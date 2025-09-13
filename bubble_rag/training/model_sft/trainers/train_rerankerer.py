"""Reranker模型训练脚本（兼容性保留）

注意：建议使用统一的train.py脚本，设置TRAIN_TYPE=reranker
该文件主要用于向后兼容
"""
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from model_sft.utils.common_utils import init_swanlab, push_to_hub
from model_sft.utils.data_loader import DataLoader
from model_sft.utils.evaluation import UnifiedEvaluator

# 配置日志
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Reranker模型训练主函数"""
    load_dotenv()
    init_swanlab()

    # 1. 模型配置
    model_name = os.getenv("MODEL_NAME_OR_PATH", "distilbert-base-uncased")
    output_dir = os.getenv("OUTPUT_DIR", 
        f"output/training_reranker_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    
    logger.info(f"开始Reranker模型训练: {model_name}")
    logger.info(f"输出目录: {output_dir}")

    # 2. 初始化CrossEncoder模型
    model = CrossEncoder(model_name, num_labels=1)

    # 处理tokenizer的pad token
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.model.config.pad_token_id = model.tokenizer.pad_token_id
    logger.info("已初始化CrossEncoder模型")

    # 3. 数据加载和验证
    data_loader = DataLoader()
    train_dataset, eval_dataset, test_dataset = data_loader.load_all_splits()
    
    data_loader.validate_dataset(train_dataset)
    target_column = data_loader.get_target_column(train_dataset)
    
    logger.info(f"训练数据集: {train_dataset}")
    logger.info(f"目标列: {target_column}")

    # 4. 定义损失函数（支持多数据集）
    try:
        if data_loader.is_multi_dataset(train_dataset):
            # 多数据集：为每个数据集创建BinaryCrossEntropyLoss
            losses = {}
            for dataset_name, dataset in train_dataset.items():
                losses[dataset_name] = BinaryCrossEntropyLoss(model)
                logger.info(f"为数据集 {dataset_name} 创建BinaryCrossEntropyLoss")
            loss = losses
        else:
            # 单数据集
            loss = BinaryCrossEntropyLoss(model)
            logger.info("使用BinaryCrossEntropyLoss损失函数")
    except Exception as e:
        logger.warning(f"创建损失函数时出错: {e}，默认使用BinaryCrossEntropyLoss")
        loss = BinaryCrossEntropyLoss(model)

    # 5. 训练配置
    short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    run_name = f"reranker-{short_model_name}"

    # 6. 创建评估器
    evaluator_factory = UnifiedEvaluator("reranker")
    evaluators = evaluator_factory.create_evaluators_from_datasets(
        eval_dataset, test_dataset, target_column, run_name
    )

    # 7. 基线模型评估
    dev_evaluator = None
    if eval_dataset is not None and 'dev' in evaluators:
        dev_evaluator = evaluators['dev']
        dev_results = evaluator_factory.evaluate_model(model, dev_evaluator)
        logger.info(f"基线模型验证结果: {dev_results}")

    # 8. 训练参数配置
    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(os.getenv("NUM_TRAIN_EPOCHS", "3")),
        per_device_train_batch_size=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "16")),
        per_device_eval_batch_size=int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", "16")),
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-5")),
        warmup_ratio=float(os.getenv("WARMUP_RATIO", "0.1")),
        fp16=os.getenv("BF16", "False").lower() == "false",
        bf16=os.getenv("BF16", "False").lower() == "true",
        eval_strategy=os.getenv("EVAL_STRATEGY", "steps"),
        eval_steps=int(os.getenv("EVAL_STEPS", "1000")),
        save_strategy=os.getenv("SAVE_STRATEGY", "steps"),
        save_steps=int(os.getenv("SAVE_STEPS", "500")),
        save_total_limit=int(os.getenv("SAVE_TOTAL_LIMIT", "3")),
        logging_steps=int(os.getenv("LOGGING_STEPS", "100")),
        run_name=run_name,
        gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1")),
        report_to=os.getenv("REPORT_TO", "none")
    )

    # 9. 创建训练器并开始训练
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "loss": loss,
    }
    
    # 添加可选参数
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if dev_evaluator is not None:
        trainer_kwargs["evaluator"] = dev_evaluator
        
    trainer = CrossEncoderTrainer(**trainer_kwargs)
    trainer.train()

    # 10. 测试集评估
    if test_dataset is not None and 'test' in evaluators:
        test_evaluator = evaluators['test']
        test_results = evaluator_factory.evaluate_model(model, test_evaluator)
        logger.info(f"训练后模型测试结果: {test_results}")

    # 11. 保存训练后的模型
    final_output_dir = f"{output_dir}/final_model"
    model.save_pretrained(final_output_dir)
    logger.info(f"模型已保存到: {final_output_dir}")

    # 12. 推送到Hugging Face Hub（可选）
    # push_to_hub(model, model_name, final_output_dir)
    
    return model, final_output_dir

if __name__ == "__main__":
    main()