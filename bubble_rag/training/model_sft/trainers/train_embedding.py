"""Embedding模型训练脚本（兼容性保留）

注意：建议使用统一的train.py脚本，设置TRAIN_TYPE=embedding
该文件主要用于向后兼容
"""
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss, ContrastiveLoss
from sentence_transformers.training_args import BatchSamplers
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from model_sft.utils.common_utils import init_swanlab, push_to_hub, create_embedding_loss
from model_sft.utils.data_loader import DataLoader
from model_sft.utils.evaluation import UnifiedEvaluator

# 配置日志
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def train_func():
    """Embedding模型训练函数"""
    logger.info("🚨 正在执行旧版训练脚本 trainers/train_embedding.py")
    load_dotenv()
    init_swanlab()

    # 1. 模型配置
    model_name = os.getenv("MODEL_NAME_OR_PATH", "distilbert-base-uncased")
    clean_model_name = model_name.replace("/", "-")
    output_dir = os.getenv("OUTPUT_DIR", 
        f"output/training_stsbenchmark_{clean_model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    
    logger.info(f"开始Embedding模型训练: {model_name}")
    logger.info(f"输出目录: {output_dir}")
    
    # 初始化模型
    model = SentenceTransformer(model_name)

    # 2. 数据加载和验证
    data_loader = DataLoader()
    train_dataset, eval_dataset, test_dataset = data_loader.load_all_splits()
    
    data_loader.validate_dataset(train_dataset)
    target_column = data_loader.get_target_column(train_dataset)
    
    logger.info(f"训练数据集: {train_dataset}")
    logger.info(f"目标列: {target_column}")

    # 3. 根据数据类型选择损失函数（支持多数据集）
    try:
        if data_loader.is_multi_dataset(train_dataset):
            # 多数据集：为每个数据集创建对应损失函数
            losses = {}
            for dataset_name, dataset in train_dataset.items():
                losses[dataset_name] = create_embedding_loss(model, dataset, target_column, dataset_name)
            loss = losses
            logger.info(f"为多个数据集创建了损失函数: {list(losses.keys())}")
        else:
            # 单数据集：创建单个损失函数
            loss = create_embedding_loss(model, train_dataset, target_column)
    except Exception as e:
        logger.warning(f"创建损失函数时出错: {e}，默认使用CosineSimilarityLoss")
        loss = CosineSimilarityLoss(model)

    # 4. 训练配置
    short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    run_name = f"embedding-{short_model_name}"

    args = SentenceTransformerTrainingArguments(
        output_dir=os.getenv("OUTPUT_DIR", output_dir),
        num_train_epochs=int(os.getenv("NUM_TRAIN_EPOCHS", "3")),
        per_device_train_batch_size=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "16")),
        per_device_eval_batch_size=int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", "16")),
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-5")),
        warmup_ratio=float(os.getenv("WARMUP_RATIO", "0.1")),
        lr_scheduler_type=os.getenv("LR_SCHEDULER_TYPE", "linear"),
        fp16=os.getenv("BF16", "False").lower() == "false",
        bf16=os.getenv("BF16", "False").lower() == "true",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy=os.getenv("EVAL_STRATEGY", "steps"),
        eval_steps=os.getenv("EVAL_STEPS", None),
        save_strategy=os.getenv("SAVE_STRATEGY", "steps"),
        save_steps=int(os.getenv("SAVE_STEPS", 500)),
        save_total_limit=int(os.getenv("SAVE_TOTAL_LIMIT", None)),
        logging_strategy=os.getenv("LOGGING_STRATEGY", "steps"),
        logging_steps=int(os.getenv("LOGGING_STEPS", 500)),
        run_name=run_name,
        gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1")),
        report_to=os.getenv("REPORT_TO", "none"),
        max_steps=int(os.getenv("MAX_STEPS", "-1")),
    )

    # 5. 创建评估器
    evaluator_factory = UnifiedEvaluator("embedding")
    evaluators = evaluator_factory.create_evaluators_from_datasets(
        eval_dataset, test_dataset, target_column, run_name
    )

    # 6. 基线模型评估
    dev_evaluator = None
    if eval_dataset is not None and 'dev' in evaluators:
        dev_evaluator = evaluators['dev']
        dev_results = evaluator_factory.evaluate_model(model, dev_evaluator)
        logger.info(f"基线模型验证结果: {dev_results}")

    # 7. 创建训练器并开始训练
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
        
    trainer = SentenceTransformerTrainer(**trainer_kwargs)
    trainer.train()
        
    # 8. 测试集评估
    if test_dataset is not None and 'test' in evaluators:
        test_evaluator = evaluators['test']
        test_results = evaluator_factory.evaluate_model(model, test_evaluator)
        logger.info(f"训练后模型测试结果: {test_results}")

    # 9. 保存训练后的模型
    save_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    logger.info(f"模型已保存到: {save_dir}")

    # 10. 推送到Hugging Face Hub（可选）
    # push_to_hub(model, model_name, save_dir)
    
    return model, save_dir

if __name__ == "__main__":
    train_func()

