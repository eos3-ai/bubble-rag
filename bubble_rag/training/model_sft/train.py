"""
统一的模型训练入口脚本
通过 TRAIN_TYPE 环境变量控制训练类型，只在训练逻辑层区分不同模型
"""
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
from bubble_rag.training.mysql_service.entity.training_dataset_models import DatasetInfo
from bubble_rag.training.mysql_service.entity.training_task_models import safe_get_session
from .enums import TrainingStatus

# 🔧 子进程模式：CUDA_VISIBLE_DEVICES已在子进程启动前设置好
# 无需手动重置CUDA上下文，子进程首次导入torch时会自动读取正确的环境变量
print(f"🔧 训练子进程启动，CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")

# 现在才导入torch相关库
# 注意：BatchSamplers 和其他工具类需要延迟导入，避免过早初始化torch

# 设置日志
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_training_device() -> str:
    """
    获取训练设备配置（环境变量已在API层设置完成）
    
    Returns:
        设备字符串，'cpu' 或 'cuda'
    """
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    
    if cuda_visible:
        # 有CUDA_VISIBLE_DEVICES，使用GPU训练
        import torch
        if torch.cuda.is_available():
            visible_gpus = cuda_visible.split(",")
            gpu_count = torch.cuda.device_count()  # 这是可见的GPU数量，不是系统总数
            logger.info(f"🔧 CUDA_VISIBLE_DEVICES: {cuda_visible}")
            logger.info(f"🖥️  将使用 {len(visible_gpus)} 个指定的GPU进行训练")
            
            # 显示每个可见GPU的详细信息
            for i in range(gpu_count):  # 遍历torch能看到的GPU
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    original_id = visible_gpus[i] if i < len(visible_gpus) else "unknown"
                    logger.info(f"   GPU {i} (原始GPU{original_id}): {gpu_name} ({gpu_memory:.1f}GB)")
                except Exception as e:
                    logger.info(f"   GPU {i}: 信息获取失败 - {e}")
            return "cuda"
        else:
            logger.warning("⚠️  设置了CUDA_VISIBLE_DEVICES但系统不支持CUDA，回退到CPU")
            return "cpu"
    else:
        # 没有CUDA_VISIBLE_DEVICES，可能是CPU模式或auto模式
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"🖥️  未指定CUDA_VISIBLE_DEVICES，检测到 {gpu_count} 个GPU可用")
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                except:
                    logger.info(f"   GPU {i}: 信息获取失败")
            return "cuda"
        else:
            logger.info("🖥️  系统不支持CUDA，使用CPU训练")
            return "cpu"


def _prepare_model_for_training(model, device: str):
    """
    准备模型进行训练
    
    Args:
        model: 训练模型
        device: 设备，如 'cuda' 或 'cpu'
        
    Returns:
        配置好的模型
    """
    import torch
    
    try:
        # 检查并移除任何DataParallel包装
        if hasattr(model, '_modules'):
            for module_name, module in model._modules.items():
                if isinstance(module, torch.nn.DataParallel):
                    # 提取原始模块，移除DataParallel包装
                    original_module = module.module
                    model._modules[module_name] = original_module
                    logger.info(f"🔓 移除模块 {module_name} 的DataParallel包装")
        
        logger.info(f"✅ 模型训练准备完成")
        return model
        
    except Exception as e:
        logger.error(f"❌ 模型准备失败: {str(e)}")
        logger.warning(f"⚠️  将继续使用原始模型配置")
        return model


def _create_training_config(output_dir: str, run_name: str, training_params: dict = None) -> dict:
    """
    创建训练配置字典，完全使用参数传递方式，避免环境变量依赖
    
    Args:
        output_dir: 输出目录
        run_name: 运行名称
        training_params: 训练参数字典（来自接口传参）
        
    Returns:
        训练配置字典
    """
    from .models.training_parameters import TrainingParametersManager
    
    # 运行时参数，不能通过参数管理器配置
    runtime_params = {
        "output_dir": output_dir,
        "run_name": run_name,
    }
    
    # 准备传递给参数管理器的配置
    if training_params:
        # 直接使用传入的所有参数，完全避免环境变量
        logger.info(f"使用接口传入的训练参数: {list(training_params.keys())}")
        param_config = dict(training_params)  # 创建副本避免修改原参数
    else:
        # 如果没有传入参数，使用空配置（将使用TrainingParameters的默认值）
        logger.info("未传入训练参数，使用默认配置")
        param_config = {}
    
    try:
        # 创建参数管理器，直接从配置字典加载（不依赖环境变量）
        param_manager = TrainingParametersManager()
        param_manager.load_from_config(param_config)
        
        # 获取验证过的参数字典
        config = param_manager.get_training_args_dict()
        
        # 添加运行时参数
        config.update(runtime_params)
        
        # 特殊处理用户日志目录
        user_logging_dir = training_params.get("user_logging_dir") if training_params else None
        if user_logging_dir and not config.get("logging_dir"):
            config["logging_dir"] = user_logging_dir
            logger.info(f"使用用户指定的日志目录: {user_logging_dir}")
            
        logger.info("✅ 训练配置创建成功（纯参数传递模式）")
        return config
        
    except Exception as e:
        logger.error(f"训练参数验证失败: {e}")
        logger.warning("回退到基础配置模式")
        
        # 回退到基础配置（不使用环境变量，只有必要的默认值）
        return _create_basic_training_config(output_dir, run_name, training_params)


def _create_training_config_legacy(output_dir: str, run_name: str) -> dict:
    """
    旧版训练配置创建方法（向后兼容）
    
    Args:
        output_dir: 输出目录
        run_name: 运行名称
        
    Returns:
        训练配置字典
    """
    # 项目自定义的环境变量，不是HuggingFace训练参数，需要排除
    project_env_vars = {
        "TRAIN_TYPE", "MODEL_NAME_OR_PATH", "OUTPUT_DIR", "SAMPLE_SIZE",
        "DATASET_NAME_OR_PATH", "DEVICE",
        # 向后兼容的环境变量
        "TRAIN_DATASET", "EVAL_DATASET", "TEST_DATASET",
        # SwanLab相关环境变量
        "SWANLAB_API_KEY", "SWANLAB_WORKSPACE", "SWANLAB_PROJECT", 
        "SWANLAB_EXPERIMENT", "SWANLAB_MODE"
    }
    
    # 基础配置，只设置运行时确定的参数
    config = {
        "output_dir": output_dir,
        "run_name": run_name,
    }
    
    # 只添加HuggingFace训练相关的环境变量
    # 定义HuggingFace训练参数的前缀和已知参数
    hf_training_params = {
        "NUM_TRAIN_EPOCHS", "PER_DEVICE_TRAIN_BATCH_SIZE", "PER_DEVICE_EVAL_BATCH_SIZE",
        "LEARNING_RATE", "WARMUP_RATIO", "LR_SCHEDULER_TYPE", "BF16", "FP16",
        "EVAL_STRATEGY", "EVAL_STEPS", "SAVE_STRATEGY", "SAVE_STEPS", "SAVE_TOTAL_LIMIT",
        "LOGGING_STEPS", "LOGGING_STRATEGY", "LOGGING_DIR", "GRADIENT_ACCUMULATION_STEPS", "MAX_STEPS",
        "BATCH_SAMPLER", "DATALOADER_NUM_WORKERS", "WEIGHT_DECAY", "ADAM_BETA1", "ADAM_BETA2",
        "ADAM_EPSILON", "MAX_GRAD_NORM", "SEED", "DATALOADER_DROP_LAST", "EVAL_ACCUMULATION_STEPS",
        "LOAD_BEST_MODEL_AT_END", "METRIC_FOR_BEST_MODEL", "GREATER_IS_BETTER", "IGNORE_DATA_SKIP",
        "RESUME_FROM_CHECKPOINT", "PUSH_TO_HUB", "HUB_MODEL_ID", "HUB_STRATEGY", "HUB_TOKEN",
        "PREDICTION_LOSS_ONLY", "REMOVE_UNUSED_COLUMNS", "LABEL_NAMES", "LOCAL_RANK", "DEEPSPEED",
        "OPTIM", "GROUP_BY_LENGTH", "LENGTH_COLUMN_NAME", "REPORT_TO", "DDPBACKEND"
    }
    
    # 定义需要类型转换的参数
    int_params = {
        "num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "eval_steps", "save_steps", "save_total_limit",
        "logging_steps", "max_steps", "dataloader_num_workers", "eval_accumulation_steps",
        "seed", "local_rank"
    }
    
    float_params = {
        "learning_rate", "warmup_ratio", "weight_decay", "adam_beta1", "adam_beta2",
        "adam_epsilon", "max_grad_norm"
    }
    
    bool_params = {
        "bf16", "fp16", "dataloader_drop_last", "load_best_model_at_end",
        "greater_is_better", "ignore_data_skip", "push_to_hub", "prediction_loss_only",
        "remove_unused_columns", "group_by_length"
    }

    # 添加符合条件的环境变量
    for key, value in os.environ.items():
        key_upper = key.upper()
        if (key_upper not in project_env_vars and 
            (key_upper in hf_training_params or 
             key.lower() in ['run_name', 'output_dir'])):
            # 将环境变量名转换为训练参数名（小写）
            param_name = key.lower()
            
            # 类型转换
            try:
                if param_name in int_params:
                    config[param_name] = int(value)
                elif param_name in float_params:
                    config[param_name] = float(value)
                elif param_name in bool_params:
                    # 处理布尔值
                    config[param_name] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    config[param_name] = value
            except (ValueError, TypeError) as e:
                logger.warning(f"无法转换参数 {param_name} 的值 '{value}': {e}")
                config[param_name] = value
            
            logger.info(f"传递训练参数: {param_name} = {config[param_name]}")
    
    # 注意：Legacy函数不再使用环境变量，避免环境污染
    # report_to 参数应该通过参数传递方式获取，而不是环境变量
    
    return config


def _create_basic_training_config(output_dir: str, run_name: str, training_params: dict = None) -> dict:
    """
    创建基础训练配置，完全不依赖环境变量，使用传入参数和合理默认值
    
    Args:
        output_dir: 输出目录
        run_name: 运行名称
        training_params: 训练参数字典
        
    Returns:
        训练配置字典
    """
    # 基础默认配置
    config = {
        # 运行时参数
        "output_dir": output_dir,
        "run_name": run_name,
        
        # 训练基础参数默认值
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        
        # 评估和保存策略
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps", 
        "save_steps": 100,
        "save_total_limit": 3,
        
        # 日志配置 - 将从用户参数中覆盖
        "logging_steps": 100,  # 默认值，会被用户参数覆盖
        "logging_strategy": "steps",
        
        # 优化器配置
        "optim": "adamw_hf",
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        
        # 数据加载
        "dataloader_num_workers": 0,
        "dataloader_drop_last": False,
        
        # 其他配置
        "seed": 42,
        "bf16": False,
        "fp16": False,
        "gradient_accumulation_steps": 1,
        "remove_unused_columns": True,
        "prediction_loss_only": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    
    # 如果有传入的参数，使用传入的参数覆盖默认值
    if training_params:
        logger.info(f"使用传入参数覆盖默认配置: {list(training_params.keys())}")
        for key, value in training_params.items():
            if value is not None:  # 只覆盖非None的值
                config[key] = value
                logger.debug(f"参数覆盖: {key} = {value}")
    
    logger.info("✅ 基础训练配置创建成功（纯参数模式，无环境变量依赖）")
    return config


def _initialize_model_and_loss(train_type: str, model_name: str, train_dataset, target_column: str, training_config: dict):
    """
    根据训练类型初始化模型、损失函数和训练参数，支持多数据集和多损失函数
    
    Args:
        train_type: 训练类型 ('embedding' 或 'reranker')
        model_name: 模型名称
        train_dataset: 训练数据集（可能是单个Dataset或Dict[str, Dataset]）
        target_column: 目标列名
        training_config: 训练配置
        
    Returns:
        (model, loss, args) 元组，其中loss可能是单个损失函数或Dict[str, 损失函数]
    """
    from .utils.data_loader import DataLoader
    hf_subset = training_config.get('HF_subset') if training_config else None
    data_loader = DataLoader(
        hf_subset=hf_subset,
        train_sample_size=training_config.get('train_sample_size', 0),
        eval_sample_size=training_config.get('eval_sample_size', 0),
        test_sample_size=training_config.get('test_sample_size', 0)
    )
    is_multi_dataset = data_loader.is_multi_dataset(train_dataset)
    
    if train_type == "embedding":
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import CosineSimilarityLoss, ContrastiveLoss, MultipleNegativesRankingLoss
        
        # 初始化embedding模型
        try:
            # 获取设备配置并验证CUDA环境
            device = _get_training_device()
            
            # 创建模型前再次验证设备配置
            import os
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
            logger.info(f"🔧 模型初始化前的设备检查:")
            logger.info(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
            logger.info(f"   计算得到的设备: {device}")
            
            # 尝试用ModelScope下载模型到统一缓存目录，然后用SentenceTransformer加载
            try:
                from modelscope import snapshot_download
                from bubble_rag.server_config import TRAINING_CACHE
                
                logger.info(f"尝试从ModelScope下载模型: {model_name}")
                model_dir = snapshot_download(model_name, cache_dir=TRAINING_CACHE)
                logger.info(f"✅ ModelScope下载成功，路径: {model_dir}")
                
                # 从本地路径加载
                model = SentenceTransformer(model_dir)
                logger.info(f"✅ embedding模型从ModelScope缓存初始化成功: {model_name}")
            except ImportError:
                logger.info("ModelScope未安装，使用HuggingFace方式")
                model = SentenceTransformer(model_name)
                logger.info(f"✅ embedding模型从HuggingFace初始化成功: {model_name}")
            except Exception as ms_error:
                logger.warning(f"ModelScope下载失败: {ms_error}，回退到HuggingFace")
                model = SentenceTransformer(model_name)
                logger.info(f"✅ embedding模型从HuggingFace初始化成功: {model_name}")
            
            # 验证模型实际使用的设备
            if hasattr(model, 'device'):
                logger.info(f"🖥️  模型实际所在设备: {model.device}")
            else:
                logger.info(f"🖥️  模型设备信息: 无法直接获取")
            
            # 准备模型进行训练
            model = _prepare_model_for_training(model, device)
        except Exception as e:
            # 如果网络失败，尝试本地缓存模式
            if "couldn't connect" in str(e).lower() or "connection" in str(e).lower():
                logger.warning(f"网络连接失败，尝试使用本地缓存加载模型: {model_name}")
                try:
                    # 设置离线模式环境变量
                    import os
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    model = SentenceTransformer(model_name, local_files_only=True)
                    logger.info(f"✅ embedding模型从本地缓存初始化成功: {model_name}")
                except Exception as cache_error:
                    error_msg = f"embedding模型初始化失败: {model_name}，网络不可用且本地缓存未找到。错误: {str(cache_error)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"embedding模型初始化失败: {model_name}, 错误: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        if is_multi_dataset:
            # 多数据集：对每个数据集应用列过滤并创建对应的损失函数
            from .utils.evaluation import UnifiedEvaluator
            temp_evaluator_factory = UnifiedEvaluator(train_type)
            
            losses = {}
            filtered_train_dataset = {}
            
            for dataset_name, dataset in train_dataset.items():
                # 为每个数据集单独确定目标列名
                dataset_target_column = temp_evaluator_factory._get_dataset_target_column(dataset)
                
                # 过滤数据集列：只保留前两列作为输入列 + 目标列
                column_names = dataset.column_names
                if len(column_names) >= 3:
                    # 确保只有3列：sentence1, sentence2, target_column
                    input_columns = [col for col in column_names if col != dataset_target_column][:2]
                    columns_to_keep = input_columns + [dataset_target_column]
                    filtered_dataset = dataset.select_columns(columns_to_keep)
                    logger.info(f"Embedding数据集 '{dataset_name}' 列过滤: {column_names} → {columns_to_keep}")
                else:
                    filtered_dataset = dataset
                
                filtered_train_dataset[dataset_name] = filtered_dataset
                
                # 使用过滤后的数据集创建损失函数
                from .utils.common_utils import create_embedding_loss
                loss_func = create_embedding_loss(model, filtered_dataset, dataset_target_column, dataset_name)
                losses[dataset_name] = loss_func
            
            loss = losses
            # 更新训练数据集为过滤后的版本
            train_dataset = filtered_train_dataset
            logger.info(f"为多个embedding数据集创建了损失函数: {list(losses.keys())}")
        else:
            # 单数据集：应用列过滤并创建单个损失函数
            from .utils.evaluation import UnifiedEvaluator
            temp_evaluator_factory = UnifiedEvaluator(train_type)
            dataset_target_column = temp_evaluator_factory._get_dataset_target_column(train_dataset)
            
            # 过滤数据集列：只保留前两列作为输入列 + 目标列
            column_names = train_dataset.column_names
            if len(column_names) >= 3:
                # 确保只有3列：sentence1, sentence2, target_column
                input_columns = [col for col in column_names if col != dataset_target_column][:2]
                columns_to_keep = input_columns + [dataset_target_column]
                train_dataset = train_dataset.select_columns(columns_to_keep)
                logger.info(f"单Embedding数据集列过滤: {column_names} → {columns_to_keep}")
            
            # 使用过滤后的数据集创建损失函数
            from .utils.common_utils import create_embedding_loss
            loss = create_embedding_loss(model, train_dataset, dataset_target_column)
        
        # 过滤掉多进程和sample_size相关的参数，这些参数不属于训练参数类
        filtered_config = {k: v for k, v in training_config.items() 
                          if k not in ['nproc_per_node', 'local_rank', 'master_port', 'master_addr', 
                                     'train_sample_size', 'eval_sample_size', 'test_sample_size']}
        
        # 添加进度条控制，避免多进度条重叠
        filtered_config['disable_tqdm'] = False  # 启用主进度条
        
        args = SentenceTransformerTrainingArguments(**filtered_config)
        
    elif train_type == "reranker":
        from sentence_transformers.cross_encoder import CrossEncoder
        from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
        from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
        
        # 初始化reranker模型
        try:
            # 获取设备配置并验证CUDA环境
            device = _get_training_device()
            
            # 创建模型前再次验证设备配置
            import os
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
            logger.info(f"🔧 reranker模型初始化前的设备检查:")
            logger.info(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
            logger.info(f"   计算得到的设备: {device}")

            # 尝试用ModelScope下载模型到统一缓存目录，然后用CrossEncoder加载
            try:
                from modelscope import snapshot_download
                from bubble_rag.server_config import TRAINING_CACHE
                
                logger.info(f"尝试从ModelScope下载模型: {model_name}")
                model_dir = snapshot_download(model_name, cache_dir=TRAINING_CACHE)
                logger.info(f"✅ ModelScope下载成功，路径: {model_dir}")
                
                # 从本地路径加载
                model = CrossEncoder(model_dir, num_labels=1)
                logger.info(f"✅ reranker模型从ModelScope缓存初始化成功: {model_name}")
            except ImportError:
                logger.info("ModelScope未安装，使用HuggingFace方式")
                model = CrossEncoder(model_name, num_labels=1)
                logger.info("📱 使用环境变量控制的GPU初始化CrossEncoder")
            except Exception as ms_error:
                logger.warning(f"ModelScope下载失败: {ms_error}，回退到HuggingFace")
                model = CrossEncoder(model_name, num_labels=1)
                logger.info("📱 使用环境变量控制的GPU初始化CrossEncoder")
            
            actual_device = getattr(model, 'device', 'unknown')
            logger.info(f"✅ reranker模型初始化成功: {model_name}")
            logger.info(f"🖥️  模型实际所在设备: {actual_device}")
            
            # 额外检查模型的内部设备
            if hasattr(model, 'model') and hasattr(model.model, 'device'):
                logger.info(f"🖥️  模型内部device属性: {model.model.device}")
            
            # 准备模型进行训练
            model = _prepare_model_for_training(model, device)
        except Exception as e:
            # 如果是网络连接相关错误，尝试离线模式
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["connection", "network", "timeout", "huggingface.co", "could not connect"]):
                logger.warning(f"⚠️ 网络连接失败，尝试使用本地缓存: {str(e)}")
                try:
                    # 使用本地缓存模式
                    model = CrossEncoder(model_name, num_labels=1,local_files_only=True)
                    logger.info(f"✅ reranker模型使用本地缓存初始化成功: {model_name}")
                except Exception as local_error:
                    error_msg = f"reranker模型初始化失败(网络和本地缓存都失败): {model_name}, 网络错误: {str(e)}, 本地错误: {str(local_error)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"reranker模型初始化失败: {model_name}, 错误: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        # 处理tokenizer pad token
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        model.model.config.pad_token_id = model.tokenizer.pad_token_id
        
        if is_multi_dataset:
            # 多数据集：根据每个数据集的任务类型选择损失函数
            from sentence_transformers.cross_encoder.losses.MSELoss import MSELoss
            from .utils.evaluation import UnifiedEvaluator
            
            # 创建临时的evaluator_factory来获取目标列名
            temp_evaluator_factory = UnifiedEvaluator(train_type)
            
            losses = {}
            filtered_train_dataset = {}
            
            for dataset_name, dataset in train_dataset.items():
                # 为每个数据集单独确定目标列名和任务类型
                dataset_target_column = temp_evaluator_factory._get_dataset_target_column(dataset)
                labels = list(dataset[dataset_target_column])
                unique_labels = set(labels)
                
                # 过滤数据集列：只保留前两列作为输入列 + 目标列
                column_names = dataset.column_names
                if len(column_names) >= 3:
                    # 确保只有3列：sentence1, sentence2, target_column
                    input_columns = [col for col in column_names if col != dataset_target_column][:2]
                    columns_to_keep = input_columns + [dataset_target_column]
                    filtered_dataset = dataset.select_columns(columns_to_keep)
                    logger.info(f"数据集 '{dataset_name}' 列过滤: {column_names} → {columns_to_keep}")
                else:
                    filtered_dataset = dataset
                
                filtered_train_dataset[dataset_name] = filtered_dataset
                
                # 判断任务类型：只有标签严格为0/1时才是二分类
                is_binary_classification = (
                    len(unique_labels) <= 2 and 
                    all(label in [0, 1] for label in unique_labels)
                )
                
                if is_binary_classification:
                    losses[dataset_name] = BinaryCrossEntropyLoss(model)
                    logger.info(f"数据集 '{dataset_name}' 使用BinaryCrossEntropyLoss（二分类任务，标签: {sorted(unique_labels)}）")
                else:
                    # 回归任务使用MSE Loss
                    # 在创建MSELoss前验证数据集格式
                    mse_dataset_columns = len(filtered_dataset.column_names)
                    if mse_dataset_columns != 3:
                        logger.error(f"❌ MSELoss要求3列数据集，但数据集 '{dataset_name}' 有{mse_dataset_columns}列: {filtered_dataset.column_names}")
                        raise ValueError(f"MSELoss数据集格式错误: {dataset_name}有{mse_dataset_columns}列而非3列")
                    
                    losses[dataset_name] = MSELoss(model)
                    logger.info(f"数据集 '{dataset_name}' 使用MSELoss（回归任务，标签范围: {min(labels):.2f}-{max(labels):.2f}），数据集列数: {mse_dataset_columns}")
            loss = losses
            # 更新训练数据集为过滤后的版本
            train_dataset = filtered_train_dataset
        else:
            # 单数据集：根据数据集类型选择损失函数
            from .utils.evaluation import UnifiedEvaluator
            temp_evaluator_factory = UnifiedEvaluator(train_type)
            dataset_target_column = temp_evaluator_factory._get_dataset_target_column(train_dataset)
            labels = list(train_dataset[dataset_target_column])
            unique_labels = set(labels)
            
            # 过滤数据集列：只保留前两列作为输入列 + 目标列
            column_names = train_dataset.column_names
            if len(column_names) >= 3:
                # 确保只有3列：sentence1, sentence2, target_column
                input_columns = [col for col in column_names if col != dataset_target_column][:2]
                columns_to_keep = input_columns + [dataset_target_column]
                train_dataset = train_dataset.select_columns(columns_to_keep)
                logger.info(f"单数据集列过滤: {column_names} → {columns_to_keep}")
            
            is_binary_classification = (
                len(unique_labels) <= 2 and 
                all(label in [0, 1] for label in unique_labels)
            )
            
            if is_binary_classification:
                loss = BinaryCrossEntropyLoss(model)
                logger.info(f"使用BinaryCrossEntropyLoss损失函数（二分类任务，标签: {sorted(unique_labels)}）")
            else:
                # 在创建MSELoss前验证数据集格式
                mse_dataset_columns = len(train_dataset.column_names)
                if mse_dataset_columns != 3:
                    logger.error(f"❌ 单数据集MSELoss要求3列，但有{mse_dataset_columns}列: {train_dataset.column_names}")
                    raise ValueError(f"单数据集MSELoss格式错误: 有{mse_dataset_columns}列而非3列")
                
                from sentence_transformers.cross_encoder.losses.MSELoss import MSELoss
                loss = MSELoss(model)
                logger.info(f"使用MSELoss损失函数（回归任务，标签范围: {min(labels):.2f}-{max(labels):.2f}），数据集列数: {mse_dataset_columns}")
        
        # 过滤掉多进程和sample_size相关的参数，这些参数不属于训练参数类
        filtered_config = {k: v for k, v in training_config.items() 
                          if k not in ['nproc_per_node', 'local_rank', 'master_port', 'master_addr', 
                                     'train_sample_size', 'eval_sample_size', 'test_sample_size']}
        
        # 添加进度条控制，避免多进度条重叠  
        filtered_config['disable_tqdm'] = False  # 启用主进度条
        
        args = CrossEncoderTrainingArguments(**filtered_config)
    else:
        raise ValueError(f"不支持的训练类型: {train_type}. 只支持 'embedding' 或 'reranker'")
    
    return model, loss, args, train_dataset



def _create_trainer(train_type: str, model, args, train_dataset, eval_dataset, loss, dev_evaluator):
    """
    创建对应类型的训练器
    
    Args:
        train_type: 训练类型
        model: 模型实例
        args: 训练参数
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        loss: 损失函数
        dev_evaluator: 验证评估器
        
    Returns:
        训练器实例
    """
    # 导入UnifiedEvaluator用于数据集处理
    from .utils.evaluation import UnifiedEvaluator
    
    # 检查评估配置，如果没有评估数据或评估器，则禁用评估
    has_eval_data = eval_dataset is not None
    has_evaluator = dev_evaluator is not None
    
    # 如果设置了评估策略但没有评估数据或评估器，则修改评估策略为"no"
    if hasattr(args, 'eval_strategy') and args.eval_strategy != "no":
        if not has_eval_data and not has_evaluator:
            logger.warning("设置了eval_strategy但没有提供eval_dataset或evaluator，将自动设置eval_strategy='no'")
            args.eval_strategy = "no"
    
    # 验证训练数据集的列数（调试用）
    if isinstance(train_dataset, dict):
        for dataset_name, dataset in train_dataset.items():
            column_count = len(dataset.column_names)
            logger.info(f"🔍 训练器构建前验证 - 数据集 '{dataset_name}' 列数: {column_count}, 列名: {dataset.column_names}")
            if column_count != 3:
                logger.error(f"❌ 数据集 '{dataset_name}' 列数异常！应为3列，实际为{column_count}列")
    else:
        column_count = len(train_dataset.column_names)
        logger.info(f"🔍 训练器构建前验证 - 训练数据集列数: {column_count}, 列名: {train_dataset.column_names}")
        if column_count != 3:
            logger.error(f"❌ 训练数据集列数异常！应为3列，实际为{column_count}列")
    
    # 构建训练器参数
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "loss": loss,
    }
    
    # 添加可选参数
    if has_eval_data:
        # 对验证数据集应用相同的列过滤逻辑
        filtered_eval_dataset = eval_dataset
        if isinstance(eval_dataset, dict):
            # 多数据集：对每个数据集应用列过滤
            filtered_eval_dataset = {}
            for dataset_name, dataset in eval_dataset.items():
                if dataset is not None:
                    # 为验证数据集确定目标列名
                    temp_evaluator_factory = UnifiedEvaluator(train_type)
                    dataset_target_column = temp_evaluator_factory._get_dataset_target_column(dataset)
                    
                    # 应用列过滤
                    column_names = dataset.column_names
                    if len(column_names) >= 3:
                        input_columns = [col for col in column_names if col != dataset_target_column][:2]
                        columns_to_keep = input_columns + [dataset_target_column]
                        filtered_dataset = dataset.select_columns(columns_to_keep)
                        logger.info(f"验证数据集 '{dataset_name}' 列过滤: {column_names} → {columns_to_keep}")
                        filtered_eval_dataset[dataset_name] = filtered_dataset
                    else:
                        filtered_eval_dataset[dataset_name] = dataset
        elif eval_dataset is not None:
            # 单数据集：应用列过滤
            temp_evaluator_factory = UnifiedEvaluator(train_type)
            dataset_target_column = temp_evaluator_factory._get_dataset_target_column(eval_dataset)
            
            column_names = eval_dataset.column_names
            if len(column_names) >= 3:
                input_columns = [col for col in column_names if col != dataset_target_column][:2]
                columns_to_keep = input_columns + [dataset_target_column]
                filtered_eval_dataset = eval_dataset.select_columns(columns_to_keep)
                logger.info(f"单验证数据集列过滤: {column_names} → {columns_to_keep}")
        
        # 验证验证数据集的列数
        if isinstance(filtered_eval_dataset, dict):
            for dataset_name, dataset in filtered_eval_dataset.items():
                if dataset is not None:
                    column_count = len(dataset.column_names)
                    logger.info(f"🔍 验证数据集 '{dataset_name}' 列数: {column_count}, 列名: {dataset.column_names}")
                    if column_count != 3:
                        logger.error(f"❌ 验证数据集 '{dataset_name}' 列数异常！应为3列，实际为{column_count}列")
        elif filtered_eval_dataset is not None:
            column_count = len(filtered_eval_dataset.column_names)
            logger.info(f"🔍 验证数据集列数: {column_count}, 列名: {filtered_eval_dataset.column_names}")
            if column_count != 3:
                logger.error(f"❌ 验证数据集列数异常！应为3列，实际为{column_count}列")
        
        trainer_kwargs["eval_dataset"] = filtered_eval_dataset
    if has_evaluator:
        trainer_kwargs["evaluator"] = dev_evaluator
    
    # 🔍 调试训练器参数
    logger.info(f"🔍 训练器参数调试:")
    logger.info(f"   args.per_device_train_batch_size: {getattr(args, 'per_device_train_batch_size', 'NOT SET')}")
    logger.info(f"   args.gradient_accumulation_steps: {getattr(args, 'gradient_accumulation_steps', 'NOT SET')}")
    logger.info(f"   args.num_train_epochs: {getattr(args, 'num_train_epochs', 'NOT SET')}")
    logger.info(f"   args.max_steps: {getattr(args, 'max_steps', 'NOT SET')}")
    
    if isinstance(train_dataset, dict):
        total_samples = sum(len(ds) for ds in train_dataset.values())
        logger.info(f"   实际训练数据集总大小: {total_samples}")
        for name, ds in train_dataset.items():
            logger.info(f"   数据集 '{name}' 大小: {len(ds)}")
    else:
        logger.info(f"   训练数据集大小: {len(train_dataset) if hasattr(train_dataset, '__len__') else '无法计算'}")
    
    # 根据类型创建训练器
    if train_type == "embedding":
        from sentence_transformers import SentenceTransformerTrainer
        trainer = SentenceTransformerTrainer(**trainer_kwargs)
    elif train_type == "reranker":
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
        trainer = CrossEncoderTrainer(**trainer_kwargs)
    else:
        raise ValueError(f"不支持的训练类型: {train_type}. 只支持 'embedding' 或 'reranker'")
    
    return trainer

def main(progress_callback=None, training_config=None):
    """主训练函数，根据TRAIN_TYPE环境变量选择训练模式
    
    Args:
        progress_callback: 进度回调函数
        training_config: 训练配置字典，包含:
            - report_to: 报告工具 (swanlab, tensorboard, etc.)
            - user_logging_dir: 用户日志目录
            - 其他训练参数...
    """
    logger.info("🔥 正在执行统一训练脚本 train.py")
    load_dotenv()
    
    # 🔧 获取任务ID用于全局异常处理
    task_id = None
    if training_config:
        task_id = training_config.get("task_id") 
    
    # SwanLab配置处理：从training_config中获取配置
    training_config = training_config or {}
    report_to = training_config.get("report_to", "")
    logger.info(f"报告工具配置: report_to='{report_to}'")
    
    # 初始化SwanLab（如果配置了的话）
    from .enums.training_parameter_enums import ReportTo
    if report_to == ReportTo.SWANLAB or report_to == ReportTo.SWANLAB.value:
        try:
            from .utils.common_utils import init_swanlab
            # 传入完整的training_config，让SwanLab配置类处理所有参数
            init_swanlab(training_config=training_config)
        except Exception as e:
            logger.warning(f"SwanLab初始化失败，继续训练: {e}")
    
    # 1. 从training_config获取配置（完全不依赖环境变量）
    train_type = training_config.get("train_type", "embedding").lower()
    model_name = training_config.get("model_name_or_path", "distilbert-base-uncased")
    
    # 如果没有提供output_dir，生成默认路径
    if not training_config.get("output_dir"):
        output_dir = f"output/training_{train_type}_{model_name.replace('/', '-')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        output_dir = training_config.get("output_dir")
    
    # 获取任务ID（只从training_config获取）
    task_id = training_config.get("task_id")
    if task_id:
        logger.info(f"使用任务ID: {task_id}")
    else:
        logger.warning("未获取到任务ID，部分功能可能无法正常工作")
        
    logger.info(f"训练配置 - 类型: {train_type}, 模型: {model_name}, 输出目录: {output_dir}")
    
    # 注册输出目录到临时文件管理器（用于异常情况下的清理）
    from .utils.temp_file_manager import temp_file_manager
    temp_file_manager.register_temp_dir(output_dir)
    
    logger.info(f"开始训练，训练类型: {train_type}")
    logger.info(f"模型: {model_name}")
    logger.info(f"输出目录: {output_dir}")

    # 2. 统一的数据加载
    from .utils.data_loader import DataLoader
    hf_subset = training_config.get('HF_subset') if training_config else None
    data_loader = DataLoader(
        hf_subset=hf_subset,
        train_sample_size=training_config.get('train_sample_size', 0),
        eval_sample_size=training_config.get('eval_sample_size', 0),
        test_sample_size=training_config.get('test_sample_size', 0)
    )
    
    # 从训练配置中获取数据集路径
    dataset_path = training_config.get("dataset_name_or_path")
    logger.info(f"🔧 使用数据集路径: {dataset_path}")
    
    train_dataset, eval_dataset, test_dataset = data_loader.load_all_splits(dataset_path)
    
    # 检查训练数据集是否成功加载
    if train_dataset is None:
        dataset_name_or_path = training_config.get("dataset_name_or_path", "未指定数据集")
        error_msg = f"训练数据集加载失败: {dataset_name_or_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 🔍 调试：检查数据集结构
    logger.info(f"🔍 train_dataset 类型: {type(train_dataset)}")
    if data_loader.is_multi_dataset(train_dataset):
        logger.info(f"🔍 多数据集键名: {list(train_dataset.keys())}")
        for name, dataset in train_dataset.items():
            logger.info(f"🔍 数据集 '{name}' 类型: {type(dataset)}")
            if hasattr(dataset, 'column_names'):
                logger.info(f"🔍 数据集 '{name}' 列名: {dataset.column_names}")
            else:
                logger.error(f"🔍 数据集 '{name}' 没有 column_names 属性!")
    else:
        logger.info(f"🔍 单数据集列名: {train_dataset.column_names if hasattr(train_dataset, 'column_names') else '无 column_names 属性'}")
    
    data_loader.validate_dataset(train_dataset)
    target_column = data_loader.get_target_column(train_dataset)
    
    # 🔧 标准化数据集列名以符合sentence-transformers要求
    logger.info("📋 开始数据集列名标准化...")
    train_dataset = data_loader.standardize_dataset_columns(train_dataset, target_column)
    
    # 标准化后需要更新目标列名称
    if target_column not in ["label", "score"]:
        # 根据数据类型确定新的目标列名
        if data_loader.is_multi_dataset(train_dataset):
            first_dataset = next(iter(train_dataset.values()))
            # 检查 first_dataset 是否为 Dataset 对象
            if not hasattr(first_dataset, 'column_names'):
                logger.error(f"多数据集中的第一个数据集不是 Dataset 对象: {type(first_dataset)}")
                logger.error(f"多数据集结构: {list(train_dataset.keys()) if isinstance(train_dataset, dict) else 'Not a dict'}")
                raise ValueError("多数据集中的数据集对象格式异常")
        else:
            first_dataset = train_dataset
        
        sample_value = first_dataset[target_column][0] if target_column in first_dataset.column_names else None
        if sample_value is not None:
            if isinstance(sample_value, int) or (isinstance(sample_value, float) and sample_value.is_integer()):
                target_column = "label"
            else:
                target_column = "score"
        logger.info(f"🔄 目标列名已更新为: {target_column}")
    
    # 🔧 同样标准化验证和测试数据集
    if eval_dataset is not None:
        logger.info("📋 标准化验证数据集列名...")
        eval_dataset = data_loader.standardize_dataset_columns(eval_dataset, target_column)
        
    if test_dataset is not None:
        logger.info("📋 标准化测试数据集列名...")
        test_dataset = data_loader.standardize_dataset_columns(test_dataset, target_column)
    
    logger.info(f"训练数据集: {train_dataset}")
    logger.info(f"目标列: {target_column}")

    # 建立数据源映射（在更大的作用域中，供后续损失函数更新使用）
    def generate_data_source_id(index: int, base_name: str) -> str:
        """生成数据源ID"""
        return str(index + 1)  # 1, 2, 3, ...
    
    data_source_mapping = {}
    if data_loader.is_multi_dataset(train_dataset) and isinstance(train_dataset, dict):
        for idx, base_name in enumerate(train_dataset.keys()):
            data_source_mapping[base_name] = generate_data_source_id(idx, base_name)
    else:
        # 单数据集：使用固定的数据源ID和基础名称
        dataset_name_or_path = training_config.get("dataset_name_or_path", "unknown")
        base_name = data_loader._extract_dataset_base_name(dataset_name_or_path)
        data_source_mapping[base_name] = "1"


    # 记录数据集信息到数据库
    if task_id:
        try:
            dataset_name_or_path = training_config.get("dataset_name_or_path", "unknown")
                    
            # 记录训练数据集
            if train_dataset:
                if data_loader.is_multi_dataset(train_dataset):
                    # 多数据集：每个数据源分配独立的ID
                    for base_name, dataset in train_dataset.items():
                        data_source_id = data_source_mapping[base_name]  # 使用统一映射
                        
                        TrainingDatasetService.record_dataset_info(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            dataset_name=base_name,  # 🔧 去掉split后缀，只保存基础名称
                            dataset_base_name=base_name,
                            dataset_path=dataset_name_or_path,
                            dataset_type="auto",
                            split_type="train",
                            dataset=dataset,
                            target_column=target_column,
                            loss_function=None,
                            evaluator=None,
                            hf_subset=training_config.get('HF_subset'),  # HF_subset配置
                            configured_sample_size=training_config.get('train_sample_size', 0)  # 新增：样本大小配置
                        )
                else:
                    # 单数据集：使用固定的数据源ID
                    base_name = data_loader._extract_dataset_base_name(dataset_name_or_path)
                    data_source_id = generate_data_source_id(0, base_name)
                    
                    TrainingDatasetService.record_dataset_info(
                        task_id=task_id,
                        data_source_id=data_source_id,
                        dataset_name=base_name,  # 🔧 去掉split后缀，只保存基础名称
                        dataset_base_name=base_name,
                        dataset_path=dataset_name_or_path,
                        dataset_type="auto",
                        split_type="train",
                        dataset=train_dataset,
                        target_column=target_column,
                        loss_function=None,
                        evaluator=None,
                        hf_subset=training_config.get('HF_subset'),  # HF_subset配置
                        configured_sample_size=training_config.get('train_sample_size', 0)  # 新增：样本大小配置
                    )
            
            # 记录验证数据集
            if eval_dataset:
                if data_loader.is_multi_dataset(eval_dataset):
                    # 多数据集：为每个数据源记录验证集
                    for base_name, dataset in eval_dataset.items():
                        data_source_id = data_source_mapping.get(base_name, f"ds_{hash(base_name) % 1000:03d}")  # 使用映射或回退
                        
                        TrainingDatasetService.record_dataset_info(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            dataset_name=base_name,  # 🔧 去掉split后缀，只保存基础名称
                            dataset_base_name=base_name,
                            dataset_path=dataset_name_or_path,
                            dataset_type="auto",
                            split_type="eval",
                            dataset=dataset,
                            target_column=target_column,
                            loss_function=None,
                            evaluator=None,
                            hf_subset=training_config.get('HF_subset'),  # HF_subset配置
                            configured_sample_size=training_config.get('eval_sample_size', 0)  # 新增：样本大小配置
                        )
                else:
                    # 单数据集：使用相同的数据源ID
                    base_name = data_loader._extract_dataset_base_name(dataset_name_or_path)
                    data_source_id = generate_data_source_id(0, base_name)
                    
                    TrainingDatasetService.record_dataset_info(
                        task_id=task_id,
                        data_source_id=data_source_id,
                        dataset_name=base_name,  # 🔧 去掉split后缀，只保存基础名称
                        dataset_base_name=base_name,
                        dataset_path=dataset_name_or_path,
                        dataset_type="auto",
                        split_type="eval",
                        dataset=eval_dataset,
                        target_column=target_column,
                        loss_function=None,
                        evaluator=None,
                        hf_subset=training_config.get('HF_subset'),  # HF_subset配置
                        configured_sample_size=training_config.get('eval_sample_size', 0)  # 新增：样本大小配置
                    )
            
            # 记录测试数据集
            if test_dataset:
                if data_loader.is_multi_dataset(test_dataset):
                    # 多数据集：为每个数据源记录测试集
                    for base_name, dataset in test_dataset.items():
                        data_source_id = data_source_mapping.get(base_name, f"ds_{hash(base_name) % 1000:03d}")  # 使用映射或回退
                        
                        TrainingDatasetService.record_dataset_info(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            dataset_name=base_name,  # 🔧 去掉split后缀，只保存基础名称
                            dataset_base_name=base_name,
                            dataset_path=dataset_name_or_path,
                            dataset_type="auto",
                            split_type="test",
                            dataset=dataset,
                            target_column=target_column,
                            loss_function=None,
                            evaluator=None,
                            hf_subset=training_config.get('HF_subset'),  # HF_subset配置
                            configured_sample_size=training_config.get('test_sample_size', 0)  # 新增：样本大小配置
                        )
                else:
                    # 单数据集：使用相同的数据源ID
                    base_name = data_loader._extract_dataset_base_name(dataset_name_or_path)
                    data_source_id = generate_data_source_id(0, base_name)
                    
                    TrainingDatasetService.record_dataset_info(
                        task_id=task_id,
                        data_source_id=data_source_id,
                        dataset_name=base_name,  # 🔧 去掉split后缀，只保存基础名称
                        dataset_base_name=base_name,
                        dataset_path=dataset_name_or_path,
                        dataset_type="auto",
                        split_type="test",
                        dataset=test_dataset,
                        target_column=target_column,
                        loss_function=None,
                        evaluator=None,
                        hf_subset=training_config.get('HF_subset'),  # HF_subset配置
                        configured_sample_size=training_config.get('test_sample_size', 0)  # 新增：样本大小配置
                    )
            
            logger.info("数据集信息记录成功")
        except Exception as e:
            logger.warning(f"记录数据集信息失败（不影响训练）: {e}")

    # 3. 创建统一的训练配置
    short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    run_name = f"{train_type}-{short_model_name}"
    
    # 构建训练配置字典（embedding和reranker共用）
    training_config_dict = _create_training_config(output_dir, run_name, training_config)


    # 4. 根据训练类型初始化模型、损失函数和训练参数
    model, loss, args, train_dataset = _initialize_model_and_loss(train_type, model_name, train_dataset, target_column, training_config_dict)
    
    # 4.1. 损失函数创建后，更新数据集信息中的损失函数名称
    if task_id:
        try:
            # 获取实际使用的损失函数名称
            from .utils.data_loader import DataLoader
            hf_subset = training_config_dict.get('HF_subset') if training_config_dict else None
            data_loader = DataLoader(
                hf_subset=hf_subset,
                train_sample_size=training_config_dict.get('train_sample_size', 0),
                eval_sample_size=training_config_dict.get('eval_sample_size', 0),
                test_sample_size=training_config_dict.get('test_sample_size', 0)
            )
            is_multi_dataset = data_loader.is_multi_dataset(train_dataset)
            
            if is_multi_dataset and isinstance(loss, dict):
                # 多数据集情况：为每个数据源更新对应的损失函数
                for dataset_name, loss_func in loss.items():
                    actual_loss_name = type(loss_func).__name__
                    data_source_id = data_source_mapping.get(dataset_name, f"ds_{hash(dataset_name) % 1000:03d}")  # 使用统一映射
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="train",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新多数据集损失函数: {data_source_id}-train -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新数据源 {data_source_id} 训练集损失函数失败: {e}")
                
                # 多数据集情况：验证集和测试集使用对应同名训练数据集的损失函数
                
                # 更新验证集损失函数（如果存在验证集且为多数据集）
                if eval_dataset and isinstance(eval_dataset, dict):
                    for dataset_name in eval_dataset.keys():
                        data_source_id = data_source_mapping.get(dataset_name, f"ds_{hash(dataset_name) % 1000:03d}")  # 使用统一映射
                        # 查找对应的训练数据集损失函数
                        if dataset_name in loss:
                            corresponding_loss_name = type(loss[dataset_name]).__name__
                            try:
                                TrainingDatasetService.update_dataset_loss_function_by_source(
                                    task_id=task_id,
                                    data_source_id=data_source_id,
                                    split_type="eval",
                                    loss_function=corresponding_loss_name
                                )
                                logger.info(f"更新多验证数据集损失函数: {data_source_id}-eval -> {corresponding_loss_name} (绑定到 {data_source_id}-train)")
                            except Exception as e:
                                logger.warning(f"更新验证数据源 {data_source_id} 损失函数失败: {e}")
                        else:
                            logger.warning(f"验证数据集 {dataset_name} 没有对应的训练数据集损失函数")
                
                # 更新测试集损失函数（如果存在测试集且为多数据集）
                if test_dataset and isinstance(test_dataset, dict):
                    for dataset_name in test_dataset.keys():
                        data_source_id = data_source_mapping.get(dataset_name, f"ds_{hash(dataset_name) % 1000:03d}")  # 使用统一映射
                        # 查找对应的训练数据集损失函数
                        if dataset_name in loss:
                            corresponding_loss_name = type(loss[dataset_name]).__name__
                            try:
                                TrainingDatasetService.update_dataset_loss_function_by_source(
                                    task_id=task_id,
                                    data_source_id=data_source_id,
                                    split_type="test",
                                    loss_function=corresponding_loss_name
                                )
                                logger.info(f"更新多测试数据集损失函数: {data_source_id}-test -> {corresponding_loss_name} (绑定到 {data_source_id}-train)")
                            except Exception as e:
                                logger.warning(f"更新测试数据源 {data_source_id} 损失函数失败: {e}")
                        else:
                            logger.warning(f"测试数据集 {dataset_name} 没有对应的训练数据集损失函数")
                            
            else:
                # 单数据集情况：直接获取损失函数类名
                actual_loss_name = type(loss).__name__
                logger.info(f"更新单数据集损失函数信息: {actual_loss_name}")
                
                # 单数据集使用固定的数据源ID
                data_source_id = "1"
                
                # 更新训练数据集的损失函数
                if train_dataset:
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="train",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新单训练数据集损失函数: {data_source_id}-train -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新训练数据集损失函数失败: {e}")
                
                # 单数据集情况：验证集和测试集也使用相同的损失函数
                if eval_dataset:
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="eval",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新单验证数据集损失函数: {data_source_id}-eval -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新验证数据集损失函数失败: {e}")
                
                if test_dataset:
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="test",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新单测试数据集损失函数: {data_source_id}-test -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新测试数据集损失函数失败: {e}")
                    
        except Exception as e:
            logger.warning(f"更新数据集损失函数信息失败: {e}")
    
    # 4.5. 模型加载成功后，更新模型信息到数据库
    try:
        from .services.task_manager import task_manager
        
        # 使用前面获取的任务ID（已经从training_config或环境变量获取）
        if task_id:
            # 构建模型信息
            model_info_update = {
                "validation": {
                    "valid": True,
                    "message": "模型加载成功",
                    "details": {
                        "type": "validated",
                        "name": model_name
                    }
                },
                "recommended_training_types": [train_type],
                "compatibility": {
                    "supported": True,
                    "model_type": "loaded",
                    "notes": ["模型已成功加载"]
                }
            }
            
            # 获取模型维度（支持embedding和reranker模型）
            embedding_dimension = None
            if train_type == "embedding" and hasattr(model, 'get_sentence_embedding_dimension'):
                try:
                    embedding_dimension = model.get_sentence_embedding_dimension()
                    logger.info(f"获取到embedding模型维度: {embedding_dimension}")
                except Exception as dim_e:
                    logger.warning(f"获取embedding模型维度失败: {str(dim_e)}")
            elif train_type == "reranker":
                # 对于reranker模型，尝试多种方法获取维度
                try:
                    # 方法1: 通过模型的tokenizer和config获取hidden_size
                    if hasattr(model, 'model') and hasattr(model.model, 'config') and hasattr(model.model.config, 'hidden_size'):
                        embedding_dimension = model.model.config.hidden_size
                        logger.info(f"获取到reranker模型维度 (方法1): {embedding_dimension}")
                    # 方法2: 通过encode方法测试获取维度
                    elif hasattr(model, 'encode'):
                        test_texts = ["test"]
                        try:
                            # 某些reranker模型的encode方法返回embedding
                            test_embedding = model.encode(test_texts)
                            if hasattr(test_embedding, 'shape') and len(test_embedding.shape) > 1:
                                embedding_dimension = test_embedding.shape[1]
                                logger.info(f"获取到reranker模型维度 (方法2): {embedding_dimension}")
                        except:
                            pass
                    # 方法3: 检查是否有classifier层来推断维度
                    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
                        embedding_dimension = model.classifier.in_features
                        logger.info(f"获取到reranker模型维度 (方法3): {embedding_dimension}")
                    
                    if not embedding_dimension:
                        logger.info("无法自动获取reranker模型维度，将使用默认值或跳过")
                        
                except Exception as dim_e:
                    logger.warning(f"获取reranker模型维度失败: {str(dim_e)}")
            
            # 如果获取到了维度，添加到模型信息中
            if embedding_dimension:
                model_info_update["embedding_dimension"] = embedding_dimension
            
            # 更新到数据库
            task_manager.update_model_info_after_loading(task_id, model_info_update)
        else:
            logger.info("未找到任务ID，跳过模型信息更新")
    except Exception as update_e:
        logger.warning(f"更新模型信息到数据库失败，不影响训练继续: {str(update_e)}")
    
    # 输出 Tensorboard 日志路径信息
    if (training_config.get('report_to') == ReportTo.TENSORBOARD or 
        training_config.get('report_to') == ReportTo.TENSORBOARD.value):
        logger.info(f"🔥 Tensorboard 已启用!")
        
        # 检查用户是否明确指定了 logging_dir
        user_logging_dir = training_config.get("user_logging_dir") if training_config else None
        if user_logging_dir:
            logger.info(f"📊 Tensorboard 日志目录（用户指定）: {user_logging_dir}")
            logger.info(f"🌐 启动 Tensorboard 命令: tensorboard --logdir=\"{user_logging_dir}\" --host=127.0.0.1 --port=6006")
        else:
            # HuggingFace 自动生成了 logging_dir，显示实际路径
            if hasattr(args, 'logging_dir') and args.logging_dir:
                actual_log_dir = str(args.logging_dir).replace('\\', '/')  # 统一使用正斜杠
                logger.info(f"📊 Tensorboard 日志目录（HuggingFace 自动生成）: {actual_log_dir}")
                logger.info(f"🌐 启动 Tensorboard 命令: tensorboard --logdir=\"{actual_log_dir}\" --host=127.0.0.1 --port=6006")
            else:
                logger.info(f"📊 Tensorboard 日志目录: HuggingFace 将自动在 {output_dir} 下创建 runs/<时间戳> 目录")
                logger.info(f"🌐 启动 Tensorboard 命令: tensorboard --logdir=\"{output_dir}/runs\" --host=127.0.0.1 --port=6006")
                logger.info(f"💡 提示: 训练开始后查看 {output_dir}/runs 目录下的实际日志文件夹")
        
        logger.info(f"🔗 访问地址: http://127.0.0.1:6006")

    # 5. 统一的评估器创建
    from .utils.evaluation import UnifiedEvaluator
    evaluator_factory = UnifiedEvaluator(train_type)
    evaluators = evaluator_factory.create_evaluators_from_datasets(
        eval_dataset, test_dataset, target_column, run_name
    )
    
    # 5.1. 评估器创建后，更新数据集信息中的评估器类型
    if task_id:
        try:
            from .utils.data_loader import DataLoader
            hf_subset = training_config_dict.get('HF_subset') if training_config_dict else None
            data_loader = DataLoader(
                hf_subset=hf_subset,
                train_sample_size=training_config_dict.get('train_sample_size', 0),
                eval_sample_size=training_config_dict.get('eval_sample_size', 0),
                test_sample_size=training_config_dict.get('test_sample_size', 0)
            )
            
            # 更新验证数据集的评估器类型
            if eval_dataset and evaluators.get('dev'):
                evaluator = evaluators['dev']
                
                # 检查是否为多数据集
                if data_loader.is_multi_dataset(eval_dataset):
                    # 多数据集情况：从SequentialEvaluator中提取子评估器
                    if isinstance(eval_dataset, dict) and hasattr(evaluator, 'evaluators'):
                        # SequentialEvaluator包含多个子评估器
                        dataset_names = list(eval_dataset.keys())
                        sub_evaluators = evaluator.evaluators
                        
                        for dataset_name, sub_evaluator in zip(dataset_names, sub_evaluators):
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    sub_evaluator_name = type(sub_evaluator).__name__
                                    
                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="eval",
                                        evaluator=sub_evaluator_name
                                    )
                                    logger.info(f"更新多数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {sub_evaluator_name}")
                                else:
                                    logger.warning(f"找不到数据集 {dataset_name} 的数据源映射")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                    else:
                        # 不是SequentialEvaluator，所有数据集使用同一个评估器
                        evaluator_name = type(evaluator).__name__
                        for dataset_name in eval_dataset.keys():
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="eval",
                                        evaluator=evaluator_name
                                    )
                                    logger.info(f"更新多数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                else:
                    # 单数据集情况：为该数据源更新评估器类型
                    evaluator_name = type(evaluator).__name__
                    try:
                        # 单数据集也要通过data_source_id更新
                        for dataset_name, data_source_id in data_source_mapping.items():
                            TrainingDatasetService.update_dataset_evaluator_by_source(
                                task_id=task_id,
                                data_source_id=data_source_id,
                                split_type="eval",
                                evaluator=evaluator_name
                            )
                            logger.info(f"更新单验证数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                    except Exception as e:
                        logger.warning(f"更新单验证数据集评估器类型失败: {e}")
                    
        except Exception as e:
            logger.warning(f"更新验证数据集评估器类型失败: {e}")

            # 更新测试数据集的评估器类型  
            if test_dataset and evaluators.get('test'):
                evaluator = evaluators['test']
                
                # 检查是否为多数据集
                if data_loader.is_multi_dataset(test_dataset):
                    # 多数据集情况：从SequentialEvaluator中提取子评估器
                    if isinstance(test_dataset, dict) and hasattr(evaluator, 'evaluators'):
                        # SequentialEvaluator包含多个子评估器
                        dataset_names = list(test_dataset.keys())
                        sub_evaluators = evaluator.evaluators
                        
                        for dataset_name, sub_evaluator in zip(dataset_names, sub_evaluators):
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    sub_evaluator_name = type(sub_evaluator).__name__
                                    
                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="test",
                                        evaluator=sub_evaluator_name
                                    )
                                    logger.info(f"更新多数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {sub_evaluator_name}")
                                else:
                                    logger.warning(f"找不到数据集 {dataset_name} 的数据源映射")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                    else:
                        # 不是SequentialEvaluator，所有数据集使用同一个评估器
                        evaluator_name = type(evaluator).__name__
                        for dataset_name in test_dataset.keys():
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="test",
                                        evaluator=evaluator_name
                                    )
                                    logger.info(f"更新多数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                else:
                    # 单数据集情况：为该数据源更新评估器类型
                    evaluator_name = type(evaluator).__name__
                    try:
                        # 单数据集也要通过data_source_id更新
                        for dataset_name, data_source_id in data_source_mapping.items():
                            TrainingDatasetService.update_dataset_evaluator_by_source(
                                task_id=task_id,
                                data_source_id=data_source_id,
                                split_type="test",
                                evaluator=evaluator_name
                            )
                            logger.info(f"更新单测试数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                    except Exception as e:
                        logger.warning(f"更新单测试数据集评估器类型失败: {e}")
                
        except Exception as e:
            logger.warning(f"更新数据集评估器类型信息失败: {e}")

    # 6. 评估基线模型
    dev_evaluator = None
    if eval_dataset is not None:
        # 检查是否为多数据集
        if data_loader.is_multi_dataset(eval_dataset):
            # 多数据集：创建 SequentialEvaluator
            dev_evaluator = evaluator_factory.create_multi_evaluator(
                eval_dataset, target_column, run_name
            )
        elif 'dev' in evaluators:
            # 单数据集：使用单个评估器
            dev_evaluator = evaluators['dev']
        
        # 评估基线模型
        if dev_evaluator is not None:
            dev_results = evaluator_factory.evaluate_model(model, dev_evaluator)
            print(f"Base model dev results: {dev_results}")
            
            # 保存基线评估结果到数据库
            if task_id and dev_results:
                try:
                    # 获取验证集的数据集ID
                    eval_datasets = TrainingDatasetService.get_datasets_by_job_and_split(task_id, "eval")
                    for eval_dataset_info in eval_datasets:
                        dataset_id = eval_dataset_info["id"]
                        dataset_name = eval_dataset_info["dataset_name"]
                        
                        TrainingDatasetService.update_eval_results(
                            dataset_id=dataset_id,
                            base_results=dev_results
                        )
                        logger.info(f"✅ 基线验证结果已保存: {dataset_name}")
                except Exception as e:
                    logger.warning(f"保存基线验证结果失败: {e}")
        else:
            logger.info("没有有效的验证评估器，跳过基线模型验证集评估")
    
    # 训练前评估测试集基线
    base_test_evaluator = None
    if test_dataset is not None:
        # 检查是否为多数据集
        if data_loader.is_multi_dataset(test_dataset):
            # 多数据集：创建 SequentialEvaluator
            base_test_evaluator = evaluator_factory.create_multi_evaluator(
                test_dataset, target_column, run_name
            )
        elif 'test' in evaluators:
            # 单数据集：使用单个评估器
            base_test_evaluator = evaluators['test']
        
        # 评估基线模型
        if base_test_evaluator is not None:
            base_test_results = evaluator_factory.evaluate_model(model, base_test_evaluator)
            print(f"Base model test results: {base_test_results}")
            
            # 保存基线评估结果到数据库
            if task_id and base_test_results:
                try:
                    # 获取测试集的数据集ID
                    test_datasets = TrainingDatasetService.get_datasets_by_job_and_split(task_id, "test")
                    for test_dataset_info in test_datasets:
                        dataset_id = test_dataset_info["id"]
                        dataset_name = test_dataset_info["dataset_name"]
                        
                        TrainingDatasetService.update_eval_results(
                            dataset_id=dataset_id,
                            base_results=base_test_results
                        )
                        logger.info(f"✅ 基线测试结果已保存: {dataset_name}")
                except Exception as e:
                    logger.warning(f"保存基线测试结果失败: {e}")
        else:
            logger.info("没有有效的测试评估器，跳过基线模型测试集评估")

    # 更新训练状态为运行中
    if task_id:
        try:
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            from .enums import TrainingStatus
            training_task_service.update_task_status(task_id, TrainingStatus.RUNNING.value)
            logger.info("训练状态已更新为运行中")
        except Exception as e:
            logger.warning(f"更新训练状态失败（不影响训练）: {e}")

    # 7. 创建训练器并开始训练
    trainer = _create_trainer(train_type, model, args, train_dataset, eval_dataset, loss, dev_evaluator)
    
    # 添加数据保存回调
    if task_id:
        # 🔧 重要：保存真正的原始方法，避免无限递归
        original_log = getattr(trainer, 'log', None)
        # 如果log已经被包装过，尝试找到真正的原始方法
        if hasattr(original_log, '__wrapped__'):
            original_log = getattr(original_log, '__wrapped__', original_log)
        
        def wrapped_log(logs, start_time=None):
            nonlocal step_count  # 确保可以修改外层的step_count
            try:
                # 调用原始log方法
                if original_log and callable(original_log):
                    result = original_log(logs, start_time)
                else:
                    result = None
                
                # ✅ 启用loss本地文件保存功能
                if logs and any(key in logs for key in ['train_loss', 'eval_loss', 'loss']):
                    try:
                        from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
                        
                        # 如果logs中没有step，使用并增加step_count
                        if 'step' not in logs:
                            step_count += 1
                        
                        # 获取当前步数和epoch信息
                        current_step = logs.get('step', step_count)  # 使用step_count作为fallback
                        current_epoch = logs.get('epoch', None)
                        
                        # 构建loss指标字典
                        loss_metrics = {}
                        for key, value in logs.items():
                            if 'loss' in key.lower() or key in ['accuracy', 'f1', 'precision', 'recall']:
                                loss_metrics[key] = value
                        
                        if loss_metrics:
                            # 获取loss管理器并保存记录
                            loss_manager = get_loss_manager(output_dir, task_id)
                            loss_manager.save_loss_record(current_step, loss_metrics, current_epoch)
                            
                            logger.debug(f"📊 Loss已保存到本地文件: step={current_step}, metrics={list(loss_metrics.keys())}")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ 保存loss到本地文件失败: {e}")
                
                # 保存训练评估器的评估结果
                if logs and dev_evaluator is not None:
                    try:
                        # 检查是否有评估结果（各种可能的键名）
                        eval_keys = [k for k in logs.keys() if any(eval_word in k.lower() for eval_word in ['eval', 'dev', 'accuracy', 'spearman', 'pearson'])]
                        if eval_keys:
                            current_step = logs.get('step', step_count)  # 使用step_count作为fallback
                            current_epoch = logs.get('epoch', 0)
                            
                            # 提取评估结果
                            eval_results = {k: logs[k] for k in eval_keys if k not in ['eval_loss']}
                            if eval_results:
                                # 获取验证集数据集ID并保存评估结果
                                eval_datasets = TrainingDatasetService.get_datasets_by_job_and_split(task_id, "eval")
                                for eval_dataset_info in eval_datasets:
                                    dataset_id = eval_dataset_info["id"]
                                    TrainingDatasetService.add_training_evaluator_evaluation(
                                        dataset_id=dataset_id,
                                        eval_results=eval_results,
                                        step=current_step,
                                        epoch=current_epoch
                                    )
                                
                                logger.debug(f"💾 训练评估结果已保存: step={current_step}, results={eval_results}")
                    except Exception as e:
                        logger.warning(f"保存训练评估结果失败: {e}")
                
                return result
            except Exception as e:
                logger.warning(f"回调函数执行失败: {e}")
                # 🛡️ 递归保护：确保原始功能不受影响，但避免无限递归
                if original_log and callable(original_log) and original_log != wrapped_log:
                    try:
                        return original_log(logs, start_time)
                    except Exception as inner_e:
                        logger.error(f"原始log方法也失败: {inner_e}")
                        return None
                return None
        
        # 替换log方法，并添加__wrapped__属性用于递归保护
        wrapped_log.__wrapped__ = original_log
        trainer.log = wrapped_log
    
    # 如果有进度回调，使用更可靠的回调机制
    if progress_callback:
        step_count = 0
        max_steps = args.max_steps if hasattr(args, 'max_steps') and args.max_steps > 0 else None
        if max_steps is None and hasattr(args, 'num_train_epochs'):
            # 更准确的步数估算 - 添加详细调试
            try:
                # 详细调试信息
                logger.info(f"🔍 调试数据集大小计算:")
                logger.info(f"   train_dataset类型: {type(train_dataset)}")
                logger.info(f"   train_dataset有__len__: {hasattr(train_dataset, '__len__')}")
                logger.info(f"   train_dataset是dict: {isinstance(train_dataset, dict)}")
                
                # 检查trainer内部数据集
                if hasattr(trainer, 'train_dataset'):
                    logger.info(f"   trainer.train_dataset类型: {type(trainer.train_dataset)}")
                    logger.info(f"   trainer.train_dataset有__len__: {hasattr(trainer.train_dataset, '__len__')}")
                    if hasattr(trainer.train_dataset, '__len__'):
                        logger.info(f"   trainer.train_dataset大小: {len(trainer.train_dataset)}")
                else:
                    logger.info(f"   trainer没有train_dataset属性")
                
                # 优先使用trainer内部数据集进行计算
                actual_dataset = train_dataset
                if hasattr(trainer, 'train_dataset') and trainer.train_dataset is not None:
                    actual_dataset = trainer.train_dataset
                    logger.info("🎯 使用trainer.train_dataset进行步数计算")
                else:
                    logger.info("🎯 使用原始train_dataset进行步数计算")
                
                if hasattr(actual_dataset, '__len__'):
                    dataset_size = len(actual_dataset)
                    logger.info(f"✅ 单数据集大小: {dataset_size}")
                elif isinstance(actual_dataset, dict):
                    # 多数据集情况
                    logger.info(f"   字典键: {list(actual_dataset.keys())}")
                    individual_sizes = []
                    for name, ds in actual_dataset.items():
                        logger.info(f"   数据集 '{name}' 类型: {type(ds)}")
                        if hasattr(ds, '__len__'):
                            size = len(ds)
                            individual_sizes.append(f"{name}: {size}")
                            logger.info(f"   数据集 '{name}' 大小: {size}")
                        else:
                            individual_sizes.append(f"{name}: 无法计算")
                            logger.info(f"   数据集 '{name}' 无法计算大小")
                    dataset_size = sum(len(ds) for ds in actual_dataset.values() if hasattr(ds, '__len__'))
                    logger.info(f"✅ 多数据集详情: {', '.join(individual_sizes)}, 总计: {dataset_size}")
                else:
                    dataset_size = 1000  # 默认值
                    logger.warning(f"⚠️ 无法确定数据集大小，实际数据集类型: {type(actual_dataset)}，使用默认值1000")
                
                # 获取GPU数量
                import torch
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                if hasattr(args, 'device') and args.device and ',' in str(args.device):
                    # 如果指定了多个设备，计算设备数量
                    num_gpus = len(str(args.device).split(','))
                
                batch_size = getattr(args, 'per_device_train_batch_size', 16)
                gradient_accumulation = getattr(args, 'gradient_accumulation_steps', 1)
                effective_batch_size = batch_size * gradient_accumulation * num_gpus
                steps_per_epoch = max(1, dataset_size // effective_batch_size)
                calculated_max_steps = steps_per_epoch * args.num_train_epochs
                logger.info(f"数据集大小: {dataset_size}, GPU数量: {num_gpus}, 每设备批次: {batch_size}, 梯度累积: {gradient_accumulation}")
                logger.info(f"有效批次大小: {effective_batch_size}, 每轮步数: {steps_per_epoch}, 计算出的总步数: {calculated_max_steps}")
                
                # 🔧 修复epoch显示0.0的问题：确保合理的训练步数
                if calculated_max_steps < 10:  # 如果步数过少，保持epoch-based训练
                    # logger.warning(f"⚠️ 计算出的训练步数过少({calculated_max_steps})，保持epoch-based训练")
                    # 确保epoch-based训练的参数设置正确
                    args.max_steps = -1  # 禁用step-based训练
                    # num_train_epochs 保持原值，不修改
                    logger.info(f"✅ 使用epoch-based训练: {args.num_train_epochs} epochs (max_steps设为-1)")
                else:
                    # 步数合理，使用step-based训练
                    args.max_steps = calculated_max_steps
                    args.num_train_epochs = -1  # 禁用epoch-based训练
                    logger.info(f"✅ 使用step-based训练: {calculated_max_steps} steps (num_train_epochs设为-1)")
            except Exception as e:
                logger.warning(f"无法估算训练步数: {e}, 使用默认值")
                max_steps = 1000
        
        # 🔍 最终参数调试信息
        final_max_steps = getattr(args, 'max_steps', None)
        final_num_epochs = getattr(args, 'num_train_epochs', None)
        logger.info(f"🎯 最终训练参数: max_steps={final_max_steps}, num_train_epochs={final_num_epochs}")
        
        # 使用更可靠的回调机制 - 同时包装多个方法
        original_log = trainer.log if hasattr(trainer, 'log') else None
        original_training_step = None
        
        # 尝试包装训练步方法（更直接的进度追踪）
        if hasattr(trainer, 'training_step'):
            original_training_step = trainer.training_step
            
            def wrapped_training_step(*args, **kwargs):
                nonlocal step_count
                result = original_training_step(*args, **kwargs)
                step_count += 1
                
                # 添加调试日志
                if step_count % 10 == 0:
                    logger.info(f"🔧 training_step被调用: 第{step_count}步")
                
                if max_steps and step_count <= max_steps:
                    try:
                        progress_callback(step_count, max_steps, "训练中")
                        if step_count % 10 == 0:
                            logger.info(f"📞 progress_callback调用成功: {step_count}/{max_steps}")
                    except KeyboardInterrupt:
                        logger.info("检测到训练停止信号，中断训练")
                        if hasattr(trainer, 'state'):
                            trainer.state.should_epoch_stop = True
                            trainer.state.should_training_stop = True
                        raise
                    except Exception as e:
                        logger.error(f"❌ progress_callback调用失败: {e}")
                
                return result
            
            trainer.training_step = wrapped_training_step
        
        # 整合进度回调到已有的log包装中
        if hasattr(trainer, 'log') and hasattr(trainer.log, '__wrapped__'):
            # 如果log已经被包装（数据保存回调），则添加进度功能
            existing_wrapped_log = trainer.log
            existing_original_log = getattr(existing_wrapped_log, '__wrapped__', None)
            
            def combined_wrapped_log(logs, start_time=None):
                nonlocal step_count
                result = None
                
                # 首先调用原始log方法（避免递归）
                try:
                    if existing_original_log and callable(existing_original_log):
                        result = existing_original_log(logs, start_time)
                except Exception as e:
                    logger.error(f"原始log方法调用失败: {e}")
                
                # ✅ 启用loss本地文件保存功能
                try:
                    if logs and task_id and any(key in logs for key in ['train_loss', 'eval_loss', 'loss']):
                        from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
                        
                        # 如果logs中没有step，使用并增加step_count
                        if 'step' not in logs:
                            step_count += 1
                        
                        # 获取当前步数和epoch信息
                        current_step = logs.get('step', step_count)  # 使用step_count作为fallback
                        current_epoch = logs.get('epoch', None)
                        
                        # 构建loss指标字典
                        loss_metrics = {}
                        for key, value in logs.items():
                            if 'loss' in key.lower() or key in ['accuracy', 'f1', 'precision', 'recall']:
                                loss_metrics[key] = value
                        
                        if loss_metrics:
                            # 获取loss管理器并保存记录
                            loss_manager = get_loss_manager(output_dir, task_id)
                            loss_manager.save_loss_record(current_step, loss_metrics, current_epoch)
                            
                            logger.debug(f"📊 Loss已保存到本地文件: step={current_step}, metrics={list(loss_metrics.keys())}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ 保存loss到本地文件失败: {e}")
                
                # 最后执行进度回调
                try:
                    # 🔧 处理进度回调（优先使用step信息，fallback到step_count）
                    if logs and 'step' in logs:
                        current_step = logs['step']
                        if max_steps and current_step <= max_steps:
                            progress_callback(current_step, max_steps, "训练中")
                            if current_step % 10 == 0:
                                logger.info(f"📞 log进度回调成功: {current_step}/{max_steps}")
                    elif not original_training_step:
                        # 如果logs中没有step信息且training_step未被包装，使用step_count
                        step_count += 1
                        if step_count % 10 == 0:
                            logger.info(f"🔧 log方法被调用: 第{step_count}步")
                        
                        if max_steps and step_count <= max_steps:
                            progress_callback(step_count, max_steps, "训练中")
                            if step_count % 10 == 0:
                                logger.info(f"📞 log中的progress_callback调用成功: {step_count}/{max_steps}")
                except KeyboardInterrupt:
                    logger.info("检测到训练停止信号，中断训练")
                    if hasattr(trainer, 'state'):
                        trainer.state.should_epoch_stop = True
                        trainer.state.should_training_stop = True
                    raise
                except Exception as e:
                    logger.error(f"❌ 进度回调失败: {e}")
                
                return result
            
            combined_wrapped_log.__wrapped__ = existing_original_log
            trainer.log = combined_wrapped_log
        
        logger.info(f"已设置训练进度回调，预估总步数: {max_steps}")
        # 初始进度更新
        try:
            progress_callback(0, max_steps or 1, "开始训练")
            logger.info("✅ 初始进度回调调用成功")
        except Exception as e:
            logger.error(f"❌ 初始进度回调调用失败: {e}")
            
        # 调试信息
        if hasattr(trainer, 'training_step'):
            logger.info("✅ 训练器有training_step方法，已包装")
        else:
            logger.warning("⚠️ 训练器没有training_step方法")
            
        if hasattr(trainer, 'log'):
            logger.info("✅ 训练器有log方法，已包装")
        else:
            logger.warning("⚠️ 训练器没有log方法")
    
    # 训练前的最后设备检查
    logger.info(f"🔍 训练开始前的设备检查:")
    logger.info(f"   模型设备: {getattr(model, 'device', 'unknown')}")
    if hasattr(trainer, 'model'):
        logger.info(f"   训练器模型设备: {getattr(trainer.model, 'device', 'unknown')}")
    
    # 训练前最终验证trainer内部的数据集格式
    if hasattr(trainer, 'train_dataset'):
        internal_train_dataset = trainer.train_dataset
        if isinstance(internal_train_dataset, dict):
            logger.info(f"🔍 Trainer内部多数据集验证:")
            for name, dataset in internal_train_dataset.items():
                logger.info(f"   数据集 '{name}': {len(dataset.column_names)}列 - {dataset.column_names}")
        else:
            logger.info(f"🔍 Trainer内部单数据集验证: {len(internal_train_dataset.column_names)}列 - {internal_train_dataset.column_names}")
    
    # 检查trainer的签名列设置
    if hasattr(trainer, '_signature_columns'):
        logger.info(f"🔍 Trainer签名列: {trainer._signature_columns}")
    else:
        # 强制设置签名列
        trainer._set_signature_columns_if_needed()
        logger.info(f"🔍 Trainer设置后的签名列: {trainer._signature_columns}")
    
    # 检查remove_unused_columns设置
    logger.info(f"🔍 remove_unused_columns设置: {trainer.args.remove_unused_columns}")
    
    # 强制禁用remove_unused_columns以确保数据格式控制
    if trainer.args.remove_unused_columns:
        logger.info("🔧 强制设置 remove_unused_columns=False 以确保数据格式控制")
        trainer.args.remove_unused_columns = False
    
    # 检查trainer的输出目录设置
    logger.info(f"🔍 Trainer输出目录: {trainer.args.output_dir}")
    
    # 确保输出目录存在
    import os
    if not os.path.exists(trainer.args.output_dir):
        os.makedirs(trainer.args.output_dir, exist_ok=True)
        logger.info(f"🔧 创建输出目录: {trainer.args.output_dir}")
    
    # 创建eval子目录（CrossEncoderCorrelationEvaluator需要）
    eval_dir = os.path.join(trainer.args.output_dir, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
        logger.info(f"🔧 创建评估输出目录: {eval_dir}")
    
    logger.info(f"✅ 使用output_dir: {trainer.args.output_dir}")
    logger.info(f"✅ 使用eval_dir: {eval_dir}")
    
    # 为评估器预创建可能需要的子目录结构
    # CrossEncoderCorrelationEvaluator 会创建以评估器名称命名的子目录
    try:
        # 获取模型名称，用于构建评估器目录名 - 需要与评估器实际使用的名称一致
        model_short_name = os.path.basename(trainer.model.config.name_or_path) if hasattr(trainer.model, 'config') and hasattr(trainer.model.config, 'name_or_path') else "model"
        logger.info(f"🔍 模型名称调试: {model_short_name}")
        
        # 可能的评估器目录名变体 - 覆盖不同的命名规则
        potential_eval_subdirs = [
            # 原始格式
            f"CrossEncoderCorrelationEvaluator_{model_short_name}",
            f"CrossEncoderClassificationEvaluator_{model_short_name}",
            # 带reranker前缀的格式
            f"CrossEncoderCorrelationEvaluator_reranker-{model_short_name}",
            f"CrossEncoderClassificationEvaluator_reranker-{model_short_name}",
            # 带sentence-transformers后缀的格式
            f"CrossEncoderCorrelationEvaluator_{model_short_name}-sentence-transformers",
            f"CrossEncoderCorrelationEvaluator_reranker-{model_short_name}-sentence-transformers",
        ]
        
        for subdir in potential_eval_subdirs:
            eval_subdir_path = os.path.join(eval_dir, subdir)
            os.makedirs(eval_subdir_path, exist_ok=True)
            logger.info(f"🔧 预创建评估器目录: {subdir}")
        logger.info(f"✅ 预创建评估器子目录完成")
    except Exception as e:
        logger.warning(f"预创建评估器子目录失败，但继续训练: {e}")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("训练被用户停止")
        raise
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

    # 8. 训练完成后的最终评估
    if progress_callback:
        progress_callback(1, 1, "评估模型性能")
    
    # 8.1 验证集最终评估
    if eval_dataset is not None:
        # 检查是否为多数据集
        if data_loader.is_multi_dataset(eval_dataset):
            # 多数据集：创建 SequentialEvaluator
            final_eval_evaluator = evaluator_factory.create_multi_evaluator(
                eval_dataset, target_column, run_name
            )
        elif 'dev' in evaluators:
            # 单数据集：使用单个评估器
            final_eval_evaluator = evaluators['dev']
        else:
            final_eval_evaluator = None
        
        # 评估训练后验证集
        if final_eval_evaluator is not None:
            final_eval_results = evaluator_factory.evaluate_model(model, final_eval_evaluator)
            print(f"Trained model eval results: {final_eval_results}")
            
            # 保存验证集最终评估结果到数据库
            if task_id and final_eval_results:
                try:
                    # 获取验证集的数据集ID
                    eval_datasets = TrainingDatasetService.get_datasets_by_job_and_split(task_id, "eval")
                    for eval_dataset_info in eval_datasets:
                        dataset_id = eval_dataset_info["id"]
                        dataset_name = eval_dataset_info["dataset_name"]
                        
                        TrainingDatasetService.update_eval_results(
                            dataset_id=dataset_id,
                            final_results=final_eval_results
                        )
                        logger.info(f"✅ 验证集最终结果已保存: {dataset_name}")
                except Exception as e:
                    logger.warning(f"保存验证集最终结果失败: {e}")
        else:
            logger.info("没有有效的验证评估器，跳过验证集最终评估")
    
    # 8.2 测试集最终评估
    if test_dataset is not None:
        # 检查是否为多数据集
        if data_loader.is_multi_dataset(test_dataset):
            # 多数据集：创建 SequentialEvaluator
            test_evaluator = evaluator_factory.create_multi_evaluator(
                test_dataset, target_column, run_name
            )
        elif 'test' in evaluators:
            # 单数据集：使用单个评估器
            test_evaluator = evaluators['test']
        else:
            test_evaluator = None
        
        # 评估训练后模型
        if test_evaluator is not None:
            test_results = evaluator_factory.evaluate_model(model, test_evaluator)
            print(f"Trained model test results: {test_results}")
            
            # 保存最终评估结果到数据库
            if task_id and test_results:
                try:
                    # 获取测试集的数据集ID
                    test_datasets = TrainingDatasetService.get_datasets_by_job_and_split(task_id, "test")
                    for test_dataset_info in test_datasets:
                        dataset_id = test_dataset_info["id"]
                        dataset_name = test_dataset_info["dataset_name"]
                        
                        TrainingDatasetService.update_eval_results(
                            dataset_id=dataset_id,
                            final_results=test_results
                        )
                        logger.info(f"✅ 最终测试结果已保存: {dataset_name}")
                except Exception as e:
                    logger.warning(f"保存最终测试结果失败: {e}")
        else:
            logger.info("没有有效的测试评估器，跳过测试集评估")

    # 9. 保存模型
    if progress_callback:
        progress_callback(1, 1, "保存模型")
        
    save_dir = os.path.join(output_dir, "final_model")
    try:
        logger.info(f"开始保存模型到: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"创建保存目录成功: {save_dir}")
        
        # 保存模型
        model.save_pretrained(save_dir)
        logger.info(f"✅ 模型保存成功: {save_dir}")
        
        # 检查保存的文件
        saved_files = os.listdir(save_dir)
        logger.info(f"保存的文件列表: {saved_files}")
        
        # push_to_hub(model, model_name, save_dir)

        # 训练成功完成，从临时文件管理器中移除输出目录（因为这是成功的输出）
        temp_file_manager.unregister_temp_dir(output_dir)
        
        # 🔧 更新训练状态为已完成（只有在子进程中运行时才更新，避免重复更新）
        if task_id and not hasattr(training_config, '_is_multiprocess_child'):
            try:
                # 使用统一的任务管理器更新逻辑
                from .services.task_manager import task_manager
                task_manager.complete_task(task_id, save_dir)
                task_manager.update_task_progress(task_id, 100.0, "训练完成")
                
                # 🔧 更新数据库任务状态为SUCCEEDED（与unified_training_service保持一致）
                from .enums import TrainingStatus
                from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                training_task_service.update_task_status(task_id, TrainingStatus.SUCCEEDED.value)
                training_task_service.update_task_result(task_id, final_model_path=save_dir)
                logger.info(f"✅ 数据库任务状态已更新为SUCCEEDED: {task_id}")
                
                # 更新进程状态为已完成
                from .enums.training_task_enums import ProcessStatus
                training_task_service.update_process_info(task_id, None, ProcessStatus.TERMINATED.value)
                
                logger.info(f"训练状态已更新为已完成，模型保存在: {save_dir}")
                
                # ✅ 完成loss管理器的最终化处理并保存汇总到数据库
                try:
                    from bubble_rag.training.model_sft.utils.loss_manager import cleanup_loss_manager
                    
                    # 获取最终训练指标
                    final_metrics = {
                        "final_model_path": save_dir,
                        "saved_files": saved_files,
                        "training_completed": True
                    }
                    
                    # 清理loss管理器并获取数据库汇总信息
                    loss_summary = cleanup_loss_manager(task_id, final_metrics)
                    
                    if loss_summary:
                        # 将loss汇总信息保存到数据库
                        try:
                            import json
                            loss_data_json = json.dumps(loss_summary, ensure_ascii=False)
                            training_task_service.update_task_result(task_id, loss_data=loss_data_json)
                            logger.info(f"✅ Loss汇总信息已保存到数据库: {len(loss_summary)} 项指标")
                        except Exception as db_e:
                            logger.warning(f"保存loss汇总到数据库失败: {db_e}")
                    
                    logger.info("✅ Loss管理器已完成最终化处理")
                except Exception as loss_e:
                    logger.warning(f"Loss管理器清理失败: {loss_e}")
                    
            except Exception as e:
                logger.warning(f"更新训练完成状态失败（不影响结果）: {e}")
        
        logger.info(f"训练完成，模型保存在: {save_dir}")
        return model, save_dir
        
    except Exception as e:
        # 更新训练状态为失败
        if task_id:
            try:
                from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                from .enums import TrainingStatus
                error_msg = f"模型保存失败: {str(e)}"
                training_task_service.update_task_status(task_id, TrainingStatus.FAILED.value)
                training_task_service.update_task_result(task_id, error_message=error_msg)
                logger.info("训练状态已更新为失败")
            except Exception as status_e:
                logger.warning(f"更新训练失败状态失败: {status_e}")
        
        logger.error(f"❌ 保存模型失败: {str(e)}", exc_info=True)
        logger.error(f"保存目录: {save_dir}")
        logger.error(f"输出目录: {output_dir}")
        raise Exception(f"模型保存失败: {str(e)}")
    
    except Exception as e:
        # 🔧 全局异常处理：确保任何训练失败都能正确更新任务状态
        if task_id:
            try:
                from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                from .enums import TrainingStatus
                error_msg = f"训练执行失败: {str(e)}"
                training_task_service.update_task_status(task_id, TrainingStatus.FAILED.value)
                training_task_service.update_task_result(task_id, error_message=error_msg)
                logger.info(f"✅ 全局异常处理：任务状态已更新为FAILED: {task_id}")
            except Exception as status_e:
                logger.warning(f"全局异常处理：更新训练失败状态失败: {status_e}")
        
        logger.error(f"❌ 训练执行失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()