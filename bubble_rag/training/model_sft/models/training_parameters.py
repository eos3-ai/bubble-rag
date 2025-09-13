"""
HuggingFace 训练参数管理器
统一管理训练参数，避免使用环境变量方式，改用结构化参数管理
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, validator
from enum import Enum

from ..enums.training_parameter_enums import (
    ReportTo, OptimType, LRSchedulerType, EvalStrategy, 
    SaveStrategy, LoggingStrategy
)

logger = logging.getLogger(__name__)



class TrainingParameters(BaseModel):
    """HuggingFace训练参数管理类"""
    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True,
        validate_assignment=True
    )
    
    # 基础训练参数
    num_train_epochs: int = Field(default=3, ge=1, description="训练轮数")
    per_device_train_batch_size: int = Field(default=16, ge=1, description="每设备训练批次大小")
    per_device_eval_batch_size: int = Field(default=16, ge=1, description="每设备评估批次大小")
    learning_rate: float = Field(default=5e-5, gt=0, description="学习率")
    weight_decay: float = Field(default=0.0, ge=0, description="权重衰减")
    
    # 学习率调度
    warmup_ratio: float = Field(default=0.1, ge=0, le=1, description="预热比例")
    lr_scheduler_type: LRSchedulerType = Field(default=LRSchedulerType.LINEAR, description="学习率调度器类型")
    
    # 优化器参数
    optim: OptimType = Field(default=OptimType.ADAMW_TORCH, description="优化器类型")
    adam_beta1: float = Field(default=0.9, ge=0, le=1, description="Adam beta1参数")
    adam_beta2: float = Field(default=0.999, ge=0, le=1, description="Adam beta2参数")
    adam_epsilon: float = Field(default=1e-8, gt=0, description="Adam epsilon参数")
    max_grad_norm: float = Field(default=1.0, gt=0, description="梯度裁剪最大值")
    
    # 混合精度
    bf16: bool = Field(default=False, description="是否使用bf16混合精度")
    fp16: bool = Field(default=False, description="是否使用fp16混合精度")
    
    # 评估和保存
    eval_strategy: EvalStrategy = Field(default=EvalStrategy.STEPS, description="评估策略")
    eval_steps: Optional[int] = Field(default=500, ge=1, description="评估步数间隔")
    save_strategy: SaveStrategy = Field(default=SaveStrategy.STEPS, description="保存策略")
    save_steps: Optional[int] = Field(default=500, ge=1, description="保存步数间隔")
    save_total_limit: int = Field(default=2, ge=1, description="保存模型的最大数量")
    
    # 日志记录
    logging_strategy: LoggingStrategy = Field(default=LoggingStrategy.STEPS, description="日志策略")
    logging_steps: int = Field(default=100, ge=1, description="日志记录步数间隔")
    logging_dir: Optional[str] = Field(default=None, description="日志目录")
    
    # 梯度累积和数据加载
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="梯度累积步数")
    dataloader_num_workers: int = Field(default=0, ge=0, description="数据加载器工作进程数")
    dataloader_drop_last: bool = Field(default=False, description="是否丢弃最后不完整的批次")
    
    # 训练控制
    max_steps: Optional[int] = Field(default=None, description="最大训练步数")
    seed: int = Field(default=42, description="随机种子")
    
    # 评估控制
    eval_accumulation_steps: Optional[int] = Field(default=None, description="评估累积步数")
    load_best_model_at_end: bool = Field(default=False, description="是否在训练结束时加载最佳模型")
    metric_for_best_model: Optional[str] = Field(default=None, description="用于判断最佳模型的指标")
    greater_is_better: Optional[bool] = Field(default=None, description="指标是否越大越好")
    
    # 恢复和检查点
    resume_from_checkpoint: Optional[str] = Field(default=None, description="从检查点恢复训练")
    ignore_data_skip: bool = Field(default=False, description="是否忽略数据跳过")
    
    # Hub相关
    push_to_hub: bool = Field(default=False, description="是否推送到Hub")
    hub_model_id: Optional[str] = Field(default=None, description="Hub模型ID")
    hub_strategy: str = Field(default="every_save", description="Hub推送策略")
    hub_token: Optional[str] = Field(default=None, description="Hub访问令牌")
    
    # 数据处理
    prediction_loss_only: bool = Field(default=False, description="是否只返回预测损失")
    remove_unused_columns: bool = Field(default=True, description="是否移除未使用的列")
    label_names: Optional[list] = Field(default=None, description="标签列名称列表")
    group_by_length: bool = Field(default=False, description="是否按长度分组")
    length_column_name: Optional[str] = Field(default=None, description="长度列名称")
    
    # 分布式训练
    local_rank: int = Field(default=-1, description="本地rank")
    nproc_per_node: int = Field(default=1, ge=1, description="每节点进程数")
    
    # 报告工具
    report_to: Optional[Union[ReportTo, str, list]] = Field(default=ReportTo.NONE, description="报告工具")
    
    # SwanLab相关参数（可选）
    swanlab_api_key: Optional[str] = Field(default=None, description="SwanLab API密钥")
    swanlab_workspace: Optional[str] = Field(default=None, description="SwanLab工作空间")
    swanlab_project: Optional[str] = Field(default=None, description="SwanLab项目名")
    swanlab_experiment: Optional[str] = Field(default=None, description="SwanLab实验名称")
    swanlab_mode: Optional[str] = Field(default=None, description="SwanLab模式 (local/cloud)")
    
    # 数据集采样参数
    train_sample_size: Optional[int] = Field(default=0, ge=0, le=10000000, description="训练数据集样本数量限制，0表示不限制")
    eval_sample_size: Optional[int] = Field(default=0, ge=0, le=10000000, description="验证数据集样本数量限制，0表示不限制")
    test_sample_size: Optional[int] = Field(default=0, ge=0, le=10000000, description="测试数据集样本数量限制，0表示不限制")
    
    # 运行时参数（不从配置文件加载）
    output_dir: str = Field(default="./output", description="输出目录")
    run_name: Optional[str] = Field(default=None, description="运行名称")
    user_logging_dir: Optional[str] = Field(default=None, description="用户指定的日志目录")
    
    @validator('bf16', 'fp16')
    def validate_precision(cls, v, values):
        """验证混合精度设置，确保不同时启用bf16和fp16"""
        if v and values.get('bf16' if 'fp16' in values else 'fp16'):
            raise ValueError("不能同时启用bf16和fp16")
        return v
    
    @validator('eval_steps')
    def validate_eval_steps(cls, v, values):
        """验证评估步数设置"""
        eval_strategy = values.get('eval_strategy')
        if eval_strategy == EvalStrategy.STEPS and v is None:
            raise ValueError("使用steps评估策略时必须设置eval_steps")
        return v
    
    @validator('save_steps')
    def validate_save_steps(cls, v, values):
        """验证保存步数设置"""
        save_strategy = values.get('save_strategy')
        if save_strategy == SaveStrategy.STEPS and v is None:
            raise ValueError("使用steps保存策略时必须设置save_steps")
        return v
    
    @validator('report_to')
    def validate_report_to(cls, v):
        """验证报告工具设置"""
        if v is None:
            return v
        
        # 如果是列表，验证每个元素
        if isinstance(v, list):
            valid_values = [item.value for item in ReportTo]
            for item in v:
                if isinstance(item, str) and item not in valid_values:
                    raise ValueError(f"无效的报告工具: {item}, 支持的工具: {valid_values}")
            return v
        
        # 如果是字符串，验证是否在枚举中
        if isinstance(v, str):
            valid_values = [item.value for item in ReportTo]
            if v not in valid_values:
                raise ValueError(f"无效的报告工具: {v}, 支持的工具: {valid_values}")
            
        return v


class TrainingParametersManager:
    """训练参数管理器"""
    
    def __init__(self, custom_params: Optional[Dict[str, Any]] = None):
        """
        初始化训练参数管理器
        
        Args:
            custom_params: 自定义参数字典
        """
        self.custom_params = custom_params or {}
        self._training_params: Optional[TrainingParameters] = None
    
    def load_from_config(self, config_dict: Dict[str, Any]) -> 'TrainingParametersManager':
        """
        从配置字典加载参数
        
        Args:
            config_dict: 配置字典
            
        Returns:
            self，支持链式调用
        """
        # 合并自定义参数
        merged_config = {**config_dict, **self.custom_params}
        
        try:
            self._training_params = TrainingParameters(**merged_config)
            logger.info("训练参数加载成功")
            self._log_parameters()
        except Exception as e:
            logger.error(f"训练参数验证失败: {e}")
            raise ValueError(f"无效的训练参数配置: {e}")
        
        return self
    
    def load_from_environment(self, env_prefix: str = "") -> 'TrainingParametersManager':
        """
        从环境变量加载参数（用于向后兼容）
        
        Args:
            env_prefix: 环境变量前缀
            
        Returns:
            self，支持链式调用
        """
        # HuggingFace训练参数映射
        hf_params_mapping = {
            "NUM_TRAIN_EPOCHS": "num_train_epochs",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "per_device_train_batch_size",
            "PER_DEVICE_EVAL_BATCH_SIZE": "per_device_eval_batch_size",
            "LEARNING_RATE": "learning_rate",
            "WARMUP_RATIO": "warmup_ratio",
            "LR_SCHEDULER_TYPE": "lr_scheduler_type",
            "BF16": "bf16",
            "FP16": "fp16",
            "EVAL_STRATEGY": "eval_strategy",
            "EVAL_STEPS": "eval_steps",
            "SAVE_STRATEGY": "save_strategy",
            "SAVE_STEPS": "save_steps",
            "SAVE_TOTAL_LIMIT": "save_total_limit",
            "LOGGING_STEPS": "logging_steps",
            "LOGGING_STRATEGY": "logging_strategy",
            "LOGGING_DIR": "logging_dir",
            "GRADIENT_ACCUMULATION_STEPS": "gradient_accumulation_steps",
            "MAX_STEPS": "max_steps",
            "DATALOADER_NUM_WORKERS": "dataloader_num_workers",
            "WEIGHT_DECAY": "weight_decay",
            "ADAM_BETA1": "adam_beta1",
            "ADAM_BETA2": "adam_beta2",
            "ADAM_EPSILON": "adam_epsilon",
            "MAX_GRAD_NORM": "max_grad_norm",
            "SEED": "seed",
            "DATALOADER_DROP_LAST": "dataloader_drop_last",
            "EVAL_ACCUMULATION_STEPS": "eval_accumulation_steps",
            "LOAD_BEST_MODEL_AT_END": "load_best_model_at_end",
            "METRIC_FOR_BEST_MODEL": "metric_for_best_model",
            "GREATER_IS_BETTER": "greater_is_better",
            "IGNORE_DATA_SKIP": "ignore_data_skip",
            "RESUME_FROM_CHECKPOINT": "resume_from_checkpoint",
            "PUSH_TO_HUB": "push_to_hub",
            "HUB_MODEL_ID": "hub_model_id",
            "HUB_STRATEGY": "hub_strategy",
            "HUB_TOKEN": "hub_token",
            "PREDICTION_LOSS_ONLY": "prediction_loss_only",
            "REMOVE_UNUSED_COLUMNS": "remove_unused_columns",
            "LABEL_NAMES": "label_names",
            "LOCAL_RANK": "local_rank",
            "OPTIM": "optim",
            "GROUP_BY_LENGTH": "group_by_length",
            "LENGTH_COLUMN_NAME": "length_column_name",
            "REPORT_TO": "report_to",
            "NPROC_PER_NODE": "nproc_per_node",
            # SwanLab相关参数
            "SWANLAB_API_KEY": "swanlab_api_key",
            "SWANLAB_WORKSPACE": "swanlab_workspace", 
            "SWANLAB_PROJECT": "swanlab_project",
            "SWANLAB_EXPERIMENT": "swanlab_experiment",
            "SWANLAB_MODE": "swanlab_mode",
            # 数据集采样参数
            "TRAIN_SAMPLE_SIZE": "train_sample_size",
            "EVAL_SAMPLE_SIZE": "eval_sample_size", 
            "TEST_SAMPLE_SIZE": "test_sample_size"
        }
        
        config = {}
        
        # 从环境变量读取参数
        for env_key, param_key in hf_params_mapping.items():
            full_env_key = f"{env_prefix}{env_key}" if env_prefix else env_key
            env_value = os.getenv(full_env_key)
            
            if env_value is not None:
                # 类型转换
                try:
                    if param_key in ['num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size',
                                   'gradient_accumulation_steps', 'eval_steps', 'save_steps', 'save_total_limit',
                                   'logging_steps', 'max_steps', 'dataloader_num_workers', 'eval_accumulation_steps',
                                   'seed', 'local_rank', 'nproc_per_node', 'train_sample_size', 'eval_sample_size', 'test_sample_size']:
                        config[param_key] = int(env_value)
                    elif param_key in ['learning_rate', 'warmup_ratio', 'weight_decay', 'adam_beta1', 'adam_beta2',
                                     'adam_epsilon', 'max_grad_norm']:
                        config[param_key] = float(env_value)
                    elif param_key in ['bf16', 'fp16', 'dataloader_drop_last', 'load_best_model_at_end',
                                     'greater_is_better', 'ignore_data_skip', 'push_to_hub', 'prediction_loss_only',
                                     'remove_unused_columns', 'group_by_length']:
                        config[param_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        config[param_key] = env_value
                        
                    logger.debug(f"从环境变量加载参数: {param_key} = {config[param_key]}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"环境变量 {full_env_key} 值转换失败: {e}")
        
        # 合并自定义参数
        config.update(self.custom_params)
        
        return self.load_from_config(config)
    
    def get_training_args_dict(self) -> Dict[str, Any]:
        """
        获取训练参数字典，用于传递给HuggingFace训练器
        
        Returns:
            训练参数字典
        """
        if self._training_params is None:
            raise ValueError("训练参数未初始化，请先调用load_from_config或load_from_environment")
        
        # 转换为字典并排除None值
        params_dict = self._training_params.model_dump(exclude_none=True, by_alias=True)
        
        # 特殊处理某些参数
        if 'report_to' in params_dict and isinstance(params_dict['report_to'], str):
            if params_dict['report_to'] in ['none', '']:
                params_dict['report_to'] = []
            else:
                params_dict['report_to'] = [params_dict['report_to']]
        
        return params_dict
    
    def get_training_parameters(self) -> TrainingParameters:
        """
        获取训练参数对象
        
        Returns:
            训练参数对象
        """
        if self._training_params is None:
            raise ValueError("训练参数未初始化，请先调用load_from_config或load_from_environment")
        
        return self._training_params
    
    def update_parameters(self, **kwargs) -> 'TrainingParametersManager':
        """
        更新训练参数
        
        Args:
            **kwargs: 要更新的参数
            
        Returns:
            self，支持链式调用
        """
        if self._training_params is None:
            raise ValueError("训练参数未初始化，请先调用load_from_config或load_from_environment")
        
        # 创建更新后的参数字典
        current_dict = self._training_params.model_dump()
        current_dict.update(kwargs)
        
        # 重新验证和创建参数对象
        try:
            self._training_params = TrainingParameters(**current_dict)
            logger.info(f"训练参数更新成功: {list(kwargs.keys())}")
        except Exception as e:
            logger.error(f"训练参数更新失败: {e}")
            raise ValueError(f"无效的参数更新: {e}")
        
        return self
    
    def _log_parameters(self):
        """记录当前训练参数"""
        if self._training_params is None:
            return
        
        logger.info("=== 训练参数配置 ===")
        params_dict = self._training_params.model_dump()
        
        # 按类别组织参数输出
        categories = {
            "基础参数": ["num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size", 
                       "learning_rate", "weight_decay"],
            "优化器": ["optim", "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm"],
            "学习率调度": ["lr_scheduler_type", "warmup_ratio"],
            "混合精度": ["bf16", "fp16"],
            "评估策略": ["eval_strategy", "eval_steps", "load_best_model_at_end"],
            "保存策略": ["save_strategy", "save_steps", "save_total_limit"],
            "日志记录": ["logging_strategy", "logging_steps", "logging_dir", "report_to"],
            "数据处理": ["gradient_accumulation_steps", "dataloader_num_workers", "dataloader_drop_last"],
            "数据集采样": ["train_sample_size", "eval_sample_size", "test_sample_size"],
            "SwanLab配置": ["swanlab_api_key", "swanlab_workspace", "swanlab_project", "swanlab_experiment", "swanlab_mode"]
        }
        
        for category, param_keys in categories.items():
            category_params = {k: v for k, v in params_dict.items() 
                             if k in param_keys and v is not None}
            if category_params:
                logger.info(f"{category}: {category_params}")