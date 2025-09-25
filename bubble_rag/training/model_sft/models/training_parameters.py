"""
训练参数管理器 - 升级到 Pydantic v2
明确区分官方训练参数和自定义参数
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum

from ..enums.training_parameter_enums import (
    ReportTo, OptimType, LRSchedulerType,
    EvalStrategy, SaveStrategy, LoggingStrategy, HubStrategy, SwanLabMode, DDPBackend
)

logger = logging.getLogger(__name__)



class TrainingParameters(BaseModel):
    """
    训练参数管理类 - 基于官方 CrossEncoderTrainingArguments 规范
    明确区分官方训练参数和自定义参数
    """
    model_config = ConfigDict(
        extra='forbid',  # 禁止额外字段，增加安全性
        validate_assignment=True,  # 赋值时验证
        str_strip_whitespace=True,  # 自动去除字符串空白
        protected_namespaces=(),  # 解决model_name_or_path冲突警告
    )

    # ============================================================================
    # 官方训练参数 (会传给 CrossEncoderTrainingArguments / SentenceTransformerTrainingArguments)
    # ============================================================================

    # === 基础训练参数 ===
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    overwrite_output_dir: bool = Field(default=False, description="是否覆盖输出目录")
    do_train: bool = Field(default=False, description="是否进行训练")
    do_eval: bool = Field(default=False, description="是否进行评估")
    do_predict: bool = Field(default=False, description="是否进行预测")
    num_train_epochs: float = Field(default=3.0, gt=0, description="训练轮次")
    per_device_train_batch_size: int = Field(default=8, ge=1, description="每设备训练批次大小")
    per_device_eval_batch_size: int = Field(default=8, ge=1, description="每设备评估批次大小")

    # === 学习率和优化器参数 ===
    learning_rate: float = Field(default=5e-5, gt=0, description="学习率")
    warmup_ratio: float = Field(default=0.0, ge=0, le=1, description="预热比例")
    warmup_steps: int = Field(default=0, ge=0, description="预热步数")
    weight_decay: float = Field(default=0.0, ge=0, description="权重衰减")
    adam_beta1: float = Field(default=0.9, ge=0, le=1, description="Adam beta1")
    adam_beta2: float = Field(default=0.999, ge=0, le=1, description="Adam beta2")
    adam_epsilon: float = Field(default=1e-8, gt=0, description="Adam epsilon")
    max_grad_norm: float = Field(default=1.0, gt=0, description="最大梯度范数")

    # === 训练策略参数 ===
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="梯度累积步数")
    eval_strategy: EvalStrategy = Field(default=EvalStrategy.NO, description="评估策略: no, steps, epoch")
    save_strategy: SaveStrategy = Field(default=SaveStrategy.STEPS, description="保存策略: no, steps, epoch")
    logging_strategy: LoggingStrategy = Field(default=LoggingStrategy.STEPS, description="日志策略: no, steps, epoch")
    logging_steps: float = Field(default=500, gt=0, description="日志记录步数")
    max_steps: int = Field(default=-1, description="最大训练步数，-1表示使用epoch")

    # === 步数参数（可选） ===
    eval_steps: Optional[float] = Field(default=None, gt=0, description="评估步数")
    save_steps: float = Field(default=500, gt=0, description="保存步数")
    save_total_limit: Optional[int] = Field(default=None, ge=1, description="保存模型数量限制")

    # === 数据加载参数 ===
    dataloader_num_workers: int = Field(default=0, ge=0, description="数据加载器工作线程数")
    dataloader_drop_last: bool = Field(default=False, description="丢弃最后一个不完整的batch")
    eval_accumulation_steps: Optional[int] = Field(default=None, ge=1, description="评估累积步数")

    # === 混合精度训练 ===
    bf16: bool = Field(default=False, description="使用bf16混合精度")
    fp16: bool = Field(default=False, description="使用fp16混合精度")
    gradient_checkpointing: bool = Field(default=False, description="使用梯度检查点")

    # === 模型评估和保存 ===
    load_best_model_at_end: Optional[bool] = Field(default=False, description="训练结束时加载最佳模型")
    metric_for_best_model: Optional[str] = Field(default=None, description="最佳模型指标")
    greater_is_better: Optional[bool] = Field(default=None, description="指标越大越好")

    # === 系统和调试参数 ===
    seed: int = Field(default=42, description="随机种子")
    disable_tqdm: Optional[bool] = Field(default=None, description="禁用进度条")
    report_to: Optional[Union[ReportTo, str, List[str]]] = Field(default=ReportTo.NONE, description="报告工具")
    run_name: Optional[str] = Field(default=None, description="运行名称")

    # === 数据处理参数 ===
    remove_unused_columns: Optional[bool] = Field(default=True, description="移除未使用的列")
    group_by_length: bool = Field(default=False, description="按长度分组")
    length_column_name: Optional[str] = Field(default="length", description="长度列名称")
    label_names: Optional[List[str]] = Field(default=None, description="标签名称列表")

    # === 学习率调度器 ===
    lr_scheduler_type: LRSchedulerType = Field(default=LRSchedulerType.LINEAR, description="学习率调度器类型")

    # === 优化器 ===
    optim: OptimType = Field(default=OptimType.ADAMW_TORCH, description="优化器类型")

    # === Hub相关参数 ===
    push_to_hub: bool = Field(default=False, description="推送到Hub")
    hub_model_id: Optional[str] = Field(default=None, description="Hub模型ID")
    hub_strategy: HubStrategy = Field(default=HubStrategy.EVERY_SAVE, description="Hub策略")
    hub_token: Optional[str] = Field(default=None, description="Hub token")

    # === 日志和检查点 ===
    logging_dir: Optional[str] = Field(default=None, description="日志目录")
    resume_from_checkpoint: Optional[str] = Field(default=None, description="从检查点恢复")

    # === 分布式训练 ===
    ddp_backend: Optional[DDPBackend] = Field(default=None, description="DDP后端")

    # === DeepSpeed ===
    deepspeed: Optional[str] = Field(default=None, description="DeepSpeed配置文件路径")

    # === 其他重要参数 ===
    prediction_loss_only: bool = Field(default=False, description="仅预测损失")
    ignore_data_skip: bool = Field(default=False, description="忽略数据跳过")

    # ============================================================================
    # 自定义参数 (不会传给官方 TrainingArguments，由我们的系统内部使用)
    # ============================================================================

    # === SwanLab配置 ===
    swanlab_api_key: Optional[str] = Field(default=None, description="SwanLab API密钥")
    swanlab_workspace: Optional[str] = Field(default=None, description="SwanLab工作空间")
    swanlab_project: Optional[str] = Field(default=None, description="SwanLab项目名")
    swanlab_experiment: Optional[str] = Field(default=None, description="SwanLab实验名称")
    swanlab_mode: Optional[SwanLabMode] = Field(default=None, description="SwanLab模式 (local/cloud)")

    # === 数据集采样参数 ===
    train_sample_size: Optional[int] = Field(default=-1, ge=-1, le=10000000, description="训练数据集样本数量限制，-1表示不限制，0表示不使用该数据集")
    eval_sample_size: Optional[int] = Field(default=-1, ge=-1, le=10000000, description="验证数据集样本数量限制，-1表示不限制，0表示不使用该数据集")
    test_sample_size: Optional[int] = Field(default=-1, ge=-1, le=10000000, description="测试数据集样本数量限制，-1表示不限制，0表示不使用该数据集")

    # === 运行时参数 ===
    user_logging_dir: Optional[str] = Field(default=None, description="前端展示用的日志目录，默认为{output_dir}/logs")
    user_eval_dir: Optional[str] = Field(default=None, description="前端展示用的评估结果目录，默认为{output_dir}/eval")


    # === 数据集和模型路径（由系统管理，不传给训练器）===
    dataset_name_or_path: Optional[str] = Field(default=None, description="数据集路径")
    model_name_or_path: Optional[str] = Field(default=None, description="模型路径")
    train_type: Optional[str] = Field(default=None, description="训练类型")
    task_id: Optional[str] = Field(default=None, description="任务ID")
    task_name: Optional[str] = Field(default=None, description="任务名称")
    description: Optional[str] = Field(default=None, description="任务描述")
    HF_subset: Optional[str] = Field(default=None, description="HuggingFace数据集子集")
    device: Optional[str] = Field(default=None, description="设备信息")

    # === 模型参数（不传递给训练器，由系统内部处理）===
    max_seq_length: Optional[int] = Field(default=512, gt=0, description="最大序列长度")
    



    @field_validator('report_to')
    @classmethod
    def validate_report_to(cls, v):
        """验证报告工具设置"""
        if v is None:
            return v

        # 如果是ReportTo枚举，直接返回其值
        if isinstance(v, ReportTo):
            return v.value

        # 如果是列表，验证每个元素
        if isinstance(v, list):
            valid_values = [item.value for item in ReportTo]
            for item in v:
                if isinstance(item, str) and item not in valid_values:
                    logger.warning(f"无效的报告工具: {item}, 支持的工具: {valid_values}")
            return v

        # 如果是字符串，验证是否在枚举中
        if isinstance(v, str):
            valid_values = [item.value for item in ReportTo]
            if v not in valid_values:
                logger.warning(f"无效的报告工具: {v}, 支持的工具: {valid_values}")

        return v

    def model_post_init(self, __context) -> None:
        """模型初始化后的验证"""
        # 验证bf16和fp16不能同时启用
        if self.bf16 and self.fp16:
            raise ValueError("bf16和fp16不能同时启用")

        # 验证steps策略下必须设置对应的步数
        if self.eval_strategy == EvalStrategy.STEPS and self.eval_steps is None:
            raise ValueError("eval_strategy为steps时必须设置eval_steps")
        if self.save_strategy == SaveStrategy.STEPS and self.save_steps is None:
            raise ValueError("save_strategy为steps时必须设置save_steps")

    # ============================================================================
    # 参数分离方法 - 明确区分官方训练参数和自定义参数
    # ============================================================================

    @classmethod
    def get_official_training_param_names(cls) -> set:
        """获取官方训练参数名称列表（会传给 *TrainingArguments）"""
        return {
            # 基础训练参数
            'output_dir', 'overwrite_output_dir', 'do_train', 'do_eval', 'do_predict',
            'num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size',

            # 学习率和优化器参数
            'learning_rate', 'warmup_ratio', 'warmup_steps', 'weight_decay',
            'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm',

            # 训练策略参数
            'gradient_accumulation_steps', 'eval_strategy', 'save_strategy',
            'logging_strategy', 'logging_steps', 'max_steps',

            # 步数参数
            'eval_steps', 'save_steps', 'save_total_limit',

            # 数据加载参数
            'dataloader_num_workers', 'dataloader_drop_last', 'eval_accumulation_steps',

            # 混合精度训练
            'bf16', 'fp16', 'gradient_checkpointing',

            # 模型评估和保存
            'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better',

            # 系统和调试参数
            'seed', 'disable_tqdm', 'run_name',
            # 注意：report_to 被排除，我们用它来控制环境变量而不是直接传递给TrainingArguments

            # 数据处理参数
            'remove_unused_columns', 'group_by_length', 'length_column_name', 'label_names',

            # 学习率调度器和优化器
            'lr_scheduler_type', 'optim',

            # Hub相关参数
            'push_to_hub', 'hub_model_id', 'hub_strategy', 'hub_token',

            # 日志和检查点
            'logging_dir', 'resume_from_checkpoint',

            # 分布式训练和DeepSpeed
            'ddp_backend', 'deepspeed',

            # 其他重要参数
            'prediction_loss_only', 'ignore_data_skip',

            # 报告工具（需要传递但进行特殊处理）
            'report_to'
        }

    @classmethod
    def get_custom_param_names(cls) -> set:
        """获取自定义参数名称列表（不会传给官方 TrainingArguments）"""
        return {
            # SwanLab配置
            'swanlab_api_key', 'swanlab_workspace', 'swanlab_project', 'swanlab_experiment', 'swanlab_mode',

            # 数据集采样参数
            'train_sample_size', 'eval_sample_size', 'test_sample_size',

            # 运行时参数
            'user_logging_dir', 'user_eval_dir',

            # 数据集和模型路径
            'dataset_name_or_path', 'model_name_or_path', 'train_type',
            'task_id', 'task_name', 'description', 'HF_subset', 'device',

            # 模型参数（不传递给训练器，由系统内部处理）
            'max_seq_length'
        }


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

            # 自动设置前端展示目录为固定格式
            if self._training_params.output_dir:
                output_dir = self._training_params.output_dir
                auto_user_logging_dir = f"{output_dir}/logs"
                auto_user_eval_dir = f"{output_dir}/eval"

                # 更新参数
                updated_config = merged_config.copy()
                updated_config['user_logging_dir'] = auto_user_logging_dir
                updated_config['user_eval_dir'] = auto_user_eval_dir
                self._training_params = TrainingParameters(**updated_config)

                logger.info(f"自动设置前端展示目录:")
                logger.info(f"  - 用户日志目录: {auto_user_logging_dir}")
                logger.info(f"  - 用户评估目录: {auto_user_eval_dir}")

            logger.info("训练参数加载成功")
            self._log_parameters()
        except Exception as e:
            logger.error(f"训练参数验证失败: {e}")
            raise ValueError(f"无效的训练参数配置: {e}")

        return self
    
    def get_training_args_dict(self) -> Dict[str, Any]:
        """
        获取官方训练参数字典，用于传递给 CrossEncoderTrainingArguments / SentenceTransformerTrainingArguments

        Returns:
            只包含官方训练参数的字典
        """
        if self._training_params is None:
            raise ValueError("训练参数未初始化，请先调用load_from_config")

        # 获取所有参数（不排除None值，因为save_total_limit=None是有意义的）
        all_params = self._training_params.model_dump(exclude_unset=True)

        # 只保留官方训练参数
        official_param_names = TrainingParameters.get_official_training_param_names()
        training_args_dict = {
            k: v for k, v in all_params.items()
            if k in official_param_names
        }

        # 特殊处理某些参数
        if 'report_to' in training_args_dict:
            report_to_value = training_args_dict['report_to']

            # 处理枚举类型
            if hasattr(report_to_value, 'value'):
                report_to_value = report_to_value.value

            # 处理字符串和列表
            if isinstance(report_to_value, str):
                if report_to_value in ['none', '']:
                    training_args_dict['report_to'] = []
                else:
                    training_args_dict['report_to'] = [report_to_value]
            elif isinstance(report_to_value, list):
                # 已经是列表，检查是否包含none
                filtered_list = [item for item in report_to_value if item not in ['none', '']]
                training_args_dict['report_to'] = filtered_list if filtered_list else []
        else:
            # 关键修复：如果没有设置report_to，显式设置为空列表，避免HuggingFace自动检测报告工具
            training_args_dict['report_to'] = []
            logger.info("未设置report_to参数，显式设置为空列表以禁用所有报告工具")

        logger.info(f"获取官方训练参数: {len(training_args_dict)} 个参数")
        return training_args_dict

    def get_custom_params_dict(self) -> Dict[str, Any]:
        """
        获取自定义参数字典，用于系统内部处理

        Returns:
            只包含自定义参数的字典
        """
        if self._training_params is None:
            raise ValueError("训练参数未初始化，请先调用load_from_config")

        # 获取所有参数（不排除None值）
        all_params = self._training_params.model_dump(exclude_unset=True)

        # 只保留自定义参数
        custom_param_names = TrainingParameters.get_custom_param_names()
        custom_params_dict = {
            k: v for k, v in all_params.items()
            if k in custom_param_names
        }

        logger.info(f"获取自定义参数: {len(custom_params_dict)} 个参数")
        return custom_params_dict
    
    def get_training_parameters(self) -> TrainingParameters:
        """
        获取训练参数对象
        
        Returns:
            训练参数对象
        """
        if self._training_params is None:
            raise ValueError("训练参数未初始化，请先调用load_from_config")
        
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
            raise ValueError("训练参数未初始化，请先调用load_from_config")
        
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