"""
训练参数相关的枚举定义
包含所有HuggingFace训练参数相关的枚举类型
"""

from enum import Enum


class ReportTo(str, Enum):
    """报告工具枚举"""
    NONE = "none"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    SWANLAB = "swanlab"


class OptimType(str, Enum):
    """优化器类型枚举"""
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    ADAFACTOR = "adafactor"


class LRSchedulerType(str, Enum):
    """学习率调度器类型枚举"""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class EvalStrategy(str, Enum):
    """评估策略枚举"""
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SaveStrategy(str, Enum):
    """保存策略枚举"""
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class LoggingStrategy(str, Enum):
    """日志策略枚举"""
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"