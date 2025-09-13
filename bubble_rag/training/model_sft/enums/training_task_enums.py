"""
训练相关的统一枚举定义
统一所有训练状态、类型等枚举，避免不一致问题
"""

from enum import Enum


class TrainingStatus(str, Enum):
    """训练状态枚举"""
    PENDING = "PENDING"          # 等待中
    RUNNING = "RUNNING"          # 运行中
    SUCCEEDED = "SUCCEEDED"      # 成功完成
    STOPPED = "STOPPED"          # 已停止（用户主动停止）
    FAILED = "FAILED"            # 失败


class TrainingType(str, Enum):
    """训练类型枚举"""
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class ProcessStatus(str, Enum):
    """进程状态枚举 - 简化版本"""
    RUNNING = "RUNNING"          # 进程正常运行中
    STOPPED = "STOPPED"          # 进程已正常停止  
    TERMINATED = "TERMINATED"    # 已被系统终止的进程
    UNKNOWN = "UNKNOWN"          # 进程状态未知
    
    @classmethod
    def get_manageable_statuses(cls):
        """获取可管理的进程状态（可以进行停止操作）"""
        return [cls.RUNNING]
    
    @classmethod
    def get_final_statuses(cls):
        """获取最终状态（不会再转换的状态）"""
        return [cls.STOPPED, cls.TERMINATED]
    
    @classmethod
    def get_active_statuses(cls):
        """获取活跃状态（进程可能还在运行）"""
        return [cls.RUNNING, cls.UNKNOWN]


class DatasetStatus(str, Enum):
    """数据集状态枚举"""
    PENDING = "pending"          # 待加载
    LOADED = "loaded"            # 加载成功
    FAILED = "failed"            # 加载失败


class DatasetType(str, Enum):
    """数据集类型枚举"""
    HUGGINGFACE = "huggingface"
    LOCAL_FILE = "local_file"
    LOCAL_FOLDER = "local_folder"
    FAILED = "failed"


class SplitType(str, Enum):
    """数据集分割类型枚举"""
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


class EvaluationStatus(str, Enum):
    """数据集评估执行状态枚举"""
    NOT_EVALUATED = "not_evaluated"        # 未执行评估/训练
    BASE_EVALUATED = "base_evaluated"      # 已基线评估（测试集初始测试，验证集初始验证）
    TRAINING_EVALUATED = "training_evaluated"  # 已训练评估（仅验证集的训练过程评估）
    FINAL_EVALUATED = "final_evaluated"    # 已最终评估（测试集最终测试，训练集已训练完成）
    IN_PROGRESS = "in_progress"            # 执行中
    FAILED = "failed"                      # 执行失败


