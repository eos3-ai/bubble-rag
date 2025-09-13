"""
训练模型定义模块
"""
from .training_task import TrainingTask, TrainingTaskCreateRequest, TrainingTaskResponse, TrainingTaskListResponse, TrainingStatus, TrainingType

__all__ = [
    "TrainingTask",
    "TrainingTaskCreateRequest", 
    "TrainingTaskResponse",
    "TrainingTaskListResponse",
    "TrainingStatus",
    "TrainingType"
]