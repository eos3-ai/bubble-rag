"""训练数据集元信息数据库模型"""

from bubble_rag.training.mysql_service import get_session
from pydantic import ConfigDict
from sqlmodel import SQLModel, Field, DateTime, VARCHAR, TEXT, Integer, Column, Relationship, JSON, Index, CheckConstraint
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from bubble_rag.utils.snowflake_utils import gen_id
from bubble_rag.training.model_sft.enums import TrainingType, DatasetStatus, DatasetType, SplitType, EvaluationStatus


# TrainingJob类已移除，统一使用training_tasks表

class DatasetInfo(SQLModel, table=True):
    """数据集信息表"""
    __tablename__ = "dataset_info"
    __table_args__ = (
        Index('idx_task_id', 'task_id'),
        Index('idx_dataset_status', 'dataset_status'),
        Index('idx_evaluation_status', 'evaluation_status'),
        Index('idx_split_type', 'split_type'),
        Index('idx_dataset_type', 'dataset_type'),
        Index('idx_task_source', 'task_id', 'data_source_id'),  # 新增：数据源分组索引
        Index('idx_task_source_split', 'task_id', 'data_source_id', 'split_type'),  # 新增：精确查询索引
        CheckConstraint('total_samples >= 0', name='chk_positive_samples_dataset'),
        CheckConstraint('configured_sample_size >= 0', name='chk_positive_configured_sample_size'),
        {'comment': '数据集元信息'}
    )
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(primary_key=True, max_length=32, nullable=False, default_factory=gen_id)
    task_id: str = Field(sa_column=Column(VARCHAR(64), comment='关联的训练任务ID', nullable=False))
    data_source_id: str = Field(sa_column=Column(VARCHAR(64), comment='数据源标识ID，用于分组同源的train/eval/test', nullable=False))  # 新增
    dataset_name: str = Field(sa_column=Column(VARCHAR(256), comment='数据集基础名称（不含分割后缀，如squad, nli）', nullable=False))
    dataset_base_name: str = Field(sa_column=Column(VARCHAR(256), comment='与dataset_name保持一致（向后兼容）', nullable=False))
    HF_subset: Optional[str] = Field(sa_column=Column(VARCHAR(64), comment='实际使用的HuggingFace子配置名称', nullable=True))  # 新增
    dataset_path: str = Field(sa_column=Column(TEXT, comment='数据集路径或名称', nullable=False))
    dataset_type: DatasetType = Field(sa_column=Column(VARCHAR(32), comment='数据集类型: huggingface, local_file, local_folder, failed', nullable=False))
    split_type: SplitType = Field(sa_column=Column(VARCHAR(32), comment='数据集分割类型: train, eval, test', nullable=False))
    
    # 数据集状态信息
    dataset_status: DatasetStatus = Field(default=DatasetStatus.PENDING, sa_column=Column(VARCHAR(32), comment='数据集状态: pending, loaded, failed', nullable=False))
    evaluation_status: EvaluationStatus = Field(default=EvaluationStatus.NOT_EVALUATED, sa_column=Column(VARCHAR(32), comment='评估执行状态: not_evaluated, base_evaluated, training_evaluated, final_evaluated, in_progress, failed', nullable=False))
    error_message: Optional[str] = Field(default=None, sa_column=Column(TEXT, comment='错误信息（加载失败时）'))
    
    # 数据集内容信息
    total_samples: int = Field(default=0, sa_column=Column(Integer, comment='样本总数', nullable=False))
    configured_sample_size: int = Field(default=0, sa_column=Column(Integer, comment='实际使用的样本数量（受配置限制影响）', nullable=False))
    target_column: str = Field(sa_column=Column(VARCHAR(64), comment='目标列名: score, label', nullable=False))
    label_type: str = Field(sa_column=Column(VARCHAR(32), comment='标签数据类型: int, float', nullable=False))
    column_names: Optional[Dict[str, Any]] = Field(sa_column=Column(JSON, comment='数据集列名信息'))
    
    # 训练配置信息
    loss_function: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(128), comment='使用的损失函数', nullable=True))
    evaluator: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(128), comment='使用的评估器', nullable=True))
    
    # 评估结果（可选）
    base_eval_results: Optional[Dict[str, Any]] = Field(sa_column=Column(JSON, comment='基线模型在该数据集上的评估结果'))
    final_eval_results: Optional[Dict[str, Any]] = Field(sa_column=Column(JSON, comment='训练后模型在该数据集上的最终评估结果'))
    loss: Optional[List[Dict[str, Any]]] = Field(sa_column=Column(JSON, comment='训练过程中的损失历史（训练集记录train_loss，验证集记录eval_loss）'))
    training_evaluator_evaluation: Optional[List[Dict[str, Any]]] = Field(sa_column=Column(JSON, comment='训练过程中评估器的评估结果历史（仅验证集有此数据，测试集为null）'))
    
    create_time: datetime = Field(default_factory=datetime.now, nullable=False)
    update_time: datetime = Field(default_factory=datetime.now, nullable=False)
    


# 关系定义已移除，将使用training_tasks表进行关联