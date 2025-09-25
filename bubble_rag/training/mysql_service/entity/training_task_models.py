"""训练任务数据库模型"""

import json
from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, DateTime, VARCHAR, TEXT, Integer, Column, Float, create_engine, Session, Index, CheckConstraint, Relationship, ForeignKey
from contextlib import contextmanager
import logging
from pydantic import ConfigDict

# 使用项目统一的数据库配置
from bubble_rag.server_config import MYSQL_URL
from bubble_rag.training.model_sft.enums import TrainingStatus, TrainingType, ProcessStatus

logger = logging.getLogger(__name__)


class TrainingTaskDB(SQLModel, table=True):
    """训练任务数据库模型"""
    __tablename__ = "training_tasks"
    __table_args__ = (
        Index('idx_service_instance_id', 'service_instance_id'),
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
        Index('idx_status_service', 'status', 'service_instance_id'),
        Index('idx_train_type', 'train_type'),
        Index('idx_username', 'username'),
        Index('idx_user_status', 'username', 'status'),
        Index('idx_base_task_id', 'base_task_id'),
        CheckConstraint('progress >= 0 AND progress <= 100', name='chk_progress_range'),
        CheckConstraint('started_at IS NULL OR started_at >= created_at', name='chk_start_after_create'),
        CheckConstraint('completed_at IS NULL OR completed_at >= started_at', name='chk_complete_after_start'),
        {'comment': '训练任务表V4-用户权限支持'}
    )
    model_config = ConfigDict(protected_namespaces=())

    # 基础信息
    task_id: str = Field(primary_key=True, max_length=64, nullable=False, description="任务唯一ID")
    task_name: Optional[str] = Field(sa_column=Column(VARCHAR(256), comment='任务名称'))
    description: Optional[str] = Field(sa_column=Column(TEXT, comment='任务描述'))
    
    # 训练配置 - 根据需求添加的核心字段
    train_type: str = Field(sa_column=Column(VARCHAR(32), comment='训练类型: embedding, reranker', nullable=False))
    dataset_name_or_path: str = Field(sa_column=Column(TEXT, comment='数据集名称或路径', nullable=False))
    HF_subset: Optional[str] = Field(sa_column=Column(VARCHAR(64), comment='HuggingFace数据集子配置名称', nullable=True))
    output_dir: str = Field(sa_column=Column(TEXT, comment='模型输出路径', nullable=False))
    model_name_or_path: Optional[str] = Field(sa_column=Column(TEXT, comment='模型名称或路径'))
    embedding_dim: Optional[int] = Field(sa_column=Column(Integer, comment='模型维度'))
    device: Optional[str] = Field(sa_column=Column(VARCHAR(64), comment='训练设备: cpu, cuda:0, cuda:1, auto等', nullable=True))
    
    # 状态信息 - 根据需求的状态定义
    status: TrainingStatus = Field(default=TrainingStatus.PENDING, sa_column=Column(VARCHAR(32), comment='训练状态: PENDING(等待中), RUNNING(运行中), SUCCEEDED(成功), STOPPED(已停止), FAILED(失败)', nullable=False))
    progress: float = Field(default=0.0, sa_column=Column(Float, comment='训练进度(0.0-100.0)', nullable=False))
    
    # 时间信息
    created_at: datetime = Field(default_factory=datetime.now, sa_column=Column(DateTime, comment='创建时间', nullable=False))
    updated_at: datetime = Field(default_factory=datetime.now, sa_column=Column(DateTime, comment='更新时间', nullable=False))
    started_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime, comment='开始时间'))
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime, comment='完成时间'))
    
    # 结果信息
    final_model_path: Optional[str] = Field(sa_column=Column(TEXT, comment='最终模型路径'))
    error_message: Optional[str] = Field(sa_column=Column(TEXT, comment='错误信息'))
    loss_data: Optional[str] = Field(sa_column=Column(TEXT, comment='训练loss汇总数据JSON'))
    
    # 额外配置信息
    training_params: Optional[str] = Field(sa_column=Column(TEXT, comment='训练参数JSON'))
    
    # 服务实例管理字段（技术层面）
    service_instance_id: Optional[str] = Field(
        sa_column=Column(VARCHAR(128), comment='启动该任务的服务实例ID', nullable=True)
    )
    service_startup_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, comment='服务实例启动时间，用于孤儿进程检测', nullable=True)
    )

    # 重启关系字段
    base_task_id: Optional[str] = Field(
        default=None,
        sa_column=Column(VARCHAR(64), comment='重启源任务ID', nullable=True)
    )
    restart_count: int = Field(
        default=0,
        sa_column=Column(Integer, comment='被重启次数', nullable=False)
    )

    # 用户权限管理字段（业务层面）
    username: str = Field(
        sa_column=Column(VARCHAR(64), ForeignKey("users.username"), comment='关联用户名（外键）', nullable=False)
    )

    # 进程管理字段
    process_pid: Optional[int] = Field(
        sa_column=Column(Integer, comment='训练进程PID', nullable=True)
    )
    process_status: ProcessStatus = Field(
        default=ProcessStatus.STOPPED,
        sa_column=Column(VARCHAR(32), comment='进程状态: RUNNING, STOPPED, TERMINATED, UNKNOWN', nullable=False)
    )

    @classmethod
    def from_training_task(cls, task, training_params: Dict[str, Any] = None, username: str = 'admin', service_instance_id: str = None):
        """从TrainingTask对象创建数据库记录"""
        return cls(
            task_id=task.task_id,
            task_name=task.task_name,
            description=task.description,
            train_type=task.train_type,
            dataset_name_or_path=task.dataset_name_or_path,
            HF_subset=getattr(task, 'HF_subset', None),  # 添加HF_subset字段映射
            output_dir=task.output_dir,
            model_name_or_path=task.model_name_or_path,
            device=task.device,  # 添加device字段映射
            embedding_dim=getattr(task, 'embedding_dimension', None),
            status=cls._map_status(task.status),
            progress=task.progress,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            final_model_path=task.final_model_path,
            error_message=task.error_message,
            training_params=json.dumps(training_params, ensure_ascii=False) if training_params else None,
            # 重启关系字段映射
            base_task_id=getattr(task, 'base_task_id', None),
            restart_count=getattr(task, 'restart_count', 0),
            # 服务实例管理字段映射（技术层面）
            service_instance_id=service_instance_id or getattr(task, 'service_instance_id', None),
            process_pid=getattr(task, 'process_pid', None),
            process_status=ProcessStatus.STOPPED,  # 默认状态，会在进程启动时更新
            # 用户权限字段映射（业务层面）- 修复字段名称
            username=username or getattr(task, 'username', 'admin')
        )
    
    @staticmethod
    def _map_status(training_task_status: str) -> TrainingStatus:
        """映射TrainingTask状态到数据库状态"""
        # 如果已经是TrainingStatus枚举值，直接返回
        if isinstance(training_task_status, TrainingStatus):
            return training_task_status
        
        # 尝试直接匹配枚举值
        try:
            return TrainingStatus(training_task_status)
        except ValueError:
            # 处理一些旧格式的映射
            mapping = {
                "completed": TrainingStatus.SUCCEEDED,
                "cancelled": TrainingStatus.STOPPED,
                "finished": TrainingStatus.SUCCEEDED
            }
            return mapping.get(training_task_status.lower(), TrainingStatus.PENDING)
    
    def update_from_training_task(self, task, username: str = None):
        """从TrainingTask对象更新数据库记录"""
        self.status = self._map_status(task.status)
        self.progress = task.progress
        self.started_at = task.started_at
        self.completed_at = task.completed_at
        self.final_model_path = task.final_model_path
        self.error_message = task.error_message
        self.device = task.device  # 添加device字段更新
        self.HF_subset = getattr(task, 'HF_subset', None)  # 添加HF_subset字段更新
        self.embedding_dim = getattr(task, 'embedding_dimension', None)
        # 重启关系字段更新
        self.base_task_id = getattr(task, 'base_task_id', None)
        self.restart_count = getattr(task, 'restart_count', 0)
        # 服务实例管理字段更新（技术层面）
        self.service_instance_id = getattr(task, 'service_instance_id', None)
        self.process_pid = getattr(task, 'process_pid', None)
        # 用户权限字段更新（业务层面）- 修复字段名称
        if username is not None:
            self.username = username
        # process_status 由专门的 update_process_info 方法管理


# 数据库引擎和会话管理
engine = None

def get_engine():
    """获取数据库引擎"""
    global engine
    if engine is None:
        engine = create_engine(MYSQL_URL, echo=False)
    return engine

def create_tables():
    """创建数据库表"""
    try:
        # 确保导入所有模型，这样SQLModel.metadata才能包含所有表定义
        from bubble_rag.training.mysql_service.entity.user_models import UserDB
        import hashlib

        engine = get_engine()
        SQLModel.metadata.create_all(engine)

        # 确保默认admin用户存在
        with Session(engine) as session:
            try:
                # 检查admin用户是否已存在
                from sqlmodel import select
                admin_user = session.exec(select(UserDB).where(UserDB.username == 'admin')).first()

                if not admin_user:
                    # 创建默认admin用户
                    default_admin = UserDB(
                        username='admin',
                        user_password=hashlib.sha256('admin'.encode()).hexdigest(),
                        user_role='admin',
                        display_name='系统管理员'
                    )
                    session.add(default_admin)
                    session.commit()
                    print("默认admin用户创建成功 (用户名: admin, 密码: admin)")
                else:
                    print("默认admin用户已存在")
            except Exception as user_error:
                print(f"默认用户创建失败: {user_error}")
                # 不影响整体表创建流程

        return True
    except Exception as e:
        print(f"创建数据库表失败: {e}")
        return False

def get_session():
    """获取数据库会话"""
    engine = get_engine()
    return Session(engine)

@contextmanager
def safe_get_session():
    """
    安全的数据库会话上下文管理器
    简化版本，确保会话正确关闭
    """
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()