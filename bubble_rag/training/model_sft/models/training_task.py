"""
训练任务数据模型
用于存储和管理训练任务的完整信息
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from ..enums import TrainingStatus, TrainingType

class TrainingTask(BaseModel):
    """训练任务模型"""
    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True
    )
    
    
    # 基础信息
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务唯一ID")
    task_name: Optional[str] = Field(default=None, description="任务名称")
    description: Optional[str] = Field(default=None, description="任务描述")
    
    # 训练配置
    train_type: TrainingType = Field(description="训练类型")
    model_name_or_path: str = Field(description="模型名称或路径")
    dataset_name_or_path: str = Field(description="数据集名称或路径")
    HF_subset: Optional[str] = Field(default=None, description="HuggingFace数据集的子配置名称，如'pair-score'、'pair-class'等")
    output_dir: str = Field(description="输出目录")
    device: Optional[str] = Field(default="auto", description="训练设备")
    
    # 模型信息
    model_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="模型详细信息")
    embedding_dimension: Optional[int] = Field(default=None, description="Embedding模型的维度")
    model_size_mb: Optional[float] = Field(default=None, description="模型大小(MB)")
    
    # 训练参数
    training_params: Dict[str, Any] = Field(default_factory=dict, description="训练参数")
    
    # 数据集信息
    dataset_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="数据集信息")
    train_samples: Optional[int] = Field(default=None, description="训练样本数量")
    eval_samples: Optional[int] = Field(default=None, description="验证样本数量")
    test_samples: Optional[int] = Field(default=None, description="测试样本数量")
    
    # 状态信息
    status: TrainingStatus = Field(default=TrainingStatus.PENDING, description="训练状态")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="训练进度(0-100)")
    
    # 时间信息
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    duration_seconds: Optional[float] = Field(default=None, description="训练时长(秒)")
    
    # 结果信息
    final_model_path: Optional[str] = Field(default=None, description="最终模型保存路径")
    checkpoints: List[str] = Field(default_factory=list, description="检查点路径列表")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="训练指标")
    logs: List[str] = Field(default_factory=list, description="训练日志")
    
    # 错误信息
    error_message: Optional[str] = Field(default=None, description="错误信息")
    error_traceback: Optional[str] = Field(default=None, description="错误堆栈")
    
    # 配置快照
    config_snapshot: Dict[str, Any] = Field(default_factory=dict, description="完整配置快照")
    env_snapshot: Dict[str, Any] = Field(default_factory=dict, description="环境变量快照")
    
    # 服务实例信息
    service_instance_id: Optional[str] = Field(default=None, description="创建任务的服务实例ID")
    process_pid: Optional[int] = Field(default=None, description="训练进程PID")
    
    
    def start_training(self):
        """开始训练"""
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.now()
        self.progress = 0.0
    
    def complete_training(self, final_model_path: str, metrics: Dict[str, Any] = None):
        """完成训练"""
        self.status = TrainingStatus.SUCCEEDED
        self.completed_at = datetime.now()
        self.progress = 100.0
        self.final_model_path = final_model_path
        
        if metrics:
            self.metrics.update(metrics)
        
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def fail_training(self, error_message: str, error_traceback: str = None):
        """训练失败"""
        self.status = TrainingStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.error_traceback = error_traceback
        
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def cancel_training(self):
        """取消训练"""
        self.status = TrainingStatus.STOPPED
        self.completed_at = datetime.now()
        
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def update_progress(self, progress: float, log_message: str = None):
        """更新进度"""
        self.progress = max(0.0, min(100.0, progress))
        
        if log_message:
            self.add_log(log_message)
    
    def add_log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # 限制日志数量，避免占用过多内存
        if len(self.logs) > 1000:
            self.logs = self.logs[-800:]  # 保留最近800条
    
    def add_checkpoint(self, checkpoint_path: str):
        """添加检查点"""
        if checkpoint_path not in self.checkpoints:
            self.checkpoints.append(checkpoint_path)
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """估算剩余时间(秒)"""
        if not self.started_at or self.progress <= 0:
            return None
        
        elapsed = (datetime.now() - self.started_at).total_seconds()
        if self.progress >= 100:
            return 0
        
        # 基于当前进度估算剩余时间
        estimated_total = elapsed / (self.progress / 100)
        return estimated_total - elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """获取任务摘要信息"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "train_type": self.train_type,
            "model_name_or_path": self.model_name_or_path,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "embedding_dimension": self.embedding_dimension,
            "final_model_path": self.final_model_path,
            "error_message": self.error_message
        }

class TrainingTaskCreateRequest(BaseModel):
    """创建训练任务请求"""
    model_config = ConfigDict(protected_namespaces=())
    
    task_name: Optional[str] = Field(default=None, description="任务名称")
    description: Optional[str] = Field(default=None, description="任务描述")
    train_type: TrainingType = Field(description="训练类型")
    model_name_or_path: str = Field(description="模型名称或路径")
    dataset_name_or_path: str = Field(description="数据集名称或路径")
    HF_subset: Optional[str] = Field(default=None, description="HuggingFace数据集的子配置名称，如'pair-score'、'pair-class'等")
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    device: Optional[str] = Field(default="auto", description="训练设备，如 'cpu'、'cuda:0'、'cuda:1' 或 'auto' (自动检测)")
    training_params: Dict[str, Any] = Field(default_factory=dict, description="训练参数")

class TrainingTaskResponse(BaseModel):
    """训练任务响应"""
    success: bool = Field(description="是否成功")
    message: str = Field(description="响应消息")
    task: Optional[TrainingTask] = Field(default=None, description="训练任务")

class TrainingTaskListResponse(BaseModel):
    """训练任务列表响应"""
    success: bool = Field(description="是否成功")
    message: str = Field(description="响应消息")
    tasks: List[TrainingTask] = Field(default_factory=list, description="训练任务列表")
    total: int = Field(description="总数量")
