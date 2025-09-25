"""
统一的训练配置模型
供单进程和多进程训练接口复用
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class UnifiedTrainingConfig(BaseModel):
    """统一的训练配置 - 单进程和多进程训练共用"""
    model_config = ConfigDict(protected_namespaces=())
    
    # === 基础配置 ===
    train_type: str = Field(description="训练类型: embedding 或 reranker")
    model_name_or_path: str = Field(description="模型名称或路径")
    dataset_name_or_path: str = Field(description="数据集名称或路径")
    HF_subset: Optional[str] = Field(default=None, description="HuggingFace数据集的子配置名称，如'pair-score'、'pair-class'等")
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    
    # === 任务标识 ===
    task_name: Optional[str] = Field(default=None, description="任务名称（用户自定义，如未设置则自动生成）")
    description: Optional[str] = Field(default=None, description="任务描述")
    
    # === 设备配置 ===
    device: Optional[str] = Field(
        default="auto", 
        description="训练设备: cpu, auto, cuda:0, cuda:1, cuda:0,cuda:1, auto:2 等"
    )
    
    # === 基础训练参数 ===
    num_train_epochs: Optional[int] = Field(default=3, description="训练轮数")
    per_device_train_batch_size: Optional[int] = Field(default=8, description="每个设备的训练批次大小")
    per_device_eval_batch_size: Optional[int] = Field(default=8, description="每个设备的评估批次大小")
    gradient_accumulation_steps: Optional[int] = Field(default=1, description="梯度累积步数")
    learning_rate: Optional[float] = Field(default=5e-5, description="学习率")
    weight_decay: Optional[float] = Field(default=0.01, description="权重衰减")
    lr_scheduler_type: Optional[str] = Field(default="linear", description="学习率调度器类型")
    warmup_steps: Optional[int] = Field(default=0, description="预热步数")
    warmup_ratio: Optional[float] = Field(default=0.0, description="预热比例")
    
    # === 模型参数 ===
    max_seq_length: Optional[int] = Field(default=512, description="最大序列长度")
    embedding_dim: Optional[int] = Field(default=None, description="嵌入维度（仅embedding类型需要）")
    
    # === 训练控制 ===
    save_strategy: Optional[str] = Field(default="epoch", description="保存策略: steps, epoch, no")
    save_steps: Optional[int] = Field(default=500, description="保存步数间隔")
    save_total_limit: Optional[int] = Field(default=3, description="最大保存checkpoint数量")
    eval_strategy: Optional[str] = Field(default="epoch", description="评估策略: steps, epoch, no")
    eval_steps: Optional[int] = Field(default=500, description="评估步数间隔")
    logging_steps: Optional[int] = Field(default=500, description="日志记录步数间隔")
    
    # === 硬件优化 ===
    fp16: Optional[bool] = Field(default=False, description="是否使用混合精度训练")
    bf16: Optional[bool] = Field(default=False, description="是否使用BF16精度")
    gradient_checkpointing: Optional[bool] = Field(default=False, description="是否使用梯度检查点")
    dataloader_num_workers: Optional[int] = Field(default=4, description="数据加载器工作进程数")
    dataloader_drop_last: Optional[bool] = Field(default=False, description="丢弃最后一个不完整的batch")
    eval_accumulation_steps: Optional[int] = Field(default=None, ge=1, description="评估累积步数")
    
    # === 高级配置 ===
    seed: Optional[int] = Field(default=42, description="随机种子")
    resume_from_checkpoint: Optional[str] = Field(default=None, description="从checkpoint恢复训练的路径")
    report_to: Optional[List[str]] = Field(default=None, description="上报训练指标的工具列表")
    
    # === 数据集采样 ===
    train_sample_size: Optional[int] = Field(default=-1, ge=-1, le=10000000, description="训练数据集样本数量限制，-1表示不限制，0表示不使用该数据集")
    eval_sample_size: Optional[int] = Field(default=-1, ge=-1, le=10000000, description="验证数据集样本数量限制，-1表示不限制，0表示不使用该数据集")
    test_sample_size: Optional[int] = Field(default=-1, ge=-1, le=10000000, description="测试数据集样本数量限制，-1表示不限制，0表示不使用该数据集")

    # === 前端展示目录参数 ===
    user_logging_dir: Optional[str] = Field(default=None, description="前端展示用的日志目录，默认为{output_dir}/logs")
    user_eval_dir: Optional[str] = Field(default=None, description="前端展示用的评估结果目录，默认为{output_dir}/eval")

    # === 扩展参数 ===
    training_params: Optional[Dict[str, Any]] = Field(default=None, description="额外的训练参数")

    def to_training_task_request(self):
        """转换为 TrainingTaskCreateRequest"""
        from bubble_rag.training.model_sft.models.training_task import TrainingTaskCreateRequest
        
        return TrainingTaskCreateRequest(
            task_name=self.task_name,
            description=self.description, 
            train_type=self.train_type,
            model_name_or_path=self.model_name_or_path,
            dataset_name_or_path=self.dataset_name_or_path,
            output_dir=self.output_dir,
            device=self.device,
            training_params=self.model_dump()  # 使用 model_dump() 替代 dict()
        )


class TrainingActionRequest(BaseModel):
    """训练操作请求"""
    action: str = Field(description="操作类型: start, stop, pause, resume")
    task_id: Optional[str] = Field(default=None, description="任务ID")
    reason: Optional[str] = Field(default=None, description="操作原因")


class BatchTrainingRequest(BaseModel):
    """批量训练请求"""
    configs: List[UnifiedTrainingConfig] = Field(description="训练配置列表")
    sequential: Optional[bool] = Field(default=False, description="是否顺序执行（否则并行）")
    max_concurrent: Optional[int] = Field(default=None, description="最大并发数量")


class TrainingStatusQuery(BaseModel):
    """训练状态查询"""
    task_ids: Optional[List[str]] = Field(default=None, description="指定任务ID列表，为空则查询所有")
    status_filter: Optional[List[str]] = Field(default=None, description="状态过滤器")
    time_range: Optional[Dict[str, str]] = Field(default=None, description="时间范围过滤")
    include_details: Optional[bool] = Field(default=False, description="是否包含详细信息")


class DatasetInfoResponse(BaseModel):
    """数据集信息响应"""
    id: str
    dataset_name: str
    dataset_path: str
    dataset_type: str
    split_type: str
    dataset_status: str
    evaluation_status: str
    error_message: Optional[str] = None
    total_samples: int
    target_column: str
    data_type: str
    loss_function: str
    evaluator_type: str
    base_eval_results: Optional[dict] = None
    final_eval_results: Optional[dict] = None
    create_time: str
    update_time: str


class ApiResponse(BaseModel):
    """统一API响应格式"""
    code: int = 200
    msg: str = "success"
    data: Optional[dict] = None


