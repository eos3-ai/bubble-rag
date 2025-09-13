"""
训练任务管理服务
负责训练任务的创建、管理和存储
"""
import os
import logging
import json
import threading
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..models.training_task import TrainingTask, TrainingTaskCreateRequest, TrainingStatus, TrainingType
from .config_service import config_service

logger = logging.getLogger(__name__)

class TaskManager:
    """训练任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, TrainingTask] = {}
        self.tasks_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tasks_storage.json"
        )
        # 线程锁保护共享资源
        self._lock = threading.RLock()
        self._load_tasks()
    
    def create_task(self, request: TrainingTaskCreateRequest, task_id: str = None, service_instance_id: str = None) -> TrainingTask:
        """
        创建训练任务
        
        Args:
            request: 创建请求
            task_id: 预分配的任务ID，如果为None则自动生成
            service_instance_id: 创建任务的服务实例ID
            
        Returns:
            训练任务
        """
        with self._lock:
            try:
                # 生成输出目录
                if not request.output_dir:
                    output_dir = config_service.generate_output_path(
                        request.train_type, 
                        request.model_name_or_path,
                        request.task_name or ""
                    )
                else:
                    output_dir = request.output_dir
            
                # 创建任务
                task_kwargs = {
                    "task_name": request.task_name or f"{request.train_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "description": request.description,
                    "train_type": request.train_type,
                    "model_name_or_path": request.model_name_or_path,
                    "dataset_name_or_path": request.dataset_name_or_path,
                    "output_dir": output_dir,
                    "device": request.device or "auto",
                    "training_params": request.training_params,
                    "service_instance_id": service_instance_id
                }
                
                # 如果提供了预分配的task_id，则使用它
                if task_id:
                    task_kwargs["task_id"] = task_id
                
                task = TrainingTask(**task_kwargs)
                
                # 保存配置快照
                task.config_snapshot = {
                    "train_type": task.train_type,
                    "model_name_or_path": task.model_name_or_path,
                    "dataset_name_or_path": task.dataset_name_or_path,
                    "output_dir": task.output_dir,
                    "device": task.device,
                    "training_params": task.training_params
                }
                
                # 环境变量快照
                task.env_snapshot = dict(os.environ)
                
                # 存储任务
                self.tasks[task.task_id] = task
                self._save_tasks()
                
                task.add_log(f"训练任务创建成功: {task.task_name}")
                
                return task
                
            except Exception as e:
                logger.error(f"创建训练任务失败: {str(e)}", exc_info=True)
                raise
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取训练任务"""
        with self._lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TrainingTask]:
        """获取所有训练任务"""
        with self._lock:
            return list(self.tasks.values())
    
    def get_tasks_by_status(self, status: TrainingStatus) -> List[TrainingTask]:
        """按状态获取训练任务"""
        with self._lock:
            return [task for task in self.tasks.values() if task.status == status]
    
    def update_task(self, task_id: str, updates: Dict) -> Optional[TrainingTask]:
        """
        更新训练任务
        
        Args:
            task_id: 任务ID
            updates: 更新内容
            
        Returns:
            更新后的任务
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            self._save_tasks()
            return task
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否删除成功
        """
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self._save_tasks()
                return True
            return False
    
    def start_task(self, task_id: str) -> Optional[TrainingTask]:
        """开始训练任务"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            task.start_training()
            self._save_tasks()
            return task
    
    def complete_task(self, task_id: str, final_model_path: str, metrics: Dict = None) -> Optional[TrainingTask]:
        """完成训练任务"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            task.complete_training(final_model_path, metrics)
            self._save_tasks()
            
            # 🔧 同步更新数据库中的进程状态为STOPPED
            try:
                from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                from bubble_rag.training.model_sft.enums.training_task_enums import ProcessStatus
                
                training_task_service.update_process_info(
                    task_id=task_id,
                    process_pid=task.process_pid,  # 🔧 保留PID用于审计追踪
                    process_status=ProcessStatus.STOPPED.value
                )
                logger.info(f"✅ 训练完成，进程状态已更新为STOPPED: {task_id}")
            except Exception as e:
                logger.warning(f"更新完成任务的进程状态失败: {str(e)}")
            
            return task
    
    def fail_task(self, task_id: str, error_message: str, error_traceback: str = None) -> Optional[TrainingTask]:
        """训练任务失败"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            task.fail_training(error_message, error_traceback)
            self._save_tasks()
            return task
    
    def cancel_task(self, task_id: str) -> Optional[TrainingTask]:
        """取消训练任务"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            task.cancel_training()
            self._save_tasks()
            return task
    
    def update_task_progress(self, task_id: str, progress: float, log_message: str = None) -> Optional[TrainingTask]:
        """更新任务进度（内存实时更新 + 数据库1%节流更新）"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            # 获取上次数据库更新的进度
            last_db_progress = getattr(task, '_last_db_progress', -1)
            
            # 总是更新内存中的进度（实时）
            task.update_progress(progress, log_message)
            
            # 数据库更新策略：进度变化超过1%才更新（提高同步频率）
            progress_change = abs(progress - last_db_progress)
            should_update_db = (
                progress_change >= 1 or  # 进度变化1%以上
                (progress >= 100 and task.status == TrainingStatus.SUCCEEDED) or  # 🔧 只有状态为SUCCEEDED时才允许100%进度写入数据库
                last_db_progress == -1  # 首次更新
            )
            
            if should_update_db:
                # 保存到本地文件
                self._save_tasks()
                
                # 同步更新数据库进度
                try:
                    from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                    training_task_service.update_task_status(task_id, task.status, progress)
                    
                    # 记录数据库更新的进度
                    task._last_db_progress = progress
                    
                    logger.info(f"进度已同步到数据库: {progress:.1f}% (变化: {progress_change:.1f}%)")
                except Exception as e:
                    logger.warning(f"同步进度到数据库失败: {e}")
            else:
                logger.debug(f"内存进度更新: {progress:.1f}% (数据库暂不更新，变化: {progress_change:.1f}%)")
            
            return task
    
    def add_task_log(self, task_id: str, message: str) -> Optional[TrainingTask]:
        """添加任务日志"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            task.add_log(message)
            self._save_tasks()
            return task
    
    def update_model_info_after_loading(self, task_id: str, model_info: dict):
        """训练时更新模型信息到数据库"""
        try:
            task = self.get_task(task_id)
            if task:
                task.model_info = model_info
                task.add_log("✅ 模型信息已更新")
                
                # 更新embedding维度（支持embedding和reranker模型）
                embedding_dim = None
                if "embedding_dimension" in model_info:
                    task.embedding_dimension = model_info["embedding_dimension"]
                    embedding_dim = task.embedding_dimension
                    model_type_name = "embedding" if task.train_type == TrainingType.EMBEDDING else "reranker"
                    task.add_log(f"✅ {model_type_name}模型维度已更新: {task.embedding_dimension}")
                
                # 保存到本地文件
                self._save_tasks()
                
                # 同步到数据库
                try:
                    from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                    if embedding_dim:
                        training_task_service.update_task_result(task_id, embedding_dim=embedding_dim)
                        logger.info(f"✅ embedding维度已同步到数据库: {embedding_dim}")
                    else:
                        # 对于非embedding任务，也要确保数据库和内存数据同步
                        training_task_service.save_training_task(task)
                        logger.info(f"✅ 任务信息已同步到数据库")
                except Exception as db_e:
                    logger.warning(f"同步模型信息到数据库失败: {db_e}")
                
                logger.info(f"任务 {task_id} 的模型信息已更新")
        except Exception as e:
            logger.error(f"更新模型信息失败: {str(e)}")
            # 不抛出异常，不影响训练继续
    
    def _save_tasks(self):
        """保存任务到文件"""
        try:
            tasks_data = {}
            for task_id, task in self.tasks.items():
                tasks_data[task_id] = task.dict()
            
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"保存训练任务失败: {str(e)}", exc_info=True)
    
    def _load_tasks(self):
        """从文件加载任务"""
        # 注意：初始化时不需要锁，因为只有一个线程
        try:
            if os.path.exists(self.tasks_file):
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                
                for task_id, task_data in tasks_data.items():
                    try:
                        task = TrainingTask(**task_data)
                        self.tasks[task_id] = task
                    except Exception as e:
                        logger.warning(f"加载训练任务 {task_id} 失败: {str(e)}")
                        
                logger.info(f"加载了 {len(self.tasks)} 个训练任务")
            else:
                logger.info("未找到训练任务存储文件，将创建新的")
                
        except Exception as e:
            logger.error(f"加载训练任务失败: {str(e)}", exc_info=True)
    
    def cleanup_old_tasks(self, days: int = 30):
        """清理旧任务"""
        with self._lock:
            try:
                cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
                tasks_to_remove = []
                
                for task_id, task in self.tasks.items():
                    if task.created_at.timestamp() < cutoff_date and task.status in [
                        TrainingStatus.SUCCEEDED, TrainingStatus.FAILED, TrainingStatus.STOPPED
                    ]:
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                
                if tasks_to_remove:
                    self._save_tasks()
                    logger.info(f"清理了 {len(tasks_to_remove)} 个旧任务")
                    
            except Exception as e:
                logger.error(f"清理旧任务失败: {str(e)}", exc_info=True)

# 全局任务管理器实例
task_manager = TaskManager()