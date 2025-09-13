"""训练任务服务
管理训练任务的数据库操作
"""
import traceback
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlmodel import select
import json

from bubble_rag.training.mysql_service.entity.training_task_models import TrainingTaskDB, get_session, safe_get_session, create_tables
from bubble_rag.training.model_sft.models.training_task import TrainingTask
from bubble_rag.training.model_sft.enums import TrainingStatus
from bubble_rag.training.model_sft.utils.error_handler import handle_database_error, with_error_handling


class TrainingTaskService:
    """训练任务数据库服务"""
    
    def __init__(self):
        """初始化训练任务服务"""
        self.ensure_tables_created()
    
    @staticmethod
    def normalize_status(status: str) -> str:
        """统一状态格式转换为标准格式"""
        # 直接返回TrainingStatus枚举值，如果无效则返回PENDING
        try:
            if hasattr(TrainingStatus, status.upper()):
                return getattr(TrainingStatus, status.upper()).value
            # 处理一些常见的映射
            mapping = {
                'completed': TrainingStatus.SUCCEEDED.value,
                'cancelled': TrainingStatus.STOPPED.value,
                'finished': TrainingStatus.SUCCEEDED.value
            }
            return mapping.get(status.lower(), TrainingStatus.PENDING.value)
        except:
            return TrainingStatus.PENDING.value
    
    @staticmethod
    def _task_db_to_dict(task_db: TrainingTaskDB) -> Dict[str, Any]:
        """将TrainingTaskDB对象转换为字典，避免会话分离问题"""
        return {
            "task_id": task_db.task_id,
            "task_name": task_db.task_name,
            "description": task_db.description,
            "train_type": task_db.train_type,
            "model_name_or_path": task_db.model_name_or_path,
            "dataset_name_or_path": task_db.dataset_name_or_path,
            "output_dir": task_db.output_dir,
            "status": task_db.status,
            "progress": task_db.progress,
            "created_at": task_db.created_at,
            "updated_at": task_db.updated_at,
            "started_at": task_db.started_at,
            "completed_at": task_db.completed_at,
            "error_message": task_db.error_message,
            "final_model_path": task_db.final_model_path,
            "training_params": task_db.training_params,
            "embedding_dim": task_db.embedding_dim,
            "service_instance_id": task_db.service_instance_id,
            "process_pid": task_db.process_pid,
            "process_status": task_db.process_status
        }
    
    def ensure_tables_created(self) -> bool:
        """确保数据库表已创建"""
        try:
            return create_tables()
        except Exception as e:
            print(f"数据库表创建失败: {e}")
            return False
    
    @with_error_handling(context="database", default_return=False)
    def save_training_task(self, task: TrainingTask, training_params: Dict[str, Any] = None, service_instance_id: str = None) -> bool:
        """保存训练任务到数据库"""
        with safe_get_session() as session:
            # 检查是否已存在
            existing = session.get(TrainingTaskDB, task.task_id)
            if existing:
                # 更新现有记录
                existing.update_from_training_task(task)
                if training_params:
                    existing.training_params = json.dumps(training_params, ensure_ascii=False)
                if service_instance_id:
                    existing.service_instance_id = service_instance_id
            else:
                # 创建新记录
                db_task = TrainingTaskDB.from_training_task(task, training_params)
                # 🔧 service_instance_id 应该在进程实际启动时设置，而不是任务创建时
                # 这样可以确保只有真正运行过的任务才有服务实例归属
                if service_instance_id:
                    db_task.service_instance_id = service_instance_id
                session.add(db_task)
            
            session.commit()
            return True
    
    @with_error_handling(context="database", default_return=None)
    def get_training_task(self, task_id: str) -> Optional[TrainingTaskDB]:
        """根据任务ID获取训练任务"""
        try:
            with safe_get_session() as session:
                result = session.get(TrainingTaskDB, task_id)
                if result:
                    session.expunge(result)
                return result
        except Exception as e:
            print(f"根据任务ID获取训练任务失败: {e}")
            return None
    
    def get_all_training_tasks(self, limit: int = 100, offset: int = 0) -> List[TrainingTaskDB]:
        """获取所有训练任务"""
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).order_by(TrainingTaskDB.created_at.desc()).limit(limit).offset(offset)
                results = session.exec(statement).all()
                # 分离所有对象
                for result in results:
                    session.expunge(result)
                return list(results)
        except Exception as e:
            print(f"获取所有训练任务失败: {e}")
            return []
    
    def get_latest_training_task(self) -> Optional[TrainingTaskDB]:
        """获取最新的训练任务"""
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).order_by(TrainingTaskDB.created_at.desc()).limit(1)
                result = session.exec(statement).first()
                
                if result:
                    # 分离对象以避免会话关闭后的访问问题
                    session.expunge(result)
                return result
        except Exception as e:
            print(f"获取最新训练任务失败: {e}")
            return None
    
    def get_training_tasks_by_status(self, status: str) -> List[TrainingTaskDB]:
        """根据状态获取训练任务"""
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).where(TrainingTaskDB.status == status).order_by(TrainingTaskDB.created_at.desc())
                results = session.exec(statement).all()
                
                # 分离所有对象
                for result in results:
                    session.expunge(result)
                return list(results)
        except Exception as e:
            print(f"根据状态获取训练任务失败: {e}")
            return []
    
    def get_training_tasks_by_type(self, train_type: str) -> List[TrainingTaskDB]:
        """根据训练类型获取训练任务"""
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).where(TrainingTaskDB.train_type == train_type).order_by(TrainingTaskDB.created_at.desc())
                results = session.exec(statement).all()
                # 分离所有对象
                for result in results:
                    session.expunge(result)
                return list(results)
        except Exception as e:
            print(f"根据训练类型获取训练任务失败: {e}")
            return []
    
    def update_task_status(self, task_id: str, status: str, progress: float = None) -> bool:
        """更新训练任务状态"""
        try:
            with safe_get_session() as session:
                task = session.get(TrainingTaskDB, task_id)
                if not task:
                    return False
                
                # 使用统一的状态转换
                task.status = self.normalize_status(status)
                if progress is not None:
                    task.progress = progress
                
                # 根据状态更新时间
                normalized_status = task.status  # 已经转换过的状态
                if normalized_status == "RUNNING" and not task.started_at:
                    task.started_at = datetime.now()
                elif normalized_status in ["SUCCEEDED", "STOPPED", "FAILED"]:
                    if not task.completed_at:
                        task.completed_at = datetime.now()
                
                session.commit()
                return True
        except Exception as e:
            print(f"更新训练任务状态失败: {e}")
            return False
    
    def update_task_result(self, task_id: str, final_model_path: str = None, error_message: str = None, embedding_dim: int = None, loss_data: str = None) -> bool:
        """更新训练任务结果"""
        try:
            with safe_get_session() as session:
                task = session.get(TrainingTaskDB, task_id)
                if not task:
                    return False
                
                if final_model_path:
                    task.final_model_path = final_model_path
                if error_message:
                    task.error_message = error_message
                if embedding_dim:
                    task.embedding_dim = embedding_dim
                if loss_data:
                    task.loss_data = loss_data
                
                session.commit()
                return True
        except Exception as e:
            print(f"更新训练任务结果失败: {e}")
            return False
    
    def delete_training_task(self, task_id: str) -> bool:
        """删除训练任务"""
        try:
            with safe_get_session() as session:
                task = session.get(TrainingTaskDB, task_id)
                if not task:
                    return False
                
                session.delete(task)
                session.commit()
                return True
        except Exception as e:
            print(f"删除训练任务失败: {e}")
            return False
    
    def get_task_count(self) -> int:
        """获取任务总数"""
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB)
                results = session.exec(statement).all()
                return len(results)
        except Exception as e:
            print(f"获取任务总数失败: {e}")
            return 0
    
    def get_task_stats(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        try:
            with safe_get_session() as session:
                # 统计各状态的任务数量（兼容大小写）
                pending_count = len(session.exec(select(TrainingTaskDB).where(TrainingTaskDB.status.in_(["PENDING", "pending"]))).all())
                running_count = len(session.exec(select(TrainingTaskDB).where(TrainingTaskDB.status.in_(["RUNNING", "running", "training"]))).all())
                succeeded_count = len(session.exec(select(TrainingTaskDB).where(TrainingTaskDB.status.in_(["SUCCEEDED", "succeeded", "finished"]))).all())
                stopped_count = len(session.exec(select(TrainingTaskDB).where(TrainingTaskDB.status.in_(["STOPPED", "stopped"]))).all())
                failed_count = len(session.exec(select(TrainingTaskDB).where(TrainingTaskDB.status.in_(["FAILED", "failed"]))).all())
                
                # 统计各类型的任务数量
                embedding_count = len(session.exec(select(TrainingTaskDB).where(TrainingTaskDB.train_type == "embedding")).all())
                reranker_count = len(session.exec(select(TrainingTaskDB).where(TrainingTaskDB.train_type == "reranker")).all())
                
                return {
                    "total": pending_count + running_count + succeeded_count + stopped_count + failed_count,
                    "by_status": {
                        "PENDING": pending_count,
                        "RUNNING": running_count,
                        "SUCCEEDED": succeeded_count,
                        "STOPPED": stopped_count,
                        "FAILED": failed_count
                    },
                    "by_type": {
                        "embedding": embedding_count,
                        "reranker": reranker_count
                    }
                }
        except Exception as e:
            print(f"获取任务统计信息失败: {e}")
            return {"total": 0, "by_status": {}, "by_type": {}}
    
    # === 服务实例管理方法 ===
    
    @with_error_handling(context="database", default_return=[])
    def get_tasks_by_service_instance(self, service_instance_id: str) -> List[TrainingTaskDB]:
        """获取特定服务实例的任务"""
        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == service_instance_id
            ).order_by(TrainingTaskDB.created_at.desc())
            results = session.exec(statement).all()
            # 分离所有对象
            for result in results:
                session.expunge(result)
            return list(results)
    
    @with_error_handling(context="database", default_return=[])
    def get_running_tasks_by_service(self, service_instance_id: str) -> List[TrainingTaskDB]:
        """获取特定服务实例的运行中任务"""
        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == service_instance_id,
                TrainingTaskDB.status.in_(["RUNNING", "PENDING"])
            )
            results = session.exec(statement).all()
            # 分离所有对象
            for result in results:
                session.expunge(result)
            return list(results)
    
    @with_error_handling(context="database", default_return=False)
    def update_process_info(self, task_id: str, process_pid: Optional[int], 
                           process_status: Optional[str] = None, service_instance_id: Optional[str] = None) -> bool:
        """更新任务的进程信息（统一接口，支持单进程和多进程训练）"""
        logger.info(f"🔧 开始更新进程信息: task_id={task_id}, process_pid={process_pid}, process_status={process_status}")
        
        with safe_get_session() as session:
            task = session.get(TrainingTaskDB, task_id)
            if not task:
                logger.error(f"❌ 任务不存在: {task_id}")
                return False
            
            logger.info(f"🔍 找到任务: {task_id}, 当前PID: {task.process_pid}")
            
            if process_pid is not None:
                old_pid = task.process_pid
                task.process_pid = process_pid
                logger.info(f"🔧 更新PID: {old_pid} -> {process_pid}")
                
            if process_status is not None:
                # 🔧 统一使用枚举值，确保类型一致性
                from bubble_rag.training.model_sft.enums import ProcessStatus
                if isinstance(process_status, str):
                    task.process_status = ProcessStatus(process_status)
                else:
                    task.process_status = process_status
                logger.info(f"🔧 更新进程状态: {process_status}")
                
            if service_instance_id is not None:
                task.service_instance_id = service_instance_id
                logger.info(f"🔧 更新服务实例ID: {service_instance_id}")
                
            task.updated_at = datetime.now()
            session.commit()
            
            # 验证更新是否成功
            session.refresh(task)
            logger.info(f"✅ 进程信息更新完成: task_id={task_id}, 最终PID={task.process_pid}")
            return True
    
    @with_error_handling(context="database", default_return=False)
    def is_task_owned_by_service(self, task_id: str, service_instance_id: str) -> bool:
        """检查任务是否属于指定的服务实例"""
        with safe_get_session() as session:
            task = session.get(TrainingTaskDB, task_id)
            if task:
                return task.service_instance_id == service_instance_id
            return False
    
    @with_error_handling(context="database", default_return=[])
    def get_orphaned_tasks(self) -> List[TrainingTaskDB]:
        """获取孤儿任务（没有service_instance_id的任务）"""
        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id.is_(None),
                TrainingTaskDB.status.in_(["RUNNING", "PENDING"])
            )
            results = session.exec(statement).all()
            # 分离所有对象
            for result in results:
                session.expunge(result)
            return list(results)
    
    @with_error_handling(context="database", default_return=[])
    def get_legacy_tasks_by_port(self, hostname: str, port: int) -> List[TrainingTaskDB]:
        """获取旧格式服务实例ID的运行中任务（用于兼容性处理）"""
        with safe_get_session() as session:
            # 查找匹配 hostname_*_port 模式的任务
            pattern = f"{hostname}_%_{port}"
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id.like(pattern),
                TrainingTaskDB.status.in_(["RUNNING", "PENDING"])
            )
            results = session.exec(statement).all()
            # 分离所有对象
            for result in results:
                session.expunge(result)
            return list(results)
    
    @with_error_handling(context="database", default_return=False)
    def migrate_legacy_service_instance_id(self, old_instance_id: str, new_instance_id: str) -> bool:
        """迁移旧的服务实例ID到新格式"""
        with safe_get_session() as session:
            # 更新所有匹配的任务
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == old_instance_id
            )
            tasks = session.exec(statement).all()
            
            updated_count = 0
            for task in tasks:
                task.service_instance_id = new_instance_id
                updated_count += 1
            
            session.commit()
            
            if updated_count > 0:
                logger.info(f"🔄 迁移了 {updated_count} 个任务从 {old_instance_id} 到 {new_instance_id}")
            
            return True

    @with_error_handling(context="database", default_return=[])
    def get_tasks_by_process_status(self, process_status: str) -> List[TrainingTaskDB]:
        """根据进程状态获取训练任务"""
        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.process_status == process_status
            )
            results = session.exec(statement).all()
            # 分离所有对象
            for result in results:
                session.expunge(result)
            return list(results)
    
    @with_error_handling(context="database", default_return={})
    def get_process_status_statistics(self) -> Dict[str, int]:
        """获取进程状态统计信息"""
        with safe_get_session() as session:
            statement = select(TrainingTaskDB.process_status).where(
                TrainingTaskDB.process_status.isnot(None)
            )
            results = session.exec(statement).all()
            
            # 统计每种状态的数量
            status_count = {}
            for status in results:
                status_count[status] = status_count.get(status, 0) + 1
                
            return status_count


# 全局训练任务服务实例
training_task_service = TrainingTaskService()

# 日志配置
import logging
logger = logging.getLogger(__name__)