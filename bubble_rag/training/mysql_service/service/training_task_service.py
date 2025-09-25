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
from bubble_rag.utils.user_manager import UserManager


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
            "service_startup_time": task_db.service_startup_time,
            "username": task_db.username,
            "process_pid": task_db.process_pid,
            "process_status": task_db.process_status,
            "loss_data": task_db.loss_data,  # 训练损失数据
            # 重启关系字段
            "base_task_id": task_db.base_task_id,
            "restart_count": task_db.restart_count
        }
    
    def ensure_tables_created(self) -> bool:
        """确保数据库表已创建"""
        try:
            return create_tables()
        except Exception as e:
            print(f"数据库表创建失败: {e}")
            return False
    
    @with_error_handling(context="database", default_return=False)
    def save_training_task(self, task: TrainingTask, training_params: Dict[str, Any] = None, service_instance_id: str = None, username: str = None) -> bool:
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
                # 更新用户信息（如果提供）
                if username:
                    existing.username = username
            else:
                # 创建新记录 - 传递正确的参数
                # 🔐 确定用户名（用于权限控制）
                if not username:
                    # 如果没有提供用户名，使用当前用户
                    current_user = UserManager.validate_and_get_user()
                    username = current_user.get('username', 'admin')

                db_task = TrainingTaskDB.from_training_task(
                    task,
                    training_params=training_params,
                    username=username,
                    service_instance_id=service_instance_id
                )

                # 新任务使用全局启动时间（这个时间在服务启动时就固定了）
                try:
                    from bubble_rag.model_sft_server import SERVICE_STARTUP_TIME
                    if SERVICE_STARTUP_TIME:
                        db_task.service_startup_time = SERVICE_STARTUP_TIME
                except ImportError:
                    # 如果无法导入，说明不是在服务环境中
                    pass
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
    
    def get_all_training_tasks(self, limit: int = 100, offset: int = 0, service_instance_id: Optional[str] = None) -> List[TrainingTaskDB]:
        """获取训练任务（建议使用get_tasks_by_service_instance代替）

        Args:
            limit: 限制返回数量
            offset: 偏移量
            service_instance_id: 服务实例ID，如果为None则返回所有任务（不推荐）

        Warning:
            不传service_instance_id参数可能导致跨服务实例数据泄露！
            建议使用 get_tasks_by_service_instance() 方法代替。
        """
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).order_by(TrainingTaskDB.created_at.desc())

                # 🔐 安全过滤：如果提供了service_instance_id，则只返回该服务的任务
                if service_instance_id is not None:
                    statement = statement.where(TrainingTaskDB.service_instance_id == service_instance_id)
                else:
                    # 记录不安全的调用
                    logger.warning("get_all_training_tasks() 被调用但没有service_instance_id过滤，可能存在数据泄露风险！")

                statement = statement.limit(limit).offset(offset)
                results = session.exec(statement).all()
                # 分离所有对象
                for result in results:
                    session.expunge(result)
                return list(results)
        except Exception as e:
            print(f"获取训练任务失败: {e}")
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
    
    def get_training_tasks_by_status(self, status: str, service_instance_id: Optional[str] = None) -> List[TrainingTaskDB]:
        """根据状态获取训练任务（建议指定service_instance_id）

        Args:
            status: 任务状态
            service_instance_id: 服务实例ID，如果为None则返回所有服务的任务（不推荐）

        Warning:
            不传service_instance_id参数可能导致跨服务实例数据泄露！
        """
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).where(TrainingTaskDB.status == status)

                # 🔐 安全过滤：如果提供了service_instance_id，则只返回该服务的任务
                if service_instance_id is not None:
                    statement = statement.where(TrainingTaskDB.service_instance_id == service_instance_id)
                else:
                    # 记录不安全的调用
                    logger.warning(f"get_training_tasks_by_status({status}) 被调用但没有service_instance_id过滤，可能存在数据泄露风险！")

                statement = statement.order_by(TrainingTaskDB.created_at.desc())
                results = session.exec(statement).all()

                # 分离所有对象
                for result in results:
                    session.expunge(result)
                return list(results)
        except Exception as e:
            print(f"根据状态获取训练任务失败: {e}")
            return []
    
    def get_training_tasks_by_type(self, train_type: str, service_instance_id: Optional[str] = None) -> List[TrainingTaskDB]:
        """根据训练类型获取训练任务（建议指定service_instance_id）

        Args:
            train_type: 训练类型
            service_instance_id: 服务实例ID，如果为None则返回所有服务的任务（不推荐）

        Warning:
            不传service_instance_id参数可能导致跨服务实例数据泄露！
        """
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).where(TrainingTaskDB.train_type == train_type)

                # 🔐 安全过滤：如果提供了service_instance_id，则只返回该服务的任务
                if service_instance_id is not None:
                    statement = statement.where(TrainingTaskDB.service_instance_id == service_instance_id)
                else:
                    # 记录不安全的调用
                    logger.warning(f"get_training_tasks_by_type({train_type}) 被调用但没有service_instance_id过滤，可能存在数据泄露风险！")

                statement = statement.order_by(TrainingTaskDB.created_at.desc())
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
            logger.error(f"更新训练任务状态失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
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
    
    def get_task_count(self, service_instance_id: Optional[str] = None) -> int:
        """获取任务总数（建议指定service_instance_id）

        Args:
            service_instance_id: 服务实例ID，如果为None则统计所有服务的任务（不推荐）

        Warning:
            不传service_instance_id参数可能导致跨服务实例数据泄露！
        """
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB)

                # 🔐 安全过滤：如果提供了service_instance_id，则只统计该服务的任务
                if service_instance_id is not None:
                    statement = statement.where(TrainingTaskDB.service_instance_id == service_instance_id)
                else:
                    # 记录不安全的调用
                    logger.warning("get_task_count() 被调用但没有service_instance_id过滤，可能存在数据泄露风险！")

                results = session.exec(statement).all()
                return len(results)
        except Exception as e:
            print(f"获取任务总数失败: {e}")
            return 0
    
    def get_task_stats(self, service_instance_id: Optional[str] = None) -> Dict[str, Any]:
        """获取任务统计信息（建议指定service_instance_id）

        Args:
            service_instance_id: 服务实例ID，如果为None则统计所有服务的任务（不推荐）

        Warning:
            不传service_instance_id参数可能导致跨服务实例数据泄露！
        """
        try:
            with safe_get_session() as session:
                base_statement = select(TrainingTaskDB)

                # 🔐 安全过滤：如果提供了service_instance_id，则只统计该服务的任务
                if service_instance_id is not None:
                    base_statement = base_statement.where(TrainingTaskDB.service_instance_id == service_instance_id)
                else:
                    # 记录不安全的调用
                    logger.warning("get_task_stats() 被调用但没有service_instance_id过滤，可能存在数据泄露风险！")

                # 统计各状态的任务数量（兼容大小写）
                pending_count = len(session.exec(base_statement.where(TrainingTaskDB.status.in_(["PENDING", "pending"]))).all())
                running_count = len(session.exec(base_statement.where(TrainingTaskDB.status.in_(["RUNNING", "running", "training"]))).all())
                succeeded_count = len(session.exec(base_statement.where(TrainingTaskDB.status.in_(["SUCCEEDED", "succeeded", "finished"]))).all())
                stopped_count = len(session.exec(base_statement.where(TrainingTaskDB.status.in_(["STOPPED", "stopped"]))).all())
                failed_count = len(session.exec(base_statement.where(TrainingTaskDB.status.in_(["FAILED", "failed"]))).all())

                # 统计各类型的任务数量
                embedding_count = len(session.exec(base_statement.where(TrainingTaskDB.train_type == "embedding")).all())
                reranker_count = len(session.exec(base_statement.where(TrainingTaskDB.train_type == "reranker")).all())

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
                    },
                    "service_instance_id": service_instance_id  # 标记统计的服务范围
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
        logger.info(f"开始更新进程信息: task_id={task_id}, process_pid={process_pid}, process_status={process_status}")
        
        with safe_get_session() as session:
            task = session.get(TrainingTaskDB, task_id)
            if not task:
                logger.error(f"❌ 任务不存在: {task_id}")
                return False
            
            logger.info(f"找到任务: {task_id}, 当前PID: {task.process_pid}")
            
            if process_pid is not None:
                old_pid = task.process_pid
                task.process_pid = process_pid
                logger.info(f"更新PID: {old_pid} -> {process_pid}")
                
            if process_status is not None:
                # 统一使用枚举值，确保类型一致性
                from bubble_rag.training.model_sft.enums import ProcessStatus
                if isinstance(process_status, str):
                    task.process_status = ProcessStatus(process_status)
                else:
                    task.process_status = process_status
                logger.info(f"更新进程状态: {process_status}")
                
            if service_instance_id is not None:
                task.service_instance_id = service_instance_id
                logger.info(f"更新服务实例ID: {service_instance_id}")
                
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

    @with_error_handling(context="database", default_return=False)
    def record_service_startup_time(self, service_instance_id: str, startup_time: datetime) -> bool:
        """记录服务实例启动时间到数据库"""
        with safe_get_session() as session:
            # 查找最新的任务，更新启动时间（如果没有任务，就不记录）
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == service_instance_id
            ).order_by(TrainingTaskDB.created_at.desc()).limit(1)

            latest_task = session.exec(statement).first()
            if latest_task:
                # 更新现有任务的启动时间
                latest_task.service_startup_time = startup_time
                latest_task.updated_at = datetime.now()
                session.add(latest_task)
                session.commit()
                logger.info(f"记录服务启动时间到任务 {latest_task.task_id}: {startup_time}")
                return True
            else:
                logger.warning(f"服务实例 {service_instance_id} 没有任务记录，无法记录启动时间")
                return False

    @with_error_handling(context="database", default_return=None)
    def get_service_startup_time(self, service_instance_id: str) -> Optional[datetime]:
        """获取服务实例启动时间"""
        with safe_get_session() as session:
            # 查找该服务实例的任务中记录的启动时间
            statement = select(TrainingTaskDB.service_startup_time).where(
                TrainingTaskDB.service_instance_id == service_instance_id,
                TrainingTaskDB.service_startup_time.isnot(None)
            ).order_by(TrainingTaskDB.created_at.desc()).limit(1)

            result = session.exec(statement).first()
            if result:
                logger.debug(f"获取到服务实例 {service_instance_id} 启动时间: {result}")
                return result
            else:
                logger.warning(f"未找到服务实例 {service_instance_id} 的启动时间记录")
                return None

    # ==================== 分层隔离方案 ====================

    # ========== 底层：纯服务隔离方法（技术层面） ==========
    # 用于孤儿进程检测、服务故障恢复、资源清理等技术功能

    @with_error_handling(context="database", default_return=[])
    def get_tasks_by_service_technical(self, service_instance_id: str, limit: int = 100, offset: int = 0) -> List[TrainingTaskDB]:
        """
        纯服务隔离获取任务（技术层面）
        用于孤儿进程检测、服务管理等技术功能
        """
        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == service_instance_id
            ).order_by(TrainingTaskDB.created_at.desc()).limit(limit).offset(offset)

            results = session.exec(statement).all()
            for result in results:
                session.expunge(result)

            return list(results)

    @with_error_handling(context="database", default_return=[])
    def get_running_tasks_by_service_technical(self, service_instance_id: str) -> List[TrainingTaskDB]:
        """
        纯服务隔离获取运行中任务（技术层面）
        用于服务实例的任务监控和管理
        """
        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == service_instance_id,
                TrainingTaskDB.status.in_(["RUNNING", "PENDING"])
            ).order_by(TrainingTaskDB.created_at.desc())

            results = session.exec(statement).all()
            for result in results:
                session.expunge(result)

            return list(results)

    # ========== 中层：纯用户权限方法（业务层面） ==========
    # 用于API接口、前端界面的业务权限控制，不关心服务分布

    @with_error_handling(context="database", default_return=[])
    def get_tasks_for_user_business(self, username: str = None, limit: int = None, offset: int = 0, user_info: dict = None) -> List[Dict[str, Any]]:
        """
        纯用户权限获取任务（业务层面）
        自动应用用户权限过滤，跨所有服务实例

        Args:
            username: 指定用户名（管理员可以查看任何用户，普通用户只能查看自己）
            limit: 限制返回数量
            offset: 偏移量
            user_info: 用户信息字典（包含username, user_role, is_admin等），如果不传则使用UserManager获取

        Returns:
            任务字典列表（包含用户信息）
        """
        # 优先使用传入的user_info，否则使用UserManager获取默认用户
        current_user = user_info if user_info else UserManager.validate_and_get_user()

        with safe_get_session() as session:
            statement = select(TrainingTaskDB).order_by(TrainingTaskDB.created_at.desc())

            # 权限过滤
            if not current_user.get('is_admin', False):
                # 普通用户只能看到自己的任务
                effective_username = current_user.get('username')
                statement = statement.where(TrainingTaskDB.username == effective_username)
            elif username:
                # 管理员指定查看某个用户的任务
                statement = statement.where(TrainingTaskDB.username == username)
            # 管理员不指定用户名时，查看所有任务

            if limit is not None:
                statement = statement.limit(limit).offset(offset)
            elif offset > 0:
                statement = statement.offset(offset)
            results = session.exec(statement).all()

            # 转换为字典并分离会话
            task_dicts = []
            for result in results:
                session.expunge(result)
                task_dicts.append(self._task_db_to_dict(result))

            return task_dicts

    @with_error_handling(context="database", default_return=[])
    def get_tasks_by_status_for_user_business(self, status: str, username: str = None) -> List[Dict[str, Any]]:
        """
        根据状态获取用户任务（业务层面）
        自动应用用户权限过滤，跨所有服务实例
        """
        current_user = UserManager.validate_and_get_user()

        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.status == self.normalize_status(status)
            ).order_by(TrainingTaskDB.created_at.desc())

            # 权限过滤
            if not current_user.get('is_admin', False):
                # 普通用户只能看到自己的任务
                effective_username = current_user.get('username')
                statement = statement.where(TrainingTaskDB.username == effective_username)
            elif username:
                # 管理员指定查看某个用户的任务
                statement = statement.where(TrainingTaskDB.username == username)

            results = session.exec(statement).all()

            # 转换为字典并分离会话
            task_dicts = []
            for result in results:
                session.expunge(result)
                task_dicts.append(self._task_db_to_dict(result))

            return task_dicts

    @with_error_handling(context="database", default_return=False)
    def can_user_access_task_business(self, task_id: str) -> bool:
        """
        检查当前用户是否可以访问任务（业务层面）
        自动应用用户权限检查，不关心服务归属
        """
        with safe_get_session() as session:
            task = session.get(TrainingTaskDB, task_id)
            if not task:
                return False

            return UserManager._can_access_task(task.username)

    @with_error_handling(context="database", default_return=0)
    def get_task_count_for_user_business(self, username: str = None) -> int:
        """
        获取用户任务总数（业务层面）
        自动应用用户权限过滤，跨所有服务实例
        """
        current_user = UserManager.validate_and_get_user()

        with safe_get_session() as session:
            statement = select(TrainingTaskDB)

            # 权限过滤
            if not current_user.get('is_admin', False):
                # 普通用户只能看到自己的任务
                effective_username = current_user.get('username')
                statement = statement.where(TrainingTaskDB.username == effective_username)
            elif username:
                # 管理员指定查看某个用户的任务
                statement = statement.where(TrainingTaskDB.username == username)

            return len(session.exec(statement).all())

    # ========== 高层：组合方法（特殊管理场景） ==========
    # 用于高级管理功能，需要同时考虑服务和用户维度

    @with_error_handling(context="database", default_return=[])
    def get_user_tasks_in_service_admin(self, service_instance_id: str, username: str = None,
                                       limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        管理员专用：获取指定服务实例中的用户任务（组合层面）
        需要管理员权限，用于高级管理功能

        使用场景：管理员需要查看"服务A中用户B的任务"

        Args:
            service_instance_id: 服务实例ID
            username: 用户名（可选）
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            任务字典列表
        """
        # 检查管理员权限
        current_user = UserManager.validate_and_get_user()
        if not current_user.get('is_admin', False):
            logger.warning(f"非管理员用户 {current_user.get('username')} 尝试访问组合查询功能")
            return []

        with safe_get_session() as session:
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == service_instance_id
            ).order_by(TrainingTaskDB.created_at.desc())

            # 可选的用户过滤
            if username:
                statement = statement.where(TrainingTaskDB.username == username)

            statement = statement.limit(limit).offset(offset)
            results = session.exec(statement).all()

            # 转换为字典并分离会话
            task_dicts = []
            for result in results:
                session.expunge(result)
                task_dicts.append(self._task_db_to_dict(result))

            return task_dicts

    @with_error_handling(context="database", default_return=[])
    def get_restart_tasks_by_base_id(self, base_task_id: str) -> List[TrainingTaskDB]:
        """
        根据基础任务ID获取所有重启任务
        """
        try:
            with safe_get_session() as session:
                statement = select(TrainingTaskDB).where(
                    TrainingTaskDB.base_task_id == base_task_id
                ).order_by(TrainingTaskDB.created_at.desc())

                results = session.exec(statement).all()

                # 分离会话以避免懒加载问题
                restart_tasks = []
                for result in results:
                    session.expunge(result)
                    restart_tasks.append(result)

                return restart_tasks

        except Exception as e:
            logger.error(f"查询重启任务失败: {e}")
            return []

    def get_tasks_with_process_pid(self, username: str = None) -> List[TrainingTaskDB]:
        """
        获取有process_pid的任务列表（用于跨服务运行任务查询）

        Args:
            username: 可选的用户名过滤，如果不传则返回所有任务

        Returns:
            List[TrainingTaskDB]: 有PID的任务列表
        """
        try:
            with safe_get_session() as session:
                # 构建查询条件
                conditions = [
                    TrainingTaskDB.process_pid.isnot(None),  # 有PID
                    TrainingTaskDB.process_pid > 0  # PID大于0
                ]

                # 添加用户过滤
                if username:
                    conditions.append(TrainingTaskDB.username == username)

                statement = select(TrainingTaskDB).where(*conditions).order_by(
                    TrainingTaskDB.started_at.desc()
                )

                results = session.exec(statement).all()

                # 分离会话以避免懒加载问题
                tasks_with_pid = []
                for result in results:
                    session.expunge(result)
                    tasks_with_pid.append(result)

                logger.debug(f"查询到 {len(tasks_with_pid)} 个有PID的任务，用户过滤: {username}")
                return tasks_with_pid

        except Exception as e:
            logger.error(f"查询有PID的任务失败: {e}")
            return []


# 全局训练任务服务实例
training_task_service = TrainingTaskService()

# 日志配置
import logging
logger = logging.getLogger(__name__)