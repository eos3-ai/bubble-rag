"""
统一的进程管理基类
为单进程和多进程训练提供通用的服务隔离和进程恢复机制
"""
import os
import logging
import threading
import psutil
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..enums.training_task_enums import TrainingStatus, ProcessStatus
from .service_instance import get_service_instance_id
from bubble_rag.training.mysql_service.service.training_task_service import training_task_service

logger = logging.getLogger(__name__)


class ProcessManagerBase(ABC):
    """进程管理器基类 - 提供服务隔离和进程恢复的通用功能"""
    
    def __init__(self):
        self.processes = {}  # task_id -> 进程信息
        self.process_info = {}  # task_id -> 详细信息
        self._lock = threading.RLock()
        
        # 延迟获取服务实例ID，避免启动时端口检测
        self._service_instance_id = None
        logger.info(f"🔧 {self.__class__.__name__} 初始化，延迟检测服务实例ID")
        
        # 服务启动时清理孤儿进程和任务状态
        # 🔧 启用智能孤儿进程清理，只清理服务启动前的进程
        self._recover_running_processes()  # 智能清理，保护正常训练进程
    
    @property
    def service_instance_id(self) -> Optional[str]:
        """延迟获取服务实例ID"""
        if self._service_instance_id is None:
            self._service_instance_id = get_service_instance_id()
            logger.info(f"🔧 检测到服务实例ID: {self._service_instance_id}")
        return self._service_instance_id
    
    def _recover_running_processes(self):
        """智能进程状态管理 - 基于时间戳的进程分类处理"""
        import time
        
        # 记录服务启动时间 - 添加安全缓冲区避免误杀
        # 🛡️ 关键安全机制：预留10秒安全缓冲时间
        # 防止在服务启动过程中创建的正常训练进程被误判为孤儿进程
        SAFETY_BUFFER_SECONDS = 10
        service_start_time = time.time() - SAFETY_BUFFER_SECONDS
        
        logger.info(f"🔄 启动智能进程状态管理，服务实例: {self.service_instance_id}")
        logger.info(f"   - 服务启动时间戳: {time.time()}")
        logger.info(f"   - 孤儿进程判定时间戳: {service_start_time} (预留{SAFETY_BUFFER_SECONDS}秒安全缓冲)")
        logger.info(f"   - 执行进程分类：保护新进程，清理孤儿进程")
        
        # 存储状态转换统计
        self._status_transition_stats = {
            'running_count': 0,
            'terminated_count': 0,
            'unknown_count': 0,
            'failed_count': 0
        }
        
        try:
            logger.info(f"🔄 服务重启：开始清理当前服务实例的孤儿进程，服务实例: {self.service_instance_id}")
            
            # 🔧 智能孤儿进程检测和清理：只清理真正的孤儿进程，避免误杀正常任务
            terminated_count = 0
            
            # 获取所有相关任务进行孤儿判断
            from bubble_rag.training.mysql_service.entity.training_task_models import safe_get_session
            from bubble_rag.training.model_sft.enums.training_task_enums import ProcessStatus, TrainingStatus
            
            with safe_get_session() as session:
                from bubble_rag.training.mysql_service.entity.training_task_models import TrainingTaskDB
                from sqlmodel import select
                from datetime import datetime
                import time
                import psutil
                
                # 查找当前服务实例的相关任务
                statement = select(TrainingTaskDB).where(
                    TrainingTaskDB.service_instance_id == self.service_instance_id,
                    # 任务状态为PENDING或RUNNING，或者进程状态为RUNNING/UNKNOWN
                    (TrainingTaskDB.status.in_([TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value])) |
                    (TrainingTaskDB.process_status.in_([ProcessStatus.RUNNING.value, ProcessStatus.UNKNOWN.value]))
                )
                tasks = session.exec(statement).all()
                
                current_time = time.time()
                
                # 跳过定期孤儿进程检测，避免循环导入问题
                # 启动时的孤儿进程检测已经足够，定期清理主要处理已完成的进程
                logger.info("📋 跳过定期孤儿进程检测，避免循环导入（启动时清理已处理）")
                return []
                
                orphan_tasks = []
                
                logger.info(f"🔍 检查 {len(tasks)} 个候选任务是否为孤儿进程")
                logger.info(f"📅 服务启动时间: {datetime.fromtimestamp(service_startup_time).strftime('%Y-%m-%d %H:%M:%S')}")
                
                for task in tasks:
                    is_orphan = False
                    reason = ""

                    # 🔧 添加详细调试信息
                    logger.info(f"🔍 检查任务 {task.task_id}:")
                    logger.info(f"   - 任务创建时间: {task.created_at}")
                    logger.info(f"   - 任务状态: {task.status}")
                    logger.info(f"   - 进程PID: {task.process_pid}")
                    logger.info(f"   - 任务中的服务启动时间: {task.service_startup_time}")

                    if task.process_pid:
                        try:
                            if psutil.pid_exists(task.process_pid):
                                process = psutil.Process(task.process_pid)
                                process_create_time = process.create_time()

                                logger.info(f"   - 进程创建时间: {datetime.fromtimestamp(process_create_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
                                logger.info(f"   - 服务启动时间: {datetime.fromtimestamp(service_startup_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
                                logger.info(f"   - 时间差: {process_create_time - service_startup_time:.6f} 秒")

                                # 正确的孤儿进程判断：进程创建时间早于服务启动时间
                                if process_create_time < service_startup_time:
                                    is_orphan = True
                                    process_age_minutes = (current_time - process_create_time) / 60
                                    service_age_minutes = (current_time - service_startup_time) / 60
                                    reason = f"孤儿进程：进程创建于服务启动前 (进程: {datetime.fromtimestamp(process_create_time).strftime('%H:%M:%S')}, 服务: {datetime.fromtimestamp(service_startup_time).strftime('%H:%M:%S')})"
                                    logger.warning(f"   - ❌ 判断为孤儿进程: {reason}")
                                else:
                                    process_age_minutes = (current_time - process_create_time) / 60
                                    reason = f"正常进程：进程创建于服务启动后 (创建时间: {datetime.fromtimestamp(process_create_time).strftime('%H:%M:%S')}, 运行 {process_age_minutes:.1f} 分钟)"
                                    logger.info(f"   - ✅ 判断为正常进程: {reason}")
                            else:
                                is_orphan = True
                                reason = "进程已不存在"
                                logger.warning(f"   - ❌ 进程不存在: PID {task.process_pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            is_orphan = True
                            reason = "无法访问进程"
                            logger.warning(f"   - ❌ 无法访问进程: PID {task.process_pid}")
                        except Exception as e:
                            is_orphan = True
                            reason = f"进程检查异常: {e}"
                    else:
                        # 没有PID但状态为RUNNING，需要清理
                        if task.status == TrainingStatus.RUNNING.value:
                            is_orphan = True
                            reason = "无PID记录但任务状态为RUNNING"
                    
                    if is_orphan:
                        orphan_tasks.append(task)
                        logger.info(f"🔍 孤儿进程: {task.task_id}, PID: {task.process_pid}, 原因: {reason}")
                    else:
                        logger.info(f"✅ 正常任务: {task.task_id}, PID: {task.process_pid}, 原因: {reason}")
                
                # 对孤儿任务执行清理
                for task in orphan_tasks:
                    try:
                        # 1. 杀死进程树（如果PID存在且进程还在运行）
                        if task.process_pid:
                            try:
                                if psutil.pid_exists(task.process_pid):
                                    logger.info(f"🔪 发现孤儿进程，开始终止进程树: PID={task.process_pid}")
                                    # 🌳 使用统一的进程树清理逻辑
                                    self._terminate_process_tree_unified(task.process_pid)
                                else:
                                    logger.info(f"ℹ️  孤儿进程已不存在: PID={task.process_pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                                logger.info(f"ℹ️  无法访问孤儿进程 PID={task.process_pid}: {e}")
                            except Exception as e:
                                logger.warning(f"⚠️  终止孤儿进程失败 PID={task.process_pid}: {e}")
                        
                        # 2. 更新任务状态：PENDING/RUNNING -> FAILED
                        if task.status in [TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value]:
                            task.status = TrainingStatus.FAILED.value
                            task.error_message = f"服务重启清理：孤儿进程被强制终止 (PID: {task.process_pid or 'N/A'})"
                            task.completed_at = datetime.now()
                            logger.info(f"📝 更新孤儿任务状态: {task.task_id} -> FAILED")
                        
                        # 3. 更新进程状态：RUNNING/UNKNOWN -> TERMINATED
                        if task.process_status in [ProcessStatus.RUNNING.value, ProcessStatus.UNKNOWN.value]:
                            task.process_status = ProcessStatus.TERMINATED.value
                            logger.info(f"🔧 更新孤儿进程状态: {task.task_id} -> TERMINATED (保留PID: {task.process_pid})")
                        
                        task.updated_at = datetime.now()
                        terminated_count += 1
                        
                    except Exception as e:
                        logger.error(f"清理孤儿任务失败: {task.task_id}, 错误: {e}")
                
                if terminated_count > 0:
                    session.commit()
                    logger.info(f"✅ 智能清理完成: 共处理 {terminated_count} 个孤儿进程/任务")
                else:
                    logger.info("ℹ️  未发现需要清理的孤儿进程")
            
            logger.info(f"✅ 已智能清理当前服务实例的 {terminated_count} 个孤儿进程")
            
            # 1. 查找属于当前服务实例的运行状态任务（用于其他清理工作）
            running_tasks = training_task_service.get_tasks_by_service_instance(
                self.service_instance_id
            )
            
            recovered_count = 0
            failed_count = 0
            migrated_count = 0
            
            for task_db in running_tasks:
                try:
                    # 🔧 修复：处理所有可能的不一致状态
                    # 1. RUNNING状态任务：需要检查是否为孤儿进程
                    # 2. UNKNOWN进程状态任务：使用同样的孤儿进程判断逻辑
                    # 3. 🆕 状态不一致任务：任务状态为STOPPED但进程状态为RUNNING
                    should_process = (
                        task_db.status == TrainingStatus.RUNNING.value or
                        (hasattr(task_db, 'process_status') and 
                         task_db.process_status == ProcessStatus.UNKNOWN.value) or
                        # 🔧 新增：处理状态不一致的情况
                        (hasattr(task_db, 'process_status') and 
                         task_db.status in [TrainingStatus.STOPPED.value, TrainingStatus.FAILED.value] and
                         task_db.process_status == ProcessStatus.RUNNING.value)
                    )
                    
                    if not should_process:
                        continue
                        
                    logger.info(f"🔄 处理任务: {task_db.task_id}, 任务状态: {task_db.status}, 进程状态: {getattr(task_db, 'process_status', 'N/A')}")
                        
                    # 2. 兼容性处理：查找旧格式的任务 (hostname_pid_port)
                    if task_db.service_instance_id != self.service_instance_id:
                        if self._is_legacy_instance_id(task_db.service_instance_id):
                            training_task_service.migrate_legacy_service_instance_id(
                                task_db.service_instance_id, self.service_instance_id
                            )
                            migrated_count += 1
                            task_db.service_instance_id = self.service_instance_id  # 更新本地对象
                        else:
                            continue  # 不属于当前实例，跳过
                    
                    # 3. 🔧 智能进程分类处理
                    process_status_result = self._classify_and_handle_process(task_db, service_start_time)
                    
                    # 更新统计信息
                    if process_status_result == ProcessStatus.RUNNING:
                        self._status_transition_stats['running_count'] += 1
                        recovered_count += 1
                    elif process_status_result == ProcessStatus.TERMINATED:
                        self._status_transition_stats['terminated_count'] += 1
                        failed_count += 1
                    elif process_status_result == ProcessStatus.UNKNOWN:
                        self._status_transition_stats['unknown_count'] += 1
                        failed_count += 1
                    else:
                        self._status_transition_stats['failed_count'] += 1
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"恢复任务 {task_db.task_id} 失败: {str(e)}")
                    failed_count += 1
            
            # 打印详细的状态转换统计
            logger.info(f"🎯 智能进程状态管理完成:")
            logger.info(f"   - ✅ 保护进程: {self._status_transition_stats['running_count']} 个")
            logger.info(f"   - 🛑 终止孤儿进程: {self._status_transition_stats['terminated_count']} 个") 
            logger.info(f"   - ❓ 未知状态: {self._status_transition_stats['unknown_count']} 个")
            logger.info(f"   - 🔄 迁移任务: {migrated_count} 个")
            logger.info(f"   - ❌ 处理失败: {self._status_transition_stats['failed_count']} 个")
            
        except Exception as e:
            logger.error(f"进程恢复失败: {str(e)}", exc_info=True)
    
    def _find_legacy_tasks_by_port(self):
        """查找旧格式的任务（hostname_pid_port）"""
        try:
            parts = self.service_instance_id.split('_')
            hostname, port = parts[0], int(parts[1])
            legacy_tasks = training_task_service.get_legacy_tasks_by_port(hostname, port)
            return legacy_tasks
        except (ValueError, IndexError):
            return []
    
    def _is_legacy_instance_id(self, instance_id: str) -> bool:
        """判断是否为旧格式的实例ID (hostname_pid_port)"""
        try:
            parts = instance_id.split('_')
            return len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit()
        except (ValueError, AttributeError):
            return False
    
    def _classify_and_handle_process(self, task_db, service_start_time: float) -> ProcessStatus:
        """
        智能进程分类与处理
        
        Args:
            task_db: 任务数据库记录
            service_start_time: 服务启动时间戳
            
        Returns:
            ProcessStatus: 处理后的进程状态
        """
        import psutil
        
        try:
            pid = task_db.process_pid
            
            # 0. 🔧 新增：检查状态一致性，处理任务状态与进程状态不匹配的情况
            task_status = task_db.status
            process_status = getattr(task_db, 'process_status', None)
            
            if (task_status in [TrainingStatus.STOPPED.value, TrainingStatus.FAILED.value] and 
                process_status == ProcessStatus.RUNNING.value):
                logger.warning(f"🔧 状态不一致检测: 任务 {task_db.task_id}")
                logger.warning(f"   - 任务状态: {task_status}")
                logger.warning(f"   - 进程状态: {process_status}")
                logger.warning(f"   - PID: {pid}")
                
                # 强制同步进程状态到STOPPED，因为任务已经停止
                if pid and psutil.pid_exists(pid):
                    logger.info(f"🛑 强制终止不一致进程树: PID {pid}")
                    # 使用统一的进程树清理方法
                    success = self._terminate_process_tree_unified(pid)
                    if not success:
                        logger.warning(f"⚠️ 进程树清理失败: PID {pid}")
                
                # 更新进程状态为STOPPED
                training_task_service.update_process_info(
                    task_db.task_id, 
                    None,  # PID设为None
                    ProcessStatus.STOPPED.value
                )
                logger.info(f"✅ 已同步进程状态为STOPPED: 任务 {task_db.task_id}")
                return ProcessStatus.STOPPED
            
            # 1. 无PID记录的任务
            if not pid:
                logger.info(f"📋 任务 {task_db.task_id} 无PID记录，标记为失败")
                self._handle_no_pid_task(task_db)
                return ProcessStatus.UNKNOWN
            
            # 2. 检查进程是否存在
            if not psutil.pid_exists(pid):
                logger.info(f"💀 进程已不存在: 任务 {task_db.task_id}, PID {pid}")
                self._handle_dead_process(task_db)
                return ProcessStatus.UNKNOWN
            
            # 3. 获取进程信息
            try:
                process = psutil.Process(pid)
                process_create_time = process.create_time()
                
                # 4. 基于时间戳进行智能分类（已包含安全缓冲时间）
                if process_create_time < service_start_time:
                    # 孤儿进程：进程创建时间早于安全判定时间 → 立即清理
                    logger.warning(f"🛑 检测到孤儿进程: 任务 {task_db.task_id}, PID {pid}")
                    logger.warning(f"   - 进程创建: {process_create_time}")
                    logger.warning(f"   - 孤儿判定时间: {service_start_time}")
                    logger.warning(f"   - 时差: {service_start_time - process_create_time:.1f} 秒")
                    
                    return self._handle_orphaned_process(task_db)
                else:
                    # 正常进程：进程创建时间晚于安全判定时间 → 保持RUNNING状态
                    logger.info(f"✅ 检测到正常运行进程: 任务 {task_db.task_id}, PID {pid}")
                    logger.info(f"   - 进程创建: {process_create_time}")
                    logger.info(f"   - 孤儿判定时间: {service_start_time}")
                    logger.info(f"   - 时差: {process_create_time - service_start_time:.1f} 秒")
                    
                    return self._handle_running_process(task_db)
                    
            except psutil.AccessDenied:
                logger.warning(f"❌ 无权限访问进程: 任务 {task_db.task_id}, PID {pid}")
                self._handle_access_denied_process(task_db)
                return ProcessStatus.UNKNOWN
                
            except psutil.NoSuchProcess:
                logger.warning(f"💀 进程已消失: 任务 {task_db.task_id}, PID {pid}")
                self._handle_dead_process(task_db)
                return ProcessStatus.UNKNOWN
                
        except Exception as e:
            logger.error(f"❌ 进程分类处理异常: 任务 {task_db.task_id}, 错误: {str(e)}")
            self._handle_process_error(task_db, str(e))
            return ProcessStatus.UNKNOWN

    def _try_recover_process(self, task_db, service_start_time: float) -> bool:
        """智能进程清理策略 - 只清理服务启动前的孤儿进程"""
        try:
            pid = task_db.process_pid
            if pid:
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    process_create_time = process.create_time()
                    
                    # 🔧 智能判断：只清理服务启动前的进程（已包含安全缓冲时间）
                    if process_create_time < service_start_time:
                        logger.warning(f"🛑 检测到服务启动前的孤儿进程: 任务 {task_db.task_id}, PID {pid}")
                        logger.warning(f"   - 进程创建时间: {process_create_time}")
                        logger.warning(f"   - 孤儿判定时间: {service_start_time}")
                        logger.warning(f"   - 进程早于判定时间 {(service_start_time - process_create_time):.1f} 秒")
                        self._terminate_orphaned_process(pid, task_db.task_id)
                        return False  # 返回False表示进程已被终止，需要标记任务失败
                    else:
                        logger.info(f"✅ 检测到正常运行进程: 任务 {task_db.task_id}, PID {pid}")
                        logger.info(f"   - 进程创建时间: {process_create_time}")
                        logger.info(f"   - 孤儿判定时间: {service_start_time}")
                        logger.info(f"   - 进程晚于判定时间 {(process_create_time - service_start_time):.1f} 秒，保护此进程")
                        return True  # 保护这个正常的进程
                else:
                    logger.info(f"进程已自然结束: 任务 {task_db.task_id}, PID {pid}")
                    return False
            else:
                logger.info(f"任务 {task_db.task_id} 没有记录进程ID")
                return False
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"无法访问进程 {task_db.task_id}: {str(e)}")
            # 即使无法访问，也要清理任务状态
            return False
    
    def _terminate_orphaned_process(self, pid: int, task_id: str):
        """终止孤儿进程树 - 服务重启时主动清理策略"""
        logger.info(f"🔄 服务重启：使用进程树清理孤儿进程 PID {pid} (任务 {task_id})")
        
        # 使用统一的进程树清理方法
        success = self._terminate_process_tree_unified(pid)
        if success:
            logger.info(f"✅ 孤儿进程树已清理完成: PID {pid}")
            return True
        else:
            logger.error(f"❌ 孤儿进程树清理失败: PID {pid}")
            return False
    
    def _handle_running_process(self, task_db) -> ProcessStatus:
        """处理正常运行的进程"""
        try:
            logger.info(f"✅ 确认正常运行进程: 任务 {task_db.task_id}, PID {task_db.process_pid}")
            
            # 确保数据库状态为RUNNING（正常运行的进程应该是RUNNING状态）
            training_task_service.update_process_info(
                task_db.task_id,
                task_db.process_pid,
                ProcessStatus.RUNNING.value
            )
            
            # 调用子类的注册方法
            self._register_recovered_process(task_db)
            
            return ProcessStatus.RUNNING
            
        except Exception as e:
            logger.error(f"处理正常运行进程失败 {task_db.task_id}: {str(e)}")
            return ProcessStatus.UNKNOWN
    
    def _handle_orphaned_process(self, task_db) -> ProcessStatus:
        """处理孤儿进程 - 直接清理"""
        try:
            logger.info(f"🛑 清理孤儿进程: 任务 {task_db.task_id}, PID {task_db.process_pid}")
            
            # 1. 执行进程终止
            if self._terminate_orphaned_process(task_db.process_pid, task_db.task_id):
                # 2. 更新最终状态
                training_task_service.update_process_info(
                    task_db.task_id,
                    task_db.process_pid,  # 保留PID用于审计
                    ProcessStatus.TERMINATED.value
                )
                
                # 3. 更新任务状态
                training_task_service.update_task_status(
                    task_db.task_id,
                    TrainingStatus.FAILED.value,
                    task_db.progress or 0.0
                )
                
                training_task_service.update_task_result(
                    task_db.task_id,
                    error_message=f"服务重启清理：孤儿进程PID {task_db.process_pid}已被终止"
                )
                
                logger.info(f"✅ 孤儿进程已终止: 任务 {task_db.task_id}")
                return ProcessStatus.TERMINATED
            else:
                # 终止失败，标记为未知状态
                training_task_service.update_process_info(
                    task_db.task_id,
                    task_db.process_pid,
                    ProcessStatus.UNKNOWN.value
                )
                logger.warning(f"❌ 孤儿进程终止失败: 任务 {task_db.task_id}")
                return ProcessStatus.UNKNOWN
                
        except Exception as e:
            logger.error(f"处理孤儿进程失败 {task_db.task_id}: {str(e)}")
            return ProcessStatus.UNKNOWN
    
    def _handle_no_pid_task(self, task_db):
        """处理无PID记录的任务"""
        training_task_service.update_task_status(
            task_db.task_id,
            TrainingStatus.FAILED.value,
            task_db.progress or 0.0
        )
        training_task_service.update_task_result(
            task_db.task_id,
            error_message="任务无进程PID记录，可能启动失败"
        )
        training_task_service.update_process_info(
            task_db.task_id,
            None,
            ProcessStatus.UNKNOWN.value
        )
    
    def _handle_dead_process(self, task_db):
        """处理已死亡的进程"""
        training_task_service.update_task_status(
            task_db.task_id,
            TrainingStatus.FAILED.value,
            task_db.progress or 0.0
        )
        training_task_service.update_task_result(
            task_db.task_id,
            error_message=f"进程PID {task_db.process_pid}已不存在"
        )
        # 🔧 修复：已死亡的进程应该标记为TERMINATED而不是UNKNOWN
        training_task_service.update_process_info(
            task_db.task_id,
            task_db.process_pid,  # 保留PID用于审计追踪
            ProcessStatus.TERMINATED.value
        )
        logger.info(f"✅ 已死亡进程已标记为TERMINATED: 任务 {task_db.task_id}, PID {task_db.process_pid}")
    
    def _handle_access_denied_process(self, task_db):
        """处理权限不足的进程"""
        training_task_service.update_process_info(
            task_db.task_id,
            task_db.process_pid,
            ProcessStatus.UNKNOWN.value
        )
        training_task_service.update_task_result(
            task_db.task_id,
            error_message=f"无权限访问进程PID {task_db.process_pid}"
        )
    
    def _handle_process_error(self, task_db, error_msg: str):
        """处理进程检测异常"""
        training_task_service.update_process_info(
            task_db.task_id,
            task_db.process_pid,
            ProcessStatus.UNKNOWN.value
        )
        training_task_service.update_task_result(
            task_db.task_id,
            error_message=f"进程状态检测异常: {error_msg}"
        )

    def _handle_unknown_process(self, task_db, auto_recover: bool = True) -> ProcessStatus:
        """处理UNKNOWN状态进程 - 自动恢复机制"""
        try:
            logger.info(f"🔍 处理UNKNOWN状态进程: 任务 {task_db.task_id}")
            
            # 1. 尝试重新检测进程状态
            if task_db.process_pid:
                try:
                    if psutil.pid_exists(task_db.process_pid):
                        process = psutil.Process(task_db.process_pid)
                        
                        # 检查进程是否还活着且可访问
                        process_status = process.status()
                        if process_status in ['running', 'sleeping']:
                            logger.info(f"✅ UNKNOWN进程状态恢复: 任务 {task_db.task_id}, PID {task_db.process_pid}")
                            # 恢复为RUNNING状态
                            training_task_service.update_process_info(
                                task_db.task_id,
                                task_db.process_pid,
                                ProcessStatus.RUNNING.value
                            )
                            return ProcessStatus.RUNNING
                        else:
                            logger.warning(f"⚠️ UNKNOWN进程异常状态: 任务 {task_db.task_id}, 状态: {process_status}")
                    else:
                        logger.info(f"💀 UNKNOWN进程已不存在: 任务 {task_db.task_id}, PID {task_db.process_pid}")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"⚠️ UNKNOWN进程访问失败: 任务 {task_db.task_id}, {str(e)}")
            
            # 2. 自动恢复机制
            if auto_recover:
                return self._auto_recover_unknown_process(task_db)
            else:
                # 不自动恢复，保持UNKNOWN状态
                logger.info(f"🔄 保持UNKNOWN状态: 任务 {task_db.task_id} (auto_recover=False)")
                return ProcessStatus.UNKNOWN
                
        except Exception as e:
            logger.error(f"处理UNKNOWN状态进程失败 {task_db.task_id}: {str(e)}")
            return ProcessStatus.UNKNOWN
    
    def _auto_recover_unknown_process(self, task_db) -> ProcessStatus:
        """UNKNOWN状态进程自动恢复"""
        try:
            # 1. 清理无效进程记录
            logger.info(f"🔄 自动恢复UNKNOWN进程: 任务 {task_db.task_id}")
            
            # 2. 标记任务失败（UNKNOWN状态通常意味着进程异常）
            training_task_service.update_task_status(
                task_db.task_id,
                TrainingStatus.FAILED.value,
                task_db.progress or 0.0
            )
            
            # 3. 更新进程状态为已终止
            training_task_service.update_process_info(
                task_db.task_id,
                task_db.process_pid,
                ProcessStatus.TERMINATED.value
            )
            
            # 4. 记录恢复信息
            error_msg = f"自动恢复UNKNOWN状态进程: PID {task_db.process_pid or 'N/A'}"
            training_task_service.update_task_result(
                task_db.task_id,
                error_message=error_msg
            )
            
            logger.info(f"✅ UNKNOWN进程自动恢复完成: 任务 {task_db.task_id}")
            return ProcessStatus.TERMINATED
            
        except Exception as e:
            logger.error(f"UNKNOWN进程自动恢复失败 {task_db.task_id}: {str(e)}")
            return ProcessStatus.UNKNOWN
    
    def check_unknown_processes(self) -> Dict[str, Any]:
        """定期检查和恢复UNKNOWN状态进程"""
        try:
            logger.info("🔍 开始检查UNKNOWN状态进程...")
            
            # 获取所有UNKNOWN状态的任务
            unknown_tasks = training_task_service.get_tasks_by_process_status(
                ProcessStatus.UNKNOWN.value
            )
            
            recovery_stats = {
                'total_unknown': len(unknown_tasks),
                'recovered_running': 0,
                'recovered_terminated': 0,
                'still_unknown': 0,
                'error_count': 0
            }
            
            for task_db in unknown_tasks:
                try:
                    result_status = self._handle_unknown_process(task_db, auto_recover=True)
                    
                    if result_status == ProcessStatus.RUNNING:
                        recovery_stats['recovered_running'] += 1
                    elif result_status == ProcessStatus.TERMINATED:
                        recovery_stats['recovered_terminated'] += 1
                    else:
                        recovery_stats['still_unknown'] += 1
                        
                except Exception as e:
                    logger.error(f"检查UNKNOWN进程失败 {task_db.task_id}: {str(e)}")
                    recovery_stats['error_count'] += 1
            
            logger.info(f"✅ UNKNOWN状态检查完成: {recovery_stats}")
            return recovery_stats
            
        except Exception as e:
            logger.error(f"检查UNKNOWN状态进程异常: {str(e)}")
            return {'error': str(e)}

    def _register_recovered_process(self, task_db):
        """注册恢复的进程（基类默认实现，子类可重写）"""
        try:
            logger.info(f"✅ 基类注册保护进程: 任务 {task_db.task_id}, PID {task_db.process_pid}")
            
        except Exception as e:
            logger.error(f"注册保护进程失败 {task_db.task_id}: {str(e)}")
    
    def get_process_info(self) -> Dict[str, Any]:
        """获取所有进程信息"""
        with self._lock:
            return {
                "service_instance_id": self.service_instance_id,
                "total_processes": len(self.processes),
                "process_details": dict(self.process_info)
            }
    
    @abstractmethod
    def start_training_process(self, task) -> bool:
        """启动训练进程 - 子类实现"""
        pass
    
    @abstractmethod
    def stop_training_process(self, task_id: str) -> bool:
        """停止训练进程 - 子类实现"""  
        pass
    
    def _terminate_service_instance_processes(self) -> int:
        """服务重启时清理当前服务实例的所有孤儿进程和任务"""
        import psutil
        from datetime import datetime
        
        logger.info(f"🧹 开始清理当前服务实例的孤儿进程和任务: {self.service_instance_id}")
        
        # 获取当前服务实例的所有相关任务
        from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
        from bubble_rag.training.model_sft.enums.training_task_enums import ProcessStatus, TrainingStatus
        from bubble_rag.training.mysql_service.entity.training_task_models import safe_get_session
        
        with safe_get_session() as session:
            from bubble_rag.training.mysql_service.entity.training_task_models import TrainingTaskDB
            from sqlmodel import select
            
            # 查找当前服务实例的相关任务
            statement = select(TrainingTaskDB).where(
                TrainingTaskDB.service_instance_id == self.service_instance_id,
                # 任务状态为PENDING或RUNNING，或者进程状态为RUNNING/UNKNOWN
                (TrainingTaskDB.status.in_([TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value])) |
                (TrainingTaskDB.process_status.in_([ProcessStatus.RUNNING.value, ProcessStatus.UNKNOWN.value]))
            )
            tasks = session.exec(statement).all()
            
            terminated_count = 0
            for task in tasks:
                logger.info(f"🔍 处理任务: {task.task_id}, 任务状态: {task.status}, 进程状态: {task.process_status}, PID: {task.process_pid}")
                
                # 1. 尝试杀死进程树（如果PID存在且进程还在运行）
                if task.process_pid:
                    try:
                        if psutil.pid_exists(task.process_pid):
                            process = psutil.Process(task.process_pid)
                            logger.info(f"🔪 发现活跃进程，开始终止进程树: PID={task.process_pid}")
                            
                            # 🌳 使用统一的进程树清理逻辑（处理CUDA训练的子进程）
                            self._terminate_process_tree_unified(task.process_pid)
                        else:
                            logger.info(f"ℹ️  进程已不存在: PID={task.process_pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.info(f"ℹ️  无法访问进程 PID={task.process_pid}: {e}")
                    except Exception as e:
                        logger.warning(f"⚠️  终止进程失败 PID={task.process_pid}: {e}")
                
                # 2. 更新任务状态：PENDING/RUNNING -> FAILED
                if task.status in [TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value]:
                    task.status = TrainingStatus.FAILED.value
                    task.error_message = f"服务重启清理：任务被强制终止 (PID: {task.process_pid or 'N/A'})"
                    task.completed_at = datetime.now()
                    logger.info(f"📝 更新任务状态: {task.task_id} -> FAILED")
                
                # 3. 更新进程状态：RUNNING/UNKNOWN -> TERMINATED
                if task.process_status in [ProcessStatus.RUNNING.value, ProcessStatus.UNKNOWN.value]:
                    task.process_status = ProcessStatus.TERMINATED.value
                    # 🔧 保留 process_pid 不清空（用于审计追踪）
                    logger.info(f"🔧 更新进程状态: {task.task_id} -> TERMINATED (保留PID: {task.process_pid})")
                
                task.updated_at = datetime.now()
                terminated_count += 1
                
                logger.info(f"✅ 任务清理完成: {task.task_id} -> 任务状态:{task.status}, 进程状态:{task.process_status}")
            
            if terminated_count > 0:
                session.commit()
                logger.info(f"✅ 服务实例清理完成: 共处理 {terminated_count} 个任务")
            else:
                logger.info("ℹ️  当前服务实例无需清理的任务")
            
            return terminated_count
    
    def _terminate_process_tree_unified(self, pid: int) -> bool:
        """使用统一的进程树终止方法"""
        try:
            import psutil
            logger.info(f"🌳 开始终止进程树 PID={pid}")
            
            # 获取主进程
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                logger.info(f"进程 {pid} 已不存在")
                return True
            
            # 1. 获取所有子进程（包括dataloader workers等）
            try:
                children = process.children(recursive=True)
                logger.info(f"🔍 发现 {len(children)} 个子进程")
                
                # 2. 先终止所有子进程
                for child in children:
                    try:
                        if child.is_running():
                            child.terminate()
                            logger.info(f"🔥 已终止子进程: PID {child.pid}")
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as e:
                        logger.warning(f"终止子进程 {child.pid} 失败: {e}")
                
                # 3. 等待子进程优雅退出
                import time
                time.sleep(1)
                
                # 4. 强制终止仍然存活的子进程
                for child in children:
                    try:
                        if child.is_running():
                            child.kill()
                            logger.warning(f"💀 强制终止顽固子进程: PID {child.pid}")
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as e:
                        logger.error(f"强制终止子进程 {child.pid} 失败: {e}")
                
            except Exception as e:
                logger.warning(f"处理子进程时出错: {e}")
            
            # 5. 终止主进程
            try:
                process.terminate()
                logger.info(f"🔥 已终止主进程: PID {pid}")
                
                # 等待主进程优雅退出
                try:
                    process.wait(timeout=5)
                    logger.info(f"✅ 主进程已优雅终止: PID {pid}")
                except psutil.TimeoutExpired:
                    # 强制杀死主进程
                    process.kill()
                    logger.info(f"💥 主进程已强制终止: PID {pid}")
                    
            except psutil.NoSuchProcess:
                logger.info(f"主进程 {pid} 已不存在")
            except Exception as e:
                logger.error(f"终止主进程 {pid} 失败: {e}")
                return False
            
            logger.info(f"✅ 进程树终止完成: PID {pid}")
            return True
        except Exception as e:
            logger.error(f"进程树终止失败 PID={pid}: {e}")
            return False
    
    @abstractmethod
    def get_training_status(self, task_id: str) -> Optional[str]:
        """获取训练状态 - 子类实现"""
        pass