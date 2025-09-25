"""
统一训练服务层
提供串行和并行两种训练模式，支持服务隔离和进程管理
"""
import os
import threading
import logging
import json
import traceback
import time
import psutil
import multiprocessing as mp
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from ..models.training_task import TrainingTaskCreateRequest, TrainingTask, TrainingStatus
from ..enums.training_task_enums import ProcessStatus
from .task_manager import task_manager
from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
from .config_service import config_service
from ..utils.gpu_resource_manager import gpu_resource_manager
# from ..utils.gpu_utils import setup_device_environment
from ..utils.service_instance import service_instance_manager
from ..utils.process_manager_base import ProcessManagerBase

logger = logging.getLogger(__name__)

class UnifiedTrainingService(ProcessManagerBase):
    """统一训练服务，支持串行(serial)和并行(parallel)两种模式"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化统一训练服务
        
        Args:
            config: 服务配置字典
        """
        super().__init__()  # 先调用父类构造函数
        self.config = config or {}
        
        # 通过property访问service_instance_id（不是直接赋值）
        instance_id = self.service_instance_id
        if not instance_id:
            error_msg = "❌ 服务实例ID创建失败！请检查配置文件中的 TRAINING_SERVER_PORT 设置。"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 初始化训练进程管理
        self.training_processes: Dict[str, mp.Process] = {}
        self._lock = threading.RLock()
        self.stop_training_flag = False
        
        # 获取训练模式配置
        self.default_mode = self.config.get("default_training_mode", "parallel")
        
        logger.info(f"✅ 统一训练服务初始化完成，服务实例ID: {instance_id}")
        logger.info(f"默认训练模式: {self.default_mode}")
        logger.info("📝 注意：孤儿进程清理已由父类 ProcessManagerBase._recover_running_processes() 完成")
    
    def start_training(self, request: TrainingTaskCreateRequest, training_mode: str = None) -> TrainingTask:
        """
        启动训练任务（统一入口）
        
        Args:
            request: 训练任务创建请求
            training_mode: 训练模式 ("serial" | "parallel")，默认使用配置的默认模式
            
        Returns:
            创建的训练任务
        """
        # 再次检查服务实例ID - 确保服务隔离
        if not self.service_instance_id:
            error_msg = "❌ 服务实例ID为空，无法创建训练任务！服务隔离失败。"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 确定训练模式
        mode = training_mode or self.default_mode
        
        if mode not in ["serial", "parallel"]:
            raise ValueError(f"不支持的训练模式: {mode}，支持的模式: serial, parallel")
        
        logger.info(f"🚀 启动训练任务，模式: {mode}, 服务实例: {self.service_instance_id}")
        
        try:
            # 创建任务，传递服务实例ID
            task = task_manager.create_task(request, service_instance_id=self.service_instance_id)
            logger.info(f"创建训练任务成功: {task.task_id}, 模式: {mode}, 服务实例: {self.service_instance_id}")
            
            # 保存到数据库（与其他训练服务保持一致）
            try:
                # 🔐 获取当前用户信息用于权限控制
                from bubble_rag.utils.user_manager import UserManager
                current_user = UserManager.validate_and_get_user()
                username = current_user.get('username', 'admin')

                training_task_service.save_training_task(
                    task,
                    request.training_params,
                    service_instance_id=self.service_instance_id,
                    username=username
                )
                logger.info(f"任务已保存到数据库: {task.task_id} (用户: {username})")
            except Exception as db_error:
                logger.warning(f"保存任务到数据库失败（但任务已创建）: {str(db_error)}")
                # 不抛出异常，允许任务继续执行，后续状态更新时会再次尝试保存
            
            # 根据模式执行不同的训练逻辑
            if mode == "serial":
                return self._execute_serial_training(task)
            else:  # parallel
                return self._execute_parallel_training(task)
                
        except Exception as e:
            error_msg = f"启动训练失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    def _execute_serial_training(self, task: TrainingTask) -> TrainingTask:
        """
        执行串行训练（一次只能运行一个任务）
        
        Args:
            task: 训练任务
            
        Returns:
            训练任务
        """
        with self._lock:
            # 检查是否有正在运行的任务
            running_tasks = self.get_running_processes()
            if running_tasks:
                running_task_ids = list(running_tasks.keys())
                error_msg = f"串行训练模式下已有任务正在运行: {running_task_ids}，请等待完成后再启动新任务"
                logger.error(error_msg)
                task_manager.fail_task(task.task_id, error_msg)
                raise RuntimeError(error_msg)
        
        logger.info(f"🔄 执行串行训练: {task.task_id}")
        return self._start_training_process(task, mode="serial")
    
    def _execute_parallel_training(self, task: TrainingTask) -> TrainingTask:
        """
        执行并行训练（可以同时运行多个任务）
        
        Args:
            task: 训练任务
            
        Returns:
            训练任务
        """
        logger.info(f"🔄 执行并行训练: {task.task_id}")
        return self._start_training_process(task, mode="parallel")
    
    def _start_training_process(self, task: TrainingTask, mode: str) -> TrainingTask:
        """
        启动训练子进程
        
        Args:
            task: 训练任务
            mode: 训练模式
            
        Returns:
            训练任务
        """
        # 最终检查：确保有服务实例ID才能创建进程
        if not self.service_instance_id:
            error_msg = f"❌ 服务实例ID为空，拒绝创建训练进程！任务: {task.task_id}"
            logger.error(error_msg)
            task_manager.fail_task(task.task_id, error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # 分配GPU资源
            if mode == "parallel":
                allocated_device = gpu_resource_manager.allocate_gpus_for_task(
                    task.task_id, 
                    task.device
                )
            else:  # serial
                # 串行模式使用配置的设备或auto
                allocated_device = self.config.get("allocated_device") or task.device or "auto"
            
            logger.info(f"🔧 任务 {task.task_id} 分配设备: {allocated_device}")
            
            # 更新任务对象的device字段为实际分配的设备
            task.device = allocated_device
            task_manager.update_task(task.task_id, {"device": allocated_device})  # 更新内存中的任务
            
            # 更新数据库中的device字段
            try:
                # 🔐 获取当前用户信息
                from bubble_rag.utils.user_manager import UserManager
                current_user = UserManager.validate_and_get_user()
                username = current_user.get('username', 'admin')

                training_task_service.save_training_task(
                    task,
                    task.training_params,
                    service_instance_id=self.service_instance_id,
                    username=username
                )
                logger.info(f"任务device字段已更新到数据库: {task.task_id} -> {allocated_device}")
            except Exception as db_error:
                logger.warning(f"更新任务device到数据库失败: {str(db_error)}")
            
            # 构建训练配置字典（结构化参数传递）
            training_config = dict(task.training_params)
            
            # 添加任务核心信息到training_config
            training_config.update({
                "task_id": task.task_id,
                "train_type": task.train_type,
                "model_name_or_path": task.model_name_or_path,
                "dataset_name_or_path": task.dataset_name_or_path,
                "output_dir": task.output_dir,
                "device": allocated_device
                # 注意：training_mode 不应该加入 training_config，它是服务层控制参数
            })
            
            logger.info(f"传递训练配置参数: {list(training_config.keys())}")
            
            # 创建进程参数
            process_args = {
                'task_id': task.task_id,
                'service_instance_id': self.service_instance_id,
                'allocated_device': allocated_device,
                'training_config': training_config,
                'task_config': task.model_dump(),  # 使用 model_dump() 而不是 dict()
                'training_mode': mode
            }
            
            # 启动multiprocessing.Process
            process = mp.Process(
                target=UnifiedTrainingService._run_training_in_process,
                args=(process_args,),
                name=f"{mode}-training-{task.task_id[:8]}"
            )
            
            process.start()
            
            # 记录进程信息到数据库
            try:
                from ..enums import ProcessStatus
                success = training_task_service.update_process_info(
                    task.task_id, 
                    process_pid=process.pid,
                    process_status=ProcessStatus.RUNNING.value,
                    service_instance_id=self.service_instance_id
                )
                if success:
                    logger.info(f"✅ 记录进程信息到数据库成功: 任务={task.task_id}, PID={process.pid}, 状态=RUNNING, 模式={mode}")
                else:
                    logger.error(f"❌ 记录进程信息到数据库失败: update_process_info返回False")
            except Exception as e:
                logger.error(f"❌ 记录进程信息异常: {str(e)}", exc_info=True)
            
            # 🔧 同步进程PID到任务管理器的内存任务对象
            try:
                task.process_pid = process.pid
                task_manager._save_tasks()  # 保存到本地文件
                logger.info(f"✅ 已同步PID到任务管理器: {task.task_id} -> PID={process.pid}")
            except Exception as e:
                logger.warning(f"同步PID到任务管理器失败: {str(e)}")
            
            # 保存进程引用
            with self._lock:
                self.training_processes[task.task_id] = process
                self.process_info[task.task_id] = {
                    'pid': process.pid,
                    'started_at': datetime.now(),
                    'status': ProcessStatus.RUNNING.value,
                    'mode': mode,
                    'service_instance_id': self.service_instance_id
                }
            
            # 进程已启动成功，保持PENDING状态，等待真正开始训练时再更新为RUNNING
            try:
                # 不调用start_task，让任务保持PENDING状态，等待子进程真正开始训练时才更新为RUNNING
                training_task_service.update_task_status(task.task_id, TrainingStatus.PENDING.value)
                logger.info(f"✅ 任务状态更新成功: {task.task_id} (PENDING - 等待训练开始)")
            except Exception as status_error:
                logger.warning(f"任务状态更新失败（但进程已启动）: {str(status_error)}")
            
            logger.info(f"✅ 训练任务已启动: {task.task_id}, 模式: {mode}, PID: {process.pid}")
            return task
            
        except Exception as e:
            error_msg = f"启动训练进程失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # 只有在进程启动前失败时才标记任务失败
            try:
                task_manager.fail_task(task.task_id, error_msg, traceback.format_exc())
            except Exception as db_error:
                logger.warning(f"更新任务失败状态失败: {str(db_error)}")
            
            # 释放资源
            if mode == "parallel":
                gpu_resource_manager.release_gpus_for_task(task.task_id)
            
            raise RuntimeError(error_msg)
    
    @staticmethod
    def _run_training_in_process(process_args: dict):
        """
        在子进程中执行训练的实际逻辑
        
        Args:
            process_args: 进程参数字典
        """
        try:
            # 提取参数
            task_id = process_args['task_id']
            service_instance_id = process_args['service_instance_id']
            allocated_device = process_args['allocated_device']
            training_config = process_args['training_config']
            task_config = process_args['task_config']
            training_mode = process_args['training_mode']
            
            logger.info(f"🚀 开始执行训练任务: {task_id} (模式: {training_mode})")
            logger.info(f"📦 服务实例ID: {service_instance_id}")
            logger.info(f"🔧 分配设备: {allocated_device}")
            
            # 设置设备环境变量（子进程环境隔离）
            if allocated_device and allocated_device != "auto":
                os.environ["CUDA_VISIBLE_DEVICES"] = allocated_device.replace("cuda:", "")
                logger.info(f"🔧 设置CUDA设备: {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            # 创建进度回调函数
            def progress_callback(current_step: int, total_steps: int, stage: str = "training"):
                """进度回调函数"""
                try:
                    # 检测异常的total_steps值（保留作为安全检查）
                    if total_steps <= 0:
                        logger.warning(f"🚨 检测到异常的total_steps: {total_steps}, 跳过进度更新")
                        return

                    # 计算进度百分比，但防止过早设置100%
                    if total_steps > 0:
                        raw_progress = (current_step / total_steps) * 100.0

                        # 🔧 防止在训练完成前设置100%进度：将99.5%以上的进度限制为99.5%
                        # 只有当训练真正完成时，才会在completion handler中设置100%
                        progress = min(raw_progress, 99.5) if stage.lower() in ["training", "训练中"] else raw_progress
                    else:
                        progress = 0.0
                    
                    message = f"{stage}: {current_step}/{total_steps}"
                    
                    # 更新任务管理器
                    from ..services.task_manager import task_manager
                    task_manager.update_task_progress(task_id, progress, message)
                    
                    # 🔧 避免直接更新数据库，使用task_manager的1%节流机制
                    # 这样可以避免频繁的数据库更新，并且进度更新逻辑更统一
                    
                    logger.info(f"训练进度 {progress:.1f}%: {message}")
                except Exception as e:
                    logger.warning(f"更新进度失败: {e}")
            
            # 导入重构后的训练函数
            from ..train import main

            # 执行训练（状态更新将在训练循环真正开始时进行）
            model, save_dir = main(
                progress_callback=progress_callback,
                training_config=training_config
            )
            
            logger.info(f"✅ 训练完成: {task_id}")
            logger.info(f"📁 模型保存路径: {save_dir}")
            
            # 更新任务状态
            from ..services.task_manager import task_manager
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            
            # 🔧 先更新任务管理器（这会设置状态为SUCCEEDED和进度100%）
            task_manager.complete_task(task_id, save_dir)
            
            # 🔧 再更新数据库（此时task_manager.update_task_progress会允许100%进度写入）
            task_manager.update_task_progress(task_id, 100.0, "训练完成")
            
            # 🔧 更新数据库任务状态为SUCCEEDED
            from bubble_rag.training.model_sft.enums.training_task_enums import TrainingStatus
            training_task_service.update_task_status(task_id, TrainingStatus.SUCCEEDED.value)
            training_task_service.update_task_result(task_id, final_model_path=save_dir)
            logger.info(f"✅ 数据库任务状态已更新为SUCCEEDED: {task_id}")
            
            # 🔧 训练成功完成后释放GPU资源
            try:
                from ..utils.gpu_resource_manager import gpu_resource_manager
                success = gpu_resource_manager.release_gpus_for_task(task_id)
                if success:
                    logger.info(f"✅ 训练成功完成，已释放任务 {task_id} 的GPU资源")
                else:
                    logger.warning(f"常规GPU释放失败，尝试强制释放")
                    gpu_resource_manager.force_release_gpu_for_task(task_id)

                # 增强GPU清理（静态方法中不能使用self，直接调用GPU资源管理器）
                logger.info("GPU资源已通过gpu_resource_manager释放")
            except Exception as gpu_error:
                logger.critical(f"❌ 严重错误：训练完成后GPU资源释放失败！尝试强制恢复。任务: {task_id}, 错误: {gpu_error}")
                try:
                    gpu_resource_manager.force_release_gpu_for_task(task_id)
                    logger.warning(f"🔧 强制GPU释放已执行")
                except Exception as force_error:
                    logger.critical(f"❌ 强制GPU释放也失败！任务 {task_id} 资源可能永久泄漏: {force_error}")
            
        except Exception as e:
            error_msg = f"训练执行失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # 更新失败状态
            try:
                from ..services.task_manager import task_manager
                from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                from bubble_rag.training.model_sft.enums import TrainingStatus

                task_manager.fail_task(task_id, error_msg, traceback.format_exc())
                training_task_service.update_task_status(task_id, TrainingStatus.FAILED.value)
                training_task_service.update_task_result(task_id, error_message=error_msg)

                # 🔧 关键修复：训练失败时也要释放GPU资源
                try:
                    from ..utils.gpu_resource_manager import gpu_resource_manager
                    success = gpu_resource_manager.release_gpus_for_task(task_id)
                    if success:
                        logger.info(f"🔓 训练失败，已释放任务 {task_id} 的GPU资源")
                    else:
                        logger.warning(f"常规GPU释放失败，尝试强制释放")
                        gpu_resource_manager.force_release_gpu_for_task(task_id)
                except Exception as gpu_error:
                    logger.error(f"❌ 训练失败时释放GPU资源失败: {gpu_error}")
                    try:
                        gpu_resource_manager.force_release_gpu_for_task(task_id)
                        logger.warning(f"🔧 强制GPU释放已执行")
                    except Exception as force_error:
                        logger.critical(f"❌ 强制GPU释放也失败！任务 {task_id} 资源可能永久泄漏: {force_error}")

            except Exception as update_error:
                logger.error(f"更新失败状态时出错: {update_error}")
            
            # 抛出异常以设置进程退出码
            raise
        finally:
            # 无论什么模式都释放GPU资源（作为最终保险）
            try:
                from ..utils.gpu_resource_manager import gpu_resource_manager
                success = gpu_resource_manager.release_gpus_for_task(task_id)
                if success:
                    logger.info(f"🔧 Finally块：确保释放任务 {task_id} 的GPU资源")
                else:
                    logger.warning(f"Finally块：常规GPU释放失败，尝试强制释放")
                    gpu_resource_manager.force_release_gpu_for_task(task_id)

                # 增强GPU清理（最终保险）- 静态方法中不能使用self
                logger.info("Finally块：GPU资源已通过gpu_resource_manager释放")
            except Exception as e:
                logger.critical(f"❌ 严重错误：Finally块GPU资源释放失败！尝试强制恢复。任务: {task_id}, 错误: {e}")
                try:
                    gpu_resource_manager.force_release_gpu_for_task(task_id)
                    logger.warning(f"🔧 强制GPU释放已执行")
                except Exception as force_error:
                    logger.critical(f"❌ 强制GPU释放也失败！任务 {task_id} 资源可能永久泄漏: {force_error}")
    
    def stop_training(self, task_id: str) -> bool:
        """
        停止指定的训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功停止
        """
        try:
            with self._lock:
                process = self.training_processes.get(task_id)
                if not process:
                    logger.warning(f"未找到任务 {task_id} 的进程")
                    # 🔧 增强停止机制：通过数据库查询PID并尝试停止
                    return self._stop_training_by_database(task_id)
                
                if not process.is_alive():
                    logger.info(f"任务 {task_id} 进程已结束，仍需清理GPU资源")
                    self.training_processes.pop(task_id, None)

                    # 🔧 即使进程已结束，也要清理GPU资源
                    task_manager.cancel_task(task_id)
                    training_task_service.update_task_status(task_id, TrainingStatus.STOPPED.value)

                    # 清理GPU资源
                    success = self._enhanced_gpu_cleanup(task_id)
                    if not success:
                        logger.warning(f"进程已结束但GPU清理失败: {task_id}")

                    return True
                
                # 🌳 使用统一的进程树终止方法
                pid = process.pid
                logger.info(f"🌳 使用统一方法终止训练进程树 (任务: {task_id}, PID: {pid})")
                
                # 先使用统一的进程树清理方法
                success = self._terminate_process_tree_by_pid(pid)
                
                if not success:
                    # 如果统一方法失败，回退到原来的multiprocessing方式
                    logger.warning(f"统一方法失败，回退到multiprocessing终止方式")
                    try:
                        process.terminate()
                        process.join(timeout=30)
                        logger.info(f"✅ 训练进程已通过multiprocessing终止")
                    except:
                        logger.warning(f"进程终止超时，使用强制终止")
                    process.kill()
                    process.join()
                    logger.info(f"💀 训练进程已强制终止")
                
                # 清理进程引用
                self.training_processes.pop(task_id, None)
                if task_id in self.process_info:
                    from ..enums.training_task_enums import ProcessStatus
                    self.process_info[task_id]['status'] = ProcessStatus.STOPPED.value
                
                # 更新任务状态
                task_manager.cancel_task(task_id)
                training_task_service.update_task_status(task_id, TrainingStatus.STOPPED.value)
                
                # 🔧 更新数据库中的进程状态
                try:
                    from ..enums.training_task_enums import ProcessStatus
                    # 获取当前任务的PID信息
                    current_task = task_manager.get_task(task_id)
                    current_pid = current_task.process_pid if current_task else None
                    
                    training_task_service.update_process_info(
                        task_id=task_id,
                        process_pid=current_pid,  # 🔧 保留PID用于审计追踪
                        process_status=ProcessStatus.STOPPED.value
                    )
                    logger.info(f"✅ 已更新进程状态为STOPPED: {task_id} (保留PID: {current_pid})")
                except Exception as update_error:
                    logger.warning(f"更新进程状态失败: {update_error}")
                
                # 🔧 增强GPU资源清理
                success = self._enhanced_gpu_cleanup(task_id)
                if not success:
                    logger.warning(f"增强GPU清理失败，建议检查GPU状态")

                logger.info(f"✅ 已停止训练任务: {task_id}")
                return True
                
        except Exception as e:
            logger.error(f"停止训练任务失败: {str(e)}", exc_info=True)
            return False
    
    def _stop_training_by_database(self, task_id: str) -> bool:
        """
        通过数据库中的PID停止训练任务（回退机制）
        当内存中的进程引用丢失时使用
        """
        import psutil
        
        try:
            # 从数据库获取任务信息
            task_db = training_task_service.get_training_task(task_id)
            if not task_db:
                logger.error(f"数据库中未找到任务: {task_id}")
                return False
            
            # 检查是否有PID记录
            if not task_db.process_pid:
                logger.warning(f"任务 {task_id} 没有PID记录，直接清理状态和GPU资源")
                # 直接更新状态为停止
                task_manager.cancel_task(task_id)
                training_task_service.update_task_status(task_id, TrainingStatus.STOPPED.value)
                # 🔧 更新进程状态
                from ..enums.training_task_enums import ProcessStatus
                training_task_service.update_process_info(task_id, None, ProcessStatus.STOPPED.value)

                # 🔧 清理GPU资源
                success = self._enhanced_gpu_cleanup(task_id)
                if not success:
                    logger.warning(f"无PID任务GPU清理失败: {task_id}")

                return True
            
            pid = task_db.process_pid
            logger.info(f"🔧 尝试通过PID {pid} 停止训练任务 {task_id}")
            
            # 检查进程是否存在
            if not psutil.pid_exists(pid):
                logger.info(f"进程 {pid} 已不存在，更新任务状态并清理GPU资源")
                task_manager.cancel_task(task_id)
                training_task_service.update_task_status(task_id, TrainingStatus.STOPPED.value)
                # 🔧 更新进程状态
                from ..enums.training_task_enums import ProcessStatus
                training_task_service.update_process_info(task_id, None, ProcessStatus.STOPPED.value)

                # 🔧 清理GPU资源
                success = self._enhanced_gpu_cleanup(task_id)
                if not success:
                    logger.warning(f"已结束进程GPU清理失败: {task_id}")

                return True
            
            # 获取进程对象并停止
            try:
                process = psutil.Process(pid)
                process_name = process.name()
                
                # 验证这确实是我们的训练进程
                if 'python' not in process_name.lower():
                    logger.warning(f"PID {pid} 不是Python进程: {process_name}")
                    return False
                
                # 🔧 使用统一的进程树终止方法
                logger.info(f"🛑 检测到训练进程 {pid} ({process_name})")
                logger.info(f"🌳 使用统一方法终止进程树（包括所有子进程）")
                
                # 调用统一的进程树清理方法
                success = self._terminate_process_tree_by_pid(pid)
                
                # 无论进程终止成功与否，都要更新状态和清理资源
                # 更新状态
                task_manager.cancel_task(task_id)
                training_task_service.update_task_status(task_id, TrainingStatus.STOPPED.value)
                # 🔧 更新进程状态
                from ..enums.training_task_enums import ProcessStatus
                training_task_service.update_process_info(task_id, None, ProcessStatus.STOPPED.value)
                
                # 🔧 增强GPU资源清理
                gpu_success = self._enhanced_gpu_cleanup(task_id)
                if not gpu_success:
                    logger.warning(f"增强GPU清理失败，建议检查GPU状态")

                if success:
                    logger.info(f"✅ 进程树已通过统一方法成功终止: PID {pid}")
                    return True
                else:
                    logger.warning(f"⚠️ 统一方法失败，但任务状态已更新: PID {pid}")
                    return False
                
            except psutil.NoSuchProcess:
                logger.info(f"进程 {pid} 已结束，更新任务状态并清理GPU资源")
                task_manager.cancel_task(task_id)
                training_task_service.update_task_status(task_id, TrainingStatus.STOPPED.value)
                # 🔧 更新进程状态
                from ..enums.training_task_enums import ProcessStatus
                training_task_service.update_process_info(task_id, None, ProcessStatus.STOPPED.value)

                # 🔧 清理GPU资源
                success = self._enhanced_gpu_cleanup(task_id)
                if not success:
                    logger.warning(f"NoSuchProcess异常GPU清理失败: {task_id}")

                return True
            except psutil.AccessDenied:
                logger.error(f"无权限访问进程 {pid}")
                return False
                
        except Exception as e:
            logger.error(f"通过数据库PID停止训练失败: {str(e)}", exc_info=True)
            return False

    def get_running_processes(self) -> Dict[str, Dict]:
        """
        获取正在运行的训练进程
        
        Returns:
            正在运行的进程信息字典
        """
        with self._lock:
            running = {}
            for task_id, process in list(self.training_processes.items()):
                if process.is_alive():
                    running[task_id] = {
                        'pid': process.pid,
                        'process_info': self.process_info.get(task_id, {}),
                        'process': process
                    }
                else:
                    # 清理已结束的进程
                    self.training_processes.pop(task_id, None)
                    if task_id in self.process_info:
                        from ..enums.training_task_enums import ProcessStatus
                        self.process_info[task_id]['status'] = ProcessStatus.STOPPED.value

                    # 🔧 关键修复：清理进程时同时释放GPU资源
                    try:
                        from ..utils.gpu_resource_manager import gpu_resource_manager
                        if gpu_resource_manager.release_gpus_for_task(task_id):
                            logger.info(f"✅ 进程清理时释放任务 {task_id} 的GPU资源")
                        else:
                            logger.warning(f"⚠️ 进程清理时GPU释放失败，尝试强制释放: {task_id}")
                            gpu_resource_manager.force_release_gpu_for_task(task_id)
                    except Exception as gpu_e:
                        logger.error(f"❌ 进程清理时GPU资源释放失败: {task_id}, 错误: {gpu_e}")
            
            return running
    
    def get_training_status(self, task_id: str) -> Optional[Dict]:
        """
        获取训练状态（实现抽象方法）
        
        Args:
            task_id: 任务ID
            
        Returns:
            训练状态信息
        """
        try:
            # 从任务管理器获取任务信息
            task = task_manager.get_task(task_id)
            if not task:
                return None
            
            # 获取进程信息
            with self._lock:
                process = self.training_processes.get(task_id)
                process_info = self.process_info.get(task_id, {})
            
            return {
                "task_info": task.get_summary(),
                "process_alive": process.is_alive() if process else False,
                "process_info": process_info,
                "service_instance_id": self.service_instance_id
            }
            
        except Exception as e:
            logger.error(f"获取训练状态失败: {str(e)}")
            return None
    
    def cleanup_completed_processes(self):
        """清理已完成的进程并更新数据库状态"""
        with self._lock:
            completed_tasks = []
            for task_id, process in list(self.training_processes.items()):
                if not process.is_alive():
                    completed_tasks.append(task_id)
                    process.join()  # 确保进程资源被释放
                    
            for task_id in completed_tasks:
                self.training_processes.pop(task_id, None)
                if task_id in self.process_info:
                    self.process_info[task_id]['status'] = 'COMPLETED'
                
                # 🔧 进程监控检测到训练完成，清理GPU资源
                try:
                    from ..utils.gpu_resource_manager import gpu_resource_manager
                    gpu_resource_manager.release_gpus_for_task(task_id)
                    logger.info(f"🔍 进程监控：检测到任务 {task_id} 完成，已释放GPU资源")
                    
                    # 增强GPU清理
                    gpu_success = self._enhanced_gpu_cleanup(task_id)
                    if not gpu_success:
                        logger.warning(f"进程监控：增强GPU清理失败")
                except Exception as gpu_error:
                    logger.error(f"❌ 进程监控GPU清理失败，尝试强制恢复。任务: {task_id}, 错误: {gpu_error}")
                    try:
                        gpu_resource_manager.force_release_gpu_for_task(task_id)
                        logger.warning(f"🔧 强制GPU释放已执行")
                    except Exception as force_error:
                        logger.critical(f"❌ 强制GPU释放也失败！任务 {task_id} 资源可能永久泄漏: {force_error}")
                
                # 更新数据库中的进程状态
                try:
                    from ..enums import ProcessStatus
                    # 获取当前任务的PID信息
                    current_task = task_manager.get_task(task_id)
                    current_pid = current_task.process_pid if current_task else None
                    
                    training_task_service.update_process_info(
                        task_id=task_id,
                        process_pid=current_pid,  # 🔧 保留PID用于审计追踪
                        process_status=ProcessStatus.STOPPED.value
                    )
                    logger.info(f"✅ 更新数据库进程状态: 任务={task_id}, 状态=STOPPED (保留PID: {current_pid})")
                except Exception as e:
                    logger.warning(f"更新进程状态失败: 任务={task_id}, 错误={e}")
                    
            if completed_tasks:
                logger.info(f"清理了 {len(completed_tasks)} 个已完成的进程: {completed_tasks}")


    # 实现抽象基类的必需方法
    def start_training_process(self, task) -> bool:
        """启动训练进程 - 抽象方法实现"""
        try:
            # 这里直接调用内部的训练启动方法
            result_task = self._start_training_process(task, self.default_mode)
            return result_task is not None
        except Exception as e:
            logger.error(f"启动训练进程失败: {e}")
            return False

    def stop_training_process(self, task_id: str) -> bool:
        """停止训练进程 - 抽象方法实现"""
        return self.stop_training(task_id)

    def check_unknown_processes(self) -> Dict[str, Any]:
        """检查和恢复UNKNOWN状态进程 - 代理到进程管理器"""
        try:
            # 代理到ProcessManagerBase的check_unknown_processes方法
            return super().check_unknown_processes()
        except Exception as e:
            logger.error(f"检查UNKNOWN状态进程失败: {str(e)}")
            return {'error': str(e), 'total_unknown': 0, 'recovered_running': 0, 'recovered_terminated': 0, 'still_unknown': 0, 'error_count': 1}

    def delete_task(self, task_id: str) -> tuple[bool, str]:
        """
        删除训练任务
        
        功能：
        1. 如果任务正在运行，先停止并杀死进程
        2. 更新任务和进程状态
        3. 从内存和数据库中删除任务记录
        
        返回：
        - (success: bool, message: str)
        """
        try:
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            from bubble_rag.training.model_sft.services.task_manager import task_manager
            
            logger.info(f"🗑️ 开始删除任务: {task_id}")
            
            # 1. 检查任务是否存在
            task_db = training_task_service.get_training_task(task_id)
            if not task_db:
                return False, f"任务不存在: {task_id}"
            
            # 2. 检查任务是否正在运行，如果是则先停止
            memory_task = task_manager.get_task(task_id)
            is_running = task_id in self.processes
            
            if is_running or (memory_task and memory_task.status == "RUNNING"):
                logger.info(f"🛑 任务正在运行，先停止: {task_id}")
                
                # 停止训练任务和进程
                stop_success = self.stop_training(task_id)
                if not stop_success:
                    logger.warning(f"停止任务失败，但继续删除流程: {task_id}")
                
                # 强制杀死进程（如果还在运行）
                if task_id in self.processes:
                    try:
                        process_info = self.processes[task_id]
                        pid = process_info.get("pid")
                        if pid:
                            import psutil
                            try:
                                process = psutil.Process(pid)
                                # 🌳 使用统一的进程树终止方法（删除任务时也要彻底清理）
                                logger.info(f"🔫 删除任务：使用进程树清理方法 PID={pid}")
                                success = self._terminate_process_tree_by_pid(pid)
                                if success:
                                    logger.info(f"✅ 进程树清理成功: PID={pid}")
                                else:
                                    logger.warning(f"⚠️ 进程树清理失败，但继续删除流程: PID={pid}")

                                # 🔧 关键修复：删除任务时也要确保GPU资源释放
                                logger.info(f"🔧 删除任务：强制清理GPU资源 {task_id}")
                                gpu_success = self._enhanced_gpu_cleanup(task_id)
                                if not gpu_success:
                                    logger.warning(f"删除任务时GPU清理失败，建议检查GPU状态")

                            except psutil.NoSuchProcess:
                                logger.info(f"进程已不存在: PID={pid}")
                                # 进程不存在时也要清理GPU资源
                                logger.info(f"🔧 进程已不存在，清理GPU资源 {task_id}")
                                self._enhanced_gpu_cleanup(task_id)
                            except Exception as e:
                                logger.warning(f"杀死进程失败: PID={pid}, 错误={e}")
                                # 进程清理失败时也要尝试清理GPU资源
                                logger.info(f"🔧 进程清理失败，强制清理GPU资源 {task_id}")
                                self._enhanced_gpu_cleanup(task_id)
                    except Exception as e:
                        logger.warning(f"处理运行进程失败: {e}")
                        # 异常情况下也要尝试清理GPU资源
                        logger.info(f"🔧 异常情况，强制清理GPU资源 {task_id}")
                        try:
                            self._enhanced_gpu_cleanup(task_id)
                        except Exception as gpu_error:
                            logger.error(f"异常情况下GPU清理失败: {gpu_error}")

                    # 从运行进程列表中移除
                    self.processes.pop(task_id, None)

            # 🔧 额外保险：通过进程名查找并清理可能遗漏的进程
            logger.info(f"🔧 删除任务：通过进程名检查遗漏的训练进程 {task_id}")
            try:
                self._cleanup_processes_by_name(task_id)
            except Exception as cleanup_error:
                logger.warning(f"通过进程名清理失败: {cleanup_error}")

            # 🔧 额外保险：无论前面的清理是否成功，都再次尝试GPU清理
            logger.info(f"🔧 删除任务最终保险：确保GPU资源清理 {task_id}")
            try:
                final_gpu_success = self._enhanced_gpu_cleanup(task_id)
                if final_gpu_success:
                    logger.info(f"✅ 删除任务GPU清理最终确认成功: {task_id}")
                else:
                    logger.warning(f"⚠️ 删除任务GPU清理最终确认失败: {task_id}")
            except Exception as final_gpu_error:
                logger.error(f"删除任务GPU清理最终确认异常: {final_gpu_error}")
            
            # 3. 从内存中删除任务
            if memory_task:
                logger.info(f"🧠 从内存删除任务: {task_id}")
                task_manager.delete_task(task_id)
            
            # 4. 更新数据库中的任务和进程状态为已终止
            logger.info(f"📊 更新数据库状态: {task_id}")
            
            # 导入ProcessStatus枚举
            from ..enums.training_task_enums import ProcessStatus
            from ..enums import TrainingStatus
            
            # 更新任务状态和结果
            training_task_service.update_task_status(
                task_id=task_id,
                status=TrainingStatus.STOPPED.value
            )
            training_task_service.update_task_result(
                task_id=task_id,
                error_message="任务已删除"
            )
            
            # 更新进程状态
            training_task_service.update_process_info(
                task_id=task_id,
                process_pid=None,  # 任务删除时清空PID
                process_status=ProcessStatus.TERMINATED.value
            )
            
            # 5. 删除关联的数据集记录
            logger.info(f"🗂️ 删除关联数据集: {task_id}")
            dataset_deleted_count = 0
            dataset_message = ""
            try:
                from bubble_rag.training.mysql_service.service.training_dataset_service import training_dataset_service
                dataset_deleted_count, dataset_message = training_dataset_service.delete_datasets_by_task(task_id)
                logger.info(f"📊 数据集删除结果: {dataset_message}")
            except Exception as e:
                logger.warning(f"删除数据集失败，但继续删除任务: {e}")
                dataset_message = f"数据集删除失败: {str(e)}"

            # 6. 从数据库中删除任务记录
            logger.info(f"🗄️ 从数据库删除任务记录: {task_id}")
            db_success = training_task_service.delete_training_task(task_id)

            if db_success:
                dataset_info = f"，同时删除了 {dataset_deleted_count} 个数据集记录" if dataset_deleted_count > 0 else ""
                logger.info(f"✅ 任务删除成功: {task_id}{dataset_info}")
                return True, f"任务 {task_id} 已成功删除{dataset_info}"
            else:
                logger.error(f"❌ 数据库删除失败: {task_id}")
                return False, "从数据库删除任务失败"
                
        except Exception as e:
            logger.error(f"删除任务失败: {task_id}, 错误: {str(e)}", exc_info=True)
            return False, f"删除任务时发生异常: {str(e)}"

    def _terminate_process_tree_by_pid(self, pid: int) -> bool:
        """统一的进程树终止方法 - 适用于所有场景（手动停止、服务重启等）"""
        try:
            import psutil
            logger.info(f"🌳 开始终止进程树 PID={pid}")
            
            # 获取主进程
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                logger.info(f"进程 {pid} 已不存在")
                return True
            
            # 1. 获取所有子进程
            children = []
            try:
                children = process.children(recursive=True)
                logger.info(f"🔍 发现 {len(children)} 个子进程需要终止")
                for child in children:
                    try:
                        logger.info(f"   子进程: PID {child.pid}, 名称: {child.name()}")
                    except:
                        logger.info(f"   子进程: PID {child.pid}, 名称: 获取失败")
            except psutil.NoSuchProcess:
                logger.warning(f"主进程 {pid} 已不存在")
                return True
            except Exception as e:
                logger.warning(f"获取子进程失败: {e}")
            
            # 2. 先终止所有子进程（dataloader workers等）
            for child in children:
                try:
                    child.terminate()
                    logger.info(f"🔥 已终止子进程: PID {child.pid}")
                except psutil.NoSuchProcess:
                    logger.info(f"子进程 {child.pid} 已不存在")
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
            logger.error(f"终止进程树失败 PID={pid}: {e}")
            return False

    def _enhanced_gpu_cleanup(self, task_id: str) -> bool:
        """
        增强GPU资源清理机制
        不仅释放GPU资源管理器中的分配，还清理可能残留的GPU内存

        Returns:
            bool: True if successful, False if failed
        """
        success = True
        try:
            # 1. 释放GPU资源管理器分配
            release_success = gpu_resource_manager.release_gpus_for_task(task_id)
            if release_success:
                logger.info(f"🔧 已释放GPU资源管理器中的任务 {task_id} 资源")
            else:
                logger.warning(f"GPU资源管理器释放失败，尝试强制释放")
                success = gpu_resource_manager.force_release_gpu_for_task(task_id)
        except Exception as e:
            logger.error(f"❌ GPU资源管理器资源释放失败，尝试强制恢复。错误: {e}")
            try:
                success = gpu_resource_manager.force_release_gpu_for_task(task_id)
                logger.warning(f"🔧 强制GPU释放已执行")
            except Exception as force_error:
                logger.critical(f"❌ 强制GPU释放也失败！任务 {task_id} 资源可能永久泄漏: {force_error}")
                success = False
        
        try:
            # 2. 强制清理CUDA内存（如果可用）
            import torch
            if torch.cuda.is_available():
                # 清空CUDA缓存
                torch.cuda.empty_cache()
                logger.info(f"🧹 已清空CUDA内存缓存")
                
                # 获取GPU内存使用情况
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_cached = torch.cuda.memory_reserved(i)
                    if memory_allocated > 0 or memory_cached > 0:
                        logger.info(f"GPU {i}: 已分配={memory_allocated/1024/1024:.1f}MB, 缓存={memory_cached/1024/1024:.1f}MB")
                        # 尝试进一步清理
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except:
                            pass
        except ImportError:
            # PyTorch不可用，跳过CUDA清理
            pass
        except Exception as e:
            logger.warning(f"清理CUDA内存失败: {e}")
        
        try:
            # 3. 系统级GPU进程清理（可选，用于极端情况）
            import subprocess
            import platform
            
            # 尝试查找可能的残留GPU进程
            if platform.system() == "Linux":
                try:
                    # 使用nvidia-smi查看GPU进程
                    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout:
                        gpu_processes = result.stdout.strip().split('\n')
                        logger.info(f"🔍 当前GPU进程: {len(gpu_processes)}个")
                        for proc in gpu_processes:
                            if proc.strip():
                                logger.info(f"   GPU进程: {proc}")
                except subprocess.TimeoutExpired:
                    logger.warning("nvidia-smi查询超时")
                except Exception as e:
                    logger.debug(f"nvidia-smi查询失败: {e}")
        except Exception as e:
            logger.debug(f"系统级GPU检查失败: {e}")
        
        logger.info(f"✅ GPU资源清理完成: 任务 {task_id}")
        return success

    def _cleanup_processes_by_name(self, task_id: str) -> bool:
        """
        通过进程名和命令行查找并清理可能遗漏的训练进程
        这是一个补充清理机制，用于捕获PID-based清理可能遗漏的进程

        Args:
            task_id: 任务ID

        Returns:
            bool: True if successful cleanup, False if issues found
        """
        try:
            import psutil
            import re
            import os

            logger.info(f"🔍 开始按进程名清理遗漏的训练进程: {task_id}")

            # 定义可能的训练进程名称模式
            training_patterns = [
                r'python.*train.*',
                r'.*accelerate.*',
                r'.*torch.*distributed.*',
                r'.*deepspeed.*',
                r'.*transformers.*',
                r'.*train_model.*',
                r'.*sft_training.*'
            ]

            # 搜索包含task_id的进程
            task_id_patterns = [
                task_id,  # 直接匹配task_id
                f'task_id.*{task_id}',  # 命令行参数包含task_id
                f'{task_id}.*train',  # task_id在训练命令中
            ]

            found_processes = []
            terminated_count = 0

            # 遍历所有进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    proc_info = proc.info
                    pid = proc_info['pid']
                    name = proc_info['name'] or ''
                    cmdline = ' '.join(proc_info['cmdline'] or [])

                    # 跳过系统进程和当前Python进程
                    if pid <= 1 or pid == os.getpid():
                        continue

                    # 检查是否是训练相关进程
                    is_training_process = False
                    for pattern in training_patterns:
                        if re.search(pattern, name, re.IGNORECASE) or re.search(pattern, cmdline, re.IGNORECASE):
                            is_training_process = True
                            break

                    # 检查是否包含task_id
                    contains_task_id = False
                    for pattern in task_id_patterns:
                        if re.search(pattern, cmdline, re.IGNORECASE):
                            contains_task_id = True
                            break

                    # 如果是训练进程且包含task_id，则需要清理
                    if is_training_process and contains_task_id:
                        found_processes.append({
                            'pid': pid,
                            'name': name,
                            'cmdline': cmdline[:200],  # 限制长度
                            'create_time': proc_info['create_time']
                        })

                        logger.warning(f"🎯 发现遗漏的训练进程: PID={pid}, 名称={name}")
                        logger.info(f"   命令行: {cmdline[:100]}...")

                        # 尝试终止这个进程
                        try:
                            process = psutil.Process(pid)
                            process.terminate()
                            logger.info(f"🔥 已终止遗漏进程: PID={pid}")

                            # 等待进程退出
                            try:
                                process.wait(timeout=3)
                                logger.info(f"✅ 遗漏进程已优雅终止: PID={pid}")
                                terminated_count += 1
                            except psutil.TimeoutExpired:
                                # 强制杀死
                                try:
                                    process.kill()
                                    logger.warning(f"💀 强制终止遗漏进程: PID={pid}")
                                    terminated_count += 1
                                except:
                                    logger.error(f"强制终止遗漏进程失败: PID={pid}")

                        except psutil.NoSuchProcess:
                            logger.info(f"遗漏进程已不存在: PID={pid}")
                            terminated_count += 1
                        except Exception as e:
                            logger.error(f"终止遗漏进程失败: PID={pid}, 错误={e}")

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # 进程可能已经消失或无权限访问，跳过
                    continue
                except Exception as e:
                    logger.debug(f"检查进程失败: {e}")
                    continue

            if found_processes:
                logger.warning(f"⚠️ 发现 {len(found_processes)} 个遗漏的训练进程，已终止 {terminated_count} 个")
                return terminated_count == len(found_processes)
            else:
                logger.info(f"✅ 未发现遗漏的训练进程: {task_id}")
                return True

        except Exception as e:
            logger.error(f"按进程名清理失败: {task_id}, 错误: {e}")
            return False

# 全局统一训练服务实例
unified_training_service = UnifiedTrainingService()