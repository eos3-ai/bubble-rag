"""
GPU资源动态分配管理器
基于现有的device参数架构，实现多任务GPU动态分配

支持用户权限控制：
- 技术隔离：继续使用服务实例ID进行孤儿进程检测
- 业务隔离：增加用户权限控制，管理员可全局管理GPU资源
"""

import os
import threading
import logging
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """GPU资源动态分配管理器"""
    
    def __init__(self):
        # 核心分配数据（保持原有结构）
        self.gpu_allocations: Dict[int, str] = {}  # {gpu_id: task_id}
        self.task_gpus: Dict[str, Set[int]] = {}   # {task_id: {gpu_ids}}
        self.allocation_times: Dict[str, datetime] = {}  # {task_id: allocation_time}

        # 新增：用户信息追踪
        self.task_users: Dict[str, str] = {}  # {task_id: username}
        self.task_user_roles: Dict[str, str] = {}  # {task_id: user_role}

        self._lock = threading.RLock()
        self.max_gpus = self._detect_max_gpus()

        logger.info(f"GPU资源管理器初始化完成，检测到 {self.max_gpus} 个GPU")

        # 延迟GPU清理，避免循环导入问题
        self._schedule_delayed_cleanup()
    
    def _detect_max_gpus(self) -> int:
        """检测系统GPU数量"""
        try:
            # 方法1: 使用nvidia-ml-py
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            return gpu_count
        except:
            try:
                # 方法2: 使用torch
                import torch
                if torch.cuda.is_available():
                    return torch.cuda.device_count()
            except:
                pass
        
        # 方法3: 默认假设4个GPU（可配置）
        logger.warning("无法检测GPU数量，使用默认值4")
        return int(os.environ.get('MAX_GPUS', '4'))
    
    def allocate_gpus_for_task(self, task_id: str, device_request: str = "auto", username: str = None, user_role: str = "user") -> Optional[str]:
        """
        为任务分配GPU资源

        Args:
            task_id: 任务ID
            device_request: 设备请求，支持:
                - "auto": 自动分配1个GPU
                - "cpu": 使用CPU
                - "cuda:0": 指定GPU 0
                - "cuda:0,cuda:1": 指定多个GPU
                - "auto:2": 自动分配2个GPU
            username: 用户名（用于权限控制和资源追踪）
            user_role: 用户角色（admin/user）

        Returns:
            设备字符串，如 "cuda:0,cuda:1" 或 "cpu" 或 None（分配失败）
        """
        with self._lock:
            try:
                # CPU模式
                if device_request == "cpu":
                    # 记录用户信息（即使是CPU模式也要记录）
                    if username:
                        self.task_users[task_id] = username
                        self.task_user_roles[task_id] = user_role
                    logger.info(f"任务 {task_id} (用户: {username}) 使用CPU模式")
                    return "cpu"
                
                # 解析设备请求
                requested_gpus, num_gpus = self._parse_device_request(device_request)
                
                if requested_gpus is not None:
                    # 指定GPU模式
                    if self._can_allocate_specific_gpus(requested_gpus):
                        return self._do_allocate_gpus(task_id, requested_gpus, username, user_role)
                    else:
                        logger.error(f"指定的GPU {requested_gpus} 不可用")
                        return None
                else:
                    # 自动分配模式
                    available_gpus = self._get_available_gpus(num_gpus)
                    if available_gpus:
                        return self._do_allocate_gpus(task_id, available_gpus, username, user_role)
                    else:
                        logger.error(f"无法自动分配 {num_gpus} 个GPU，当前可用GPU不足")
                        return None
                        
            except Exception as e:
                logger.error(f"GPU分配失败: {e}")
                return None
    
    def _parse_device_request(self, device_request: str) -> tuple[Optional[List[int]], int]:
        """
        解析设备请求
        
        Returns:
            (指定的GPU列表或None, 请求的GPU数量)
        """
        if device_request == "auto":
            return None, 1
        
        if device_request.startswith("auto:"):
            try:
                num_gpus = int(device_request.split(":")[1])
                return None, num_gpus
            except (IndexError, ValueError):
                logger.warning(f"无法解析auto请求: {device_request}，使用默认1个GPU")
                return None, 1
        
        if device_request.startswith("cuda:"):
            try:
                gpu_ids = []
                for part in device_request.split(","):
                    part = part.strip()
                    if part.startswith("cuda:"):
                        gpu_id = int(part.split(":")[1])
                        if 0 <= gpu_id < self.max_gpus:
                            gpu_ids.append(gpu_id)
                        else:
                            logger.warning(f"GPU ID {gpu_id} 超出范围 [0, {self.max_gpus-1}]")
                
                return gpu_ids if gpu_ids else None, len(gpu_ids)
                
            except (IndexError, ValueError) as e:
                logger.warning(f"无法解析CUDA请求: {device_request}, 错误: {e}")
                return None, 1
        
        logger.warning(f"未识别的设备请求: {device_request}，使用auto模式")
        return None, 1
    
    def _can_allocate_specific_gpus(self, gpu_ids: List[int]) -> bool:
        """检查指定的GPU是否可以分配"""
        # 首先清理失败任务的GPU分配
        self._cleanup_failed_task_allocations()

        for gpu_id in gpu_ids:
            if gpu_id >= self.max_gpus:
                logger.warning(f"GPU {gpu_id} 超出系统GPU数量 ({self.max_gpus})")
                return False
            if gpu_id in self.gpu_allocations:
                allocated_task = self.gpu_allocations[gpu_id]
                logger.warning(f"GPU {gpu_id} 已被任务 {allocated_task} 占用")
                return False
        return True
    
    def _get_available_gpus(self, num_needed: int) -> Optional[List[int]]:
        """获取可用的GPU"""
        # 首先清理失败任务的分配
        self._cleanup_failed_task_allocations()

        available = []
        for gpu_id in range(self.max_gpus):
            if gpu_id not in self.gpu_allocations:
                available.append(gpu_id)
                if len(available) >= num_needed:
                    break

        logger.info(f"请求 {num_needed} 个GPU，找到 {len(available)} 个可用GPU: {available}")
        return available[:num_needed] if len(available) >= num_needed else None
    
    def _do_allocate_gpus(self, task_id: str, gpu_ids: List[int], username: str = None, user_role: str = "user") -> str:
        """执行GPU分配"""
        # 分配GPU
        for gpu_id in gpu_ids:
            self.gpu_allocations[gpu_id] = task_id

        # 记录任务GPU映射
        self.task_gpus[task_id] = set(gpu_ids)
        self.allocation_times[task_id] = datetime.now()

        # 记录用户信息
        if username:
            self.task_users[task_id] = username
            self.task_user_roles[task_id] = user_role

        # 生成CUDA设备字符串
        if len(gpu_ids) == 1:
            device_str = f"cuda:{gpu_ids[0]}"
        else:
            device_str = ",".join([f"cuda:{gpu_id}" for gpu_id in gpu_ids])

        user_info = f" (用户: {username}, 角色: {user_role})" if username else ""
        logger.info(f"为任务 {task_id}{user_info} 分配GPU: {device_str} (物理GPU: {gpu_ids})")
        return device_str
    
    def release_gpus_for_task(self, task_id: str) -> bool:
        """释放任务的GPU资源"""
        with self._lock:
            try:
                if task_id not in self.task_gpus:
                    logger.debug(f"任务 {task_id} 没有分配GPU资源")
                    return True

                # 获取任务的GPU
                gpu_ids = self.task_gpus[task_id]

                # 释放GPU
                for gpu_id in gpu_ids:
                    if gpu_id in self.gpu_allocations:
                        del self.gpu_allocations[gpu_id]

                # 清理记录
                del self.task_gpus[task_id]
                if task_id in self.allocation_times:
                    del self.allocation_times[task_id]

                # 清理用户信息
                username = self.task_users.pop(task_id, None)
                user_role = self.task_user_roles.pop(task_id, None)

                user_info = f" (用户: {username})" if username else ""
                logger.info(f"🔓 释放任务 {task_id}{user_info} 的GPU资源: {list(gpu_ids)}")
                return True

            except Exception as e:
                logger.error(f"释放GPU资源失败: {e}")
                # 强制清理：即使出错也要尝试清理记录，避免永久泄漏
                self._force_cleanup_task_records(task_id)
                return False

    def _force_cleanup_task_records(self, task_id: str):
        """强制清理任务记录，避免永久资源泄漏"""
        try:
            logger.warning(f"强制清理任务 {task_id} 的GPU记录")

            # 强制删除任务GPU映射
            if task_id in self.task_gpus:
                gpu_ids = self.task_gpus[task_id]
                logger.info(f"强制释放GPU: {list(gpu_ids)}")

                # 强制清理GPU分配记录
                for gpu_id in gpu_ids:
                    if gpu_id in self.gpu_allocations:
                        del self.gpu_allocations[gpu_id]
                        logger.info(f"强制清理GPU {gpu_id} 分配记录")

                # 强制清理任务记录
                del self.task_gpus[task_id]

            # 强制清理分配时间记录
            if task_id in self.allocation_times:
                del self.allocation_times[task_id]

            # 强制清理用户信息记录
            username = self.task_users.pop(task_id, None)
            user_role = self.task_user_roles.pop(task_id, None)
            user_info = f" (用户: {username})" if username else ""

            logger.info(f"强制清理任务 {task_id}{user_info} GPU记录完成")

        except Exception as force_error:
            logger.critical(f"强制清理也失败！任务 {task_id} 的GPU资源可能永久泄漏: {force_error}")

    def force_release_gpu_for_task(self, task_id: str) -> bool:
        """强制释放指定任务的GPU资源（用于异常情况下的资源恢复）"""
        with self._lock:
            logger.warning(f"🚨 强制释放任务 {task_id} 的GPU资源")

            try:
                # 无条件清理所有相关记录
                gpu_ids = []
                if task_id in self.task_gpus:
                    gpu_ids = list(self.task_gpus[task_id])

                # 强制清理GPU分配
                for gpu_id in list(self.gpu_allocations.keys()):
                    if self.gpu_allocations[gpu_id] == task_id:
                        del self.gpu_allocations[gpu_id]
                        logger.info(f"强制释放GPU {gpu_id}")

                # 强制清理任务记录
                if task_id in self.task_gpus:
                    del self.task_gpus[task_id]
                if task_id in self.allocation_times:
                    del self.allocation_times[task_id]

                # 强制清理用户信息
                username = self.task_users.pop(task_id, None)
                user_role = self.task_user_roles.pop(task_id, None)
                user_info = f" (用户: {username})" if username else ""

                logger.info(f"强制释放任务 {task_id}{user_info} GPU资源完成: {gpu_ids}")
                return True

            except Exception as e:
                logger.critical(f"强制释放也失败！系统可能需要重启来恢复GPU资源: {e}")
                return False

    def _cleanup_failed_task_allocations(self):
        """清理失败任务的GPU分配"""
        failed_tasks = []

        # 检查所有已分配GPU的任务状态
        for task_id in list(self.task_gpus.keys()):
            if self._is_task_failed_or_stopped(task_id):
                failed_tasks.append(task_id)

        # 释放失败任务的GPU
        for task_id in failed_tasks:
            logger.info(f"检测到完成/失败任务，清理GPU分配: {task_id}")
            self.release_gpus_for_task(task_id)

            # 额外的GPU内存清理
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    logger.debug(f"已清理任务 {task_id} 的GPU内存")
            except:
                pass

    def _is_task_failed_or_stopped(self, task_id: str) -> bool:
        """检查任务是否已失败或停止"""
        try:
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            task = training_task_service.get_training_task(task_id)
            if task and task.status in ["FAILED", "STOPPED", "SUCCEEDED"]:
                return True
        except Exception as e:
            logger.debug(f"检查任务状态失败: {e}")
        return False

    def force_cleanup_task(self, task_id: str) -> bool:
        """强制清理指定任务的GPU资源"""
        logger.info(f"🔨 强制清理任务 {task_id} 的GPU资源")
        return self.release_gpus_for_task(task_id)

    def _get_gpu_memory_info(self, gpu_id: int) -> Dict:
        """获取GPU显存信息"""
        try:
            try:
                import pynvml
            except ImportError:
                # 如果pynvml不可用，返回空信息
                raise Exception("pynvml not available")

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # 获取显存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total
            used_memory = mem_info.used
            free_memory = mem_info.free
            
            # 获取GPU利用率
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = None
                memory_util = None
            
            # 获取GPU名称和温度
            try:
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            except:
                gpu_name = f"GPU {gpu_id}"
            
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
            
            return {
                "gpu_name": gpu_name,
                "memory": {
                    "total": total_memory,
                    "used": used_memory,
                    "free": free_memory,
                    "total_gb": round(total_memory / 1024**3, 2),
                    "used_gb": round(used_memory / 1024**3, 2),
                    "free_gb": round(free_memory / 1024**3, 2),
                    "usage_percent": round((used_memory / total_memory) * 100, 1) if total_memory > 0 else 0
                },
                "utilization": {
                    "gpu_percent": gpu_util,
                    "memory_percent": memory_util
                },
                "temperature": temperature
            }
        except Exception as e:
            logger.debug(f"获取GPU {gpu_id} 详细信息失败: {e}")
            return {
                "gpu_name": f"GPU {gpu_id}",
                "memory": {
                    "total": None,
                    "used": None,
                    "free": None,
                    "total_gb": None,
                    "used_gb": None,
                    "free_gb": None,
                    "usage_percent": None
                },
                "utilization": {
                    "gpu_percent": None,
                    "memory_percent": None
                },
                "temperature": None,
                "error": str(e)
            }

    def get_resource_status(self) -> Dict:
        """获取资源状态报告（包含显存信息）"""
        with self._lock:
            # 主动清理已结束的任务，确保GPU状态实时准确
            self._cleanup_failed_task_allocations()

            total_gpus = self.max_gpus
            # 注意：分配的GPU数量会在后面的循环中动态计算，因为可能有实时清理
            
            # 构建详细状态
            gpu_details = {}
            total_memory_gb = 0
            used_memory_gb = 0
            
            for gpu_id in range(total_gpus):
                # 获取GPU硬件信息
                gpu_hw_info = self._get_gpu_memory_info(gpu_id)
                
                # 基础分配状态
                if gpu_id in self.gpu_allocations:
                    task_id = self.gpu_allocations[gpu_id]
                    alloc_time = self.allocation_times.get(task_id)
                    username = self.task_users.get(task_id)
                    user_role = self.task_user_roles.get(task_id)

                    # 实时检查任务状态，如果任务已结束则立即清理GPU
                    should_release = False
                    task_status = "UNKNOWN"

                    try:
                        # 检查数据库中的任务状态
                        if self._is_task_failed_or_stopped(task_id):
                            logger.info(f"实时检测到任务 {task_id} 已结束，立即释放GPU {gpu_id}")
                            should_release = True
                            task_status = "FINISHED"
                        else:
                            # 任务仍在运行，获取最新状态
                            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                            task_db = training_task_service.get_training_task(task_id)
                            task_status = task_db.status if task_db else "NOT_FOUND"
                    except Exception as e:
                        logger.warning(f"检查任务 {task_id} 状态失败: {e}")
                        should_release = False

                    if should_release:
                        # 立即释放GPU并跳过这个GPU的状态构建
                        try:
                            self.release_gpus_for_task(task_id)
                            logger.info(f"实时清理完成: 任务 {task_id} GPU {gpu_id}")
                            # 重新构建为free状态
                            gpu_status = {
                                "status": "free",
                                "task_id": None,
                                "username": None,
                                "user_role": None,
                                "allocated_at": None,
                                "duration": None
                            }
                        except Exception as release_error:
                            logger.error(f"实时释放GPU失败: {release_error}")
                            # 如果释放失败，仍显示为已分配但标记为异常
                            gpu_status = {
                                "status": "allocated",
                                "task_id": task_id,
                                "username": username,
                                "user_role": user_role,
                                "allocated_at": alloc_time.isoformat() if alloc_time else None,
                                "duration": str(datetime.now() - alloc_time) if alloc_time else None,
                                "task_status": task_status,
                                "warning": "任务已结束但GPU释放失败"
                            }
                    else:
                        # 任务仍在运行，正常显示分配状态
                        gpu_status = {
                            "status": "allocated",
                            "task_id": task_id,
                            "username": username,
                            "user_role": user_role,
                            "allocated_at": alloc_time.isoformat() if alloc_time else None,
                            "duration": str(datetime.now() - alloc_time) if alloc_time else None,
                            "task_status": task_status
                        }
                else:
                    gpu_status = {
                        "status": "free",
                        "task_id": None,
                        "username": None,
                        "user_role": None,
                        "allocated_at": None,
                        "duration": None
                    }
                
                # 合并硬件信息
                gpu_details[gpu_id] = {**gpu_status, **gpu_hw_info}
                
                # 累计内存统计
                if gpu_hw_info["memory"]["total_gb"]:
                    total_memory_gb += gpu_hw_info["memory"]["total_gb"]
                if gpu_hw_info["memory"]["used_gb"]:
                    used_memory_gb += gpu_hw_info["memory"]["used_gb"]

            # 重新计算分配统计（实时清理后可能已变化）
            allocated_gpus = len(self.gpu_allocations)
            free_gpus = total_gpus - allocated_gpus

            return {
                "total_gpus": total_gpus,
                "allocated_gpus": allocated_gpus,
                "free_gpus": free_gpus,
                "utilization_rate": allocated_gpus / total_gpus if total_gpus > 0 else 0,
                "memory_summary": {
                    "total_gb": round(total_memory_gb, 2),
                    "used_gb": round(used_memory_gb, 2),
                    "free_gb": round(total_memory_gb - used_memory_gb, 2),
                    "usage_percent": round((used_memory_gb / total_memory_gb) * 100, 1) if total_memory_gb > 0 else 0
                },
                "gpu_details": gpu_details,
                "active_tasks": len(self.task_gpus)
            }

    def get_resource_status_for_user(self, username: str = None, user_role: str = "user") -> Dict:
        """
        获取用户可见的资源状态报告

        Args:
            username: 用户名，None表示获取当前用户
            user_role: 用户角色，admin可以看到所有资源

        Returns:
            Dict: 过滤后的资源状态
        """
        # 获取完整的资源状态
        full_status = self.get_resource_status()

        # 管理员可以看到所有信息
        if user_role == "admin":
            return full_status

        # 普通用户只能看到自己的任务信息
        if username:
            filtered_gpu_details = {}
            user_allocated_gpus = 0

            for gpu_id, gpu_info in full_status["gpu_details"].items():
                if gpu_info["status"] == "allocated":
                    if gpu_info["username"] == username:
                        # 显示自己的GPU分配
                        filtered_gpu_details[gpu_id] = gpu_info
                        user_allocated_gpus += 1
                    else:
                        # 隐藏其他用户的详细信息
                        filtered_gpu_details[gpu_id] = {
                            "status": "allocated",
                            "task_id": "***",
                            "username": "***",
                            "user_role": "***",
                            "allocated_at": "***",
                            "duration": "***",
                            "gpu_name": gpu_info.get("gpu_name", f"GPU {gpu_id}"),
                            "memory": gpu_info.get("memory", {}),
                            "utilization": gpu_info.get("utilization", {}),
                            "temperature": gpu_info.get("temperature")
                        }
                else:
                    # 显示空闲的GPU
                    filtered_gpu_details[gpu_id] = gpu_info

            # 返回过滤后的状态
            return {
                "total_gpus": full_status["total_gpus"],
                "allocated_gpus": full_status["allocated_gpus"],
                "free_gpus": full_status["free_gpus"],
                "user_allocated_gpus": user_allocated_gpus,  # 新增：用户分配的GPU数量
                "utilization_rate": full_status["utilization_rate"],
                "memory_summary": full_status["memory_summary"],
                "gpu_details": filtered_gpu_details,
                "active_tasks": full_status["active_tasks"],
                "user_tasks": len([tid for tid, uid in self.task_users.items() if uid == username])  # 新增：用户任务数量
            }

        # 如果没有提供用户ID，返回基础统计信息
        return {
            "total_gpus": full_status["total_gpus"],
            "allocated_gpus": full_status["allocated_gpus"],
            "free_gpus": full_status["free_gpus"],
            "utilization_rate": full_status["utilization_rate"],
            "memory_summary": full_status["memory_summary"]
        }
    
    def cleanup_stale_allocations(self, max_age_hours: int = 24):
        """清理长时间未释放的分配（防止资源泄漏）"""
        with self._lock:
            current_time = datetime.now()
            stale_tasks = []
            
            for task_id, alloc_time in self.allocation_times.items():
                if current_time - alloc_time > timedelta(hours=max_age_hours):
                    stale_tasks.append(task_id)
            
            for task_id in stale_tasks:
                logger.warning(f"清理过期的GPU分配: 任务 {task_id}")
                self.release_gpus_for_task(task_id)
            
            return len(stale_tasks)

    def _cleanup_completed_tasks_on_startup(self):
        """
        启动时恢复全局GPU分配状态
        修复服务重启后GPU资源管理不一致的问题

        全局恢复策略：
        1. 从数据库恢复所有服务实例的GPU使用情况
        2. 清理已完成任务的GPU资源记录
        3. 重建所有正在运行任务的GPU分配记录
        4. 确保GPU资源的全局一致性
        """
        try:
            logger.info("启动时检查数据库中的GPU分配状态...")

            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service

            # 清空内存中的分配记录
            self.gpu_allocations.clear()
            self.task_gpus.clear()
            self.allocation_times.clear()
            # 清空用户信息记录
            self.task_users.clear()
            self.task_user_roles.clear()

            # 全局GPU管理：恢复所有服务实例的GPU分配状态
            # 避免服务重启时清空其他服务的GPU分配记录
            try:
                all_status_tasks = training_task_service.get_all_training_tasks(limit=1000)
                logger.info(f"全局GPU恢复：检查所有服务的任务，共 {len(all_status_tasks)} 个任务")
            except Exception as e:
                logger.warning(f"获取任务失败: {e}")
                # 发生错误时采用安全策略：清空所有GPU分配
                all_status_tasks = []

            completed_statuses = ["SUCCEEDED", "FAILED", "STOPPED"]
            running_statuses = ["RUNNING", "PENDING"]
            cleaned_count = 0
            restored_count = 0

            # 筛选使用GPU的任务
            tasks_with_device = [task for task in all_status_tasks if task.device and task.device != "cpu"]
            logger.info(f"发现 {len(tasks_with_device)} 个使用GPU的任务")

            for task in tasks_with_device:
                task_id = task.task_id
                device = task.device
                status = task.status

                try:
                    if status in completed_statuses:
                        # 已完成的任务：确保GPU资源已清理
                        logger.info(f"已完成任务 {task_id} (状态: {status}), 确保GPU资源清理")
                        # 这里不需要调用release，因为内存中本来就是空的
                        cleaned_count += 1

                    elif status in running_statuses:
                        # 正在运行的任务：检查进程状态后决定是否重建GPU分配记录
                        logger.info(f"检查运行中任务 {task_id} (状态: {status}) 的实际进程状态")

                        # 关键改进：检查实际进程状态
                        should_restore_gpu = False
                        process_check_result = "UNKNOWN"

                        try:
                            # 检查进程是否真的还在运行
                            if hasattr(task, 'process_pid') and task.process_pid:
                                import psutil
                                if psutil.pid_exists(task.process_pid):
                                    try:
                                        process = psutil.Process(task.process_pid)
                                        if process.is_running() and 'python' in process.name().lower():
                                            should_restore_gpu = True
                                            process_check_result = "RUNNING"
                                            logger.info(f"任务 {task_id} 进程 {task.process_pid} 确实在运行，恢复GPU分配")
                                        else:
                                            process_check_result = "NOT_PYTHON"
                                            logger.warning(f"任务 {task_id} PID {task.process_pid} 存在但不是Python进程: {process.name()}")
                                    except psutil.NoSuchProcess:
                                        process_check_result = "PROCESS_DEAD"
                                        logger.warning(f"任务 {task_id} 进程 {task.process_pid} 已死亡")
                                else:
                                    process_check_result = "PID_NOT_EXISTS"
                                    logger.warning(f"任务 {task_id} PID {task.process_pid} 不存在")
                            else:
                                process_check_result = "NO_PID"
                                logger.warning(f"任务 {task_id} 没有PID记录，可能是旧任务")

                        except Exception as process_error:
                            logger.error(f"检查任务 {task_id} 进程状态失败: {process_error}")
                            process_check_result = "CHECK_FAILED"

                        if should_restore_gpu:
                            # 进程确实在运行，恢复GPU分配
                            gpu_ids = self._parse_device_to_gpu_ids(device)
                            if gpu_ids:
                                # 重建分配记录
                                for gpu_id in gpu_ids:
                                    self.gpu_allocations[gpu_id] = task_id
                                self.task_gpus[task_id] = set(gpu_ids)
                                self.allocation_times[task_id] = datetime.now()

                                # 恢复用户信息
                                if hasattr(task, 'username') and task.username:
                                    self.task_users[task_id] = task.username
                                    # 从用户信息推断用户角色，或使用默认值
                                    self.task_user_roles[task_id] = "user"  # 默认为普通用户

                                restored_count += 1
                                username_info = f" (用户: {task.username})" if hasattr(task, 'username') and task.username else ""
                                logger.info(f"已重建任务 {task_id} 的GPU分配: {gpu_ids}{username_info}")
                        else:
                            # 进程不在运行，标记任务为失败并清理
                            logger.warning(f"任务 {task_id} 数据库状态为{status}但进程不存在({process_check_result})，标记为失败")
                            try:
                                # 更新数据库中的任务状态
                                training_task_service.update_task_status(task_id, "FAILED")
                                cleaned_count += 1
                                logger.info(f"已将孤儿任务 {task_id} 标记为FAILED")
                            except Exception as update_error:
                                logger.error(f"更新孤儿任务状态失败: {update_error}")

                except Exception as task_error:
                    logger.warning(f"处理任务 {task_id} 时出错: {task_error}")

            logger.info(f"全局GPU状态恢复完成: 清理了{cleaned_count}个已完成任务, 重建了{restored_count}个运行中任务的GPU分配")

        except Exception as e:
            logger.error(f"启动时GPU状态恢复失败: {e}")

    def _parse_device_to_gpu_ids(self, device: str) -> List[int]:
        """解析device字符串为GPU ID列表"""
        try:
            if not device or device == "cpu":
                return []

            gpu_ids = []
            # 处理 "cuda:0,cuda:1" 格式
            if "," in device:
                parts = device.split(",")
                for part in parts:
                    part = part.strip()
                    if part.startswith("cuda:"):
                        gpu_id = int(part.split(":")[1])
                        gpu_ids.append(gpu_id)
            else:
                # 处理 "cuda:0" 格式
                if device.startswith("cuda:"):
                    gpu_id = int(device.split(":")[1])
                    gpu_ids.append(gpu_id)

            return gpu_ids
        except Exception as e:
            logger.warning(f"解析设备字符串失败 '{device}': {e}")
            return []

    def _schedule_delayed_cleanup(self):
        """延迟执行GPU清理，避免循环导入"""
        import threading
        import time

        def delayed_cleanup():
            # 等待3秒让服务完全初始化
            time.sleep(3)
            try:
                logger.info("开始延迟GPU清理...")
                self._cleanup_completed_tasks_on_startup()
            except Exception as e:
                logger.error(f"延迟GPU清理失败: {e}")

        # 在后台线程中执行清理
        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()


# 全局GPU资源管理器实例
gpu_resource_manager = GPUResourceManager()