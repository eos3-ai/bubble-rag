"""
GPU资源动态分配管理器
基于现有的device参数架构，实现多任务GPU动态分配
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
        self.gpu_allocations: Dict[int, str] = {}  # {gpu_id: task_id}
        self.task_gpus: Dict[str, Set[int]] = {}   # {task_id: {gpu_ids}}
        self.allocation_times: Dict[str, datetime] = {}  # {task_id: allocation_time}
        self._lock = threading.RLock()
        self.max_gpus = self._detect_max_gpus()
        
        logger.info(f"🔧 GPU资源管理器初始化完成，检测到 {self.max_gpus} 个GPU")
    
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
    
    def allocate_gpus_for_task(self, task_id: str, device_request: str = "auto") -> Optional[str]:
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
        
        Returns:
            设备字符串，如 "cuda:0,cuda:1" 或 "cpu" 或 None（分配失败）
        """
        with self._lock:
            try:
                # CPU模式
                if device_request == "cpu":
                    logger.info(f"✅ 任务 {task_id} 使用CPU模式")
                    return "cpu"
                
                # 解析设备请求
                requested_gpus, num_gpus = self._parse_device_request(device_request)
                
                if requested_gpus is not None:
                    # 指定GPU模式
                    if self._can_allocate_specific_gpus(requested_gpus):
                        return self._do_allocate_gpus(task_id, requested_gpus)
                    else:
                        logger.error(f"❌ 指定的GPU {requested_gpus} 不可用")
                        return None
                else:
                    # 自动分配模式
                    available_gpus = self._get_available_gpus(num_gpus)
                    if available_gpus:
                        return self._do_allocate_gpus(task_id, available_gpus)
                    else:
                        logger.error(f"❌ 无法自动分配 {num_gpus} 个GPU，当前可用GPU不足")
                        return None
                        
            except Exception as e:
                logger.error(f"❌ GPU分配失败: {e}")
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
        for gpu_id in gpu_ids:
            if gpu_id in self.gpu_allocations:
                return False
        return True
    
    def _get_available_gpus(self, num_needed: int) -> Optional[List[int]]:
        """获取可用的GPU"""
        available = []
        for gpu_id in range(self.max_gpus):
            if gpu_id not in self.gpu_allocations:
                available.append(gpu_id)
                if len(available) >= num_needed:
                    break
        
        return available[:num_needed] if len(available) >= num_needed else None
    
    def _do_allocate_gpus(self, task_id: str, gpu_ids: List[int]) -> str:
        """执行GPU分配"""
        # 分配GPU
        for gpu_id in gpu_ids:
            self.gpu_allocations[gpu_id] = task_id
        
        # 记录任务GPU映射
        self.task_gpus[task_id] = set(gpu_ids)
        self.allocation_times[task_id] = datetime.now()
        
        # 生成CUDA设备字符串
        if len(gpu_ids) == 1:
            device_str = f"cuda:{gpu_ids[0]}"
        else:
            device_str = ",".join([f"cuda:{gpu_id}" for gpu_id in gpu_ids])
        
        logger.info(f"✅ 为任务 {task_id} 分配GPU: {device_str} (物理GPU: {gpu_ids})")
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
                
                logger.info(f"🔓 释放任务 {task_id} 的GPU资源: {list(gpu_ids)}")
                return True
                
            except Exception as e:
                logger.error(f"释放GPU资源失败: {e}")
                return False
    
    def _get_gpu_memory_info(self, gpu_id: int) -> Dict:
        """获取GPU显存信息"""
        try:
            import pynvml
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
            total_gpus = self.max_gpus
            allocated_gpus = len(self.gpu_allocations)
            free_gpus = total_gpus - allocated_gpus
            
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
                    gpu_status = {
                        "status": "allocated",
                        "task_id": task_id,
                        "allocated_at": alloc_time.isoformat() if alloc_time else None,
                        "duration": str(datetime.now() - alloc_time) if alloc_time else None
                    }
                else:
                    gpu_status = {
                        "status": "free",
                        "task_id": None,
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
    
    def cleanup_stale_allocations(self, max_age_hours: int = 24):
        """清理长时间未释放的分配（防止资源泄漏）"""
        with self._lock:
            current_time = datetime.now()
            stale_tasks = []
            
            for task_id, alloc_time in self.allocation_times.items():
                if current_time - alloc_time > timedelta(hours=max_age_hours):
                    stale_tasks.append(task_id)
            
            for task_id in stale_tasks:
                logger.warning(f"⚠️ 清理过期的GPU分配: 任务 {task_id}")
                self.release_gpus_for_task(task_id)
            
            return len(stale_tasks)


# 全局GPU资源管理器实例
gpu_resource_manager = GPUResourceManager()