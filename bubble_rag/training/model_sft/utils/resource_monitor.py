"""
资源监控和清理工具
监控系统资源使用情况并进行自动清理
"""
import psutil
import threading
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """资源使用情况"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    open_files: int
    thread_count: int


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, 
                 check_interval: int = 60,  # 检查间隔秒数
                 cpu_threshold: float = 90.0,  # CPU使用率阈值
                 memory_threshold: float = 85.0,  # 内存使用率阈值
                 monitor_threads: bool = False):  # 是否监控线程数，默认禁用
        self.check_interval = check_interval
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.monitor_threads = monitor_threads
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 添加警告去重机制
        self._last_warnings = {
            'cpu': 0,
            'memory': 0, 
            'threads': 0,
            'files': 0
        }
        self._warning_cooldown = 300  # 5分钟内不重复相同警告
        
    def start_monitoring(self):
        """开始监控"""
        with self._lock:
            if self.is_running:
                logger.warning("资源监控已在运行")
                return
                
            self.is_running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="resource-monitor",
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("资源监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            if not self.is_running:
                return
                
            self.is_running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            logger.info("资源监控已停止")
    
    def enable_thread_monitoring(self, enable: bool = True):
        """启用或禁用线程监控"""
        with self._lock:
            self.monitor_threads = enable
            status = "启用" if enable else "禁用"
            logger.info(f"线程监控已{status}")
    
    def disable_thread_monitoring(self):
        """禁用线程监控"""
        self.enable_thread_monitoring(False)
    
    def get_current_usage(self) -> ResourceUsage:
        """获取当前资源使用情况"""
        try:
            process = psutil.Process()
            
            # CPU使用率
            cpu_percent = process.cpu_percent()
            
            # 内存使用情况
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            memory_used_mb = memory_info.rss / (1024 * 1024)
            
            # 磁盘使用情况
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # 打开的文件数
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # 线程数
            try:
                thread_count = process.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                thread_count = 0
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_usage_percent=disk_usage_percent,
                open_files=open_files,
                thread_count=thread_count
            )
        except Exception as e:
            logger.error(f"获取资源使用情况失败: {str(e)}")
            return ResourceUsage(0, 0, 0, 0, 0, 0)
    
    def _monitor_loop(self):
        """监控循环"""
        logger.info("资源监控循环已启动")
        
        while self.is_running:
            try:
                usage = self.get_current_usage()
                self._log_usage(usage)
                
                # 检查是否需要警告（带去重机制）
                current_time = time.time()
                
                if usage.cpu_percent > self.cpu_threshold:
                    if current_time - self._last_warnings['cpu'] > self._warning_cooldown:
                        logger.warning(f"CPU使用率过高: {usage.cpu_percent:.1f}%")
                        self._last_warnings['cpu'] = current_time
                
                if usage.memory_percent > self.memory_threshold:
                    if current_time - self._last_warnings['memory'] > self._warning_cooldown:
                        logger.warning(f"内存使用率过高: {usage.memory_percent:.1f}% ({usage.memory_used_mb:.1f}MB)")
                        self._last_warnings['memory'] = current_time
                
                if self.monitor_threads and usage.thread_count > 50:  # 线程数过多（仅在启用时检查）
                    if current_time - self._last_warnings['threads'] > self._warning_cooldown:
                        logger.warning(f"线程数过多: {usage.thread_count} (将在5分钟后再次提醒)")
                        self._last_warnings['threads'] = current_time
                
                if usage.open_files > 1000:  # 打开文件过多
                    if current_time - self._last_warnings['files'] > self._warning_cooldown:
                        logger.warning(f"打开文件数过多: {usage.open_files} (将在5分钟后再次提醒)")
                        self._last_warnings['files'] = current_time
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"资源监控循环出错: {str(e)}")
                time.sleep(self.check_interval)
        
        logger.info("资源监控循环已结束")
    
    def _log_usage(self, usage: ResourceUsage):
        """记录资源使用情况"""
        logger.debug(
            f"资源使用情况 - CPU: {usage.cpu_percent:.1f}%, "
            f"内存: {usage.memory_percent:.1f}% ({usage.memory_used_mb:.1f}MB), "
            f"磁盘: {usage.disk_usage_percent:.1f}%, "
            f"文件: {usage.open_files}, "
            f"线程: {usage.thread_count}"
        )
    
    def cleanup_resources(self) -> Dict[str, Any]:
        """清理资源"""
        logger.info("开始资源清理...")
        cleanup_results = {
            "success": True,
            "actions": [],
            "errors": []
        }
        
        try:
            # 强制垃圾回收
            import gc
            collected = gc.collect()
            cleanup_results["actions"].append(f"垃圾回收: 回收了 {collected} 个对象")
            
            # 获取清理后的资源使用情况
            usage = self.get_current_usage()
            cleanup_results["final_usage"] = {
                "cpu_percent": usage.cpu_percent,
                "memory_percent": usage.memory_percent,
                "memory_used_mb": usage.memory_used_mb,
                "thread_count": usage.thread_count,
                "open_files": usage.open_files
            }
            
            logger.info("资源清理完成")
            
        except Exception as e:
            error_msg = f"资源清理失败: {str(e)}"
            logger.error(error_msg)
            cleanup_results["success"] = False
            cleanup_results["errors"].append(error_msg)
        
        return cleanup_results


# 全局资源监控器实例
resource_monitor = ResourceMonitor()