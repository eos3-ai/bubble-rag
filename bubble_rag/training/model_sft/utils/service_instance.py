"""
服务实例标识管理工具
用于多服务实例场景下的进程归属管理
"""

import os
import socket
import psutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ServiceInstanceManager:
    """服务实例管理器"""
    
    def __init__(self):
        self._instance_id: Optional[str] = None
        self._port_detected: bool = False  # 端口检测成功标记
        self._detection_attempted: bool = False
    
    def get_service_instance_id(self) -> Optional[str]:
        """获取唯一的服务实例标识，如果端口未检测到则持续尝试"""
        if not self._detection_attempted:
            self._instance_id = self._generate_instance_id()
            self._detection_attempted = True
        
        # 如果之前没检测到端口，继续尝试检测
        if not self._port_detected and self._instance_id:
            self._try_update_with_port()
        
        return self._instance_id
    
    def _get_port_from_config(self) -> Optional[int]:
        """从配置文件获取训练服务端口"""
        try:
            from bubble_rag.server_config import TRAINING_SERVER_PORT
            logger.debug(f"从配置文件读取到端口: {TRAINING_SERVER_PORT}")
            return TRAINING_SERVER_PORT
        except ImportError as e:
            logger.debug(f"无法导入配置文件中的训练服务端口: {e}")
            return None
        except Exception as e:
            logger.error(f"读取配置文件端口时出错: {e}")
            return None
    
    def _generate_instance_id(self) -> Optional[str]:
        """生成基于hostname+端口的稳定服务实例ID，支持容器化部署"""
        # 优先使用TRAINING_SERVICE_ID配置作为hostname（用于容器化部署）
        try:
            from bubble_rag.server_config import TRAINING_SERVICE_ID
            hostname = TRAINING_SERVICE_ID or socket.gethostname()
            using_config_id = bool(TRAINING_SERVICE_ID)
        except ImportError:
            hostname = socket.gethostname()
            using_config_id = False
        
        # 只从配置文件读取端口
        config_port = self._get_port_from_config()
        if config_port:
            self._port_detected = True  # 标记端口获取成功
            instance_id = f"{hostname}_{config_port}"
            if using_config_id:
                logger.info(f"✅ 服务实例ID (TRAINING_SERVICE_ID): {instance_id}")
            else:
                logger.info(f"✅ 服务实例ID (系统hostname): {instance_id}")
                logger.warning("⚠️  建议设置TRAINING_SERVICE_ID环境变量以避免容器重启时hostname变化问题")
            return instance_id
        
        # 如果配置文件中没有端口，则服务实例ID为空
        logger.error("❌ 配置文件中未找到训练服务端口，无法生成服务实例ID")
        return None
    
    def _try_update_with_port(self):
        """尝试更新服务实例ID（只使用配置文件端口）"""
        if self._port_detected:
            return  # 已经获取到端口，无需再尝试
        
        # 只尝试从配置文件获取端口
        config_port = self._get_port_from_config()
        if config_port:
            try:
                from bubble_rag.server_config import TRAINING_SERVICE_ID
                hostname = TRAINING_SERVICE_ID or socket.gethostname()
            except ImportError:
                hostname = socket.gethostname()
            old_instance_id = self._instance_id
            self._instance_id = f"{hostname}_{config_port}"
            self._port_detected = True
            logger.info(f"🔄 服务实例ID已更新: {old_instance_id} → {self._instance_id}")
        else:
            logger.debug("配置文件中仍未找到训练服务端口")
    
    
    def is_same_instance(self, instance_id: Optional[str]) -> bool:
        """检查是否为同一个服务实例"""
        return self.get_service_instance_id() == instance_id
    
    def force_port_detection(self) -> bool:
        """强制执行端口检测更新，返回是否成功检测到端口"""
        if self._port_detected:
            logger.info("端口已检测成功，无需强制更新")
            return True
        
        self._try_update_with_port()
        return self._port_detected
    
    def get_instance_info(self) -> dict:
        """获取实例详细信息"""
        instance_id = self.get_service_instance_id()
        parts = instance_id.split('_') if instance_id else []
        config_port = self._get_port_from_config()
        
        try:
            hostname = parts[0] if len(parts) > 0 else socket.gethostname()
            port = int(parts[1]) if len(parts) > 1 else config_port
            
            return {
                "instance_id": instance_id,
                "hostname": hostname,
                "pid": os.getpid(),  # 当前进程PID
                "port": port,
                "is_current": True,
                "config_port": config_port,  # 配置文件端口
                "port_detected": self._port_detected,  # 端口获取状态
                "detection_attempted": self._detection_attempted,  # 检测尝试状态
                "format": "hostname_port"  # 格式标识
            }
        except (ValueError, IndexError):
            return {
                "instance_id": instance_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "port": config_port,
                "is_current": True,
                "config_port": config_port,
                "port_detected": self._port_detected,
                "detection_attempted": self._detection_attempted,
                "format": "hostname_port"
            }


# 全局服务实例管理器
service_instance_manager = ServiceInstanceManager()

# 便捷函数
def get_service_instance_id() -> Optional[str]:
    """获取当前服务实例ID"""
    return service_instance_manager.get_service_instance_id()

def is_same_service_instance(instance_id: Optional[str]) -> bool:
    """检查是否为同一个服务实例"""
    return service_instance_manager.is_same_instance(instance_id)

def get_service_instance_info() -> dict:
    """获取服务实例信息"""
    return service_instance_manager.get_instance_info()

def force_port_detection() -> bool:
    """强制执行端口检测更新"""
    return service_instance_manager.force_port_detection()