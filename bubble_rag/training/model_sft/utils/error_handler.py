"""
统一错误处理器
处理异常、记录错误日志，提供错误恢复机制
"""
import functools
import logging
import traceback
from typing import Optional, Dict, Any, Callable, Type
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    LOW = "low"          # 轻微错误，不影响核心功能
    MEDIUM = "medium"    # 中等错误，影响部分功能
    HIGH = "high"        # 严重错误，影响核心功能
    CRITICAL = "critical"  # 关键错误，系统可能崩溃


class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self):
        self.error_count = 0
        self.error_history = []
        
    def handle_error(self, 
                    error: Exception, 
                    context: str = "",
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            severity: 错误严重程度
            task_id: 相关任务ID
            
        Returns:
            错误处理结果
        """
        self.error_count += 1
        
        error_info = {
            "error_id": self.error_count,
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "severity": severity.value,
            "task_id": task_id,
            "traceback": traceback.format_exc()
        }
        
        # 记录到历史
        self.error_history.append(error_info)
        if len(self.error_history) > 100:  # 保留最近100个错误
            self.error_history.pop(0)
        
        # 根据严重程度选择日志级别
        if severity == ErrorSeverity.LOW:
            logger.warning(f"[{context}] {type(error).__name__}: {str(error)}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.error(f"[{context}] {type(error).__name__}: {str(error)}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"[{context}] 严重错误 - {type(error).__name__}: {str(error)}", exc_info=True)
        else:  # CRITICAL
            logger.critical(f"[{context}] 关键错误 - {type(error).__name__}: {str(error)}", exc_info=True)
        
        return {
            "success": False,
            "error_id": error_info["error_id"],
            "error_type": error_info["type"],
            "message": f"{context}: {str(error)}" if context else str(error),
            "severity": severity.value
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {"total_errors": 0, "by_type": {}, "by_severity": {}}
        
        by_type = {}
        by_severity = {}
        
        for error in self.error_history:
            error_type = error["type"]
            severity = error["severity"]
            
            by_type[error_type] = by_type.get(error_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_errors": self.error_history[-10:]  # 最近10个错误
        }


def with_error_handling(context: str = "", 
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       default_return: Any = None,
                       reraise: bool = False):
    """
    错误处理装饰器
    
    Args:
        context: 错误上下文
        severity: 错误严重程度
        default_return: 发生错误时的默认返回值
        reraise: 是否重新抛出异常
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = f"{func.__name__}"
                if context:
                    error_context = f"{context}.{error_context}"
                
                error_handler.handle_error(e, error_context, severity)
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def safe_execute(func: Callable, 
                *args, 
                context: str = "",
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                default_return: Any = None,
                **kwargs) -> Any:
    """
    安全执行函数，捕获并处理异常
    
    Args:
        func: 要执行的函数
        *args: 函数位置参数
        context: 错误上下文
        severity: 错误严重程度
        default_return: 发生错误时的默认返回值
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或默认返回值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_context = f"{func.__name__}"
        if context:
            error_context = f"{context}.{error_context}"
        
        error_handler.handle_error(e, error_context, severity)
        return default_return


def handle_database_error(error: Exception, operation: str, task_id: Optional[str] = None) -> Dict[str, Any]:
    """处理数据库相关错误"""
    return error_handler.handle_error(
        error, 
        f"database.{operation}", 
        ErrorSeverity.HIGH,
        task_id
    )


def handle_training_error(error: Exception, operation: str, task_id: Optional[str] = None) -> Dict[str, Any]:
    """处理训练相关错误"""
    return error_handler.handle_error(
        error, 
        f"training.{operation}", 
        ErrorSeverity.HIGH,
        task_id
    )


def handle_api_error(error: Exception, endpoint: str, task_id: Optional[str] = None) -> Dict[str, Any]:
    """处理API相关错误"""
    return error_handler.handle_error(
        error, 
        f"api.{endpoint}", 
        ErrorSeverity.MEDIUM,
        task_id
    )


# 全局错误处理器实例
error_handler = ErrorHandler()