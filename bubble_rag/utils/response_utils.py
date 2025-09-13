"""响应处理公共工具类"""

from typing import Any
from bubble_rag.entity.query.response_model import SrvResult, PageResult
from loguru import logger


class ResponseBuilder:
    """统一的响应构建器"""
    
    @staticmethod
    def success(data: Any = None, msg: str = "success") -> SrvResult:
        """构建成功响应"""
        return SrvResult(code=200, msg=msg, data=data)
    
    @staticmethod
    def error(msg: str, code: int = 500, data: Any = None) -> SrvResult:
        """构建错误响应"""
        logger.error(f"API错误: {msg} (code: {code})")
        return SrvResult(code=code, msg=msg, data=data)
    
    @staticmethod
    def bad_request(msg: str = "请求参数错误", data: Any = None) -> SrvResult:
        """构建400错误响应"""
        return ResponseBuilder.error(msg, 400, data)
    
    @staticmethod
    def not_found(msg: str = "资源不存在", data: Any = None) -> SrvResult:
        """构建404错误响应"""
        return ResponseBuilder.error(msg, 404, data)
    
    @staticmethod
    def internal_error(msg: str = "内部服务错误", data: Any = None) -> SrvResult:
        """构建500错误响应"""
        return ResponseBuilder.error(msg, 500, data)
    
    @staticmethod
    def paginated_success(
        items: list, 
        total: int, 
        page: int, 
        page_size: int,
        msg: str = "success"
    ) -> SrvResult:
        """构建分页成功响应"""
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        page_result = PageResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        return ResponseBuilder.success(page_result, msg)


class ExceptionHandler:
    """异常处理器"""
    
    @staticmethod
    def handle_database_error(e: Exception, operation: str = "数据库操作") -> SrvResult:
        """处理数据库异常"""
        error_msg = f"{operation}失败: {str(e)}"
        logger.error(error_msg)
        return ResponseBuilder.internal_error(error_msg)
    
    @staticmethod
    def handle_validation_error(e: Exception, field: str = "参数") -> SrvResult:
        """处理验证异常"""
        error_msg = f"{field}验证失败: {str(e)}"
        logger.warning(error_msg)
        return ResponseBuilder.bad_request(error_msg)
    
    @staticmethod
    def handle_business_error(e: Exception, operation: str = "业务操作") -> SrvResult:
        """处理业务逻辑异常"""
        error_msg = f"{operation}失败: {str(e)}"
        logger.warning(error_msg)
        return ResponseBuilder.bad_request(error_msg)


# 向后兼容的便捷函数
def success_response(data: Any = None, msg: str = "success") -> SrvResult:
    """成功响应（向后兼容）"""
    return ResponseBuilder.success(data, msg)


def error_response(msg: str, code: int = 500, data: Any = None) -> SrvResult:
    """错误响应（向后兼容）"""
    return ResponseBuilder.error(msg, code, data)


def paginated_response(items: list, total: int, page: int, page_size: int) -> SrvResult:
    """分页响应（向后兼容）"""
    return ResponseBuilder.paginated_success(items, total, page, page_size)