"""参数验证公共工具类"""

from typing import Any, Optional, Dict, List
import re
from loguru import logger


class ValidationError(Exception):
    """验证异常"""
    pass


class Validator:
    """参数验证器"""
    
    @staticmethod
    def require_non_empty(value: Any, field_name: str) -> Any:
        """验证非空参数"""
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValidationError(f"{field_name}不能为空")
        return value
    
    @staticmethod
    def validate_id(value: str, field_name: str = "ID") -> str:
        """验证ID格式"""
        if not value or not value.strip():
            raise ValidationError(f"{field_name}不能为空")
        
        # 基本长度检查
        value = value.strip()
        if len(value) < 1 or len(value) > 64:
            raise ValidationError(f"{field_name}长度必须在1-64字符之间")
        
        return value
    
    @staticmethod
    def validate_page_params(page: int, page_size: int) -> tuple[int, int]:
        """验证分页参数"""
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 20
        elif page_size > 1000:
            page_size = 1000
            
        return page, page_size
    
    @staticmethod
    def validate_file_path(file_path: str, field_name: str = "文件路径") -> str:
        """验证文件路径"""
        if not file_path or not file_path.strip():
            raise ValidationError(f"{field_name}不能为空")
        
        file_path = file_path.strip()
        
        # 基本安全检查
        dangerous_patterns = ['../', '..\\', '<', '>', '|', '&', ';']
        for pattern in dangerous_patterns:
            if pattern in file_path:
                raise ValidationError(f"{field_name}包含非法字符")
        
        return file_path
    
    @staticmethod
    def validate_model_name(model_name: str, field_name: str = "模型名称") -> str:
        """验证模型名称"""
        if not model_name or not model_name.strip():
            raise ValidationError(f"{field_name}不能为空")
        
        model_name = model_name.strip()
        
        # 长度检查
        if len(model_name) > 256:
            raise ValidationError(f"{field_name}长度不能超过256字符")
        
        return model_name
    
    @staticmethod
    def validate_port(port: int, field_name: str = "端口") -> int:
        """验证端口号"""
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValidationError(f"{field_name}必须在1-65535范围内")
        
        return port
    
    @staticmethod
    def validate_embedding_dim(dim: int, field_name: str = "向量维度") -> int:
        """验证向量维度"""
        if not isinstance(dim, int) or dim < 1:
            raise ValidationError(f"{field_name}必须为正整数")
        
        if dim > 8192:
            raise ValidationError(f"{field_name}不能超过8192")
        
        return dim
    
    @staticmethod
    def validate_chunk_size(size: int, field_name: str = "分块大小") -> int:
        """验证分块大小"""
        if not isinstance(size, int) or size < 1:
            raise ValidationError(f"{field_name}必须为正整数")
        
        if size > 8192:
            raise ValidationError(f"{field_name}不能超过8192")
        
        return size


class ModelValidator:
    """模型相关验证器"""
    
    VALID_MODEL_TYPES = {0, 1, 2}  # 0-embedding, 1-rerank, 2-llm
    
    @staticmethod
    def validate_model_type(model_type: int) -> int:
        """验证模型类型"""
        if model_type not in ModelValidator.VALID_MODEL_TYPES:
            raise ValidationError(f"模型类型必须为: {', '.join(map(str, ModelValidator.VALID_MODEL_TYPES))}")
        return model_type
    
    @staticmethod
    def validate_model_config(config_dict: Dict) -> Dict:
        """验证模型配置"""
        required_fields = ['model_name', 'model_type']
        
        for field in required_fields:
            if field not in config_dict or not config_dict[field]:
                raise ValidationError(f"模型配置缺少必填字段: {field}")
        
        # 验证字段
        config_dict['model_name'] = Validator.validate_model_name(config_dict['model_name'])
        config_dict['model_type'] = ModelValidator.validate_model_type(config_dict['model_type'])
        
        return config_dict


# 向后兼容的便捷函数
def validate_required_param(value: Any, field_name: str) -> Any:
    """验证必填参数（向后兼容）"""
    return Validator.require_non_empty(value, field_name)


def validate_id_param(value: str, field_name: str = "ID") -> str:
    """验证ID参数（向后兼容）"""
    return Validator.validate_id(value, field_name)


def validate_pagination(page: int, page_size: int) -> tuple[int, int]:
    """验证分页参数（向后兼容）"""
    return Validator.validate_page_params(page, page_size)