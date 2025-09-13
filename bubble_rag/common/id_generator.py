"""ID生成器工具类"""

import uuid
import string
import random
from bubble_rag.utils.snowflake_utils import gen_id as snowflake_gen_id


class IdGenerator:
    """统一的ID生成器"""
    
    @staticmethod
    def generate_snowflake_id() -> str:
        """生成Snowflake ID"""
        return snowflake_gen_id()
    
    @staticmethod
    def generate_uuid() -> str:
        """生成标准UUID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_uuid_without_hyphen() -> str:
        """生成不带横线的UUID"""
        return str(uuid.uuid4()).replace("-", "")
    
    @staticmethod
    def generate_english_uuid(length: int = 32) -> str:
        """生成只包含英文字母的UUID"""
        letters = string.ascii_letters
        return ''.join(random.choice(letters) for _ in range(length))
    
    @staticmethod
    def format_uuid_with_groups(uuid_str: str, groups: int = 4) -> str:
        """格式化UUID，添加连字符分组"""
        if not uuid_str:
            return ""
        
        group_length = len(uuid_str) // groups
        formatted = '-'.join(
            uuid_str[i:i + group_length] 
            for i in range(0, len(uuid_str), group_length)
        )
        return formatted


# 向后兼容的便捷函数
def generate_id() -> str:
    """生成默认ID（Snowflake）"""
    return IdGenerator.generate_snowflake_id()


def generate_collection_name() -> str:
    """生成集合名称（英文UUID）"""
    return IdGenerator.generate_english_uuid(32)


def generate_uuid() -> str:
    """生成UUID"""
    return IdGenerator.generate_uuid()


def generate_uuid_without_hyphen() -> str:
    """生成不带横线的UUID"""
    return IdGenerator.generate_uuid_without_hyphen()