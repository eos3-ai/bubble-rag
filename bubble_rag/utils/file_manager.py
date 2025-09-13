"""文件操作公共工具类"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, BinaryIO
from loguru import logger


class FileManager:
    """文件管理器"""
    
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.doc', '.docx', '.pdf', '.html', '.htm',
        '.xls', '.xlsx', '.csv', '.ppt', '.pptx'
    }
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """验证文件扩展名"""
        if not filename:
            return False
        
        file_path = Path(filename)
        extension = file_path.suffix.lower()
        return extension in FileManager.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小"""
        try:
            return os.path.getsize(file_path)
        except (OSError, FileNotFoundError) as e:
            logger.error(f"获取文件大小失败: {file_path}, 错误: {str(e)}")
            return 0
    
    @staticmethod
    def validate_file_size(file_path: str) -> bool:
        """验证文件大小"""
        file_size = FileManager.get_file_size(file_path)
        return 0 < file_size <= FileManager.MAX_FILE_SIZE
    
    @staticmethod
    def calculate_file_md5(file_path: str) -> Optional[str]:
        """计算文件MD5值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (OSError, FileNotFoundError) as e:
            logger.error(f"计算文件MD5失败: {file_path}, 错误: {str(e)}")
            return None
    
    @staticmethod
    def calculate_content_md5(content: bytes) -> str:
        """计算内容MD5值"""
        return hashlib.md5(content).hexdigest()
    
    @staticmethod
    def get_mime_type(filename: str) -> Optional[str]:
        """获取文件MIME类型"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type
    
    @staticmethod
    def ensure_directory(dir_path: str) -> bool:
        """确保目录存在"""
        try:
            os.makedirs(dir_path, exist_ok=True)
            return True
        except OSError as e:
            logger.error(f"创建目录失败: {dir_path}, 错误: {str(e)}")
            return False
    
    @staticmethod
    def safe_remove_file(file_path: str) -> bool:
        """安全删除文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"文件删除成功: {file_path}")
            return True
        except OSError as e:
            logger.error(f"文件删除失败: {file_path}, 错误: {str(e)}")
            return False
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """获取文件信息"""
        info = {
            'exists': os.path.exists(file_path),
            'size': 0,
            'md5': None,
            'mime_type': None,
            'extension': None
        }
        
        if info['exists']:
            info['size'] = FileManager.get_file_size(file_path)
            info['md5'] = FileManager.calculate_file_md5(file_path)
            info['mime_type'] = FileManager.get_mime_type(file_path)
            info['extension'] = Path(file_path).suffix.lower()
        
        return info


class FileValidator:
    """文件验证器"""
    
    @staticmethod
    def validate_upload_file(filename: str, content: bytes) -> Tuple[bool, str]:
        """验证上传文件"""
        # 检查文件名
        if not filename:
            return False, "文件名不能为空"
        
        # 检查扩展名
        if not FileManager.validate_file_extension(filename):
            return False, f"不支持的文件类型，支持的格式: {', '.join(FileManager.SUPPORTED_EXTENSIONS)}"
        
        # 检查内容大小
        if len(content) == 0:
            return False, "文件内容不能为空"
        
        if len(content) > FileManager.MAX_FILE_SIZE:
            return False, f"文件大小不能超过 {FileManager.MAX_FILE_SIZE // (1024*1024)}MB"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_file_path(file_path: str) -> Tuple[bool, str]:
        """验证文件路径"""
        if not file_path:
            return False, "文件路径不能为空"
        
        if not os.path.exists(file_path):
            return False, "文件不存在"
        
        if not os.path.isfile(file_path):
            return False, "路径不是文件"
        
        if not FileManager.validate_file_size(file_path):
            return False, "文件大小不符合要求"
        
        filename = os.path.basename(file_path)
        if not FileManager.validate_file_extension(filename):
            return False, "不支持的文件类型"
        
        return True, "验证通过"


# 向后兼容的便捷函数
def get_file_md5(file_path: str) -> Optional[str]:
    """获取文件MD5（向后兼容）"""
    return FileManager.calculate_file_md5(file_path)


def validate_file_type(filename: str) -> bool:
    """验证文件类型（向后兼容）"""
    return FileManager.validate_file_extension(filename)


def get_file_size(file_path: str) -> int:
    """获取文件大小（向后兼容）"""
    return FileManager.get_file_size(file_path)