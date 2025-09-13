"""
临时文件管理器
负责管理训练过程中产生的临时文件和目录，确保资源清理
"""
import os
import shutil
import logging
import threading
from typing import Set
from pathlib import Path

logger = logging.getLogger(__name__)

class TemporaryFileManager:
    """临时文件管理器"""
    
    def __init__(self):
        self.temp_files: Set[str] = set()
        self.temp_dirs: Set[str] = set()
        self._lock = threading.RLock()
    
    def register_temp_file(self, filepath: str):
        """注册临时文件"""
        with self._lock:
            self.temp_files.add(os.path.abspath(filepath))
            logger.debug(f"注册临时文件: {filepath}")
    
    def register_temp_dir(self, dirpath: str):
        """注册临时目录"""
        with self._lock:
            self.temp_dirs.add(os.path.abspath(dirpath))
            logger.debug(f"注册临时目录: {dirpath}")
    
    def unregister_temp_file(self, filepath: str):
        """取消注册临时文件（当文件被正常处理时）"""
        with self._lock:
            abs_path = os.path.abspath(filepath)
            self.temp_files.discard(abs_path)
            logger.debug(f"取消注册临时文件: {filepath}")
    
    def unregister_temp_dir(self, dirpath: str):
        """取消注册临时目录（当目录被正常处理时）"""
        with self._lock:
            abs_path = os.path.abspath(dirpath)
            self.temp_dirs.discard(abs_path)
            logger.debug(f"取消注册临时目录: {dirpath}")
    
    def cleanup_all(self) -> dict:
        """清理所有注册的临时文件和目录"""
        cleanup_results = {
            "files_cleaned": 0,
            "dirs_cleaned": 0,
            "files_failed": [],
            "dirs_failed": [],
            "total_size_freed": 0
        }
        
        with self._lock:
            # 清理临时文件
            for filepath in list(self.temp_files):
                try:
                    if os.path.exists(filepath):
                        # 获取文件大小用于统计
                        try:
                            file_size = os.path.getsize(filepath)
                            cleanup_results["total_size_freed"] += file_size
                        except:
                            pass
                        
                        os.remove(filepath)
                        cleanup_results["files_cleaned"] += 1
                        logger.info(f"✅ 清理临时文件: {filepath}")
                    
                    self.temp_files.discard(filepath)
                    
                except Exception as e:
                    cleanup_results["files_failed"].append({
                        "path": filepath,
                        "error": str(e)
                    })
                    logger.error(f"❌ 清理临时文件失败: {filepath}, 错误: {e}")
            
            # 清理临时目录
            for dirpath in list(self.temp_dirs):
                try:
                    if os.path.exists(dirpath):
                        # 获取目录大小用于统计
                        try:
                            dir_size = self._get_dir_size(dirpath)
                            cleanup_results["total_size_freed"] += dir_size
                        except:
                            pass
                        
                        shutil.rmtree(dirpath)
                        cleanup_results["dirs_cleaned"] += 1
                        logger.info(f"✅ 清理临时目录: {dirpath}")
                    
                    self.temp_dirs.discard(dirpath)
                    
                except Exception as e:
                    cleanup_results["dirs_failed"].append({
                        "path": dirpath,
                        "error": str(e)
                    })
                    logger.error(f"❌ 清理临时目录失败: {dirpath}, 错误: {e}")
        
        # 输出清理统计
        total_cleaned = cleanup_results["files_cleaned"] + cleanup_results["dirs_cleaned"]
        total_failed = len(cleanup_results["files_failed"]) + len(cleanup_results["dirs_failed"])
        size_mb = cleanup_results["total_size_freed"] / (1024 * 1024)
        
        logger.info(f"🧹 临时文件清理完成: 成功清理 {total_cleaned} 项，失败 {total_failed} 项，释放空间 {size_mb:.2f}MB")
        
        return cleanup_results
    
    def cleanup_by_pattern(self, pattern: str, base_dir: str = None) -> dict:
        """根据模式清理临时文件"""
        import glob
        
        cleanup_results = {
            "files_cleaned": 0,
            "files_failed": [],
            "total_size_freed": 0
        }
        
        try:
            search_pattern = pattern
            if base_dir:
                search_pattern = os.path.join(base_dir, pattern)
            
            matching_files = glob.glob(search_pattern, recursive=True)
            
            for filepath in matching_files:
                try:
                    if os.path.isfile(filepath):
                        file_size = os.path.getsize(filepath)
                        os.remove(filepath)
                        cleanup_results["files_cleaned"] += 1
                        cleanup_results["total_size_freed"] += file_size
                        logger.info(f"✅ 按模式清理文件: {filepath}")
                        
                        # 从注册列表中移除
                        with self._lock:
                            self.temp_files.discard(os.path.abspath(filepath))
                            
                except Exception as e:
                    cleanup_results["files_failed"].append({
                        "path": filepath,
                        "error": str(e)
                    })
                    logger.error(f"❌ 按模式清理文件失败: {filepath}, 错误: {e}")
        
        except Exception as e:
            logger.error(f"❌ 模式匹配失败: {pattern}, 错误: {e}")
        
        return cleanup_results
    
    def _get_dir_size(self, dirpath: str) -> int:
        """计算目录大小"""
        total_size = 0
        try:
            for dirpath_inner, dirnames, filenames in os.walk(dirpath):
                for filename in filenames:
                    filepath = os.path.join(dirpath_inner, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
        except:
            pass
        return total_size
    
    def get_status(self) -> dict:
        """获取临时文件管理状态"""
        with self._lock:
            total_size = 0
            
            # 计算文件大小
            for filepath in self.temp_files:
                try:
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                except:
                    pass
            
            # 计算目录大小
            for dirpath in self.temp_dirs:
                try:
                    if os.path.exists(dirpath):
                        total_size += self._get_dir_size(dirpath)
                except:
                    pass
            
            return {
                "registered_files": len(self.temp_files),
                "registered_dirs": len(self.temp_dirs),
                "total_size_mb": total_size / (1024 * 1024),
                "files": list(self.temp_files),
                "dirs": list(self.temp_dirs)
            }

# 全局临时文件管理器实例
temp_file_manager = TemporaryFileManager()