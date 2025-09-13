"""
数据集管理服务
提供数据集的选择、加载、验证等功能
"""
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datasets import Dataset
from ..utils.data_loader import DataLoader

logger = logging.getLogger(__name__)

class DatasetService:
    """数据集服务类"""
    
    def __init__(self):
        self.data_loader = DataLoader()
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """
        列出可用的数据集选项
        
        Returns:
            数据集配置选项
        """
        return {
            "huggingface_datasets": [
                "sentence-transformers/all-nli",
                "sentence-transformers/stsb",
                "sentence-transformers/msmarco-hard-negatives",
                "sentence-transformers/quora-duplicates"
            ],
            "dataset_formats": [
                "单个HuggingFace数据集",
                "多个HuggingFace数据集（逗号分隔）",
                "本地文件夹（自动识别train/eval/test文件）",
                "单个本地文件",
                "多个本地文件（逗号分隔）"
            ],
            "supported_file_types": [".json", ".jsonl"],
            "required_columns": ["sentence1", "sentence2"],
            "target_columns": ["score", "label"]
        }
    
    def validate_dataset_path(self, dataset_path: str) -> Dict[str, Any]:
        """
        验证数据集路径的有效性
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            验证结果
        """
        try:
            if not dataset_path.strip():
                return {
                    "valid": False,
                    "message": "数据集路径不能为空",
                    "type": None
                }
            
            # 检查是否为多个路径
            paths = [p.strip() for p in dataset_path.split(",")]
            
            if len(paths) > 1:
                # 多个路径
                validation_results = []
                for path in paths:
                    result = self._validate_single_path(path)
                    validation_results.append(result)
                
                all_valid = all(r["valid"] for r in validation_results)
                return {
                    "valid": all_valid,
                    "message": "多个数据集路径验证完成",
                    "type": "multiple",
                    "details": validation_results
                }
            else:
                # 单个路径
                return self._validate_single_path(paths[0])
                
        except Exception as e:
            return {
                "valid": False,
                "message": f"验证过程出错: {str(e)}",
                "type": None
            }
    
    def _validate_single_path(self, path: str) -> Dict[str, Any]:
        """验证单个路径"""
        try:
            if os.path.isdir(path):
                return {
                    "valid": True,
                    "message": f"本地文件夹路径有效: {path}",
                    "type": "local_directory",
                    "path": path
                }
            elif os.path.isfile(path):
                return {
                    "valid": True,
                    "message": f"本地文件路径有效: {path}",
                    "type": "local_file",
                    "path": path
                }
            else:
                # 可能是HuggingFace数据集名称
                return {
                    "valid": True,
                    "message": f"可能的HuggingFace数据集: {path}",
                    "type": "huggingface",
                    "path": path
                }
        except Exception as e:
            return {
                "valid": False,
                "message": f"路径验证失败: {str(e)}",
                "type": None,
                "path": path
            }
    
    def preview_dataset(self, dataset_path: str, split: str = "train", num_samples: int = 5) -> Dict[str, Any]:
        """
        预览数据集内容
        
        Args:
            dataset_path: 数据集路径
            split: 数据分割类型
            num_samples: 预览样本数量
            
        Returns:
            数据集预览信息
        """
        try:
            # 临时设置环境变量
            original_path = os.getenv("DATASET_NAME_OR_PATH")
            os.environ["DATASET_NAME_OR_PATH"] = dataset_path
            
            try:
                dataset = self.data_loader.load_data(split)
                
                if dataset is None:
                    return {
                        "success": False,
                        "message": f"无法加载{split}分割的数据",
                        "data": None
                    }
                
                if isinstance(dataset, dict):
                    # 多数据集
                    preview_data = {}
                    for name, ds in dataset.items():
                        preview_data[name] = {
                            "num_samples": len(ds),
                            "columns": ds.column_names,
                            "samples": ds.select(range(min(num_samples, len(ds)))).to_dict()
                        }
                    
                    return {
                        "success": True,
                        "message": "多数据集预览成功",
                        "type": "multiple",
                        "data": preview_data
                    }
                else:
                    # 单数据集
                    samples = dataset.select(range(min(num_samples, len(dataset))))
                    return {
                        "success": True,
                        "message": "数据集预览成功",
                        "type": "single",
                        "data": {
                            "num_samples": len(dataset),
                            "columns": dataset.column_names,
                            "samples": samples.to_dict()
                        }
                    }
                    
            finally:
                # 恢复原始环境变量
                if original_path is not None:
                    os.environ["DATASET_NAME_OR_PATH"] = original_path
                elif "DATASET_NAME_OR_PATH" in os.environ:
                    del os.environ["DATASET_NAME_OR_PATH"]
                    
        except Exception as e:
            logger.error(f"预览数据集失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"预览数据集失败: {str(e)}",
                "data": None
            }
    
    def get_dataset_stats(self, dataset_path: str) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            数据集统计信息
        """
        try:
            # 临时设置环境变量
            original_path = os.getenv("DATASET_NAME_OR_PATH")
            os.environ["DATASET_NAME_OR_PATH"] = dataset_path
            
            try:
                train_data, eval_data, test_data = self.data_loader.load_all_splits()
                
                stats = {
                    "train": self._get_split_stats(train_data),
                    "eval": self._get_split_stats(eval_data),
                    "test": self._get_split_stats(test_data)
                }
                
                return {
                    "success": True,
                    "message": "数据集统计信息获取成功",
                    "data": stats
                }
                
            finally:
                # 恢复原始环境变量
                if original_path is not None:
                    os.environ["DATASET_NAME_OR_PATH"] = original_path
                elif "DATASET_NAME_OR_PATH" in os.environ:
                    del os.environ["DATASET_NAME_OR_PATH"]
                    
        except Exception as e:
            logger.error(f"获取数据集统计信息失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"获取数据集统计信息失败: {str(e)}",
                "data": None
            }
    
    def _get_split_stats(self, data: Union[Dataset, Dict[str, Dataset], None]) -> Dict[str, Any]:
        """获取数据分割的统计信息"""
        if data is None:
            return {
                "available": False,
                "num_datasets": 0,
                "total_samples": 0
            }
        
        if isinstance(data, dict):
            # 多数据集
            total_samples = sum(len(ds) for ds in data.values())
            return {
                "available": True,
                "type": "multiple",
                "num_datasets": len(data),
                "total_samples": total_samples,
                "datasets": {
                    name: {
                        "num_samples": len(ds),
                        "columns": ds.column_names
                    }
                    for name, ds in data.items()
                }
            }
        else:
            # 单数据集
            return {
                "available": True,
                "type": "single",
                "num_datasets": 1,
                "total_samples": len(data),
                "columns": data.column_names
            }

# 全局数据集服务实例
dataset_service = DatasetService()