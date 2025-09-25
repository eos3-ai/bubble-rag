"""
配置管理服务
提供训练配置的管理，包括输出路径、训练类型、环境变量等
"""
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigService:
    """配置服务类"""
    
    def __init__(self):
        self.env_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".env"
        )
    
    def get_training_types(self) -> Dict[str, Any]:
        """
        获取支持的训练类型和描述
        
        Returns:
            训练类型配置
        """
        return {
            "supported_types": [
                {
                    "value": "embedding",
                    "label": "Embedding模型训练",
                    "description": "训练文本嵌入模型，用于语义相似度计算和检索",
                    "use_cases": ["语义搜索", "文档检索", "相似度计算", "聚类分析"],
                    "output_type": "dense vectors"
                },
                {
                    "value": "reranker", 
                    "label": "Reranker模型训练",
                    "description": "训练重排序模型，用于对检索结果进行精确排序",
                    "use_cases": ["搜索结果重排", "问答系统", "文档排序", "相关性判断"],
                    "output_type": "relevance scores"
                }
            ],
            "default": "embedding"
        }
    
    def generate_output_path(self, train_type: str, model_name: str, custom_suffix: str = "") -> str:
        """
        生成输出路径
        
        Args:
            train_type: 训练类型
            model_name: 模型名称
            custom_suffix: 自定义后缀
            
        Returns:
            生成的输出路径
        """
        # 处理模型名称，移除路径分隔符
        safe_model_name = model_name.replace("/", "-").replace("\\", "-")
        
        # 时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 构建路径
        if custom_suffix:
            path = f"output/training_{train_type}_{safe_model_name}_{custom_suffix}_{timestamp}"
        else:
            path = f"output/training_{train_type}_{safe_model_name}_{timestamp}"
        
        return path
    
    def validate_output_path(self, output_path: str) -> Dict[str, Any]:
        """
        验证输出路径
        
        Args:
            output_path: 输出路径
            
        Returns:
            验证结果
        """
        try:
            if not output_path.strip():
                return {
                    "valid": False,
                    "message": "输出路径不能为空",
                    "suggestions": []
                }
            
            path = Path(output_path)
            
            # 检查路径格式
            if path.is_absolute() and path.exists() and path.is_file():
                return {
                    "valid": False,
                    "message": "输出路径不能是已存在的文件",
                    "suggestions": ["请指定一个文件夹路径"]
                }
            
            # 检查父目录是否可写
            try:
                parent_dir = path.parent
                if not parent_dir.exists():
                    return {
                        "valid": True,
                        "message": "输出路径有效，将创建目录结构",
                        "suggestions": [f"将创建目录: {parent_dir}"],
                        "will_create": True
                    }
                elif parent_dir.is_dir():
                    # 检查是否可写
                    test_file = parent_dir / ".write_test"
                    try:
                        test_file.touch()
                        test_file.unlink()
                        writable = True
                    except Exception:
                        writable = False
                    
                    if not writable:
                        return {
                            "valid": False,
                            "message": "父目录不可写",
                            "suggestions": ["请检查目录权限或选择其他路径"]
                        }
                else:
                    return {
                        "valid": False,
                        "message": "父路径不是目录",
                        "suggestions": ["请选择有效的目录路径"]
                    }
            except Exception as e:
                return {
                    "valid": False,
                    "message": f"路径验证失败: {str(e)}",
                    "suggestions": ["请检查路径格式和权限"]
                }
            
            # 检查路径是否已存在
            if path.exists():
                if path.is_dir():
                    return {
                        "valid": True,
                        "message": "输出路径有效，目录已存在",
                        "suggestions": ["目录已存在，训练文件将保存到此目录"],
                        "exists": True
                    }
                else:
                    return {
                        "valid": False,
                        "message": "输出路径已存在且不是目录",
                        "suggestions": ["请选择其他路径或删除已存在的文件"]
                    }
            
            return {
                "valid": True,
                "message": "输出路径有效",
                "suggestions": [],
                "will_create": True
            }
            
        except Exception as e:
            logger.error(f"验证输出路径失败: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "message": f"验证输出路径失败: {str(e)}",
                "suggestions": []
            }
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """
        获取可配置的训练参数
        
        Returns:
            训练参数配置
        """
        return {
            "basic_parameters": [
                {
                    "name": "num_train_epochs",
                    "display_name": "训练轮数",
                    "type": "integer",
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "description": "模型训练的总轮数"
                },
                {
                    "name": "per_device_train_batch_size",
                    "display_name": "训练批次大小",
                    "type": "integer", 
                    "default": 16,
                    "min": 1,
                    "max": 512,
                    "description": "每个设备的训练批次大小"
                },
                {
                    "name": "learning_rate",
                    "display_name": "学习率",
                    "type": "float",
                    "default": 2e-5,
                    "min": 1e-6,
                    "max": 1e-2,
                    "description": "模型训练的学习率"
                },
                {
                    "name": "warmup_ratio",
                    "display_name": "预热比例",
                    "type": "float",
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "学习率预热的比例"
                }
            ],
            "advanced_parameters": [
                {
                    "name": "gradient_accumulation_steps",
                    "display_name": "梯度累积步数",
                    "type": "integer",
                    "default": 1,
                    "min": 1,
                    "max": 128,
                    "description": "梯度累积的步数，用于模拟更大的批次"
                },
                {
                    "name": "max_steps",
                    "display_name": "最大训练步数",
                    "type": "integer",
                    "default": -1,
                    "min": -1,
                    "max": 1000000,
                    "description": "最大训练步数，-1表示使用epoch设置"
                },
                {
                    "name": "eval_strategy",
                    "display_name": "评估策略",
                    "type": "select",
                    "default": "steps",
                    "options": ["no", "steps", "epoch"],
                    "description": "模型评估的策略"
                },
                {
                    "name": "eval_steps",
                    "display_name": "评估步数间隔",
                    "type": "integer",
                    "default": 1000,
                    "min": 1,
                    "max": 10000,
                    "description": "每隔多少步进行一次评估"
                },
                {
                    "name": "save_strategy",
                    "display_name": "保存策略",
                    "type": "select",
                    "default": "steps",
                    "options": ["no", "steps", "epoch"],
                    "description": "模型保存的策略"
                },
                {
                    "name": "save_steps",
                    "display_name": "保存步数间隔",
                    "type": "integer",
                    "default": 500,
                    "min": 1,
                    "max": 10000,
                    "description": "每隔多少步保存一次模型"
                },
                {
                    "name": "lr_scheduler_type",
                    "display_name": "学习率调度器",
                    "type": "select",
                    "default": "linear",
                    "options": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                    "description": "学习率调度策略"
                },
                {
                    "name": "weight_decay",
                    "display_name": "权重衰减",
                    "type": "float",
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "L2正则化的权重衰减系数"
                }
            ],
            "precision_parameters": [
                {
                    "name": "bf16",
                    "display_name": "使用BF16精度",
                    "type": "boolean",
                    "default": False,
                    "description": "是否使用BF16混合精度训练（需要硬件支持）"
                },
                {
                    "name": "fp16",
                    "display_name": "使用FP16精度",
                    "type": "boolean",
                    "default": False,
                    "description": "是否使用FP16混合精度训练"
                }
            ],
            "data_parameters": [
                {
                    "name": "train_sample_size",
                    "display_name": "训练集样本数量限制",
                    "type": "integer",
                    "default": -1,
                    "min": -1,
                    "max": 10000000,
                    "description": "限制训练集每个数据集的样本数量，-1表示不限制，0表示不使用该数据集"
                },
                {
                    "name": "eval_sample_size",
                    "display_name": "验证集样本数量限制",
                    "type": "integer",
                    "default": -1,
                    "min": -1,
                    "max": 10000000,
                    "description": "限制验证集每个数据集的样本数量，-1表示不限制，0表示不使用该数据集"
                },
                {
                    "name": "test_sample_size",
                    "display_name": "测试集样本数量限制",
                    "type": "integer",
                    "default": -1,
                    "min": -1,
                    "max": 10000000,
                    "description": "限制测试集每个数据集的样本数量，-1表示不限制，0表示不使用该数据集"
                },
                {
                    "name": "dataloader_num_workers",
                    "display_name": "数据加载器工作进程数",
                    "type": "integer",
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "description": "数据加载器的工作进程数"
                }
            ]
        }
    
    def get_current_env_config(self) -> Dict[str, Any]:
        """
        获取当前环境变量配置
        
        Returns:
            当前环境变量配置
        """
        try:
            # 从.env文件读取配置
            env_config = {}
            if os.path.exists(self.env_file_path):
                with open(self.env_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_config[key.strip()] = value.strip()
            
            # 获取运行时环境变量（优先级更高）
            runtime_config = {}
            for key, value in os.environ.items():
                if key.startswith(('TRAIN_', 'MODEL_', 'OUTPUT_', 'DATASET_', 'NUM_', 'PER_', 'LEARNING_', 'WARMUP_', 'LR_', 'BF16', 'FP16', 'EVAL_', 'SAVE_', 'LOGGING_', 'GRADIENT_', 'MAX_', 'WEIGHT_', 'SAMPLE_', 'DATALOADER_')):
                    runtime_config[key] = value
            
            return {
                "success": True,
                "message": "环境配置获取成功",
                "data": {
                    "file_config": env_config,
                    "runtime_config": runtime_config,
                    "effective_config": {**env_config, **runtime_config}
                }
            }
            
        except Exception as e:
            logger.error(f"获取环境配置失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"获取环境配置失败: {str(e)}",
                "data": None
            }
    
    def update_env_config(self, config: Dict[str, str]) -> Dict[str, Any]:
        """
        更新环境变量配置
        
        Args:
            config: 要更新的配置项
            
        Returns:
            更新结果
        """
        try:
            # 读取现有配置
            existing_config = {}
            if os.path.exists(self.env_file_path):
                with open(self.env_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = ""
            
            # 解析现有配置
            lines = content.split('\n')
            config_lines = {}
            other_lines = []
            
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in stripped:
                    key, value = stripped.split('=', 1)
                    config_lines[key.strip()] = line
                else:
                    other_lines.append(line)
            
            # 更新配置
            for key, value in config.items():
                # 格式化配置行
                config_line = f"{key}={value}"
                config_lines[key] = config_line
                # 同时设置运行时环境变量
                os.environ[key] = str(value)
            
            # 重新构建文件内容
            new_lines = other_lines + list(config_lines.values())
            new_content = '\n'.join(new_lines)
            
            # 写入文件
            os.makedirs(os.path.dirname(self.env_file_path), exist_ok=True)
            with open(self.env_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "message": f"成功更新 {len(config)} 个配置项",
                "data": {
                    "updated_keys": list(config.keys()),
                    "config_file": self.env_file_path
                }
            }
            
        except Exception as e:
            logger.error(f"更新环境配置失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"更新环境配置失败: {str(e)}",
                "data": None
            }
    
    def validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证训练配置的有效性
        
        Args:
            config: 训练配置
            
        Returns:
            验证结果
        """
        try:
            errors = []
            warnings = []
            
            # 获取参数定义
            param_definitions = self.get_training_parameters()
            all_params = {}
            for category in param_definitions.values():
                for param in category:
                    all_params[param["name"]] = param
            
            # 验证每个配置项
            for key, value in config.items():
                if key.lower() in all_params:
                    param_def = all_params[key.lower()]
                    
                    # 类型验证
                    if param_def["type"] == "integer":
                        try:
                            int_value = int(value)
                            if "min" in param_def and int_value < param_def["min"]:
                                errors.append(f"{param_def['display_name']} 值 {int_value} 小于最小值 {param_def['min']}")
                            if "max" in param_def and int_value > param_def["max"]:
                                errors.append(f"{param_def['display_name']} 值 {int_value} 大于最大值 {param_def['max']}")
                        except ValueError:
                            errors.append(f"{param_def['display_name']} 值 '{value}' 不是有效的整数")
                    
                    elif param_def["type"] == "float":
                        try:
                            float_value = float(value)
                            if "min" in param_def and float_value < param_def["min"]:
                                errors.append(f"{param_def['display_name']} 值 {float_value} 小于最小值 {param_def['min']}")
                            if "max" in param_def and float_value > param_def["max"]:
                                errors.append(f"{param_def['display_name']} 值 {float_value} 大于最大值 {param_def['max']}")
                        except ValueError:
                            errors.append(f"{param_def['display_name']} 值 '{value}' 不是有效的浮点数")
                    
                    elif param_def["type"] == "select":
                        if value not in param_def["options"]:
                            errors.append(f"{param_def['display_name']} 值 '{value}' 不在有效选项中: {param_def['options']}")
                    
                    elif param_def["type"] == "boolean":
                        if str(value).lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
                            errors.append(f"{param_def['display_name']} 值 '{value}' 不是有效的布尔值")
            
            # 逻辑验证
            if "bf16" in config and "fp16" in config:
                if str(config["bf16"]).lower() == 'true' and str(config["fp16"]).lower() == 'true':
                    errors.append("不能同时启用BF16和FP16精度")
            
            if "eval_strategy" in config and config["eval_strategy"] == "steps":
                if "eval_steps" not in config:
                    warnings.append("使用steps评估策略时建议设置eval_steps参数")
            
            if "save_strategy" in config and config["save_strategy"] == "steps":
                if "save_steps" not in config:
                    warnings.append("使用steps保存策略时建议设置save_steps参数")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "message": "配置验证完成" if len(errors) == 0 else f"发现 {len(errors)} 个错误"
            }
            
        except Exception as e:
            logger.error(f"验证训练配置失败: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "errors": [f"验证过程出错: {str(e)}"],
                "warnings": [],
                "message": "配置验证失败"
            }

# 全局配置服务实例
config_service = ConfigService()