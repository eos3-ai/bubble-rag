"""
Configuration builder for training.

Handles creation and validation of training configurations using Pydantic.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..models.training_parameters import TrainingParameters, TrainingParametersManager

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """Builds and validates training configurations using Pydantic."""
    
    def __init__(self, training_config: Dict[str, Any]):
        """
        Initialize config builder.
        
        Args:
            training_config: Raw training configuration dictionary
        """
        self.raw_config = training_config
        
    def build_training_config(self, train_type: str) -> Dict[str, Any]:
        """
        Build validated training configuration using Pydantic.

        Args:
            train_type: Training type ('embedding' or 'reranker')

        Returns:
            Validated training configuration dictionary
        """
        logger.info(f"构建 {train_type} 训练配置")

        # 准备配置数据
        config_data = self.raw_config.copy()

        # 确保输出目录存在（因为官方默认是None，我们需要设置默认值）
        if not config_data.get("output_dir"):
            config_data["output_dir"] = self._generate_default_output_dir(train_type)



        # 处理SwanLab配置
        config_data = self._handle_swanlab_config(config_data)

        # 使用升级后的 TrainingParametersManager 进行验证和转换
        try:
            param_manager = TrainingParametersManager()
            param_manager.load_from_config(config_data)

            # 获取官方训练参数（会传给 *TrainingArguments）
            training_args_dict = param_manager.get_training_args_dict()

            logger.info(f"训练配置构建完成: {len(training_args_dict)} 个官方训练参数")
            return training_args_dict

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            # 如果有详细错误信息，记录下来
            if hasattr(e, 'errors'):
                for error in e.errors():
                    logger.error(f"配置错误: {error.get('loc', [])} - {error.get('msg', '')} (输入值: {error.get('input', 'N/A')})")
            raise ValueError(f"训练配置验证失败: {e}") from e


    def _generate_default_output_dir(self, train_type: str) -> str:
        """Generate default output directory."""
        model_name = self.raw_config.get("model_name_or_path", "unknown-model")
        model_name = model_name.replace("/", "-")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return f"output/training_{train_type}_{model_name}_{timestamp}"
    

    def _handle_swanlab_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SwanLab configuration to prevent initialization errors."""
        report_to = config.get('report_to', 'none')
        logger.info(f"🔍 SwanLab配置检查 - report_to: {report_to}, 类型: {type(report_to)}")

        # 检查是否启用了SwanLab
        if isinstance(report_to, str) and 'swanlab' in report_to.lower():
            # 检查是否有API密钥
            swanlab_api_key = self.raw_config.get('swanlab_api_key', '')
            if not swanlab_api_key:
                logger.warning("SwanLab已配置但缺少API密钥，将禁用SwanLab")
                # 从report_to中移除swanlab
                if report_to == 'swanlab':
                    config['report_to'] = 'none'
                else:
                    # 处理多个报告工具的情况
                    report_tools = [tool.strip() for tool in report_to.split(',')]
                    filtered_tools = [tool for tool in report_tools if 'swanlab' not in tool.lower()]
                    config['report_to'] = ','.join(filtered_tools) if filtered_tools else 'none'

                logger.info(f"SwanLab已从report_to中移除，当前report_to: {config['report_to']}")

        return config

    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return {
            'dataset_name_or_path': self.raw_config.get('dataset_name_or_path'),
            'HF_subset': self.raw_config.get('HF_subset'),
            'train_sample_size': self.raw_config.get('train_sample_size', -1),
            'eval_sample_size': self.raw_config.get('eval_sample_size', -1),
            'test_sample_size': self.raw_config.get('test_sample_size', -1)
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-related configuration."""
        return {
            'model_name_or_path': self.raw_config.get('model_name_or_path'),
            'train_type': self.raw_config.get('train_type', 'embedding').lower()
        }