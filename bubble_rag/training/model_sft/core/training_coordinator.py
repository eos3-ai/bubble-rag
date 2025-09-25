"""
Training coordinator for orchestrating the training process.

Manages the overall training workflow and delegates to appropriate trainers.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Callable
from .base_trainer import TrainingResult

logger = logging.getLogger(__name__)


class TrainingCoordinator:
    """
    Coordinates the training process and manages trainer selection.
    
    This class serves as the main entry point for training, handling
    trainer selection and orchestrating the overall training workflow.
    """
    
    def __init__(self, training_config: Dict[str, Any]):
        """
        Initialize training coordinator.
        
        Args:
            training_config: Complete training configuration dictionary
        """
        self.config = training_config
        self.train_type = training_config.get('train_type', 'embedding').lower()
        
        logger.info(f"初始化训练协调器，训练类型: {self.train_type}")
    
    def execute_training(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """
        Execute the complete training workflow.

        Args:
            progress_callback: Optional callback function for progress reporting

        Returns:
            TrainingResult containing the trained model and metadata

        Raises:
            ValueError: If train_type is not supported
        """
        logger.info(f"🎯 开始执行 {self.train_type} 训练")

        # Validate training type
        if self.train_type not in ['embedding', 'reranker']:
            raise ValueError(f"不支持的训练类型: {self.train_type}，仅支持 'embedding' 或 'reranker'")

        # Clean GPU environment before training
        from .device_manager import DeviceManager
        DeviceManager.cleanup_gpu_environment()

        # Create appropriate trainer
        trainer = self._create_trainer()

        # Execute training
        try:
            result = trainer.train(progress_callback)
            logger.info(f"🎉 训练协调器完成 {self.train_type} 训练")
            return result
        except Exception as e:
            logger.error(f"💥 训练协调器执行失败: {e}")
            raise
    
    def _create_trainer(self):
        """Create and return the appropriate trainer instance."""
        if self.train_type == 'embedding':
            from ..trainers.embedding_trainer import EmbeddingTrainer
            return EmbeddingTrainer(self.config)
        elif self.train_type == 'reranker':
            from ..trainers.reranker_trainer import RerankerTrainer
            return RerankerTrainer(self.config)
        else:
            # This should never happen due to validation above
            raise ValueError(f"无法创建训练器，未知类型: {self.train_type}")
    
    def validate_configuration(self) -> bool:
        """
        Validate the training configuration.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = [
            'train_type',
            'model_name_or_path',
            'dataset_name_or_path'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in self.config or not self.config[field]:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"缺少必需的配置字段: {missing_fields}")
        
        # Validate train_type
        if self.config['train_type'].lower() not in ['embedding', 'reranker']:
            raise ValueError(f"无效的训练类型: {self.config['train_type']}")
        
        logger.info("✅ 训练配置验证通过")
        return True
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training configuration.
        
        Returns:
            Dictionary containing training summary information
        """
        return {
            'train_type': self.train_type,
            'model_name': self.config.get('model_name_or_path'),
            'dataset_path': self.config.get('dataset_name_or_path'),
            'output_dir': self.config.get('output_dir'),
            'num_epochs': self.config.get('num_train_epochs', 'unknown'),
            'batch_size': self.config.get('per_device_train_batch_size', 'unknown'),
            'learning_rate': self.config.get('learning_rate', 'unknown')
        }