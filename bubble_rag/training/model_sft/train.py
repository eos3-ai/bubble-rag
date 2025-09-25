"""
Unified training orchestrator for bubble_rag.

This lightweight orchestrator delegates training to specialized trainers
while maintaining backward compatibility with the existing API.
"""

import logging
from typing import Dict, Any, Optional, Callable, Tuple
from dotenv import load_dotenv

from .core.training_coordinator import TrainingCoordinator
from .utils.common_utils import init_swanlab

logger = logging.getLogger(__name__)


def main(progress_callback: Optional[Callable] = None, training_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    """
    Main training function with refactored architecture.
    
    Args:
        progress_callback: Optional progress callback function
        training_config: Training configuration dictionary containing:
            - train_type: Training type ('embedding' or 'reranker')
            - model_name_or_path: Model name or path
            - dataset_name_or_path: Dataset name or path  
            - output_dir: Output directory
            - num_train_epochs: Number of training epochs
            - per_device_train_batch_size: Batch size per device
            - learning_rate: Learning rate
            - ... other training parameters
            
    Returns:
        Tuple of (trained_model, save_directory)
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
    """
    logger.info("正在执行统一训练脚本 - 重构版本")
    
    # Load environment variables
    load_dotenv()
    
    # Validate input
    if training_config is None:
        raise ValueError("training_config 不能为空")
    
    # Get task ID for global exception handling
    task_id = training_config.get("task_id")
    
    try:
        # Initialize SwanLab if configured
        if training_config.get('swanlab_api_key'):
            logger.info("初始化 SwanLab")
            try:
                init_swanlab(training_config=training_config)
            except Exception as e:
                logger.warning(f"SwanLab 初始化失败，继续训练: {e}")
        
        # Create training coordinator
        coordinator = TrainingCoordinator(training_config)
        
        # Validate configuration
        coordinator.validate_configuration()
        
        # Log training summary
        summary = coordinator.get_training_summary()
        logger.info("训练配置摘要:")
        for key, value in summary.items():
            logger.info(f"   {key}: {value}")
        
        # Execute training
        logger.info("开始执行训练")
        result = coordinator.execute_training(progress_callback)
        
        # Log success
        logger.info(f"训练完成！模型保存至: {result.save_dir}")
        
        return result.model, result.save_dir
        
    except Exception as e:
        error_msg = f"训练执行失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Handle task status update if task_id is available
        if task_id:
            try:
                from .services.task_manager import task_manager
                from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
                from .enums import TrainingStatus
                
                task_manager.fail_task(task_id, error_msg, None)
                training_task_service.update_task_status(task_id, TrainingStatus.FAILED.value)
                training_task_service.update_task_result(task_id, error_message=error_msg)
                
            except Exception as update_error:
                logger.error(f"更新失败状态时出错: {update_error}")
        
        # Re-raise the original exception
        raise RuntimeError(error_msg) from e


def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    coordinator = TrainingCoordinator(config)
    return coordinator.validate_configuration()


def get_supported_training_types() -> list[str]:
    """
    Get list of supported training types.
    
    Returns:
        List of supported training type strings
    """
    return ['embedding', 'reranker']


def create_default_config(train_type: str, model_name: str, dataset_path: str) -> Dict[str, Any]:
    """
    Create default training configuration.
    
    Args:
        train_type: Training type ('embedding' or 'reranker')
        model_name: Model name or path
        dataset_path: Dataset path
        
    Returns:
        Default configuration dictionary
    """
    if train_type not in get_supported_training_types():
        raise ValueError(f"不支持的训练类型: {train_type}")
    
    return {
        'train_type': train_type,
        'model_name_or_path': model_name,
        'dataset_name_or_path': dataset_path,
        'num_train_epochs': 3,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'logging_steps': 10,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'bf16': False,
        'fp16': False,
    }


if __name__ == "__main__":
    # Example usage
    example_config = {
        'train_type': 'embedding',
        'model_name_or_path': 'distilbert-base-uncased',
        'dataset_name_or_path': 'sentence-transformers/all-nli',
        'output_dir': './output/test',
        'num_train_epochs': 1,
        'per_device_train_batch_size': 8,
        'learning_rate': 2e-5,
    }
    
    try:
        model, save_dir = main(training_config=example_config)
        print(f"训练成功完成！模型保存在: {save_dir}")
    except Exception as e:
        print(f"训练失败: {e}")