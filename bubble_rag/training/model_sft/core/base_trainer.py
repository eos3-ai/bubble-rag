"""
Abstract base trainer for all training types.

Defines the common interface and shared functionality for embedding and reranker training.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .device_manager import DeviceManager
from .config_builder import ConfigBuilder
from ..utils.data_loader import DataLoader
from ..utils.evaluation import UnifiedEvaluator
from ..models.training_parameters import TrainingParametersManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Training result container."""
    model: Any
    save_dir: str
    final_metrics: Optional[Dict[str, float]] = None
    training_stats: Optional[Dict[str, Any]] = None


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Defines the common training pipeline and abstract methods that 
    must be implemented by specific trainer types.
    """
    
    def __init__(self, training_config: Dict[str, Any]):
        """
        Initialize base trainer.
        
        Args:
            training_config: Complete training configuration dictionary
        """
        self.raw_config = training_config
        self.config_builder = ConfigBuilder(training_config)
        self.device_manager = DeviceManager()
        self.data_loader = None
        self.evaluator_factory = None
        
        # Extract common configurations
        self.model_config = self.config_builder.get_model_config()
        self.data_config = self.config_builder.get_data_config()
        self.train_type = self.model_config['train_type']

        # Cache for target column to avoid repeated computation
        self._cached_target_column = None

        logger.info(f"初始化 {self.train_type} 训练器")
    
    def train(self, progress_callback=None) -> TrainingResult:
        """
        Execute the complete training pipeline.
        
        Args:
            progress_callback: Optional progress callback function
            
        Returns:
            TrainingResult containing model and metadata
        """
        logger.info(f"开始 {self.train_type} 训练")
        task_id = self.raw_config.get('task_id')

        try:
            # Step 1: Initialize components
            self._initialize_components()
            
            # Step 2: Load and prepare data
            datasets = self._load_datasets()

            # Step 2.5: Record dataset information to database
            self._record_dataset_info(datasets)

            # Step 3: Initialize model and loss
            model, loss = self._initialize_model_and_loss(datasets['train'])

            # Step 3.5: Update loss function information in database
            self._update_loss_function_info(datasets, loss)

            # Step 3.6: Update model information in database
            self._update_model_info(model)

            # Step 4: Create training configuration
            training_args = self._create_training_args()
            
            # Step 5: Create evaluators (matching original train.py logic)
            evaluators = self._create_evaluators_from_datasets(datasets)
            evaluator = evaluators.get('dev')

            # Step 5.5: Update evaluator information in database
            self._update_evaluator_info(datasets, evaluators)

            # Step 6: Create trainer instance
            trainer = self.create_trainer_instance(
                model=model,
                args=training_args,
                train_dataset=datasets['train'],
                eval_dataset=datasets.get('eval'),
                loss=loss,
                evaluator=evaluator
            )
            
            # Step 7: Pre-training baseline evaluation
            self._perform_baseline_evaluation(model, datasets, evaluators)

            # Step 8: Show Tensorboard logging info (matching original train.py)
            from .base_trainer_tensorboard import show_tensorboard_info
            show_tensorboard_info(self, training_args)

            # Step 9: Update task status to RUNNING - 真正开始训练
            self._update_task_status_running(task_id)

            # Step 10: Execute training
            result = self._execute_training(trainer, model, progress_callback)

            # Step 11: 训练完成，设置为PENDING状态进行最终评估
            self._update_task_status_post_training_evaluation(task_id)

            # Step 12: Post-training final evaluation
            self._perform_final_evaluation(result.model, datasets, evaluators)

            # Step 13: Update task status to SUCCEEDED (matching original train.py)
            self._update_task_status_succeeded(task_id, result.save_dir)

            logger.info(f"{self.train_type} 训练完成")
            return result

        except Exception as e:
            # Update task status to FAILED (matching original train.py)
            self._update_task_status_failed(task_id, str(e))
            logger.error(f"{self.train_type} 训练失败: {e}")
            raise
        finally:
            # 最终保险：无论成功失败都确保释放GPU资源
            if task_id:
                try:
                    from ..utils.gpu_resource_manager import gpu_resource_manager
                    success = gpu_resource_manager.release_gpus_for_task(task_id)
                    if success:
                        logger.info(f"Finally块：确保释放任务 {task_id} 的GPU资源")
                    else:
                        logger.warning(f"Finally块：常规GPU释放失败，尝试强制释放")
                        gpu_resource_manager.force_release_gpu_for_task(task_id)
                except Exception as e:
                    logger.critical(f"严重错误：Finally块GPU资源释放失败！尝试强制恢复。任务: {task_id}, 错误: {e}")
                    try:
                        from ..utils.gpu_resource_manager import gpu_resource_manager
                        gpu_resource_manager.force_release_gpu_for_task(task_id)
                        logger.warning(f"强制GPU释放已执行，请检查资源状态")
                    except Exception as force_error:
                        logger.critical(f"强制GPU释放也失败！任务 {task_id} 的GPU资源可能永久泄漏: {force_error}")
    
    def _initialize_components(self):
        """Initialize data loader and evaluator factory."""
        self.data_loader = DataLoader(
            hf_subset=self.data_config.get('HF_subset'),
            train_sample_size=self.data_config.get('train_sample_size', -1),
            eval_sample_size=self.data_config.get('eval_sample_size', -1),
            test_sample_size=self.data_config.get('test_sample_size', -1)
        )
        
        self.evaluator_factory = UnifiedEvaluator(self.train_type)
    
    def _load_datasets(self) -> Dict[str, Any]:
        """Load and return datasets."""
        dataset_path = self.data_config['dataset_name_or_path']
        logger.info(f"加载数据集: {dataset_path}")
        
        train_dataset, eval_dataset, test_dataset = self.data_loader.load_all_splits(dataset_path)
        
        if train_dataset is None:
            raise ValueError(f"训练数据集加载失败: {dataset_path}")
        
        return {
            'train': train_dataset,
            'eval': eval_dataset,
            'test': test_dataset
        }
    
    def _initialize_model_and_loss(self, train_dataset) -> Tuple[Any, Any]:
        """Initialize model and loss function."""
        model_name = self.model_config['model_name_or_path']
        
        # Initialize model
        model = self.initialize_model(model_name)
        
        # Prepare model for training
        user_device = self.raw_config.get('device')
        device = self.device_manager.get_training_device(user_device)
        model = self.device_manager.prepare_model_for_training(model, device)
        
        # Create loss function
        loss = self.create_loss_function(model, train_dataset)
        
        return model, loss
    
    def _check_gpu_bf16_compatibility(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查GPU的bf16兼容性并自动调整配置

        Args:
            config: 训练配置字典

        Returns:
            Dict[str, Any]: 调整后的配置
        """
        config_copy = config.copy()

        if config_copy.get('bf16', False):
            logger.info("检测到bf16=True，开始GPU兼容性检查")

            try:
                import torch
                if torch.cuda.is_available():
                    # 检查GPU是否支持bf16
                    device = torch.cuda.current_device()
                    gpu_capability = torch.cuda.get_device_capability(device)

                    # Ampere架构（计算能力8.0+）才完全支持bf16
                    supports_bf16 = gpu_capability[0] >= 8

                    if not supports_bf16:
                        logger.warning(f"GPU计算能力 {gpu_capability[0]}.{gpu_capability[1]} 不支持bf16，自动降级为fp16")
                        config_copy['bf16'] = False
                        config_copy['fp16'] = True
                    else:
                        logger.info(f"GPU计算能力 {gpu_capability[0]}.{gpu_capability[1]} 支持bf16")
                else:
                    logger.warning("CUDA不可用，bf16已禁用")
                    config_copy['bf16'] = False
                    config_copy['fp16'] = False

            except Exception as gpu_check_error:
                logger.warning(f"GPU兼容性检查失败，禁用bf16: {gpu_check_error}")
                config_copy['bf16'] = False
                config_copy['fp16'] = True

        return config_copy

    def _create_training_args(self) -> Any:
        """Create training arguments."""
        config = self.config_builder.build_training_config(self.train_type)

        # 调试：记录关键参数
        logger.info(f"[DEBUG] 训练配置关键参数: metric_for_best_model={config.get('metric_for_best_model')}, "
                   f"load_best_model_at_end={config.get('load_best_model_at_end')}, "
                   f"greater_is_better={config.get('greater_is_better')}")

        # 兼容性修复：如果metric_for_best_model设置为不存在的eval_loss，自动修正
        if config.get('metric_for_best_model') == 'eval_loss':
            logger.warning("检测到过时的metric_for_best_model='eval_loss'，自动修正为'eval_sequential_score'")
            config['metric_for_best_model'] = 'eval_sequential_score'
            config['greater_is_better'] = True

        # 🔧 GPU兼容性检测：检查bf16支持
        config = self._check_gpu_bf16_compatibility(config)

        return self.create_training_args(config)
    
    def _create_evaluators_from_datasets(self, datasets) -> Dict[str, Any]:
        """Create evaluators from datasets (matching original train.py logic)."""
        # Get target column using the same logic as original train.py
        target_column = self._get_target_column(datasets)

        # Get run name (model name for evaluator naming)
        model_name = self.model_config.get('model_name_or_path', 'unknown')
        short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
        run_name = f"{self.train_type}-{short_model_name}"

        # 获取数据源映射
        data_source_mapping = self._get_data_source_mapping(datasets['train'])
        # 保存到实例变量，供后续分离逻辑使用
        self._current_data_source_mapping = data_source_mapping
        logger.info(f"[DEBUG] 设置_current_data_source_mapping: {data_source_mapping}")

        evaluators = {}

        # 创建验证评估器（使用data_source_mapping确保source_id命名）
        if datasets.get('eval') is not None:
            if self.data_loader.is_multi_dataset(datasets['eval']):
                # 多数据集：使用create_multi_evaluator并传递data_source_mapping
                evaluators['dev'] = self.evaluator_factory.create_multi_evaluator(
                    datasets['eval'], target_column, f"{run_name}-eval", data_source_mapping
                )
                logger.info(f"创建多数据集验证评估器成功，使用source_id命名: {run_name}-eval")
            else:
                # 单数据集：使用source_id作为评估器名称确保命名一致
                dataset_name = list(datasets['eval'].keys())[0] if isinstance(datasets['eval'], dict) else 'default'
                evaluator_name = data_source_mapping.get(dataset_name, "1") if data_source_mapping else "1"
                evaluators['dev'] = self.evaluator_factory.create_evaluator(
                    datasets['eval'], target_column, evaluator_name
                )
                logger.info(f"创建单数据集验证评估器成功，使用source_id命名: {evaluator_name}")

        # 创建测试评估器（使用data_source_mapping确保source_id命名）
        if datasets.get('test') is not None:
            if self.data_loader.is_multi_dataset(datasets['test']):
                # 多数据集：使用create_multi_evaluator并传递data_source_mapping
                evaluators['test'] = self.evaluator_factory.create_multi_evaluator(
                    datasets['test'], target_column, f"{run_name}-test", data_source_mapping
                )
                logger.info(f"创建多数据集测试评估器成功，使用source_id命名: {run_name}-test")
            else:
                # 单数据集：使用source_id作为评估器名称确保命名一致
                dataset_name = list(datasets['test'].keys())[0] if isinstance(datasets['test'], dict) else 'default'
                evaluator_name = data_source_mapping.get(dataset_name, "1") if data_source_mapping else "1"
                evaluators['test'] = self.evaluator_factory.create_evaluator(
                    datasets['test'], target_column, evaluator_name
                )
                logger.info(f"创建单数据集测试评估器成功，使用source_id命名: {evaluator_name}")

        return evaluators

    def _create_evaluator(self, datasets) -> Optional[Any]:
        """Create evaluator if evaluation dataset exists."""
        if datasets.get('eval') is None:
            logger.info("没有评估数据集，跳过评估器创建")
            return None

        return self.create_evaluator(datasets['eval'])

    def _record_dataset_info(self, datasets: Dict[str, Any]):
        """Record dataset information to database (matching original train.py)."""
        task_id = self.raw_config.get('task_id')
        if not task_id:
            logger.warning("缺少task_id，跳过数据集信息记录")
            return

        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService

            # 获取数据集路径和配置
            dataset_name_or_path = self.data_config.get('dataset_name_or_path', 'unknown')

            # 获取目标列名（使用与train.py相同的逻辑）
            target_column = self._get_target_column(datasets)

            # 生成数据源映射 (完全匹配原版train.py的逻辑)
            data_source_mapping = {}
            train_dataset = datasets.get('train')

            if self.data_loader.is_multi_dataset(train_dataset) and isinstance(train_dataset, dict):
                for idx, base_name in enumerate(train_dataset.keys()):
                    data_source_mapping[base_name] = self._generate_data_source_id(idx, base_name)
            else:
                # 单数据集：使用固定的数据源ID和基础名称
                base_name = self.data_loader._extract_dataset_base_name(dataset_name_or_path)
                data_source_mapping[base_name] = "1"  # 原版train.py的固定值

            # 记录训练数据集
            if datasets.get('train'):
                self._record_split_dataset_info(
                    TrainingDatasetService, task_id, datasets['train'], 'train',
                    dataset_name_or_path, data_source_mapping, target_column
                )

            # 记录验证数据集
            if datasets.get('eval'):
                self._record_split_dataset_info(
                    TrainingDatasetService, task_id, datasets['eval'], 'eval',
                    dataset_name_or_path, data_source_mapping, target_column
                )

            # 记录测试数据集
            if datasets.get('test'):
                self._record_split_dataset_info(
                    TrainingDatasetService, task_id, datasets['test'], 'test',
                    dataset_name_or_path, data_source_mapping, target_column
                )

            logger.info("数据集信息记录成功")

        except Exception as e:
            logger.warning(f"记录数据集信息失败（不影响训练）: {e}")

    def _record_split_dataset_info(self, TrainingDatasetService, task_id: str, dataset: Any,
                                   split_type: str, dataset_name_or_path: str,
                                   data_source_mapping: Dict[str, str], target_column: str):
        """Record dataset information for a specific split (matching original train.py)."""

        # 获取样本大小配置
        sample_size_key = f'{split_type}_sample_size'
        global_configured_sample_size = self.data_config.get(sample_size_key, -1)

        if self.data_loader.is_multi_dataset(dataset):
            # 多数据集：每个数据源分配独立的ID
            for base_name, ds in dataset.items():
                data_source_id = data_source_mapping[base_name]  # 使用统一映射

                # 按照原版train.py逻辑：直接保存用户配置值，0表示无限制
                configured_sample_size = global_configured_sample_size

                TrainingDatasetService.record_dataset_info(
                    task_id=task_id,
                    data_source_id=data_source_id,
                    dataset_name=base_name,  # 去掉split后缀，只保存基础名称
                    dataset_base_name=base_name,
                    dataset_path=dataset_name_or_path,
                    dataset_type="auto",
                    split_type=split_type,
                    dataset=ds,
                    target_column="auto",  # 让record_dataset_info自动识别
                    loss_function=None,
                    evaluator=None,
                    hf_subset=self.data_config.get('HF_subset'),  # HF_subset配置
                    configured_sample_size=configured_sample_size  # 修复：使用各数据集的实际大小
                )
        else:
            # 单数据集：使用固定的数据源ID
            base_name = self.data_loader._extract_dataset_base_name(dataset_name_or_path)
            data_source_id = self._generate_data_source_id(0, base_name)

            # 按照原版train.py逻辑：直接保存用户配置值，0表示无限制
            configured_sample_size = global_configured_sample_size

            TrainingDatasetService.record_dataset_info(
                task_id=task_id,
                data_source_id=data_source_id,
                dataset_name=base_name,  # 去掉split后缀，只保存基础名称
                dataset_base_name=base_name,
                dataset_path=dataset_name_or_path,
                dataset_type="auto",
                split_type=split_type,
                dataset=dataset,
                target_column="auto",  # 让record_dataset_info自动识别
                loss_function=None,
                evaluator=None,
                hf_subset=self.data_config.get('HF_subset'),  # HF_subset配置
                configured_sample_size=configured_sample_size  # 修复：使用实际大小
            )

    def _generate_data_source_id(self, index: int, base_name: str) -> str:
        """Generate data source ID (matching original train.py logic)."""
        return str(index + 1)  # Original train.py uses simple "1", "2", "3"

    def _get_target_column(self, datasets: Dict[str, Any]) -> str:
        """Get target column for this training type (matching original train.py logic)."""
        # Use cached value if available
        if self._cached_target_column is not None:
            return self._cached_target_column

        train_dataset = datasets['train']

        # Use data_loader to get the initial target column
        target_column = self.data_loader.get_target_column(train_dataset)

        # Standardize based on data type (matching train.py logic)
        sample_value = None
        try:
            if self.data_loader.is_multi_dataset(train_dataset):
                # For multi-dataset, check first dataset
                first_dataset_name = next(iter(train_dataset.keys()))
                first_dataset = train_dataset[first_dataset_name]
                if target_column in first_dataset.column_names and len(first_dataset) > 0:
                    # Access underlying arrow table to avoid formatting issues
                    sample_value = first_dataset.data.column(target_column).to_pylist()[0] if hasattr(first_dataset, 'data') and first_dataset.data else None
            else:
                # For single dataset
                if target_column in train_dataset.column_names and len(train_dataset) > 0:
                    # Access underlying arrow table to avoid formatting issues
                    sample_value = train_dataset.data.column(target_column).to_pylist()[0] if hasattr(train_dataset, 'data') and train_dataset.data else None
        except Exception as e:
            logger.warning(f"获取样本值失败，使用默认列名: {e}")
            # Fall back to reranker/embedding type-based default
            if self.train_type == "embedding":
                target_column = "score"
            else:  # reranker
                target_column = "label"
            sample_value = None

        if sample_value is not None:
            if isinstance(sample_value, int) or (isinstance(sample_value, float) and sample_value.is_integer()):
                target_column = "label"
            else:
                target_column = "score"

        # Cache the result
        self._cached_target_column = target_column
        logger.info(f"目标列名已更新为: {target_column}")
        return target_column

    def _update_loss_function_info(self, datasets: Dict[str, Any], loss):
        """Update loss function information in database (matching original train.py)."""
        task_id = self.raw_config.get('task_id')
        if not task_id:
            logger.warning("缺少task_id，跳过损失函数信息更新")
            return

        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService

            # 获取数据源映射（复用 _record_dataset_info 中的逻辑）
            data_source_mapping = self._get_data_source_mapping(datasets['train'])

            # 检查是否为多数据集场景
            train_dataset = datasets['train']
            is_multi_dataset = self.data_loader.is_multi_dataset(train_dataset)

            if is_multi_dataset and isinstance(loss, dict):
                # 多数据集情况：为每个数据源更新对应的损失函数
                for dataset_name, loss_func in loss.items():
                    actual_loss_name = type(loss_func).__name__
                    data_source_id = data_source_mapping.get(dataset_name, f"ds_{hash(dataset_name) % 1000:03d}")
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="train",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新多数据集损失函数: {data_source_id}-train -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新数据源 {data_source_id} 训练集损失函数失败: {e}")

                # 多数据集情况：验证集和测试集使用对应同名训练数据集的损失函数
                if datasets.get('eval') and isinstance(datasets['eval'], dict):
                    for dataset_name in datasets['eval'].keys():
                        data_source_id = data_source_mapping.get(dataset_name, f"ds_{hash(dataset_name) % 1000:03d}")
                        if dataset_name in loss:
                            corresponding_loss_name = type(loss[dataset_name]).__name__
                            try:
                                TrainingDatasetService.update_dataset_loss_function_by_source(
                                    task_id=task_id,
                                    data_source_id=data_source_id,
                                    split_type="eval",
                                    loss_function=corresponding_loss_name
                                )
                                logger.info(f"更新多验证数据集损失函数: {data_source_id}-eval -> {corresponding_loss_name}")
                            except Exception as e:
                                logger.warning(f"更新验证数据源 {data_source_id} 损失函数失败: {e}")

                if datasets.get('test') and isinstance(datasets['test'], dict):
                    for dataset_name in datasets['test'].keys():
                        data_source_id = data_source_mapping.get(dataset_name, f"ds_{hash(dataset_name) % 1000:03d}")
                        if dataset_name in loss:
                            corresponding_loss_name = type(loss[dataset_name]).__name__
                            try:
                                TrainingDatasetService.update_dataset_loss_function_by_source(
                                    task_id=task_id,
                                    data_source_id=data_source_id,
                                    split_type="test",
                                    loss_function=corresponding_loss_name
                                )
                                logger.info(f"更新多测试数据集损失函数: {data_source_id}-test -> {corresponding_loss_name}")
                            except Exception as e:
                                logger.warning(f"更新测试数据源 {data_source_id} 损失函数失败: {e}")

            else:
                # 单数据集情况：直接获取损失函数类名
                actual_loss_name = type(loss).__name__
                logger.info(f"更新单数据集损失函数信息: {actual_loss_name}")

                # 单数据集使用固定的数据源ID
                data_source_id = "1"

                # 更新训练数据集的损失函数
                if datasets.get('train'):
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="train",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新单训练数据集损失函数: {data_source_id}-train -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新训练数据集损失函数失败: {e}")

                # 单数据集情况：验证集和测试集也使用相同的损失函数
                if datasets.get('eval'):
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="eval",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新单验证数据集损失函数: {data_source_id}-eval -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新验证数据集损失函数失败: {e}")

                if datasets.get('test'):
                    try:
                        TrainingDatasetService.update_dataset_loss_function_by_source(
                            task_id=task_id,
                            data_source_id=data_source_id,
                            split_type="test",
                            loss_function=actual_loss_name
                        )
                        logger.info(f"更新单测试数据集损失函数: {data_source_id}-test -> {actual_loss_name}")
                    except Exception as e:
                        logger.warning(f"更新测试数据集损失函数失败: {e}")

        except Exception as e:
            logger.warning(f"更新数据集损失函数信息失败: {e}")

    def _get_data_source_mapping(self, train_dataset) -> Dict[str, str]:
        """Get data source mapping (extracted from _record_dataset_info)."""
        data_source_mapping = {}
        dataset_name_or_path = self.data_config.get('dataset_name_or_path', 'unknown')

        if self.data_loader.is_multi_dataset(train_dataset) and isinstance(train_dataset, dict):
            for idx, base_name in enumerate(train_dataset.keys()):
                data_source_mapping[base_name] = self._generate_data_source_id(idx, base_name)
        else:
            # 单数据集：使用固定的数据源ID和基础名称
            base_name = self.data_loader._extract_dataset_base_name(dataset_name_or_path)
            data_source_mapping[base_name] = self._generate_data_source_id(0, base_name)  # 保持与_record_dataset_info一致

        return data_source_mapping

    def _update_model_info(self, model):
        """Update model information in database (matching original train.py)."""
        task_id = self.raw_config.get('task_id')
        if not task_id:
            logger.warning("缺少task_id，跳过模型信息更新")
            return

        try:
            from bubble_rag.training.model_sft.services.task_manager import task_manager

            model_name = self.model_config.get('model_name_or_path', 'unknown')

            # 构建模型信息（完全匹配原版train.py）
            model_info_update = {
                "validation": {
                    "valid": True,
                    "message": "模型加载成功",
                    "details": {
                        "type": "validated",
                        "name": model_name
                    }
                },
                "recommended_training_types": [self.train_type],
                "compatibility": {
                    "supported": True,
                    "model_type": "loaded",
                    "notes": ["模型已成功加载"]
                }
            }

            # 获取模型维度（支持embedding和reranker模型）
            embedding_dimension = None
            if self.train_type == "embedding" and hasattr(model, 'get_sentence_embedding_dimension'):
                try:
                    embedding_dimension = model.get_sentence_embedding_dimension()
                    logger.info(f"获取到embedding模型维度: {embedding_dimension}")
                except Exception as dim_e:
                    logger.warning(f"获取embedding模型维度失败: {str(dim_e)}")
            elif self.train_type == "reranker":
                # 对于reranker模型，尝试多种方法获取维度
                try:
                    # 方法1: 通过模型的tokenizer和config获取hidden_size
                    if hasattr(model, 'model') and hasattr(model.model, 'config') and hasattr(model.model.config, 'hidden_size'):
                        embedding_dimension = model.model.config.hidden_size
                        logger.info(f"获取到reranker模型维度 (方法1): {embedding_dimension}")
                    # 方法2: 通过encode方法测试获取维度
                    elif hasattr(model, 'encode'):
                        test_texts = ["test"]
                        try:
                            # 某些reranker模型的encode方法返回embedding
                            test_embedding = model.encode(test_texts)
                            if hasattr(test_embedding, 'shape') and len(test_embedding.shape) > 1:
                                embedding_dimension = test_embedding.shape[1]
                                logger.info(f"获取到reranker模型维度 (方法2): {embedding_dimension}")
                        except:
                            pass
                    # 方法3: 检查是否有classifier层来推断维度
                    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
                        embedding_dimension = model.classifier.in_features
                        logger.info(f"获取到reranker模型维度 (方法3): {embedding_dimension}")

                    if not embedding_dimension:
                        logger.info("无法自动获取reranker模型维度，将使用默认值或跳过")

                except Exception as dim_e:
                    logger.warning(f"获取reranker模型维度失败: {str(dim_e)}")

            # 如果获取到了维度，添加到模型信息中
            if embedding_dimension:
                model_info_update["embedding_dimension"] = embedding_dimension

            # 更新到数据库
            task_manager.update_model_info_after_loading(task_id, model_info_update)
            logger.info("模型信息更新成功")

        except Exception as update_e:
            logger.warning(f"更新模型信息到数据库失败，不影响训练继续: {str(update_e)}")

    def _update_evaluator_info(self, datasets: Dict[str, Any], evaluators: Dict[str, Any]):
        """Update evaluator information in database (matching original train.py)."""
        task_id = self.raw_config.get('task_id')
        if not task_id:
            logger.warning("缺少task_id，跳过评估器信息更新")
            return

        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService

            # 获取数据源映射
            data_source_mapping = self._get_data_source_mapping(datasets['train'])

            # 更新验证数据集的评估器类型
            if datasets.get('eval') and evaluators.get('dev'):
                eval_dataset = datasets['eval']
                evaluator = evaluators['dev']

                # 检查是否为多数据集
                if self.data_loader.is_multi_dataset(eval_dataset):
                    # 多数据集情况：从SequentialEvaluator中提取子评估器
                    if isinstance(eval_dataset, dict) and hasattr(evaluator, 'evaluators'):
                        # SequentialEvaluator包含多个子评估器
                        dataset_names = list(eval_dataset.keys())
                        sub_evaluators = evaluator.evaluators

                        for dataset_name, sub_evaluator in zip(dataset_names, sub_evaluators):
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    sub_evaluator_name = type(sub_evaluator).__name__

                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="eval",
                                        evaluator=sub_evaluator_name
                                    )
                                    logger.info(f"更新多数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {sub_evaluator_name}")
                                else:
                                    logger.warning(f"找不到数据集 {dataset_name} 的数据源映射")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                    else:
                        # 不是SequentialEvaluator，所有数据集使用同一个评估器
                        evaluator_name = type(evaluator).__name__
                        for dataset_name in eval_dataset.keys():
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="eval",
                                        evaluator=evaluator_name
                                    )
                                    logger.info(f"更新多数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                else:
                    # 单数据集情况：为该数据源更新评估器类型
                    evaluator_name = type(evaluator).__name__
                    try:
                        # 单数据集也要通过data_source_id更新
                        for dataset_name, data_source_id in data_source_mapping.items():
                            TrainingDatasetService.update_dataset_evaluator_by_source(
                                task_id=task_id,
                                data_source_id=data_source_id,
                                split_type="eval",
                                evaluator=evaluator_name
                            )
                            logger.info(f"更新单验证数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                    except Exception as e:
                        logger.warning(f"更新单验证数据集评估器类型失败: {e}")

            # 更新测试数据集的评估器类型
            if datasets.get('test') and evaluators.get('test'):
                test_dataset = datasets['test']
                evaluator = evaluators['test']

                # 检查是否为多数据集
                if self.data_loader.is_multi_dataset(test_dataset):
                    # 多数据集情况：从SequentialEvaluator中提取子评估器
                    if isinstance(test_dataset, dict) and hasattr(evaluator, 'evaluators'):
                        # SequentialEvaluator包含多个子评估器
                        dataset_names = list(test_dataset.keys())
                        sub_evaluators = evaluator.evaluators

                        for dataset_name, sub_evaluator in zip(dataset_names, sub_evaluators):
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    sub_evaluator_name = type(sub_evaluator).__name__

                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="test",
                                        evaluator=sub_evaluator_name
                                    )
                                    logger.info(f"更新多测试数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {sub_evaluator_name}")
                                else:
                                    logger.warning(f"找不到数据集 {dataset_name} 的数据源映射")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                    else:
                        # 不是SequentialEvaluator，所有数据集使用同一个评估器
                        evaluator_name = type(evaluator).__name__
                        for dataset_name in test_dataset.keys():
                            try:
                                if dataset_name in data_source_mapping:
                                    data_source_id = data_source_mapping[dataset_name]
                                    TrainingDatasetService.update_dataset_evaluator_by_source(
                                        task_id=task_id,
                                        data_source_id=data_source_id,
                                        split_type="test",
                                        evaluator=evaluator_name
                                    )
                                    logger.info(f"更新多测试数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                            except Exception as e:
                                logger.warning(f"更新数据集 {dataset_name} 评估器类型失败: {e}")
                else:
                    # 单数据集情况：为该数据源更新评估器类型
                    evaluator_name = type(evaluator).__name__
                    try:
                        # 单数据集也要通过data_source_id更新
                        for dataset_name, data_source_id in data_source_mapping.items():
                            TrainingDatasetService.update_dataset_evaluator_by_source(
                                task_id=task_id,
                                data_source_id=data_source_id,
                                split_type="test",
                                evaluator=evaluator_name
                            )
                            logger.info(f"更新单测试数据集评估器类型: {dataset_name} (源ID: {data_source_id}) -> {evaluator_name}")
                    except Exception as e:
                        logger.warning(f"更新单测试数据集评估器类型失败: {e}")

        except Exception as e:
            logger.warning(f"更新评估器信息失败: {e}")

    def _execute_training(self, trainer, model, progress_callback) -> TrainingResult:
        """Execute the actual training."""
        # 总是设置基础的task_manager进度跟踪（匹配原版train.py）
        task_id = self.raw_config.get('task_id')
        if task_id:
            try:
                from bubble_rag.training.model_sft.services.task_manager import task_manager
                task_manager.update_task_progress(task_id, 0.0, "开始训练")
            except Exception as tm_e:
                logger.warning(f"初始化task_manager进度失败: {tm_e}")

        # Setup progress callback if provided
        if progress_callback:
            # 修复：有外层进度回调时，只设置外层回调，避免双重更新
            self._setup_progress_callback(trainer, progress_callback)
            # 但仍需要设置loss数据收集
            self._setup_loss_collection(trainer)
        else:
            # 即使没有进度回调，也要设置loss数据收集和基础进度跟踪 (matching original train.py)
            self._setup_loss_collection(trainer)
            # 设置基础的progress跟踪（仅在无外层回调时使用）
            self._setup_basic_progress_tracking(trainer)

        # 保存训练元数据到loss文件（包含数据源映射等信息）
        self._save_training_metadata(trainer)

        # 确保评估器目录存在（匹配原版train.py逻辑）
        self._ensure_eval_directories(trainer)

        # Run training
        trainer.train()

        # 修复：训练完成时不使用硬编码的(1,1)，而是使用正确的步数
        # 训练真正完成时，进度应该接近100%，不需要额外的进度回调
        # 最终的100%进度由任务完成状态更新时设置
        logger.info("训练执行完成，跳过硬编码的训练完成进度回调")

        # 总是更新task_manager中的进度到99%（匹配原版train.py，100%由成功状态更新处理）
        task_id = self.raw_config.get('task_id')
        if task_id:
            try:
                from bubble_rag.training.model_sft.services.task_manager import task_manager
                task_manager.update_task_progress(task_id, 99.0, "保存模型")
                logger.info("训练完成，进度更新到99%")
            except Exception as tm_e:
                logger.warning(f"更新完成时task_manager进度失败: {tm_e}")

        # Save model
        save_dir = trainer.args.output_dir
        try:
            logger.info(f"开始保存模型到: {save_dir}")
            trainer.save_model(save_dir)

            # 验证保存是否成功
            if os.path.exists(save_dir) and os.listdir(save_dir):
                logger.info(f"模型已成功保存到: {save_dir}")

                # 列出保存的文件
                saved_files = os.listdir(save_dir)
                logger.info(f"保存的文件: {saved_files}")
            else:
                logger.error(f"模型保存验证失败: 目录不存在或为空 - {save_dir}")

        except Exception as save_e:
            logger.error(f"模型保存失败: {save_e}")
            # 不要抛出异常，继续执行其他清理工作
            import traceback
            logger.error(f"保存错误详情: {traceback.format_exc()}")

        return TrainingResult(
            model=model,
            save_dir=save_dir
        )

    def _perform_baseline_evaluation(self, model, datasets, evaluators):
        """Perform baseline evaluation before training (matching original train.py)."""
        logger.info("开始训练前基线评估")

        # Validation set baseline evaluation (matching train.py lines 1455-1489)
        dev_evaluator = None
        if datasets.get('eval') is not None:
            # Check if multi-dataset (same logic as original)
            if self.data_loader.is_multi_dataset(datasets['eval']):
                # Multi-dataset: create SequentialEvaluator
                target_column = self._get_target_column(datasets)
                model_name = self.model_config.get('model_name_or_path', 'unknown')
                short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
                run_name = f"{self.train_type}-{short_model_name}"

                # 获取数据源映射
                data_source_mapping = self._get_data_source_mapping(datasets['train'])
                dev_evaluator = self.evaluator_factory.create_multi_evaluator(
                    datasets['eval'], target_column, run_name, data_source_mapping
                )
            elif 'dev' in evaluators:
                # Single dataset: use single evaluator
                dev_evaluator = evaluators['dev']

            # Evaluate baseline model
            if dev_evaluator is not None:
                try:
                    dev_results = self.evaluator_factory.evaluate_model(model, dev_evaluator)
                    logger.info(f"验证集基线评估结果: {dev_results}")

                    task_id = self.raw_config.get('task_id')
                    if task_id:
                        # 检测是否为SequentialEvaluator，从结果中按前缀分别保存
                        from sentence_transformers.evaluation import SequentialEvaluator
                        if isinstance(dev_evaluator, SequentialEvaluator):
                            logger.info(f"检测到SequentialEvaluator，从结果中按前缀提取{len(dev_evaluator.evaluators)}个子评估器的基线结果")

                            # 从SequentialEvaluator结果中按前缀提取各source_id的结果
                            self._save_sequential_evaluator_results_by_prefix(task_id, dev_results, 'eval', dev_evaluator, is_baseline=True)
                        else:
                            # 非SequentialEvaluator，单数据集使用source_id="1"
                            if dev_results:
                                self._save_baseline_results_by_source_id(task_id, dev_results, 'eval', "1")

                except Exception as e:
                    logger.warning(f"验证集基线评估失败: {e}")
            else:
                logger.info("没有有效的验证评估器，跳过基线模型验证集评估")

        # Test set baseline evaluation (matching train.py lines 1491-1526)
        base_test_evaluator = None
        if datasets.get('test') is not None:
            # Check if multi-dataset (same logic as original)
            if self.data_loader.is_multi_dataset(datasets['test']):
                # Multi-dataset: create SequentialEvaluator
                target_column = self._get_target_column(datasets)
                model_name = self.model_config.get('model_name_or_path', 'unknown')
                short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
                run_name = f"{self.train_type}-{short_model_name}"

                # 获取数据源映射
                data_source_mapping = self._get_data_source_mapping(datasets['train'])
                base_test_evaluator = self.evaluator_factory.create_multi_evaluator(
                    datasets['test'], target_column, run_name, data_source_mapping
                )
            elif 'test' in evaluators:
                # Single dataset: use single evaluator
                base_test_evaluator = evaluators['test']

            # Evaluate baseline model
            if base_test_evaluator is not None:
                try:
                    base_test_results = self.evaluator_factory.evaluate_model(model, base_test_evaluator)
                    logger.info(f"测试集基线评估结果: {base_test_results}")

                    task_id = self.raw_config.get('task_id')
                    if task_id:
                        # 检测是否为SequentialEvaluator，从结果中按前缀分别保存
                        from sentence_transformers.evaluation import SequentialEvaluator
                        if isinstance(base_test_evaluator, SequentialEvaluator):
                            logger.info(f"检测到SequentialEvaluator，从结果中按前缀提取{len(base_test_evaluator.evaluators)}个子评估器的基线结果")

                            # 从SequentialEvaluator结果中按前缀提取各source_id的结果
                            self._save_sequential_evaluator_results_by_prefix(task_id, base_test_results, 'test', base_test_evaluator, is_baseline=True)
                        else:
                            # 非SequentialEvaluator，单数据集使用source_id="1"
                            if base_test_results:
                                self._save_baseline_results_by_source_id(task_id, base_test_results, 'test', "1")

                except Exception as e:
                    logger.warning(f"测试集基线评估失败: {e}")
            else:
                logger.info("没有有效的测试评估器，跳过基线模型测试集评估")

    def _perform_final_evaluation(self, model, datasets, evaluators):
        """Perform final evaluation after training (matching original train.py)."""
        logger.info("开始训练后最终评估")

        # Final validation set evaluation (matching train.py lines 1942-1978)
        if datasets.get('eval') is not None:
            # Check if multi-dataset (same logic as original)
            if self.data_loader.is_multi_dataset(datasets['eval']):
                # Multi-dataset: create SequentialEvaluator
                target_column = self._get_target_column(datasets)
                model_name = self.model_config.get('model_name_or_path', 'unknown')
                short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
                run_name = f"{self.train_type}-{short_model_name}"

                # 获取数据源映射
                data_source_mapping = self._get_data_source_mapping(datasets['train'])
                final_eval_evaluator = self.evaluator_factory.create_multi_evaluator(
                    datasets['eval'], target_column, run_name, data_source_mapping
                )
            elif 'dev' in evaluators:
                # Single dataset: use single evaluator
                final_eval_evaluator = evaluators['dev']
            else:
                final_eval_evaluator = None

            # Evaluate trained validation set
            if final_eval_evaluator is not None:
                try:
                    # 运行评估器（包括SequentialEvaluator会自动运行所有子评估器）
                    final_eval_results = self.evaluator_factory.evaluate_model(model, final_eval_evaluator)
                    logger.info(f"验证集最终评估结果: {final_eval_results}")

                    task_id = self.raw_config.get('task_id')
                    if task_id:
                        # 检测是否为SequentialEvaluator，从结果中按前缀分别保存
                        from sentence_transformers.evaluation import SequentialEvaluator
                        if isinstance(final_eval_evaluator, SequentialEvaluator):
                            logger.info(f"检测到SequentialEvaluator，从结果中按前缀提取{len(final_eval_evaluator.evaluators)}个子评估器的最终结果")

                            # 从SequentialEvaluator结果中按前缀提取各source_id的结果
                            self._save_sequential_evaluator_results_by_prefix(task_id, final_eval_results, 'eval', final_eval_evaluator, is_baseline=False)
                        else:
                            # 非SequentialEvaluator，单数据集使用source_id="1"
                            if final_eval_results:
                                self._save_final_results_by_source_id(task_id, final_eval_results, 'eval', "1")

                except Exception as e:
                    logger.warning(f"验证集最终评估失败: {e}")
            else:
                logger.info("没有有效的验证评估器，跳过验证集最终评估")

        # Final test set evaluation (matching train.py lines 1980+)
        if datasets.get('test') is not None:
            # Check if multi-dataset (same logic as original)
            if self.data_loader.is_multi_dataset(datasets['test']):
                # Multi-dataset: create SequentialEvaluator
                target_column = self._get_target_column(datasets)
                model_name = self.model_config.get('model_name_or_path', 'unknown')
                short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
                run_name = f"{self.train_type}-{short_model_name}"

                # 获取数据源映射
                data_source_mapping = self._get_data_source_mapping(datasets['train'])
                test_evaluator = self.evaluator_factory.create_multi_evaluator(
                    datasets['test'], target_column, run_name, data_source_mapping
                )
            elif 'test' in evaluators:
                # Single dataset: use single evaluator
                test_evaluator = evaluators['test']
            else:
                test_evaluator = None

            # Evaluate trained test set
            if test_evaluator is not None:
                try:
                    # 运行评估器（包括SequentialEvaluator会自动运行所有子评估器）
                    test_results = self.evaluator_factory.evaluate_model(model, test_evaluator)
                    logger.info(f"测试集最终评估结果: {test_results}")

                    task_id = self.raw_config.get('task_id')
                    if task_id:
                        # 检测是否为SequentialEvaluator，从结果中按前缀分别保存
                        from sentence_transformers.evaluation import SequentialEvaluator
                        if isinstance(test_evaluator, SequentialEvaluator):
                            logger.info(f"检测到SequentialEvaluator，从结果中按前缀提取{len(test_evaluator.evaluators)}个子评估器的最终结果")

                            # 从SequentialEvaluator结果中按前缀提取各source_id的结果
                            self._save_sequential_evaluator_results_by_prefix(task_id, test_results, 'test', test_evaluator, is_baseline=False)
                        else:
                            # 非SequentialEvaluator，单数据集使用source_id="1"
                            if test_results:
                                self._save_final_results_by_source_id(task_id, test_results, 'test', "1")

                except Exception as e:
                    logger.warning(f"测试集最终评估失败: {e}")
            else:
                logger.info("没有有效的测试评估器，跳过测试集最终评估")

    def _save_baseline_results_by_source_id(self, task_id: str, results: dict, split_type: str, source_id: str):
        """Save baseline evaluation results to database by source_id."""
        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
            from bubble_rag.training.mysql_service.entity.training_task_models import safe_get_session
            from bubble_rag.training.mysql_service.entity.training_dataset_models import DatasetInfo
            from sqlmodel import select

            # 在同一个会话中完成查询和更新
            with safe_get_session() as session:
                statement = select(DatasetInfo).where(
                    DatasetInfo.task_id == task_id,
                    DatasetInfo.data_source_id == source_id,
                    DatasetInfo.split_type == split_type
                )
                dataset_info = session.exec(statement).first()

                if dataset_info:
                    dataset_id = dataset_info.id
                    dataset_name = dataset_info.dataset_name

                    # 在同一个会话外调用更新方法
                    TrainingDatasetService.update_eval_results(
                        dataset_id=dataset_id,
                        base_results=results
                    )
                    logger.info(f"{split_type}集基线结果已保存到数据集 '{dataset_name}' (source_id: {source_id})")
                else:
                    logger.warning(f"未找到source_id '{source_id}' 和split_type '{split_type}' 对应的数据集记录")

        except Exception as e:
            logger.warning(f"按source_id保存{split_type}集基线结果失败: {e}")

    def _save_final_results_by_source_id(self, task_id: str, results: dict, split_type: str, source_id: str):
        """Save final evaluation results to database by source_id."""
        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
            from bubble_rag.training.mysql_service.entity.training_task_models import safe_get_session
            from bubble_rag.training.mysql_service.entity.training_dataset_models import DatasetInfo
            from sqlmodel import select

            # 在同一个会话中完成查询和更新
            with safe_get_session() as session:
                statement = select(DatasetInfo).where(
                    DatasetInfo.task_id == task_id,
                    DatasetInfo.data_source_id == source_id,
                    DatasetInfo.split_type == split_type
                )
                dataset_info = session.exec(statement).first()

                if dataset_info:
                    dataset_id = dataset_info.id
                    dataset_name = dataset_info.dataset_name

                    # 在同一个会话外调用更新方法
                    TrainingDatasetService.update_eval_results(
                        dataset_id=dataset_id,
                        final_results=results
                    )
                    logger.info(f"{split_type}集最终结果已保存到数据集 '{dataset_name}' (source_id: {source_id})")
                else:
                    logger.warning(f"未找到source_id '{source_id}' 和split_type '{split_type}' 对应的数据集记录")

        except Exception as e:
            logger.warning(f"按source_id保存{split_type}集最终结果失败: {e}")

    def _save_sequential_evaluator_results_by_prefix(self, task_id: str, sequential_results: dict, split_type: str,
                                                   sequential_evaluator, is_baseline: bool = True):
        """从SequentialEvaluator结果中按前缀提取各source_id的结果并分别保存."""
        try:
            # 获取所有source_id（即evaluator names）
            source_ids = [getattr(evaluator, 'name', 'unknown') for evaluator in sequential_evaluator.evaluators]
            logger.info(f"从SequentialEvaluator结果中提取source_ids: {source_ids}")

            # 为每个source_id提取对应的metrics
            for source_id in source_ids:
                try:
                    # 提取带有该source_id前缀的所有metrics
                    prefix = f"{source_id}_"
                    source_results = {}

                    for key, value in sequential_results.items():
                        if key.startswith(prefix):
                            # 去掉前缀，保留原始metric名称
                            original_key = key[len(prefix):]
                            source_results[original_key] = value

                    if source_results:
                        logger.info(f"source_id '{source_id}' 的{'基线' if is_baseline else '最终'}结果: {source_results}")

                        # 按source_id保存结果
                        if is_baseline:
                            self._save_baseline_results_by_source_id(task_id, source_results, split_type, source_id)
                        else:
                            self._save_final_results_by_source_id(task_id, source_results, split_type, source_id)
                    else:
                        logger.warning(f"未找到source_id '{source_id}' 对应的metrics（前缀: {prefix}）")

                except Exception as e:
                    logger.warning(f"处理source_id '{source_id}' 的结果失败: {e}")

        except Exception as e:
            result_type = "基线" if is_baseline else "最终"
            logger.warning(f"从SequentialEvaluator结果按前缀提取{result_type}结果失败: {e}")
    
    def _setup_progress_callback(self, trainer, progress_callback):
        """Setup progress callback for trainer (matching original train.py logic)."""
        if not progress_callback:
            return

        try:
            # 初始化步骤计数器和进度跟踪（匹配原train.py逻辑）
            step_count = 0
            last_reported_progress = -1  # 上次报告的进度百分比，用于1%节流
            max_steps = getattr(trainer.args, 'max_steps', None) if hasattr(trainer, 'args') else None
            logger.info(f"进度回调初始化 - 预设max_steps: {max_steps}")

            # 尝试从trainer.state获取实际的max_steps（训练开始后可用）
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps') and trainer.state.max_steps > 0:
                actual_max_steps = trainer.state.max_steps
                logger.info(f"从trainer.state获取实际max_steps: {actual_max_steps}")
                max_steps = actual_max_steps

            # 修复：max_steps=-1 是有效值，表示epoch-based训练，不需要估算
            if (max_steps is None or (max_steps <= 0 and max_steps != -1)) and hasattr(trainer, 'args') and hasattr(trainer.args, 'num_train_epochs'):
                logger.info("需要估算max_steps，开始计算...")
                # 估算最大步数（匹配原train.py逻辑）
                try:
                    if hasattr(trainer, 'train_dataset') and hasattr(trainer.train_dataset, '__len__'):
                        dataset_size = len(trainer.train_dataset)
                    elif hasattr(trainer, '_train_dataset') and hasattr(trainer._train_dataset, '__len__'):
                        dataset_size = len(trainer._train_dataset)
                    else:
                        dataset_size = 1000  # 默认估算

                    batch_size = getattr(trainer.args, 'per_device_train_batch_size', 16) if hasattr(trainer, 'args') else 16
                    gradient_accumulation_steps = getattr(trainer.args, 'gradient_accumulation_steps', 1) if hasattr(trainer, 'args') else 1
                    num_epochs = getattr(trainer.args, 'num_train_epochs', 3) if hasattr(trainer, 'args') else 3

                    # 修复max_steps计算：包含GPU数量（匹配原train.py逻辑）
                    import torch
                    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
                    steps_per_epoch = max(1, dataset_size // effective_batch_size)
                    max_steps = int(steps_per_epoch * num_epochs)

                    logger.info(f"详细计算参数: 数据集大小={dataset_size}, GPU数量={num_gpus}, 每设备批次={batch_size}, 梯度累积={gradient_accumulation_steps}")
                    logger.info(f"计算结果: 有效批次大小={effective_batch_size}, 每轮步数={steps_per_epoch}, 估算最大步数={max_steps}")
                except Exception as e:
                    logger.warning(f"估算最大步数失败: {e}，使用默认值1000")
                    max_steps = 1000

            # 直接包装trainer的training_step方法（匹配原train.py逻辑）
            if hasattr(trainer, 'training_step'):
                original_training_step = trainer.training_step

                def wrapped_training_step(*args, **kwargs):
                    nonlocal step_count, last_reported_progress
                    result = original_training_step(*args, **kwargs)

                    # 使用trainer.state.global_step而不是自己计数，确保与实际训练步数同步
                    if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step'):
                        step_count = trainer.state.global_step
                    else:
                        step_count += 1  # 回退方案

                    # 添加调试日志（匹配原train.py）
                    if step_count % 10 == 0:
                        logger.info(f"training_step被调用: 第{step_count}步")

                    # 实现1%进度节流更新策略
                    try:
                        # 动态获取trainer.state.max_steps（如果可用且合理）
                        runtime_max_steps = None
                        if hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps') and trainer.state.max_steps > 0:
                            runtime_max_steps = trainer.state.max_steps

                        # 优先使用运行时max_steps，然后是估算值，最后是默认值
                        if runtime_max_steps and runtime_max_steps > 0:
                            effective_max_steps = runtime_max_steps
                        elif max_steps and max_steps > 0:
                            effective_max_steps = max_steps
                        else:
                            effective_max_steps = 1000


                        # 计算当前进度百分比
                        current_progress = min(99.0, (step_count / effective_max_steps) * 100)

                        # 只有进度增加1%以上时才更新（节流策略）
                        progress_change = current_progress - last_reported_progress
                        should_update = (
                            progress_change >= 1.0 or  # 进度变化1%以上
                            last_reported_progress == -1 or  # 首次更新
                            step_count % 100 == 0  # 每100步强制更新一次（防止长时间无更新）
                        )

                        if should_update:
                            # 调用外部进度回调
                            progress_callback(step_count, effective_max_steps, "训练中")

                            # 更新上次报告的进度
                            last_reported_progress = current_progress

                            logger.info(f"进度更新(1%节流): {current_progress:.1f}% (步数: {step_count}/{effective_max_steps}, 变化: +{progress_change:.1f}%)")

                            # 注意：不再直接调用task_manager.update_task_progress，避免与外层progress_callback重复
                        elif step_count % 50 == 0:
                            # 定期显示当前进度（但不更新数据库）
                            logger.debug(f"当前进度: {current_progress:.1f}% (步数: {step_count}/{effective_max_steps}, 未达1%更新阈值)")
                    except KeyboardInterrupt:
                        logger.info("检测到训练停止信号，中断训练")
                        if hasattr(trainer, 'state'):
                            trainer.state.should_epoch_stop = True
                            trainer.state.should_training_stop = True
                        raise
                    except Exception as e:
                        logger.error(f"progress_callback调用失败: {e}")

                    return result

                trainer.training_step = wrapped_training_step
                logger.info("成功包装training_step方法进行进度追踪")

            # 包装trainer的log方法来收集loss数据（匹配原train.py逻辑）
            if hasattr(trainer, 'log'):
                original_log = trainer.log
                task_id = self.raw_config.get('task_id')
                output_dir = getattr(trainer.args, 'output_dir', './output') if hasattr(trainer, 'args') else './output'

                def wrapped_log(logs, start_time=None):
                    try:
                        # 调用原始log方法 - 统一使用原train.py的调用方式
                        if original_log and callable(original_log):
                            result = original_log(logs, start_time)
                        else:
                            result = None
                    except Exception as e:
                        logger.error(f"原始log方法调用失败: {e}")
                        result = None


                    # 添加基于log的进度更新作为fallback (匹配原train.py)
                    try:
                        if logs and task_id:
                            # 尝试从logs中获取进度信息
                            if 'step' in logs and hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps'):
                                current_step = logs['step']
                                max_steps = trainer.state.max_steps

                                if max_steps and max_steps > 0:
                                    from bubble_rag.training.model_sft.services.task_manager import task_manager
                                    progress_percentage = min(99.0, (current_step / max_steps) * 100)
                                    task_manager.update_task_progress(task_id, progress_percentage, f"训练中: {current_step}/{max_steps}")

                                    if current_step % 10 == 0:
                                        logger.info(f"基于log的进度更新: {progress_percentage:.1f}% (步数: {current_step}/{max_steps})")
                    except Exception as e:
                        if logs and logs.get('step', 0) % 20 == 0:  # 减少错误日志频率
                            logger.warning(f"基于log的进度更新失败: {e}")

                    return result

                trainer.log = wrapped_log
                logger.info("成功包装log方法进行loss数据收集")

            # 初始进度回调（匹配原train.py）
            try:
                # 修复：直接使用已经正确计算的步数，不依赖不可靠的初始max_steps
                # 等训练开始后通过training_step回调来更新进度，避免错误的初始回调
                logger.info("⏭️ 跳过初始progress_callback，等待训练开始后通过training_step回调更新进度")
                # 不调用初始回调，让第一次training_step回调来设置正确的total_steps
            except Exception as e:
                logger.error(f"初始progress_callback设置失败: {e}")

        except Exception as e:
            logger.warning(f"设置进度回调失败: {e}")

    def _setup_loss_collection(self, trainer):
        """Setup loss data collection without progress callback (matching original train.py)."""
        try:
            logger.info(f"[DEBUG] 开始设置loss收集，trainer类型: {type(trainer)}")
            # 包装trainer的log方法来收集loss数据（匹配原train.py逻辑）
            if hasattr(trainer, 'log'):
                logger.info(f"[DEBUG] trainer有log方法: {type(trainer.log)}")
                original_log = trainer.log
                task_id = self.raw_config.get('task_id')
                output_dir = getattr(trainer.args, 'output_dir', './output') if hasattr(trainer, 'args') else './output'
                step_count = 0

                def wrapped_log(logs, start_time=None):
                    nonlocal step_count
                    try:
                        # Log包装正常工作

                        # 调用原始log方法 - 统一使用原train.py的调用方式
                        if original_log and callable(original_log):
                            result = original_log(logs, start_time)
                        else:
                            result = None
                    except Exception as e:
                        logger.error(f"原始log方法调用失败: {e}")
                        result = None

                    # 启用loss本地文件保存功能 - 与train.py逻辑保持一致
                    try:
                        # 只要有loss或eval字段就记录，扩展以包含所有eval指标
                        has_loss = any(key in logs for key in ['train_loss', 'eval_loss', 'loss'])
                        has_eval = any('eval' in key.lower() for key in logs.keys())
                        if logs and task_id and (has_loss or has_eval):
                            from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager

                            # 区分training和evaluation log的step计算
                            if 'step' in logs:
                                # 如果logs中有step，直接使用（这通常是正确的）
                                current_step = logs['step']
                                logger.debug(f"使用logs中的step: {current_step}")
                            else:
                                # 当logs中没有step时，需要根据log类型确定step
                                is_eval_log = any('eval' in key.lower() for key in logs.keys())

                                if is_eval_log:
                                    # evaluation log: 根据eval_steps计算
                                    # eval通常在特定步数触发，我们需要从trainer获取当前真实步数
                                    try:
                                        if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step'):
                                            current_step = trainer.state.global_step
                                            logger.debug(f"Eval log - 使用trainer.state.global_step: {current_step}")
                                        else:
                                            # 回退：估算eval步数
                                            eval_steps = 100  # 默认值
                                            if hasattr(trainer, 'args') and hasattr(trainer.args, 'eval_steps'):
                                                eval_steps = trainer.args.eval_steps
                                            current_step = eval_steps * (step_count // 2 + 1)  # 粗略估算
                                            logger.debug(f"Eval log - 估算步数: {current_step}")
                                    except:
                                        current_step = step_count * 10  # 最后的回退
                                else:
                                    # training log: 根据logging_steps计算
                                    step_count += 1
                                    logging_steps = 10  # 默认值
                                    if hasattr(trainer, 'args') and hasattr(trainer.args, 'logging_steps'):
                                        logging_steps = trainer.args.logging_steps

                                    current_step = step_count * logging_steps
                                    logger.debug(f"Training log - 计算步数: 回调#{step_count} × {logging_steps} = {current_step}")

                            current_epoch = logs.get('epoch', None)

                            # 构建loss指标字典 - 包含所有数值型指标
                            loss_metrics = {}
                            for key, value in logs.items():
                                if isinstance(value, (int, float)):
                                    # 包含loss、eval指标，以及常见的评估指标
                                    if any(keyword in key.lower() for keyword in ['loss', 'eval', 'accuracy', 'f1', 'precision', 'recall', 'pearson', 'spearman']):
                                        loss_metrics[key] = value

                            if loss_metrics:
                                loss_manager = get_loss_manager(output_dir, task_id)

                                # 检查是否是混合的eval记录，需要按数据源分离
                                is_mixed_eval = self._is_mixed_eval_record(loss_metrics)
                                logger.info(f"[DEBUG] 混合eval检测: {is_mixed_eval}, 指标: {list(loss_metrics.keys())}")
                                logger.info(f"[DEBUG] _current_data_source_mapping存在: {hasattr(self, '_current_data_source_mapping')}")
                                if hasattr(self, '_current_data_source_mapping'):
                                    logger.info(f"[DEBUG] _current_data_source_mapping: {self._current_data_source_mapping}")

                                if is_mixed_eval:
                                    # 按数据源分离并保存多条记录
                                    logger.info(f"调用_save_separated_eval_records进行loss名称转换")
                                    self._save_separated_eval_records(loss_manager, current_step, loss_metrics, current_epoch)
                                else:
                                    # 普通记录，直接保存
                                    logger.info(f"保存单条记录: step={current_step}, metrics={list(loss_metrics.keys())}")
                                    data_source_mapping = getattr(self, '_current_data_source_mapping', {})
                                    loss_manager.save_loss_record(current_step, loss_metrics, current_epoch, True, data_source_mapping)

                                logger.info(f"Loss已保存到本地文件: step={current_step}, metrics={list(loss_metrics.keys())}")

                                # 保存评估结果到数据库 (包含eval_loss和其他eval指标)
                                eval_metrics = {k: v for k, v in loss_metrics.items() if k.startswith('eval_')}
                                if eval_metrics:
                                    try:
                                        # 传递data_source_mapping给评估结果保存函数
                                        data_source_mapping = getattr(self, '_current_data_source_mapping', {})
                                        self._save_training_evaluation_results(task_id, eval_metrics, current_step, current_epoch, data_source_mapping)
                                        logger.info(f"评估结果已保存到数据库: step={current_step}, metrics={list(eval_metrics.keys())}")
                                    except Exception as e:
                                        logger.warning(f"保存评估结果到数据库失败: {e}")

                            else:
                                logger.debug(f"⏭️ 没有有效的loss指标可保存: {list(logs.keys())}")

                    except Exception as e:
                        logger.warning(f"保存loss到本地文件失败: {e}")

                    return result

                trainer.log = wrapped_log
                logger.info(f"[DEBUG] 成功包装log方法进行loss数据收集，新log方法: {type(trainer.log)}")
            else:
                logger.warning("[DEBUG] trainer没有log方法，无法包装")

        except Exception as e:
            logger.warning(f"设置loss数据收集失败: {e}")

    def _setup_basic_progress_tracking(self, trainer):
        """Setup basic progress tracking without external progress callback (matching original train.py)."""
        try:
            task_id = self.raw_config.get('task_id')
            if not task_id:
                return

            # 初始化步骤计数器和进度跟踪（匹配原train.py逻辑）
            step_count = 0
            last_reported_progress = -1  # 上次报告的进度百分比，用于1%节流
            max_steps = getattr(trainer.args, 'max_steps', None) if hasattr(trainer, 'args') else None
            logger.info(f"基础进度跟踪初始化 - 预设max_steps: {max_steps}")

            # 修复：max_steps=-1 是有效值，表示epoch-based训练，不需要估算
            if (max_steps is None or (max_steps <= 0 and max_steps != -1)) and hasattr(trainer, 'args') and hasattr(trainer.args, 'num_train_epochs'):
                logger.info("基础进度跟踪需要估算max_steps，开始计算...")
                # 估算最大步数（匹配原train.py逻辑）
                try:
                    if hasattr(trainer, 'train_dataset') and hasattr(trainer.train_dataset, '__len__'):
                        dataset_size = len(trainer.train_dataset)
                    elif hasattr(trainer, '_train_dataset') and hasattr(trainer._train_dataset, '__len__'):
                        dataset_size = len(trainer._train_dataset)
                    else:
                        dataset_size = 1000  # 默认估算

                    batch_size = getattr(trainer.args, 'per_device_train_batch_size', 16) if hasattr(trainer, 'args') else 16
                    gradient_accumulation_steps = getattr(trainer.args, 'gradient_accumulation_steps', 1) if hasattr(trainer, 'args') else 1
                    num_epochs = getattr(trainer.args, 'num_train_epochs', 3) if hasattr(trainer, 'args') else 3

                    # 修复max_steps计算：包含GPU数量（匹配原train.py逻辑）
                    import torch
                    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
                    steps_per_epoch = max(1, dataset_size // effective_batch_size)
                    max_steps = int(steps_per_epoch * num_epochs)

                    logger.info(f"基础进度跟踪 - 详细计算参数: 数据集大小={dataset_size}, GPU数量={num_gpus}, 每设备批次={batch_size}, 梯度累积={gradient_accumulation_steps}")
                    logger.info(f"基础进度跟踪 - 计算结果: 有效批次大小={effective_batch_size}, 每轮步数={steps_per_epoch}, 估算最大步数={max_steps}")
                except Exception as e:
                    logger.warning(f"估算最大步数失败: {e}，使用默认值1000")
                    max_steps = 1000

            # 包装trainer的training_step方法进行进度跟踪
            if hasattr(trainer, 'training_step'):
                original_training_step = trainer.training_step

                def wrapped_training_step(*args, **kwargs):
                    nonlocal step_count, last_reported_progress
                    result = original_training_step(*args, **kwargs)
                    step_count += 1

                    # 实现1%进度节流更新策略（基础进度跟踪）
                    try:
                        from bubble_rag.training.model_sft.services.task_manager import task_manager

                        # 动态获取trainer.state.max_steps（如果可用且合理）
                        runtime_max_steps = None
                        if hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps') and trainer.state.max_steps > 0:
                            runtime_max_steps = trainer.state.max_steps

                        # 优先使用运行时max_steps，然后是估算值，最后是默认值
                        if runtime_max_steps and runtime_max_steps > 0:
                            effective_max_steps = runtime_max_steps
                        elif max_steps and max_steps > 0:
                            effective_max_steps = max_steps
                        else:
                            effective_max_steps = 1000

                        # 计算当前进度百分比
                        current_progress = min(99.0, (step_count / effective_max_steps) * 100)

                        # 只有进度增加1%以上时才更新（节流策略）
                        progress_change = current_progress - last_reported_progress
                        should_update = (
                            progress_change >= 1.0 or  # 进度变化1%以上
                            last_reported_progress == -1 or  # 首次更新
                            step_count % 100 == 0  # 每100步强制更新一次（防止长时间无更新）
                        )

                        if should_update:
                            task_manager.update_task_progress(task_id, current_progress, f"训练中: {step_count}/{effective_max_steps}")

                            # 更新上次报告的进度
                            last_reported_progress = current_progress

                            logger.info(f"基础进度跟踪(1%节流): {current_progress:.1f}% (步数: {step_count}/{effective_max_steps}, 变化: +{progress_change:.1f}%)")
                        elif step_count % 50 == 0:
                            # 定期显示当前进度（但不更新数据库）
                            logger.debug(f"基础进度: {current_progress:.1f}% (步数: {step_count}/{effective_max_steps}, 未达1%更新阈值)")
                    except Exception as tm_e:
                        if step_count % 100 == 0:  # 减少错误日志频率
                            logger.warning(f"基础进度跟踪失败: {tm_e}")

                    return result

                trainer.training_step = wrapped_training_step
                logger.info("成功设置基础进度跟踪")

        except Exception as e:
            logger.warning(f"设置基础进度跟踪失败: {e}")

    def _is_mixed_eval_record(self, loss_metrics: dict) -> bool:
        """检查是否是包含多个数据源的混合eval记录"""
        eval_keys = [key for key in loss_metrics.keys() if 'eval' in key.lower()]
        if len(eval_keys) < 2:
            return False

        # 动态检测数据源：利用现有的映射构建函数
        source_ids, dataset_to_source_mapping = self._extract_data_sources_and_build_mapping(eval_keys)

        # 如果检测到多个数据源，就是混合记录
        total_sources = len(source_ids) + len(dataset_to_source_mapping)
        return total_sources > 1

    def _extract_data_sources_and_build_mapping(self, eval_keys: list) -> tuple:
        """
        从eval指标键中提取数据源信息并建立映射关系

        Returns:
            tuple: (source_ids, dataset_to_source_mapping)
            - source_ids: set of source_ids
            - dataset_to_source_mapping: {dataset_name: source_id} 的动态映射
        """
        source_ids = set()
        dataset_to_source_mapping = {}

        # 直接使用已建立的data_source_mapping，避免重复解析和硬编码
        if hasattr(self, '_current_data_source_mapping'):
            # 收集所有source_ids
            for source_id in self._current_data_source_mapping.values():
                source_ids.add(source_id)

            # 建立数据集名称到source_id的映射
            for full_name, source_id in self._current_data_source_mapping.items():
                dataset_to_source_mapping[full_name] = source_id
                # 同时为简化名称建立映射（如果是路径格式）
                if '/' in full_name:
                    simple_name = full_name.split('/')[-1]
                    dataset_to_source_mapping[simple_name] = source_id

        return source_ids, dataset_to_source_mapping

    def _save_separated_eval_records(self, loss_manager, current_step: int, loss_metrics: dict, current_epoch):
        """将混合的eval记录按数据源分离保存并统一命名"""
        logger.info(f"分离保存eval记录: step={current_step}, 原始指标数={len(loss_metrics)}")

        # 获取所有eval键，提取数据源和映射关系
        eval_keys = [key for key in loss_metrics.keys() if 'eval' in key.lower()]
        source_ids, dataset_to_source_mapping = self._extract_data_sources_and_build_mapping(eval_keys)
        logger.info(f"检测到数据源: {source_ids}")
        logger.info(f"🔗 数据集映射: {dataset_to_source_mapping}")

        # 按数据源分离指标
        source_metrics = {}
        common_metrics = {}

        # 初始化每个数据源的指标字典
        for source_id in source_ids:
            source_metrics[source_id] = {}

        # 分类和转换指标
        for key, value in loss_metrics.items():
            assigned_to_source = False

            # 检查格式1: eval_{source_id}_* (已经是标准格式)
            for source_id in source_ids:
                if key.startswith(f'eval_{source_id}_'):
                    source_metrics[source_id][key] = value
                    assigned_to_source = True
                    break

            # 检查格式2: 其他格式，通过数据集名称匹配转换
            if not assigned_to_source:
                for dataset_name, source_id in dataset_to_source_mapping.items():
                    if dataset_name in key:
                        # 转换指标名称为统一格式
                        metric_name = key.split('_')[-1]  # 提取最后的指标名(loss, runtime等)
                        new_key = f'eval_{source_id}_{metric_name}'
                        source_metrics[source_id][new_key] = value
                        assigned_to_source = True
                        logger.debug(f"转换指标名称: {key} → {new_key}")
                        break

            # 如果不属于任何特定数据源，则归为通用指标
            if not assigned_to_source:
                # 排除已知的数据源特定指标，保留通用指标（如sequential_score等）
                is_source_specific = (
                    any(key.startswith(f'eval_{source_id}_') for source_id in source_ids) or
                    any(dataset_name in key for dataset_name in dataset_to_source_mapping.keys())
                )
                if not is_source_specific:
                    common_metrics[key] = value

        # 为每个数据源保存记录，支持同step内的合并
        for source_id, metrics in source_metrics.items():
            if metrics:  # 只有当该数据源有指标时才保存
                source_record = {**metrics, **common_metrics}
                # 使用step+source_id作为合并键
                merge_key = f"step_{current_step}_source_{source_id}"
                loss_manager.save_or_merge_loss_record(current_step, source_record, current_epoch, merge_key)
                logger.info(f"source_id {source_id} 记录已保存/合并: {list(metrics.keys())}")

        # 如果没有检测到数据源但有指标，保存为单条记录
        if not source_ids and loss_metrics:
            logger.warning(f"未检测到数据源，保存为单条记录")
            data_source_mapping = getattr(self, '_current_data_source_mapping', {})
            loss_manager.save_loss_record(current_step, loss_metrics, current_epoch, True, data_source_mapping)


    def _perform_training_evaluation(self, trainer, current_step):
        """Perform evaluation during training (matching train.py behavior)."""
        try:
            # This matches the periodic evaluation logic from original train.py
            if hasattr(trainer, 'evaluate'):
                eval_results = trainer.evaluate()
                if eval_results:
                    logger.info(f"训练中评估 (Step {current_step}): {eval_results}")
        except Exception as e:
            logger.warning(f"训练中评估失败: {e}")

    def _save_training_evaluation_results(self, task_id: str, eval_logs: dict, step: int, epoch: float, data_source_mapping: Dict[str, str] = None):
        """Save evaluation results during training to database."""
        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService

            # 使用evaluation_result工具保存评估结果到数据库
            from bubble_rag.training.model_sft.utils.evaluation_result import save_evaluation_results_to_database
            save_evaluation_results_to_database(task_id, eval_logs, step, epoch, data_source_mapping)

        except Exception as e:
            logger.warning(f"保存训练评估结果失败: {e}")
    
    # Abstract methods that must be implemented by subclasses

    def _ensure_eval_directories(self, trainer):
        """确保评估器所需的目录存在"""
        import os

        # 确保主输出目录存在
        if not os.path.exists(trainer.args.output_dir):
            os.makedirs(trainer.args.output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {trainer.args.output_dir}")

        # 初始化eval_dir变量
        eval_dir = None

        # 使用配置中定义的user_eval_dir，而不是运行时创建
        try:
            # 从配置中获取自定义参数，包括user_eval_dir
            param_manager = TrainingParametersManager()
            param_manager.load_from_config(self.raw_config)
            custom_params = param_manager.get_custom_params_dict()

            eval_dir = custom_params.get('user_eval_dir')
            if eval_dir:
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir, exist_ok=True)
                    logger.info(f"创建评估输出目录: {eval_dir}")

                logger.info(f"使用output_dir: {trainer.args.output_dir}")
                logger.info(f"使用user_eval_dir: {eval_dir}")
            else:
                logger.warning("未找到user_eval_dir配置，使用默认路径")
                eval_dir = os.path.join(trainer.args.output_dir, "eval")

        except Exception as e:
            logger.error(f"获取user_eval_dir失败: {e}")
            # 回退到原来的方式
            eval_dir = os.path.join(trainer.args.output_dir, "eval")

        # 确保eval_dir存在
        if eval_dir and not os.path.exists(eval_dir):
            os.makedirs(eval_dir, exist_ok=True)
            logger.info(f"创建评估输出目录: {eval_dir}")

        # 注：评估器的 __call__ 方法有 output_path 参数，直接指定输出目录即可

        # # 为评估器预创建可能需要的子目录结构
        # # CrossEncoderCorrelationEvaluator 会创建以评估器名称命名的子目录
        # try:
        #     # 获取模型名称，用于构建评估器目录名 - 需要与评估器实际使用的名称一致
        #     model_short_name = os.path.basename(trainer.model.config.name_or_path) if hasattr(trainer.model, 'config') and hasattr(trainer.model.config, 'name_or_path') else "model"
        #     logger.info(f"模型名称调试: {model_short_name}")

        #     # 可能的评估器目录名变体 - 覆盖不同的命名规则
        #     potential_eval_subdirs = [
        #         # 原始格式
        #         f"CrossEncoderCorrelationEvaluator_{model_short_name}",
        #         f"CrossEncoderClassificationEvaluator_{model_short_name}",
        #         # 带reranker前缀的格式
        #         f"CrossEncoderCorrelationEvaluator_reranker-{model_short_name}",
        #         f"CrossEncoderClassificationEvaluator_reranker-{model_short_name}",
        #         # 带sentence-transformers后缀的格式
        #         f"CrossEncoderCorrelationEvaluator_{model_short_name}-sentence-transformers",
        #         f"CrossEncoderCorrelationEvaluator_reranker-{model_short_name}-sentence-transformers",
        #         # 特殊的eval格式（根据错误信息）
        #         f"CrossEncoderCorrelationEvaluator_reranker-{model_short_name}-eval-sentence-transformers",
        #     ]

        #     for subdir in potential_eval_subdirs:
        #         eval_subdir_path = os.path.join(eval_dir, subdir)
        #         os.makedirs(eval_subdir_path, exist_ok=True)
        #         logger.info(f"预创建评估器目录: {subdir}")
        #     logger.info(f"预创建评估器子目录完成")
        # except Exception as e:
        #     logger.warning(f"预创建评估器子目录失败，但继续训练: {e}")

        logger.info(f"评估目录准备完成，使用简化版本（评估器将通过output_path参数自行创建文件）")

    @abstractmethod
    def initialize_model(self, model_name: str) -> Any:
        """
        Initialize the model for this training type.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Initialized model instance
        """
        pass
    
    @abstractmethod
    def create_loss_function(self, model: Any, train_dataset: Any) -> Any:
        """
        Create loss function for this training type.
        
        Args:
            model: The initialized model
            train_dataset: Training dataset
            
        Returns:
            Loss function instance
        """
        pass
    
    @abstractmethod
    def create_training_args(self, config: Dict[str, Any]) -> Any:
        """
        Create training arguments for this training type.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Training arguments instance
        """
        pass
    
    @abstractmethod
    def create_trainer_instance(self, model, args, train_dataset, eval_dataset, loss, evaluator) -> Any:
        """
        Create trainer instance for this training type.
        
        Args:
            model: The initialized model
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            loss: Loss function
            evaluator: Evaluator (optional)
            
        Returns:
            Trainer instance
        """
        pass
    
    @abstractmethod
    def create_evaluator(self, eval_dataset: Any) -> Any:
        """
        Create evaluator for this training type.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluator instance
        """
        pass

    # Task Status Management Methods (matching original train.py)

    def _update_task_status_running(self, task_id: str):
        """Update task status to RUNNING (matching original train.py)."""
        if not task_id:
            logger.warning("task_id为空，跳过状态更新")
            return

        logger.info(f"开始更新任务状态为RUNNING: {task_id}")

        try:
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            from ..enums import TrainingStatus

            # 增强错误处理：记录详细的更新过程
            logger.info(f"正在调用 training_task_service.update_task_status(task_id={task_id}, status={TrainingStatus.RUNNING.value}, progress=0.0)")

            success = training_task_service.update_task_status(task_id, TrainingStatus.RUNNING.value, progress=0.0)

            if success:
                logger.info(f"数据库状态更新成功: RUNNING, 进度=0.0%")
            else:
                logger.error(f"数据库状态更新失败: update_task_status返回False")
                # 即使数据库更新失败，也要更新内存状态

            # 同时更新task_manager中的状态和进度，确保内存和数据库同步
            try:
                from ..services.task_manager import task_manager

                # 强制更新内存中的任务状态为RUNNING
                task = task_manager.get_task(task_id)
                if task:
                    logger.info(f"更新内存任务状态: {task.status} -> RUNNING")
                    task.status = TrainingStatus.RUNNING.value
                    task.started_at = task.started_at or datetime.now()

                    # 重置 _last_db_progress 属性，确保进度同步正常工作
                    if hasattr(task, '_last_db_progress'):
                        task._last_db_progress = -1
                        logger.info("已重置任务的数据库进度跟踪状态")

                    # 强制更新任务进度为0%，确保数据库同步
                    task_manager.update_task_progress(task_id, 0.0, "训练开始", force_db_update=True)
                    logger.info("内存状态和进度已强制重置: RUNNING, 0%")
                else:
                    logger.warning(f"在task_manager中未找到任务: {task_id}")

            except Exception as tm_e:
                logger.error(f"重置task_manager状态失败: {tm_e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")

            logger.info(f"训练状态更新完成: {task_id}")

        except Exception as e:
            logger.error(f"更新训练状态失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")

            # 即使出现异常，也要尝试更新内存状态，确保训练可以继续
            try:
                from ..services.task_manager import task_manager
                task = task_manager.get_task(task_id)
                if task:
                    task.status = TrainingStatus.RUNNING.value
                    logger.warning(f"异常情况下强制更新内存状态为RUNNING: {task_id}")
            except Exception as fallback_e:
                logger.error(f"异常情况下内存状态更新也失败: {fallback_e}")

    def _update_task_status_post_training_evaluation(self, task_id: str):
        """训练完成后，设置PENDING状态进行最终评估，进度100%"""
        if not task_id:
            return
        try:
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            from bubble_rag.training.model_sft.services.task_manager import task_manager
            from ..enums import TrainingStatus

            # 更新任务管理器：进度100%，状态说明正在最终评估
            task_manager.update_task_progress(task_id, 100.0, "训练完成，正在进行最终评估...")

            # 更新数据库状态为PENDING（表示正在进行最终评估）
            training_task_service.update_task_status(task_id, TrainingStatus.PENDING.value)
            logger.info(f"训练完成，状态设为PENDING进行最终评估: {task_id}, 进度100%")

        except Exception as e:
            logger.warning(f"更新最终评估状态失败（不影响训练）: {e}")

    def _update_task_status_succeeded(self, task_id: str, save_dir: str):
        """Update task status to SUCCEEDED (matching original train.py)."""
        if not task_id:
            return

        try:
            from ..services.task_manager import task_manager
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            from ..enums import TrainingStatus
            from ..enums.training_task_enums import ProcessStatus

            # Update task manager (this sets status to SUCCEEDED and progress to 100%)
            task_manager.complete_task(task_id, save_dir)
            task_manager.update_task_progress(task_id, 100.0, "训练完成")

            # Update database task status to SUCCEEDED (matching unified_training_service)
            training_task_service.update_task_status(task_id, TrainingStatus.SUCCEEDED.value)
            training_task_service.update_task_result(task_id, final_model_path=save_dir)
            logger.info(f"数据库任务状态已更新为SUCCEEDED: {task_id}")

            # 立即释放GPU资源 - 确保与数据库状态同步
            try:
                from ..utils.gpu_resource_manager import gpu_resource_manager
                success = gpu_resource_manager.release_gpus_for_task(task_id)
                if success:
                    logger.info(f"任务成功完成，立即释放GPU资源: {task_id}")
                else:
                    logger.warning(f"常规GPU释放失败，尝试强制释放")
                    gpu_resource_manager.force_release_gpu_for_task(task_id)
            except Exception as gpu_error:
                logger.critical(f"严重错误：训练成功后GPU资源释放失败！尝试强制恢复。任务: {task_id}, 错误: {gpu_error}")
                try:
                    gpu_resource_manager.force_release_gpu_for_task(task_id)
                    logger.warning(f"强制GPU释放已执行")
                except Exception as force_error:
                    logger.critical(f"强制GPU释放也失败！任务 {task_id} 资源可能永久泄漏: {force_error}")

            # Update process status to TERMINATED
            training_task_service.update_process_info(task_id, None, ProcessStatus.TERMINATED.value)

            logger.info(f"训练状态已更新为已完成，模型保存在: {save_dir}")

            # 完成loss管理器的最终化处理并保存汇总到数据库 (matching original train.py)
            try:
                from bubble_rag.training.model_sft.utils.loss_manager import cleanup_loss_manager

                # 获取最终训练指标
                final_metrics = {
                    "final_model_path": save_dir,
                    "training_completed": True
                }

                # 清理loss管理器并获取数据库汇总信息
                loss_summary = cleanup_loss_manager(task_id, final_metrics)

                if loss_summary:
                    # 将loss汇总信息保存到数据库
                    try:
                        import json
                        loss_data_json = json.dumps(loss_summary, ensure_ascii=False)
                        training_task_service.update_task_result(task_id, loss_data=loss_data_json)
                        logger.info(f"Loss汇总信息已保存到数据库: {len(loss_summary)} 项指标")
                    except Exception as db_e:
                        logger.warning(f"保存loss汇总到数据库失败: {db_e}")

                logger.info("Loss管理器已完成最终化处理")
            except Exception as loss_e:
                logger.warning(f"Loss管理器清理失败: {loss_e}")

        except Exception as e:
            logger.warning(f"更新训练成功状态失败: {e}")

    def _update_task_status_failed(self, task_id: str, error_msg: str):
        """Update task status to FAILED (matching original train.py)."""
        if not task_id:
            return

        try:
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            from ..enums import TrainingStatus
            from ..services.task_manager import task_manager

            # Update task manager
            task_manager.fail_task(task_id, error_msg, None)

            # Update database
            training_task_service.update_task_status(task_id, TrainingStatus.FAILED.value)
            training_task_service.update_task_result(task_id, error_message=error_msg)
            logger.info(f"全局异常处理：任务状态已更新为FAILED: {task_id}")

            # 立即释放GPU资源 - 确保与数据库状态同步
            try:
                from ..utils.gpu_resource_manager import gpu_resource_manager
                success = gpu_resource_manager.release_gpus_for_task(task_id)
                if success:
                    logger.info(f"任务失败完成，立即释放GPU资源: {task_id}")
                else:
                    logger.warning(f"常规GPU释放失败，尝试强制释放")
                    gpu_resource_manager.force_release_gpu_for_task(task_id)
            except Exception as gpu_error:
                logger.critical(f"严重错误：训练失败时GPU资源释放失败！尝试强制恢复。任务: {task_id}, 错误: {gpu_error}")
                try:
                    gpu_resource_manager.force_release_gpu_for_task(task_id)
                    logger.warning(f"强制GPU释放已执行")
                except Exception as force_error:
                    logger.critical(f"强制GPU释放也失败！任务 {task_id} 资源可能永久泄漏: {force_error}")

        except Exception as status_e:
            logger.warning(f"全局异常处理：更新训练失败状态失败: {status_e}")

    def _save_training_metadata(self, trainer):
        """保存训练元数据到loss文件，包含数据源映射等信息"""
        try:
            task_id = self.raw_config.get('task_id')
            if not task_id:
                logger.warning("缺少task_id，跳过元数据保存")
                return

            output_dir = getattr(trainer.args, 'output_dir', './output') if hasattr(trainer, 'args') else './output'

            from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
            loss_manager = get_loss_manager(output_dir, task_id)

            # 构建元数据
            metadata = {
                "train_type": self.train_type,
                "model_config": self.model_config,
                "data_config": self.data_config,
                "data_source_mapping": getattr(self, '_current_data_source_mapping', {}),
            }

            # 保存元数据
            loss_manager.save_metadata(metadata)
            logger.info(f"训练元数据已保存，包含数据源映射: {metadata.get('data_source_mapping', {})}")

        except Exception as e:
            logger.warning(f"保存训练元数据失败: {e}")