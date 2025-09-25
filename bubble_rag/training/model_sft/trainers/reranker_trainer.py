"""
Reranker model trainer.

Specialized trainer for reranker models using CrossEncoder.
"""

import logging
import os
from typing import Any, Dict, Optional, Union

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.losses.MSELoss import MSELoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

from ..core.base_trainer import BaseTrainer
from ..utils.evaluation import UnifiedEvaluator

logger = logging.getLogger(__name__)


class RerankerTrainer(BaseTrainer):
    """
    Specialized trainer for reranker models.
    
    Handles CrossEncoder model training with appropriate loss functions
    and training arguments.
    """
    
    def initialize_model(self, model_name: str) -> CrossEncoder:
        """
        Initialize CrossEncoder model.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Initialized CrossEncoder model
        """
        logger.info(f"初始化 CrossEncoder 模型: {model_name}")
        
        try:
            # Try ModelScope download first (if available)
            try:
                from modelscope import snapshot_download
                from bubble_rag.server_config import TRAINING_CACHE
                
                logger.info(f"尝试从 ModelScope 下载模型: {model_name}")
                model_dir = snapshot_download(model_name, cache_dir=TRAINING_CACHE)
                model = CrossEncoder(model_dir)
                logger.info(f"✅ ModelScope 下载成功，路径: {model_dir}")
                
            except ImportError:
                logger.info("ModelScope 未安装，使用 HuggingFace 方式")
                model = CrossEncoder(model_name)
                logger.info(f"✅ HuggingFace 下载成功: {model_name}")
                
            except Exception as e:
                # Fallback to HuggingFace if ModelScope fails
                logger.warning(f"ModelScope 下载失败: {e}，回退到 HuggingFace")
                model = CrossEncoder(model_name)
                logger.info(f"✅ HuggingFace 下载成功: {model_name}")
            
            # Handle tokenizer pad token
            if model.tokenizer.pad_token is None:
                logger.info("设置 tokenizer pad_token")
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
            logger.info(f"CrossEncoder 模型初始化成功: {model_name}")
            return model
            
        except Exception as e:
            # Try offline mode as last resort
            if "couldn't connect" in str(e).lower() or "connection" in str(e).lower():
                logger.warning(f"网络连接失败，尝试使用本地缓存加载模型: {model_name}")
                try:
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    model = CrossEncoder(model_name, local_files_only=True)
                    if model.tokenizer.pad_token is None:
                        model.tokenizer.pad_token = model.tokenizer.eos_token
                    logger.info(f"✅ 本地缓存加载成功: {model_name}")
                    return model
                except Exception as cache_error:
                    error_msg = f"reranker模型初始化失败: {model_name}，网络不可用且本地缓存未找到。错误: {str(cache_error)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"reranker模型初始化失败: {model_name}, 错误: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    def create_loss_function(self, model: CrossEncoder, train_dataset: Any) -> Any:
        """
        Create appropriate loss function for reranker training.
        
        Args:
            model: The CrossEncoder model
            train_dataset: Training dataset (can be single dataset or dict of datasets)
            
        Returns:
            Loss function or dict of loss functions for multi-dataset training
        """
        logger.info("创建 reranker 损失函数")
        
        if isinstance(train_dataset, dict):
            # Multi-dataset case: create loss function for each dataset
            losses = {}
            
            for dataset_name, dataset in train_dataset.items():
                # Get target column for this dataset
                target_column = self.evaluator_factory._get_dataset_target_column(dataset)
                
                # Apply column filtering: keep only 2 input columns + target column
                column_names = dataset.column_names
                if len(column_names) >= 3:
                    input_columns = [col for col in column_names if col != target_column][:2]
                    columns_to_keep = input_columns + [target_column]
                    filtered_dataset = dataset.select_columns(columns_to_keep)
                    logger.info(f"Reranker数据集 '{dataset_name}' 列过滤: {column_names} → {columns_to_keep}")
                else:
                    filtered_dataset = dataset
                
                # Update the dataset in the training data
                train_dataset[dataset_name] = filtered_dataset
                
                # Create loss function based on target column type
                loss_func = self._create_loss_for_dataset(model, filtered_dataset, target_column, dataset_name)
                losses[dataset_name] = loss_func
            
            logger.info(f"为多个reranker数据集创建了损失函数: {list(losses.keys())}")
            return losses
        else:
            # Single dataset case
            target_column = self.evaluator_factory._get_dataset_target_column(train_dataset)
            
            # Apply column filtering
            column_names = train_dataset.column_names
            if len(column_names) >= 3:
                input_columns = [col for col in column_names if col != target_column][:2]
                columns_to_keep = input_columns + [target_column]
                train_dataset = train_dataset.select_columns(columns_to_keep)
                logger.info(f"单Reranker数据集列过滤: {column_names} → {columns_to_keep}")
            
            # Create single loss function
            loss = self._create_loss_for_dataset(model, train_dataset, target_column)
            logger.info("单reranker数据集损失函数创建完成")
            return loss
    
    def _create_loss_for_dataset(self, model: CrossEncoder, dataset: Any, target_column: str, dataset_name: str = None) -> Any:
        """Create loss function for a specific dataset."""
        try:
            # Check target column data type
            if target_column in dataset.column_names:
                # Sample a few values to determine data type
                sample_values = dataset[target_column][:min(10, len(dataset))]
                
                # Check if all values are integers (classification task)
                is_classification = all(isinstance(val, (int, bool)) or 
                                      (isinstance(val, float) and val.is_integer()) 
                                      for val in sample_values)
                
                if is_classification:
                    loss = BinaryCrossEntropyLoss(model)
                    loss_name = "BinaryCrossEntropyLoss（分类任务）"
                else:
                    loss = MSELoss(model)
                    loss_name = "MSELoss（回归任务）"
            else:
                # Default to MSE if no target column
                loss = MSELoss(model)
                loss_name = "MSELoss（默认）"
            
            dataset_info = f"数据集 '{dataset_name}'" if dataset_name else "数据集"
            logger.info(f"为 {dataset_info} 创建损失函数: {loss_name}")
            return loss
            
        except Exception as e:
            logger.warning(f"损失函数创建失败: {e}，使用默认 MSELoss")
            return MSELoss(model)
    
    def create_training_args(self, config: Dict[str, Any]) -> CrossEncoderTrainingArguments:
        """
        Create CrossEncoderTrainingArguments.

        Args:
            config: Training configuration dictionary (已经通过base_trainer进行了GPU兼容性检测)

        Returns:
            CrossEncoderTrainingArguments instance
        """
        logger.info("创建 CrossEncoderTrainingArguments")

        try:
            args = CrossEncoderTrainingArguments(**config)
            logger.info("CrossEncoderTrainingArguments 创建成功")
            logger.info(f"🔍 [DEBUG] 混合精度设置: bf16={getattr(args, 'bf16', None)}, fp16={getattr(args, 'fp16', None)}")
            return args
        except Exception as e:
            logger.error(f"CrossEncoderTrainingArguments 创建失败: {e}")
            raise
    
    def create_trainer_instance(self, model, args, train_dataset, eval_dataset, loss, evaluator) -> CrossEncoderTrainer:
        """
        Create CrossEncoderTrainer instance.
        
        Args:
            model: CrossEncoder model
            args: CrossEncoderTrainingArguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            loss: Loss function
            evaluator: Evaluator (optional)
            
        Returns:
            CrossEncoderTrainer instance
        """
        logger.info("创建 CrossEncoderTrainer")
        
        # Build trainer arguments
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "loss": loss,
        }
        
        # Add evaluation dataset if available
        if eval_dataset is not None:
            # Apply column filtering to evaluation dataset
            if isinstance(eval_dataset, dict):
                # Multi-dataset evaluation
                filtered_eval_dataset = {}
                for dataset_name, dataset in eval_dataset.items():
                    if dataset is not None:
                        target_column = self.evaluator_factory._get_dataset_target_column(dataset)
                        column_names = dataset.column_names
                        if len(column_names) >= 3:
                            input_columns = [col for col in column_names if col != target_column][:2]
                            columns_to_keep = input_columns + [target_column]
                            filtered_eval_dataset[dataset_name] = dataset.select_columns(columns_to_keep)
                            logger.info(f"Reranker验证数据集 '{dataset_name}' 列过滤: {column_names} → {columns_to_keep}")
                        else:
                            filtered_eval_dataset[dataset_name] = dataset
                
                trainer_kwargs["eval_dataset"] = filtered_eval_dataset
            else:
                # Single dataset evaluation
                target_column = self.evaluator_factory._get_dataset_target_column(eval_dataset)
                column_names = eval_dataset.column_names
                if len(column_names) >= 3:
                    input_columns = [col for col in column_names if col != target_column][:2]
                    columns_to_keep = input_columns + [target_column]
                    filtered_eval_dataset = eval_dataset.select_columns(columns_to_keep)
                    logger.info(f"单验证数据集列过滤: {column_names} → {columns_to_keep}")
                else:
                    filtered_eval_dataset = eval_dataset
                
                trainer_kwargs["eval_dataset"] = filtered_eval_dataset
        
        # Add evaluator if available
        if evaluator is not None:
            trainer_kwargs["evaluator"] = evaluator
        
        try:
            trainer = CrossEncoderTrainer(**trainer_kwargs)
            logger.info("CrossEncoderTrainer 创建成功")
            return trainer
        except Exception as e:
            logger.error(f"CrossEncoderTrainer 创建失败: {e}")
            raise
    
    def create_evaluator(self, eval_dataset: Any) -> Optional[Any]:
        """
        Create evaluator for reranker model evaluation.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluator instance or None if no evaluation needed
        """
        logger.info("创建 reranker 评估器")
        
        try:
            # Use the unified evaluator factory with correct API
            evaluator = self.evaluator_factory.create_evaluator(
                dataset=eval_dataset,
                target_column="label",  # Default for reranker
                name=f"reranker_evaluator"
            )
            
            if evaluator:
                logger.info("Reranker 评估器创建成功")
            else:
                logger.info("无需创建评估器")
            
            return evaluator
            
        except Exception as e:
            logger.warning(f"评估器创建失败: {e}，跳过评估")
            return None