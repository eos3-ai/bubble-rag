"""
Embedding model trainer.

Specialized trainer for sentence embedding models using SentenceTransformers.
"""

import logging
import os
from typing import Any, Dict, Optional, Union

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss, ContrastiveLoss, MultipleNegativesRankingLoss, CoSENTLoss

from ..core.base_trainer import BaseTrainer
from ..utils.common_utils import create_embedding_loss

logger = logging.getLogger(__name__)


class EmbeddingTrainer(BaseTrainer):
    """
    Specialized trainer for embedding models.
    
    Handles SentenceTransformer model training with appropriate loss functions
    and training arguments.
    """
    
    def initialize_model(self, model_name: str) -> SentenceTransformer:
        """
        Initialize SentenceTransformer model.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Initialized SentenceTransformer model
        """
        logger.info(f"初始化 SentenceTransformer 模型: {model_name}")
        
        try:
            # Try ModelScope download first (if available)
            try:
                from modelscope import snapshot_download
                from bubble_rag.server_config import TRAINING_CACHE
                
                logger.info(f"尝试从 ModelScope 下载模型: {model_name}")
                model_dir = snapshot_download(model_name, cache_dir=TRAINING_CACHE)
                model = SentenceTransformer(model_dir)
                logger.info(f"✅ ModelScope 下载成功，路径: {model_dir}")
                
            except ImportError:
                logger.info("ModelScope 未安装，使用 HuggingFace 方式")
                model = SentenceTransformer(model_name)
                logger.info(f"✅ HuggingFace 下载成功: {model_name}")
                
            except Exception as e:
                # Fallback to HuggingFace if ModelScope fails
                logger.warning(f"ModelScope 下载失败: {e}，回退到 HuggingFace")
                model = SentenceTransformer(model_name)
                logger.info(f"✅ HuggingFace 下载成功: {model_name}")
            
            logger.info(f"SentenceTransformer 模型初始化成功: {model_name}")
            return model
            
        except Exception as e:
            # Try offline mode as last resort
            if "couldn't connect" in str(e).lower() or "connection" in str(e).lower():
                logger.warning(f"网络连接失败，尝试使用本地缓存加载模型: {model_name}")
                try:
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    model = SentenceTransformer(model_name, local_files_only=True)
                    logger.info(f"✅ 本地缓存加载成功: {model_name}")
                    return model
                except Exception as cache_error:
                    error_msg = f"embedding模型初始化失败: {model_name}，网络不可用且本地缓存未找到。错误: {str(cache_error)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"embedding模型初始化失败: {model_name}, 错误: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    def create_loss_function(self, model: SentenceTransformer, train_dataset: Any) -> Any:
        """
        Create appropriate loss function for embedding training.
        
        Args:
            model: The SentenceTransformer model
            train_dataset: Training dataset (can be single dataset or dict of datasets)
            
        Returns:
            Loss function or dict of loss functions for multi-dataset training
        """
        logger.info("创建 embedding 损失函数")
        
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
                    logger.info(f"Embedding数据集 '{dataset_name}' 列过滤: {column_names} → {columns_to_keep}")
                else:
                    filtered_dataset = dataset
                
                # Update the dataset in the training data
                train_dataset[dataset_name] = filtered_dataset
                
                # Create loss function for this dataset
                loss_func = create_embedding_loss(model, filtered_dataset, target_column, dataset_name)
                losses[dataset_name] = loss_func
            
            logger.info(f"为多个embedding数据集创建了损失函数: {list(losses.keys())}")
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
                logger.info(f"单Embedding数据集列过滤: {column_names} → {columns_to_keep}")
            
            # Create single loss function
            loss = create_embedding_loss(model, train_dataset, target_column)
            logger.info("单embedding数据集损失函数创建完成")
            return loss
    
    def create_training_args(self, config: Dict[str, Any]) -> SentenceTransformerTrainingArguments:
        """
        Create SentenceTransformerTrainingArguments.

        Args:
            config: Training configuration dictionary (已经通过base_trainer进行了GPU兼容性检测)

        Returns:
            SentenceTransformerTrainingArguments instance
        """
        logger.info("创建 SentenceTransformerTrainingArguments")

        try:
            args = SentenceTransformerTrainingArguments(**config)
            logger.info("SentenceTransformerTrainingArguments 创建成功")

            # 调试：记录实际的训练参数值
            logger.info(f"🔍 [DEBUG] 实际训练参数: metric_for_best_model={getattr(args, 'metric_for_best_model', None)}, "
                       f"load_best_model_at_end={getattr(args, 'load_best_model_at_end', None)}, "
                       f"greater_is_better={getattr(args, 'greater_is_better', None)}")
            logger.info(f"🔍 [DEBUG] 混合精度设置: bf16={getattr(args, 'bf16', None)}, fp16={getattr(args, 'fp16', None)}")

            return args
        except Exception as e:
            logger.error(f"SentenceTransformerTrainingArguments 创建失败: {e}")
            raise
    
    def create_trainer_instance(self, model, args, train_dataset, eval_dataset, loss, evaluator) -> SentenceTransformerTrainer:
        """
        Create SentenceTransformerTrainer instance.
        
        Args:
            model: SentenceTransformer model
            args: SentenceTransformerTrainingArguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            loss: Loss function
            evaluator: Evaluator (optional)
            
        Returns:
            SentenceTransformerTrainer instance
        """
        logger.info("创建 SentenceTransformerTrainer")
        
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
                            logger.info(f"Embedding验证数据集 '{dataset_name}' 列过滤: {column_names} → {columns_to_keep}")
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
            trainer = SentenceTransformerTrainer(**trainer_kwargs)
            logger.info("SentenceTransformerTrainer 创建成功")
            return trainer
        except Exception as e:
            logger.error(f"SentenceTransformerTrainer 创建失败: {e}")
            raise
    
    def create_evaluator(self, eval_dataset: Any) -> Optional[Any]:
        """
        Create evaluator for embedding model evaluation.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluator instance or None if no evaluation needed
        """
        logger.info("创建 embedding 评估器")
        
        try:
            # Use the unified evaluator factory with correct API
            evaluator = self.evaluator_factory.create_evaluator(
                dataset=eval_dataset,
                target_column="score",  # Default for embedding
                name=f"embedding_evaluator"
            )
            
            if evaluator:
                logger.info("Embedding 评估器创建成功")
            else:
                logger.info("无需创建评估器")
            
            return evaluator
            
        except Exception as e:
            logger.warning(f"评估器创建失败: {e}，跳过评估")
            return None