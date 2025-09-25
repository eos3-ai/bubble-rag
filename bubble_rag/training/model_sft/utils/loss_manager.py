"""
训练Loss本地文件管理器
负责将训练过程中的loss数据保存到本地文件，避免数据库性能问题
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock

logger = logging.getLogger(__name__)


class LossManager:
    """训练Loss本地文件管理器"""
    
    def __init__(self, output_dir: str, task_id: str):
        """
        初始化Loss管理器
        
        Args:
            output_dir: 训练输出目录
            task_id: 训练任务ID
        """
        self.output_dir = output_dir
        self.task_id = task_id
        self.lock = Lock()
        
        # 设置日志目录结构: {output_dir}/logs/training/{task_id}/
        self.logs_dir = Path(output_dir) / "logs" / "training" / task_id
        self.loss_history_file = self.logs_dir / "loss_history.jsonl"
        self.training_metrics_file = self.logs_dir / "training_metrics.json"
        self.metadata_file = self.logs_dir / "metadata.json"  # 新增：元数据文件
        
        # 创建目录
        self._ensure_directories()
        
        # 初始化训练指标
        self.training_metrics = {
            "task_id": task_id,
            "start_time": datetime.now().isoformat(),
            "total_steps": 0,
            "epochs_completed": 0,
            "best_train_loss": None,
            "best_eval_loss": None,
            "loss_records_count": 0,
            "last_updated": None,
            "epoch_summaries": [],  # 每个epoch的汇总信息
            "final_results": {}     # 最终评估结果
        }
        
        logger.info(f"Loss管理器初始化完成: {self.logs_dir}")
    
    def _ensure_directories(self):
        """确保目录结构存在"""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"创建日志目录: {self.logs_dir}")
        except Exception as e:
            logger.error(f"创建日志目录失败: {e}")
            raise
    
    def save_loss_record(self, step: int, metrics: Dict[str, Any], epoch: Optional[float] = None, save_to_db: bool = True, data_source_mapping: Dict[str, str] = None):
        """
        保存单次loss记录到JSONL文件
        
        Args:
            step: 训练步数
            metrics: 指标字典（包含train_loss, eval_loss等）
            epoch: 当前epoch（可选）
        """
        with self.lock:
            try:
                # 构建记录
                record = {
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                    **metrics
                }
                
                if epoch is not None:
                    record["epoch"] = epoch
                
                # 写入JSONL文件（每行一个JSON）
                with open(self.loss_history_file, 'a', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
                
                # 更新训练指标
                self._update_training_metrics(step, metrics, epoch)

                # 保存到数据库
                if save_to_db:
                    self._save_loss_to_database(metrics, step, epoch, data_source_mapping)

                logger.debug(f"保存loss记录: step={step}, metrics={list(metrics.keys())}")

            except Exception as e:
                logger.error(f"保存loss记录失败: {e}")

    def save_or_merge_loss_record(self, step: int, metrics: Dict[str, Any], epoch: Optional[float] = None, merge_key: str = None):
        """
        保存或合并loss记录，支持同step内的合并

        Args:
            step: 训练步数
            metrics: 指标字典
            epoch: 当前epoch
            merge_key: 合并键，用于标识可以合并的记录
        """
        with self.lock:
            try:
                # 如果没有提供merge_key，直接保存
                if merge_key is None:
                    self.save_loss_record(step, metrics, epoch)
                    return

                # 尝试读取现有记录并查找可合并的记录
                existing_records = []
                merged = False

                if self.loss_history_file.exists():
                    with open(self.loss_history_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                existing_records.append(json.loads(line.strip()))

                # 查找同step的记录进行合并
                for i, record in enumerate(existing_records):
                    if (record.get('step') == step and
                        record.get('epoch') == epoch):

                        # 检查是否是同一数据源（通过指标前缀判断）
                        record_source_ids = set()
                        new_source_ids = set()

                        for key in record.keys():
                            if key.startswith('eval_') and '_' in key[5:]:
                                parts = key[5:].split('_')
                                if parts[0].isdigit():
                                    record_source_ids.add(parts[0])

                        for key in metrics.keys():
                            if key.startswith('eval_') and '_' in key[5:]:
                                parts = key[5:].split('_')
                                if parts[0].isdigit():
                                    new_source_ids.add(parts[0])

                        # 如果有相同的source_id，则合并
                        if record_source_ids & new_source_ids:
                            existing_records[i].update(metrics)
                            merged = True
                            logger.info(f"🔗 合并同step记录: step={step}, source_ids={new_source_ids}")
                            break

                if not merged:
                    # 没有找到可合并的记录，添加新记录
                    new_record = {
                        "step": step,
                        "timestamp": datetime.now().isoformat(),
                        **metrics
                    }
                    if epoch is not None:
                        new_record["epoch"] = epoch
                    existing_records.append(new_record)
                    logger.info(f"添加新记录: step={step}, metrics={list(metrics.keys())}")

                # 重写整个文件
                with open(self.loss_history_file, 'w', encoding='utf-8') as f:
                    for record in existing_records:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')

                # 更新训练指标（只在添加新记录时更新）
                if not merged:
                    self._update_training_metrics(step, metrics, epoch)

            except Exception as e:
                logger.error(f"保存/合并loss记录失败: {e}")
                # 回退到普通保存
                self.save_loss_record(step, metrics, epoch)

    def _update_training_metrics(self, step: int, metrics: Dict[str, Any], epoch: Optional[float]):
        """更新训练指标汇总"""
        try:
            # 更新基础信息
            self.training_metrics["total_steps"] = max(self.training_metrics["total_steps"], step)
            self.training_metrics["loss_records_count"] += 1
            self.training_metrics["last_updated"] = datetime.now().isoformat()
            
            if epoch is not None:
                self.training_metrics["epochs_completed"] = max(
                    self.training_metrics["epochs_completed"], epoch
                )
            
            # 更新最佳loss
            if "train_loss" in metrics:
                train_loss = metrics["train_loss"]
                if (self.training_metrics["best_train_loss"] is None or 
                    train_loss < self.training_metrics["best_train_loss"]):
                    self.training_metrics["best_train_loss"] = train_loss
            
            if "eval_loss" in metrics:
                eval_loss = metrics["eval_loss"]
                if (self.training_metrics["best_eval_loss"] is None or 
                    eval_loss < self.training_metrics["best_eval_loss"]):
                    self.training_metrics["best_eval_loss"] = eval_loss
            
            # 保存到文件
            with open(self.training_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_metrics, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"更新训练指标失败: {e}")
    
    def finalize_epoch(self, epoch: int, epoch_metrics: Dict[str, Any]):
        """
        完成一个epoch，保存epoch汇总信息
        
        Args:
            epoch: epoch编号
            epoch_metrics: epoch汇总指标
        """
        with self.lock:
            try:
                epoch_summary = {
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    **epoch_metrics
                }
                
                # 添加到epoch汇总列表
                self.training_metrics["epoch_summaries"].append(epoch_summary)
                
                # 保存到文件
                with open(self.training_metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_metrics, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Epoch {epoch} 汇总已保存: {list(epoch_metrics.keys())}")
                
            except Exception as e:
                logger.error(f"保存epoch汇总失败: {e}")
    
    def get_loss_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取loss历史记录
        
        Args:
            limit: 限制返回的记录数量
            
        Returns:
            loss记录列表
        """
        try:
            if not self.loss_history_file.exists():
                return []
            
            records = []
            with open(self.loss_history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            
            # 应用限制
            if limit and len(records) > limit:
                records = records[-limit:]  # 返回最近的记录
            
            return records
            
        except Exception as e:
            logger.error(f"获取loss历史失败: {e}")
            return []
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """获取训练指标汇总"""
        try:
            if self.training_metrics_file.exists():
                with open(self.training_metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.training_metrics
        except Exception as e:
            logger.error(f"获取训练指标失败: {e}")
            return self.training_metrics
    
    def get_database_summary(self) -> Dict[str, Any]:
        """
        获取用于数据库存储的汇总信息
        
        Returns:
            数据库汇总信息字典
        """
        try:
            summary = {
                "task_id": self.task_id,
                "loss_file_path": str(self.loss_history_file),
                "metrics_file_path": str(self.training_metrics_file),
                "total_records": self.training_metrics.get("loss_records_count", 0),
                "training_duration_seconds": self.training_metrics.get("training_duration_seconds"),
                "best_train_loss": self.training_metrics.get("best_train_loss"),
                "best_eval_loss": self.training_metrics.get("best_eval_loss"),
                "total_steps": self.training_metrics.get("total_steps", 0),
                "epochs_completed": self.training_metrics.get("epochs_completed", 0),
                "start_time": self.training_metrics.get("start_time"),
                "end_time": self.training_metrics.get("end_time"),
                "status": self.training_metrics.get("status", "running"),
                "epoch_summaries": self.training_metrics.get("epoch_summaries", []),
                "final_metrics": self.training_metrics.get("final_metrics", {})
            }
            return summary
        except Exception as e:
            logger.error(f"生成数据库汇总信息失败: {e}")
            return {}

    def finalize_training(self, final_metrics: Optional[Dict[str, Any]] = None):
        """
        完成训练，保存最终指标
        
        Args:
            final_metrics: 最终训练指标
        """
        with self.lock:
            try:
                # 更新完成时间
                self.training_metrics["end_time"] = datetime.now().isoformat()
                self.training_metrics["status"] = "completed"
                
                # 添加最终指标
                if final_metrics:
                    self.training_metrics["final_metrics"] = final_metrics
                
                # 计算训练时长
                if "start_time" in self.training_metrics:
                    start_time = datetime.fromisoformat(self.training_metrics["start_time"])
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    self.training_metrics["training_duration_seconds"] = duration

                    # 同时保存格式化的时长
                    self.training_metrics["duration_formatted"] = self._format_duration(duration)

                # 保存最终指标
                with open(self.training_metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_metrics, f, ensure_ascii=False, indent=2)
                
                logger.info(f"训练指标已完成保存: {self.training_metrics_file}")
                
                # 返回数据库汇总信息供调用者使用
                return self.get_database_summary()
                
            except Exception as e:
                logger.error(f"完成训练指标保存失败: {e}")
                return None

    def save_metadata(self, metadata: Dict[str, Any]):
        """
        保存训练元数据到文件

        Args:
            metadata: 元数据字典，包含数据源映射、评估器类型等信息
        """
        with self.lock:
            try:
                # 添加时间戳
                metadata["created_at"] = datetime.now().isoformat()
                metadata["task_id"] = self.task_id

                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                logger.info(f"元数据已保存: {self.metadata_file}")
                logger.debug(f"元数据内容: {list(metadata.keys())}")

            except Exception as e:
                logger.error(f"保存元数据失败: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        读取训练元数据

        Returns:
            元数据字典，如果文件不存在则返回空字典
        """
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.debug(f"读取元数据成功: {list(metadata.keys())}")
                return metadata
            else:
                logger.debug("元数据文件不存在，返回空字典")
                return {}
        except Exception as e:
            logger.error(f"读取元数据失败: {e}")
            return {}

    def _format_duration(self, seconds: Optional[float]) -> Optional[str]:
        """将秒数转换为人类可读格式"""
        if seconds is None:
            return None

        total_seconds = int(seconds)
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if days > 0:
            return f"{days}天{hours}时{minutes}分{secs}秒"
        elif hours > 0:
            return f"{hours}时{minutes}分{secs}秒"
        elif minutes > 0:
            return f"{minutes}分{secs}秒"
        else:
            return f"{secs}秒"

    def _save_loss_to_database(self, loss_metrics: Dict[str, Any], step: int, epoch: Optional[float] = None, data_source_mapping: Dict[str, str] = None):
        """保存loss数据到数据库"""
        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService

            for loss_key, loss_value in loss_metrics.items():
                if 'loss' in loss_key.lower() and isinstance(loss_value, (int, float)):
                    # 训练loss：存储到第一个数据源的训练集
                    if 'train' in loss_key.lower() or loss_key.lower() == 'loss':
                        train_datasets = TrainingDatasetService.get_datasets_by_job_and_split(self.task_id, 'train')
                        if train_datasets:
                            dataset_info = train_datasets[0]
                            dataset_id = dataset_info["id"]
                            TrainingDatasetService.add_loss_record(
                                dataset_id=dataset_id,
                                loss_value=loss_value,
                                step=step,
                                epoch=epoch
                            )
                            logger.debug(f"训练Loss已保存到数据库: dataset_id={dataset_id}, step={step}, {loss_key}={loss_value}")

                    # 评估loss：使用统一的映射逻辑按数据源精确存储
                    elif 'eval' in loss_key.lower() and loss_key.startswith('eval_'):
                        # 使用统一的映射逻辑处理eval loss
                        eval_metrics = {loss_key: loss_value}
                        from bubble_rag.training.model_sft.utils.evaluation_result import _separate_eval_results_by_source
                        source_eval_results = _separate_eval_results_by_source(eval_metrics, data_source_mapping)

                        for source_id, source_results in source_eval_results.items():
                            loss_metrics_for_source = {k: v for k, v in source_results.items() if 'loss' in k}
                            if loss_metrics_for_source:
                                # 获取所有eval数据集并筛选匹配的source_id
                                eval_datasets = TrainingDatasetService.get_datasets_by_job_and_split(self.task_id, "eval")
                                matching_dataset = None
                                for dataset in eval_datasets:
                                    if dataset.get("data_source_id") == source_id:
                                        matching_dataset = dataset
                                        break

                                if matching_dataset:
                                    dataset_id = matching_dataset["id"]
                                    for metric_name, metric_value in loss_metrics_for_source.items():
                                        TrainingDatasetService.add_loss_record(
                                            dataset_id=dataset_id,
                                            loss_value=metric_value,
                                            step=step,
                                            epoch=epoch
                                        )
                                        logger.debug(f"评估Loss已保存到数据库: source_id={source_id}, dataset_id={dataset_id}, step={step}, {metric_name}={metric_value}")
        except Exception as e:
            logger.warning(f"保存loss到数据库失败: {e}")



# 全局loss管理器实例字典 {task_id: LossManager}
_loss_managers: Dict[str, LossManager] = {}
_managers_lock = Lock()


def get_loss_manager(output_dir: str, task_id: str) -> LossManager:
    """获取或创建loss管理器实例"""
    with _managers_lock:
        if task_id not in _loss_managers:
            _loss_managers[task_id] = LossManager(output_dir, task_id)
        return _loss_managers[task_id]


def cleanup_loss_manager(task_id: str, final_metrics: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    清理loss管理器实例
    
    Args:
        task_id: 任务ID
        final_metrics: 最终训练指标
        
    Returns:
        数据库汇总信息，用于存储到数据库
    """
    with _managers_lock:
        if task_id in _loss_managers:
            try:
                # 完成训练并获取汇总信息
                summary = _loss_managers[task_id].finalize_training(final_metrics)
                del _loss_managers[task_id]
                logger.info(f"🧹 清理loss管理器: {task_id}")
                return summary
            except Exception as e:
                logger.error(f"清理loss管理器失败: {e}")
                return None
        return None