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
        
        logger.info(f"✅ Loss管理器初始化完成: {self.logs_dir}")
    
    def _ensure_directories(self):
        """确保目录结构存在"""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 创建日志目录: {self.logs_dir}")
        except Exception as e:
            logger.error(f"❌ 创建日志目录失败: {e}")
            raise
    
    def save_loss_record(self, step: int, metrics: Dict[str, Any], epoch: Optional[float] = None):
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
                
                logger.debug(f"📊 保存loss记录: step={step}, metrics={list(metrics.keys())}")
                
            except Exception as e:
                logger.error(f"❌ 保存loss记录失败: {e}")
    
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
            logger.error(f"❌ 更新训练指标失败: {e}")
    
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
                
                logger.info(f"📊 Epoch {epoch} 汇总已保存: {list(epoch_metrics.keys())}")
                
            except Exception as e:
                logger.error(f"❌ 保存epoch汇总失败: {e}")
    
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
            logger.error(f"❌ 获取loss历史失败: {e}")
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
            logger.error(f"❌ 获取训练指标失败: {e}")
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
            logger.error(f"❌ 生成数据库汇总信息失败: {e}")
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
                
                # 保存最终指标
                with open(self.training_metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_metrics, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✅ 训练指标已完成保存: {self.training_metrics_file}")
                
                # 返回数据库汇总信息供调用者使用
                return self.get_database_summary()
                
            except Exception as e:
                logger.error(f"❌ 完成训练指标保存失败: {e}")
                return None


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
                logger.error(f"❌ 清理loss管理器失败: {e}")
                return None
        return None