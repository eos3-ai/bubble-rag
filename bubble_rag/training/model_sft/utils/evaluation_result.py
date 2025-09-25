"""
标准化评估结果处理

提供统一的评估结果包装和处理功能
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from .metric_registry import get_metric_registry, MetricRegistry

@dataclass
class EvaluationResult:
    """标准化评估结果"""
    source_id: str                    # 数据源ID
    dataset_name: str                # 数据集名称
    evaluator_name: str              # 评估器名称
    step: int                        # 训练步数
    epoch: Optional[float] = None    # 训练轮数
    metrics: Dict[str, float] = None # 评估指标
    metadata: Dict[str, Any] = None  # 元数据

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}

    def add_metric(self, name: str, value: float):
        """添加评估指标"""
        self.metrics[name] = value

    def get_standardized_metrics(self) -> Dict[str, float]:
        """获取标准化的指标名称 (eval_{source_id}_{metric})"""
        standardized = {}
        for metric_name, value in self.metrics.items():
            standardized_name = f"eval_{self.source_id}_{metric_name}"
            standardized[standardized_name] = value
        return standardized

    def get_frontend_format(self, registry: MetricRegistry = None) -> Dict[str, Any]:
        """获取前端格式的数据"""
        if registry is None:
            registry = get_metric_registry()

        # 获取元数据
        metric_names = list(self.metrics.keys())
        frontend_metadata = registry.get_frontend_metadata(metric_names)

        return {
            "step": self.step,
            "epoch": self.epoch,
            "source_id": self.source_id,
            "dataset_name": self.dataset_name,
            "evaluator_name": self.evaluator_name,
            "metrics": self.metrics,
            "standardized_metrics": self.get_standardized_metrics(),
            "metadata": {
                **self.metadata,
                **frontend_metadata
            }
        }

class EvaluationResultProcessor:
    """评估结果处理器"""

    def __init__(self):
        self.registry = get_metric_registry()

    def extract_evaluation_results_from_logs(self, loss_data: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """从loss日志数据中提取评估结果"""
        results = []

        for record in loss_data:
            step = record.get('step', 0)
            epoch = record.get('epoch')

            # 按数据源分组指标
            source_metrics = self._group_metrics_by_source(record)

            for source_id, metrics in source_metrics.items():
                if metrics:  # 只处理有指标的数据源
                    # 推断评估器信息
                    evaluator_name = self._infer_evaluator_name(metrics)

                    result = EvaluationResult(
                        source_id=source_id,
                        dataset_name=f"source_{source_id}",  # 默认名称，后续会被映射
                        evaluator_name=evaluator_name,
                        step=step,
                        epoch=epoch,
                        metrics=metrics
                    )

                    results.append(result)

        return results

    def _group_metrics_by_source(self, record: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """按数据源分组指标"""
        source_metrics = {}

        for key, value in record.items():
            if key.startswith('eval_') and isinstance(value, (int, float)):
                # 解析 eval_{source_id}_{metric_name} 格式
                parts = key[5:].split('_')  # 去掉 'eval_' 前缀
                if len(parts) >= 2 and parts[0].isdigit():
                    source_id = parts[0]
                    metric_name = '_'.join(parts[1:])

                    if source_id not in source_metrics:
                        source_metrics[source_id] = {}

                    source_metrics[source_id][metric_name] = value

        return source_metrics

    def _infer_evaluator_name(self, metrics: Dict[str, float]) -> str:
        """从指标推断评估器名称"""
        metric_names = list(metrics.keys())
        evaluator_type = self.registry.infer_evaluator_type_from_metrics(metric_names)

        if evaluator_type:
            # 查找匹配的评估器
            for evaluator_name, evaluator_info in self.registry.evaluators.items():
                if evaluator_info.evaluator_type == evaluator_type:
                    return evaluator_name

        return "UnknownEvaluator"

    def apply_dataset_mapping(self, results: List[EvaluationResult],
                            source_mapping: Dict[str, str]) -> List[EvaluationResult]:
        """应用数据集名称映射"""
        for result in results:
            if result.source_id in source_mapping:
                result.dataset_name = source_mapping[result.source_id]

        return results

    def convert_to_frontend_format(self, results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """转换为前端格式"""
        return [result.get_frontend_format(self.registry) for result in results]

    def enhance_loss_data_with_metadata(self, loss_data: List[Dict[str, Any]],
                                      source_mapping: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """为loss数据增强元数据"""
        enhanced_data = []

        for record in loss_data:
            enhanced_record = record.copy()

            # 提取评估指标
            eval_metrics = {}
            for key, value in record.items():
                if key.startswith('eval_') and isinstance(value, (int, float)):
                    eval_metrics[key] = value

            if eval_metrics:
                # 获取元数据
                metric_names = []
                for key in eval_metrics.keys():
                    # 从 eval_{source_id}_{metric_name} 提取 metric_name
                    parts = key[5:].split('_')  # 去掉 'eval_' 前缀
                    if len(parts) >= 2:
                        metric_name = '_'.join(parts[1:])
                        metric_names.append(metric_name)

                if metric_names:
                    frontend_metadata = self.registry.get_frontend_metadata(metric_names)
                    enhanced_record['evaluation_metadata'] = frontend_metadata

            enhanced_data.append(enhanced_record)

        return enhanced_data

# 全局处理器实例
evaluation_result_processor = EvaluationResultProcessor()

def get_evaluation_result_processor() -> EvaluationResultProcessor:
    """获取全局评估结果处理器"""
    return evaluation_result_processor

def save_evaluation_results_to_database(task_id: str, eval_results: Dict[str, Any], step: int, epoch: Optional[float] = None, data_source_mapping: Dict[str, str] = None):
    """保存评估器结果到数据库"""
    try:
        from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
        import logging
        logger = logging.getLogger(__name__)

        # 步骤1: 统一格式转换 - 将所有非标准格式转换为 eval_{source_id}_{metric} 格式
        converted_results = _convert_all_to_standard_format(eval_results, data_source_mapping)
        logger.debug(f"格式转换完成: {list(converted_results.keys())}")

        # 步骤2a: 先处理loss数据 - 保存到loss字段
        loss_results = {k: v for k, v in converted_results.items() if 'loss' in k.lower()}
        if loss_results:
            loss_source_results = _split_by_source_id(loss_results)
            eval_datasets = TrainingDatasetService.get_datasets_by_job_and_split(task_id, "eval")
            for source_id, source_loss_data in loss_source_results.items():
                matching_dataset = None
                for dataset in eval_datasets:
                    if dataset.get("data_source_id") == source_id:
                        matching_dataset = dataset
                        break

                if matching_dataset:
                    dataset_id = matching_dataset["id"]
                    for loss_key, loss_value in source_loss_data.items():
                        if 'loss' in loss_key.lower() and isinstance(loss_value, (int, float)):
                            TrainingDatasetService.add_loss_record(
                                dataset_id=dataset_id,
                                loss_value=loss_value,
                                step=step,
                                epoch=epoch
                            )
                            logger.debug(f"💾 数据源{source_id}的loss已保存: dataset_id={dataset_id}, step={step}, {loss_key}={loss_value}")

        # 步骤2b: 筛除loss - 只保留非loss的评估指标
        non_loss_results = {k: v for k, v in converted_results.items() if 'loss' not in k.lower()}
        logger.debug(f"🗑️ 筛除loss后: {list(non_loss_results.keys())}")

        # 步骤3: 按数据源分离
        source_eval_results = _split_by_source_id(non_loss_results)
        logger.debug(f"按数据源分离: {list(source_eval_results.keys())}")

        # 步骤4: 为每个数据源保存评估指标到数据库
        eval_datasets = TrainingDatasetService.get_datasets_by_job_and_split(task_id, "eval")
        for source_id, source_results in source_eval_results.items():
            matching_dataset = None
            for dataset in eval_datasets:
                if dataset.get("data_source_id") == source_id:
                    matching_dataset = dataset
                    break

            if matching_dataset:
                dataset_id = matching_dataset["id"]
                TrainingDatasetService.add_training_evaluator_evaluation(
                    dataset_id=dataset_id,
                    eval_results=source_results,
                    step=step,
                    epoch=epoch
                )
                logger.debug(f"💾 数据源{source_id}评估结果已保存: dataset_id={dataset_id}, step={step}, metrics={list(source_results.keys())}")

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"保存评估结果到数据库失败: {e}")

def _convert_all_to_standard_format(eval_results: Dict[str, Any], data_source_mapping: Dict[str, str] = None) -> Dict[str, Any]:
    """步骤1: 将所有指标转换为标准格式"""
    converted = {}

    # 处理数据集名称映射
    dataset_to_source = {}
    if data_source_mapping:
        for full_name, source_id in data_source_mapping.items():
            if '/' in full_name:
                dataset_name = full_name.split('/')[-1]
                dataset_to_source[dataset_name] = source_id
            dataset_to_source[full_name] = source_id

    for key, value in eval_results.items():
        if key.startswith('eval_'):
            # 检查是否已经是标准格式 eval_{数字}_*
            remaining = key[5:]  # 去掉'eval_'
            parts = remaining.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                # 已经是标准格式，直接保留
                converted[key] = value
            else:
                # 非标准格式，需要转换
                converted_key = None
                for dataset_name, source_id in dataset_to_source.items():
                    if dataset_name in key:
                        # 提取指标名
                        dataset_pos = key.find(dataset_name)
                        metric_suffix = key[dataset_pos + len(dataset_name):]
                        if metric_suffix.startswith('_'):
                            metric_suffix = metric_suffix[1:]

                        converted_key = f"eval_{source_id}_{metric_suffix}"
                        break

                if converted_key:
                    converted[converted_key] = value
                else:
                    # 无法转换，保持原样（如统一指标）
                    converted[key] = value
        else:
            # 非eval字段，直接保留
            converted[key] = value

    return converted

def _split_by_source_id(eval_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """步骤3: 按数据源ID分离（所有数据已经是标准格式）"""
    source_results = {}
    unified_metrics = {}

    # 收集所有source_id
    source_ids = set()
    for key in eval_results.keys():
        if key.startswith('eval_'):
            parts = key[5:].split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                source_ids.add(parts[0])

    for key, value in eval_results.items():
        if key.startswith('eval_'):
            parts = key[5:].split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                # 标准格式：分配到对应数据源
                source_id = parts[0]
                if source_id not in source_results:
                    source_results[source_id] = {}
                source_results[source_id][key] = value
            else:
                # 统一指标：添加到所有数据源
                metric_name = key[5:]  # 去掉eval_前缀
                unified_metrics[metric_name] = value
        else:
            # 非eval字段（如epoch）：添加到所有数据源
            unified_metrics[key] = value

    # 将统一指标添加到所有数据源
    for source_id in source_results.keys():
        source_results[source_id].update(unified_metrics)

    return source_results

def _separate_eval_results_by_source(eval_results: Dict[str, Any], data_source_mapping: Dict[str, str] = None) -> Dict[str, Dict[str, Any]]:
    """按数据源分离评估结果"""
    source_eval_results = {}

    # 构建数据集名称到source_id的映射（通用方法，和本地文件逻辑一致）
    dataset_to_source_mapping = {}
    if data_source_mapping:
        # 从eval keys中提取数据集名称
        eval_keys = [key for key in eval_results.keys() if key.startswith('eval_')]
        dataset_names = set()
        for key in eval_keys:
            # 提取可能的数据集名称（去掉eval_前缀和后面的metric）
            remaining = key[5:]  # 去掉'eval_'
            if not remaining.split('_')[0].isdigit():  # 不是标准格式
                # 可能包含数据集名称，尝试提取
                parts = remaining.split('_')
                for i in range(1, len(parts)):
                    potential_dataset = '_'.join(parts[:i])
                    dataset_names.add(potential_dataset)

        # 建立映射关系
        for dataset_name in dataset_names:
            for full_name, source_id in data_source_mapping.items():
                if dataset_name in full_name:
                    dataset_to_source_mapping[dataset_name] = source_id
                    break

    for key, value in eval_results.items():
        if key.startswith('eval_'):
            assigned_to_source = False
            remaining = key[5:]  # 去掉'eval_'

            # 格式1: eval_{source_id}_{metric} (标准格式)
            parts = remaining.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                source_id = parts[0]
                metric_name = '_'.join(parts[1:])
                if source_id not in source_eval_results:
                    source_eval_results[source_id] = {}
                source_eval_results[source_id][metric_name] = value
                assigned_to_source = True

            # 格式2: 其他格式，通过dataset名称匹配，转换为标准格式
            elif not assigned_to_source and dataset_to_source_mapping:
                for dataset_name, source_id in dataset_to_source_mapping.items():
                    if dataset_name in key:
                        # 提取指标名并转换为标准格式
                        # eval_sentence-transformers/stsb_runtime -> eval_2_runtime
                        remaining = key[5:]  # 去掉'eval_'
                        if dataset_name in remaining:
                            # 找到数据集名称位置，提取后面的指标名
                            dataset_pos = remaining.find(dataset_name)
                            metric_suffix = remaining[dataset_pos + len(dataset_name):]
                            if metric_suffix.startswith('_'):
                                metric_suffix = metric_suffix[1:]  # 去掉开头的下划线

                            # 生成标准格式的key
                            standard_key = f"eval_{source_id}_{metric_suffix}"
                        else:
                            # 备用方案
                            metric_suffix = key.split('_')[-1]
                            standard_key = f"eval_{source_id}_{metric_suffix}"

                        if source_id not in source_eval_results:
                            source_eval_results[source_id] = {}
                        source_eval_results[source_id][standard_key] = value
                        assigned_to_source = True
                        break

    # 处理统一指标（如eval_sequential_score），添加到所有数据源中
    unified_metrics = {}
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            assigned_to_any_source = False

            # 检查是否已被分配到某个数据源
            # 1. 检查标准格式 eval_{source_id}_*
            parts = key[5:].split('_')
            if len(parts) >= 2 and parts[0].isdigit() and parts[0] in source_eval_results:
                assigned_to_any_source = True

            # 2. 检查非标准格式（通过数据集名称匹配）
            if not assigned_to_any_source and dataset_to_source_mapping:
                for dataset_name in dataset_to_source_mapping.keys():
                    if dataset_name in key:
                        assigned_to_any_source = True
                        break

            # 如果没有被分配到任何数据源，认为是统一指标
            if not assigned_to_any_source:
                metric_name = key[5:]  # 去掉eval_前缀
                unified_metrics[metric_name] = value

    # 处理非eval字段（如epoch），添加到所有数据源中
    non_eval_fields = {}
    for key, value in eval_results.items():
        if not key.startswith('eval_'):
            non_eval_fields[key] = value

    # 将统一指标和非eval字段添加到所有数据源中
    for source_id in source_eval_results.keys():
        # 添加统一指标
        for metric_name, metric_value in unified_metrics.items():
            source_eval_results[source_id][metric_name] = metric_value
        # 添加非eval字段（如epoch）
        for field_name, field_value in non_eval_fields.items():
            source_eval_results[source_id][field_name] = field_value

    return source_eval_results