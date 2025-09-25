"""
评估指标注册器和抽象化组件

提供统一的评估指标管理、分类和前端展示支持
"""
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

class TaskType(Enum):
    """任务类型枚举"""
    REGRESSION = "regression"      # 回归任务
    CLASSIFICATION = "classification"  # 分类任务

class EvaluatorType(Enum):
    """评估器类型枚举"""
    CORRELATION = "correlation"    # 相关性评估器
    CLASSIFICATION = "classification"  # 分类评估器
    EMBEDDING_SIMILARITY = "embedding_similarity"  # embedding相似度
    BINARY_CLASSIFICATION = "binary_classification"  # 二分类

class MetricCategory(Enum):
    """指标分类枚举"""
    CORRELATION = "correlation"    # 相关性指标
    CLASSIFICATION = "classification"  # 分类指标
    PERFORMANCE = "performance"    # 性能指标
    LOSS = "loss"                 # 损失指标

@dataclass
class MetricInfo:
    """指标信息"""
    name: str                    # 指标名称
    display_name: str           # 显示名称
    category: MetricCategory    # 指标分类
    description: str            # 描述
    range_min: Optional[float] = None  # 最小值
    range_max: Optional[float] = None  # 最大值
    higher_is_better: bool = True      # 数值越高越好

@dataclass
class EvaluatorInfo:
    """评估器信息"""
    name: str                   # 评估器名称
    evaluator_type: EvaluatorType  # 评估器类型
    task_type: TaskType         # 任务类型
    metrics: List[str]          # 支持的指标列表
    description: str            # 描述

class MetricRegistry:
    """评估指标注册器"""

    def __init__(self):
        self.metrics: Dict[str, MetricInfo] = {}
        self.evaluators: Dict[str, EvaluatorInfo] = {}
        self._initialize_default_metrics()
        self._initialize_default_evaluators()

    def _initialize_default_metrics(self):
        """初始化默认指标"""
        # 相关性指标
        self.register_metric(MetricInfo(
            name="pearson",
            display_name="皮尔逊相关系数",
            category=MetricCategory.CORRELATION,
            description="皮尔逊线性相关系数，衡量两个变量的线性关系",
            range_min=-1.0,
            range_max=1.0,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="spearman",
            display_name="斯皮尔曼相关系数",
            category=MetricCategory.CORRELATION,
            description="斯皮尔曼等级相关系数，衡量两个变量的单调关系",
            range_min=-1.0,
            range_max=1.0,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="cosine_pearson",
            display_name="余弦相似度皮尔逊系数",
            category=MetricCategory.CORRELATION,
            description="基于余弦相似度的皮尔逊相关系数",
            range_min=-1.0,
            range_max=1.0,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="cosine_spearman",
            display_name="余弦相似度斯皮尔曼系数",
            category=MetricCategory.CORRELATION,
            description="基于余弦相似度的斯皮尔曼相关系数",
            range_min=-1.0,
            range_max=1.0,
            higher_is_better=True
        ))

        # 分类指标
        self.register_metric(MetricInfo(
            name="accuracy",
            display_name="准确率",
            category=MetricCategory.CLASSIFICATION,
            description="分类准确率，正确分类样本数/总样本数",
            range_min=0.0,
            range_max=1.0,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="f1",
            display_name="F1分数",
            category=MetricCategory.CLASSIFICATION,
            description="F1分数，精确率和召回率的调和平均",
            range_min=0.0,
            range_max=1.0,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="precision",
            display_name="精确率",
            category=MetricCategory.CLASSIFICATION,
            description="精确率，真正例/(真正例+假正例)",
            range_min=0.0,
            range_max=1.0,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="recall",
            display_name="召回率",
            category=MetricCategory.CLASSIFICATION,
            description="召回率，真正例/(真正例+假负例)",
            range_min=0.0,
            range_max=1.0,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="ap",
            display_name="平均精度",
            category=MetricCategory.CLASSIFICATION,
            description="Average Precision，平均精度",
            range_min=0.0,
            range_max=1.0,
            higher_is_better=True
        ))

        # 损失指标
        self.register_metric(MetricInfo(
            name="loss",
            display_name="损失值",
            category=MetricCategory.LOSS,
            description="训练或评估损失值",
            range_min=None,
            range_max=None,
            higher_is_better=False
        ))

        # 性能指标
        self.register_metric(MetricInfo(
            name="runtime",
            display_name="运行时间(秒)",
            category=MetricCategory.PERFORMANCE,
            description="评估运行时间，单位秒",
            range_min=0.0,
            range_max=None,
            higher_is_better=False
        ))

        self.register_metric(MetricInfo(
            name="samples_per_second",
            display_name="样本处理速度",
            category=MetricCategory.PERFORMANCE,
            description="每秒处理的样本数",
            range_min=0.0,
            range_max=None,
            higher_is_better=True
        ))

        self.register_metric(MetricInfo(
            name="steps_per_second",
            display_name="步骤处理速度",
            category=MetricCategory.PERFORMANCE,
            description="每秒处理的步骤数",
            range_min=0.0,
            range_max=None,
            higher_is_better=True
        ))

    def _initialize_default_evaluators(self):
        """初始化默认评估器"""
        # CrossEncoderCorrelationEvaluator
        self.register_evaluator(EvaluatorInfo(
            name="CrossEncoderCorrelationEvaluator",
            evaluator_type=EvaluatorType.CORRELATION,
            task_type=TaskType.REGRESSION,
            metrics=["pearson", "spearman"],
            description="CrossEncoder相关性评估器，用于回归任务"
        ))

        # CrossEncoderClassificationEvaluator
        self.register_evaluator(EvaluatorInfo(
            name="CrossEncoderClassificationEvaluator",
            evaluator_type=EvaluatorType.CLASSIFICATION,
            task_type=TaskType.CLASSIFICATION,
            metrics=["accuracy", "f1", "precision", "recall"],
            description="CrossEncoder分类评估器，用于分类任务"
        ))

        # EmbeddingSimilarityEvaluator
        self.register_evaluator(EvaluatorInfo(
            name="EmbeddingSimilarityEvaluator",
            evaluator_type=EvaluatorType.EMBEDDING_SIMILARITY,
            task_type=TaskType.REGRESSION,
            metrics=["cosine_pearson", "cosine_spearman"],
            description="Embedding相似度评估器，用于回归任务"
        ))

        # BinaryClassificationEvaluator
        self.register_evaluator(EvaluatorInfo(
            name="BinaryClassificationEvaluator",
            evaluator_type=EvaluatorType.BINARY_CLASSIFICATION,
            task_type=TaskType.CLASSIFICATION,
            metrics=["accuracy", "ap", "f1", "precision", "recall"],
            description="二分类评估器，用于分类任务"
        ))

    def register_metric(self, metric_info: MetricInfo):
        """注册指标"""
        self.metrics[metric_info.name] = metric_info

    def register_evaluator(self, evaluator_info: EvaluatorInfo):
        """注册评估器"""
        self.evaluators[evaluator_info.name] = evaluator_info

    def get_metric_info(self, metric_name: str) -> Optional[MetricInfo]:
        """获取指标信息"""
        return self.metrics.get(metric_name)

    def get_evaluator_info(self, evaluator_name: str) -> Optional[EvaluatorInfo]:
        """获取评估器信息"""
        return self.evaluators.get(evaluator_name)

    def get_metrics_by_category(self, category: MetricCategory) -> List[MetricInfo]:
        """按分类获取指标"""
        return [info for info in self.metrics.values() if info.category == category]

    def infer_evaluator_type_from_metrics(self, metric_names: List[str]) -> Optional[EvaluatorType]:
        """从指标名称推断评估器类型"""
        metric_set = set(metric_names)

        # 检查相关性指标
        correlation_metrics = {"pearson", "spearman", "cosine_pearson", "cosine_spearman"}
        if metric_set & correlation_metrics:
            if "cosine_pearson" in metric_set or "cosine_spearman" in metric_set:
                return EvaluatorType.EMBEDDING_SIMILARITY
            else:
                return EvaluatorType.CORRELATION

        # 检查分类指标
        classification_metrics = {"accuracy", "f1", "precision", "recall", "ap"}
        if metric_set & classification_metrics:
            if "ap" in metric_set:
                return EvaluatorType.BINARY_CLASSIFICATION
            else:
                return EvaluatorType.CLASSIFICATION

        return None

    def get_frontend_metadata(self, metric_names: List[str]) -> Dict[str, Any]:
        """获取前端所需的元数据"""
        evaluator_type = self.infer_evaluator_type_from_metrics(metric_names)

        # 获取指标分类统计
        categories = {}
        for metric_name in metric_names:
            metric_info = self.get_metric_info(metric_name)
            if metric_info:
                category = metric_info.category.value
                if category not in categories:
                    categories[category] = []
                categories[category].append({
                    "name": metric_name,
                    "display_name": metric_info.display_name,
                    "description": metric_info.description,
                    "range_min": metric_info.range_min,
                    "range_max": metric_info.range_max,
                    "higher_is_better": metric_info.higher_is_better
                })

        task_type = None
        if evaluator_type:
            # 推断任务类型
            correlation_types = {EvaluatorType.CORRELATION, EvaluatorType.EMBEDDING_SIMILARITY}
            if evaluator_type in correlation_types:
                task_type = TaskType.REGRESSION.value
            else:
                task_type = TaskType.CLASSIFICATION.value

        return {
            "evaluator_type": evaluator_type.value if evaluator_type else None,
            "task_type": task_type,
            "metric_categories": categories,
            "available_metrics": metric_names,
            "chart_suggestions": self._get_chart_suggestions(categories)
        }

    def _get_chart_suggestions(self, categories: Dict[str, List]) -> List[str]:
        """根据指标分类建议图表类型"""
        suggestions = []

        if "correlation" in categories:
            suggestions.append("correlation_scatter_plot")
            suggestions.append("correlation_line_chart")

        if "classification" in categories:
            suggestions.append("classification_bar_chart")
            suggestions.append("classification_radar_chart")

        if "loss" in categories:
            suggestions.append("loss_line_chart")

        if "performance" in categories:
            suggestions.append("performance_bar_chart")

        if not suggestions:
            suggestions.append("generic_line_chart")

        return suggestions

# 全局注册器实例
metric_registry = MetricRegistry()

def get_metric_registry() -> MetricRegistry:
    """获取全局指标注册器"""
    return metric_registry