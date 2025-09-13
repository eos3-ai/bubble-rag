import os
import logging
import traceback
import swanlab

# 配置全局日志格式
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


class SwanLabConfig:
    """SwanLab配置管理类，封装所有SwanLab相关的字段和操作"""
    
    def __init__(self, report_to: str = "", api_key: str = "", workspace: str = "", 
                 project: str = "", experiment_name: str = "", mode: str = ""):
        """
        初始化SwanLab配置
        
        Args:
            report_to: 报告工具类型，从训练参数传入
            api_key: SwanLab API密钥
            workspace: SwanLab工作空间
            project: SwanLab项目名
            experiment_name: 实验名称
            mode: SwanLab模式 (local/cloud)
        """
        # 只使用传入的参数，避免从环境变量读取（防止主进程环境污染）
        self.report_to = report_to
        self.api_key = api_key or ""
        self.workspace = workspace or ""
        self.project = project or ""
        self.experiment_name = experiment_name or ""
        self.mode = mode or ""
    
    @property
    def is_enabled(self) -> bool:
        """检查是否启用SwanLab - 只有明确设置为swanlab时才启用"""
        from ..enums.training_parameter_enums import ReportTo
        return (self.report_to == ReportTo.SWANLAB or 
                self.report_to == ReportTo.SWANLAB.value or
                (isinstance(self.report_to, str) and self.report_to.lower() == "swanlab"))
    
    @property
    def should_use_local_mode(self) -> bool:
        """检查是否应该使用本地模式"""
        return not self.api_key or self.mode.lower() == "local"
    
    def setup_environment(self):
        """设置SwanLab环境变量"""
        if not self.is_enabled:
            return
            
        if self.should_use_local_mode:
            logger.info("SwanLab使用本地模式")
            os.environ["SWANLAB_MODE"] = "local"
        elif self.api_key:
            # 确保环境变量设置正确
            os.environ["SWANLAB_API_KEY"] = self.api_key
            if self.workspace:
                os.environ["SWANLAB_WORKSPACE"] = self.workspace
            if self.project:
                os.environ["SWANLAB_PROJECT"] = self.project
            if self.experiment_name:
                os.environ["SWANLAB_EXPERIMENT"] = self.experiment_name
    
    def login(self):
        """登录SwanLab - 只有启用时才执行"""
        if not self.is_enabled:
            logger.debug("SwanLab未启用，跳过登录")
            return
            
        if self.api_key and not self.should_use_local_mode:
            try:
                swanlab.login(api_key=self.api_key)
                logger.info("SwanLab登录成功")
            except Exception as e:
                logger.warning(f"SwanLab登录失败，切换到本地模式: {e}")
                os.environ["SWANLAB_MODE"] = "local"
        else:
            logger.info("SwanLab使用本地模式，跳过登录")
    
    def init_experiment(self):
        """初始化SwanLab实验 - 只有启用时才执行"""
        if not self.is_enabled:
            logger.debug("SwanLab未启用，跳过实验初始化")
            return
            
        try:
            init_kwargs = {}
            if self.project:
                init_kwargs["project"] = self.project
            if self.workspace and not self.should_use_local_mode:
                init_kwargs["workspace"] = self.workspace
            if self.experiment_name:
                init_kwargs["experiment_name"] = self.experiment_name
                
            swanlab.init(**init_kwargs)
            logger.info(f"SwanLab实验初始化成功: {init_kwargs}")
        except Exception as e:
            logger.error(f"SwanLab实验初始化失败: {e}")
            raise
    
    @classmethod
    def from_training_config(cls, training_config: dict) -> 'SwanLabConfig':
        """
        从训练配置创建SwanLab配置
        
        Args:
            training_config: 训练配置字典
            
        Returns:
            SwanLabConfig实例
        """
        return cls(
            report_to=training_config.get("report_to", ""),
            api_key=training_config.get("swanlab_api_key", ""),
            workspace=training_config.get("swanlab_workspace", ""),
            project=training_config.get("swanlab_project", ""),
            experiment_name=training_config.get("swanlab_experiment", ""),
            mode=training_config.get("swanlab_mode", "")
        )
    
    def get_env_vars(self) -> set:
        """获取SwanLab相关的环境变量名称集合"""
        return {
            "SWANLAB_API_KEY", "SWANLAB_WORKSPACE", "SWANLAB_PROJECT", 
            "SWANLAB_EXPERIMENT", "SWANLAB_MODE"
        }


def init_swanlab(training_config: dict = None, report_to: str = ""):
    """
    初始化 SwanLab 实验跟踪工具，支持分布式训练
    仅在主进程且配置了 SwanLab 时启动
    
    Args:
        training_config: 完整的训练配置字典，包含SwanLab相关参数
        report_to: 报告工具配置（向后兼容参数）
    """
    try:
        from accelerate import PartialState
        state = PartialState()
        
        # 只在主进程中初始化SwanLab
        if not state.is_main_process:
            logger.debug("非主进程，跳过SwanLab初始化")
            return
            
        # 确定report_to配置（完全不依赖环境变量）
        if training_config:
            # 从training_config获取
            config = SwanLabConfig.from_training_config(training_config)
        else:
            # 使用传入的report_to参数（不再回退到环境变量）
            config = SwanLabConfig(report_to=report_to)
        
        logger.info(f"SwanLab配置检查: report_to='{config.report_to}', is_enabled={config.is_enabled}")
        
        if config.is_enabled:
            logger.info("开始初始化SwanLab...")
            
            # 设置环境变量（确保HuggingFace训练器能正确识别）
            config.setup_environment()
            
            # 登录和初始化实验
            config.login()
            config.init_experiment()
            
            logger.info("SwanLab初始化完成")
        else:
            logger.debug("SwanLab未启用，跳过初始化")
            
    except ImportError:
        logger.warning("accelerate库未安装，无法检测分布式训练状态，假设为单进程运行")
        # 在没有accelerate的情况下，假设为单进程
        if training_config:
            config = SwanLabConfig.from_training_config(training_config)
        else:
            effective_report_to = report_to
            config = SwanLabConfig(report_to=effective_report_to)
            
        if config.is_enabled:
            logger.info("初始化SwanLab（单进程模式）...")
            config.setup_environment()
            config.login()
            config.init_experiment()
    except Exception as e:
        logger.error(f"SwanLab初始化失败: {e}")
        # 不抛出异常，允许训练继续

def create_embedding_loss(model, dataset, target_column, dataset_name=None):
    """为embedding模型创建损失函数"""
    from sentence_transformers.losses import CosineSimilarityLoss, ContrastiveLoss, MultipleNegativesRankingLoss, CoSENTLoss
    
    try:
        # 检查数据集是否为None
        if dataset is None:
            logger.error("数据集为None，无法创建损失函数")
            return CoSENTLoss(model)  # 返回默认损失函数
        
        # 检查数据集格式和目标列类型
        if target_column in dataset.column_names:
            # 安全地获取数据类型
            try:
                dtype_str = str(dataset[target_column].dtype)
            except AttributeError:
                # 如果直接访问dtype失败，尝试获取第一个值的类型
                sample_value = dataset[target_column][0]
                dtype_str = str(type(sample_value).__name__)
            
            # 检查是否有anchor, positive, negative格式 (三元组)
            if all(col in dataset.column_names for col in ['anchor', 'positive', 'negative']):
                loss = MultipleNegativesRankingLoss(model)
                loss_name = "MultipleNegativesRankingLoss（三元组）"
            # 检查标签列名称和数据类型，优先根据列名判断
            elif target_column == "label" or dtype_str in ['int64', 'int32', 'int']:
                loss = ContrastiveLoss(model)
                if target_column == "label":
                    loss_name = "ContrastiveLoss（label列，适用于pair-class数据）"
                else:
                    loss_name = "ContrastiveLoss（整数标签）"
            elif target_column == "score" or dtype_str in ['float64', 'float32', 'float']:
                loss = CosineSimilarityLoss(model)
                if target_column == "score":
                    loss_name = "CosineSimilarityLoss（score列，适用于pair-score数据）"
                else:
                    loss = CoSENTLoss(model)
                    loss_name = "CoSENTLoss（浮点分数，更先进的相似度损失）"
            # 默认使用CoSENTLoss（回归任务）
            else:
                loss = CoSENTLoss(model)
                loss_name = "CoSENTLoss（默认，更先进的相似度损失）"
        else:
            # 如果没有目标列，检查是否为(anchor, positive)格式
            if all(col in dataset.column_names for col in ['anchor', 'positive']):
                loss = MultipleNegativesRankingLoss(model)
                loss_name = "MultipleNegativesRankingLoss（正负样本对）"
            else:
                loss = CoSENTLoss(model)
                loss_name = "CoSENTLoss（默认）"
        
        prefix = f"数据集 {dataset_name}: " if dataset_name else ""
        logger.info(f"{prefix}使用{loss_name}")
        return loss
        
    except Exception as e:
        logger.warning(f"检查数据类型时出错: {e}，默认使用CoSENTLoss")
        return CoSENTLoss(model)


def push_to_hub(model, model_name, final_output_dir):
    """
    将训练好的模型推送到 Hugging Face Hub
    
    Args:
        model: 训练好的模型实例
        model_name: 原始模型名称
        final_output_dir: 模型保存目录（用于错误提示）
    """
    # 提取模型名称（去除路径前缀）
    clean_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    hub_name = f"{clean_model_name}-sts"
    
    try:
        logger.info(f"正在推送模型到Hub: {hub_name}")
        model.push_to_hub(hub_name)
        logger.info(f"模型成功推送到Hub: {hub_name}")
    except Exception as e:
        logger.error(
            f"模型推送到 Hugging Face Hub 失败: {e}\n{traceback.format_exc()}"
            f"手动上传步骤:\n"
            f"1. 运行: huggingface-cli login\n"
            f"2. 加载模型: model = SentenceTransformer('{final_output_dir}')\n"
            f"3. 推送模型: model.push_to_hub('{hub_name}')"
        )