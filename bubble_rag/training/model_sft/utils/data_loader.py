import os
import logging
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Optional, Tuple, Dict, List, Union
import json

logger = logging.getLogger(__name__)

class DataLoader:
    """统一的数据加载器，与训练类型无关，只负责数据加载"""
    
    # 默认HF subset配置
    DEFAULT_HF_SUBSET = "pair-class"
    
    def __init__(self, hf_subset: Optional[str] = None,
                 train_sample_size: int = -1,
                 eval_sample_size: int = -1,
                 test_sample_size: int = -1):
        """
        初始化数据加载器
        
        Args:
            hf_subset: HuggingFace数据集的子配置名称，支持：
                     - 单个配置: 'pair-score'
                     - 多个配置: 'pair-score,pair-class,custom-config'
                     - 如果HF数据集数量超过subset配置数量，超出部分使用默认的pair-class
            train_sample_size: 训练集样本数量限制，-1表示不限制，0表示不使用该数据集
            eval_sample_size: 验证集样本数量限制，-1表示不限制，0表示不使用该数据集
            test_sample_size: 测试集样本数量限制，-1表示不限制，0表示不使用该数据集
        """
        # 解析HF_subset为列表
        if hf_subset and isinstance(hf_subset, str):
            if "," in hf_subset:
                # 分割subset配置，支持空配置和显式的none/null
                parts = [s.strip() for s in hf_subset.split(",")]
                self.hf_subsets = []
                for part in parts:
                    if part == "" or part.lower() in ["none", "null"]:
                        # 空字符串、none、null都表示无配置
                        self.hf_subsets.append(None)
                    else:
                        self.hf_subsets.append(part)
            else:
                if hf_subset.lower() in ["none", "null"]:
                    self.hf_subsets = [None]
                else:
                    self.hf_subsets = [hf_subset]
        elif isinstance(hf_subset, list):
            self.hf_subsets = hf_subset
        else:
            self.hf_subsets = []
        
        # 保持向后兼容
        self.hf_subset = hf_subset
        
        # 存储样本大小限制参数，进行参数验证
        # 所有数据集都允许-1表示不限制，0表示不使用该数据集
        self.train_sample_size = train_sample_size if train_sample_size >= -1 else -1
        self.eval_sample_size = eval_sample_size if eval_sample_size >= -1 else -1
        self.test_sample_size = test_sample_size if test_sample_size >= -1 else -1

        # 如果参数不合法，记录警告
        if train_sample_size < -1:
            logger.warning(f"train_sample_size不能小于-1，已重置为-1: {train_sample_size}")
        if eval_sample_size < -1:
            logger.warning(f"eval_sample_size不能小于-1，已重置为-1: {eval_sample_size}")
        if test_sample_size < -1:
            logger.warning(f"test_sample_size不能小于-1，已重置为-1: {test_sample_size}")
            
        logger.info(f"初始化数据加载器，HF_subset: {hf_subset}, 解析后的subset列表: {self.hf_subsets}")
        logger.info(f"样本数量限制: train={self.train_sample_size}, eval={self.eval_sample_size}, test={self.test_sample_size}")
        
    def load_data(self, split_type: str = "train", dataset_path: str = None) -> Union[Optional[Dataset], Dict[str, Dataset]]:
        """
        根据传入参数加载数据集，支持单个数据集或多个数据集
        
        Args:
            split_type: 数据集分割类型 ('train', 'dev', 'test')
            dataset_path: 数据集路径，必须提供
            
        Returns:
            Dataset、Dict[str, Dataset]或None
            - 单个数据集返回Dataset
            - 多个数据集返回Dict[str, Dataset]，键为数据集名称
        """
        if not dataset_path:
            logger.error("未提供数据集路径")
            return None
        
        return self._load_from_unified_path(dataset_path, split_type)
    
    def _is_huggingface_dataset_names(self, dataset_path: str) -> bool:
        """
        判断单个路径是否为Hugging Face数据集名称
        
        Args:
            dataset_path: 单个数据集路径字符串（不包含逗号）
            
        Returns:
            True表示是HuggingFace数据集名称，False表示是本地文件路径
        """
        path = dataset_path.strip()
        
        # 注意：不能因为指定了HF_subset就强制判断所有路径为HF数据集
        # HF_subset只影响已识别的HF数据集的配置，不影响数据集类型识别
        
        # 检查是否包含文件扩展名（本地文件的特征）
        if path.endswith(('.json', '.jsonl', '.txt', '.csv')):
            return False
        
        # 检查是否以 ./ 或 ../ 开头（明确的相对路径）
        if path.startswith('./') or path.startswith('../'):
            return False
        
        # 检查是否为 HuggingFace 标准格式 org/repo
        if '/' in path:
            parts = path.split('/')
            if len(parts) == 2 and parts[0] and parts[1]:
                # 符合 org/repo 格式，很可能是 HuggingFace 数据集
                return True
            elif len(parts) > 2:
                # 多级路径，可能是本地路径
                return False
        
        # 检查是否包含本地路径分隔符（Windows）
        if os.path.sep in path and os.path.sep != '/':
            return False
        
        # 检查是否为绝对路径
        if os.path.isabs(path):
            return False
        
        # 检查是否存在作为本地文件或文件夹
        if os.path.exists(path):
            return False
        
        # 如果不符合本地文件特征，假定为HuggingFace数据集名称
        return True

    def _load_from_unified_path(self, dataset_path: str, split_type: str) -> Union[Optional[Dataset], Dict[str, Dataset]]:
        """
        统一的数据加载方法，按优先级顺序尝试加载
        
        Args:
            dataset_path: 数据集路径，可能是：
                1. Hugging Face数据集名称（如：sentence-transformers/all-nli）
                2. 逗号分隔的多个Hub数据集名称
                3. 本地文件路径（.json/.jsonl）
                4. 逗号分隔的多个文件路径（仅用于训练集）
                5. 本地文件夹路径
            split_type: 数据集分割类型
            
        Returns:
            Dataset、Dict[str, Dataset]或None
        """
        # 检查是否为纯本地文件（非混合）
        if not "," in dataset_path and not self._is_huggingface_dataset_names(dataset_path) and os.path.isfile(dataset_path):
            # 单个本地文件，仅支持训练集
            if split_type != "train":
                logger.info(f"单个本地文件仅支持训练集，{split_type}分割返回None")
                return None
        
        # 检查是否为混合路径（包含多种类型）
        if "," in dataset_path:
            # 混合路径：可能包含HuggingFace数据集、本地文件和本地文件夹
            return self._load_from_mixed_sources(dataset_path, split_type)
        
        # 单个路径的处理逻辑：根据路径类型直接调用对应方法
        if self._is_huggingface_dataset_names(dataset_path):
            # HuggingFace数据集
            logger.info(f"识别为HuggingFace数据集: {dataset_path}")
            try:
                result = self._load_from_hub_unified(dataset_path, split_type)
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"HuggingFace数据集加载失败: {e}")
        else:
            # 本地路径（文件或文件夹）
            logger.info(f"识别为本地路径: {dataset_path}")
            try:
                result = self._load_from_local_unified(dataset_path, split_type)
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"本地路径加载失败: {e}")
        
        # 优先级3: 使用默认数据集（对所有分割类型都尝试） - 暂时禁用
        logger.error(f"所有加载方式失败，不自动使用默认数据集: split={split_type}")
        # default_result = self._load_default_dataset(split_type)
        # if default_result is not None:
        #     logger.info(f"成功使用默认数据集: {split_type}")
        #     return default_result
        
        logger.warning(f"无法加载数据集: {dataset_path}, split={split_type}")
        return None
    
    def _load_from_local_unified(self, path: str, split_type: str) -> Union[Optional[Dataset], Dict[str, Dataset]]:
        """从单个本地路径加载数据（不处理逗号分隔的多路径）"""
        if os.path.isdir(path):
            # 单个文件夹路径
            return self._load_from_directory(path, split_type)
        elif os.path.isfile(path):
            # 单个文件路径
            return self._load_from_json_single(path, split_type)
        else:
            logger.error(f"路径不存在: {path}")
            if split_type == "train":
                raise ValueError(f"训练数据路径不存在: {path}")
            return None
    
    def _load_from_hub_unified(self, dataset_name: str, split_type: str) -> Union[Optional[Dataset], Dict[str, Dataset]]:
        """从Hub加载单个数据集（不处理逗号分隔的多数据集）"""
        # 对于单个HF数据集，使用第一个subset配置（如果有的话）
        if self.hf_subsets:
            subset_to_use = self.hf_subsets[0]
            logger.info(f"单个HF数据集使用第一个subset配置: {subset_to_use}")
            
            # 如果用户为单个数据集指定了多个subset，给出警告
            if len(self.hf_subsets) > 1:
                unused_subsets = self.hf_subsets[1:]
                logger.warning(f"单个HF数据集 '{dataset_name}' 指定了多个subset配置，只使用第一个: {subset_to_use}，忽略: {unused_subsets}")
                logger.warning(f"如果要使用多个subset，请提供多个数据集路径，用逗号分隔")
            
            # 创建临时loader实例来使用特定的subset
            temp_loader = DataLoader(hf_subset=subset_to_use)
            return temp_loader._load_from_hub_single(dataset_name, split_type)
        else:
            # 没有指定subset，使用默认配置
            return self._load_from_hub_single(dataset_name, split_type)
    
    def _load_from_hub_single(self, dataset_name: str, split_type: str) -> Optional[Dataset]:
        """从Hugging Face Hub加载单个数据集，支持本地缓存模式和分割名称自动回退"""
        logger.info(f"从Hub加载数据集: {dataset_name}, split: {split_type}, HF_subset: {self.hf_subset}")
        
        # 确定要使用的subset配置
        if self.hf_subsets:
            # 如果有多个subset，只使用第一个（适用于单个HF数据集的场景）
            actual_subset = self.hf_subsets[0]
            # 添加智能回退：指定配置 -> 常见配置 -> 无配置
            config_alternatives = [actual_subset, "pair-class", "pair-score", None]
            logger.info(f"使用HF_subset配置: {actual_subset} (来自列表: {self.hf_subsets})，带智能回退")
        elif self.hf_subset is not None and not "," in str(self.hf_subset):
            # 向后兼容：单个subset字符串
            config_alternatives = [self.hf_subset, "pair-class", "pair-score", None]
            logger.info(f"使用指定的HF_subset配置: {self.hf_subset}，带智能回退")
        elif self.hf_subset is None:
            # 明确指定了None，优先无配置但保留回退
            config_alternatives = [None, "pair-class", "pair-score"]
            logger.info("使用明确指定的无配置，带有回退机制")
        else:
            # 没有有效的subset配置，使用默认值和无配置回退
            config_alternatives = ["pair-class", "pair-score", None]
            logger.info("使用配置回退策略: pair-class -> pair-score -> 无配置")
        
        # 定义分割名称的回退顺序
        split_alternatives = [split_type]
        if split_type == "eval":
            split_alternatives = ["eval", "dev", "validation"]  # 优先尝试原名，然后回退
        elif split_type == "dev":
            split_alternatives = ["dev", "eval", "validation"]  # 优先尝试原名，然后回退
        elif split_type == "validation":
            split_alternatives = ["validation", "eval", "dev"]  # 优先尝试原名，然后回退
        
        # 尝试每个分割名称和配置组合
        for split_attempt, actual_split in enumerate(split_alternatives):
            if split_attempt > 0:
                logger.info(f"回退尝试分割名称: {split_type} -> {actual_split}")
            
            for config_attempt, config in enumerate(config_alternatives):
                if config_attempt > 0:
                    config_desc = "无配置" if config is None else config
                    logger.info(f"回退尝试配置: {config_desc}")
                
                try:
                    # 优先尝试ModelScope加载
                    try:
                        from modelscope.msdatasets import MsDataset
                        logger.info(f"优先尝试从ModelScope加载数据集: {dataset_name}")
                        
                        if config is None:
                            ms_dataset = MsDataset.load(dataset_name, split=actual_split)
                        else:
                            ms_dataset = MsDataset.load(dataset_name, subset_name=config, split=actual_split)
                        
                        # 转换为HuggingFace Dataset格式
                        from datasets import Dataset
                        dataset = Dataset.from_dict(ms_dataset.to_dict())
                        
                        if split_attempt > 0:
                            logger.info(f"ModelScope成功使用回退分割名称: {actual_split}")
                        if config_attempt > 0:
                            config_desc = "无配置" if config is None else config
                            logger.info(f"ModelScope成功使用回退配置: {config_desc}")
                        logger.info(f"成功从ModelScope加载数据集: {dataset_name}")
                        return self._apply_sample_size(dataset, split_type)
                        
                    except ImportError:
                        logger.info("ModelScope未安装，回退到HuggingFace")
                        # 回退到HuggingFace加载
                        if config is None:
                            dataset = load_dataset(dataset_name, split=actual_split)
                        else:
                            dataset = load_dataset(dataset_name, config, split=actual_split)
                        if split_attempt > 0:
                            logger.info(f"HuggingFace成功使用回退分割名称: {actual_split}")
                        if config_attempt > 0:
                            config_desc = "无配置" if config is None else config
                            logger.info(f"HuggingFace成功使用回退配置: {config_desc}")
                        logger.info(f"成功从HuggingFace加载数据集: {dataset_name}")
                        return self._apply_sample_size(dataset, split_type)
                    
                    except Exception as ms_error:
                        logger.warning(f"ModelScope加载失败: {ms_error}，回退到HuggingFace")
                        # ModelScope失败，回退到HuggingFace
                        if config is None:
                            dataset = load_dataset(dataset_name, split=actual_split)
                        else:
                            dataset = load_dataset(dataset_name, config, split=actual_split)
                        if split_attempt > 0:
                            logger.info(f"HuggingFace成功使用回退分割名称: {actual_split}")
                        if config_attempt > 0:
                            config_desc = "无配置" if config is None else config
                            logger.info(f"HuggingFace成功使用回退配置: {config_desc}")
                        logger.info(f"成功从HuggingFace加载数据集: {dataset_name}")
                        return self._apply_sample_size(dataset, split_type)
                except Exception as e:
                    # 检查是否是配置不存在的问题
                    if ("Unknown config" in str(e) or "BuilderConfig" in str(e) or "Config name is missing" in str(e)) and config_attempt < len(config_alternatives) - 1:
                        # 提取可用配置信息
                        if "Available:" in str(e):
                            available_configs = str(e).split("Available:")[1].strip()
                            logger.info(f"配置 '{config}' 不存在，可用配置: {available_configs}，尝试下一个配置...")
                        elif "Config name is missing" in str(e):
                            logger.info(f"数据集要求指定配置（当前为None），尝试下一个配置...")
                        else:
                            logger.info(f"配置 '{config}' 不存在，尝试下一个配置...")
                        continue
                    
                    # 检查是否是分割名称问题
                    if "Unknown split" in str(e) and split_attempt < len(split_alternatives) - 1:
                        logger.info(f"分割 '{actual_split}' 不存在，尝试下一个分割...")
                        break  # 跳出内层配置循环，尝试下一个分割
                    
                    # 最后尝试本地缓存（只在第一次尝试时）
                    if split_attempt == 0 and config_attempt == 0:
                        logger.warning(f"数据集加载失败，尝试本地缓存: {e}")
                        try:
                            logger.info(f"尝试从本地缓存加载数据集: {dataset_name}")
                            # 尝试使用local_files_only参数
                            try:
                                if config is None:
                                    dataset = load_dataset(dataset_name, split=actual_split, local_files_only=True)
                                else:
                                    dataset = load_dataset(dataset_name, config, split=actual_split, local_files_only=True)
                            except TypeError as type_error:
                                # 如果local_files_only参数不支持，则不使用该参数
                                if "unexpected keyword argument" in str(type_error) and "local_files_only" in str(type_error):
                                    logger.info(f"数据集 {dataset_name} 不支持local_files_only参数，尝试不使用该参数")
                                    if config is None:
                                        dataset = load_dataset(dataset_name, split=actual_split)
                                    else:
                                        dataset = load_dataset(dataset_name, config, split=actual_split)
                                else:
                                    raise type_error
                            logger.info(f"成功从本地缓存加载数据集: {dataset_name}")
                            return self._apply_sample_size(dataset, split_type)
                        except Exception as cache_error:
                            logger.error(f"本地缓存也失败: {cache_error}")
                    
                    # 如果是最后一次尝试，记录详细错误信息
                    if split_attempt == len(split_alternatives) - 1 and config_attempt == len(config_alternatives) - 1:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ["connection", "network", "timeout", "huggingface.co"]):
                            logger.error(f"网络连接问题导致数据集加载失败: {dataset_name}")
                            logger.error("建议: 1) 检查网络连接 2) 使用本地数据集文件 3) 配置代理")
                        else:
                            logger.error(f"数据集加载失败: {e}")
                            if "Unknown split" in str(e):
                                logger.error(f"尝试的所有分割名称都不存在: {split_alternatives}")
                            if "Unknown config" in str(e):
                                logger.error(f"尝试的所有配置都不存在: {config_alternatives}")
        
        return None
    
    def _load_from_local_multiple(self, split_type: str) -> Union[Dataset, Dict[str, Dataset]]:
        """从本地文件/文件夹加载多个数据集"""
        path = os.getenv(f"{split_type.upper()}_DATASET")
        
        if os.path.isdir(path):
            # 文件夹：加载其中所有JSON文件
            return self._load_from_directory(path, split_type)
        elif "," in path:
            # 逗号分隔的多个文件路径
            return self._load_from_multiple_files(path, split_type)
        else:
            # 单个文件，保持向后兼容
            return self._load_from_json_single(path, split_type)
    
    def _load_from_directory(self, dir_path: str, split_type: str) -> Union[Dataset, Dict[str, Dataset]]:
        """从文件夹中加载数据集，智能识别标准命名的文件"""
        logger.info(f"从文件夹加载数据集: {dir_path}")
        
        # 标准文件名映射
        standard_files = {
            'train': ['train_data.jsonl', 'train_data.json', 'train.jsonl', 'train.json'],
            'eval': ['eval_data.jsonl', 'eval_data.json', 'eval.jsonl', 'eval.json', 'val_data.jsonl', 'val.jsonl','val.json','val_data.json', 'dev_data.jsonl', 'dev_data.json', 'dev.jsonl', 'dev.json'],
            'dev': ['eval_data.jsonl', 'eval_data.json', 'eval.jsonl', 'eval.json', 'val_data.jsonl', 'val.jsonl','val.json','val_data.json', 'dev_data.jsonl', 'dev_data.json', 'dev.jsonl', 'dev.json'],
            'test': ['test_data.jsonl', 'test_data.json', 'test.jsonl', 'test.json']
        }
        
        # 检查是否有标准命名的文件
        found_files = {}
        for split, possible_names in standard_files.items():
            for name in possible_names:
                file_path = os.path.join(dir_path, name)
                if os.path.exists(file_path):
                    found_files[split] = file_path
                    break
        
        if found_files:
            # 如果找到标准命名文件，只加载对应split的数据
            if split_type in found_files:
                file_path = found_files[split_type]
                logger.info(f"加载标准数据文件: {file_path}")
                try:
                    dataset = load_dataset("json", data_files=file_path, split="train")
                    return self._apply_sample_size(dataset, split_type)
                except Exception as e:
                    logger.error(f"加载标准数据文件失败: {e}")
                    if split_type == "train":
                        raise ValueError(f"训练数据加载失败: {e}")
                    return None
            else:
                logger.info(f"文件夹中没有{split_type}分割的标准数据文件")
                return None
        
        else:
            # 如果没有标准命名文件，返回None
            logger.warning(f"文件夹中未找到{split_type}分割的标准命名文件")
            if split_type == "train":
                raise ValueError(f"文件夹中未找到训练数据的标准命名文件")
            return None
    
    
    def _extract_dataset_base_name(self, path: str) -> str:
        """
        提取数据集基础名称（不含分割后缀）
        
        Args:
            path: 数据集路径
            
        Returns:
            标准化的数据集基础名称
        """
        if self._is_huggingface_dataset_names(path):
            # HuggingFace数据集：保持原始名称不变
            return path
        elif os.path.isdir(path):
            # 文件夹：使用路径最后部分
            return os.path.basename(path.rstrip(os.path.sep))
        elif os.path.isfile(path):
            # 文件：使用不带后缀的文件名
            filename = os.path.basename(path)
            # 移除常见的数据文件后缀
            for ext in ['.json', '.jsonl', '.txt', '.csv']:
                if filename.endswith(ext):
                    filename = filename[:-len(ext)]
                    break
            return filename
        else:
            # 未知类型，直接返回路径
            return path

    def _load_from_mixed_sources(self, mixed_paths: str, split_type: str) -> Union[Dataset, Dict[str, Dataset]]:
        """从混合数据源加载数据集，支持HF_subset按HF数据集顺序分配"""
        path_list = [path.strip() for path in mixed_paths.split(",")]
        datasets = {}
        
        # 第一步：识别所有HF数据集并记录索引
        hf_datasets_info = []  # [(原始索引, 路径)]
        for i, path in enumerate(path_list):
            if self._is_huggingface_dataset_names(path):
                hf_datasets_info.append((i, path))
        
        # 记录subset分配情况
        if self.hf_subsets:
            logger.info(f"HF数据集数量: {len(hf_datasets_info)}, subset配置数量: {len(self.hf_subsets)}")
            if len(hf_datasets_info) > len(self.hf_subsets):
                logger.info(f"HF数据集多于subset配置，{len(hf_datasets_info) - len(self.hf_subsets)}个数据集将使用默认subset: {self.DEFAULT_HF_SUBSET}")
            elif len(self.hf_subsets) > len(hf_datasets_info):
                unused_subsets = self.hf_subsets[len(hf_datasets_info):]
                logger.info(f"subset配置多于HF数据集，以下subset将被忽略: {unused_subsets}")
        
        # 第二步：按顺序处理所有数据集
        hf_dataset_counter = 0  # HF数据集计数器
        
        for original_index, path in enumerate(path_list):
            try:
                dataset = None
                dataset_base_name = self._extract_dataset_base_name(path)
                
                # 判断路径类型并加载
                if self._is_huggingface_dataset_names(path):
                    # HF数据集：获取对应的subset
                    subset_for_this_dataset = self._get_hf_subset_for_dataset(hf_dataset_counter)
                    
                    logger.info(f"从HuggingFace加载数据集: {path} (HF数据集#{hf_dataset_counter}, subset: {subset_for_this_dataset})")
                    
                    # 临时创建专门的DataLoader实例来处理这个特定的subset，保持sample_size配置
                    temp_loader = DataLoader(
                        hf_subset=subset_for_this_dataset,
                        train_sample_size=self.train_sample_size,
                        eval_sample_size=self.eval_sample_size,
                        test_sample_size=self.test_sample_size
                    )
                    dataset = temp_loader._load_from_hub_single(path, split_type)
                    
                    hf_dataset_counter += 1
                    
                elif os.path.isdir(path):
                    # 本地文件夹：不需要subset
                    logger.info(f"从文件夹加载数据集: {path}")
                    dataset = self._load_from_directory(path, split_type)
                    
                elif os.path.isfile(path):
                    # 本地文件：不需要subset，仅训练集支持
                    if split_type != "train":
                        logger.info(f"本地文件仅支持训练集，跳过验证/测试集加载: {path}")
                        continue
                    logger.info(f"从文件加载数据集: {path}")
                    dataset = load_dataset("json", data_files=path, split="train")
                    dataset = self._apply_sample_size(dataset, split_type)
                else:
                    logger.warning(f"路径不存在或无法识别，跳过: {path}")
                    continue
                
                # 添加成功加载的数据集
                if dataset is not None:
                    datasets[dataset_base_name] = dataset
                    logger.info(f"成功加载数据集: {dataset_base_name} (来源: {path})")
                
            except Exception as e:
                logger.error(f"加载数据集失败: {path}, 错误: {e}")
                continue
        
        if not datasets and split_type == "train":
            raise ValueError(f"所有混合数据源加载失败")
        
        # 混合数据源总是返回Dict格式（即使只有1个成功），保持一致性
        return datasets if datasets else None
    
    def _load_from_json_single(self, json_path: str, split_type: str) -> Optional[Dataset]:
        """从单个JSON文件加载数据集"""
        logger.info(f"从JSON文件加载数据集: {json_path}")
        
        try:
            dataset = load_dataset("json", data_files=json_path, split="train")
            return self._apply_sample_size(dataset, split_type)
        except Exception as e:
            logger.error(f"从JSON文件加载数据集失败: {e}")
            if split_type == "train":
                raise ValueError(f"训练数据加载失败: {e}")
            return None
    
    def _load_default_dataset(self, split_type: str) -> Optional[Dataset]:
        """加载默认示例数据集，自动映射分割名称"""
        logger.info(f"使用默认数据集: sentence-transformers/all-nli")
        try:
            # 映射分割名称：sentence-transformers/all-nli 使用 dev 而不是 eval
            actual_split = "dev" if split_type == "eval" else split_type
            
            # 尝试不同的配置
            for config in ["pair-class", "pair-score"]:
                try:
                    dataset = load_dataset("sentence-transformers/all-nli", config, split=actual_split)
                    if split_type != actual_split:
                        logger.info(f"成功加载默认数据集 {split_type} 分割（实际使用 {actual_split}）")
                    if config == "pair-class":
                        logger.info(f"成功使用pair-class配置加载默认数据集")
                    return self._apply_sample_size(dataset, split_type)
                except Exception as config_error:
                    if config == "pair-class":
                        logger.warning(f"默认数据集配置 {config} 失败: {config_error}")
                    continue
            
            logger.warning(f"默认数据集的所有配置都失败")
            return None
        except Exception as e:
            logger.warning(f"默认数据集的{split_type}分割不存在: {e}")
            return None
    
    def _apply_sample_size(self, dataset: Union[Dataset, Dict[str, Dataset]], split_type: str = "train") -> Union[Dataset, Dict[str, Dataset]]:
        """
        应用样本数量限制
        
        Args:
            dataset: 数据集，可以是单个Dataset或Dict[str, Dataset]
            split_type: 分割类型，用于确定使用哪个sample_size参数
            
        Returns:
            限制样本数量后的数据集
        """
        # 根据split_type选择对应的sample_size
        size_map = {
            "train": self.train_sample_size,
            "eval": self.eval_sample_size,
            "dev": self.eval_sample_size,  # dev等同于eval
            "test": self.test_sample_size
        }
        sample_size = size_map.get(split_type, 0)

        # 处理新的采样逻辑
        if sample_size == 0:
            # 0 表示不使用该数据集，返回None
            logger.info(f"数据集采样: {split_type} 设置为0，不使用该数据集")
            return None
        elif sample_size == -1:
            # -1 表示不限制，即使不需要采样，也记录原始大小
            if isinstance(dataset, dict):
                for name, ds in dataset.items():
                    ds._original_total_samples = len(ds)
            else:
                dataset._original_total_samples = len(dataset)
            logger.info(f"数据集采样: {split_type} 设置为-1，不限制样本数量")
            return dataset
            
        # 处理多数据集场景
        if isinstance(dataset, dict):
            result = {}
            for name, ds in dataset.items():
                original_size = len(ds)
                if original_size > sample_size:
                    sampled_ds = ds.select(range(sample_size))
                    # 在采样后的数据集上记录原始大小
                    sampled_ds._original_total_samples = original_size
                    result[name] = sampled_ds
                    logger.info(f"数据集 {name}({split_type}): {original_size} → {sample_size}")
                else:
                    # 即使没有采样，也记录原始大小
                    ds._original_total_samples = original_size
                    result[name] = ds
                    logger.info(f"数据集 {name}({split_type}): {original_size} (无需限制)")
            return result
        else:
            # 单数据集场景
            original_size = len(dataset)
            if original_size > sample_size:
                result = dataset.select(range(sample_size))
                # 在采样后的数据集上记录原始大小
                result._original_total_samples = original_size
                logger.info(f"数据集({split_type}): {original_size} → {sample_size}")
                return result
            else:
                # 即使没有采样，也记录原始大小
                dataset._original_total_samples = original_size
                logger.info(f"数据集({split_type}): {original_size} (无需限制)")
                return dataset
    
    def load_all_splits(self, dataset_path: str = None) -> Tuple[Union[Dataset, Dict[str, Dataset]], 
                                      Optional[Union[Dataset, Dict[str, Dataset]]], 
                                      Optional[Union[Dataset, Dict[str, Dataset]]]]:
        """
        加载所有数据分割，支持单个数据集或多个数据集
        
        Args:
            dataset_path: 数据集路径，必须提供
        
        Returns:
            (train_dataset, eval_dataset, test_dataset)
            每个可能是Dataset、Dict[str, Dataset]或None
        """
        train_data = self.load_data("train", dataset_path)
        # 加载验证集：让底层的 _load_from_hub_single 自动处理分割名称回退
        # （"eval" -> "dev" -> "validation"）
        eval_data = self.load_data("eval", dataset_path)
        test_data = self.load_data("test", dataset_path)
        
        return train_data, eval_data, test_data
    
    def is_multi_dataset(self, data) -> bool:
        """检查是否为多数据集"""
        return isinstance(data, dict)
    
    def get_dataset_names(self, data) -> List[str]:
        """获取数据集名称列表"""
        if self.is_multi_dataset(data):
            return list(data.keys())
        else:
            return ["single_dataset"]
    
    def _get_hf_subset_for_dataset(self, hf_dataset_index: int) -> Optional[str]:
        """
        根据HF数据集的索引获取对应的subset
        
        Args:
            hf_dataset_index: 该数据集在所有HF数据集中的索引（0-based）
            
        Returns:
            对应的subset配置，超出部分返回None（尝试无配置加载）
        """
        if not self.hf_subsets or len(self.hf_subsets) == 0:
            # 没有指定任何subset，使用默认值
            return self.DEFAULT_HF_SUBSET
        
        # 添加边界检查
        if hf_dataset_index < 0:
            logger.warning(f"数据集索引为负数: {hf_dataset_index}，使用默认配置")
            return self.DEFAULT_HF_SUBSET
            
        if hf_dataset_index < len(self.hf_subsets):
            # 有对应的subset配置
            return self.hf_subsets[hf_dataset_index]
        else:
            # HF数据集数量大于subset数量，使用默认配置
            logger.info(f"数据集索引 {hf_dataset_index} 超出配置数量 {len(self.hf_subsets)}，使用默认配置: {self.DEFAULT_HF_SUBSET}")
            return self.DEFAULT_HF_SUBSET
    
    def get_target_column(self, data: Union[Dataset, Dict[str, Dataset]]) -> str:
        """
        获取目标列名，支持单个数据集或多个数据集
        
        Args:
            data: 数据集或数据集字典
            
        Returns:
            目标列名 ('score' 或 'label')
        """
        if self.is_multi_dataset(data):
            # 多数据集：使用第一个数据集确定目标列
            first_dataset = next(iter(data.values()))
            return self._get_single_target_column(first_dataset)
        else:
            return self._get_single_target_column(data)
    
    def _get_single_target_column(self, dataset: Dataset) -> str:
        """获取单个数据集的目标列名：三列格式直接使用第三列"""
        column_names = dataset.column_names
        
        # 三列格式：直接使用第三列作为目标列
        if len(column_names) == 3:
            target_col = column_names[2]
            logger.info(f"使用第三列作为目标列: '{target_col}'")
            return target_col
        
        # 兼容旧格式：优先使用标准列名
        if "score" in column_names:
            return "score"
        elif "label" in column_names:
            return "label"
        
        # 其他情况报错
        raise ValueError(f"无法确定目标列。数据集列名: {column_names}，请使用3列格式或包含'score'/'label'列")
    
    def _standardize_dataset_columns(self, dataset: Dataset, target_column: str) -> Dataset:
        """
        标准化数据集列名以符合sentence-transformers要求
        
        根据官方文档要求：
        1. 标签列必须命名为 "label" 或 "score"
        2. 其他列名称无关紧要，只有顺序重要
        
        Args:
            dataset: 输入数据集
            target_column: 目标列名（当前使用的标签列名）
            
        Returns:
            标准化后的数据集
        """
        column_names = dataset.column_names
        
        # 如果目标列已经是标准名称，则不需要重命名
        if target_column in ["label", "score"]:
            logger.info(f"目标列 '{target_column}' 已经是标准名称，无需重命名")
            return dataset
        
        # 检查目标列的数据类型来确定重命名目标
        sample_value = dataset[target_column][0]
        if isinstance(sample_value, int) or (isinstance(sample_value, float) and sample_value.is_integer()):
            # 整数类型 → 重命名为 "label"
            new_name = "label"
            logger.info(f"将目标列 '{target_column}' 重命名为 'label'（检测到整数类型）")
        else:
            # 浮点数类型 → 重命名为 "score" 
            new_name = "score"
            logger.info(f"将目标列 '{target_column}' 重命名为 'score'（检测到浮点类型）")
        
        # 创建列名映射
        rename_mapping = {target_column: new_name}
        
        # 执行重命名
        standardized_dataset = dataset.rename_columns(rename_mapping)
        logger.info(f"数据集列名标准化完成: {column_names} → {standardized_dataset.column_names}")
        
        return standardized_dataset
    
    def standardize_dataset_columns(self, data: Union[Dataset, Dict[str, Dataset]], target_column: str) -> Union[Dataset, Dict[str, Dataset]]:
        """
        对单个或多个数据集进行列名标准化
        
        Args:
            data: 数据集或数据集字典
            target_column: 目标列名
            
        Returns:
            标准化后的数据集
        """
        if self.is_multi_dataset(data):
            # 多数据集：对每个数据集进行标准化
            standardized_datasets = {}
            for name, dataset in data.items():
                if dataset is not None:
                    standardized_datasets[name] = self._standardize_dataset_columns(dataset, target_column)
                    logger.info(f"数据集 '{name}' 列名标准化完成")
                else:
                    standardized_datasets[name] = dataset
            return standardized_datasets
        else:
            # 单数据集
            return self._standardize_dataset_columns(data, target_column)
    
    def validate_dataset(self, data: Union[Dataset, Dict[str, Dataset]]) -> bool:
        """
        验证数据集格式，支持单个数据集或多个数据集
        
        Args:
            data: 数据集或数据集字典
            
        Returns:
            是否有效
        """
        if self.is_multi_dataset(data):
            # 多数据集：验证所有数据集
            for name, dataset in data.items():
                try:
                    self._validate_single_dataset(dataset)
                    logger.info(f"数据集 {name} 验证通过")
                except ValueError as e:
                    raise ValueError(f"数据集 {name} 验证失败: {e}")
        else:
            self._validate_single_dataset(data)
        
        return True
    
    def _validate_single_dataset(self, dataset: Dataset) -> bool:
        """验证单个数据集格式：必须为三列（文本1，文本2，标签）"""
        column_names = dataset.column_names
        
        # 检查列数量：必须为3列
        if len(column_names) != 3:
            raise ValueError(f"数据集必须为3列格式（文本1，文本2，标签），当前有{len(column_names)}列: {column_names}")
        
        # 验证前两列是否为文本类型
        try:
            sample = dataset[0] if len(dataset) > 0 else None
            if sample:
                col1, col2, col3 = column_names[0], column_names[1], column_names[2]
                
                # 检查前两列是文本
                if not isinstance(sample[col1], str):
                    raise ValueError(f"第1列 '{col1}' 必须是文本类型，当前类型: {type(sample[col1])}")
                if not isinstance(sample[col2], str):
                    raise ValueError(f"第2列 '{col2}' 必须是文本类型，当前类型: {type(sample[col2])}")
                
                # 检查第三列是数值类型（int, float）或可转换为数值
                label_value = sample[col3]
                if not self._is_valid_label(label_value):
                    raise ValueError(f"第3列 '{col3}' 必须是数值类型（int/float）或可转换为数值，当前值: {label_value} ({type(label_value).__name__})")
                
                label_type = "float" if isinstance(label_value, float) or self._is_float_like(label_value) else "int"
                logger.info(f"数据集验证通过: 文本1='{col1}', 文本2='{col2}', 标签='{col3}' ({label_type})")
                
        except (IndexError, KeyError) as e:
            raise ValueError(f"无法验证数据集格式: {e}")
        
        return True
    
    def _is_valid_label(self, value) -> bool:
        """检查值是否为有效的标签（数值类型或可转换为数值）"""
        # 直接是数值类型
        if isinstance(value, (int, float)):
            return True
        
        # 字符串数值
        try:
            float(str(value))
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_float_like(self, value) -> bool:
        """检查值是否为浮点数类型"""
        try:
            if isinstance(value, float):
                return True
            if isinstance(value, str) and ('.' in value or 'e' in value.lower()):
                float(value)
                return True
            return False
        except (ValueError, TypeError):
            return False

# 保持向后兼容的函数
def load_training_data(split_type):
    loader = DataLoader()
    return loader.load_data(split_type)