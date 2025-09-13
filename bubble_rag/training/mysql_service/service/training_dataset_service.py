"""训练数据集信息服务"""

import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from sqlmodel import select, Session
from datasets import Dataset

from bubble_rag.training.mysql_service.entity.training_task_models import safe_get_session
from bubble_rag.training.mysql_service.entity.training_dataset_models import (
    DatasetInfo, SplitType, DatasetType, DatasetStatus, EvaluationStatus
)

logger = logging.getLogger(__name__)


def create_dataset_tables():
    """创建数据集相关表"""
    try:
        from sqlmodel import create_engine
        from bubble_rag.server_config import MYSQL_URL
        from bubble_rag.training.mysql_service.entity.training_dataset_models import DatasetInfo
        
        engine = create_engine(MYSQL_URL)
        DatasetInfo.metadata.create_all(engine)
        logger.info("数据集表创建成功")
    except Exception as e:
        logger.error(f"创建数据集表失败: {e}")


# 模块加载时创建表
create_dataset_tables()


class TrainingDatasetService:
    """训练数据集信息管理服务"""
    
    @staticmethod
    def record_dataset_info(
        task_id: str,
        data_source_id: str,           # 数据源ID
        dataset_name: str,             # 数据集基础名称（不含分割后缀）
        dataset_base_name: str,        # 与dataset_name保持一致（向后兼容）
        dataset_path: str,
        dataset_type: str,
        split_type: str,
        dataset: Union[Dataset, Dict[str, Dataset]],
        target_column: str,
        loss_function: Optional[str],
        evaluator: Optional[str],
        hf_subset: Optional[str] = None,  # HF子配置参数
        configured_sample_size: int = 0   # 新增：配置的样本大小限制
    ) -> List[str]:
        """
        记录数据集信息
        
        Args:
            task_id: 训练任务ID
            data_source_id: 数据源标识ID (用于分组同源的train/eval/test，如ds_001)
            dataset_name: 数据集基础名称 (不含分割后缀，如squad, nli) 
            dataset_base_name: 与dataset_name保持一致 (向后兼容)
            dataset_path: 数据集路径
            dataset_type: 数据集类型 (huggingface/local_file/local_folder)
            split_type: 分割类型 (train/eval/test)
            dataset: 数据集对象
            target_column: 目标列名
            loss_function: 损失函数名称
            evaluator: 评估器类型
            hf_subset: HuggingFace数据集子配置名称 (如pair-score, pair-class)
            
        Returns:
            创建的数据集信息ID列表
        """
        dataset_ids = []
        
        with safe_get_session() as session:
            # 处理多数据集情况
            if isinstance(dataset, dict):
                for name, ds in dataset.items():
                    if ds is not None:
                        # 为每个数据集自动判断类型
                        individual_dataset_type = TrainingDatasetService._determine_dataset_type(name, dataset_path)
                        # 新设计：dataset_name不含split后缀，只保存基础名称
                        dataset_id = TrainingDatasetService._create_single_dataset_info(
                            session, task_id, data_source_id, name, name,
                            name,  # 对于多数据集，使用各自的名称作为路径
                            individual_dataset_type, split_type, ds,
                            target_column, loss_function, evaluator, hf_subset, configured_sample_size
                        )
                        dataset_ids.append(dataset_id)
            else:
                # 单数据集
                single_dataset_type = TrainingDatasetService._determine_dataset_type(dataset_path, dataset_path) if dataset_type == "auto" else DatasetType(dataset_type)
                dataset_id = TrainingDatasetService._create_single_dataset_info(
                    session, task_id, data_source_id, dataset_name, dataset_base_name,
                    dataset_path, single_dataset_type, split_type, dataset, 
                    target_column, loss_function, evaluator, hf_subset, configured_sample_size
                )
                dataset_ids.append(dataset_id)
            
            session.commit()
        
        logger.info(f"记录数据集信息成功, 任务ID: {task_id}, 数据集数量: {len(dataset_ids)}")
        return dataset_ids

    @staticmethod
    def get_data_sources_by_task(task_id: str) -> List[str]:
        """获取任务的所有数据源ID列表"""
        with safe_get_session() as session:
            statement = select(DatasetInfo.data_source_id).where(
                DatasetInfo.task_id == task_id
            ).distinct()
            result = session.exec(statement).all()
            return [r for r in result]
    
    @staticmethod
    def get_splits_by_source(task_id: str, data_source_id: str) -> Dict[str, Any]:
        """获取指定数据源的所有分割信息"""
        with safe_get_session() as session:
            statement = select(DatasetInfo).where(
                DatasetInfo.task_id == task_id,
                DatasetInfo.data_source_id == data_source_id
            )
            splits = session.exec(statement).all()
            
            result = {
                "data_source_id": data_source_id,
                "base_name": splits[0].dataset_base_name if splits else "",
                "path": splits[0].dataset_path if splits else "",
                "splits": {}
            }
            
            for split in splits:
                result["splits"][split.split_type] = {
                    "id": split.id,
                    "dataset_name": split.dataset_name,
                    "status": split.dataset_status,
                    "samples": split.total_samples,  # 原始数据集总样本数
                    "actual_samples": split.configured_sample_size,  # 实际训练使用的样本数
                    "loss_function": split.loss_function,
                    "evaluator": split.evaluator,
                    "target_column": split.target_column,
                    "label_type": split.label_type,
                    "base_eval_results": split.base_eval_results,
                    "final_eval_results": split.final_eval_results
                }
            
            return result
    
    @staticmethod
    def get_source_performance_summary(task_id: str) -> Dict[str, Dict]:
        """获取所有数据源的性能摘要"""
        source_ids = TrainingDatasetService.get_data_sources_by_task(task_id)
        summary = {}
        
        for source_id in source_ids:
            source_info = TrainingDatasetService.get_splits_by_source(task_id, source_id)
            splits = source_info["splits"]
            
            summary[source_id] = {
                "base_name": source_info["base_name"],
                "path": source_info["path"],
                "total_samples": sum(s["samples"] for s in splits.values()),  # 原始样本总数
                "actual_total_samples": sum(s["actual_samples"] for s in splits.values()),  # 实际使用样本总数
                "available_splits": list(splits.keys()),
                "loss_function": splits.get("train", {}).get("loss_function", ""),
                "best_eval_result": splits.get("eval", {}).get("final_eval_results", {}),
                "test_result": splits.get("test", {}).get("final_eval_results", {})
            }
        
        return summary

    @staticmethod
    def get_split_by_source_and_type(task_id: str, data_source_id: str, split_type: str) -> Optional[Any]:
        """获取指定数据源和分割类型的数据集信息"""
        with safe_get_session() as session:
            statement = select(DatasetInfo).where(
                DatasetInfo.task_id == task_id,
                DatasetInfo.data_source_id == data_source_id,
                DatasetInfo.split_type == split_type
            )
            return session.exec(statement).first()
    
    @staticmethod
    def _determine_dataset_type(dataset_name: str, original_path: str) -> DatasetType:
        """
        自动判断数据集类型
        
        Args:
            dataset_name: 数据集名称（可能是HuggingFace数据集名、文件名或文件夹名）
            original_path: 原始的数据集路径配置
            
        Returns:
            数据集类型枚举: DatasetType.HUGGINGFACE, DatasetType.LOCAL_FILE, DatasetType.LOCAL_FOLDER
        """
        import os
        
        import re
        
        # 1. 文件扩展名检查 → LOCAL_FILE
        if any(dataset_name.endswith(ext) for ext in ['.json', '.jsonl', '.txt', '.csv', '.parquet', '.arrow']):
            return DatasetType.LOCAL_FILE
        
        # 2. 本地文件夹特征检查 → LOCAL_FOLDER
        
        # 包含Windows路径分隔符
        if '\\' in dataset_name:
            return DatasetType.LOCAL_FOLDER
        
        # 超过两级路径（多于一个/）
        if dataset_name.count('/') > 1:
            return DatasetType.LOCAL_FOLDER
        
        # 绝对路径检查
        if (dataset_name.startswith('/') or                           # Unix绝对路径
            re.match(r'^[A-Za-z]:[/\\]', dataset_name) or            # Windows盘符路径 C:/, D:\
            dataset_name.startswith('\\\\')):                        # UNC路径 \\server\share
            return DatasetType.LOCAL_FOLDER
        
        # 相对路径标识
        if (dataset_name.startswith('./') or 
            dataset_name.startswith('../')):
            return DatasetType.LOCAL_FOLDER
        
        # 本地存在性检查
        if os.path.exists(dataset_name):
            return DatasetType.LOCAL_FILE if os.path.isfile(dataset_name) else DatasetType.LOCAL_FOLDER
        
        # 3. 默认判断 → HUGGINGFACE
        # 包括 sentence-transformers/all-nli, microsoft/DialoGPT-medium, squad 等
        return DatasetType.HUGGINGFACE
    
    @staticmethod
    def _create_single_dataset_info(
        session: Session,
        task_id: str,
        data_source_id: str,           # 数据源ID
        dataset_name: str,             # 数据集基础名称（不含分割后缀）
        dataset_base_name: str,        # 与dataset_name保持一致（向后兼容）
        dataset_path: str,
        dataset_type: DatasetType,
        split_type: str,
        dataset: Dataset,
        target_column: str,
        loss_function: Optional[str],
        evaluator: Optional[str],
        hf_subset: Optional[str] = None,  # 新增：HF子配置
        configured_sample_size: int = 0   # 新增：配置的样本大小限制
    ) -> str:
        """创建单个数据集信息记录"""
        try:
            # 获取数据类型和列信息
            if target_column in dataset.column_names:
                sample_value = dataset[target_column][0]
                label_type = "int" if isinstance(sample_value, int) else "float"
            else:
                label_type = "unknown"
            
            # 提取数据集列信息
            column_info = {}
            if dataset is not None and hasattr(dataset, 'column_names'):
                for col_name in dataset.column_names:
                    try:
                        # 获取第一个非空值来推断列类型
                        sample_val = dataset[col_name][0]
                        if isinstance(sample_val, str):
                            column_info[col_name] = "string"
                        elif isinstance(sample_val, int):
                            column_info[col_name] = "int"
                        elif isinstance(sample_val, float):
                            column_info[col_name] = "float"
                        elif isinstance(sample_val, bool):
                            column_info[col_name] = "bool"
                        elif isinstance(sample_val, list):
                            column_info[col_name] = "list"
                        else:
                            column_info[col_name] = str(type(sample_val).__name__)
                    except:
                        column_info[col_name] = "unknown"
            
            dataset_info = DatasetInfo(
                task_id=task_id,
                data_source_id=data_source_id,        # 数据源ID
                dataset_name=dataset_name,            # 基础名称（不含分割后缀）
                dataset_base_name=dataset_base_name,  # 与dataset_name保持一致
                HF_subset=hf_subset,                  # HF子配置
                dataset_path=dataset_path,
                dataset_type=dataset_type,
                split_type=SplitType(split_type),
                dataset_status=DatasetStatus.LOADED,  # 成功加载
                error_message=None,
                total_samples=len(dataset),
                configured_sample_size=min(len(dataset), configured_sample_size) if configured_sample_size > 0 else len(dataset),  # 实际使用的样本数量
                target_column=target_column,
                label_type=label_type,
                column_names=column_info,
                loss_function=loss_function,
                evaluator=evaluator
            )
            
            session.add(dataset_info)
            session.flush()  # 获取ID但不提交
            return dataset_info.id
            
        except Exception as e:
            # 创建失败记录
            logger.error(f"创建数据集信息记录失败: {e}")
            dataset_info = DatasetInfo(
                task_id=task_id,
                data_source_id=data_source_id,        # 数据源ID
                dataset_name=dataset_name,            # 基础名称（不含分割后缀）
                dataset_base_name=dataset_base_name,  # 与dataset_name保持一致
                HF_subset=hf_subset,                  # HF子配置
                dataset_path=dataset_path,
                dataset_type=DatasetType.FAILED,
                split_type=SplitType(split_type),
                dataset_status=DatasetStatus.FAILED,
                error_message=str(e),
                total_samples=0,
                configured_sample_size=configured_sample_size if configured_sample_size > 0 else 0,  # 失败时记录原始配置
                target_column=target_column,
                label_type="unknown",
                column_names={},
                loss_function=loss_function,
                evaluator=evaluator
            )
            
            session.add(dataset_info)
            session.flush()
            return dataset_info.id
    
    @staticmethod
    def update_eval_results(
        dataset_id: str,
        base_results: Optional[Dict[str, Any]] = None,
        final_results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新数据集评估结果
        
        Args:
            dataset_id: 数据集信息ID
            base_results: 基线评估结果
            final_results: 最终评估结果
            
        Returns:
            是否更新成功
        """
        with safe_get_session() as session:
            statement = select(DatasetInfo).where(DatasetInfo.id == dataset_id)
            dataset_info = session.exec(statement).first()
            
            if not dataset_info:
                logger.error(f"数据集信息不存在: {dataset_id}")
                return False
            
            if base_results:
                dataset_info.base_eval_results = base_results
            if final_results:
                dataset_info.final_eval_results = final_results
                
            dataset_info.update_time = datetime.now()
            session.add(dataset_info)
            session.commit()
            
            # 自动更新评估状态
            TrainingDatasetService.auto_update_evaluation_status_from_results(dataset_id)
            
            logger.info(f"更新数据集评估结果成功: {dataset_id}")
            return True
    
    @staticmethod
    def add_training_evaluator_evaluation(
        dataset_id: str,
        eval_results: Dict[str, Any],
        step: int,
        epoch: Optional[float] = None
    ) -> bool:
        """
        添加训练过程中的评估器评估结果（仅用于验证集）
        
        Args:
            dataset_id: 数据集信息ID
            eval_results: 评估器评估结果
            step: 训练步数
            epoch: 训练轮次（可选）
            
        Returns:
            是否添加成功
        """
        with safe_get_session() as session:
            statement = select(DatasetInfo).where(DatasetInfo.id == dataset_id)
            dataset_info = session.exec(statement).first()
            
            if not dataset_info:
                logger.error(f"数据集信息不存在: {dataset_id}")
                return False
            
            # 构造评估记录
            eval_record = {
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "results": eval_results
            }
            
            # 获取现有历史记录
            if dataset_info.training_evaluator_evaluation is None:
                dataset_info.training_evaluator_evaluation = []
            
            # 添加新记录
            dataset_info.training_evaluator_evaluation.append(eval_record)
            dataset_info.update_time = datetime.now()
            
            session.add(dataset_info)
            session.commit()
            
            # 自动更新评估状态
            TrainingDatasetService.auto_update_evaluation_status_from_results(dataset_id)
            
            logger.info(f"添加训练评估器评估结果成功: {dataset_id}, step: {step}")
            return True
    
    @staticmethod
    def add_loss_record(
        dataset_id: str,
        loss_value: float,
        step: int,
        epoch: Optional[float] = None
    ) -> bool:
        """
        添加训练过程中的损失记录
        
        Args:
            dataset_id: 数据集信息ID
            loss_value: 损失值
            step: 训练步数
            epoch: 训练轮次（可选）
            
        Returns:
            是否添加成功
        """
        with safe_get_session() as session:
            statement = select(DatasetInfo).where(DatasetInfo.id == dataset_id)
            dataset_info = session.exec(statement).first()
            
            if not dataset_info:
                logger.error(f"数据集信息不存在: {dataset_id}")
                return False
            
            # 构造损失记录
            loss_record = {
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "loss": loss_value
            }
            
            # 获取现有历史记录
            if dataset_info.loss is None:
                dataset_info.loss = []
            
            # 添加新记录
            dataset_info.loss.append(loss_record)
            dataset_info.update_time = datetime.now()
            
            session.add(dataset_info)
            session.commit()
            
            # 自动更新评估状态（对于训练集，有loss即表示已训练）
            TrainingDatasetService.auto_update_evaluation_status_from_results(dataset_id)
            
            logger.info(f"添加损失记录成功: {dataset_id}, step: {step}, loss: {loss_value}")
            return True
    
    @staticmethod
    def update_dataset_loss_function(task_id: str, split_type: str, loss_function: str) -> bool:
        """
        更新数据集记录中的损失函数信息
        
        Args:
            task_id: 训练任务ID
            split_type: 数据集分割类型 ("train", "eval", "test")
            loss_function: 实际使用的损失函数名称
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with safe_get_session() as session:
                # 查找对应的数据集记录
                statement = select(DatasetInfo).where(
                    DatasetInfo.task_id == task_id,
                    DatasetInfo.split_type == SplitType(split_type)
                )
                dataset_info = session.exec(statement).first()
                
                if not dataset_info:
                    logger.warning(f"未找到数据集记录: task_id={task_id}, split_type={split_type}")
                    return False
                
                # 更新损失函数信息
                dataset_info.loss_function = loss_function
                dataset_info.update_time = datetime.now()
                
                session.add(dataset_info)
                session.commit()
                
                logger.info(f"更新数据集损失函数成功: task_id={task_id}, split_type={split_type}, loss_function={loss_function}")
                return True
                
        except Exception as e:
            logger.error(f"更新数据集损失函数失败: task_id={task_id}, split_type={split_type}, 错误: {e}")
            return False
    
    @staticmethod
    def update_dataset_loss_function_by_source(task_id: str, data_source_id: str, split_type: str, loss_function: str) -> bool:
        """
        根据数据源ID和分割类型更新损失函数（推荐方法）
        
        Args:
            task_id: 训练任务ID
            data_source_id: 数据源ID
            split_type: 分割类型 ("train", "eval", "test")
            loss_function: 实际使用的损失函数名称
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with safe_get_session() as session:
                # 根据数据源ID和分割类型查找记录
                statement = select(DatasetInfo).where(
                    DatasetInfo.task_id == task_id,
                    DatasetInfo.data_source_id == data_source_id,
                    DatasetInfo.split_type == split_type
                )
                dataset_info = session.exec(statement).first()
                
                if not dataset_info:
                    logger.error(f"未找到数据集: task_id={task_id}, data_source_id={data_source_id}, split_type={split_type}")
                    return False
                
                dataset_info.loss_function = loss_function
                dataset_info.update_time = datetime.now()
                session.add(dataset_info)
                session.commit()
                
                logger.info(f"更新数据集损失函数成功: {data_source_id}-{split_type} -> {loss_function}")
                return True
                
        except Exception as e:
            logger.error(f"更新数据集损失函数失败: {e}")
            return False

    @staticmethod
    def update_dataset_evaluator(task_id: str, split_type: str, evaluator: str) -> bool:
        """
        更新数据集记录中的评估器类型信息
        
        Args:
            task_id: 训练任务ID
            split_type: 数据集分割类型 ("train", "eval", "test")
            evaluator: 实际使用的评估器类型名称
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with safe_get_session() as session:
                # 查找对应的数据集记录
                statement = select(DatasetInfo).where(
                    DatasetInfo.task_id == task_id,
                    DatasetInfo.split_type == SplitType(split_type)
                )
                dataset_info = session.exec(statement).first()
                
                if not dataset_info:
                    logger.warning(f"未找到数据集记录: task_id={task_id}, split_type={split_type}")
                    return False
                
                # 更新评估器类型信息
                dataset_info.evaluator = evaluator
                dataset_info.update_time = datetime.now()
                
                session.add(dataset_info)
                session.commit()
                
                logger.info(f"更新数据集评估器类型成功: task_id={task_id}, split_type={split_type}, evaluator={evaluator}")
                return True
                
        except Exception as e:
            logger.error(f"更新数据集评估器类型失败: task_id={task_id}, split_type={split_type}, 错误: {e}")
            return False
    
    @staticmethod
    def update_dataset_evaluator_by_source(task_id: str, data_source_id: str, split_type: str, evaluator: str) -> bool:
        """
        根据数据源ID和分割类型更新评估器类型（推荐方法）
        
        Args:
            task_id: 训练任务ID
            data_source_id: 数据源ID
            split_type: 分割类型 ("eval", "test")
            evaluator: 评估器类型名称
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with safe_get_session() as session:
                # 根据数据源ID和分割类型查找记录
                statement = select(DatasetInfo).where(
                    DatasetInfo.task_id == task_id,
                    DatasetInfo.data_source_id == data_source_id,
                    DatasetInfo.split_type == split_type
                )
                dataset_info = session.exec(statement).first()
                
                if not dataset_info:
                    logger.error(f"未找到数据集: task_id={task_id}, data_source_id={data_source_id}, split_type={split_type}")
                    return False
                
                dataset_info.evaluator = evaluator
                dataset_info.update_time = datetime.now()
                session.add(dataset_info)
                session.commit()
                
                logger.info(f"更新数据集评估器类型成功: {data_source_id}-{split_type} -> {evaluator}")
                return True
                
        except Exception as e:
            logger.error(f"更新数据集评估器类型失败: {e}")
            return False

    @staticmethod
    def get_datasets_by_job_and_split(job_id: str, split_type: str) -> List[Dict[str, Any]]:
        """
        根据训练任务ID和分割类型获取数据集信息
        
        Args:
            job_id: 训练任务ID
            split_type: 分割类型 (train/eval/test)
            
        Returns:
            数据集信息字典列表，避免SQLAlchemy会话绑定问题
        """
        with safe_get_session() as session:
            statement = select(DatasetInfo).where(
                DatasetInfo.task_id == job_id,
                DatasetInfo.split_type == split_type
            )
            datasets = session.exec(statement).all()
            
            # 转换为字典列表，避免会话绑定问题
            return [
                {
                    "id": str(dataset.id),
                    "dataset_name": dataset.dataset_name,
                    "data_source_id": dataset.data_source_id,
                    "split_type": dataset.split_type,
                    "task_id": dataset.task_id
                }
                for dataset in datasets
            ]

    @staticmethod
    def get_datasets_with_eval_results_by_task(task_id: str) -> List[Dict[str, Any]]:
        """
        根据训练任务ID获取数据集信息及评估结果
        
        Args:
            task_id: 训练任务ID
            
        Returns:
            包含评估结果的数据集信息字典列表
        """
        with safe_get_session() as session:
            statement = select(DatasetInfo).where(DatasetInfo.task_id == task_id)
            datasets = session.exec(statement).all()
            
            # 转换为字典列表，包含评估结果
            return [
                {
                    "id": str(dataset.id),
                    "dataset_name": dataset.dataset_name,
                    "data_source_id": dataset.data_source_id,
                    "split_type": dataset.split_type,
                    "task_id": dataset.task_id,
                    "base_eval_results": dataset.base_eval_results,
                    "final_eval_results": dataset.final_eval_results,
                    "evaluation_status": dataset.evaluation_status.value if hasattr(dataset.evaluation_status, 'value') else dataset.evaluation_status if dataset.evaluation_status else None,
                    "create_time": dataset.create_time.isoformat() if dataset.create_time else None,
                    "update_time": dataset.update_time.isoformat() if dataset.update_time else None,
                    "configured_sample_size": dataset.configured_sample_size
                }
                for dataset in datasets
            ]
    
    @staticmethod
    def update_evaluation_status(dataset_id: str, evaluation_status: EvaluationStatus) -> bool:
        """
        更新数据集的评估执行状态
        
        Args:
            dataset_id: 数据集信息ID
            evaluation_status: 新的评估状态
            
        Returns:
            是否更新成功
        """
        try:
            with safe_get_session() as session:
                statement = select(DatasetInfo).where(DatasetInfo.id == dataset_id)
                dataset_info = session.exec(statement).first()
                
                if not dataset_info:
                    logger.error(f"数据集信息不存在: {dataset_id}")
                    return False
                
                dataset_info.evaluation_status = evaluation_status
                dataset_info.update_time = datetime.now()
                
                session.add(dataset_info)
                session.commit()
                
                logger.info(f"更新数据集评估状态成功: {dataset_id} -> {evaluation_status}")
                return True
                
        except Exception as e:
            logger.error(f"更新数据集评估状态失败: {dataset_id}, 错误: {e}")
            return False
    
    @staticmethod
    def update_evaluation_status_by_task_and_split(task_id: str, split_type: str, evaluation_status: EvaluationStatus) -> bool:
        """
        根据任务ID和分割类型更新评估状态
        
        Args:
            task_id: 训练任务ID
            split_type: 分割类型 ("train", "eval", "test")
            evaluation_status: 新的评估状态
            
        Returns:
            是否更新成功
        """
        try:
            with safe_get_session() as session:
                statement = select(DatasetInfo).where(
                    DatasetInfo.task_id == task_id,
                    DatasetInfo.split_type == SplitType(split_type)
                )
                datasets = session.exec(statement).all()
                
                updated_count = 0
                for dataset_info in datasets:
                    dataset_info.evaluation_status = evaluation_status
                    dataset_info.update_time = datetime.now()
                    session.add(dataset_info)
                    updated_count += 1
                
                session.commit()
                
                logger.info(f"批量更新评估状态成功: task_id={task_id}, split_type={split_type}, 更新了{updated_count}个数据集 -> {evaluation_status}")
                return True
                
        except Exception as e:
            logger.error(f"批量更新评估状态失败: task_id={task_id}, split_type={split_type}, 错误: {e}")
            return False
    
    @staticmethod
    def auto_update_evaluation_status_from_results(dataset_id: str) -> bool:
        """
        根据已有的评估结果自动更新评估状态
        
        Args:
            dataset_id: 数据集信息ID
            
        Returns:
            是否更新成功
        """
        try:
            with safe_get_session() as session:
                statement = select(DatasetInfo).where(DatasetInfo.id == dataset_id)
                dataset_info = session.exec(statement).first()
                
                if not dataset_info:
                    logger.error(f"数据集信息不存在: {dataset_id}")
                    return False
                
                # 根据分割类型和已有数据自动判断状态
                old_status = dataset_info.evaluation_status
                new_status = EvaluationStatus.NOT_EVALUATED
                
                if dataset_info.split_type == SplitType.TRAIN:
                    # 训练集：有loss数据即为已训练
                    if dataset_info.loss and len(dataset_info.loss) > 0:
                        new_status = EvaluationStatus.FINAL_EVALUATED
                    else:
                        new_status = EvaluationStatus.NOT_EVALUATED
                        
                elif dataset_info.split_type == SplitType.EVAL:
                    # 验证集：检查最终评估 > 训练评估器评估 > 基线评估 > 未评估
                    if dataset_info.final_eval_results:
                        new_status = EvaluationStatus.FINAL_EVALUATED  
                    elif dataset_info.training_evaluator_evaluation and len(dataset_info.training_evaluator_evaluation) > 0:
                        new_status = EvaluationStatus.TRAINING_EVALUATED
                    elif dataset_info.base_eval_results:
                        new_status = EvaluationStatus.BASE_EVALUATED
                    else:
                        new_status = EvaluationStatus.NOT_EVALUATED
                        
                elif dataset_info.split_type == SplitType.TEST:
                    # 测试集：检查最终评估 > 基线评估 > 未评估
                    if dataset_info.final_eval_results:
                        new_status = EvaluationStatus.FINAL_EVALUATED
                    elif dataset_info.base_eval_results:
                        new_status = EvaluationStatus.BASE_EVALUATED
                    else:
                        new_status = EvaluationStatus.NOT_EVALUATED
                
                # 只在状态确实改变时更新
                if new_status != old_status:
                    dataset_info.evaluation_status = new_status
                    dataset_info.update_time = datetime.now()
                    session.add(dataset_info)
                    session.commit()
                    logger.info(f"自动更新评估状态: {dataset_id} {old_status} -> {new_status}")
                else:
                    logger.debug(f"评估状态无需更新: {dataset_id} 保持为 {new_status}")
                
                return True
                
        except Exception as e:
            logger.error(f"自动更新评估状态失败: {dataset_id}, 错误: {e}")
            return False
    
