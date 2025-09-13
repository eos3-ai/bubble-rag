"""
统一训练API路由
支持串行(serial)和并行(parallel)训练模式的统一接口
"""
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, Any, List
from datetime import datetime

from bubble_rag.entity.query.response_model import SrvResult
from bubble_rag.training.model_sft.services.unified_training_service import unified_training_service
from bubble_rag.training.model_sft.services.dataset_service import dataset_service
from bubble_rag.training.model_sft.services.model_service import model_service
from bubble_rag.training.model_sft.services.config_service import config_service
from bubble_rag.training.model_sft.models.training_task import TrainingTaskCreateRequest
from bubble_rag.training.model_sft.models.unified_config import UnifiedTrainingConfig
from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
from bubble_rag.training.model_sft.enums import TrainingStatus
from bubble_rag.training.model_sft.enums.training_task_enums import ProcessStatus
from bubble_rag.training.model_sft.utils.error_handler import handle_api_error
from bubble_rag.training.model_sft.utils.gpu_resource_manager import gpu_resource_manager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class StartTrainingRequest(UnifiedTrainingConfig):
    """启动训练请求模型（继承统一配置）"""
    training_mode: Optional[str] = Field(default="parallel", description="训练模式: serial(串行) 或 parallel(并行)")

# 公共数据转换函数
def convert_task_to_dict(task) -> Dict[str, Any]:
    """将训练任务对象转换为字典格式"""
    task_data = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "description": task.description,
        "train_type": task.train_type,
        "dataset_name_or_path": task.dataset_name_or_path,
        "output_dir": task.output_dir,
        "device": task.device,
        "status": task.status,
        "progress": task.progress,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "duration_seconds": task.duration_seconds,
        "final_model_path": task.final_model_path,
        "error_message": task.error_message,
        "training_params": task.training_params or {}
    }
    return task_data

@router.post("/start_training")
def start_training(request: StartTrainingRequest):
    """
    启动训练任务（统一接口）
    
    支持两种训练模式：
    - serial: 串行训练，一次只能运行一个任务
    - parallel: 并行训练，可同时运行多个任务
    """
    try:
        # 检查服务实例ID - 确保服务隔离功能正常
        if not unified_training_service.service_instance_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法创建训练任务！"
            )
        # 提取训练模式
        training_mode = request.training_mode or "parallel"
        
        # 分离训练模式和其他参数
        request_data = request.model_dump(exclude={"training_mode"}, exclude_unset=True)
        
        # 分离核心任务参数和训练参数
        core_task_fields = {
            "task_name", "description", "train_type", "model_name_or_path", 
            "dataset_name_or_path", "HF_subset", "output_dir", "device"
        }
        
        core_params = {k: v for k, v in request_data.items() if k in core_task_fields}
        training_params = {k: v for k, v in request_data.items() if k not in core_task_fields}
        
        # 合并用户提供的training_params
        if request.training_params:
            training_params.update(request.training_params)
        
        # 直接使用Pydantic的TrainingParameters进行验证
        try:
            from bubble_rag.training.model_sft.models.training_parameters import TrainingParameters
            validated_training_params = TrainingParameters(**training_params)
            training_params_dict = validated_training_params.model_dump(exclude_unset=True)
            
            logger.info(f"训练参数验证成功，共 {len(training_params_dict)} 个参数")
            
        except Exception as e:
            logger.error(f"训练参数验证失败: {e}")
            # 如果是 Pydantic 验证错误，提供更友好的错误信息
            if hasattr(e, 'errors'):
                error_details = []
                for error in e.errors():
                    field = '.'.join(str(loc) for loc in error['loc'])
                    error_details.append(f"{field}: {error['msg']}")
                error_msg = "参数验证失败: " + "; ".join(error_details)
            else:
                error_msg = f"训练参数验证失败: {e}"
            
            return SrvResult(code=400, msg=error_msg)
        
        # 创建训练任务请求
        training_request = TrainingTaskCreateRequest(
            task_name=core_params.get("task_name"),
            description=core_params.get("description"),
            train_type=core_params["train_type"],
            model_name_or_path=core_params["model_name_or_path"],
            dataset_name_or_path=core_params["dataset_name_or_path"],
            HF_subset=core_params.get("HF_subset"),
            output_dir=core_params.get("output_dir"),
            device=core_params.get("device", "auto"),
            training_params=training_params_dict
        )
        
        logger.info(f"启动训练任务请求，模式: {training_mode}")
        logger.info(f"训练参数: {list(training_request.training_params.keys())}")
        
        # 使用统一训练服务启动任务
        task = unified_training_service.start_training(training_request, training_mode=training_mode)
        
        # 构建响应数据
        response_data = convert_task_to_dict(task)
        response_data["training_mode"] = training_mode
        
        return SrvResult(
            code=200,
            msg=f"训练任务启动成功 (模式: {training_mode})",
            data=response_data
        )
        
    except ValidationError as ve:
        logger.error(f"请求参数验证失败: {ve}")
        error_details = []
        for error in ve.errors():
            field = '.'.join(str(loc) for loc in error['loc']) 
            error_details.append(f"{field}: {error['msg']}")
        return SrvResult(
            code=422,
            msg=f"请求参数验证失败: {'; '.join(error_details)}"
        )
    except Exception as e:
        logger.error(f"启动训练任务失败: {str(e)}", exc_info=True)
        handle_api_error(e, "start_training")  # 记录错误到日志
        return SrvResult(
            code=500,
            msg=f"启动训练任务失败: {str(e)}"
        )

@router.post("/stop_training")
def stop_training(task_id: str = Query(..., description="任务ID")):
    """停止训练任务"""
    try:
        # 🔐 安全检查：验证任务是否属于当前服务实例
        task_db = training_task_service.get_training_task(task_id)
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        # 检查服务实例归属权限
        current_service_id = unified_training_service.service_instance_id
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务停止任务被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403, 
                msg=f"权限不足：任务属于其他服务实例，无法停止"
            )
        
        success = unified_training_service.stop_training(task_id)
        
        if success:
            return SrvResult(
                code=200,
                msg="训练任务已停止",
                data={"task_id": task_id, "stopped": True}
            )
        else:
            return SrvResult(code=500, msg=f"停止训练任务失败: {task_id}")
            
    except Exception as e:
        logger.error(f"停止训练任务失败: {str(e)}", exc_info=True)
        handle_api_error(e, "stop_training")
        return SrvResult(code=500, msg=f"停止训练任务失败: {str(e)}")

@router.get("/tasks/{task_id}")
def get_task_detail(task_id: str):
    """获取任务详情（从数据库获取完整信息，包含持久化的进度状态）"""
    try:
        # 🔧 从数据库获取完整任务信息（持久化数据）
        task_db = training_task_service.get_training_task(task_id)
        
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        # 🔐 安全检查：验证任务是否属于当前服务实例
        current_service_id = unified_training_service.service_instance_id
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务查询任务被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403,
                msg=f"权限不足：任务属于其他服务实例，无法查看详情"
            )
        
        # 获取运行进程信息（实时状态）
        running_processes = unified_training_service.get_running_processes()
        is_running = task_id in running_processes
        process_info = running_processes.get(task_id, {}).get('process_info', {}) if is_running else None
        
        # 🔧 修复进度显示问题：使用与progress接口相同的混合数据源策略
        # 确保两个接口返回一致的进度数据
        real_time_progress = task_db.progress or 0.0  # 优先使用数据库进度（可靠）
        memory_progress = 0.0
        sync_status = "unknown"
        
        try:
            from bubble_rag.training.model_sft.services.task_manager import task_manager
            memory_task = task_manager.get_task(task_id)
            if memory_task:
                memory_progress = memory_task.progress
                # 计算同步状态
                sync_diff = abs(real_time_progress - memory_progress)
                sync_status = "synced" if sync_diff < 1 else "out_of_sync"
                logger.debug(f"任务详情API: 数据库进度={real_time_progress}%, 内存进度={memory_progress}%, 同步状态={sync_status}")
            else:
                logger.debug(f"内存中未找到任务，使用数据库进度: {real_time_progress}%")
                sync_status = "memory_not_found"
        except Exception as e:
            logger.warning(f"获取内存任务失败，使用数据库进度: {e}")
            sync_status = "memory_error"
            
        # 确保进度在合理范围内
        real_time_progress = max(0.0, min(100.0, real_time_progress))
        
        # 🔧 调试日志：输出最终使用的进度值
        logger.info(f"🔍 任务详情API调试: task_id={task_id}, 数据库原始进度={task_db.progress}, 最终进度={real_time_progress}, 同步状态={sync_status}")
        
        # 计算预估剩余时间（使用实时进度）
        estimated_time = None
        if task_db.started_at and real_time_progress > 0 and real_time_progress < 100:
            elapsed = (datetime.now() - task_db.started_at).total_seconds()
            estimated_total = elapsed / (real_time_progress / 100)
            estimated_time = max(0, estimated_total - elapsed)
        
        # 📊 详细任务信息
        task_detail = {
            # 基础信息
            "task_id": task_db.task_id,
            "task_name": task_db.task_name,
            "description": task_db.description,
            "train_type": task_db.train_type,
            "model_name_or_path": task_db.model_name_or_path,
            "dataset_name_or_path": task_db.dataset_name_or_path,
            "HF_subset": getattr(task_db, 'HF_subset', None),  # 新增字段
            "output_dir": task_db.output_dir,
            "device": task_db.device,
            
            # 状态信息
            "status": task_db.status,
            "progress": real_time_progress,
            "is_running": is_running,
            
            # 时间信息
            "created_at": task_db.created_at.isoformat() if task_db.created_at else None,
            "started_at": task_db.started_at.isoformat() if task_db.started_at else None,
            "completed_at": task_db.completed_at.isoformat() if task_db.completed_at else None,
            "estimated_time_remaining": estimated_time,
            
            # 结果信息
            "final_model_path": task_db.final_model_path,
            "error_message": task_db.error_message,
            
            # 进程信息（如果正在运行）
            "process_info": process_info,
            
            # 训练参数
            "training_params": task_db.training_params
        }
        
        return SrvResult(code=200, msg="获取任务详情成功", data=task_detail)
            
    except Exception as e:
        logger.error(f"获取任务详情失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_task_detail")
        return SrvResult(code=500, msg=f"获取任务详情失败: {str(e)}")

@router.get("/tasks/{task_id}/datasets")
def get_task_datasets(task_id: str):
    """获取任务的训练数据集信息 - 仅当前服务实例"""
    try:
        # 🔐 安全检查：先验证任务是否属于当前服务实例
        task_db = training_task_service.get_training_task(task_id)
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法获取数据集信息！"
            )
        
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务查询数据集被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403,
                msg=f"权限不足：任务属于其他服务实例，无法查看数据集信息"
            )
        
        # 📊 获取任务的数据集信息
        try:
            # 获取所有数据源
            data_sources = TrainingDatasetService.get_data_sources_by_task(task_id)
            
            if not data_sources:
                return SrvResult(
                    code=200,
                    msg="任务暂无数据集信息",
                    data={
                        "task_id": task_id,
                        "data_sources": [],
                        "summary": {
                            "total_sources": 0,
                            "total_samples": 0,
                            "actual_total_samples": 0,
                            "available_splits": []
                        }
                    }
                )
            
            # 获取详细数据集信息
            datasets_info = []
            total_samples = 0
            actual_total_samples = 0
            all_splits = set()
            
            for source_id in data_sources:
                source_info = TrainingDatasetService.get_splits_by_source(task_id, source_id)
                datasets_info.append({
                    "data_source_id": source_id,
                    "dataset_base_name": source_info["base_name"],
                    "dataset_path": source_info["path"],
                    "splits": source_info["splits"]
                })
                
                # 统计信息
                for split_info in source_info["splits"].values():
                    total_samples += split_info.get("samples", 0)
                    actual_total_samples += split_info.get("actual_samples", 0)
                    all_splits.add(split_info.get("split_type", "unknown"))
            
            # 获取性能摘要
            performance_summary = TrainingDatasetService.get_source_performance_summary(task_id)
            
            datasets_data = {
                "task_id": task_id,
                "data_sources": datasets_info,
                "performance_summary": performance_summary,
                "summary": {
                    "total_sources": len(data_sources),
                    "total_samples": total_samples,  # 原始数据集总样本数
                    "actual_total_samples": actual_total_samples,  # 实际训练使用总样本数
                    "available_splits": list(all_splits),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return SrvResult(
                code=200,
                msg="获取任务数据集信息成功",
                data=datasets_data
            )
            
        except Exception as dataset_error:
            logger.error(f"获取数据集信息失败: {dataset_error}")
            return SrvResult(
                code=500,
                msg=f"获取数据集信息失败: {str(dataset_error)}"
            )
            
    except Exception as e:
        logger.error(f"获取任务数据集失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_task_datasets")
        return SrvResult(code=500, msg=f"获取任务数据集失败: {str(e)}")

@router.get("/tasks/{task_id}/training_metrics")
def get_task_training_metrics(task_id: str, limit: Optional[int] = Query(None, ge=1, le=10000, description="限制返回的loss记录数量")):
    """获取任务的训练指标和loss历史 - 仅当前服务实例"""
    try:
        # 🔐 安全检查：先验证任务是否属于当前服务实例
        task_db = training_task_service.get_training_task(task_id)
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法获取训练指标！"
            )
        
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务查询训练指标被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403,
                msg=f"权限不足：任务属于其他服务实例，无法查看训练指标"
            )
        
        # 📊 获取训练指标和loss历史
        try:
            # 检查本地loss文件是否存在
            output_dir = task_db.output_dir
            if not output_dir:
                return SrvResult(
                    code=404,
                    msg="任务输出目录未配置，无法获取训练指标"
                )
            
            from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
            
            # 创建或获取loss管理器
            loss_manager = get_loss_manager(output_dir, task_id)
            
            # 获取训练指标汇总
            training_metrics = loss_manager.get_training_metrics()
            
            # 获取loss历史记录
            loss_history = loss_manager.get_loss_history(limit=limit)
            
            # 构建响应数据
            metrics_data = {
                "task_id": task_id,
                "training_metrics": training_metrics,
                "loss_history": loss_history,
                "loss_history_count": len(loss_history),
                "total_loss_records": training_metrics.get("loss_records_count", 0),
                "files_info": {
                    "loss_history_file": str(loss_manager.loss_history_file),
                    "training_metrics_file": str(loss_manager.training_metrics_file),
                    "loss_history_exists": loss_manager.loss_history_file.exists(),
                    "training_metrics_exists": loss_manager.training_metrics_file.exists()
                }
            }
            
            return SrvResult(
                code=200,
                msg="获取训练指标成功",
                data=metrics_data
            )
            
        except Exception as metrics_error:
            logger.error(f"获取训练指标失败: {metrics_error}")
            return SrvResult(
                code=500,
                msg=f"获取训练指标失败: {str(metrics_error)}"
            )
            
    except Exception as e:
        logger.error(f"获取任务训练指标失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_task_training_metrics")
        return SrvResult(code=500, msg=f"获取任务训练指标失败: {str(e)}")

@router.get("/tasks/{task_id}/progress")
def get_task_progress(task_id: str):
    """获取任务实时进度（从内存获取，高频轮询优化）"""
    try:
        # 🚀 直接从内存获取实时进度（避免数据库查询延迟）
        from bubble_rag.training.model_sft.services.task_manager import task_manager
        task = task_manager.get_task(task_id)
        
        if not task:
            # 如果内存中没有，尝试从数据库验证任务是否存在
            task_db = training_task_service.get_training_task(task_id)
            if not task_db:
                return SrvResult(code=404, msg=f"未找到任务: {task_id}")
            
            # 🔐 安全检查：验证任务归属
            current_service_id = unified_training_service.service_instance_id
            if task_db.service_instance_id != current_service_id:
                logger.warning(f"🚫 跨服务查询进度被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
                return SrvResult(
                    code=403,
                    msg=f"权限不足：任务属于其他服务实例，无法查看进度"
                )
            
            # 任务存在但不在内存中（可能已完成），返回数据库中的状态
            progress_data = {
                "task_id": task_db.task_id,
                "status": task_db.status,
                "progress": task_db.progress,
                "is_running": False,
                "timestamp": datetime.now().isoformat(),
                "source": "database"  # 标识数据来源
            }
            return SrvResult(code=200, msg="获取任务进度成功（来源：数据库）", data=progress_data)
        
        # 🔐 安全检查：验证任务归属（只在找到内存任务时检查）
        try:
            task_db = training_task_service.get_training_task(task_id)
            if task_db:
                current_service_id = unified_training_service.service_instance_id
                if task_db.service_instance_id != current_service_id:
                    logger.warning(f"🚫 跨服务查询进度被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
                    return SrvResult(
                        code=403,
                        msg=f"权限不足：任务属于其他服务实例，无法查看进度"
                    )
        except Exception as e:
            logger.warning(f"安全检查失败，继续返回进度: {e}")
        
        # 获取运行进程信息  
        running_processes = unified_training_service.get_running_processes()
        is_running = task_id in running_processes
        
        # 🚀 实时进度信息（从内存获取，高频轮询优化）
        # 🔧 修复进度同步问题：优先从数据库获取最新进度，确保准确性
        try:
            task_db = training_task_service.get_training_task(task_id)
            if task_db and task_db.service_instance_id == unified_training_service.service_instance_id:
                # 使用数据库中的最新进度数据，因为它更可靠
                db_progress = task_db.progress or 0
                db_status = task_db.status
                
                # 但是使用内存中的is_running状态（更实时）
                memory_is_running = is_running
                
                progress_data = {
                    "task_id": task.task_id,
                    "status": db_status,  # 使用数据库状态
                    "progress": db_progress,  # 使用数据库进度
                    "progress_percentage": db_progress,  # 兼容字段
                    "is_running": memory_is_running,  # 使用内存运行状态
                    "timestamp": datetime.now().isoformat(),
                    "source": "hybrid",  # 混合数据源
                    # 添加训练详情
                    "training_details": {
                        "memory_progress": task.progress,
                        "database_progress": db_progress,
                        "sync_status": "synced" if abs(task.progress - db_progress) < 1 else "out_of_sync",
                        "task_started_at": task.started_at.isoformat() if task.started_at else None,
                        "process_running": memory_is_running
                    }
                }
                
                logger.info(f"进度API混合模式: 任务{task_id} DB进度={db_progress}%, 内存进度={task.progress}%, 运行状态={memory_is_running}")
            else:
                # 回退到纯内存模式
                progress_data = {
                    "task_id": task.task_id,
                    "status": task.status,
                    "progress": task.progress,
                    "progress_percentage": task.progress,
                    "is_running": is_running,
                    "timestamp": datetime.now().isoformat(),
                    "source": "memory_fallback"
                }
        except Exception as e:
            logger.warning(f"获取数据库进度失败，使用内存数据: {e}")
            # 如果数据库查询失败，使用原有内存数据
            progress_data = {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "progress_percentage": task.progress,
                "is_running": is_running,
                "timestamp": datetime.now().isoformat(),
                "source": "memory_emergency"
            }
        
        return SrvResult(code=200, msg="获取任务进度成功（来源：内存）", data=progress_data)
        
    except Exception as e:
        logger.error(f"获取任务进度失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_task_progress")
        return SrvResult(code=500, msg=f"获取任务进度失败: {str(e)}")

@router.get("/training_logs")
def get_training_logs(
    task_id: str = Query(..., description="任务ID"),
    lines: int = Query(50, description="获取日志行数", ge=1, le=1000)
):
    """
    获取训练日志和loss数据
    
    功能：
    1. 获取训练过程中的文本日志信息
    2. 获取完整的loss历史数据（train_loss, eval_loss等）
    3. 提供统一的训练监控数据接口
    
    Args:
        task_id: 训练任务ID
        lines: 获取的日志行数（1-1000）
        
    Returns:
        包含日志和loss数据的综合信息
    """
    try:
        # 🔐 安全检查：先验证任务是否属于当前服务实例
        task_db = training_task_service.get_training_task(task_id)
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        current_service_id = unified_training_service.service_instance_id
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务查询日志被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403,
                msg=f"权限不足：任务属于其他服务实例，无法查看日志"
            )
        
        # 从任务管理器获取任务信息
        from bubble_rag.training.model_sft.services.task_manager import task_manager
        task = task_manager.get_task(task_id)
        
        if not task:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        # 获取最近的日志
        recent_logs = task.logs[-lines:] if task.logs else []
        
        # 🆕 同时获取loss数据
        loss_data = []
        try:
            from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
            output_dir = task_db.output_dir or "/tmp/training_output"
            loss_manager = get_loss_manager(output_dir, task_id)
            loss_data = loss_manager.get_loss_history()
        except Exception as e:
            logger.warning(f"获取loss数据失败: {e}")
        
        return SrvResult(
            code=200,
            msg="获取训练日志成功",
            data={
                "task_id": task.task_id,
                "logs": recent_logs,
                "total_logs": len(task.logs) if task.logs else 0,
                "requested_lines": lines,
                "loss_data": loss_data,
                "total_loss_records": len(loss_data)
            }
        )
        
    except Exception as e:
        logger.error(f"获取训练日志失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_training_logs")
        return SrvResult(code=500, msg=f"获取训练日志失败: {str(e)}")

@router.get("/running_tasks")
def get_running_tasks():
    """获取正在运行的训练任务列表 - 仅返回当前服务实例的任务"""
    try:
        # 🔐 安全检查：确保只获取当前服务实例的任务
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法获取运行任务列表！"
            )
        
        running_processes = unified_training_service.get_running_processes()
        
        tasks_info = []
        for task_id, process_info in running_processes.items():
            # 🔐 验证任务归属权限
            task_db = training_task_service.get_training_task(task_id)
            if not task_db or task_db.service_instance_id != current_service_id:
                logger.warning(f"🚫 跨服务获取运行任务被过滤: 任务 {task_id}")
                continue
            
            # 从任务管理器获取任务详细信息
            from bubble_rag.training.model_sft.services.task_manager import task_manager
            task = task_manager.get_task(task_id)
            
            if task:
                # 🔧 使用混合数据源获取最准确的进度（类似progress接口）
                try:
                    # 获取数据库中的最新进度
                    actual_progress = task.progress  # 默认使用内存进度
                    actual_status = task.status
                    
                    if task_db:
                        db_progress = task_db.progress or 0
                        db_status = task_db.status
                        
                        # 优先使用数据库进度（更可靠）
                        actual_progress = db_progress
                        actual_status = db_status
                        
                        logger.debug(f"运行任务 {task_id}: DB进度={db_progress}%, 内存进度={task.progress}%")
                except Exception as e:
                    logger.warning(f"获取任务 {task_id} 最新进度失败: {e}")
                    actual_progress = task.progress
                    actual_status = task.status
                
                task_info = convert_task_to_dict(task)
                # 使用最准确的进度和状态
                task_info["progress"] = actual_progress
                task_info["status"] = actual_status
                
                task_info["process_info"] = {
                    "pid": process_info["pid"],
                    "mode": process_info["process_info"].get("mode", "unknown"),
                    "started_at": process_info["process_info"].get("started_at").isoformat() if process_info["process_info"].get("started_at") and hasattr(process_info["process_info"].get("started_at"), "isoformat") else None
                }
                tasks_info.append(task_info)
        
        return SrvResult(
            code=200,
            msg="获取运行中任务列表成功",
            data={
                "running_tasks": tasks_info,
                "total": len(tasks_info)
            }
        )
        
    except Exception as e:
        logger.error(f"获取运行中任务列表失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_running_tasks")
        return SrvResult(code=500, msg=f"获取运行中任务列表失败: {str(e)}")

@router.get("/tasks")
def get_tasks(
    limit: int = Query(20, description="返回记录数限制", ge=1, le=100),
    offset: int = Query(0, description="偏移量", ge=0),
    status: Optional[str] = Query(None, description="按状态过滤: PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED"),
    train_type: Optional[str] = Query(None, description="按训练类型过滤: embedding, reranker")
):
    """获取任务列表（支持分页和过滤）- 仅返回当前服务实例的任务"""
    try:
        # 🔐 安全检查：确保只获取当前服务实例的任务
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法获取任务列表！"
            )
        
        # 🔧 从数据库获取当前服务实例的任务列表（支持过滤）
        all_service_tasks = training_task_service.get_tasks_by_service_instance(current_service_id)
        
        # 应用过滤条件
        filtered_tasks = all_service_tasks
        if status:
            filtered_tasks = [t for t in filtered_tasks if t.status == status]
        if train_type:
            filtered_tasks = [t for t in filtered_tasks if t.train_type == train_type]
        
        # 计算总数并应用分页
        total_count = len(filtered_tasks)
        tasks = filtered_tasks[offset:offset + limit]
        
        # 获取运行进程信息（用于is_running状态）
        running_processes = unified_training_service.get_running_processes()
        
        # 转换为列表格式（概览信息）
        tasks_data = []
        for task in tasks:
            is_running = task.task_id in running_processes
            
            # 📋 任务概览信息（比详情轻量）
            task_overview = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "train_type": task.train_type,
                "model_name_or_path": task.model_name_or_path,
                "dataset_name_or_path": task.dataset_name_or_path[:50] + "..." if len(task.dataset_name_or_path or "") > 50 else task.dataset_name_or_path,  # 截断长路径
                "HF_subset": getattr(task, 'HF_subset', None),
                "status": task.status,
                "progress": task.progress,
                "is_running": is_running,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error_message": task.error_message[:100] + "..." if task.error_message and len(task.error_message) > 100 else task.error_message  # 截断长错误信息
            }
            tasks_data.append(task_overview)
        
        # total_count 已经在上面设置好了
        
        return SrvResult(
            code=200,
            msg="获取任务列表成功",
            data={
                "tasks": tasks_data,
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        )
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_tasks")
        return SrvResult(code=500, msg=f"获取任务列表失败: {str(e)}")

# 配置和验证接口（复用原有逻辑）
@router.post("/config/validate")
def validate_config(request: UnifiedTrainingConfig):
    """验证训练配置"""
    try:
        is_valid, message = config_service.validate_training_config(request.model_dump())
        
        return SrvResult(
            code=200,
            msg="配置验证完成",
            data={"valid": is_valid, "message": message}
        )
        
    except Exception as e:
        logger.error(f"验证训练配置失败: {str(e)}", exc_info=True)
        handle_api_error(e, "validate_config")
        return SrvResult(code=500, msg=f"验证训练配置失败: {str(e)}")

@router.get("/gpu/status")
def get_gpu_status():
    """获取GPU资源状态"""
    try:
        gpu_status = gpu_resource_manager.get_resource_status()
        return SrvResult(code=200, msg="获取GPU状态成功", data=gpu_status)
        
    except Exception as e:
        logger.error(f"获取GPU状态失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_gpu_status")
        return SrvResult(code=500, msg=f"获取GPU状态失败: {str(e)}")

# 数据集相关接口（复用原有逻辑）
@router.get("/datasets/list")
def list_available_datasets():
    """列出可用的数据集"""
    try:
        datasets = dataset_service.list_available_datasets()
        return SrvResult(code=200, msg="获取数据集列表成功", data=datasets)
        
    except Exception as e:
        logger.error(f"获取数据集列表失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_datasets")
        return SrvResult(code=500, msg=f"获取数据集列表失败: {str(e)}")

@router.post("/datasets/validate")
def validate_dataset(dataset_path: str = Query(..., description="数据集路径")):
    """验证数据集"""
    try:
        is_valid, message, sample_data = dataset_service.validate_dataset(dataset_path)
        
        return SrvResult(
            code=200,
            msg="数据集验证完成",
            data={
                "valid": is_valid,
                "message": message,
                "sample_data": sample_data
            }
        )
        
    except Exception as e:
        logger.error(f"验证数据集失败: {str(e)}", exc_info=True)
        handle_api_error(e, "validate_dataset")
        return SrvResult(code=500, msg=f"验证数据集失败: {str(e)}")

@router.post("/datasets/preview")
def preview_dataset(
    dataset_path: str = Query(..., description="数据集路径"),
    max_samples: int = Query(5, description="预览样本数量", ge=1, le=50)
):
    """预览数据集"""
    try:
        preview_data = dataset_service.preview_dataset(dataset_path, max_samples)
        return SrvResult(code=200, msg="数据集预览成功", data=preview_data)
        
    except Exception as e:
        logger.error(f"预览数据集失败: {str(e)}", exc_info=True)
        handle_api_error(e, "preview_dataset")
        return SrvResult(code=500, msg=f"预览数据集失败: {str(e)}")

@router.get("/service/health")
def get_service_health():
    """获取服务健康状态和实例信息"""
    try:
        from bubble_rag.training.model_sft.utils.service_instance import service_instance_manager
        
        service_id = unified_training_service.service_instance_id
        health_status = "healthy" if service_id else "critical"
        
        health_data = {
            "service_instance_id": service_id,
            "health_status": health_status,
            "service_isolation": service_id is not None,
            "default_training_mode": unified_training_service.default_mode,
            "running_tasks_count": len(unified_training_service.get_running_processes()),
            "instance_info": service_instance_manager.get_instance_info() if service_id else None,
            "timestamp": datetime.now().isoformat()
        }
        
        if not service_id:
            health_data["warning"] = "❌ 服务实例ID为空，服务隔离功能异常！"
        
        return SrvResult(code=200, msg="服务健康状态获取成功", data=health_data)
        
    except Exception as e:
        logger.error(f"获取服务健康状态失败: {str(e)}", exc_info=True)
        handle_api_error(e, "service_health")
        return SrvResult(code=500, msg=f"获取服务健康状态失败: {str(e)}")

@router.get("/process/status/stats")
def get_process_status_statistics():
    """获取进程状态统计信息 - 仅当前服务实例"""
    try:
        # 🔐 安全检查：确保只获取当前服务实例的统计
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法获取统计信息！"
            )
        
        # 获取当前服务实例的任务统计
        service_tasks = training_task_service.get_tasks_by_service_instance(current_service_id)
        stats = {}
        for task in service_tasks:
            process_status = task.process_status or ProcessStatus.UNKNOWN.value
            stats[process_status] = stats.get(process_status, 0) + 1
        
        # 添加更详细的统计信息
        detailed_stats = {
            "total_tasks": sum(stats.values()),
            "status_breakdown": stats,
            "manageable_count": sum(stats.get(status, 0) for status in ProcessStatus.get_manageable_statuses()),
            "active_count": sum(stats.get(status, 0) for status in ProcessStatus.get_active_statuses()),
            "final_count": sum(stats.get(status, 0) for status in ProcessStatus.get_final_statuses()),
            "unknown_count": stats.get(ProcessStatus.UNKNOWN.value, 0),
            "timestamp": datetime.now().isoformat()
        }
        
        return SrvResult(
            code=200,
            msg="获取进程状态统计成功",
            data=detailed_stats
        )
        
    except Exception as e:
        logger.error(f"获取进程状态统计失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_process_status_stats")
        return SrvResult(code=500, msg=f"获取进程状态统计失败: {str(e)}")

@router.get("/process/status/unknown")
def get_unknown_process_tasks():
    """获取UNKNOWN状态的进程任务 - 仅当前服务实例"""
    try:
        # 🔐 安全检查：确保只获取当前服务实例的任务
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法获取UNKNOWN任务！"
            )
        
        # 获取当前服务实例的UNKNOWN状态任务
        service_tasks = training_task_service.get_tasks_by_service_instance(current_service_id)
        unknown_tasks = [task for task in service_tasks if (task.process_status or ProcessStatus.UNKNOWN.value) == ProcessStatus.UNKNOWN.value]
        
        tasks_data = []
        for task in unknown_tasks:
            task_dict = training_task_service._task_db_to_dict(task)
            tasks_data.append({
                "task_id": task_dict["task_id"],
                "task_name": task_dict["task_name"],
                "process_pid": task_dict["process_pid"],
                "process_status": task_dict["process_status"],
                "status": task_dict["status"],
                "created_at": task_dict["created_at"],
                "updated_at": task_dict["updated_at"],
                "service_instance_id": task_dict["service_instance_id"]
            })
        
        return SrvResult(
            code=200,
            msg="获取UNKNOWN状态任务成功",
            data={
                "unknown_tasks": tasks_data,
                "count": len(tasks_data),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"获取UNKNOWN状态任务失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_unknown_tasks")
        return SrvResult(code=500, msg=f"获取UNKNOWN状态任务失败: {str(e)}")

@router.post("/process/status/recovery/unknown")
def recover_unknown_processes():
    """主动触发UNKNOWN状态进程恢复 - 仅当前服务实例"""
    try:
        # 🔐 安全检查：确保只恢复当前服务实例的进程
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法恢复进程！"
            )
        
        # 通过统一训练服务触发恢复（只处理当前服务实例的任务）
        recovery_stats = unified_training_service.check_unknown_processes()
        
        return SrvResult(
            code=200,
            msg="UNKNOWN状态进程恢复完成",
            data={
                "recovery_result": recovery_stats,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"UNKNOWN状态进程恢复失败: {str(e)}", exc_info=True)
        handle_api_error(e, "recover_unknown_processes")
        return SrvResult(code=500, msg=f"UNKNOWN状态进程恢复失败: {str(e)}")

@router.get("/process/status/health")
def get_process_health_status():
    """获取进程健康状态监控信息 - 仅当前服务实例"""
    try:
        # 🔐 安全检查：确保只获取当前服务实例的健康状态
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法获取健康状态！"
            )
        
        # 获取当前服务实例的任务统计
        service_tasks = training_task_service.get_tasks_by_service_instance(current_service_id)
        stats = {}
        for task in service_tasks:
            process_status = task.process_status or ProcessStatus.UNKNOWN.value
            stats[process_status] = stats.get(process_status, 0) + 1
        
        # 计算健康评分
        total_tasks = sum(stats.values())
        if total_tasks == 0:
            health_score = 100
            health_level = "excellent"
        else:
            # 健康评分: (RUNNING + STOPPED) / total * 100
            healthy_count = stats.get(ProcessStatus.RUNNING.value, 0) + stats.get(ProcessStatus.STOPPED.value, 0)
            health_score = round((healthy_count / total_tasks) * 100, 2)
            
            if health_score >= 90:
                health_level = "excellent"
            elif health_score >= 75:
                health_level = "good"
            elif health_score >= 50:
                health_level = "warning"
            else:
                health_level = "critical"
        
        health_data = {
            "health_score": health_score,
            "health_level": health_level,
            "total_processes": total_tasks,
            "status_breakdown": stats,
            "alerts": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 生成告警信息
        unknown_count = stats.get(ProcessStatus.UNKNOWN.value, 0)
        if unknown_count > 0:
            health_data["alerts"].append(f"发现 {unknown_count} 个UNKNOWN状态进程，建议检查")
        
        orphaned_count = stats.get(ProcessStatus.ORPHANED.value, 0)
        if orphaned_count > 0:
            health_data["alerts"].append(f"发现 {orphaned_count} 个ORPHANED状态进程，已自动清理")
        
        terminated_count = stats.get(ProcessStatus.TERMINATED.value, 0)
        if terminated_count > total_tasks * 0.3 and total_tasks > 0:
            health_data["alerts"].append(f"TERMINATED进程占比过高 ({terminated_count}/{total_tasks})")
        
        return SrvResult(
            code=200,
            msg="获取进程健康状态成功",
            data=health_data
        )
        
    except Exception as e:
        logger.error(f"获取进程健康状态失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_process_health")
        return SrvResult(code=500, msg=f"获取进程健康状态失败: {str(e)}")

@router.delete("/tasks/{task_id}")
def delete_training_task(task_id: str):
    """
    删除训练任务
    
    功能：
    1. 检查任务归属权限
    2. 如果任务正在运行，先停止并杀死进程
    3. 更新任务和进程状态
    4. 从内存和数据库中删除任务记录
    """
    try:
        # 🔐 安全检查：先验证任务是否属于当前服务实例
        task_db = training_task_service.get_training_task(task_id)
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        current_service_id = unified_training_service.service_instance_id
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务删除任务被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403,
                msg=f"权限不足：任务属于其他服务实例，无法删除"
            )
        
        # 🗑️ 调用统一训练服务删除任务
        success, message = unified_training_service.delete_task(task_id)
        
        if success:
            return SrvResult(
                code=200,
                msg="任务删除成功",
                data={
                    "task_id": task_id,
                    "deleted": True,
                    "message": message
                }
            )
        else:
            return SrvResult(code=500, msg=f"删除任务失败: {message}")
            
    except Exception as e:
        logger.error(f"删除训练任务失败: {str(e)}", exc_info=True)
        handle_api_error(e, "delete_task")
        return SrvResult(code=500, msg=f"删除任务失败: {str(e)}")


@router.get("/tasks/{task_id}/loss_data")
def get_task_loss_data(
    task_id: str,
    limit: Optional[int] = Query(None, description="限制返回的记录数量，不指定则返回所有记录"),
    loss_type: Optional[str] = Query("all", description="指定loss类型: train, eval, all")
):
    """
    获取训练任务的完整loss数据
    
    功能：
    1. 从LossManager获取指定任务的所有loss历史数据
    2. 支持过滤loss类型（train_loss, eval_loss, 或全部）
    3. 支持限制返回记录数量
    4. 返回完整的loss数据供前端绘制曲线图
    
    Args:
        task_id: 训练任务ID
        limit: 可选，限制返回最近的N条记录
        loss_type: 可选，指定返回的loss类型（train, eval, all，默认all）
    
    Returns:
        包含完整loss数据的结果
    """
    try:
        logger.info(f"获取任务loss数据: {task_id}, limit={limit}, loss_type={loss_type}")
        
        # 🔐 安全检查：验证任务是否存在且属于当前服务实例
        task_db = training_task_service.get_training_task(task_id)
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        current_service_id = unified_training_service.service_instance_id
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务获取loss数据被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403,
                msg=f"权限不足：任务属于其他服务实例，无法获取loss数据"
            )
        
        # 📊 获取loss数据
        from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
        
        # 构造output_dir路径（从任务配置中获取）
        output_dir = task_db.output_dir or "/tmp/training_output"
        
        # 获取LossManager实例
        loss_manager = get_loss_manager(output_dir, task_id)
        
        # 获取所有loss历史记录
        all_records = loss_manager.get_loss_history(limit=limit)
        
        # 根据loss_type过滤数据
        filtered_records = []
        for record in all_records:
            filtered_record = {
                "step": record.get("step"),
                "timestamp": record.get("timestamp"),
                "epoch": record.get("epoch")
            }
            
            # 根据loss_type添加相应的loss数据
            if loss_type in ["all", "train"] and "train_loss" in record:
                filtered_record["train_loss"] = record["train_loss"]
            if loss_type in ["all", "eval"] and "eval_loss" in record:
                filtered_record["eval_loss"] = record["eval_loss"]
            if loss_type in ["all"] and "loss" in record:
                # 兼容性：有些记录可能只有'loss'字段
                filtered_record["loss"] = record["loss"]
            
            # 只保留有loss数据的记录
            has_loss_data = any(key in filtered_record for key in ["train_loss", "eval_loss", "loss"])
            if has_loss_data:
                filtered_records.append(filtered_record)
        
        # 构建响应数据
        response_data = {
            "task_id": task_id,
            "total_records": len(filtered_records),
            "loss_type": loss_type,
            "loss_data": filtered_records,
            "task_status": task_db.status,
            "last_updated": filtered_records[-1]["timestamp"] if filtered_records else None
        }
        
        logger.info(f"✅ 成功获取任务 {task_id} 的loss数据: {len(filtered_records)} 条记录")
        
        return SrvResult(
            code=200,
            msg="获取loss数据成功",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"获取任务loss数据失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_task_loss_data")
        return SrvResult(code=500, msg=f"获取loss数据失败: {str(e)}")


@router.get("/tasks/{task_id}/eval_results")
def get_task_eval_results(task_id: str):
    """
    获取训练任务的评估结果
    
    功能：
    1. 获取指定任务的基线评估结果（base_eval_results）和最终评估结果（final_eval_results）
    2. 包含所有数据集（train/eval/test）的评估数据
    3. 返回训练过程中的最佳loss值（best_train_loss, best_eval_loss）
    4. 返回完整的评估结果供前端展示
    
    Args:
        task_id: 训练任务ID
    
    Returns:
        包含基线和最终评估结果以及最佳loss值的数据
    """
    try:
        logger.info(f"获取任务评估结果: {task_id}")
        
        # 🔐 安全检查：验证任务是否存在且属于当前服务实例
        task_db = training_task_service.get_training_task(task_id)
        if not task_db:
            return SrvResult(code=404, msg=f"未找到任务: {task_id}")
        
        current_service_id = unified_training_service.service_instance_id
        if task_db.service_instance_id != current_service_id:
            logger.warning(f"🚫 跨服务获取评估结果被拒绝: 任务 {task_id} 属于服务 {task_db.service_instance_id}，当前服务 {current_service_id}")
            return SrvResult(
                code=403,
                msg=f"权限不足：任务属于其他服务实例，无法获取评估结果"
            )
        
        # 📊 获取数据集评估结果
        from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
        
        datasets_with_results = TrainingDatasetService.get_datasets_with_eval_results_by_task(task_id)
        
        # 📈 获取训练指标（包含best_train_loss和best_eval_loss）
        training_metrics = None
        try:
            from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
            output_dir = task_db.output_dir or "/tmp/training_output"
            loss_manager = get_loss_manager(output_dir, task_id)
            training_metrics = loss_manager.get_training_metrics()
        except Exception as e:
            logger.warning(f"获取训练指标失败: {e}")
        
        # 按数据集类型分组整理结果
        eval_results = {
            "train": [],
            "eval": [],
            "test": []
        }
        
        total_base_results = 0
        total_final_results = 0
        
        for dataset in datasets_with_results:
            split_type = dataset["split_type"]
            dataset_result = {
                "dataset_id": dataset["id"],
                "dataset_name": dataset["dataset_name"],
                "data_source_id": dataset["data_source_id"],
                "base_eval_results": dataset["base_eval_results"],
                "final_eval_results": dataset["final_eval_results"],
                "evaluation_status": dataset["evaluation_status"],
                "configured_sample_size": dataset["configured_sample_size"],
                "last_updated": dataset["update_time"]
            }
            
            # 统计评估结果数量
            if dataset["base_eval_results"]:
                total_base_results += 1
            if dataset["final_eval_results"]:
                total_final_results += 1
            
            if split_type in eval_results:
                eval_results[split_type].append(dataset_result)
        
        # 构建响应数据
        response_data = {
            "task_id": task_id,
            "task_status": task_db.status,
            "task_name": task_db.task_name,
            "train_type": task_db.train_type,
            "best_train_loss": training_metrics.get("best_train_loss") if training_metrics else None,
            "best_eval_loss": training_metrics.get("best_eval_loss") if training_metrics else None,
            "evaluation_summary": {
                "total_datasets": len(datasets_with_results),
                "datasets_with_base_results": total_base_results,
                "datasets_with_final_results": total_final_results,
                "train_datasets": len(eval_results["train"]),
                "eval_datasets": len(eval_results["eval"]),
                "test_datasets": len(eval_results["test"])
            },
            "eval_results": eval_results
        }
        
        logger.info(f"✅ 成功获取任务 {task_id} 的评估结果: {len(datasets_with_results)} 个数据集")
        
        return SrvResult(
            code=200,
            msg="获取评估结果成功",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"获取任务评估结果失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_task_eval_results")
        return SrvResult(code=500, msg=f"获取评估结果失败: {str(e)}")