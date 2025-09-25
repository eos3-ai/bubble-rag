"""
统一训练API路由
支持串行(serial)和并行(parallel)训练模式的统一接口
"""
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import uuid

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
from loguru import logger

# 权限检查助手函数
def can_access_task(task_username: str, current_user: dict) -> bool:
    """检查用户是否可以访问指定任务"""
    # 管理员可以访问所有任务
    if current_user.get('is_admin', False):
        return True
    # 普通用户只能访问自己的任务
    return task_username == current_user.get('username')

def is_admin_user(current_user: dict) -> bool:
    """检查是否为管理员用户"""
    return current_user.get('is_admin', False)

router = APIRouter()

def validate_task_access(task_id: str, username: Optional[str] = None) -> tuple:
    """
    统一的任务访问验证函数

    Args:
        task_id: 任务ID
        username: 用户名（可选，不传则默认为admin用户）

    Returns:
        tuple: (current_user, task_db)

    Raises:
        HTTPException: 当任务不存在或权限不足时
    """
    from bubble_rag.utils.user_manager import validate_user
    from fastapi import HTTPException

    # 验证用户身份
    current_user = validate_user(username)

    # 获取任务信息
    task_db = training_task_service.get_training_task(task_id)
    if not task_db:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    # 权限检查：管理员可以访问所有任务，普通用户只能访问自己的任务
    if (current_user['user_role'] != 'admin' and
        task_db.username != current_user['username']):
        raise HTTPException(
            status_code=403,
            detail=f"权限不足：无法访问其他用户的任务（任务属于用户: {task_db.username}）"
        )

    return current_user, task_db

class StartTrainingRequest(UnifiedTrainingConfig):
    """启动训练请求模型（继承统一配置）"""
    training_mode: Optional[str] = Field(default="parallel", description="训练模式: serial(串行) 或 parallel(并行)")
    base_task_id: Optional[str] = Field(default=None, description="重启源任务ID（可选，用于记录重启关系）")
    username: Optional[str] = Field(default=None, description="用户名（可选，不传则默认为admin用户）")


class StopTrainingRequest(BaseModel):
    """停止训练请求模型"""
    task_id: str = Field(description="要停止的任务ID")
    username: Optional[str] = Field(default=None, description="用户名（可选，不传则默认为admin用户）")


# 公共数据转换函数
def convert_task_to_dict(task) -> Dict[str, Any]:
    """将训练任务对象转换为字典格式"""
    task_data = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "description": task.description,
        "train_type": task.train_type,
        "model_name_or_path": task.model_name_or_path,  # 基础模型路径
        "dataset_name_or_path": task.dataset_name_or_path,
        "HF_subset": getattr(task, 'HF_subset', None),  # HuggingFace子集
        "output_dir": task.output_dir,
        "device": task.device,
        "embedding_dim": getattr(task, 'embedding_dim', None),  # 模型维度
        "status": task.status,
        "progress": task.progress,
        "username": getattr(task, 'username', None),
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,  # 更新时间
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "duration_seconds": _calculate_duration_seconds(task),
        "final_model_path": task.final_model_path,
        "error_message": task.error_message,
        "loss_data": getattr(task, 'loss_data', None),  # 训练损失数据
        # 训练参数（兼容内存对象和数据库对象）
        "training_params": _parse_training_params(task.training_params),
        # 进程管理字段
        "process_pid": getattr(task, 'process_pid', None),
        "process_status": getattr(task, 'process_status', None),
        # 服务管理字段
        "service_instance_id": getattr(task, 'service_instance_id', None),
        "service_startup_time": getattr(task, 'service_startup_time', None),
        # 重启关系字段
        "base_task_id": getattr(task, 'base_task_id', None),
        "restart_count": getattr(task, 'restart_count', 0)
    }
    return task_data

def _calculate_duration_seconds(task) -> Optional[float]:
    """计算训练时长（秒）- 通用动态计算函数"""
    if not task.started_at:
        return None

    if task.completed_at:
        # 已完成的任务：使用完成时间 - 开始时间
        return (task.completed_at - task.started_at).total_seconds()
    else:
        # 正在运行的任务：使用当前时间 - 开始时间
        return (datetime.now() - task.started_at).total_seconds()

def _calculate_duration_from_dict(task_dict: Dict[str, Any]) -> Optional[float]:
    """从字典格式的任务计算训练时长（秒）"""
    started_at_str = task_dict.get("started_at")
    if not started_at_str:
        return None

    try:
        # 将ISO格式字符串转换为datetime对象
        if isinstance(started_at_str, str):
            started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
        else:
            started_at = started_at_str

        completed_at_str = task_dict.get("completed_at")
        if completed_at_str:
            if isinstance(completed_at_str, str):
                completed_at = datetime.fromisoformat(completed_at_str.replace('Z', '+00:00'))
            else:
                completed_at = completed_at_str
            # 已完成的任务：使用完成时间 - 开始时间
            return (completed_at - started_at).total_seconds()
        else:
            # 正在运行的任务：使用当前时间 - 开始时间
            return (datetime.now() - started_at).total_seconds()
    except Exception as e:
        logger.warning(f"计算任务时长失败: {e}")
        return None

def _get_task_duration_fields(task_db) -> Dict[str, Any]:
    """获取任务训练时长字段 - 简单动态计算"""
    duration_seconds = _calculate_duration_seconds(task_db)
    duration_formatted = _format_duration(duration_seconds)

    return {
        "training_duration_seconds": duration_seconds,
        "duration_formatted": duration_formatted
    }

def _format_duration(seconds: Optional[float]) -> Optional[str]:
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

def _check_process_running(pid: int) -> bool:
    """跨服务检查指定PID的进程是否正在运行"""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # 如果没有psutil，使用系统调用
        try:
            import os
            import signal
            os.kill(pid, 0)  # 发送信号0检查进程是否存在
            return True
        except (OSError, ProcessLookupError):
            return False
    except Exception as e:
        logger.warning(f"检查进程 {pid} 状态失败: {e}")
        return False

def _parse_training_params(training_params) -> Dict[str, Any]:
    """
    解析训练参数，兼容不同的输入格式

    Args:
        training_params: 训练参数，可能是字典、JSON字符串或None

    Returns:
        Dict[str, Any]: 解析后的参数字典
    """
    if not training_params:
        return {}

    if isinstance(training_params, dict):
        # 内存中的TrainingTask对象，training_params已经是字典
        return training_params

    if isinstance(training_params, str):
        # 数据库的TrainingTaskDB对象，training_params是JSON字符串
        try:
            return json.loads(training_params)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"解析训练参数JSON失败: {e}")
            return {}

    # 其他情况，返回空字典
    logger.warning(f"未知的训练参数格式: {type(training_params)}")
    return {}

def _calculate_estimated_completion_time_from_dict(task_dict: Dict[str, Any]) -> Optional[str]:
    """从字典格式的任务计算预估完成时间"""
    started_at_str = task_dict.get("started_at")
    progress = task_dict.get("progress", 0)
    status = task_dict.get("status")

    # 对于已完成的任务，返回特殊标识
    if status in ["SUCCEEDED", "FAILED", "STOPPED"]:
        if status == "SUCCEEDED":
            return "已完成"
        elif status == "FAILED":
            return "已失败"
        else:
            return "已停止"

    # 只对有开始时间且进度在合理范围内的运行中任务计算剩余时间
    if not started_at_str or progress <= 0 or progress >= 100:
        return None

    try:
        # 将时间字符串转换为datetime对象
        if isinstance(started_at_str, datetime):
            # 如果已经是datetime对象，直接使用
            started_at = started_at_str
        elif isinstance(started_at_str, str):
            try:
                # 尝试标准ISO格式
                started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
            except ValueError:
                # 尝试其他常见格式
                try:
                    started_at = datetime.strptime(started_at_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    logger.warning(f"无法解析时间格式: {started_at_str}")
                    return None
        else:
            logger.warning(f"started_at类型不支持: {type(started_at_str)}")
            return None

        # 计算已运行时间
        elapsed = (datetime.now() - started_at).total_seconds()

        # 根据进度估算总时间和剩余时间
        estimated_total = elapsed / (progress / 100)
        estimated_remaining = max(0, estimated_total - elapsed)

        # 计算预估完成时间
        estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining)

        # 格式化为易读的时间字符串
        return estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning(f"计算预估完成时间失败: task_id={task_dict.get('task_id')}, error={e}")
        return None


@router.post("/start_training")
def start_training(request: StartTrainingRequest):
    """
    启动训练任务（统一接口）
    
    支持两种训练模式：
    - serial: 串行训练，一次只能运行一个任务
    - parallel: 并行训练，可同时运行多个任务
    """
    try:
        # 🔐 验证并获取用户信息
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(request.username)
        logger.info(f"用户 {current_user['username']} (角色: {current_user['user_role']}) 请求启动训练任务")

        # 检查服务实例ID - 确保服务隔离功能正常
        if not unified_training_service.service_instance_id:
            return SrvResult(
                code=500,
                msg="❌ 服务隔离功能异常：服务实例ID为空，无法创建训练任务！"
            )
        # 提取训练模式
        training_mode = request.training_mode or "parallel"
        
        # 分离训练模式和其他参数，排除用户身份字段和控制字段
        request_data = request.model_dump(exclude={"training_mode", "username", "base_task_id"}, exclude_none=True)

        # 分离核心任务参数和训练参数
        core_task_fields = {
            "task_name", "description", "train_type", "model_name_or_path",
            "dataset_name_or_path", "HF_subset", "output_dir", "device"
        }

        # 排除的字段：核心任务参数 + 路由控制参数 + 用户身份字段
        excluded_fields = core_task_fields | {"training_mode", "username", "base_task_id"}

        core_params = {k: v for k, v in request_data.items() if k in core_task_fields}
        training_params = {k: v for k, v in request_data.items() if k not in excluded_fields}
        
        # 合并用户提供的training_params
        if request.training_params:
            training_params.update(request.training_params)
        
        # 直接使用Pydantic的TrainingParameters进行验证
        try:
            from bubble_rag.training.model_sft.models.training_parameters import TrainingParameters
            validated_training_params = TrainingParameters(**training_params)
            training_params_dict = validated_training_params.model_dump(exclude_none=True)
            
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
        logger.info(f"核心参数: model={training_request.model_name_or_path}, dataset={training_request.dataset_name_or_path}, train_type={training_request.train_type}")
        logger.info(f"训练超参数 ({len(training_request.training_params)}个): {list(training_request.training_params.keys())}")

        # 处理重启关系和智能命名
        base_task = None
        if request.base_task_id:
            logger.info(f"检测到重启关系，源任务: {request.base_task_id}")
            # 验证源任务存在和权限
            base_task = training_task_service.get_training_task(request.base_task_id)
            if base_task:
                if can_access_task(base_task.username, current_user):
                    logger.info(f"重启关系验证通过，将记录重启关系")
                else:
                    logger.warning(f"用户 {current_user['username']} 无权访问源任务 {request.base_task_id}")
                    request.base_task_id = None  # 清除无权限的重启关系
                    base_task = None
            else:
                logger.warning(f"源任务 {request.base_task_id} 不存在，清除重启关系")
                request.base_task_id = None
                base_task = None

        # 智能生成任务名称
        if not request.task_name or not request.task_name.strip():
            if base_task:
                # 重启任务命名：源任务名 + 重启次数
                restart_count = getattr(base_task, 'restart_count', 0) + 1
                generated_name = f"{base_task.task_name}_restart_{restart_count}"
                logger.info(f"为重启任务生成名称: {generated_name}")
            else:
                # 普通任务命名：使用默认逻辑
                from datetime import datetime
                timestamp = datetime.now().strftime("%m%d_%H%M%S")
                generated_name = f"training_task_{timestamp}"
                logger.info(f"为新任务生成名称: {generated_name}")

            # 更新训练请求中的任务名称
            training_request.task_name = generated_name

        # 使用统一训练服务启动任务
        task = unified_training_service.start_training(training_request, training_mode=training_mode)

        # 设置重启关系信息
        if request.base_task_id:
            task.base_task_id = request.base_task_id
            # 更新任务描述，添加重启标记
            if task.description:
                task.description = f"{task.description} | 🔄 重启自任务 {request.base_task_id}"
            else:
                task.description = f"🔄 重启自任务 {request.base_task_id}"
            logger.info(f"已设置重启关系: {task.task_id} -> 重启自 {request.base_task_id}")

        # 🔐 任务创建后立即更新用户信息到数据库
        try:
            training_task_service.save_training_task(
                task,
                task.training_params,
                service_instance_id=unified_training_service.service_instance_id,
                username=current_user['username']
            )
            logger.info(f"任务 {task.task_id} 用户信息已保存: {current_user['username']} ({current_user['user_role']})")

            # 更新源任务的重启计数
            if request.base_task_id:
                try:
                    base_task = training_task_service.get_training_task(request.base_task_id)
                    if base_task:
                        base_task.restart_count = getattr(base_task, 'restart_count', 0) + 1
                        training_task_service.save_training_task(base_task)
                        logger.info(f"源任务 {request.base_task_id} 重启计数已更新: {base_task.restart_count}")
                except Exception as restart_count_error:
                    logger.warning(f"更新源任务重启计数失败: {str(restart_count_error)}")

        except Exception as user_save_error:
            logger.warning(f"保存任务用户信息失败: {str(user_save_error)}")
            # 不影响任务创建流程，继续执行
        
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
def stop_training(request: StopTrainingRequest):
    """停止训练任务"""
    try:
        # 🔐 验证任务访问权限
        current_user, task_db = validate_task_access(request.task_id, request.username)

        success = unified_training_service.stop_training(request.task_id)

        if success:
            return SrvResult(
                code=200,
                msg="训练任务已停止",
                data={"task_id": request.task_id, "stopped": True, "user": current_user['username']}
            )
        else:
            return SrvResult(code=500, msg=f"停止训练任务失败: {request.task_id}")
            
    except Exception as e:
        logger.error(f"停止训练任务失败: {str(e)}", exc_info=True)
        handle_api_error(e, "stop_training")
        return SrvResult(code=500, msg=f"停止训练任务失败: {str(e)}")

@router.get("/tasks/{task_id}")
def get_task_detail(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取任务详情（从数据库获取完整信息，包含持久化的进度状态）"""
    try:
        # 🔐 验证任务访问权限
        current_user, task_db = validate_task_access(task_id, username)
        
        # 🌐 跨服务获取运行进程信息（实时状态）
        # 通过数据库PID检查实际运行状态，而不是依赖本地服务实例
        is_running = False
        process_info = None
        if task_db.process_pid:
            is_running = _check_process_running(task_db.process_pid)
            if is_running:
                process_info = {
                    'pid': task_db.process_pid,
                    'status': task_db.process_status,
                    'cross_service': True
                }
        
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

        # 🔧 修复：状态与进度一致性检查
        if task_db.status in ["SUCCEEDED", "FAILED", "STOPPED"] and real_time_progress == 0:
            # 如果任务已经结束但进度为0，设置进度为100%（针对SUCCESS）或保持0（针对FAILED/STOPPED）
            if task_db.status == "SUCCEEDED":
                real_time_progress = 100.0
                logger.info(f"🔧 修复任务进度: {task_id} 状态={task_db.status}, 进度从0%修正为100%")

        # 🔧 调试日志：输出最终使用的进度值
        logger.info(f"🔍 任务详情API调试: task_id={task_id}, 数据库原始进度={task_db.progress}, 最终进度={real_time_progress}, 同步状态={sync_status}")
        
        # 计算预估剩余时间（使用实时进度）
        estimated_time = None
        estimated_completion_time = None
        if task_db.started_at and real_time_progress > 0 and real_time_progress < 100:
            elapsed = (datetime.now() - task_db.started_at).total_seconds()
            estimated_total = elapsed / (real_time_progress / 100)
            estimated_time = max(0, estimated_total - elapsed)
            # 计算预计完成时间点
            from datetime import timedelta
            estimated_completion_time = (datetime.now() + timedelta(seconds=estimated_time)).isoformat()
        
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
            "duration_seconds": _calculate_duration_seconds(task_db),
            "estimated_completion_time": estimated_completion_time,
            
            # 结果信息
            "final_model_path": task_db.final_model_path,
            "error_message": task_db.error_message,
            
            # 进程信息（如果正在运行）
            "process_info": process_info,
            
            # 训练参数（解析为对象便于前端使用）
            "training_params": json.loads(task_db.training_params) if task_db.training_params else {}
        }
        
        return SrvResult(code=200, msg="获取任务详情成功", data=task_detail)
            
    except Exception as e:
        logger.error(f"获取任务详情失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_task_detail")
        return SrvResult(code=500, msg=f"获取任务详情失败: {str(e)}")

@router.get("/tasks/{task_id}/datasets")
def get_task_datasets(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取任务的训练数据集信息 - 基于用户权限"""
    try:
        # 🔐 验证任务访问权限
        current_user, task_db = validate_task_access(task_id, username)
        
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
def get_task_training_metrics(
    task_id: str,
    limit: Optional[int] = Query(None, description="限制返回的loss记录数量，不传则获取全部"),
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取任务的训练指标和loss历史 - 仅当前服务实例"""
    try:
        # 🔐 验证任务访问权限
        current_user, task_db = validate_task_access(task_id, username)
        
        # 🌐 支持跨服务查询：已通过用户权限验证，允许查询训练日志
        # 服务隔离仅用于进程管理，不限制数据查询
        
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
def get_task_progress(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取任务实时进度（从内存获取，高频轮询优化）"""
    try:
        # 🔐 验证任务访问权限
        current_user, task_db = validate_task_access(task_id, username)

        # 🚀 直接从内存获取实时进度（避免数据库查询延迟）
        from bubble_rag.training.model_sft.services.task_manager import task_manager
        task = task_manager.get_task(task_id)

        if not task:
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
        
        # 🌐 支持跨服务查询：已通过用户权限验证，允许查询任务进度
        # 服务隔离仅用于进程管理，不限制数据查询
        
        # 🌐 跨服务获取运行进程信息
        # 通过数据库PID检查实际运行状态，而不是依赖本地服务实例
        task_db = training_task_service.get_training_task(task_id)
        is_running = False
        if task_db and task_db.process_pid:
            is_running = _check_process_running(task_db.process_pid)
        
        # 🚀 实时进度信息（从内存获取，高频轮询优化）
        # 🔧 修复进度同步问题：优先从数据库获取最新进度，确保准确性
        try:
            task_db = training_task_service.get_training_task(task_id)
            if task_db and task_db.service_instance_id == unified_training_service.service_instance_id:
                # 使用数据库中的最新进度数据，因为它更可靠
                db_progress = task_db.progress or 0
                db_status = task_db.status

                # 🔧 修复：状态与进度一致性检查
                if db_status in ["SUCCEEDED", "FAILED", "STOPPED"] and db_progress == 0:
                    if db_status == "SUCCEEDED":
                        db_progress = 100.0
                        logger.info(f"🔧 修复任务进度: {task_id} 状态={db_status}, 进度从0%修正为100%")

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


def _aggregate_local_cache_data(task_id: str, raw_loss_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将本地缓存的原始数据聚合为规范化格式
    Args:
        task_id: 任务ID
        raw_loss_data: 原始本地缓存数据（每条记录独立）
    Returns:
        规范化的聚合数据（相同step的数据合并）
    """
    from collections import defaultdict

    if not raw_loss_data:
        return []

    # 按step聚合数据
    step_data = defaultdict(dict)
    data_sources = {}
    all_metric_names = set()

    for record in raw_loss_data:
        step = record.get('step')
        if step is None:
            continue

        # 初始化step数据
        if step not in step_data:
            step_data[step] = {
                'step': step,
                'epoch': record.get('epoch'),
                'timestamp': record.get('timestamp')
            }

        # 聚合所有指标到同一条记录中
        for key, value in record.items():
            if key not in ['step', 'epoch', 'timestamp']:
                step_data[step][key] = value

                # 收集评估指标信息用于生成元数据
                if key.startswith('eval_') and '_' in key[5:]:
                    parts = key[5:].split('_', 1)
                    if len(parts) >= 2:
                        source_id, metric_name = parts[0], parts[1]
                        if not metric_name.endswith('_loss') and metric_name not in ['runtime', 'second', 'steps_per_second', 'samples_per_second']:
                            all_metric_names.add(metric_name)

    # 获取数据源映射信息
    try:
        from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
        datasets = []
        try:
            # 尝试获取数据库中的数据源信息
            with TrainingDatasetService.safe_get_session() as session:
                from bubble_rag.training.mysql_service.entity.training_dataset_models import DatasetInfo
                from sqlmodel import select
                statement = select(DatasetInfo).where(DatasetInfo.task_id == task_id, DatasetInfo.split_type == 'eval')
                datasets = session.exec(statement).all()
        except Exception as db_e:
            logger.debug(f"从数据库获取数据源信息失败，将使用默认映射: {db_e}")

        # 构建数据源映射
        for dataset in datasets:
            data_sources[dataset.data_source_id] = {
                "name": dataset.dataset_name,
                "source_id": dataset.data_source_id
            }
    except Exception as e:
        logger.debug(f"获取数据源映射失败: {e}")

    # 生成规范化结果
    result = []
    for step in sorted(step_data.keys()):
        record = step_data[step]

        # 添加evaluation_metadata（仅对包含评估指标的记录）
        eval_metrics = [k for k in record.keys() if k.startswith('eval_') and not k.endswith('_loss')]
        if eval_metrics and all_metric_names:
            try:
                from bubble_rag.training.model_sft.utils.evaluation_result import get_evaluation_result_processor
                processor = get_evaluation_result_processor()
                frontend_metadata = processor.registry.get_frontend_metadata(list(all_metric_names))

                record['evaluation_metadata'] = {
                    **frontend_metadata,
                    "data_sources": data_sources
                }
            except Exception as meta_e:
                logger.warning(f"获取评估元数据失败: {meta_e}")

        result.append(record)

    logger.debug(f"本地缓存数据聚合完成: {len(raw_loss_data)} -> {len(result)} 条记录")
    return result


@router.get("/training_logs")
def get_training_logs(
    task_id: str = Query(..., description="任务ID"),
    lines: int = Query(None, description="获取日志行数，不传则获取全部"),
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
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
        username: 用户名（可选，用于权限验证）

    Returns:
        包含日志和loss数据的综合信息
    """
    try:
        # 验证任务访问权限（包含用户权限和服务实例权限）
        current_user, task_db = validate_task_access(task_id, username)
        
        # 从任务管理器获取任务信息
        from bubble_rag.training.model_sft.services.task_manager import task_manager
        task = task_manager.get_task(task_id)

        # 获取日志数据
        recent_logs = []
        if task:
            # 任务在内存中，使用内存中的日志
            if lines is None:
                recent_logs = task.logs if task.logs else []
            else:
                recent_logs = task.logs[-lines:] if task.logs else []
        else:
            # 任务不在内存中，任务已完成或服务重启
            # 目前从数据库获取日志功能未实现，返回空日志
            pass
        
        # 🆕 获取loss数据（优先从数据库，失败时回退到本地文件）
        loss_data = []
        data_source = "unknown"

        # 优先从数据库获取
        try:
            from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
            db_loss_data = TrainingDatasetService.get_loss_data_by_task(task_id)
            logger.info(f"🔍 数据库查询结果: 任务{task_id}, 记录数={len(db_loss_data) if db_loss_data else 0}")
            if db_loss_data:
                loss_data = db_loss_data
                data_source = "database"
                logger.info(f"✅ 从数据库获取loss数据成功: {len(loss_data)} 条记录")
            else:
                logger.info("数据库中暂无loss数据，尝试本地文件")
                raise Exception("数据库中无数据")
        except Exception as db_e:
            logger.warning(f"从数据库获取loss数据失败: {db_e}")

            # 回退到本地文件
            try:
                from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager
                output_dir = task_db.output_dir or "/tmp/training_output"
                loss_manager = get_loss_manager(output_dir, task_id)
                raw_loss_data = loss_manager.get_loss_history()

                # 🔄 聚合本地缓存数据为规范化格式
                loss_data = _aggregate_local_cache_data(task_id, raw_loss_data)
                data_source = "local_file"
                logger.info(f"✅ 从本地文件获取并聚合loss数据成功: {len(raw_loss_data)} -> {len(loss_data)} 条记录")
            except Exception as file_e:
                logger.warning(f"从本地文件获取loss数据失败: {file_e}")
                data_source = "failed"
        
        return SrvResult(
            code=200,
            msg="获取训练日志成功",
            data={
                "task_id": task.task_id if task else task_id,
                "logs": recent_logs,
                "total_logs": len(task.logs) if task and task.logs else 0,
                "requested_lines": lines,
                "loss_data": loss_data,
                "total_loss_records": len(loss_data),
                "data_source": data_source  # 标识数据来源：database, local_file, failed
            }
        )
        
    except Exception as e:
        logger.error(f"获取训练日志失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_training_logs")
        return SrvResult(code=500, msg=f"获取训练日志失败: {str(e)}")

@router.get("/running_tasks")
def get_running_tasks(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取正在运行的训练任务列表 - 跨服务查询实际运行的进程"""
    try:
        # 🔐 验证并获取用户信息
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)

        # 🌐 跨服务查询：从数据库获取有PID的任务，然后验证进程是否存活
        tasks_info = []

        # 查询数据库中有process_pid的任务（可能正在运行）
        if current_user.get('is_admin', False):
            # 管理员可以看到所有正在运行的任务
            potential_running_tasks = training_task_service.get_tasks_with_process_pid()
        else:
            # 普通用户只能看到自己的任务
            potential_running_tasks = training_task_service.get_tasks_with_process_pid(username=current_user['username'])

        for task_db in potential_running_tasks:
            # 跨服务进程验证：检查PID是否真的在运行
            if task_db.process_pid:
                is_actually_running = _check_process_running(task_db.process_pid)

                if is_actually_running:
                    # 从任务管理器获取任务详细信息（如果在当前服务中）
                    from bubble_rag.training.model_sft.services.task_manager import task_manager
                    task = task_manager.get_task(task_db.task_id)

                    if task:
                        # 内存中有任务信息，使用混合数据源
                        task_info = convert_task_to_dict(task)
                        # 优先使用数据库的进度和状态（更可靠）
                        task_info["progress"] = task_db.progress or 0
                        task_info["status"] = task_db.status
                    else:
                        # 跨服务任务，只能从数据库构建信息
                        task_info = {
                            "task_id": task_db.task_id,
                            "task_name": task_db.task_name,
                            "train_type": task_db.train_type,
                            "model_name_or_path": task_db.model_name_or_path,
                            "dataset_name_or_path": task_db.dataset_name_or_path,
                            "HF_subset": task_db.HF_subset,
                            "status": task_db.status,
                            "progress": task_db.progress or 0,
                            "username": task_db.username,
                            "created_at": task_db.created_at.isoformat() if task_db.created_at else None,
                            "started_at": task_db.started_at.isoformat() if task_db.started_at else None,
                            "completed_at": task_db.completed_at.isoformat() if task_db.completed_at else None,
                            "duration_seconds": _calculate_duration_seconds(task_db),
                            "error_message": task_db.error_message
                        }

                    # 添加格式化时长
                    task_info["duration_formatted"] = _format_duration(task_info.get("duration_seconds"))

                    # 添加跨服务进程信息
                    task_info["process_info"] = {
                        "pid": task_db.process_pid,
                        "status": task_db.process_status,
                        "cross_service": task is None,  # 标识是否为跨服务任务
                        "verified_running": True
                    }

                    tasks_info.append(task_info)
                else:
                    # 进程已死但数据库未更新，记录警告
                    logger.warning(f"🚨 检测到僵尸任务：{task_db.task_id} PID={task_db.process_pid} 进程已结束但数据库未更新")

        return SrvResult(
            code=200,
            msg="获取运行中任务列表成功",
            data={
                "running_tasks": tasks_info,
                "total": len(tasks_info),
                "query_method": "cross_service_database_pid_verification"
            }
        )

    except Exception as e:
        logger.error(f"获取运行中任务列表失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_running_tasks")
        return SrvResult(code=500, msg=f"获取运行中任务列表失败: {str(e)}")

@router.get("/tasks")
def get_tasks(
    status: Optional[str] = Query(None, description="按状态过滤: PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED"),
    train_type: Optional[str] = Query(None, description="按训练类型过滤: embedding, reranker"),
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）"),
    limit: Optional[int] = Query(None, description="限制返回数量（不传则返回所有）"),
    offset: Optional[int] = Query(0, description="偏移量（默认0）")
):
    """获取任务列表（支持过滤）- 基于用户权限返回任务"""
    try:
        # 🔐 验证并获取用户信息
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)


        if current_user.get('is_admin', False) and username is None:
            # admin用户默认查看所有任务
            all_user_tasks = training_task_service.get_tasks_for_user_business(
                username=None,  # 不传username以获取所有任务
                user_info=current_user
            )
        else:
            # 其他情况：查看指定用户的任务
            target_username = username if username else current_user['username']
            all_user_tasks = training_task_service.get_tasks_for_user_business(
                username=target_username,
                user_info=current_user
            )
        
        # 应用过滤条件（注意：all_user_tasks 是字典列表）
        filtered_tasks = all_user_tasks
        if status:
            filtered_tasks = [t for t in filtered_tasks if t.get("status") == status]
        if train_type:
            filtered_tasks = [t for t in filtered_tasks if t.get("train_type") == train_type]
        
        # 返回所有过滤后的任务
        total_count = len(filtered_tasks)
        tasks = filtered_tasks
        
        # 🌐 跨服务获取运行进程信息（用于is_running状态）
        # 构建PID映射用于快速查找运行状态
        running_task_pids = {}
        try:
            # 根据用户权限获取有PID的任务
            if current_user.get('is_admin', False):
                pid_tasks = training_task_service.get_tasks_with_process_pid()
            else:
                pid_tasks = training_task_service.get_tasks_with_process_pid(username=current_user['username'])

            for pid_task in pid_tasks:
                if pid_task.process_pid and _check_process_running(pid_task.process_pid):
                    running_task_pids[pid_task.task_id] = True
        except Exception as e:
            logger.warning(f"获取跨服务运行状态失败: {e}")
            running_task_pids = {}

        # 转换为列表格式（概览信息）- 注意：tasks 已经是字典列表
        tasks_data = []
        for task in tasks:
            task_id = task.get("task_id")
            is_running = task_id in running_task_pids

            # 获取实时进度（与任务进度接口保持一致 - 使用数据库进度优先）
            # 使用数据库进度作为主要数据源，确保与进度接口一致
            real_time_progress = task.get("progress", 0.0) or 0.0

            # 确保进度在合理范围内
            real_time_progress = max(0.0, min(100.0, real_time_progress))

            # 🔧 修复：状态与进度一致性检查
            task_status = task.get("status", "UNKNOWN")
            if task_status in ["SUCCEEDED", "FAILED", "STOPPED"] and real_time_progress == 0:
                # 如果任务已经结束但进度为0，设置进度为100%（针对SUCCESS）或保持0（针对FAILED/STOPPED）
                if task_status == "SUCCEEDED":
                    real_time_progress = 100.0
                    logger.info(f"🔧 修复任务进度: {task.get('task_id')} 状态={task_status}, 进度从0%修正为100%")

            # 📋 任务概览信息（比详情轻量）
            task_overview = {
                "task_id": task_id,
                "task_name": task.get("task_name"),
                "train_type": task.get("train_type"),
                "model_name_or_path": task.get("model_name_or_path"),
                "dataset_name_or_path": task.get("dataset_name_or_path"),
                "HF_subset": task.get("HF_subset"),
                "status": task.get("status"),
                "progress": real_time_progress,  # 使用实时进度
                "is_running": is_running,
                "username": task.get("username"),  # 添加用户名字段
                "created_at": task.get("created_at"),
                "started_at": task.get("started_at"),
                "completed_at": task.get("completed_at"),
                "error_message": task.get("error_message")
            }

            # 添加训练时长字段 - 需要创建一个临时对象来计算时长
            duration_seconds = _calculate_duration_from_dict(task)
            task_overview["training_duration_seconds"] = duration_seconds
            task_overview["duration_formatted"] = _format_duration(duration_seconds)

            # 添加预估完成时间字段（使用实时进度的task_overview）
            estimated_completion_time = _calculate_estimated_completion_time_from_dict(task_overview)
            task_overview["estimated_completion_time"] = estimated_completion_time

            tasks_data.append(task_overview)
        
        # total_count 已经在上面设置好了
        
        return SrvResult(
            code=200,
            msg="获取任务列表成功",
            data={
                "tasks": tasks_data,
                "total": total_count
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
    """获取GPU资源状态 - 全局可见（避免资源冲突）"""
    try:
        # 🔐 验证用户身份（但GPU状态对所有用户全局可见）
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user()

        # 🌍 GPU是物理资源，必须全局可见以避免资源冲突
        gpu_status = gpu_resource_manager.get_resource_status()
        return SrvResult(code=200, msg="获取GPU状态成功", data=gpu_status)

    except Exception as e:
        logger.error(f"获取GPU状态失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_gpu_status")
        return SrvResult(code=500, msg=f"获取GPU状态失败: {str(e)}")

@router.post("/gpu/cleanup")
def cleanup_gpu_resources(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """手动清理所有已完成任务的GPU资源 - 仅管理员"""
    try:
        # 验证管理员权限 - GPU资源清理属于运维操作，只允许管理员执行
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)
        if not current_user.get('is_admin', False):
            return SrvResult(
                code=403,
                msg=f"权限不足：GPU资源清理需要管理员权限，当前用户: {current_user.get('username')}"
            )

        # 强制执行清理
        gpu_resource_manager._cleanup_failed_task_allocations()

        # 获取清理后的状态
        gpu_status = gpu_resource_manager.get_resource_status()

        return SrvResult(
            code=200,
            msg="GPU资源清理完成",
            data={
                "cleaned": True,
                "current_status": gpu_status
            }
        )

    except Exception as e:
        logger.error(f"清理GPU资源失败: {str(e)}", exc_info=True)
        handle_api_error(e, "cleanup_gpu_resources")
        return SrvResult(code=500, msg=f"清理GPU资源失败: {str(e)}")

@router.post("/gpu/force_release/{task_id}")
def force_release_gpu_for_task(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """强制释放指定任务的GPU资源 - 基于用户权限"""
    try:
        # 验证任务访问权限（包含用户权限和服务实例权限）
        current_user, task_db = validate_task_access(task_id, username)

        success = gpu_resource_manager.force_cleanup_task(task_id)

        if success:
            gpu_status = gpu_resource_manager.get_resource_status()
            return SrvResult(
                code=200,
                msg=f"任务 {task_id} 的GPU资源已强制释放",
                data={
                    "task_id": task_id,
                    "released": True,
                    "current_status": gpu_status
                }
            )
        else:
            return SrvResult(
                code=404,
                msg=f"任务 {task_id} 没有分配GPU资源或释放失败"
            )

    except Exception as e:
        logger.error(f"强制释放任务 {task_id} GPU资源失败: {str(e)}", exc_info=True)
        handle_api_error(e, "force_release_gpu_for_task")
        return SrvResult(code=500, msg=f"强制释放GPU资源失败: {str(e)}")

# 数据集相关接口（复用原有逻辑）
@router.get("/datasets/list")
def list_available_datasets(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """列出可用的数据集 - 需要用户认证"""
    try:
        # 🔐 验证用户身份
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)

        datasets = dataset_service.list_available_datasets()
        return SrvResult(code=200, msg="获取数据集列表成功", data=datasets)
        
    except Exception as e:
        logger.error(f"获取数据集列表失败: {str(e)}", exc_info=True)
        handle_api_error(e, "get_datasets")
        return SrvResult(code=500, msg=f"获取数据集列表失败: {str(e)}")

@router.post("/datasets/validate")
def validate_dataset(
    dataset_path: str = Query(..., description="数据集路径"),
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """验证数据集 - 需要用户认证"""
    try:
        # 🔐 验证用户身份
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)

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
    max_samples: int = Query(5, description="预览样本数量", ge=1, le=50),
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """预览数据集 - 需要用户认证"""
    try:
        # 🔐 验证用户身份
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)

        preview_data = dataset_service.preview_dataset(dataset_path, max_samples)
        return SrvResult(code=200, msg="数据集预览成功", data=preview_data)
        
    except Exception as e:
        logger.error(f"预览数据集失败: {str(e)}", exc_info=True)
        handle_api_error(e, "preview_dataset")
        return SrvResult(code=500, msg=f"预览数据集失败: {str(e)}")

@router.get("/service/health")
def get_service_health(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取服务健康状态和实例信息 - 需要管理员权限"""
    try:
        # 验证管理员权限 - 服务健康状态属于运维信息，只允许管理员查看
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)
        if not current_user.get('is_admin', False):
            return SrvResult(
                code=403,
                msg=f"权限不足：查看服务健康状态需要管理员权限，当前用户: {current_user.get('username')}"
            )

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
def get_process_status_statistics(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取进程状态统计信息 - 仅当前服务实例，需要管理员权限"""
    try:
        # 验证管理员权限 - 进程状态统计属于运维信息，只允许管理员查看
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)
        if not current_user.get('is_admin', False):
            return SrvResult(
                code=403,
                msg=f"权限不足：查看进程状态统计需要管理员权限，当前用户: {current_user.get('username')}"
            )

        # 安全检查：确保只获取当前服务实例的统计
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
def get_unknown_process_tasks(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取UNKNOWN状态的进程任务 - 仅当前服务实例，需要管理员权限"""
    try:
        # 验证管理员权限 - UNKNOWN状态任务属于故障排查信息，只允许管理员查看
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)
        if not current_user.get('is_admin', False):
            return SrvResult(
                code=403,
                msg=f"权限不足：查看UNKNOWN状态任务需要管理员权限，当前用户: {current_user.get('username')}"
            )

        # 安全检查：确保只获取当前服务实例的任务
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="服务隔离功能异常：服务实例ID为空，无法获取UNKNOWN任务！"
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
def recover_unknown_processes(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """主动触发UNKNOWN状态进程恢复 - 仅当前服务实例，需要管理员权限"""
    try:
        # 验证管理员权限 - 进程恢复操作属于运维操作，只允许管理员执行
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)
        if not current_user.get('is_admin', False):
            return SrvResult(
                code=403,
                msg=f"权限不足：执行进程恢复需要管理员权限，当前用户: {current_user.get('username')}"
            )

        # 安全检查：确保只恢复当前服务实例的进程
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
def get_process_health_status(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """获取进程健康状态监控信息 - 仅当前服务实例，需要管理员权限"""
    try:
        # 验证管理员权限 - 进程健康状态属于运维监控信息，只允许管理员查看
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)
        if not current_user.get('is_admin', False):
            return SrvResult(
                code=403,
                msg=f"权限不足：查看进程健康状态需要管理员权限，当前用户: {current_user.get('username')}"
            )

        # 安全检查：确保只获取当前服务实例的健康状态
        current_service_id = unified_training_service.service_instance_id
        if not current_service_id:
            return SrvResult(
                code=500,
                msg="服务隔离功能异常：服务实例ID为空，无法获取健康状态！"
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
def delete_training_task(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """
    删除训练任务

    功能：
    1. 检查任务访问权限
    2. 如果任务正在运行，先停止并杀死进程
    3. 更新任务和进程状态
    4. 从内存和数据库中删除任务记录
    """
    try:
        # 🔐 验证任务访问权限
        current_user, task_db = validate_task_access(task_id, username)

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


@router.delete("/tasks/service/all")
def delete_all_service_tasks(
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """
    删除当前服务实例的所有任务

    功能：
    1. 获取当前服务实例的所有任务（包括运行中的任务）
    2. 逐一停止正在运行的任务并杀死进程
    3. 删除所有任务记录（内存和数据库）
    4. 返回删除结果统计

    注意：此操作不可逆，请谨慎使用，仅限管理员使用
    """
    try:
        # 验证管理员权限 - 此操作影响所有用户，只允许管理员执行
        from bubble_rag.utils.user_manager import validate_user
        current_user = validate_user(username)
        if not current_user.get('is_admin', False):
            return SrvResult(
                code=403,
                msg=f"权限不足：删除所有任务需要管理员权限，当前用户: {current_user.get('username')}"
            )

        current_service_id = unified_training_service.service_instance_id
        logger.info(f"管理员 {current_user.get('username')} 开始删除服务实例 {current_service_id} 的所有任务")

        # 获取当前服务实例的所有任务
        all_tasks = training_task_service.get_tasks_by_service_instance(current_service_id)
        if not all_tasks:
            return SrvResult(
                code=200,
                msg="当前服务实例没有任务需要删除",
                data={
                    "service_instance_id": current_service_id,
                    "deleted_count": 0,
                    "failed_count": 0,
                    "tasks": []
                }
            )

        logger.info(f"📊 发现 {len(all_tasks)} 个任务需要删除")

        # 统计信息
        deleted_count = 0
        failed_count = 0
        deletion_results = []

        # 逐一删除任务
        for task in all_tasks:
            try:
                task_id = task.task_id
                logger.info(f"🗑️ 删除任务: {task_id} (状态: {task.status})")

                success, message = unified_training_service.delete_task(task_id)

                if success:
                    deleted_count += 1
                    deletion_results.append({
                        "task_id": task_id,
                        "status": task.status,
                        "deleted": True,
                        "message": message
                    })
                    logger.info(f"✅ 任务 {task_id} 删除成功")
                else:
                    failed_count += 1
                    deletion_results.append({
                        "task_id": task_id,
                        "status": task.status,
                        "deleted": False,
                        "message": message
                    })
                    logger.warning(f"❌ 任务 {task_id} 删除失败: {message}")

            except Exception as task_error:
                failed_count += 1
                error_msg = str(task_error)
                deletion_results.append({
                    "task_id": task.task_id if hasattr(task, 'task_id') else 'unknown',
                    "status": getattr(task, 'status', 'unknown'),
                    "deleted": False,
                    "message": f"删除时发生异常: {error_msg}"
                })
                logger.error(f"❌ 删除任务 {getattr(task, 'task_id', 'unknown')} 时发生异常: {error_msg}")

        # 构建响应结果
        total_tasks = len(all_tasks)
        success_rate = (deleted_count / total_tasks * 100) if total_tasks > 0 else 100

        result_message = f"批量删除完成: 成功 {deleted_count}/{total_tasks}, 失败 {failed_count}/{total_tasks}, 成功率 {success_rate:.1f}%"
        logger.info(f"🎯 {result_message}")

        return SrvResult(
            code=200 if failed_count == 0 else 207,  # 207 Multi-Status for partial success
            msg=result_message,
            data={
                "service_instance_id": current_service_id,
                "total_tasks": total_tasks,
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "success_rate": f"{success_rate:.1f}%",
                "tasks": deletion_results
            }
        )

    except Exception as e:
        logger.error(f"批量删除任务失败: {str(e)}", exc_info=True)
        handle_api_error(e, "delete_all_service_tasks")
        return SrvResult(code=500, msg=f"批量删除任务失败: {str(e)}")


@router.get("/tasks/{task_id}/loss_data")
def get_task_loss_data(
    task_id: str,
    limit: Optional[int] = Query(None, description="限制返回的记录数量，不指定则返回所有记录"),
    loss_type: Optional[str] = Query("all", description="指定loss类型: train, eval, all"),
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
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
        username: 用户名（可选，用于权限验证）

    Returns:
        包含完整loss数据的结果
    """
    try:
        logger.info(f"获取任务loss数据: {task_id}, limit={limit}, loss_type={loss_type}")

        # 验证任务访问权限（包含用户权限和服务实例权限）
        current_user, task_db = validate_task_access(task_id, username)
        
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
def get_task_eval_results(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """
    获取训练任务的评估结果
    
    功能：
    1. 获取指定任务的基线评估结果（base_eval_results）和最终评估结果（final_eval_results）
    2. 包含所有数据集（train/eval/test）的评估数据
    3. 返回训练过程中的最佳loss值（best_train_loss, best_eval_loss）
    4. 返回完整的评估结果供前端展示
    
    Args:
        task_id: 训练任务ID
        username: 用户名（可选，用于权限验证）

    Returns:
        包含基线和最终评估结果以及最佳loss值的数据
    """
    try:
        logger.info(f"获取任务评估结果: {task_id}")

        # 验证任务访问权限（包含用户权限和服务实例权限）
        current_user, task_db = validate_task_access(task_id, username)
        
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


def _enhance_loss_data_with_metadata(task_id: str, raw_loss_data: List[Dict]) -> List[Dict]:
    """
    将loss数据中的source_id映射回数据集名称，并增强评估元数据

    Args:
        task_id: 训练任务ID
        raw_loss_data: 原始loss数据列表

    Returns:
        应用映射和元数据增强后的loss数据列表
    """
    try:
        from bubble_rag.training.model_sft.utils.loss_manager import get_loss_manager

        # 先尝试从loss文件的元数据中获取映射（性能更好）
        mapping = {}
        try:
            # 获取task的output_dir
            from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
            task_db = training_task_service.get_training_task(task_id)
            if task_db and task_db.output_dir:
                loss_manager = get_loss_manager(task_db.output_dir, task_id)
                metadata = loss_manager.get_metadata()
                data_source_mapping = metadata.get("data_source_mapping", {})

                # 转换映射方向：{dataset_name: source_id} → {source_id: dataset_name}
                mapping = {v: k for k, v in data_source_mapping.items()}
                logger.debug(f"从loss文件获取source_id映射: {mapping}")
        except Exception as file_error:
            logger.debug(f"从loss文件获取映射失败: {file_error}")

        # 如果loss文件中没有映射，回退到数据库查询
        if not mapping:
            try:
                from bubble_rag.training.mysql_service.service.training_dataset_service import TrainingDatasetService
                mapping = TrainingDatasetService.get_source_id_to_dataset_mapping(task_id)
                logger.debug(f"从数据库获取source_id映射: {mapping}")

                # 🆕 将数据库查询结果缓存到本地文件，下次就不用再查数据库了
                if mapping and task_db and task_db.output_dir:
                    try:
                        loss_manager = get_loss_manager(task_db.output_dir, task_id)

                        # 尝试从数据库获取更多任务信息来丰富缓存
                        cache_metadata = {
                            "data_source_mapping": {v: k for k, v in mapping.items()},  # 转换回原始格式
                            "cached_from_database": True,
                            "cache_created_at": datetime.now().isoformat()
                        }

                        # 尝试从任务记录中获取更多信息
                        try:
                            if hasattr(task_db, 'config') and task_db.config:
                                import json
                                task_config = json.loads(task_db.config) if isinstance(task_db.config, str) else task_db.config

                                # 添加训练类型和配置信息
                                if 'train_type' in task_config:
                                    cache_metadata["train_type"] = task_config['train_type']
                                if 'model_config' in task_config:
                                    cache_metadata["model_config"] = task_config['model_config']
                                if 'data_config' in task_config:
                                    cache_metadata["data_config"] = task_config['data_config']

                                logger.debug(f"从任务配置中获取额外信息: {list(cache_metadata.keys())}")
                        except Exception as config_error:
                            logger.debug(f"获取任务配置信息失败: {config_error}")

                        loss_manager.save_metadata(cache_metadata)
                        logger.info(f"✅ 已缓存数据库映射到本地文件，任务: {task_id}")
                    except Exception as cache_error:
                        logger.warning(f"缓存映射到本地失败: {cache_error}")

            except Exception as db_error:
                logger.warning(f"从数据库获取映射也失败: {db_error}")

        if not mapping:
            logger.warning(f"任务 {task_id} 没有找到source_id映射，返回原始数据")
            return raw_loss_data

        # 对每条记录应用映射
        mapped_data = []
        for record in raw_loss_data:
            mapped_record = record.copy()

            # 查找需要映射的eval指标
            keys_to_update = {}
            for key, value in record.items():
                if key.startswith('eval_'):
                    # 检查是否是source_id格式: eval_1_loss, eval_2_pearson等
                    parts = key[5:].split('_')  # 去掉'eval_'前缀
                    if len(parts) >= 2 and parts[0].isdigit():
                        source_id = parts[0]
                        metric_name = '_'.join(parts[1:])  # 重新组合指标名

                        # 查找对应的数据集名称
                        if source_id in mapping:
                            dataset_name = mapping[source_id]
                            new_key = f"eval_{dataset_name}_{metric_name}"
                            keys_to_update[key] = new_key

            # 应用映射更新
            for old_key, new_key in keys_to_update.items():
                mapped_record[new_key] = mapped_record.pop(old_key)
                logger.debug(f"映射指标名: {old_key} → {new_key}")

            mapped_data.append(mapped_record)

        # 🆕 应用元数据增强
        try:
            from bubble_rag.training.model_sft.utils.evaluation_result import get_evaluation_result_processor
            processor = get_evaluation_result_processor()
            enhanced_data = processor.enhance_loss_data_with_metadata(mapped_data, mapping)
            logger.info(f"完成loss数据反向映射和元数据增强，处理了 {len(enhanced_data)} 条记录")
            return enhanced_data
        except Exception as enhance_error:
            logger.warning(f"元数据增强失败: {enhance_error}，返回映射后的数据")
            return mapped_data

    except Exception as e:
        logger.warning(f"应用反向映射和元数据增强失败: {e}，返回原始数据")
        return raw_loss_data


# ==================== 🔄 一键重启任务接口 ====================

@router.get("/tasks/{task_id}/restart_config")
def get_task_restart_config(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """
    获取任务重启配置

    重启训练的标准流程：
    1. 用户点击"重启任务" -> 调用此接口获取原任务配置
    2. 前端跳转到训练配置页面，用返回的参数预填充表单
    3. 用户调整参数后点击"开始训练" -> 调用 POST /start_training

    这样避免了提前创建任务入库，只有用户真正启动时才创建任务
    """
    try:
        # 验证任务访问权限（包含用户权限和服务实例权限）
        current_user, task = validate_task_access(task_id, username)

        # 构建重启配置
        restart_config = {
            "base_task_id": task_id,
            "original_task_name": task.task_name,
            "original_status": task.status,
            "original_created_at": task.created_at.isoformat() if task.created_at else None,

            # 核心训练配置
            "train_type": task.train_type,
            "model_name_or_path": task.model_name_or_path,
            "dataset_name_or_path": task.dataset_name_or_path,
            "HF_subset": task.HF_subset,
            "device": task.device,

            # 训练参数
            "training_params": json.loads(task.training_params) if task.training_params else {},

            # 建议的新任务名称（基于重启计数）
            "suggested_task_name": f"{task.task_name}_restart_{getattr(task, 'restart_count', 0) + 1}" if task.task_name else f"restart_{getattr(task, 'restart_count', 0) + 1}"
        }

        # 如果是失败任务，提供失败信息用于参考
        if task.status == TrainingStatus.FAILED.value and task.error_message:
            restart_config["failure_info"] = {
                "error_message": task.error_message,
                "failed_at": task.completed_at.isoformat() if task.completed_at else None
            }

        return SrvResult(
            code=200,
            msg="获取重启配置成功",
            data=restart_config
        )

    except Exception as e:
        logger.error(f"获取任务重启配置失败: {str(e)}", exc_info=True)
        return SrvResult(code=500, msg=f"获取重启配置失败: {str(e)}")


@router.get("/tasks/{task_id}/restart_history")
def get_task_restart_history(
    task_id: str,
    username: Optional[str] = Query(None, description="用户名（可选，不传则默认为admin用户）")
):
    """
    获取任务重启历史
    查找基于此任务重启的所有任务
    """
    try:
        # 验证任务访问权限（包含用户权限和服务实例权限）
        current_user, base_task = validate_task_access(task_id, username)

        # 使用数据库字段直接查询重启任务
        restart_tasks_db = training_task_service.get_restart_tasks_by_base_id(task_id)
        restart_tasks = []

        for task in restart_tasks_db:
            task_dict = convert_task_to_dict(task)
            restart_info = {
                "task_id": task_dict["task_id"],
                "task_name": task_dict["task_name"],
                "status": task_dict["status"],
                "progress": task_dict["progress"],
                "created_at": task_dict["created_at"],
                "started_at": task_dict["started_at"],
                "completed_at": task_dict["completed_at"],
                "base_task_id": getattr(task, 'base_task_id', None)
            }
            restart_tasks.append(restart_info)

        # 按创建时间排序（最新的在前）
        restart_tasks.sort(key=lambda x: x["created_at"], reverse=True)

        return SrvResult(
            code=200,
            msg="获取重启历史成功",
            data={
                "base_task_id": task_id,
                "base_task_name": base_task.task_name,
                "restart_count_from_db": getattr(base_task, 'restart_count', 0),
                "restart_count_found": len(restart_tasks),
                "restart_tasks": restart_tasks
            }
        )

    except Exception as e:
        logger.error(f"获取任务重启历史失败: {str(e)}", exc_info=True)
        return SrvResult(code=500, msg=f"获取重启历史失败: {str(e)}")



