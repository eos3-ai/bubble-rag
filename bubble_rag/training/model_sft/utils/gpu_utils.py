"""
GPU设备管理工具函数
统一单进程和多进程训练的GPU设备处理逻辑
"""
import os
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def parse_device_to_cuda_visible_devices(device: str) -> Tuple[str, bool]:
    """
    将设备字符串转换为CUDA_VISIBLE_DEVICES格式
    
    Args:
        device: 设备字符串，支持:
            - "cpu": CPU模式
            - "auto": 自动模式（不设置限制）
            - "cuda:0": 单GPU
            - "cuda:0,cuda:1": 多GPU
    
    Returns:
        Tuple[str, bool]: (CUDA_VISIBLE_DEVICES值, 是否需要设置环境变量)
    """
    if device == "cpu":
        return "", True  # 空字符串表示CPU模式
    elif device == "auto":
        return "", False  # 不设置环境变量，让系统自动选择
    elif device and device.startswith("cuda:"):
        try:
            gpu_ids = []
            for device_part in device.split(","):
                device_part = device_part.strip()
                if device_part.startswith("cuda:"):
                    gpu_id = device_part.split(":")[1]
                    gpu_ids.append(gpu_id)
            
            if gpu_ids:
                cuda_visible_devices = ",".join(gpu_ids)
                return cuda_visible_devices, True
                
        except (IndexError, ValueError) as e:
            logger.error(f"解析设备配置失败: {device}, 错误: {e}")
            return "", False
    
    logger.warning(f"未识别的设备配置: {device}")
    return "", False


def setup_cuda_environment_for_subprocess(env: Dict[str, str], device: str) -> Dict[str, str]:
    """
    为子进程设置CUDA环境变量
    
    Args:
        env: 环境变量字典
        device: 设备配置字符串
        
    Returns:
        Dict[str, str]: 更新后的环境变量字典
    """
    cuda_visible_devices, should_set = parse_device_to_cuda_visible_devices(device)
    
    if should_set:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        if device == "cpu":
            logger.info("🔧 子进程设置为CPU模式")
        else:
            logger.info(f"🔧 子进程设置CUDA_VISIBLE_DEVICES: {cuda_visible_devices} (从 {device})")
    else:
        logger.info("🔧 子进程使用auto模式，系统将自动选择可用GPU")
    
    return env


def setup_training_params_env(env: Dict[str, str], training_params: Dict) -> Dict[str, str]:
    """
    设置训练参数环境变量
    
    Args:
        env: 环境变量字典
        training_params: 训练参数字典
        
    Returns:
        Dict[str, str]: 更新后的环境变量字典
    """
    if not training_params:
        return env
        
    # 统一的训练参数列表
    param_names = [
        "num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size",
        "learning_rate", "warmup_ratio", "lr_scheduler_type", "bf16", "fp16",
        "eval_strategy", "eval_steps", "save_strategy", "save_steps", "save_total_limit",
        "logging_steps", "logging_strategy", "logging_dir", "gradient_accumulation_steps", "max_steps",
        "batch_sampler", "dataloader_num_workers", "weight_decay", "adam_beta1", "adam_beta2",
        "adam_epsilon", "max_grad_norm", "seed", "dataloader_drop_last", "eval_accumulation_steps",
        "load_best_model_at_end", "metric_for_best_model", "greater_is_better", "ignore_data_skip",
        "resume_from_checkpoint", "push_to_hub", "hub_model_id", "hub_strategy", "hub_token",
        "prediction_loss_only", "remove_unused_columns", "label_names", "local_rank", "deepspeed",
        "optim", "group_by_length", "length_column_name", "report_to", "ddpbackend", 
        "sample_size"
    ]
    
    for param in param_names:
        if param in training_params:
            env_key = param.upper()
            env_value = str(training_params[param])
            env[env_key] = env_value
            logger.debug(f"设置环境变量: {env_key} = {env_value}")
    
    return env


def log_gpu_allocation(device_request: str, allocated_device: str, task_id: str):
    """
    记录GPU分配信息的统一日志格式
    
    Args:
        device_request: 原始设备请求
        allocated_device: 分配到的设备
        task_id: 任务ID
    """
    logger.info(f"🔧 任务 {task_id} GPU分配: {device_request} -> {allocated_device}")
    
    if allocated_device != "cpu" and allocated_device != "auto":
        cuda_visible_devices, _ = parse_device_to_cuda_visible_devices(allocated_device)
        logger.info(f"🔧 任务 {task_id} CUDA_VISIBLE_DEVICES将设置为: {cuda_visible_devices}")