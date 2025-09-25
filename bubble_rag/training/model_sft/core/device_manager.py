"""
Device management for training.

Handles GPU detection, configuration, and device allocation.
"""

import os
import logging
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device configuration and GPU setup for training."""

    @staticmethod
    def cleanup_gpu_environment():
        """
        清理GPU环境，移除可能的残留状态
        在训练开始前调用，确保干净的GPU环境
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("🧹 清理GPU环境...")

                # 清空所有GPU缓存
                torch.cuda.empty_cache()

                # 收集IPC内存
                try:
                    torch.cuda.ipc_collect()
                except:
                    pass

                # 重置CUDA上下文（如果可能）
                try:
                    torch.cuda.reset_accumulated_memory_stats()
                except:
                    pass

                logger.info(f"✅ GPU环境清理完成，可见GPU数量: {torch.cuda.device_count()}")
            else:
                logger.info("CPU模式，跳过GPU环境清理")

        except Exception as e:
            logger.error(f"❌ GPU环境清理失败，可能影响后续GPU使用。错误: {e}")

    @staticmethod
    def get_training_device(user_device=None):
        """
        获取训练设备配置（匹配原train.py逻辑）

        Args:
            user_device: 用户指定的设备配置（被忽略，因为API层会处理并设置环境变量）

        Returns:
            str: 设备字符串，'cuda' 或 'cpu'
        """
        # 完全遵循原train.py的处理逻辑：
        # 1. 用户设备参数由API层的GPUResourceManager处理并设置CUDA_VISIBLE_DEVICES
        # 2. 训练层只需要简单地检查环境变量并返回'cuda'或'cpu'

        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")

        if cuda_visible:
            # 有CUDA_VISIBLE_DEVICES，使用GPU训练
            if torch.cuda.is_available():
                visible_gpus = cuda_visible.split(",")
                gpu_count = torch.cuda.device_count()  # 这是可见的GPU数量，不是系统总数
                logger.info(f"🔧 CUDA_VISIBLE_DEVICES: {cuda_visible}")
                logger.info(f"🖥️  将使用 {len(visible_gpus)} 个指定的GPU进行训练")

                # 显示每个可见GPU的详细信息
                for i in range(gpu_count):  # 遍历torch能看到的GPU
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        original_id = visible_gpus[i] if i < len(visible_gpus) else "unknown"
                        logger.info(f"   GPU {i} (原始GPU{original_id}): {gpu_name} ({gpu_memory:.1f}GB)")
                    except Exception as e:
                        logger.info(f"   GPU {i}: 信息获取失败 - {e}")
                return "cuda"
            else:
                logger.warning("⚠️  设置了CUDA_VISIBLE_DEVICES但系统不支持CUDA，回退到CPU")
                return "cpu"
        else:
            # 没有CUDA_VISIBLE_DEVICES，可能是CPU模式或auto模式
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"🖥️  未指定CUDA_VISIBLE_DEVICES，检测到 {gpu_count} 个GPU可用")
                for i in range(gpu_count):
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                    except:
                        logger.info(f"   GPU {i}: 信息获取失败")
                return "cuda"
            else:
                logger.info("🖥️  系统不支持CUDA，使用CPU训练")
                return "cpu"
    
    @staticmethod
    def prepare_model_for_training(model, device):
        """
        为训练准备模型（匹配原train.py逻辑）

        Args:
            model: 要准备的模型
            device: 设备配置字符串，'cuda' 或 'cpu'

        Returns:
            准备好的模型
        """
        try:
            # 检查并移除任何DataParallel包装（匹配原train.py逻辑）
            if hasattr(model, '_modules'):
                for module_name, module in model._modules.items():
                    if isinstance(module, torch.nn.DataParallel):
                        # 提取原始模块，移除DataParallel包装
                        original_module = module.module
                        model._modules[module_name] = original_module
                        logger.info(f"🔓 移除模块 {module_name} 的DataParallel包装")

            logger.info(f"✅ 模型训练准备完成")
            return model

        except Exception as e:
            logger.error(f"❌ 模型准备失败: {str(e)}")
            logger.warning(f"⚠️  将继续使用原始模型配置")
            return model
    
    @staticmethod
    def get_gpu_count(device_config=None):
        """
        获取可用的GPU数量
        
        Args:
            device_config: 设备配置字符串，如 'cuda:0,1'
            
        Returns:
            int: GPU数量
        """
        if not torch.cuda.is_available():
            return 0
        
        if device_config and ',' in str(device_config):
            # 从设备配置中计算
            return len(str(device_config).split(','))
        else:
            # 使用系统检测到的GPU数量
            return torch.cuda.device_count()