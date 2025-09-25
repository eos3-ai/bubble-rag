"""
Tensorboard logging functionality for BaseTrainer.
"""

import logging

logger = logging.getLogger(__name__)


def show_tensorboard_info(trainer_instance, training_args):
    """Show Tensorboard logging information (matching original train.py)."""
    try:
        from ..enums.training_parameter_enums import ReportTo

        # Check if Tensorboard is enabled
        report_to = trainer_instance.raw_config.get('report_to', 'none')
        if (report_to == ReportTo.TENSORBOARD or report_to == ReportTo.TENSORBOARD.value or
            report_to == 'tensorboard'):
            logger.info(f"🔥 Tensorboard 已启用!")

            # Check if user specified logging_dir
            user_logging_dir = trainer_instance.raw_config.get("user_logging_dir")
            if user_logging_dir:
                logger.info(f"📊 Tensorboard 日志目录（用户指定）: {user_logging_dir}")
                logger.info(f"🌐 启动 Tensorboard 命令: tensorboard --logdir=\"{user_logging_dir}\" --host=127.0.0.1 --port=6006")
            else:
                # HuggingFace auto-generated logging_dir
                if hasattr(training_args, 'logging_dir') and training_args.logging_dir:
                    actual_log_dir = str(training_args.logging_dir).replace('\\', '/')
                    logger.info(f"📊 Tensorboard 日志目录（HuggingFace 自动生成）: {actual_log_dir}")
                    logger.info(f"🌐 启动 Tensorboard 命令: tensorboard --logdir=\"{actual_log_dir}\" --host=127.0.0.1 --port=6006")
                else:
                    output_dir = training_args.output_dir if hasattr(training_args, 'output_dir') else './output'
                    logger.info(f"📊 Tensorboard 日志目录: HuggingFace 将自动在 {output_dir} 下创建 runs/<时间戳> 目录")
                    logger.info(f"🌐 启动 Tensorboard 命令: tensorboard --logdir=\"{output_dir}/runs\" --host=127.0.0.1 --port=6006")
                    logger.info(f"💡 提示: 训练开始后查看 {output_dir}/runs 目录下的实际日志文件夹")

            logger.info(f"🔗 访问地址: http://127.0.0.1:6006")

    except Exception as e:
        logger.debug(f"显示Tensorboard信息时出错: {e}")