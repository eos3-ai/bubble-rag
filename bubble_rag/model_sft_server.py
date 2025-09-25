"""
独立的模型微调服务器
用于测试和验证模型训练接口，避免主服务器的依赖问题
"""
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入服务配置（包括GPU配置）
from bubble_rag import server_config
print(f"[CONFIG] 服务启动，GPU配置: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")

from bubble_rag.routing.unified_training_router import router as unified_training_router
from bubble_rag.routing.user_router import router as user_router

# 全局服务启动时间（用于任务创建时设置）
SERVICE_STARTUP_TIME = None

def custom_openapi():
    """自定义OpenAPI生成函数，用于捕获详细错误信息"""
    if app.openapi_schema:
        return app.openapi_schema
    
    try:
        from fastapi.openapi.utils import get_openapi
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        app.openapi_schema = openapi_schema
        return openapi_schema
    except Exception as e:
        print(f"[ERROR] OpenAPI schema generation failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        # 返回一个最小的OpenAPI schema
        return {
            "openapi": "3.0.2",
            "info": {"title": app.title, "version": app.version},
            "paths": {},
            "error": str(e)
        }

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc", 
    openapi_url="/openapi.json",
    title="Model SFT Server",
    description="模型微调服务器，提供训练相关的API接口",
    version="1.0.0"
)


# 包含统一训练路由
app.include_router(
    router=unified_training_router,
    prefix="/api/v1/unified_training",
    tags=["Unified Training"]
)

# 包含用户管理路由
app.include_router(
    router=user_router,
    prefix="/api/v1/users",
    tags=["User Management"]
)

# 设置自定义OpenAPI函数
app.openapi = custom_openapi


@app.on_event("startup")
def startup_event():
    """服务启动时初始化服务实例ID和统一训练服务"""
    print("[STARTUP] 模型训练服务器启动中...")

    # 记录服务启动时间到数据库中
    import time
    from datetime import datetime
    from bubble_rag.training.mysql_service.service.training_task_service import training_task_service

    # 立即记录服务启动时间
    global SERVICE_STARTUP_TIME
    SERVICE_STARTUP_TIME = datetime.now()
    print(f"[STARTUP] 服务启动时间: {SERVICE_STARTUP_TIME}")

    # 自动清理RPC临时文件
    try:
        print("[CLEANUP] 自动清理RPC临时文件...")
        from bubble_rag.training.model_sft.utils.temp_file_manager import temp_file_manager
        import os

        # 获取当前工作目录（bubble_rag根目录）
        current_dir = os.getcwd()

        # 清理RPC临时文件
        rpc_cleanup_result = temp_file_manager.cleanup_by_pattern(
            pattern="tmp*/_remote_module_non_scriptable.py",
            base_dir=current_dir
        )

        # 清理整个tmp目录
        import glob
        tmp_dirs = glob.glob(os.path.join(current_dir, "tmp*"))
        dirs_cleaned = 0
        for tmp_dir in tmp_dirs:
            try:
                if os.path.isdir(tmp_dir):
                    import shutil
                    shutil.rmtree(tmp_dir)
                    dirs_cleaned += 1
                    print(f"   - ✅ 清理目录: {tmp_dir}")
            except Exception as e:
                print(f"   - ❌ 清理目录失败: {tmp_dir}, 错误: {e}")

        total_cleaned = rpc_cleanup_result["files_cleaned"] + dirs_cleaned
        if total_cleaned > 0:
            print(f"[SUCCESS] RPC临时文件自动清理完成: {rpc_cleanup_result['files_cleaned']} 个文件, {dirs_cleaned} 个目录")
        else:
            print("[INFO] 未发现需要清理的RPC临时文件")

    except Exception as e:
        print(f"[WARNING] RPC临时文件自动清理失败: {e}")

    # 初始化服务实例ID
    from bubble_rag.training.model_sft.utils.service_instance import get_service_instance_id, get_service_instance_info
    
    service_instance_id = get_service_instance_id()
    if not service_instance_id:
        error_msg = "[ERROR] 服务实例ID创建失败！请检查配置文件中的 TRAINING_SERVER_PORT 设置。"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    print(f"[SUCCESS] 服务实例ID创建成功: {service_instance_id}")

    # 服务启动时间已设置为全局变量，任务创建时会自动使用
    print(f"[SUCCESS] 全局服务启动时间已设置: {SERVICE_STARTUP_TIME}")
    print("[INFO] 新创建的任务将自动使用此启动时间")

    # 显示服务实例详细信息
    instance_info = get_service_instance_info()
    print(f"[INFO] 服务实例信息:")
    print(f"   - 主机名: {instance_info.get('hostname')}")
    print(f"   - 端口: {instance_info.get('port')}")
    print(f"   - 进程ID: {instance_info.get('pid')}")
    print(f"   - 配置端口: {instance_info.get('config_port')}")
    print(f"   - 启动时间: {SERVICE_STARTUP_TIME}")
    
    # 智能孤儿任务清理：只清理真正无主的进程，避免误杀正常运行的任务
    print("[CLEANUP] 检查并清理真正的孤儿任务和进程...")
    try:
        from bubble_rag.training.mysql_service.service.training_task_service import training_task_service
        from bubble_rag.training.model_sft.enums.training_task_enums import TrainingStatus
        import psutil
        import signal
        import time
        
        # 更智能的孤儿进程检测：等待5秒让服务完全启动，避免误杀正在启动的进程
        print("   - [WAIT] 等待5秒让服务完全初始化...")
        time.sleep(5)
        
        # 查找属于当前服务实例但状态为RUNNING的所有任务
        running_tasks = training_task_service.get_running_tasks_by_service(service_instance_id)
        true_orphan_tasks = []
        
        # 智能筛选：只有进程不存在或进程创建时间早于服务启动的才是真正的孤儿进程
        # 从全局变量获取启动时间
        startup_time_dt = SERVICE_STARTUP_TIME
        startup_time = startup_time_dt.timestamp() if startup_time_dt else None

        print(f"   - [DEBUG] 全局启动时间: {startup_time_dt}")

        if startup_time is None:
            print("   - [WARNING] 全局启动时间未设置，跳过孤儿进程检测")
        else:
            current_time = time.time()

            for task in running_tasks:
                if task.process_pid:
                    try:
                        if psutil.pid_exists(task.process_pid):
                            process = psutil.Process(task.process_pid)
                            process_create_time = process.create_time()

                            # 只有进程创建时间早于服务启动时间的才是孤儿进程
                            if process_create_time < startup_time:
                                true_orphan_tasks.append(task)
                                print(f"   - 检测到孤儿进程: 任务 {task.task_id}, PID {task.process_pid}, 进程创建于服务启动前 {(startup_time - process_create_time)/60:.1f} 分钟")
                            else:
                                print(f"   - [OK] 正常运行的进程: 任务 {task.task_id}, PID {task.process_pid}, 创建时间在服务启动后 {(process_create_time - startup_time)/60:.1f} 分钟")
                        else:
                            # 进程不存在，需要清理任务状态
                            true_orphan_tasks.append(task)
                            print(f"   - [DEAD] 进程已不存在: 任务 {task.task_id}, PID {task.process_pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        true_orphan_tasks.append(task)
                        print(f"   - [ACCESS_DENIED] 无法访问进程: 任务 {task.task_id}, PID {task.process_pid}")
                else:
                    # 没有PID记录但状态为RUNNING的任务也需要清理
                    true_orphan_tasks.append(task)
                    print(f"   - 🚫 无PID记录的运行任务: 任务 {task.task_id}")
        
        orphan_tasks = true_orphan_tasks
        cleaned_count = 0
        killed_processes = 0
        
        print(f"   - 发现 {len(orphan_tasks)} 个属于服务实例 {service_instance_id} 的运行中任务")
        
        for task in orphan_tasks:
            try:
                task_killed = False
                
                # 如果任务有进程ID，使用统一进程树清理方式
                if task.process_pid:
                    try:
                        if psutil.pid_exists(task.process_pid):
                            print(f"   - [ORPHAN] 发现孤儿进程: 任务 {task.task_id}, PID {task.process_pid}")
                            
                            # 使用统一的深层次进程树清理方式
                            print(f"   - 🌳 使用进程树清理方式终止进程 {task.process_pid}")
                            from bubble_rag.training.model_sft.services.unified_training_service import training_service
                            
                            success = training_service._terminate_process_tree_by_pid(task.process_pid)
                            if success:
                                print(f"   - [TERMINATED] 进程树 {task.process_pid} 已成功终止")
                                task_killed = True
                                killed_processes += 1
                            else:
                                print(f"   - [FAILED] 进程树 {task.process_pid} 终止失败，但仍标记为已清理")
                                task_killed = True  # 即使失败也继续清理任务状态
                        else:
                            print(f"   - ℹ️  进程 {task.process_pid} 已不存在")
                            task_killed = True
                    except psutil.NoSuchProcess:
                        print(f"   - ℹ️  进程 {task.process_pid} 已不存在")
                        task_killed = True
                    except psutil.AccessDenied as e:
                        print(f"   - [PERMISSION_ERROR] 无权限操作进程 {task.process_pid}: {str(e)}")
                        task_killed = True  # 无论如何都要清理任务状态
                    except Exception as e:
                        print(f"   - [PROCESS_ERROR] 处理进程 {task.process_pid} 时出错: {str(e)}")
                        task_killed = True  # 无论如何都要清理任务状态
                else:
                    print(f"   - ℹ️  任务 {task.task_id} 没有记录进程ID")
                    task_killed = True
                
                # 无论进程是否成功杀死，都要更新任务状态为失败
                if task_killed:
                    error_msg = f"服务重启清理：进程{task.process_pid or '未知'}已终止"
                    
                    # 更新任务状态和进程信息
                    training_task_service.update_task_status(task.task_id, TrainingStatus.FAILED.value, task.progress or 0.0)
                    training_task_service.update_task_result(task.task_id, error_message=error_msg)
                    training_task_service.update_process_info(
                        task.task_id,
                        task.process_pid,
                        process_status="STOPPED"
                    )

                    # 释放孤儿任务的GPU资源
                    try:
                        from bubble_rag.training.model_sft.utils.gpu_resource_manager import gpu_resource_manager
                        gpu_resource_manager.release_gpus_for_task(task.task_id)
                        print(f"   - [GPU_RELEASED] 已释放任务 {task.task_id} 的GPU资源")
                    except Exception as gpu_error:
                        print(f"   - [GPU_ERROR] 释放GPU资源失败: {task.task_id}, 错误: {gpu_error}")

                    cleaned_count += 1
                    print(f"   - [CLEANED] 清理任务状态: {task.task_id} -> FAILED")
                
            except Exception as e:
                print(f"   - [CLEANUP_ERROR] 清理任务 {task.task_id} 时出错: {e}")
                # 即使出错也尝试更新状态
                try:
                    training_task_service.update_task_status(task.task_id, TrainingStatus.FAILED.value)
                    training_task_service.update_task_result(task.task_id, error_message=f"服务重启清理失败: {str(e)}")
                except:
                    pass
        
        print(f"[COMPLETED] 孤儿任务和进程清理完成：")
        print(f"   - 清理任务数: {cleaned_count}")
        print(f"   - 终止进程数: {killed_processes}")
        print(f"   - 服务实例: {service_instance_id}")
        
    except Exception as e:
        print(f"孤儿任务清理失败: {e}")
    
    # 初始化统一训练服务（这会触发服务隔离检查）
    try:
        from bubble_rag.training.model_sft.services.unified_training_service import unified_training_service
        print(f"[SUCCESS] 统一训练服务初始化成功，服务实例ID: {unified_training_service.service_instance_id}")
        print(f"[INFO] 默认训练模式: {unified_training_service.default_mode}")
        
        # 启动定期清理已完成进程的任务
        import threading
        import time
        
        def periodic_cleanup():
            """定期清理已完成的进程"""
            while True:
                try:
                    unified_training_service.cleanup_completed_processes()
                    time.sleep(30)  # 每30秒清理一次
                except Exception as e:
                    print(f"定期清理进程失败: {e}")
                    time.sleep(30)
        
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True, name="process-cleanup")
        cleanup_thread.start()
        print("🔄 启动定期进程清理线程")
        
    except RuntimeError as e:
        print(f"[ERROR] 统一训练服务初始化失败: {e}")
        raise
    
    print("模型训练服务器启动完成！")



@app.get("/")
def root():
    return {
        "message": "Model SFT Server is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    # 获取服务实例信息
    from bubble_rag.training.model_sft.utils.service_instance import get_service_instance_id, get_service_instance_info
    
    service_instance_id = get_service_instance_id()
    instance_info = get_service_instance_info()
    
    health_status = "healthy" if service_instance_id else "critical"
    
    return {
        "status": health_status,
        "service": "model_sft_server",
        "service_instance_id": service_instance_id,
        "service_isolation": service_instance_id is not None,
        "instance_info": instance_info,
        "gpu_allocation": "dynamic",  # 标明支持动态GPU分配
        "training_modes": {
            "unified_parallel": "并行训练模式"
        },
        "endpoints": {
            "single_process_training": "/api/v1/model_sft/start_training",
            "single_process_gpu_status": "/api/v1/model_sft/gpu_status",
            "multi_process_training": "/api/v1/mp_training/start_training",
            "mp_training_status": "/api/v1/mp_training/training_status",
            "mp_stop_training": "/api/v1/mp_training/stop_training",
            "mp_active_processes": "/api/v1/mp_training/active_processes",
            "mp_gpu_status": "/api/v1/mp_training/gpu_status",
            "unified_training": "/api/v1/unified_training/start_training",
            "unified_health": "/api/v1/unified_training/service/health",
            "tasks": "/api/v1/model_sft/db/tasks",
            "delete_task": "/api/v1/unified_training/tasks/{task_id}",
            "delete_all_service_tasks": "/api/v1/unified_training/tasks/service/all",
            "datasets": "/api/v1/model_sft/datasets/validate",
            "models": "/api/v1/model_sft/models/recommended"
        },
        "warning": "[WARNING] 服务实例ID为空，服务隔离功能异常！" if not service_instance_id else None
    }

if __name__ == "__main__":
    import uvicorn
    from bubble_rag.server_config import TRAINING_SERVER_PORT
    
    print(f"启动模型训练服务器，端口: {TRAINING_SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=TRAINING_SERVER_PORT, reload=True, workers=1)