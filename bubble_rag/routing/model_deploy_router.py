import socket
import os
import traceback
from datetime import datetime, timedelta
from fastapi import APIRouter
from loguru import logger
from sqlalchemy import func, delete as sqldelete
from sqlmodel import select, and_
from typing import Optional

from bubble_rag.databases.relation_database import SessionDep, update_model_from_dict
from bubble_rag.entity.relational.models import ModelDeploy, ModelConfig, DockerServer
from bubble_rag.entity.query.models import (
    ModelDeployReq, ModelDeployListReq, ModelDeployCreateReq,
    ModelDeployUpdateReq, ModelDeployDeleteReq, ModelOneClickDeployReq
)
from bubble_rag.entity.query.response_model import SrvResult, PageResult
from bubble_rag.training.deploying.model_deploy import ModelDeployService
from bubble_rag.utils.snowflake_utils import gen_id
from bubble_rag.server_config import CURR_SERVER_IP

router = APIRouter()


@router.post("/add_model_deploy")
async def add_model_deploy(create_req: ModelDeployCreateReq, session: SessionDep):
    """添加模型部署配置"""
    try:
        # 验证模型路径
        if not create_req.model_path.startswith("/"):
            return SrvResult(code=400, msg="模型路径必须是绝对路径", data=None)

        # 创建ModelDeploy对象
        model_deploy = ModelDeploy(
            id=gen_id(),
            model_path=create_req.model_path,
            svc_port=create_req.svc_port,
            gpus_cfg=create_req.gpus_cfg,
            run_cfg=create_req.run_cfg,
            model_type=create_req.model_type,
            docker_server_id=create_req.docker_server_id,
            container_status=0,  # 新创建的配置默认为未启动状态
            create_time=datetime.now(),
            update_time=datetime.now()
        )

        session.add(model_deploy)
        session.commit()
        session.refresh(model_deploy)

        logger.info(f"添加模型部署配置成功: {model_deploy.id}")
        return SrvResult(code=200, msg="success", data=model_deploy)

    except Exception as e:
        logger.error(f"添加模型部署配置失败: {e}")
        session.rollback()
        return SrvResult(code=500, msg=f"添加失败: {str(e)}", data=None)


@router.post("/edit_model_deploy")
async def edit_model_deploy(update_req: ModelDeployUpdateReq, session: SessionDep):
    """编辑模型部署配置"""
    try:
        # 获取现有记录
        db_deploy = session.get(ModelDeploy, update_req.id)
        if not db_deploy:
            return SrvResult(code=404, msg="模型部署配置不存在", data=None)

        # 检查是否有容器在运行
        if db_deploy.container_id:
            return SrvResult(code=400, msg="模型正在运行中，无法编辑配置，请先停止模型", data=None)

        # 准备更新数据
        update_data = update_req.dict(exclude={'id'})
        # 过滤空值
        update_data = {k: v for k, v in update_data.items() if v is not None and v != ""}

        # 验证模型路径
        if 'model_path' in update_data and not update_data['model_path'].startswith("/"):
            return SrvResult(code=400, msg="模型路径必须是绝对路径", data=None)

        # 添加更新时间
        update_data['update_time'] = datetime.now()

        # 更新记录
        update_model_from_dict(db_deploy, update_data)
        session.commit()
        session.refresh(db_deploy)

        logger.info(f"编辑模型部署配置成功: {db_deploy.id}")
        return SrvResult(code=200, msg="success", data=db_deploy)

    except Exception as e:
        logger.error(f"编辑模型部署配置失败: {e}")
        session.rollback()
        return SrvResult(code=500, msg=f"编辑失败: {str(e)}", data=None)


@router.post("/delete_model_deploy")
async def delete_model_deploy(delete_req: ModelDeployDeleteReq, session: SessionDep):
    """删除模型部署配置"""
    try:
        # 获取记录
        db_deploy = session.get(ModelDeploy, delete_req.id)
        if not db_deploy:
            return SrvResult(code=404, msg="模型部署配置不存在", data=None)

        # 检查是否有容器在运行
        if db_deploy.container_id:
            return SrvResult(code=400, msg="模型正在运行中，请先停止模型", data=None)
            # if delete_req.force_remove_container:
            #     # 强制删除容器和配置
            #     logger.warning(f"强制删除模型部署配置和容器: {delete_req.id}")
            # else:
            #     return SrvResult(code=400, msg="模型正在运行中，请先停止模型", data=None)

        # 删除记录
        session.exec(sqldelete(ModelDeploy).where(ModelDeploy.id == delete_req.id))
        session.commit()

        logger.info(f"删除模型部署配置成功: {delete_req.id}")
        return SrvResult(code=200, msg="success", data=None)

    except Exception as e:
        logger.error(f"删除模型部署配置失败: {e}")
        session.rollback()
        return SrvResult(code=500, msg=f"删除失败: {str(e)}", data=None)


@router.post("/get_model_deploy")
async def get_model_deploy(get_req: dict, session: SessionDep):
    """获取单个模型部署配置"""
    try:
        model_id = get_req.get("id")
        if not model_id:
            return SrvResult(code=400, msg="缺少模型ID参数", data=None)

        db_deploy = session.get(ModelDeploy, model_id)
        if not db_deploy:
            return SrvResult(code=404, msg="模型部署配置不存在", data=None)

        return SrvResult(code=200, msg="success", data=db_deploy)

    except Exception as e:
        logger.error(f"获取模型部署配置失败: {e}")
        return SrvResult(code=500, msg=f"获取失败: {str(e)}", data=None)


@router.post("/list_model_deploy")
async def list_model_deploy(list_req: ModelDeployListReq, session: SessionDep):
    """列出模型部署配置（分页）"""
    try:
        # 构建查询条件
        conditions = []
        if list_req.model_type is not None and list_req.model_type >= 0:
            conditions.append(ModelDeploy.model_type == list_req.model_type)

        # 构建基础查询
        base_query = select(ModelDeploy).order_by(ModelDeploy.update_time.desc())
        if conditions:
            base_query = base_query.where(and_(*conditions))

        # 获取总数
        count_query = select(func.count(ModelDeploy.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_count = session.exec(count_query).first()

        # 分页查询
        offset = (list_req.page_num - 1) * list_req.page_size
        paginated_query = base_query.offset(offset).limit(list_req.page_size)
        deployments = session.exec(paginated_query).all()
        deployments2 = []
        for deploy in deployments:
            deploy_dict = deploy.model_dump()
            deploy_dict["docker_server"] = session.get(DockerServer, deploy.docker_server_id)
            deployments2.append(deploy_dict)
        total_pages = (total_count + list_req.page_size - 1) // list_req.page_size if total_count > 0 else 1

        # 构建分页结果
        page_result = PageResult(
            total=total_count,
            page=list_req.page_num,
            page_size=list_req.page_size,
            total_pages=total_pages,
            items=deployments2
        )

        return SrvResult(code=200, msg="success", data=page_result)

    except Exception as e:
        traceback.print_exc()
        logger.error(f"列出模型部署配置失败: {e}")
        return SrvResult(code=500, msg=f"查询失败: {str(e)}", data=None)


# ========== 容器管理操作 ==========

@router.post("/one_click_deploy")
async def one_click_deploy(deploy_req: ModelOneClickDeployReq, session: SessionDep):
    """一键部署模型：自动创建部署配置、部署模型、创建模型记录"""
    try:
        logger.info(f"开始一键部署模型: {deploy_req.model_path}")

        # 1. 参数处理和验证
        # 验证model_path
        if not os.path.isabs(deploy_req.model_path):
            return SrvResult(code=400, msg="model_path必须是绝对路径", data=None)

        if not os.path.exists(deploy_req.model_path):
            logger.warning(f"模型路径不存在: {deploy_req.model_path}")
            # 注意：对于Docker容器内的路径，宿主机可能看不到，这里只是警告而不阻断

        # 验证Docker服务器是否存在
        docker_server = session.get(DockerServer, deploy_req.docker_server_id)
        if not docker_server:
            return SrvResult(code=404, msg=f"Docker服务器不存在: {deploy_req.docker_server_id}", data=None)

        # 获取served-model-name作为OpenAI调用时的model名称
        from bubble_rag.training.deploying import get_docker_service
        docker_svc = get_docker_service(docker_server)
        served_model_name = docker_svc.get_served_model_name(ModelDeploy(
            model_type=deploy_req.model_type,
            model_path=deploy_req.model_path
        ))
        
        # model_name用于OpenAI调用，应该使用served-model-name
        model_name = deploy_req.model_name or served_model_name
        # config_name用于配置标识，可以使用文件夹名称或用户指定名称
        config_name = deploy_req.config_name or os.path.basename(deploy_req.model_path)

        # 验证生成的名称不为空
        if not model_name.strip():
            return SrvResult(code=400, msg="无法获取模型名称", data=None)
        if not config_name.strip():
            return SrvResult(code=400, msg="无法从model_path提取配置名称，请手动指定config_name", data=None)

        # 2. 创建ModelDeploy对象 (暂不提交) - 不设置端口，让Docker自动分配
        model_deploy = ModelDeploy(
            id=gen_id(),
            model_path=deploy_req.model_path,
            svc_port=deploy_req.svc_port,  # 如果用户指定了端口则使用，否则为None让Docker自动分配
            gpus_cfg=deploy_req.gpus_cfg,
            run_cfg=deploy_req.run_cfg,
            model_type=deploy_req.model_type,
            container_status=0,
            docker_server_id=deploy_req.docker_server_id,
            create_time=datetime.now(),
            update_time=datetime.now()
        )

        session.add(model_deploy)
        session.commit()
        logger.info(f"创建模型部署配置: {model_deploy.id}")

        # 3. 部署模型
        deploy_result = ModelDeployService.deploy_model(
            model_deploy_id=model_deploy.id,
            docker_server_id=deploy_req.docker_server_id
        )

        if deploy_result.code != 200:
            logger.error(f"模型部署失败: {deploy_result.msg}")
            session.rollback()  # 回滚ModelDeploy创建
            return deploy_result

        logger.info(f"模型部署成功: {model_deploy.id}")

        # 3.1 获取容器实际分配的端口并更新数据库
        actual_port = None
        
        # 使用DockerAPIService的get_container_port方法获取端口
        if deploy_result.data and isinstance(deploy_result.data, dict):
            container_info = deploy_result.data.get('container')
            model_deploy_data = deploy_result.data.get('model_deploy')
            docker_server_data = deploy_result.data.get('docker_server')
            
            if container_info and container_info.get('container_id') and docker_server_data:
                try:
                    from bubble_rag.training.deploying import get_docker_service
                    docker_svc = get_docker_service(docker_server)
                    actual_port = docker_svc.get_container_port(container_info['container_id'])
                    logger.info(f"从Docker API获取到实际端口: {actual_port}")
                except Exception as e:
                    logger.warning(f"使用Docker API获取端口失败: {e}")
        
        if actual_port:
            # 更新ModelDeploy的端口信息
            model_deploy.svc_port = actual_port
            model_deploy.update_time = datetime.now()
            session.add(model_deploy)
            session.commit()
            logger.info(f"更新模型部署端口: {model_deploy.id} -> {actual_port}")
        else:
            # 如果无法获取端口，使用默认逻辑
            actual_port = model_deploy.svc_port or 8000
            logger.warning(f"无法获取容器实际端口，使用默认端口: {actual_port}")

        # 4. 创建ModelConfig记录
        # 检查配置名称是否已存在
        existing_config = session.exec(
            select(ModelConfig).where(ModelConfig.config_name == config_name)
        ).first()

        if existing_config:
            # 如果配置名称重复，添加时间戳后缀
            import time
            timestamp = int(time.time())
            config_name = f"{config_name}_{timestamp}"
            logger.warning(f"配置名称重复，自动重命名为: {config_name}")

        model_base_url = f"http://{CURR_SERVER_IP}:{actual_port}"
        model_type_str = "embedding" if deploy_req.model_type == 0 else "rerank"

        model_config = ModelConfig(
            id=gen_id(),
            config_name=config_name,
            model_base_url=model_base_url,
            model_name=model_name,
            model_type=model_type_str,
            embedding_dim=deploy_req.embedding_dim,  # 使用请求参数中的维度
            create_time=datetime.now(),
            update_time=datetime.now()
        )

        session.add(model_config)

        # 5. 统一提交所有变更
        session.commit()
        session.refresh(model_deploy)
        session.refresh(model_config)
        logger.info(f"创建模型配置记录成功: {model_config.id}")

        return SrvResult(code=200, msg="一键部署模型成功", data={
            "model_deploy": model_deploy,
            "model_config": model_config,
            "deploy_result": deploy_result.data,
            "assigned_port": actual_port
        })

    except Exception as e:
        logger.error(f"一键部署模型失败: {e}")
        session.rollback()
        return SrvResult(code=500, msg=f"一键部署失败: {str(e)}", data=None)


@router.post("/deploy_model")
async def deploy_model(deploy_req: ModelDeployReq):
    """部署模型"""
    try:
        # 如果请求中包含docker_server_id则使用，否则将从ModelDeploy对象中获取
        result = ModelDeployService.deploy_model(
            model_deploy_id=deploy_req.model_deploy_id,
            docker_server_id=deploy_req.docker_server_id if hasattr(deploy_req,
                                                                    'docker_server_id') and deploy_req.docker_server_id else None
        )
        return result
    except Exception as e:
        logger.error(f"部署模型API调用失败: {e}")
        return SrvResult(code=500, msg=f"部署模型失败: {str(e)}", data=None)


@router.post("/stop_model")
async def stop_model(deploy_req: ModelDeployReq):
    """停止并删除模型容器"""
    try:
        # 先停止容器
        stop_result = ModelDeployService.stop_model(
            model_deploy_id=deploy_req.model_deploy_id,
            docker_server_id=deploy_req.docker_server_id if hasattr(deploy_req,
                                                                    'docker_server_id') and deploy_req.docker_server_id else None
        )
        # 如果停止成功，再删除容器
        if stop_result.code == 200:
            logger.info(f"容器停止成功，开始删除容器: {deploy_req.model_deploy_id}")
            remove_result = ModelDeployService.remove_model(
                model_deploy_id=deploy_req.model_deploy_id,
                docker_server_id=deploy_req.docker_server_id if hasattr(deploy_req,
                                                                        'docker_server_id') and deploy_req.docker_server_id else None,
                force=True  # 使用force=True确保能删除已停止的容器
            )

            if remove_result.code == 200:
                logger.info(f"容器删除成功，开始删除数据库记录: {deploy_req.model_deploy_id}")
                
                # 删除成功后，同时删除数据库中的ModelDeploy记录和相关的ModelConfig记录
                try:
                    from bubble_rag.databases.relation_database import get_session
                    from bubble_rag.entity.relational.models import ModelDeploy, ModelConfig
                    from sqlmodel import select
                    
                    with next(get_session()) as session:
                        # 获取要删除的ModelDeploy记录
                        model_deploy = session.get(ModelDeploy, deploy_req.model_deploy_id)
                        if not model_deploy:
                            logger.warning(f"ModelDeploy记录不存在，跳过删除: {deploy_req.model_deploy_id}")
                        else:
                            # 删除ModelDeploy记录
                            session.delete(model_deploy)
                            session.commit()
                            logger.info(f"ModelDeploy数据库记录删除成功: {deploy_req.model_deploy_id}")
                except Exception as db_e:
                    traceback.print_exc()
                    logger.error(f"删除数据库记录失败: {db_e}")
                    # 虽然数据库删除失败，但容器已经删除成功，返回警告信息
                    container_id = None
                    if stop_result.data and isinstance(stop_result.data, dict):
                        container_id = stop_result.data.get("container_id")
                    
                    return SrvResult(code=206, msg="容器删除成功，但数据库记录删除失败", data={
                        "container_id": container_id,
                        "operation": "container_removed_db_failed",
                        "warning": str(db_e)
                    })
                
                # 安全获取container_id
                container_id = None
                if stop_result.data and isinstance(stop_result.data, dict):
                    container_id = stop_result.data.get("container_id")

                return SrvResult(code=200, msg="模型停止、删除容器和数据库记录成功", data={
                    "container_id": container_id,
                    "operation": "stop_remove_and_delete_records"
                })
            else:
                # 删除失败，但停止成功了
                error_msg = remove_result.msg or "删除操作失败，原因未知"
                logger.warning(f"容器停止成功但删除失败: {error_msg}")

                # 安全获取container_id
                container_id = None
                if stop_result.data and isinstance(stop_result.data, dict):
                    container_id = stop_result.data.get("container_id")

                return SrvResult(code=206, msg=f"模型停止成功，但删除失败: {error_msg}", data={
                    "container_id": container_id,
                    "operation": "stop_only",
                    "warning": error_msg
                })
        else:
            # 停止失败，不尝试删除
            stop_error_msg = stop_result.msg or "停止操作失败，原因未知"
            logger.error(f"容器停止失败: {stop_error_msg}")
            return stop_result

    except Exception as e:
        traceback.print_exc()
        logger.error(f"停止并删除模型API调用失败: {e}")
        return SrvResult(code=500, msg=f"停止并删除模型失败: {str(e)}", data=None)


@router.post("/remove_model")
async def remove_model(deploy_req: ModelDeployReq):
    """删除模型容器"""
    try:
        result = ModelDeployService.remove_model(
            model_deploy_id=deploy_req.model_deploy_id,
            docker_server_id=deploy_req.docker_server_id if hasattr(deploy_req,
                                                                    'docker_server_id') and deploy_req.docker_server_id else None,
            force=deploy_req.force if hasattr(deploy_req, 'force') else False
        )
        return result
    except Exception as e:
        logger.error(f"删除模型容器API调用失败: {e}")
        return SrvResult(code=500, msg=f"删除模型容器失败: {str(e)}", data=None)


@router.post("/get_model_status")
async def get_model_status(deploy_req: ModelDeployReq):
    """获取模型状态"""
    try:
        result = ModelDeployService.get_model_status(
            model_deploy_id=deploy_req.model_deploy_id,
            docker_server_id=deploy_req.docker_server_id
        )
        return result
    except Exception as e:
        logger.error(f"获取模型状态API调用失败: {e}")
        return SrvResult(code=500, msg=f"获取模型状态失败: {str(e)}", data=None)


@router.post("/list_model_deployments")
async def list_model_deployments(list_req: ModelDeployListReq):
    """列出模型部署"""
    try:
        result = ModelDeployService.list_model_deployments(
            model_type=list_req.model_type if list_req.model_type else None,
        )
        return result
    except Exception as e:
        logger.error(f"列出模型部署API调用失败: {e}")
        return SrvResult(code=500, msg=f"获取模型部署列表失败: {str(e)}", data=None)
