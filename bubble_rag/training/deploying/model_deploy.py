from datetime import datetime
from typing import Optional, List
from sqlmodel import Session, select
from loguru import logger

from contextlib import contextmanager
from bubble_rag.databases.relation_database import get_engine, get_session
from bubble_rag.entity.relational.models import ModelDeploy, DockerServer
from bubble_rag.training.deploying import get_docker_service
from bubble_rag.entity.query.response_model import SrvResult


class ModelDeployService:
    """模型部署服务"""

    @staticmethod
    def deploy_model(model_deploy_id: str, docker_server_id: str = None) -> SrvResult:
        """部署模型"""
        logger.info(f"开始部署模型: model_deploy_id={model_deploy_id}, docker_server_id={docker_server_id}")

        with next(get_session()) as session:
            try:
                # 获取模型部署配置
                model_deploy = session.get(ModelDeploy, model_deploy_id)
                if not model_deploy:
                    return SrvResult(code=404, msg="模型部署配置不存在", data=None)

                # 如果没有传入docker_server_id，则从ModelDeploy对象中获取
                if docker_server_id is None:
                    docker_server_id = model_deploy.docker_server_id
                    logger.info(f"从ModelDeploy对象获取docker_server_id: {docker_server_id}")

                if not docker_server_id:
                    return SrvResult(code=400, msg="docker_server_id未设置", data=None)

                # 获取Docker服务器配置
                docker_server = session.get(DockerServer, docker_server_id)
                if not docker_server:
                    return SrvResult(code=404, msg="Docker服务器不存在", data=None)

                # 检查是否已有容器在运行
                if model_deploy.container_id:
                    docker_svc = get_docker_service(docker_server)
                    container_status = docker_svc.get_container_status(model_deploy.container_id)
                    if container_status and container_status.get('status') == 'running':
                        return SrvResult(code=400, msg="模型已在运行中", data=container_status)

                # 创建Docker服务实例
                docker_svc = get_docker_service(docker_server)

                # 测试连接
                if not docker_svc.test_connection():
                    return SrvResult(code=500, msg="无法连接到Docker服务器", data=None)

                # 启动容器（先不提交数据库更改）
                result = docker_svc.start_vllm_container(model_deploy)

                # 容器启动成功后才更新数据库
                if result and result.get('container_id'):
                    model_deploy.container_id = result['container_id']
                    model_deploy.container_status = 1  # 设置为已启动状态
                    model_deploy.update_time = datetime.now()
                    session.add(model_deploy)  # 确保对象在session中
                    session.commit()
                    logger.info(f"容器启动成功，状态更新为已启动: {model_deploy_id}")
                else:
                    # 容器启动失败，设置为未启动状态
                    model_deploy.container_status = 0
                    model_deploy.update_time = datetime.now()
                    session.add(model_deploy)
                    session.commit()
                    logger.warning(f"容器启动失败，状态更新为未启动: {model_deploy_id}")
                    raise Exception("容器启动失败，未返回容器ID")

                logger.info(f"模型部署成功: {model_deploy_id}, 容器ID: {result['container_id']}")

                return SrvResult(code=200, msg="模型部署成功", data={
                    "model_deploy": model_deploy,
                    "container": result,
                    "docker_server": docker_server
                })

            except Exception as e:
                logger.error(f"模型部署失败: {e}")
                session.rollback()
                return SrvResult(code=500, msg=f"模型部署失败: {str(e)}", data=None)

    @staticmethod
    def stop_model(model_deploy_id: str, docker_server_id: str = None) -> SrvResult:
        """停止模型"""
        with next(get_session()) as session:
            try:
                # 获取模型部署配置
                model_deploy = session.get(ModelDeploy, model_deploy_id)
                if not model_deploy:
                    return SrvResult(code=404, msg="模型部署配置不存在", data=None)

                # 如果没有传入docker_server_id，则从ModelDeploy对象中获取
                if docker_server_id is None:
                    docker_server_id = model_deploy.docker_server_id
                    logger.info(f"从ModelDeploy对象获取docker_server_id: {docker_server_id}")

                if not docker_server_id:
                    return SrvResult(code=400, msg="docker_server_id未设置", data=None)

                # 获取Docker服务器配置
                docker_server = session.get(DockerServer, docker_server_id)
                if not docker_server:
                    return SrvResult(code=200, msg="模型停止成功")
                    # return SrvResult(code=404, msg="Docker服务器不存在", data=None)

                # 创建Docker服务实例并停止容器
                docker_svc = get_docker_service(docker_server)

                if model_deploy.container_id and model_deploy.container_id.strip():
                    if docker_svc.stop_container(model_deploy.container_id):
                        # 容器停止成功，更新状态为未启动
                        model_deploy.container_status = 0
                        model_deploy.update_time = datetime.now()
                        session.add(model_deploy)
                        session.commit()
                        logger.info(
                            f"模型停止成功，状态更新为未启动: {model_deploy_id}, 容器ID: {model_deploy.container_id}")
                        return SrvResult(code=200, msg="模型停止成功", data={"container_id": model_deploy.container_id})
                    else:
                        return SrvResult(code=500, msg="停止容器失败", data=None)
                else:
                    return SrvResult(code=200, msg="模型停止成功", data={"container_id": model_deploy.container_id})
            except Exception as e:
                logger.error(f"模型停止失败: {e}")
                return SrvResult(code=500, msg=f"模型停止失败: {str(e)}", data=None)

    @staticmethod
    def remove_model(model_deploy_id: str, docker_server_id: str = None, force: bool = False) -> SrvResult:
        """删除模型容器"""
        with next(get_session()) as session:
            try:
                # 获取模型部署配置
                model_deploy = session.get(ModelDeploy, model_deploy_id)
                if not model_deploy:
                    return SrvResult(code=404, msg="模型部署配置不存在", data=None)

                # 如果没有传入docker_server_id，则从ModelDeploy对象中获取
                if docker_server_id is None:
                    docker_server_id = model_deploy.docker_server_id
                    logger.info(f"从ModelDeploy对象获取docker_server_id: {docker_server_id}")

                if not docker_server_id:
                    return SrvResult(code=400, msg="docker_server_id未设置", data=None)

                # 获取Docker服务器配置
                docker_server = session.get(DockerServer, docker_server_id)
                if not docker_server:
                    return SrvResult(code=200, msg="模型容器删除成功")
                    # return SrvResult(code=404, msg="Docker服务器不存在", data=None)

                # 创建Docker服务实例并删除容器
                docker_svc = get_docker_service(docker_server)

                container_id = model_deploy.container_id

                # 先尝试删除容器
                if container_id and container_id.strip():
                    if docker_svc.remove_container(container_id, force):
                        # 容器删除成功后才更新数据库
                        model_deploy.container_id = None
                        model_deploy.container_status = 0  # 设置为未启动状态
                        model_deploy.update_time = datetime.now()
                        session.add(model_deploy)  # 确保对象在session中
                        session.commit()

                        logger.info(f"模型容器删除成功，状态更新为未启动: {model_deploy_id}, 容器ID: {container_id}")
                        return SrvResult(code=200, msg="模型容器删除成功", data={"container_id": container_id})
                    else:
                        # 删除失败，检查容器是否真的存在
                        container_status = docker_svc.get_container_status(container_id)
                        if not container_status:
                            # 容器不存在，清空数据库记录
                            logger.warning(f"容器{container_id}不存在，清空数据库记录，状态更新为未启动")
                            model_deploy.container_id = None
                            model_deploy.container_status = 0  # 设置为未启动状态
                            model_deploy.update_time = datetime.now()
                            session.add(model_deploy)
                            session.commit()
                            return SrvResult(code=200, msg="容器不存在，已清空记录", data={"container_id": container_id})
                        else:
                            return SrvResult(code=500, msg="删除容器失败", data={"container_status": container_status})
                else:
                    return SrvResult(code=200, msg="模型停止成功", data={"container_id": model_deploy.container_id})

            except Exception as e:
                logger.error(f"删除模型容器失败: {e}")
                session.rollback()
                return SrvResult(code=500, msg=f"删除模型容器失败: {str(e)}", data=None)

    @staticmethod
    def get_model_status(model_deploy_id: str, docker_server_id: str) -> SrvResult:
        """获取模型状态"""
        with next(get_session()) as session:
            try:
                # 获取模型部署配置
                model_deploy = session.get(ModelDeploy, model_deploy_id)
                if not model_deploy:
                    return SrvResult(code=404, msg="模型部署配置不存在", data=None)

                # 获取Docker服务器配置
                docker_server = session.get(DockerServer, docker_server_id)
                if not docker_server:
                    return SrvResult(code=404, msg="Docker服务器不存在", data=None)

                result = {
                    "model_deploy": model_deploy,
                    "docker_server": docker_server,
                    "container_status": None
                }

                # 如果有容器ID，获取容器状态
                if model_deploy.container_id:
                    docker_svc = get_docker_service(docker_server)
                    container_status = docker_svc.get_container_status(model_deploy.container_id)
                    result["container_status"] = container_status

                return SrvResult(code=200, msg="获取模型状态成功", data=result)

            except Exception as e:
                logger.error(f"获取模型状态失败: {e}")
                return SrvResult(code=500, msg=f"获取模型状态失败: {str(e)}", data=None)

    @staticmethod
    def list_model_deployments(model_type: Optional[int] = None) -> SrvResult:
        """列出模型部署"""
        with next(get_session()) as session:
            try:
                # 构建查询条件
                conditions = []
                if model_type is not None:
                    conditions.append(ModelDeploy.model_type == model_type)

                # 查询模型部署列表
                statement = select(ModelDeploy).order_by(ModelDeploy.update_time.desc())
                if conditions:
                    from sqlmodel import and_
                    statement = statement.where(and_(*conditions))

                model_deploys: List[ModelDeploy] = session.exec(statement).all()

                # 如果指定了Docker服务器，获取对应状态
                result = []
                for deploy in model_deploys:
                    deploy_info = deploy
                    result.append(deploy_info)

                return SrvResult(code=200, msg="获取模型部署列表成功", data=result)

            except Exception as e:
                logger.error(f"获取模型部署列表失败: {e}")
                return SrvResult(code=500, msg=f"获取模型部署列表失败: {str(e)}", data=None)
