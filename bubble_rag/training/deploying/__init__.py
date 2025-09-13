import traceback
from typing import Optional, Dict, Any, List
from loguru import logger
from bubble_rag.entity.relational.models import DockerServer, ModelDeploy

try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    logger.warning("Docker SDK未安装，请运行: pip install docker")
    DOCKER_AVAILABLE = False
    docker = None


class DockerAPIService:
    """Docker API管理服务"""

    def __init__(self, base_url: str):
        """初始化Docker API客户端"""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker SDK未安装，请运行: pip install docker")
        self.base_url = base_url
        self.client = None

    def _get_client(self):
        """获取Docker客户端"""
        if not self.client:
            try:
                self.client = docker.DockerClient(base_url=self.base_url)
                # 测试连接
                self.client.ping()

            except Exception as e:
                logger.error(f"连接Docker API失败: {e}")
                raise e
        return self.client

    def test_connection(self) -> bool:
        """测试Docker连接"""
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception as e:
            logger.error(f"Docker连接测试失败: {e}")
            return False

    def get_containers(self, all_containers: bool = False) -> List[Dict[str, Any]]:
        """获取容器列表"""
        try:
            client = self._get_client()
            containers = client.containers.list(all=all_containers)
            return [{
                'id': c.short_id,
                'name': c.name,
                'status': c.status,
                'image': c.image.tags[0] if c.image.tags else 'unknown',
                'created': c.attrs.get('Created', ''),
                'ports': c.ports
            } for c in containers]
        except Exception as e:
            logger.error(f"获取容器列表失败: {e}")
            raise e

    def start_vllm_container(self, model_deploy: ModelDeploy) -> Dict[str, Any]:
        """启动vllm容器"""
        logger.info(f"开始启动VLLM容器: 模型类型={model_deploy.model_type}, 模型路径={model_deploy.model_path}")

        try:
            client = self._get_client()

            # 构建容器配置
            container_config = self._build_container_config(model_deploy)
            logger.info(
                f"容器配置: {container_config['name']}, 端口:{container_config.get('ports')}, GPU:{container_config.get('runtime')}")

            # 创建并启动容器
            container = client.containers.run(**container_config)

            result = {
                'container_id': container.short_id,
                'name': container.name,
                'status': container.status,
                'ports': container.ports if hasattr(container, 'ports') else {},
                'image': container.image.tags[0] if container.image.tags else 'unknown'
            }

            logger.info(f"VLLM容器启动成功: {result}")
            return result

        except Exception as e:
            traceback.print_exc()
            error_msg = f"启动vllm容器失败: {str(e)}"
            logger.error(error_msg)

            # 提供更详细的错误信息
            if "port" in str(e).lower():
                error_msg += f" (可能是端口{model_deploy.svc_port}被占用)"
            elif "gpu" in str(e).lower() or "nvidia" in str(e).lower():
                error_msg += f" (可能是GPU配置{model_deploy.gpus_cfg}有误或GPU不可用)"
            elif "volume" in str(e).lower() or "mount" in str(e).lower():
                error_msg += f" (可能是模型路径{model_deploy.model_path}不存在或无权限)"

            raise Exception(error_msg)

    def stop_container(self, container_id: str) -> bool:
        """停止容器"""
        try:
            client = self._get_client()
            container = client.containers.get(container_id)
            container.stop()
            return True
        except Exception as e:
            logger.error(f"停止容器失败: {e}")
            return True

    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """删除容器"""
        try:
            client = self._get_client()
            container = client.containers.get(container_id)
            container.remove(force=force)
            return True
        except Exception as e:
            logger.error(f"删除容器失败: {e}")
            return False

    def get_container_status(self, container_id: str) -> Optional[Dict[str, Any]]:
        """获取容器状态"""
        try:
            client = self._get_client()
            container = client.containers.get(container_id)
            return {
                'id': container.short_id,
                'name': container.name,
                'status': container.status,
                'image': container.image.tags[0] if container.image.tags else 'unknown',
                'created': container.attrs.get('Created', ''),
                'ports': container.ports,
                'logs': container.logs(tail=50).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"获取容器状态失败: {e}")
            return None

    def _build_container_config(self, model_deploy: ModelDeploy) -> Dict[str, Any]:
        """构建容器配置"""
        # 验证模型路径
        if not model_deploy.model_path:
            raise ValueError("模型路径不能为空")

        # 验证模型路径是否为绝对路径
        if not model_deploy.model_path.startswith("/"):
            raise ValueError("模型路径必须是绝对路径")

        # 基础镜像
        image = "laiye-aifoundry-registry.cn-beijing.cr.aliyuncs.com/public/vllm-openai:v0.10.1.1"

        # 容器名称（添加时间戳避免重复）
        import time
        timestamp = str(int(time.time()))[-6:]  # 取时间戳后6位
        model_type_name = "embedding" if model_deploy.model_type == 0 else "rerank"
        container_name = f"vllm-{model_type_name}-{model_deploy.id[:8]}-{timestamp}"

        # 端口映射和验证
        ports = {}
        if model_deploy.svc_port:
            # 验证端口范围
            if not (1024 <= model_deploy.svc_port <= 65535):
                raise ValueError(f"端口号必须在1024-65535范围内，当前值: {model_deploy.svc_port}")
            ports = {'8000/tcp': model_deploy.svc_port}
        else:
            # 如果没有指定端口，让Docker自动分配随机端口
            ports = {'8000/tcp': ""}

        # GPU配置和验证
        device_requests = None
        runtime = None
        if model_deploy.gpus_cfg:
            gpus_cfg = model_deploy.gpus_cfg.strip()
            try:
                if gpus_cfg == "all":
                    device_requests = [
                        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                    ]
                    runtime = 'nvidia'
                elif gpus_cfg.startswith("device="):
                    # 解析设备ID，支持 device=0,1,2 格式
                    device_str = gpus_cfg.replace("device=", "").strip()
                    if device_str:
                        gpu_devices = [d.strip() for d in device_str.split(",") if d.strip()]
                        if gpu_devices:
                            device_requests = [
                                docker.types.DeviceRequest(device_ids=gpu_devices, capabilities=[['gpu']])
                            ]
                            runtime = 'nvidia'
                        else:
                            raise ValueError("GPU设备ID不能为空")
                    else:
                        raise ValueError("GPU设备ID不能为空")
                else:
                    raise ValueError(f"不支持的GPU配置格式: {gpus_cfg}，支持格式: 'all' 或 'device=0,1'")
            except Exception as e:
                logger.error(f"GPU配置解析失败: {e}")
                raise ValueError(f"GPU配置错误: {e}")

        # 卷挂载 - 将宿主机模型真实路径挂载到容器固定位置
        volumes = {}
        if model_deploy.model_path:
            # 验证模型路径格式
            model_path = model_deploy.model_path.strip()
            if not model_path:
                raise ValueError("模型路径不能为空或仅包含空格")

            # 检查路径中是否包含危险字符
            dangerous_chars = ['..', ';', '&', '|', '$', '`']
            if any(char in model_path for char in dangerous_chars):
                raise ValueError(f"模型路径包含不安全字符: {model_path}")

            # 直接挂载宿主机模型路径到容器的/app/model（固定位置）
            volumes[model_path] = {'bind': '/app/model', 'mode': 'ro'}
            logger.info(f"配置模型挂载: {model_path} -> /app/model")

        # 构建vllm启动命令
        command = self._build_vllm_command(model_deploy)

        # 环境变量（根据GPU配置设置）
        environment = {}
        if model_deploy.gpus_cfg:
            if "device=" in model_deploy.gpus_cfg:
                # 设置特定GPU可见性
                gpu_devices = model_deploy.gpus_cfg.replace("device=", "")
                environment['NVIDIA_VISIBLE_DEVICES'] = gpu_devices
            elif model_deploy.gpus_cfg == "all":
                environment['NVIDIA_VISIBLE_DEVICES'] = 'all'
        else:
            # 如果没有GPU配置，设置为none避免意外使用GPU
            environment['NVIDIA_VISIBLE_DEVICES'] = 'none'

        config = {
            'image': image,
            'name': container_name,
            'command': command,
            'ports': ports,
            'volumes': volumes,
            'environment': environment,
            'detach': True,
            'remove': False,
        }

        # 添加GPU相关配置
        if runtime:
            config['runtime'] = runtime
        if device_requests:
            config['device_requests'] = device_requests

        # 过滤None值
        config = {k: v for k, v in config.items() if v is not None}

        return config

    def _build_vllm_command(self, model_deploy: ModelDeploy) -> List[str]:
        """构建vllm启动命令"""
        # 容器内模型路径固定为/app/model，用户无法修改
        container_model_path = "/app/model"

        # 基础命令
        cmd = [
            "--model", container_model_path,
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]

        # 根据模型类型设置不同参数
        if model_deploy.model_type == 0:  # embedding模型
            cmd.extend([
                "--served-model-name", "embedding-model",
                "--max-model-len", "4096",
                # "--gpu-memory-utilization", "0.2"
            ])
        elif model_deploy.model_type == 1:  # rerank模型
            cmd.extend([
                "--served-model-name", "rerank-model",
                "--max-model-len", "4096",
                # "--gpu-memory-utilization", "0.2"
            ])

        # 解析自定义运行配置
        if model_deploy.run_cfg:
            try:
                # 假设run_cfg是用空格分隔的参数字符串
                custom_params = model_deploy.run_cfg.split()
                custom_params = [cp.strip() for cp in custom_params if cp not in cmd]
                cmd.extend(custom_params)
            except Exception as e:
                logger.warning(f"解析自定义配置失败: {e}")

        return cmd

    def get_served_model_name(self, model_deploy: ModelDeploy) -> str:
        """获取VLLM部署时使用的served-model-name"""
        if model_deploy.model_type == 0:  # embedding模型
            return "embedding-model"
        elif model_deploy.model_type == 1:  # rerank模型
            return "rerank-model"
        else:
            # 默认情况，使用模型文件名
            import os
            return os.path.basename(model_deploy.model_path or "unknown-model")

    def get_container_port(self, container_id: str, internal_port: str = "8000/tcp") -> Optional[int]:
        """获取容器端口映射的宿主机端口
        
        Args:
            container_id: 容器ID
            internal_port: 容器内部端口，格式如'8000/tcp'
            
        Returns:
            宿主机端口号，如果获取失败返回None
        """
        try:
            client = self._get_client()
            # 使用client.api.port()获取端口映射
            port_bindings = client.api.port(container_id, internal_port)
            
            if port_bindings and len(port_bindings) > 0:
                # port_bindings是一个列表，每个元素包含HostIp和HostPort
                host_port = port_bindings[0].get('HostPort')
                if host_port:
                    logger.info(f"获取到容器{container_id}的端口映射: {internal_port} -> {host_port}")
                    return int(host_port)
            
            logger.warning(f"无法获取容器{container_id}的端口映射: {internal_port}")
            return None
            
        except Exception as e:
            logger.error(f"获取容器端口映射失败: container_id={container_id}, port={internal_port}, error={e}")
            return None


def get_docker_service(docker_server: DockerServer) -> DockerAPIService:
    """获取Docker服务实例"""
    return DockerAPIService(docker_server.srv_base_url)
