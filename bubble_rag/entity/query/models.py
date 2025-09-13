from pydantic import BaseModel, ConfigDict
from pydantic import Field
from datetime import datetime
from typing import Optional

from bubble_rag.utils.snowflake_utils import gen_id


class ModelConfigReq(BaseModel):
    model_config: Optional[ConfigDict] = ConfigDict(protected_namespaces=())
    ## 配置名称
    config_name: Optional[str] = ""
    # 模型地址
    model_base_url: Optional[str] = ""
    # 模型名称
    model_name: Optional[str] = ""
    # 模型秘钥
    model_api_key: Optional[str] = ""
    # 模型类型 embedding rerank
    model_type: Optional[str] = ""
    # embedding 模型维度
    embedding_dim: Optional[int] = 0
    page_size: Optional[int] = 20
    page_num: Optional[int] = 1
    model_id: Optional[str] = ""


class DockerServerReq(BaseModel):
    id: Optional[str] = ""
    server_name: Optional[str] = ""
    srv_base_url: Optional[str] = ""
    page_size: Optional[int] = 20
    page_num: Optional[int] = 1
    server_id: Optional[str] = ""


class ModelDeployReq(BaseModel):
    """模型部署请求模型"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_deploy_id: str = Field(..., min_length=1, description="模型部署ID，不能为空")
    docker_server_id: str = Field(..., min_length=1, description="Docker服务器ID，不能为空")
    force: Optional[bool] = False


class ModelDeployListReq(BaseModel):
    """模型部署列表请求模型"""
    model_config = ConfigDict(protected_namespaces=())
    
    docker_server_id: Optional[str] = ""
    model_type: Optional[int] = None
    page_size: Optional[int] = 20
    page_num: Optional[int] = 1


class ModelDeployCreateReq(BaseModel):
    """模型部署创建请求模型"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_path: str = Field(..., min_length=1, description="模型在宿主机的真实绝对路径")
    docker_server_id: str = Field(..., min_length=1, description="Docker服务器ID")
    svc_port: Optional[int] = Field(None, ge=1024, le=65535, description="服务端口，范围1024-65535")
    gpus_cfg: Optional[str] = Field(None, description="GPU配置，格式: 'all' 或 'device=0,1'")
    run_cfg: Optional[str] = Field(None, description="自定义运行参数")
    model_type: int = Field(..., ge=0, le=1, description="模型类型：0=embedding, 1=rerank")


class ModelDeployUpdateReq(BaseModel):
    """模型部署更新请求模型"""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str = Field(..., min_length=1, description="模型部署ID")
    model_path: Optional[str] = Field(None, description="模型在宿主机的真实绝对路径")
    docker_server_id: Optional[str] = Field(None, min_length=1, description="Docker服务器ID")
    svc_port: Optional[int] = Field(None, ge=1024, le=65535, description="服务端口，范围1024-65535")
    gpus_cfg: Optional[str] = Field(None, description="GPU配置，格式: 'all' 或 'device=0,1'")
    run_cfg: Optional[str] = Field(None, description="自定义运行参数")
    model_type: Optional[int] = Field(None, ge=0, le=1, description="模型类型：0=embedding, 1=rerank")


class ModelDeployDeleteReq(BaseModel):
    """模型部署删除请求模型"""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str = Field(..., min_length=1, description="模型部署ID")
    force_remove_container: Optional[bool] = Field(False, description="是否强制删除容器")


class ModelOneClickDeployReq(BaseModel):
    """一键部署模型请求模型"""
    model_config = ConfigDict(protected_namespaces=())
    
    # 必传参数
    model_path: str = Field(..., min_length=1, description="模型在宿主机的真实绝对路径")
    model_type: int = Field(..., ge=0, le=1, description="模型类型：0=embedding, 1=rerank")
    docker_server_id: str = Field(..., min_length=1, description="Docker服务器ID")
    
    # 可选参数 - ModelDeploy相关
    svc_port: Optional[int] = Field(None, ge=30000, le=65535, description="服务端口，不传则自动获取30000以上可用端口")
    gpus_cfg: Optional[str] = Field("device=all", description="GPU配置，格式: 'all' 或 'device=0,1'",)
    run_cfg: Optional[str] = Field(None, description="自定义运行参数")
    
    # 可选参数 - ModelConfig相关
    config_name: Optional[str] = Field(None, description="配置名称，不传则使用模型文件名")
    model_name: Optional[str] = Field(None, description="模型名称，不传则使用模型文件名")
    embedding_dim: Optional[int] = Field(1024, ge=1, description="embedding模型维度，默认为1024")

