from bubble_rag.databases.relation_database import get_session
from pydantic import ConfigDict
from sqlmodel import SQLModel, Field, DateTime, VARCHAR, TEXT, Integer, Column, Relationship
from datetime import datetime
from typing import Optional

from bubble_rag.utils.snowflake_utils import gen_id


class ModelConfig(SQLModel, table=True):
    __tablename__ = "model_config"
    __table_args__ = {'comment': '模型配置'}
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(primary_key=True, max_length=32, nullable=False, default_factory=gen_id)
    config_name: Optional[str] = Field(sa_column=Column(VARCHAR(256), comment='配置名称', nullable=False))
    model_base_url: Optional[str] = Field(sa_column=Column(TEXT, comment='模型地址'))
    model_name: Optional[str] = Field(sa_column=Column(TEXT, comment='模型名称'))
    model_api_key: Optional[str] = Field(sa_column=Column(TEXT, comment='模型秘钥'))
    model_type: str = Field(default="embedding",
                            sa_column=Column(VARCHAR(256), comment='模型类型 embedding rerank', nullable=False))
    embedding_dim: int = Field(default=1024, sa_column=Column(Integer, comment='嵌入模型向量维度', nullable=False))
    create_time: datetime = Field(default_factory=datetime.now, nullable=False)
    update_time: datetime = Field(default_factory=datetime.now, nullable=False)


class DockerServer(SQLModel, table=True):
    __tablename__ = "docker_server"
    __table_args__ = {'comment': 'docker服务器'}

    id: str = Field(primary_key=True, max_length=32, nullable=False, default_factory=gen_id)
    server_name: Optional[str] = Field(sa_column=Column(TEXT, comment='服务器名称'))
    srv_base_url: Optional[str] = Field(sa_column=Column(TEXT, comment='服务器url'))
    create_time: datetime = Field(default_factory=datetime.now, nullable=False)
    update_time: datetime = Field(default_factory=datetime.now, nullable=False)


class ModelDeploy(SQLModel, table=True):
    """模型部署信息"""
    __tablename__ = "model_deploy"

    id: str = Field(
        primary_key=True,
        max_length=32,
        description="主键ID"
    )

    model_path: Optional[str] = Field(
        default=None,
        description="模型位置"
    )

    svc_port: Optional[int] = Field(
        default=None,
        description="docker服务端口配置，固定映射到容器的8000端口"
    )

    container_id: Optional[str] = Field(
        default=None,
        description="docker 容器id"
    )

    container_status: Optional[int] = Field(
        default=None,
        description="docker 容器状态, 0 未启动 1 已启动"
    )

    gpus_cfg: Optional[str] = Field(
        default=None,
        description="运行gpu配置 用于配置docker --gpus 参数 示例 device=0,1"
    )

    run_cfg: Optional[str] = Field(
        default=None,
        description="启动参数配置 自定义 端口号 host gpuid 等信息"
    )

    model_type: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, server_default='0',
                         comment='模型类型 0 embedding 模型 1 rerank模型'),
        description="模型类型：0-embedding模型, 1-rerank模型"
    )

    docker_server_id: str = Field(sa_column=Column(VARCHAR, comment='docker server id'))

    create_time: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )

    update_time: datetime = Field(
        default_factory=datetime.now,
        description="更新时间"
    )


class ModelDeployDetail(ModelDeploy):
    docker_server: Optional[DockerServer] = Relationship(link_model=DockerServer)
