from sqlalchemy import CHAR, Integer

from bubble_rag.entity.relational.models import ModelConfig
from sqlmodel import SQLModel, Field, VARCHAR, TEXT, Column, Relationship
from datetime import datetime
from typing import Optional

from bubble_rag.utils.snowflake_utils import gen_id


class DocKnowledgeBase(SQLModel, table=True):
    __tablename__ = "doc_knowledge_base"
    __table_args__ = {'comment': '文档知识库'}

    id: str = Field(primary_key=True, max_length=32, nullable=False, default_factory=gen_id)
    kb_name: Optional[str] = Field(sa_column=Column(VARCHAR(256), comment='知识库名称'))
    coll_name: Optional[str] = Field(sa_column=Column(TEXT, comment='milvus数据库集合名称'))
    rerank_model_id: Optional[str] = Field(sa_column=Column(CHAR(32), comment='重排序模型id'))
    embedding_model_id: Optional[str] = Field(sa_column=Column(CHAR(32), comment='向量模型id'))
    kb_desc: Optional[str] = Field(sa_column=Column(TEXT, comment='知识库备注'))
    create_time: datetime = Field(default_factory=datetime.now, nullable=False)
    update_time: datetime = Field(default_factory=datetime.now, nullable=False)


class DocKnowledgeBaseDetail(DocKnowledgeBase, table=True):
    rerank_model: ModelConfig = Relationship(link_model=ModelConfig)
    embedding_model: ModelConfig = Relationship(link_model=ModelConfig)
