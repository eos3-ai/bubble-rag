import json

from sqlalchemy import CHAR, BigInteger
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlmodel import SQLModel, Field, DATETIME, VARCHAR, TEXT, Integer, Column
from datetime import datetime
from typing import Optional
from bubble_rag.utils.snowflake_utils import gen_id


class DocFile(SQLModel, table=True):
    __tablename__ = "doc_file"
    __table_args__ = {'comment': '文档文件'}

    id: str = Field(primary_key=True, max_length=32, nullable=False, default_factory=gen_id)
    file_path: Optional[str] = Field(sa_column=Column(TEXT, comment='文件路径'))
    uncompress_path: Optional[str] = Field(sa_column=Column(TEXT, comment='文件解压路径'))
    file_size: Optional[int] = Field(sa_column=Column(Integer, comment='文件大小 byte'))
    file_md5: Optional[str] = Field(sa_column=Column(VARCHAR(32)))
    create_time: datetime = Field(default_factory=datetime.now, nullable=False)
    update_time: datetime = Field(default_factory=datetime.now, nullable=False)


class DocTask(SQLModel, table=True):
    __tablename__ = "doc_task"
    __table_args__ = {'comment': '文档解析任务'}

    id: str = Field(primary_key=True, max_length=32, nullable=False, default_factory=gen_id)
    file_id: str = Field(max_length=32, sa_column=Column(CHAR(32), comment='文件id', nullable=False))
    doc_knowledge_base_id: str = Field(max_length=32,
                                       sa_column=Column(CHAR(32), comment='所属知识库id', nullable=False))
    total_file: int = Field(default=0, sa_column=Column(Integer, comment='总文件数量', nullable=False))
    remaining_file: int = Field(default=0, sa_column=Column(Integer, comment='剩余文件数量', nullable=False))
    success_file: int = Field(default=0, sa_column=Column(Integer, comment='处理成功文件数量', nullable=False))
    curr_file_progress: int = Field(default=0, sa_column=Column(Integer, comment='当前文件处理进度', nullable=False))
    curr_filename: Optional[str] = Field(sa_column=Column(VARCHAR(256), comment='当前文件名称'))
    chunk_size: int = Field(default=512, sa_column=Column(Integer, comment='分段大小', nullable=False))
    create_time: datetime = Field(default_factory=datetime.now, nullable=False)
    update_time: datetime = Field(default_factory=datetime.now, nullable=False)


class RagDocuments(SQLModel, table=True):
    __tablename__ = "rag_documents"
    __table_args__ = {'comment': 'rag文档信息'}

    id: str = Field(primary_key=True, max_length=32, nullable=False, default_factory=gen_id)
    doc_title: Optional[str] = Field(sa_column=Column(TEXT, comment='文档标题'))
    doc_content: Optional[str] = Field(sa_column=Column(TEXT, comment='文档内容'))
    doc_file_id: str = Field(max_length=32, sa_column=Column(VARCHAR(32), comment='文档来源文件id', nullable=False))
    doc_task_id: str = Field(max_length=32, sa_column=Column(VARCHAR(32), comment='文档任务id', nullable=False))
    doc_file_name: Optional[str] = Field(sa_column=Column(TEXT, comment='文档来源文件名称'))
    doc_knowledge_base_id: str = Field(max_length=32,
                                       sa_column=Column(VARCHAR(32), comment='所属知识库id', nullable=False))
    embedding_model_id: Optional[str] = Field(sa_column=Column(CHAR(32), comment='向量模型id'))
    doc_version: Optional[int] = Field(sa_column=Column(BigInteger, comment='版本号', nullable=False),
                                       default_factory=gen_id)
    create_time: datetime = Field(default_factory=datetime.now, nullable=False)
    update_time: datetime = Field(default_factory=datetime.now, nullable=False)

