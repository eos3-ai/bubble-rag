from pydantic import BaseModel, Field
from typing import Optional

from bubble_rag.utils.snowflake_utils import gen_id


class DocTaskParam(BaseModel):
    doc_knowledge_base_id: str = ""
    page_size: Optional[int] = 20
    page_num: Optional[int] = 1


class RagDocumentsParam(BaseModel):
    doc_id: Optional[str] = ""
    doc_title: Optional[str] = ""
    doc_content: Optional[str] = ""
    doc_knowledge_base_id: Optional[str] = ""
    limit_result: Optional[int] = Field(default=20, le=40, alias="limit")
    page_size: Optional[int] = 20
    page_num: Optional[int] = 1


class DocTask(BaseModel):
    ## 是否使用语义分段 0 不使用语义 1 使用语义分段
    semantic_chunk: int = 0
    ## 分段大小
    chunk_size: int = 512
