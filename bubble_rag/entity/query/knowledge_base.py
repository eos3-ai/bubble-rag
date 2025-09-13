from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from bubble_rag.utils.snowflake_utils import gen_id


class DocKnowledgeBaseParam(BaseModel):
    kb_id: Optional[str] = ""
    kb_name: Optional[str] = ""
    rerank_model_id: Optional[str] = ""
    embedding_model_id: Optional[str] = ""
    kb_desc: Optional[str] = ""
    page_size: Optional[int] = 20
    page_num: Optional[int] = 1
