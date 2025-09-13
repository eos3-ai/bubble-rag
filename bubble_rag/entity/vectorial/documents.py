import datetime
from typing import Optional, List

from numpy import ndarray, int64, float64
from pydantic import BaseModel, ConfigDict


class MilvusRagDocuments(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    doc_title: str
    doc_content: str
    doc_ctn_dense: List = None
    doc_file_id: str
    embedding_model_id: str
    create_time: Optional[int64] = None
    update_time: Optional[int64] = None


class MilvusQueryRagDocuments(BaseModel):
    id: str
    doc_title: str
    doc_content: str
    embedding_score: float
    rerank_score: Optional[float] = 0
