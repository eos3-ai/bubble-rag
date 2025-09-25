from sqlmodel import select
from typing import List

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import delete, func

from bubble_rag.entity.vectorial.documents import MilvusQueryRagDocuments
from bubble_rag.retrieving.vectorial.documents import edit_rag_document, add_rag_doc_list_by_mysqldata, delete_rag_document
from bubble_rag.databases.relation_database import SessionDep
from bubble_rag.entity.relational.documents import DocFile, DocTask, RagDocuments
from bubble_rag.entity.relational.knowledge_base import DocKnowledgeBase
from bubble_rag.retrieving.relational.documents import add_doc_file, semantic_query_documents
from bubble_rag.retrieving.relational.rag_chat import rag_chat_response, rag_chat_response_stream
from bubble_rag.entity.query.documents import RagDocumentsParam, DocTaskParam
from bubble_rag.entity.query.request_model import RagChatParam
from bubble_rag.entity.query.response_model import SrvResult, PageResult
from bubble_rag.utils.snowflake_utils import gen_id

router = APIRouter()


@router.post("/add_file")
async def add_doc_task(
        session: SessionDep,
        files: List[UploadFile] = File(...),
):
    """添加文档处理任务，上传文件并创建处理任务"""
    doc_files: List[DocFile] = []
    for file in files:
        doc_files.append(await add_doc_file(file))
    return SrvResult(code=200, msg='success', data=doc_files)
