from sqlmodel import select
from typing import List

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import delete, func

from bubble_rag.entity.vectorial.documents import MilvusQueryRagDocuments
from bubble_rag.retrieving.vectorial.documents import edit_rag_document, add_rag_doc_list_by_mysqldata, \
    delete_rag_document
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


@router.post("/add_doc_task")
async def add_doc_task(
        session: SessionDep,
        files: List[UploadFile] = File(...),
        doc_knowledge_base_id: str = Form(...),
        ## 分段大小
        chunk_size: int = Form(...),
):
    """添加文档处理任务，上传文件并创建处理任务"""
    doc_files: List[DocFile] = []
    for file in files:
        doc_files.append(await add_doc_file(file))
    doc_tasks: List[DocTask] = []
    for doc_file in doc_files:
        doc_task = DocTask(
            file_id=doc_file.id,
            total_file=1,
            remaining_file=1,
            success_file=0,
            curr_file_progress=0,
            curr_filename="",
            doc_knowledge_base_id=doc_knowledge_base_id,
            chunk_size=chunk_size
        )
        doc_tasks.append(doc_task)
    session.add_all(doc_tasks)
    session.commit()
    doc_tasks_new = []
    for dt in doc_tasks:
        session.refresh(dt)
        doc_tasks_new.append(dt)
    logger.info("=============================== add_doc_task ===============================")
    logger.info(doc_tasks_new)
    logger.info("=============================== add_doc_task ===============================")

    return SrvResult(code=200, msg='success', data=doc_tasks_new)


@router.post("/list_doc_tasks")
async def list_doc_tasks(doc_task_param: DocTaskParam, session: SessionDep, ):
    """分页查询文档处理任务列表"""
    page_num = doc_task_param.page_num
    conditions = []
    if page_num < 1:
        page_num = 1
    conditions.append(DocTask.doc_knowledge_base_id == doc_task_param.doc_knowledge_base_id)
    page_size = doc_task_param.page_size
    offset = (doc_task_param.page_num - 1) * page_size
    statement = select(DocTask).where(*conditions).order_by(DocTask.update_time.desc())
    statement_page = select(func.count()).select_from(DocTask).where(*conditions).order_by(DocTask.update_time.desc())
    # 计算总数
    total = session.exec(statement_page).one()
    # 应用分页
    statement = statement.offset(offset).limit(page_size)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    models = session.exec(statement).all()
    page_result = PageResult(items=models, total=total, page=page_num, page_size=page_size, total_pages=total_pages)
    return SrvResult(code=200, msg='success', data=page_result)


@router.post("/semantic_query")
async def semantic_query(rag_doc_param: RagDocumentsParam, session: SessionDep, ):
    """语义搜索文档，支持向量相似度查询"""
    if rag_doc_param.doc_content and rag_doc_param.doc_content.strip():
        rag_doc_list = semantic_query_documents(rag_doc_param)
        page_result = PageResult(items=rag_doc_list, total=len(rag_doc_list), page=1, page_size=len(rag_doc_list),
                                 total_pages=1)
    else:
        page_num = rag_doc_param.page_num
        if page_num < 1:
            page_num = 1
        page_size = rag_doc_param.page_size
        offset = (rag_doc_param.page_num - 1) * page_size
        conditions = [RagDocuments.doc_knowledge_base_id == rag_doc_param.doc_knowledge_base_id]
        statement = select(RagDocuments).where(*conditions).order_by(RagDocuments.update_time.desc())
        statement_page = select(func.count()).select_from(RagDocuments).where(*conditions).order_by(
            RagDocuments.update_time.desc())
        # 计算总数
        total = session.exec(statement_page).one()
        # 应用分页
        statement = statement.offset(offset).limit(page_size)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        rag_doc_list = session.exec(statement).all()
        page_result = PageResult(items=[
            MilvusQueryRagDocuments(id=rd.id, doc_title=rd.doc_title, doc_content=rd.doc_content, embedding_score=0,
                                    rerank_score=0) for rd in rag_doc_list], total=total, page=page_num,
            page_size=page_size, total_pages=total_pages)
    return SrvResult(code=200, msg='success', data=page_result)


@router.post("/add_doc")
async def add_doc(rag_doc: RagDocumentsParam, session: SessionDep):
    """手动添加文档片段到知识库"""
    if len(rag_doc.doc_content) > 1024 * 16:
        return SrvResult(code=410, msg="chunk非法")
    doc_kb: DocKnowledgeBase = session.get(DocKnowledgeBase, rag_doc.doc_knowledge_base_id)
    rag_doc = RagDocuments(
        doc_title=rag_doc.doc_title,
        doc_content=rag_doc.doc_content,
        doc_knowledge_base_id=rag_doc.doc_knowledge_base_id,
        doc_file_id='',
        doc_task_id='',
        doc_file_name='',
        embedding_model_id=doc_kb.embedding_model_id,
        doc_version=gen_id()
    )
    session.add(rag_doc)
    logger.info("========================================= rag_doc =========================================")
    logger.info(rag_doc)
    logger.info("========================================= rag_doc =========================================")
    add_rag_doc_list_by_mysqldata(rag_doc_list=[rag_doc], knowledge_base=doc_kb)
    session.commit()
    session.refresh(rag_doc)
    return SrvResult(code=200, msg='success', data=rag_doc)


@router.post("/delete_doc")
async def delete_doc(rag_doc: RagDocumentsParam, session: SessionDep):
    """删除指定文档片段"""
    rag_doc_db = session.get(RagDocuments, rag_doc.doc_id)
    doc_kb = session.get(DocKnowledgeBase, rag_doc_db.doc_knowledge_base_id)
    session.exec(delete(RagDocuments).where(RagDocuments.id == rag_doc.doc_id))
    delete_rag_document(doc_kb.coll_name, rag_doc.doc_id)
    session.commit()
    return SrvResult(code=200, msg='success', data=None)


@router.post("/edit_doc")
async def edit_doc(rag_doc: RagDocumentsParam, session: SessionDep):
    """编辑文档片段内容"""
    rag_doc_db = session.get(RagDocuments, rag_doc.doc_id)
    rag_doc_db.doc_title = rag_doc.doc_title
    rag_doc_db.doc_content = rag_doc.doc_content
    doc_knowledge_base = session.get(DocKnowledgeBase, rag_doc_db.doc_knowledge_base_id)
    rag_doc_db.embedding_model_id = doc_knowledge_base.embedding_model_id
    edit_rag_document(doc_knowledge_base.coll_name, rag_doc=rag_doc_db, knowledge_base=doc_knowledge_base)
    session.commit()
    session.refresh(rag_doc_db)
    return SrvResult(code=200, msg='success', data=rag_doc_db)


@router.post("/rag_chat")
async def rag_chat(rag_chat_param: RagChatParam, session: SessionDep):
    """RAG对话接口，支持流式和非流式响应"""
    if rag_chat_param.stream:
        def generate():
            for chunk in rag_chat_response_stream(
                    question=rag_chat_param.question,
                    doc_knowledge_base_id=rag_chat_param.doc_knowledge_base_id,
                    limit_result=10,
                    temperature=rag_chat_param.temperature,
                    max_tokens=rag_chat_param.max_tokens
            ):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "X-Accel-Buffering": "no"  # 禁用Nginx缓冲，确保实时流式输出
            }
        )
    else:
        response = rag_chat_response(
            question=rag_chat_param.question,
            doc_knowledge_base_id=rag_chat_param.doc_knowledge_base_id,
            limit_result=rag_chat_param.limit_result,
            temperature=rag_chat_param.temperature,
            max_tokens=rag_chat_param.max_tokens
        )
        return SrvResult(code=200, msg='success', data={"answer": response})
