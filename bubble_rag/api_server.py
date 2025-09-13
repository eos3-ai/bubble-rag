import nltk

nltk.data.path.append("/app/nltk_data")

from apscheduler.jobstores.memory import MemoryJobStore
from fastapi import FastAPI, Request
from loguru import logger

from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

from bubble_rag.retrieving.relational.documents import process_all_doc_tasks
from bubble_rag.server_config import ALLOWED_ORIGINS, ALLOWED_METHODS, ALLOWED_HEADERS, SRV_BASE_RUI
from bubble_rag.routing.doc_router import router as doc_router
from bubble_rag.routing.knowledge_base_router import router as knowledge_base_router
from bubble_rag.routing.docker_server_router import router as docker_server_router
from bubble_rag.routing.model_deploy_router import router as model_deploy_router
from bubble_rag.routing.models_router import router as models_router
from bubble_rag.routing.openai_chat_router import router as openai_chat_router
from bubble_rag.routing.unified_training_router import router as training_router

app = FastAPI()

# 配置调度器（使用线程池，最大并发数为 1）
executors = {
    'default': ThreadPoolExecutor(max_workers=1)  # 全局限制并发数
}

# 使用数据库存储任务（可选，确保重启后任务不丢失）
jobstores = {
    'default': MemoryJobStore()
}

scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors)


def sequential_task():
    """需要顺序执行的任务"""
    print("================================= 开始执行任务... =================================")
    process_all_doc_tasks()
    print("================================= 任务执行完成 =================================")


def sync_documents_embedding_model():
    """同步文档向量模型 - 检查并更新不一致的文档向量"""
    from bubble_rag.databases.relation_database import get_session
    from bubble_rag.entity.relational.documents import RagDocuments
    from bubble_rag.entity.relational.knowledge_base import DocKnowledgeBase
    from bubble_rag.retrieving.vectorial.documents import delete_rag_document, add_rag_doc_list_by_mysqldata
    from loguru import logger

    print("开始同步文档向量模型...")

    total_processed_count = 0
    total_updated_count = 0
    processed_kb_count = 0

    try:
        # 获取所有知识库列表
        knowledge_base_ids = []
        with next(get_session()) as session:
            knowledge_bases = session.query(DocKnowledgeBase.id, DocKnowledgeBase.kb_name).all()
            knowledge_base_ids = [(kb.id, kb.kb_name) for kb in knowledge_bases]

        for kb_id, kb_name in knowledge_base_ids:
            try:
                # 为每个知识库开启新的session
                with next(get_session()) as kb_session:
                    # 重新查询当前知识库对象
                    knowledge_base = kb_session.query(DocKnowledgeBase).filter(
                        DocKnowledgeBase.id == kb_id
                    ).first()

                    if not knowledge_base:
                        logger.warning(f"知识库 {kb_id} 不存在，跳过")
                        continue

                    # 跳过没有向量模型的知识库
                    if not knowledge_base.embedding_model_id:
                        logger.info(f"知识库 {kb_id} ({kb_name}) 没有配置向量模型，跳过")
                        continue

                    # 查询该知识库下向量模型ID不一致的文档
                    inconsistent_docs = kb_session.query(RagDocuments).filter(
                        RagDocuments.doc_knowledge_base_id == knowledge_base.id,
                        RagDocuments.embedding_model_id != knowledge_base.embedding_model_id
                    ).all()
                    logger.info(
                        "======================================= 文档不一致 =======================================")
                    logger.info(inconsistent_docs)
                    logger.info(
                        "======================================= 文档不一致 =======================================")

                    if not inconsistent_docs:
                        logger.debug(f"知识库 {kb_id} ({kb_name}) 中的文档向量模型已一致")
                        processed_kb_count += 1
                        continue

                    logger.info(f"知识库 {kb_id} ({kb_name}) 发现 {len(inconsistent_docs)} 个向量模型不一致的文档")

                    # 获取需要处理的文档ID列表
                    doc_ids = [doc.id for doc in inconsistent_docs]

                # 处理每个文档，每个文档使用独立的session
                kb_updated_count = 0
                for doc_id in doc_ids:
                    try:
                        with next(get_session()) as doc_session:
                            # 重新查询文档和知识库对象
                            rag_doc = doc_session.query(RagDocuments).filter(
                                RagDocuments.id == doc_id
                            ).first()

                            if not rag_doc:
                                logger.warning(f"文档 {doc_id} 不存在，跳过")
                                continue

                            knowledge_base = doc_session.query(DocKnowledgeBase).filter(
                                DocKnowledgeBase.id == rag_doc.doc_knowledge_base_id
                            ).first()

                            if not knowledge_base:
                                logger.warning(f"文档 {doc_id} 对应的知识库不存在，跳过")
                                continue

                            # 再次检查是否需要更新（防止并发情况下的重复处理）
                            if rag_doc.embedding_model_id == knowledge_base.embedding_model_id:
                                logger.debug(f"文档 {doc_id} 向量模型已一致，跳过")
                                continue

                            logger.info(
                                f"处理文档 {doc_id} - 当前向量模型: {rag_doc.embedding_model_id}, 目标向量模型: {knowledge_base.embedding_model_id}")

                            # 删除Milvus中的旧数据
                            if knowledge_base.coll_name:
                                delete_rag_document(knowledge_base.coll_name, rag_doc.id)
                                logger.info(
                                    f"已删除文档 {doc_id} 在Milvus集合 {knowledge_base.coll_name} 中的旧向量数据")

                            # 更新文档的向量模型ID
                            rag_doc.embedding_model_id = knowledge_base.embedding_model_id
                            doc_session.add(rag_doc)

                            # 重新添加到Milvus（使用新的向量模型）
                            add_rag_doc_list_by_mysqldata(rag_doc_list=[rag_doc], knowledge_base=knowledge_base)
                            logger.info(f"已使用新向量模型重新添加文档 {doc_id} 到Milvus")

                            # 提交当前文档的更改
                            doc_session.commit()

                            kb_updated_count += 1
                            total_updated_count += 1

                    except Exception as e:
                        logger.error(f"处理文档 {doc_id} 时发生错误: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue

                total_processed_count += len(doc_ids)
                processed_kb_count += 1

                logger.info(f"知识库 {kb_id} ({kb_name}) 处理完成: 更新 {kb_updated_count} 个文档")

            except Exception as e:
                logger.error(f"处理知识库 {kb_id} 时发生错误: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        logger.info(
            f"文档向量模型同步完成: 处理 {processed_kb_count} 个知识库，检查 {total_processed_count} 个文档，更新 {total_updated_count} 个文档")
        print(
            f"文档向量模型同步完成: 处理 {processed_kb_count} 个知识库，检查 {total_processed_count} 个文档，更新 {total_updated_count} 个文档")

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"文档向量模型同步过程中发生错误: {str(e)}")
        print(f"文档向量模型同步失败: {str(e)}")


def get_srv_base_uri():
    """获取服务基础URI路径"""
    uri = SRV_BASE_RUI.strip().removesuffix('/')
    return uri


@app.on_event("startup")
def startup_event():
    """应用启动事件处理，初始化定时任务"""
    # 添加文档任务处理（重点配置：max_instances=1, coalesce=True）
    scheduler.add_job(
        sequential_task,
        'interval',
        seconds=3,
        id='sequential_task',
        max_instances=1,
        coalesce=True,
        replace_existing=True  # 避免重复添加相同 ID 的任务
    )

    # 添加文档向量模型同步任务（每10分钟执行一次）
    scheduler.add_job(
        sync_documents_embedding_model,
        'interval',
        seconds=5,
        id='sync_documents_embedding',
        max_instances=1,
        coalesce=True,
        replace_existing=True
    )

    scheduler.start()


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭事件处理，停止定时任务调度器"""
    scheduler.shutdown()


def valid_token(header):
    """验证请求头中的token是否有效"""
    # return header and header is not None and '1112222' in header
    return True


@app.middleware("http")
async def authorization_middleware(request: Request, call_next):
    """HTTP授权中间件，处理请求认证（当前已禁用认证检查）"""
    # token = request.headers.get("Authorization")
    # if not token or not valid_token(token):
    #     return Response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})
    # response = await call_next(request)
    # return response
    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
    expose_headers=["X-Total-Count"],
    max_age=600,
)


@app.get("/health")
async def health():
    return "OK"


## 静态资源文件
static = f"{get_srv_base_uri()}/static"

app.include_router(
    router=doc_router,
    prefix=f'{get_srv_base_uri()}/api/v1/documents',
)

app.include_router(
    router=knowledge_base_router,
    prefix=f'{get_srv_base_uri()}/api/v1/knowledge_base',
)

app.include_router(
    router=docker_server_router,
    prefix=f'{get_srv_base_uri()}/api/v1/docker_servers',
)

app.include_router(
    router=model_deploy_router,
    prefix=f'{get_srv_base_uri()}/api/v1/model_deploy',
)

app.include_router(
    router=models_router,
    prefix=f'{get_srv_base_uri()}/api/v1/models',
)

app.include_router(
    router=openai_chat_router,
    prefix=f'{get_srv_base_uri()}/api',
)

app.include_router(
    router=training_router,
    prefix=f'{get_srv_base_uri()}/api/v1/training',
    tags=["训练管理"]
)
