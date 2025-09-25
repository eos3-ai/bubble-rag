import nltk

from bubble_rag.job.documents_job import sync_documents_embedding_model, doc_tasks_parse_job

nltk.data.path.append("/app/nltk_data")

from apscheduler.jobstores.memory import MemoryJobStore
from fastapi import FastAPI, Request
from loguru import logger

from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

from bubble_rag.server_config import ALLOWED_ORIGINS, ALLOWED_METHODS, ALLOWED_HEADERS, SRV_BASE_RUI
from bubble_rag.routing.doc_router import router as doc_router
from bubble_rag.routing.file_router import router as file_router
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


def get_srv_base_uri():
    """获取服务基础URI路径"""
    uri = SRV_BASE_RUI.strip().removesuffix('/')
    return uri


@app.on_event("startup")
def startup_event():
    """应用启动事件处理，初始化定时任务"""
    scheduler.add_job(
        doc_tasks_parse_job,
        'interval',
        seconds=3,
        id='sequential_task',
        max_instances=1,
        coalesce=True,
        replace_existing=True  # 避免重复添加相同 ID 的任务
    )
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
    router=file_router,
    prefix=f'{get_srv_base_uri()}/api/v1/files',
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
