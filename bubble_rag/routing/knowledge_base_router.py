import traceback

from fastapi import APIRouter
from bubble_rag.entity.relational.models import ModelConfig
from loguru import logger
from sqlmodel import select, func

from bubble_rag.databases.relation_database import SessionDep
from bubble_rag.entity.relational.knowledge_base import DocKnowledgeBase, DocKnowledgeBaseDetail
from bubble_rag.retrieving.relational.knowledge_base import add_knowledge_base as svc_add_knowledge_base, \
    delete_knowledge_base, update_knowledge_base as svc_update_knowledge_base
from bubble_rag.entity.query.knowledge_base import DocKnowledgeBaseParam
from bubble_rag.entity.query.response_model import SrvResult, PageResult
import random
import string


def generate_english_uuid(length=32):
    """
    生成一个只包含英文字母的UUID

    参数:
    length: UUID的长度，默认为32个字符

    返回:
    只包含英文字母的UUID字符串
    """
    # 只使用英文字母（大小写）
    letters = string.ascii_letters  # 包含所有大小写字母

    # 随机选择字母生成UUID
    uuid = ''.join(random.choice(letters) for _ in range(length))

    return uuid


def format_english_uuid(uuid, groups=4):
    """
    格式化UUID，添加连字符使其看起来像标准UUID

    参数:
    uuid: 原始UUID字符串
    groups: 分组数量，默认为4组

    返回:
    格式化后的UUID字符串
    """
    group_length = len(uuid) // groups
    formatted = '-'.join(uuid[i:i + group_length] for i in range(0, len(uuid), group_length))
    return formatted


router = APIRouter()


@router.post("/add_knowledge_base")
async def add_knowledge_base(doc_db_param: DocKnowledgeBaseParam, session: SessionDep):
    """添加新知识库"""
    # coll_name = doc_db_param.kb_name + "_" + str(uuid.uuid4()).replace("-", "")
    coll_name = str(generate_english_uuid(32)).replace("-", "")
    knowledge_base = svc_add_knowledge_base(doc_db_param, coll_name)
    return SrvResult(code=200, msg='success', data=knowledge_base)


@router.post("/update_knowledge_base")
async def update_knowledge_base(doc_db_param: DocKnowledgeBaseParam, session: SessionDep):
    """
    更新知识库接口
    
    可以修改知识库的以下字段：
    - kb_name: 知识库名称
    - rerank_model_id: 重排序模型ID
    - embedding_model_id: 向量模型ID
    - kb_desc: 知识库描述
    
    注意：如果修改了embedding_model_id，系统的定时任务会自动同步相关文档的向量数据
    """
    try:
        # 验证必填参数
        if not doc_db_param.kb_id or not doc_db_param.kb_id.strip():
            return SrvResult(code=400, msg='知识库ID不能为空', data=None)
        
        # 调用服务层方法更新知识库
        updated_knowledge_base = svc_update_knowledge_base(doc_db_param)
        return SrvResult(code=200, msg='知识库更新成功', data=updated_knowledge_base)
        
    except ValueError as e:
        return SrvResult(code=400, msg=str(e), data=None)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"更新知识库失败: {str(e)}")
        logger.error(traceback.format_exc())
        return SrvResult(code=500, msg=f'更新知识库失败: {str(e)}', data=None)


@router.post("/delete_knowledge_base")
async def delete_knowledge_base_api(doc_db_param: DocKnowledgeBaseParam, session: SessionDep):
    """删除指定知识库"""
    delete_knowledge_base(kbid=doc_db_param.kb_id)
    return SrvResult(code=200, msg='success', data=None)


@router.post("/list_knowledge_base")
async def list_knowledge_base(doc_db_param: DocKnowledgeBaseParam, session: SessionDep):
    """分页查询知识库列表"""
    conditions = []
    page_num = doc_db_param.page_num
    if page_num < 1:
        page_num = 1
    page_size = doc_db_param.page_size
    offset = (doc_db_param.page_num - 1) * page_size
    if doc_db_param.kb_name and doc_db_param.kb_name.strip():
        conditions.append(DocKnowledgeBase.kb_name.like(f"%{doc_db_param.kb_name}%"))
    statement = select(DocKnowledgeBase).where(*conditions).order_by(DocKnowledgeBase.update_time.desc())
    statement_page = select(func.count()).select_from(DocKnowledgeBase).where(*conditions).order_by(
        DocKnowledgeBase.update_time.desc())
    # 计算总数
    total = session.exec(statement_page).one()
    # 应用分页
    statement = statement.offset(offset).limit(page_size)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    models = session.exec(statement).all()
    page_result = PageResult(items=models, total=total, page=page_num, page_size=page_size, total_pages=total_pages)
    return SrvResult(code=200, msg='success', data=page_result)


@router.get("/get_knowledge_base_detail")
async def get_knowledge_base_detail(doc_db_param: DocKnowledgeBaseParam, session: SessionDep):
    """获取知识库详细信息"""
    kbobj = session.get(DocKnowledgeBase, doc_db_param.kb_id)
    kb_detail = DocKnowledgeBaseDetail(**kbobj)
    kb_detail.embedding_model = session.get(ModelConfig, kbobj.embedding_model_id)
    kb_detail.rerank_model = session.get(ModelConfig, kbobj.rerank_model_id)
    return SrvResult(code=200, msg='success', data=kb_detail)
