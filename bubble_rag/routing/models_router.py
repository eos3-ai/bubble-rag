from fastapi import APIRouter
from sqlalchemy import func, delete as sqldelete
from sqlmodel import select, and_, or_

from bubble_rag.databases.relation_database import SessionDep
from bubble_rag.entity.relational.knowledge_base import DocKnowledgeBase
from bubble_rag.entity.relational.models import ModelConfig
from bubble_rag.databases.relation_database import update_model_from_dict
from bubble_rag.entity.query.models import ModelConfigReq
from bubble_rag.entity.query.response_model import SrvResult, PageResult

router = APIRouter()


@router.post("/add_model")
async def add_model(model_param: ModelConfigReq, session: SessionDep):
    """添加模型配置"""
    model_conf: ModelConfig = ModelConfig(**model_param.dict())
    session.add(model_conf)
    session.commit()
    session.refresh(model_conf)
    return SrvResult(code=200, msg='success', data=model_conf)


@router.post("/edit_model")
async def edit_model(model_param: ModelConfigReq, session: SessionDep):
    """编辑模型配置"""
    if model_param and model_param.id:
        db_pro = session.get(ModelConfigReq, model_param.id)
        update_model_from_dict(db_pro, model_param.dict(exclude={'id'}))
        session.commit()
        session.refresh(db_pro)
        return SrvResult(code=200, msg='success', data=db_pro)
    return SrvResult(code=500, msg='fail', data=None)


@router.post("/delete_model")
async def delete_model(model_param: ModelConfigReq, session: SessionDep):
    """删除模型配置，检查是否被知识库引用"""
    if model_param and model_param.model_id:
        kb_count = session.exec(select(func.count())
                                .select_from(DocKnowledgeBase)
                                .where(or_(*[DocKnowledgeBase.rerank_model_id == model_param.model_id,
                                             DocKnowledgeBase.embedding_model_id == model_param.model_id]))).one()
        if kb_count > 0:
            return SrvResult(code=401, msg='需要先删除已关联的知识库', data=None)
        session.exec(sqldelete(ModelConfig).where(ModelConfig.id == model_param.model_id))
        session.commit()
        return SrvResult(code=200, msg='success', data=None)
    return SrvResult(code=500, msg='fail', data=None)


@router.post("/list_models")
async def list_models(model_param: ModelConfigReq, session: SessionDep):
    """分页查询模型配置列表"""
    conditions = []
    page_num = model_param.page_num
    if page_num < 1:
        page_num = 1
    page_size = model_param.page_size
    offset = (model_param.page_num - 1) * page_size
    if model_param.config_name and model_param.config_name.strip():
        conditions.append(ModelConfig.config_name.like(f"%{model_param.model_name}%"))
    if model_param.model_type and model_param.model_type.strip():
        conditions.append(ModelConfig.model_type.in_([model_param.model_type]))
    if len(conditions) > 0:
        statement = select(ModelConfig).where(and_(*conditions)).order_by(ModelConfig.update_time.desc())
        statement_page = select(func.count()).where(and_(*conditions)).select_from(ModelConfig).order_by(
            ModelConfig.update_time.desc())
    else:
        statement = select(ModelConfig).order_by(ModelConfig.update_time.desc())
        statement_page = select(func.count()).select_from(ModelConfig).order_by(
            ModelConfig.update_time.desc())
    # 计算总数
    total = session.exec(statement_page).one()
    # 应用分页
    statement = statement.offset(offset).limit(page_size)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    models = session.exec(statement).all()
    page_result = PageResult(items=models, total=total, page=page_num, page_size=page_size, total_pages=total_pages)
    return SrvResult(code=200, msg='success', data=page_result)


@router.post("/list_all_models")
async def list_all_models(model_param: ModelConfigReq, session: SessionDep):
    """查询指定类型的所有模型配置"""
    conditions = [ModelConfig.model_type.in_([model_param.model_type])]
    statement = select(ModelConfig).where(and_(*conditions)).order_by(ModelConfig.update_time.desc())
    models = session.exec(statement).all()
    return SrvResult(code=200, msg='success', data=models)
