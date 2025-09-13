from bubble_rag.retrieving.vectorial.knowledge_base import create_coll
from bubble_rag.databases.relation_database import get_session
from bubble_rag.entity.relational.documents import RagDocuments, DocTask
from bubble_rag.entity.relational.knowledge_base import DocKnowledgeBase
from bubble_rag.entity.relational.models import ModelConfig
from bubble_rag.entity.query.knowledge_base import DocKnowledgeBaseParam
from sqlmodel import text as sqltext, delete as sqldelete
from bubble_rag.retrieving.vectorial.knowledge_base import delete_knowledge_base as milvus_delete_knowledge_base


def add_knowledge_base(doc_db_param: DocKnowledgeBaseParam, coll_name: str) -> DocKnowledgeBase:
    with next(get_session()) as session:
        embedding_model = session.get(ModelConfig, doc_db_param.embedding_model_id)
        coll = create_coll(coll_name, embedding_model.embedding_dim)
        knowledge_base = DocKnowledgeBase(**doc_db_param.dict())
        knowledge_base.coll_name = coll_name
        session.add(knowledge_base)
        session.commit()
        session.refresh(knowledge_base)
        return knowledge_base
    return None


def get_knowledge_base(kbid: str) -> DocKnowledgeBase:
    with next(get_session()) as session:
        return session.get(DocKnowledgeBase, kbid)


def update_knowledge_base(doc_db_param: DocKnowledgeBaseParam) -> DocKnowledgeBase:
    """更新知识库信息"""
    with next(get_session()) as session:
        # 获取现有知识库
        knowledge_base = session.get(DocKnowledgeBase, doc_db_param.kb_id)
        if not knowledge_base:
            raise ValueError(f"知识库 {doc_db_param.kb_id} 不存在")
        
        # 更新字段
        if doc_db_param.kb_name is not None and doc_db_param.kb_name.strip():
            knowledge_base.kb_name = doc_db_param.kb_name.strip()
        
        if doc_db_param.rerank_model_id is not None and doc_db_param.rerank_model_id.strip():
            # 验证重排序模型是否存在
            rerank_model = session.get(ModelConfig, doc_db_param.rerank_model_id)
            if not rerank_model:
                raise ValueError(f"重排序模型 {doc_db_param.rerank_model_id} 不存在")
            knowledge_base.rerank_model_id = doc_db_param.rerank_model_id
        
        if doc_db_param.embedding_model_id is not None and doc_db_param.embedding_model_id.strip():
            # 验证向量模型是否存在
            embedding_model = session.get(ModelConfig, doc_db_param.embedding_model_id)
            if not embedding_model:
                raise ValueError(f"向量模型 {doc_db_param.embedding_model_id} 不存在")
            knowledge_base.embedding_model_id = doc_db_param.embedding_model_id
        
        if doc_db_param.kb_desc is not None:
            knowledge_base.kb_desc = doc_db_param.kb_desc
        
        # 更新修改时间
        from datetime import datetime
        knowledge_base.update_time = datetime.now()
        
        session.add(knowledge_base)
        session.commit()
        session.refresh(knowledge_base)
        milvus_delete_knowledge_base(knowledge_base.coll_name)
        create_coll(knowledge_base.coll_name, session.query(ModelConfig).get(knowledge_base.embedding_model_id).embedding_dim)
        return knowledge_base


def delete_knowledge_base(kbid: str) -> bool:
    with next(get_session()) as session:
        knowledge_base: DocKnowledgeBase = session.get(DocKnowledgeBase, kbid)
        session.exec(sqldelete(DocKnowledgeBase).where(DocKnowledgeBase.id == kbid))
        session.exec(sqldelete(RagDocuments).where(RagDocuments.doc_knowledge_base_id == kbid))
        session.exec(statement=sqltext("""
        delete
        from doc_file df
        where df.id in (select dt.file_id from doc_task dt where dt.doc_knowledge_base_id = :kbid)"""),
                     params={"kbid": kbid}, )
        session.exec(sqldelete(DocTask).where(DocTask.doc_knowledge_base_id == kbid))
        milvus_delete_knowledge_base(knowledge_base.coll_name)
        session.commit()
    return True
