from typing import List

from loguru import logger
from numpy import int64

from bubble_rag.databases.vector_databse import milvus_client
from bubble_rag.entity.vectorial.documents import MilvusRagDocuments, MilvusQueryRagDocuments
from bubble_rag.entity.relational.documents import RagDocuments
from bubble_rag.entity.relational.knowledge_base import DocKnowledgeBase
from bubble_rag.retrieving.relational.models import get_embedding_ef, get_rerank_ef


def add_rag_doc_list_by_mysqldata(rag_doc_list: List[RagDocuments], knowledge_base: DocKnowledgeBase):
    """将MySQL中的文档数据转换为向量并添加到Milvus"""
    embedding_ef = get_embedding_ef(knowledge_base.embedding_model_id)
    milvus_rag_doc_list = []
    for rag_doc in rag_doc_list:
        milvus_rag_doc = MilvusRagDocuments(
            **rag_doc.dict(exclude={"doc_task_id", "doc_file_name", "doc_knowledge_base_id", "create_time",
                                    "update_time", "doc_version"}))
        milvus_rag_doc.create_time = int64(rag_doc.create_time.timestamp())
        milvus_rag_doc.update_time = int64(rag_doc.update_time.timestamp())
        doc_dense = embedding_ef.encode_documents(documents=[rag_doc.doc_content], )[0]
        milvus_rag_doc.doc_ctn_dense = doc_dense
        milvus_rag_doc_list.append(milvus_rag_doc)
    add_rag_documents(coll_name=knowledge_base.coll_name, rag_doc_list=milvus_rag_doc_list)


def add_rag_documents(coll_name, rag_doc_list: List[MilvusRagDocuments]):
    """批量插入文档向量数据到Milvus集合"""
    # rag_doc.dict()
    milvus_client.insert(
        collection_name=coll_name,
        data=[
            rag_doc.dict()
            for rag_doc in rag_doc_list
        ]
    )


def edit_rag_document(coll_name, rag_doc: RagDocuments, knowledge_base: DocKnowledgeBase):
    """编辑文档，先删除原有向量再重新添加"""
    delete_rag_document(coll_name, rag_doc.id)
    add_rag_doc_list_by_mysqldata(rag_doc_list=[rag_doc], knowledge_base=knowledge_base)
    return True


def delete_rag_document(coll_name, doc_id: str):
    """从Milvus集合中删除指定文档"""
    milvus_client.delete(collection_name=coll_name, ids=[doc_id])
    return True


def list_documents(
        knowledge_base: DocKnowledgeBase,
        n_results: int,
) -> List[MilvusQueryRagDocuments]:
    """列出指定知识库中的文档"""
    coll_name = knowledge_base.coll_name
    doc_hits = milvus_client.query(collection_name=coll_name, limit=n_results, filter="", output_fields=["id", ])
    logger.info("=============================== doc_hits ===============================")
    logger.info(doc_hits)
    logger.info("=============================== doc_hits ===============================")
    query_rag_docs = []
    for doc_hit in doc_hits:
        milvus_doc = milvus_client.get(collection_name=coll_name, ids=[doc_hit.get("id")])[0]
        query_rag_docs.append(MilvusQueryRagDocuments(
            id=doc_hit.get("id"),
            doc_title=milvus_doc.get("doc_title"),
            doc_content=milvus_doc.get("doc_content"),
            embedding_score=0,
        ))
    return query_rag_docs


def semantic_merge_query(
        query_ctn: str,
        knowledge_base: DocKnowledgeBase,
        n_results: int,
        n_rerank: int,
        rerank: bool = True,
) -> List[MilvusQueryRagDocuments]:
    coll_name = knowledge_base.coll_name
    logger.info("===================== coll_name =====================")
    logger.info(coll_name)
    logger.info("===================== coll_name =====================")
    embedding_ef = get_embedding_ef(knowledge_base.embedding_model_id)
    doc_hits = milvus_client.search(
        collection_name=coll_name,
        data=embedding_ef.encode_queries(queries=[query_ctn]),
        anns_field='doc_ctn_dense',
        limit=n_results,
        output_fields=['id', ],
        filter=f"embedding_model_id in ['{knowledge_base.embedding_model_id}']"
    )
    query_rag_docs = []
    for doc_hit in doc_hits[0]:
        milvus_doc = milvus_client.get(collection_name=coll_name, ids=[doc_hit.get('id')])[0]
        query_rag_docs.append(MilvusQueryRagDocuments(
            id=doc_hit.get("id"),
            doc_title=milvus_doc.get("doc_title"),
            doc_content=milvus_doc.get("doc_content"),
            embedding_score=doc_hit.get('distance'),
        ))
    logger.info("======================= query_rag_docs =======================")
    logger.info(query_rag_docs)
    logger.info("======================= query_rag_docs =======================")
    if rerank and len(query_rag_docs) > 0:
        rerank_rf = get_rerank_ef(knowledge_base.rerank_model_id)
        rerank_result = rerank_rf(query=query_ctn, documents=[rag_doc.doc_content for rag_doc in query_rag_docs],
                                  top_k=n_rerank)
        rerank_rag_docs = []
        for rr in rerank_result:
            curr_rag_doc = query_rag_docs[rr.index]
            curr_rag_doc.rerank_score = rr.score
            rerank_rag_docs.append(curr_rag_doc)
        return rerank_rag_docs
    return query_rag_docs
