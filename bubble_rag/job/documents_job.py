"""同步文档向量模型 - 检查并更新不一致的文档向量"""
import traceback

from loguru import logger
from sqlmodel import text as sqltext

from bubble_rag.databases.memery_database import redis_db
from bubble_rag.databases.relation_database import get_session
from bubble_rag.entity.relational.documents import RagDocuments, DocTask
from bubble_rag.entity.relational.knowledge_base import DocKnowledgeBase
from bubble_rag.retrieving.relational.documents import process_doc_task
from bubble_rag.retrieving.vectorial.documents import delete_rag_document, add_rag_doc_list_by_mysqldata


def sync_documents_embedding_model():
    logger.info("开始同步文档向量模型...")

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
                        lock = redis_db.lock(f"rag_doc_embedding_update_{doc_id}", ttl=1000 * 120)
                        with lock:
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
        logger.info(
            f"文档向量模型同步完成: 处理 {processed_kb_count} 个知识库，检查 {total_processed_count} 个文档，更新 {total_updated_count} 个文档")

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"文档向量模型同步过程中发生错误: {str(e)}")
        logger.info(f"文档向量模型同步失败: {str(e)}")


def doc_tasks_parse_job():
    """批量处理所有待处理的文档任务"""
    try:
        with next(get_session()) as session:
            sql = """
            select dt.*
            from doc_task dt
            where dt.success_file < dt.total_file
            order by timestampdiff(second, dt.create_time, dt.update_time) asc
            limit 10
            """
            doc_task_result = session.exec(statement=sqltext(sql), params={}).all()
            unprocess_doc_tasks = [DocTask(**dtr._asdict()) for dtr in doc_task_result]
            for doc_task in unprocess_doc_tasks:
                process_doc_task(doc_task)
    except Exception as e:
        traceback.print_exc()
