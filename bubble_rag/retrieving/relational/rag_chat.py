import json
import uuid
from typing import List
from loguru import logger

from bubble_rag.entity.vectorial.documents import MilvusQueryRagDocuments
from bubble_rag.retrieving.vectorial.documents import semantic_merge_query
from bubble_rag.retrieving.relational.knowledge_base import get_knowledge_base
from bubble_rag.utils.openai_utils import  get_client, chat_with_message, chat_with_message_stream
from bubble_rag.server_config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME


def rag_query_documents(question: str, doc_knowledge_base_id: str, limit_result: int = 10) -> List[MilvusQueryRagDocuments]:
    """RAG文档检索"""
    knowledge_base = get_knowledge_base(doc_knowledge_base_id)
    if not knowledge_base:
        logger.error(f"知识库不存在: {doc_knowledge_base_id}")
        return []

    documents = semantic_merge_query(
        query_ctn=question,
        knowledge_base=knowledge_base,
        n_results=limit_result,
        n_rerank=min(limit_result, 5),
        rerank=True
    )
    
    logger.info(f"检索到 {len(documents)} 个相关文档")
    return documents


def build_rag_context(documents: List[MilvusQueryRagDocuments]) -> str:
    """构建RAG上下文"""
    if not documents:
        return "暂无相关文档信息。"
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_part = f"文档{i}:\n{doc.doc_content}"
        context_parts.append(context_part)
    
    return "\n\n".join(context_parts)


def rag_chat_response(question: str, doc_knowledge_base_id: str, limit_result: int = 10, 
                     temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """RAG问答响应"""
    documents = rag_query_documents(question, doc_knowledge_base_id, limit_result)
    
    if not documents:
        return "抱歉，没有找到相关的文档信息来回答您的问题。"
    
    context = build_rag_context(documents)
    
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的知识助手。请基于提供的文档内容准确回答用户问题。如果文档中没有相关信息，请说明无法基于现有资料回答。"
        },
        {
            "role": "user", 
            "content": f"基于以下文档内容回答问题：\n\n{context}\n\n问题：{question}"
        }
    ]
    
    client = get_client(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    
    try:
        chat_response = chat_with_message(
            chat_client=client,
            model_name=LLM_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_times=3
        )
        return chat_response.resp_text if chat_response else "生成回答失败，请稍后重试。"
    except Exception as e:
        logger.error(f"RAG问答失败: {str(e)}")
        return f"生成回答时出现错误: {str(e)}"


def rag_chat_response_stream(question: str, doc_knowledge_base_id: str, limit_result: int = 10, 
                            temperature: float = 0.7, max_tokens: int = 2048):
    """RAG问答流式响应 - OpenAI SSE格式"""
    try:
        documents = rag_query_documents(question, doc_knowledge_base_id, limit_result)
        
        # 生成唯一的聊天ID
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        
        if not documents:
            # 没有找到文档时的响应
            chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": LLM_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "抱歉，没有找到相关的文档信息来回答您的问题。"
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            # 结束chunk
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": LLM_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        context = build_rag_context(documents)
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的知识助手。请基于提供的文档内容准确回答用户问题。如果文档中没有相关信息，请说明无法基于现有资料回答。"
            },
            {
                "role": "user", 
                "content": f"基于以下文档内容回答问题：\n\n{context}\n\n问题：{question}"
            }
        ]
        
        client = get_client(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        
        for chunk_content in chat_with_message_stream(
            chat_client=client,
            model_name=LLM_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_times=3
        ):
            # 构建OpenAI格式的chunk
            chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": LLM_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk_content
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        
        # 发送结束chunk
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk", 
            "created": int(__import__('time').time()),
            "model": LLM_MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"RAG问答流失败: {str(e)}")
        # 错误时也使用OpenAI格式
        error_chat_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        error_chunk = {
            "id": error_chat_id,
            "object": "chat.completion.chunk",
            "created": int(__import__('time').time()),
            "model": LLM_MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"生成回答时出现错误: {str(e)}"
                },
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"