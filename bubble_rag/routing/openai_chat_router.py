"""
OpenAI Chat Completions API 兼容路由
支持流式和非流式输出，兼容RAG功能
"""

import json
import uuid
import time
import traceback
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from bubble_rag.databases.relation_database import SessionDep
from bubble_rag.retrieving.relational.rag_chat import rag_query_documents, build_rag_context
from bubble_rag.utils.openai_utils import  get_client, chat_with_message, chat_with_message_stream
from bubble_rag.server_config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME
from bubble_rag.entity.query.openai_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionChunk,
    Choice, 
    StreamChoice,
    Delta,
    Message,
    MessageRole,
    Usage,
    FinishReason,
    ObjectType,
    OpenAIError,
    OpenAIErrorDetail,
    ErrorType
)

router = APIRouter()


def generate_chat_id() -> str:
    """生成聊天完成ID"""
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def estimate_tokens(text: str) -> int:
    """简单的token数量估算"""
    # 粗略估算：中文约1.5字符/token，英文约4字符/token
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def prepare_messages_for_rag(request: ChatCompletionRequest) -> List[dict]:
    """为RAG准备消息，将用户消息与检索到的文档结合"""
    if not request.doc_knowledge_base_id:
        # 如果没有指定知识库，直接返回原消息
        return [{"role": msg.role.value, "content": msg.content} for msg in request.messages]
    
    # 提取最后一个用户消息作为查询
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == MessageRole.USER:
            user_message = msg.content
            break
    
    if not user_message:
        # 没有用户消息，直接返回原消息
        return [{"role": msg.role.value, "content": msg.content} for msg in request.messages]

    logger.info("==================================== prepare_messages_for_rag user query ====================================")
    logger.info(user_message)
    logger.info("==================================== prepare_messages_for_rag user query ====================================")

    # 检索相关文档
    documents = rag_query_documents(
        question=user_message,
        doc_knowledge_base_id=request.doc_knowledge_base_id,
        limit_result=request.limit_result or 10
    )
    
    # 构建上下文
    if documents:
        context = build_rag_context(documents)
        logger.info(f"为问题检索到 {len(documents)} 个相关文档")
    else:
        context = "暂无相关文档信息。"
        logger.warning("未找到相关文档")
    
    # 重构消息列表
    messages = []
    
    # 添加系统消息（如果有的话）
    for msg in request.messages:
        if msg.role == MessageRole.SYSTEM:
            messages.append({"role": msg.role.value, "content": msg.content})
            break
    
    # 如果没有系统消息，添加默认的
    if not any(msg["role"] == "system" for msg in messages):
        messages.append({
            "role": "system",
            "content": "你是一个专业的知识助手。请基于提供的文档内容准确回答用户问题。如果文档中没有相关信息，请说明无法基于现有资料回答。"
        })
    
    # 添加历史对话（除了最后一个用户消息）
    for msg in request.messages:
        if msg.role != MessageRole.SYSTEM and msg.content != user_message:
            messages.append({"role": msg.role.value, "content": msg.content})
    
    # 添加增强后的用户消息
    enhanced_user_message = f"基于以下文档内容回答问题：\n\n{context}\n\n问题：{user_message}"
    messages.append({"role": "user", "content": enhanced_user_message})
    
    return messages


def create_error_response(error_message: str, error_type: str = ErrorType.API_ERROR, 
                         status_code: int = 500) -> HTTPException:
    """创建错误响应"""
    return HTTPException(
        status_code=status_code,
        detail=OpenAIError(
            error=OpenAIErrorDetail(
                message=error_message,
                type=error_type
            )
        ).model_dump()
    )


@router.post("/v1/chat/completions")
@router.post("/chat/completions")  # 同时支持两种路径
async def chat_completions(request: ChatCompletionRequest, session: SessionDep):
    """
    OpenAI Chat Completions API 兼容接口
    
    支持功能:
    - 完全兼容OpenAI Chat Completions API
    - 支持流式和非流式输出
    - 集成RAG功能 (通过doc_knowledge_base_id参数)
    - 支持多轮对话
    """
    
    try:
        # 参数验证
        if not request.messages:
            raise create_error_response("Messages cannot be empty", ErrorType.INVALID_REQUEST_ERROR, 400)
        
        # 生成聊天ID
        chat_id = generate_chat_id()
        created_time = int(time.time())
        
        # 准备消息
        messages = prepare_messages_for_rag(request)
        
        # 获取OpenAI客户端 - 使用动态参数或系统默认配置
        actual_base_url = request.base_url or LLM_BASE_URL
        actual_api_key = request.api_key or LLM_API_KEY
        client = get_client(base_url=actual_base_url, api_key=actual_api_key)

        logger.info("=================================== chat req ===================================")
        logger.info(request)
        logger.info(messages)
        logger.info("=================================== chat req ===================================")

        if request.stream:
            logger.info("============================== steam ==============================")
            # 流式响应
            return StreamingResponse(
                _generate_stream_response(
                    client=client,
                    messages=messages,
                    request=request,
                    chat_id=chat_id,
                    created_time=created_time
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            logger.info("============================== no steam ==============================")
            # 非流式响应
            response = await _generate_non_stream_response(
                client=client,
                messages=messages,
                request=request,
                chat_id=chat_id,
                created_time=created_time
            )
            return response
            
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Chat completions error: {str(e)}")
        logger.error(traceback.format_exc())
        raise create_error_response(f"Internal server error: {str(e)}")


async def _generate_non_stream_response(client, messages: List[dict], request: ChatCompletionRequest, 
                                      chat_id: str, created_time: int) -> ChatCompletionResponse:
    """生成非流式响应"""
    try:
        # 调用LLM
        chat_response = chat_with_message(
            chat_client=client,
            model_name=request.model,
            messages=messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2048,
            retry_times=3
        )
        
        if not chat_response or not chat_response.resp_text:
            raise create_error_response("Failed to generate response")
        
        # 计算token使用量
        prompt_text = "\n".join([msg["content"] for msg in messages])
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(chat_response.resp_text)
        total_tokens = prompt_tokens + completion_tokens
        
        # 构建响应
        response = ChatCompletionResponse(
            id=chat_id,
            object=ObjectType.CHAT_COMPLETION,
            created=created_time,
            model=request.model,
            choices=[Choice(
                index=0,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=chat_response.resp_text
                ),
                finish_reason=FinishReason.STOP
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Non-stream response generation failed: {str(e)}")
        raise create_error_response(f"Failed to generate response: {str(e)}")


async def _generate_stream_response(client, messages: List[dict], request: ChatCompletionRequest,
                                  chat_id: str, created_time: int):
    """生成流式响应"""
    try:
        # 发送第一个chunk（包含角色）
        first_chunk = ChatCompletionChunk(
            id=chat_id,
            object=ObjectType.CHAT_COMPLETION_CHUNK,
            created=created_time,
            model=request.model,
            choices=[StreamChoice(
                index=0,
                delta=Delta(role=MessageRole.ASSISTANT),
                finish_reason=None
            )]
        )
        yield f"data: {json.dumps(first_chunk.model_dump(), ensure_ascii=False)}\n\n"
        
        # 获取流式响应
        for chunk_content in chat_with_message_stream(
            chat_client=client,
            model_name=request.model,
            messages=messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2048,
            retry_times=3
        ):
            if chunk_content:  # 只有非空内容才发送
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    object=ObjectType.CHAT_COMPLETION_CHUNK,
                    created=created_time,
                    model=request.model,
                    choices=[StreamChoice(
                        index=0,
                        delta=Delta(content=chunk_content),
                        finish_reason=None
                    )]
                )
                yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
        
        # 发送结束chunk
        final_chunk = ChatCompletionChunk(
            id=chat_id,
            object=ObjectType.CHAT_COMPLETION_CHUNK,
            created=created_time,
            model=request.model,
            choices=[StreamChoice(
                index=0,
                delta=Delta(),
                finish_reason=FinishReason.STOP
            )]
        )
        yield f"data: {json.dumps(final_chunk.model_dump(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Stream response generation failed: {str(e)}")
        # 发送错误chunk
        error_chunk = ChatCompletionChunk(
            id=chat_id,
            object=ObjectType.CHAT_COMPLETION_CHUNK,
            created=int(time.time()),
            model=request.model,
            choices=[StreamChoice(
                index=0,
                delta=Delta(content=f"Error: {str(e)}"),
                finish_reason=FinishReason.STOP
            )]
        )
        yield f"data: {json.dumps(error_chunk.model_dump(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@router.get("/v1/models")
@router.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": LLM_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "bubble-rag",
                "permission": [],
                "root": LLM_MODEL_NAME,
                "parent": None
            }
        ]
    }