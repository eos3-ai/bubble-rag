"""
OpenAI Chat Completions API 兼容模型
"""

from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union, Dict, Any
import time


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """聊天消息"""
    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    name: Optional[str] = Field(None, description="消息发送者名称（可选）")


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions API 请求模型"""
    model_config = ConfigDict(protected_namespaces=())
    
    # 核心参数
    model: str = Field(default="chat-model", description="使用的模型名称")
    messages: List[Message] = Field(..., description="聊天消息列表")
    
    # 可选参数
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="核采样概率")
    max_tokens: Optional[int] = Field(default=2048, ge=1, description="生成的最大token数")
    stream: Optional[bool] = Field(default=False, description="是否开启流式响应")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="停止序列")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    user: Optional[str] = Field(default=None, description="用户标识")
    
    # RAG扩展参数
    doc_knowledge_base_id: Optional[str] = Field(default=None, description="知识库ID，用于RAG检索")
    limit_result: Optional[int] = Field(default=10, ge=1, le=20, description="RAG检索结果数量限制")
    
    # 动态模型配置参数
    base_url: Optional[str] = Field(default=None, description="模型服务的base URL，覆盖系统默认配置")
    api_key: Optional[str] = Field(default=None, description="模型服务的API密钥，覆盖系统默认配置")


class Usage(BaseModel):
    """Token使用情况"""
    prompt_tokens: int = Field(..., description="提示token数量")
    completion_tokens: int = Field(..., description="完成token数量")
    total_tokens: int = Field(..., description="总token数量")


class Choice(BaseModel):
    """完成选择"""
    index: int = Field(..., description="选择索引")
    message: Message = Field(..., description="助手回复消息")
    finish_reason: Optional[str] = Field(..., description="完成原因: stop, length, content_filter等")


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completions API 响应模型"""
    id: str = Field(..., description="聊天完成ID")
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(default_factory=lambda: int(time.time()), description="创建时间戳")
    model: str = Field(..., description="使用的模型名称")
    choices: List[Choice] = Field(..., description="生成的选择列表")
    usage: Optional[Usage] = Field(None, description="Token使用情况")
    system_fingerprint: Optional[str] = Field(None, description="系统指纹")


class Delta(BaseModel):
    """流式响应增量"""
    role: Optional[MessageRole] = Field(None, description="消息角色（仅在第一个chunk中）")
    content: Optional[str] = Field(None, description="内容增量")


class StreamChoice(BaseModel):
    """流式响应选择"""
    index: int = Field(..., description="选择索引")
    delta: Delta = Field(..., description="内容增量")
    finish_reason: Optional[str] = Field(None, description="完成原因")


class ChatCompletionChunk(BaseModel):
    """流式响应chunk"""
    id: str = Field(..., description="聊天完成ID")
    object: str = Field(default="chat.completion.chunk", description="对象类型")
    created: int = Field(default_factory=lambda: int(time.time()), description="创建时间戳")
    model: str = Field(..., description="使用的模型名称")
    choices: List[StreamChoice] = Field(..., description="生成的选择列表")
    system_fingerprint: Optional[str] = Field(None, description="系统指纹")


class OpenAIErrorDetail(BaseModel):
    """错误详情"""
    message: str = Field(..., description="错误消息")
    type: str = Field(..., description="错误类型")
    param: Optional[str] = Field(None, description="相关参数")
    code: Optional[str] = Field(None, description="错误代码")


class OpenAIError(BaseModel):
    """OpenAI API错误响应"""
    error: OpenAIErrorDetail = Field(..., description="错误详情")


# 一些常用的常量
class FinishReason:
    """完成原因常量"""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


class ObjectType:
    """对象类型常量"""
    CHAT_COMPLETION = "chat.completion"
    CHAT_COMPLETION_CHUNK = "chat.completion.chunk"


class ErrorType:
    """错误类型常量"""
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED_ERROR = "overloaded_error"