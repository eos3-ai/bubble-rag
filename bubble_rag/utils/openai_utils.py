import base64
import random
import time
import traceback
from json import loads, dumps

from langchain_core.prompts import ChatPromptTemplate
from openai import Completion, ChatCompletion, Stream, APIConnectionError, OpenAI
from openai.types.chat import ChatCompletionChunk

from loguru import logger

from bubble_rag.server_config import VLM_MODEL_NAME, VLM_BASE_URL, VLM_API_KEY


class ChatResponse:
    query: str
    resp: Completion
    resp_text: str

    def __init__(self, query: str, resp_text: str, resp: Completion):
        self.query = query
        self.resp_text = resp_text
        self.resp = resp


def get_client(base_url='http://127.0.0.1:30220/v1/', api_key='11111111111111'):
    """创建OpenAI客户端实例"""
    tmp_client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return tmp_client


def_vlm_client = get_client(base_url=VLM_BASE_URL, api_key=VLM_API_KEY)


def chat_with_prompt(
        chat_client,
        model_name,
        prompt: ChatPromptTemplate,
        prompt_args,
        timeout=5,
        max_tokens=1024 * 4,
        retry_times=10,
        temperature=0.6
):
    """使用模板进行流式对话，支持重试机制"""
    llm_query = prompt.invoke(prompt_args).to_messages()[0].content
    for i in range(retry_times):
        try:
            stream_chunk = chat_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': llm_query
                    },
                ],
                # temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                stop=["<|im_end|>"],
                stream=True,
                timeout=timeout,
            )
            llm_reply = ''
            logger.info(f'stream start')
            reason_reply = ""
            result_reply = ""
            for chunk in stream_chunk:
                reasoning_delta = chunk.choices[0].delta.reasoning_content
                if reasoning_delta is not None and isinstance(reasoning_delta, str):
                    reason_reply += reasoning_delta
                delta = chunk.choices[0].delta.content
                if delta is not None and isinstance(delta, str):
                    result_reply += delta
            if reason_reply and len(result_reply) > 0:
                llm_reply = f"<think>{reason_reply}</think>{result_reply}"
            else:
                llm_reply = result_reply
            # for chunk in stream_chunk:
            #     delta = chunk.choices[0].delta.content
            #     if delta is not None:
            #         llm_reply += delta
            logger.info(f'strea end {llm_reply}')
            llm_reply = llm_reply.strip()
            return ChatResponse(llm_query, llm_reply, stream_chunk)
        except Exception as ace:
            traceback.print_exc()
            time.sleep(3)
            logger.info(f"连接错误 重试次数 {retry_times} 3s")


def chat_with_message_stream(
        chat_client,
        model_name,
        messages: list,
        timeout=5,
        max_tokens=1024 * 16,
        retry_times=3,
        temperature=0.7,
):
    """流式聊天，返回生成器"""
    for i in range(retry_times):
        try:
            stream_chunk = chat_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["<|im_end|>"],
                stream=True,
                timeout=timeout,
            )
            
            for chunk in stream_chunk:
                delta = chunk.choices[0].delta.content
                if delta is not None and isinstance(delta, str):
                    yield delta
            return
            
        except Exception as e:
            if i == retry_times - 1:  # 最后一次重试失败
                yield f"[错误: {str(e)}]"
                return
            time.sleep(1)
            

def chat_with_message(
        chat_client,
        model_name,
        messages: list,
        timeout=5,
        max_tokens=1024 * 16,
        retry_times=10,
        temperature=0,
        top_p=1,
):
    """使用消息列表进行流式对话，支持重试机制"""
    for i in range(retry_times):
        try:
            stream_chunk = chat_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=["<|im_end|>"],
                stream=True,
                timeout=timeout,
            )
            llm_reply = ''
            logger.info(f'stream start')
            reason_reply = ""
            result_reply = ""
            for chunk in stream_chunk:
                reasoning_delta = chunk.choices[0].delta.reasoning_content if hasattr(chunk.choices[0].delta,
                                                                                      'reasoning_content') else ""
                if reasoning_delta is not None and isinstance(reasoning_delta, str):
                    reason_reply += reasoning_delta
                delta = chunk.choices[0].delta.content
                if delta is not None and isinstance(delta, str):
                    result_reply += delta
            if reason_reply and len(result_reply) > 0:
                llm_reply = f"<think>{reason_reply}</think>{result_reply}"
            else:
                llm_reply = result_reply
            logger.info(f'strea end {llm_reply}')
            llm_reply = llm_reply.strip()
            return ChatResponse(dumps(messages, ensure_ascii=False, indent='  '), llm_reply, stream_chunk)
        except Exception as ace:
            traceback.print_exc()
            time.sleep(3)
            logger.info(f"连接错误 重试次数 {retry_times} 3s")


def is_decimal(s):
    """检查字符串是否为有效的小数"""
    try:
        float(s)  # 尝试转换为浮点数
        return True  # 检查是否包含小数点
    except ValueError:
        return False  # 转换失败则不是有效小数


def get_client_by_config(client_conf):
    """根据配置随机选择加权客户端"""
    client_conf = client_conf if isinstance(client_conf, list) else loads(client_conf, strict=False)
    clients = []
    weights = []
    for conf in client_conf:
        client_base_url = conf['base_url']
        client_api_key = conf.get("api_key", "11111111111111111")
        client_weight = float(conf.get("weight", "1"))
        if client_base_url:
            clients.append(get_client(client_base_url, client_api_key))
            weights.append(client_weight)
    return random.choices(clients, weights)[0]
