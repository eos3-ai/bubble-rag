from enum import Enum

from pydantic import BaseModel, Field, conint, confloat, ConfigDict
from typing import List, Optional, Union, Dict


class RagChatParam(BaseModel):
    question: str = ""
    doc_knowledge_base_id: str = ""
    limit_result: Optional[int] = Field(default=10, le=20)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)
    max_tokens: Optional[int] = Field(default=2048, le=4096)
    stream: Optional[bool] = False


# def RagQuery(BaseModel):
#     query: str
#
#
# class RequestModel(BaseModel):
#     name: str
#     age: int
#
#
# class AnimalDrug(BaseModel):
#     desc: str
#
#
# class AnimalMedicationRequire(BaseModel):
#     target_animals: Optional[str] = None
#     indications: Optional[str] = None
#     medication: Optional[str] = None
#     drug_name: Optional[str] = None
#     user_require: Optional[str] = None
#     pet_age: Optional[float] = None
#     pet_weight: Optional[float] = None
#     unwell: Optional[bool] = False
#
#
# class MessageContent(BaseModel):
#     type: str = Field(..., description="")
#     text: str = Field(..., description="")
#
#
# class Message(BaseModel):
#     # model_config = ConfigDict(arbitrary_types_allowed=True)
#     """聊天消息结构"""
#     role: str = Field(..., description="角色，通常是 'system', 'user', 'assistant'")
#     content: Union[str | List[MessageContent]] = Field(..., description="消息内容")
#     # name: Optional[str] = Field(None, description="消息名称，可选")
#
#
# class UserHistoricalChat(BaseModel):
#     messages: List[Message] = Field(..., description="消息列表")
#
#
# class RagOrDrugFeature(BaseModel):
#     messages: List[Message] = Field(description="消息列表", )
#     # feature: int = Field(description="功能类型 0 rag 1 药品推荐")
#     model: Optional[str] = Field(description="模型名称", default='zkjy_llm')
#     temperature: Optional[float] = Field(description="", default=0.7)
#     top_p: Optional[float] = Field(default=1, )
#     top_k: Optional[float] = Field(default=1, )
#     max_tokens: Optional[int] = Field(description="", default=1000)
#     presence_penalty: Optional[float] = Field(description="", default=0)
#     frequency_penalty: Optional[float] = Field(description="", default=0)
#     search_ctrl: Optional[int] = Field(default=0)
#     rec_ctrl: Optional[int] = Field(default=0)
#     user: Optional[str] = Field(default="")
#     additional_kwargs: Optional[str] = Field(default="")
#     syspmt_user_msg: Optional[int] = Field(default=1)
#
#
# class AnimalProductParam(BaseModel):
#     ids: Optional[list[str]] = None
#     # 产品的商品名，如喵益哆、宠益星等
#     product_name: Optional[str] = ""
#     # 产品的通用名，对产品的通用描述性名称
#     generic_name: Optional[str] = ""
#     # 产品品类，如疫苗、驱虫药、药品、保健品等
#     category: Optional[str] = ""
#     # 产品所起到的作用，适用病症
#     indications: Optional[str] = ""
#     # 商品适用的动物种类
#     target_animals: Optional[str] = ""
#     # 批准文号
#     approval_number: Optional[str] = ""
#     # 产品所属品牌，当前主要为惠中动保
#     brand: Optional[str] = ""
#     # 生产企业
#     manufacturing_enterprise: Optional[str] = ""
#     # 搜索条件 0 或者条件 或者条件
#     search_type: Optional[int] = 1
#     page_size: Optional[int] = 20
#     page_num: Optional[int] = 1
#
#
# class AppConfigParam(BaseModel):
#     conf_id: Optional[str] = None
#     conf_name: Optional[str] = None
#     conf_value: Optional[str] = None
#
#
# class MessageRole(Enum):
#     USER = "user"
#     SYSTEM = "system"
#     ASSISTANT = "assistant"
#
#
# class RagOrDrugFeatureCtrl(Enum):
#     AUTO_SEARCH = 0
#     OPEN_SEARCH = 1
#     CLOSE_SEARCH = 2
#     AUTO_RECOMMEND = 0
#     OPEN_RECOMMEND = 1
#     CLOSE_RECOMMEND = 2
