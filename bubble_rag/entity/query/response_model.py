from enum import Enum
from typing import Any

from pydantic import BaseModel


class SrvResult(BaseModel):
    msg: str
    code: int
    data: object = None


class SteamStageResp(Enum):
    GENERATING = 'generating'
    SEARCHING = 'searching'

# 定义分页结果返回模型
class PageResult(BaseModel):
    items: list[object]
    total: int
    page: int
    page_size: int
    total_pages: int