"""用户数据库模型"""

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, VARCHAR, DateTime, Column
from pydantic import ConfigDict


class UserDB(SQLModel, table=True):
    """用户数据库模型"""
    __tablename__ = "users"
    __table_args__ = {
        'comment': '用户管理表'
    }
    model_config = ConfigDict(protected_namespaces=())

    # 用户基础信息
    username: str = Field(
        max_length=64,
        sa_column=Column(VARCHAR(64), comment='用户名（唯一）', primary_key=True)
    )
    user_password: str = Field(
        sa_column=Column(VARCHAR(255), comment='用户密码哈希值', nullable=False)
    )
    user_role: str = Field(
        default="admin",
        sa_column=Column(VARCHAR(16), comment='用户角色：admin=管理员，user=普通用户', nullable=False)
    )
    display_name: Optional[str] = Field(
        sa_column=Column(VARCHAR(128), comment='显示名称', nullable=True)
    )

    # 时间戳
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime, comment='创建时间', nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime, comment='更新时间', nullable=False)
    )


class UserCreate(SQLModel):
    """创建用户请求模型"""
    username: str = Field(max_length=64, description="用户名")
    user_password: str = Field(description="密码")
    user_role: str = Field(default="user", description="用户角色")
    display_name: Optional[str] = Field(default=None, description="显示名称")


class UserResponse(SQLModel):
    """用户响应模型"""
    username: str
    user_role: str
    display_name: Optional[str]
    created_at: datetime
    updated_at: datetime


class UserUpdate(SQLModel):
    """更新用户请求模型"""
    user_password: Optional[str] = Field(default=None, description="新密码")
    user_role: Optional[str] = Field(default=None, description="用户角色")
    display_name: Optional[str] = Field(default=None, description="显示名称")