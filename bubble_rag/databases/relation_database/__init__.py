import os
from typing import Annotated, Generator, Iterator

from fastapi import Depends
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy.orm import sessionmaker
import logging
from pydantic import BaseModel
from bubble_rag.server_config import MYSQL_URL

logging.basicConfig(level=logging.WARNING)
# logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG)
engine = create_engine(MYSQL_URL, max_overflow=8, echo=False, pool_recycle=180)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def get_fastapi_session() -> Iterator[Session]:
    yield Session(engine, autocommit=True)


def get_engine():
    """获取数据库引擎"""
    return engine


SessionDep = Annotated[Session, Depends(get_session)]
SqlAlchemySession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def update_model_from_dict(model: BaseModel, data: dict) -> BaseModel:
    """根据字典更新 SQLModel 对象的属性"""
    for key, value in data.items():
        if hasattr(model, key):
            setattr(model, key, value)
    return model
