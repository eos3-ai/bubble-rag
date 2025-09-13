import os
import sys

from pymilvus import connections, MilvusClient, DataType, Function, FunctionType, db
from loguru import logger

from bubble_rag.server_config import MILVUS_SERVER_IP, MILVUS_SERVER_PORT, MILVUS_DB_NAME


logger.info(f'初始化milvus MILVUS_SERVER_IP {MILVUS_SERVER_IP} MILVUS_SERVER_PORT {MILVUS_SERVER_PORT}')
conn = connections.connect(host=MILVUS_SERVER_IP, port=int(MILVUS_SERVER_PORT), db_name=MILVUS_DB_NAME)
milvus_client = MilvusClient(uri=f"http://{MILVUS_SERVER_IP}:{MILVUS_SERVER_PORT}", token='', db_name=MILVUS_DB_NAME)
logger.info('初始化milvus success')

