# 连接到 Redis
from walrus import Database

from bubble_rag.server_config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

redis_db = Database(host=REDIS_HOST, port=int(REDIS_PORT), db=0,
                    password=REDIS_PASSWORD if REDIS_PASSWORD and REDIS_PASSWORD.strip() else None)
