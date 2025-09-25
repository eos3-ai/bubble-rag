import os


# 在导入时设置环境变量
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

MYSQL_URL = os.environ.get("MYSQL_URL", "mysql+pymysql://root:admin@172.16.10.105:23306/laiye_rag_community?charset=utf8")

MILVUS_SERVER_IP = os.getenv('MILVUS_SERVER_IP', '172.16.10.105')
MILVUS_SERVER_PORT = os.getenv('MILVUS_SERVER_PORT', '19530')
MILVUS_DB_NAME = os.getenv('MILVUS_DB_NAME', 'laiye_rag_test')

REDIS_HOST = os.getenv("REDIS_HOST", "172.16.10.105")
REDIS_PORT = os.getenv("REDIS_PORT", "26379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# 从环境变量获取配置（默认开发环境配置）
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_METHODS = os.getenv("ALLOWED_METHODS", "*").split(",")
ALLOWED_HEADERS = os.getenv("ALLOWED_HEADERS", "*").split(",")
STATIC_FILES_DIR = os.getenv("STATIC_FILES_DIR", "")

## 服务器base uri
SRV_BASE_RUI = os.getenv("SRV_BASE_URI", "/bubble_rag")

UPLOAD_FILE_PATH = os.getenv('UPLOAD_SAVE_PATH', '/data/code/bubble_rag/uploads')
## mineru服务器地址
MINERU_SERVER_URL = os.getenv("MINERU_SERVER_URL", "http://172.16.10.105:28100")

## 多模型配置
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://172.16.10.105:30010/v1/")
VLM_API_KEY = os.getenv("VLM_API_KEY", "1111")
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "qwen2.5-vl-instruct")

## 聊天模型配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://172.16.10.105:26014/v1/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "111")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen3-235B-A22B")

## 当前服务器ip
CURR_SERVER_IP = os.getenv("CURR_SERVER_IP", "172.16.10.105")

## 训练服务器端口
TRAINING_SERVER_PORT = int(os.getenv("TRAINING_SERVER_PORT", "8001"))
# 训练文件路径
TRAINING_FILES_PATH = os.getenv("TRAINING_FILES_PATH", "/app/data/files")
# 训练模型路径
TRAINING_MODELS_PATH = os.getenv("TRAINING_MODELS_PATH", "/app/data/models")
# 训练输出路径
TRAINING_OUTPUT_PATH = os.getenv("TRAINING_OUTPUT_PATH", "/app/data/output")
# modelscope缓存位置
TRAINING_CACHE = os.getenv("TRAINING_CACHE", "/app/cache")
TRAINING_SERVICE_ID = os.getenv("TRAINING_SERVICE_ID", "ds0o313111")