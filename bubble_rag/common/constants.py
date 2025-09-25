"""系统常量定义"""
from enum import Enum
from enum import IntEnum
from strenum import StrEnum

NAME_LENGTH_LIMIT = 2 ** 10

IMG_BASE64_PREFIX = 'data:image/png;base64,'

SERVICE_CONF = "service_conf.yaml"

API_VERSION = "v1"
RAG_FLOW_SERVICE_NAME = "ragflow"
REQUEST_WAIT_SEC = 2
REQUEST_MAX_WAIT_SEC = 300

DATASET_NAME_LIMIT = 128

# 模型类型常量
class ModelType:
    """模型类型枚举"""
    EMBEDDING = 0  # 向量模型
    RERANK = 1     # 重排序模型
    LLM = 2        # 大语言模型


# 文件处理常量
class FileConstants:
    """文件处理相关常量"""
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.doc', '.docx', '.pdf', '.html', '.htm',
        '.xls', '.xlsx', '.csv', '.ppt', '.pptx'
    }
    
    # 文件大小限制
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_FILE_SIZE = 1  # 1 byte
    
    # 分块大小限制
    MAX_CHUNK_SIZE = 8192
    MIN_CHUNK_SIZE = 1
    DEFAULT_CHUNK_SIZE = 512


class FileType(StrEnum):
    PDF = 'pdf'
    DOC = 'doc'
    VISUAL = 'visual'
    AURAL = 'aural'
    VIRTUAL = 'virtual'
    FOLDER = 'folder'
    OTHER = "other"


class StatusEnum(Enum):
    VALID = "1"
    INVALID = "0"


class UserTenantRole(StrEnum):
    OWNER = 'owner'
    ADMIN = 'admin'
    NORMAL = 'normal'
    INVITE = 'invite'


class TenantPermission(StrEnum):
    ME = 'me'
    TEAM = 'team'


class SerializedType(IntEnum):
    PICKLE = 1
    JSON = 2

VALID_FILE_TYPES = {FileType.PDF, FileType.DOC, FileType.VISUAL, FileType.AURAL, FileType.VIRTUAL, FileType.FOLDER, FileType.OTHER}

class LLMType(StrEnum):
    CHAT = 'chat'
    EMBEDDING = 'embedding'
    SPEECH2TEXT = 'speech2text'
    IMAGE2TEXT = 'image2text'
    RERANK = 'rerank'
    TTS    = 'tts'


class ChatStyle(StrEnum):
    CREATIVE = 'Creative'
    PRECISE = 'Precise'
    EVENLY = 'Evenly'
    CUSTOM = 'Custom'


class TaskStatus(StrEnum):
    UNSTART = "0"
    RUNNING = "1"
    CANCEL = "2"
    DONE = "3"
    FAIL = "4"

VALID_TASK_STATUS     = {TaskStatus.UNSTART, TaskStatus.RUNNING, TaskStatus.CANCEL, TaskStatus.DONE, TaskStatus.FAIL}

class ParserType(StrEnum):
    PRESENTATION = "presentation"
    LAWS = "laws"
    MANUAL = "manual"
    PAPER = "paper"
    RESUME = "resume"
    BOOK = "book"
    QA = "qa"
    TABLE = "table"
    NAIVE = "naive"
    PICTURE = "picture"
    ONE = "one"
    AUDIO = "audio"
    EMAIL = "email"
    KG = "knowledge_graph"
    TAG = "tag"


class FileSource(StrEnum):
    LOCAL = ""
    KNOWLEDGEBASE = "knowledgebase"
    S3 = "s3"


class CanvasType(StrEnum):
    ChatBot = "chatbot"
    DocBot = "docbot"

KNOWLEDGEBASE_FOLDER_NAME=".knowledgebase"

