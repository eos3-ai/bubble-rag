import base64
import datetime
import hashlib
import os
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Tuple

from fastapi import UploadFile
from loguru import logger
from sqlmodel import select, text as sqltext, Session, delete as sqldelete

from bubble_rag.databases.memery_database import redis_db
from bubble_rag.entity.vectorial.documents import MilvusQueryRagDocuments
from bubble_rag.retrieving.vectorial.documents import add_rag_doc_list_by_mysqldata, semantic_merge_query, \
    list_documents
from bubble_rag.databases.relation_database import get_session
from bubble_rag.entity.relational.documents import DocFile, DocTask, RagDocuments
from bubble_rag.retrieving.relational.knowledge_base import get_knowledge_base
from bubble_rag.utils.parser.app.book import chunk as book_chunk
from bubble_rag.utils.parser.app.table import chunk as table_chunk
from bubble_rag.entity.query.documents import RagDocumentsParam
from bubble_rag.server_config import UPLOAD_FILE_PATH, MINERU_SERVER_URL, VLM_MODEL_NAME
from bubble_rag.utils.mineru_utils import mineru_parse_pdf_doc
from bubble_rag.utils.openai_utils import chat_with_message as openai_chat_with_message, def_vlm_client


async def add_doc_file(req_file: UploadFile) -> DocFile:
    """上传并保存文档文件到服务器"""
    filename = (req_file.filename)
    logger.info(f"filename: {filename}")
    fid = uuid.uuid4().hex.replace("-", '')
    # 保存文件到上传目录
    base_dir = os.path.join(UPLOAD_FILE_PATH, fid)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    file_path = os.path.join(base_dir, filename)
    logger.info(f"filepath {file_path}")
    # 以二进制写入模式打开文件
    with open(file_path, "wb") as f:
        # 读取上传文件的内容并写入到本地文件
        contents = await req_file.read()
        f.write(contents)

    if filename.endswith(".zip"):
        uncompress_dir = filename.split('.')[-2]
        uncompress_dir = os.path.join(base_dir, uncompress_dir)
        doc_file = DocFile(file_path=file_path, uncompress_path=uncompress_dir, file_size=get_file_size(file_path),
                           file_md5=get_file_md5(file_path))
        with next(get_session()) as session:
            session.add(doc_file)
            session.commit()
            # session.refresh(doc_file)
            doc_file = session.query(DocFile).get(doc_file.id)
            # doc_task = session.query(DocTask).get(doc_task.id)
            return doc_file
    else:
        uncompress_dir = filename.split('.')[-2]
        uncompress_dir = os.path.join(base_dir, uncompress_dir)
        doc_file = DocFile(file_path=file_path, uncompress_path=uncompress_dir, file_size=get_file_size(file_path),
                           file_md5=get_file_md5(file_path))
        with next(get_session()) as session:
            session.add(doc_file)
            session.commit()
            # session.refresh(doc_file)
            doc_file = session.query(DocFile).get(doc_file.id)
            return doc_file


def add_doc_file_by_path(rag_file: str, session: Session) -> DocFile:
    """通过本地文件路径添加文档文件"""
    filename = os.path.basename(rag_file)
    # logger.info(f"filename: {filename}")
    fid = uuid.uuid4().hex.replace("-", '')
    # 保存文件到上传目录
    base_dir = os.path.join(UPLOAD_FILE_PATH, fid)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    file_path = os.path.join(base_dir, filename)
    shutil.copy2(rag_file, file_path)
    logger.info(f"filepath {file_path}")
    uncompress_dir = filename.split('.')[-2]
    uncompress_dir = os.path.join(base_dir, uncompress_dir)
    doc_file = DocFile(file_path=file_path, uncompress_path=uncompress_dir, file_size=get_file_size(file_path),
                       file_md5=get_file_md5(file_path))
    session.add(doc_file)
    session.commit()
    # session.refresh(doc_file)
    doc_file = session.query(DocFile).get(doc_file.id)
    return doc_file


def get_file_size(file_path: str) -> Optional[int]:
    """
    获取文件大小（字节数）

    :param file_path: 文件路径
    :return: 文件大小（字节），如果文件不存在或出错则返回None
    """
    try:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return None
        if not os.path.isfile(file_path):
            print(f"错误: 不是一个文件 - {file_path}")
            return None

        return os.path.getsize(file_path)
    except Exception as e:
        print(f"获取文件大小出错: {str(e)}")
        return None


def get_file_md5(file_path: str, chunk_size: int = 4096) -> Optional[str]:
    """
    计算文件的MD5哈希值

    :param file_path: 文件路径
    :param chunk_size: 读取文件的块大小，默认4096字节
    :return: 文件的MD5值（32位小写字符串），如果文件不存在或出错则返回None
    """
    try:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return None
        if not os.path.isfile(file_path):
            print(f"错误: 不是一个文件 - {file_path}")
            return None

        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # 分块读取大文件，避免占用过多内存
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)

        return md5_hash.hexdigest()
    except Exception as e:
        print(f"计算文件MD5出错: {str(e)}")
        return None


def semantic_query_documents(rag_doc: RagDocumentsParam, ) -> List[MilvusQueryRagDocuments]:
    """根据文档内容进行语义查询搜索"""
    with next(get_session()) as session:
        knowledge_base = get_knowledge_base(rag_doc.doc_knowledge_base_id)
        if rag_doc.doc_content:
            milvus_rag_doc_list = semantic_merge_query(
                rag_doc.doc_content,
                knowledge_base,
                rag_doc.limit_result,
                rag_doc.limit_result,
            )
            # rag_doc_list = [session.get(RagDocuments, mdoc.id) for mdoc in milvus_rag_doc_list]
            rag_doc_list = milvus_rag_doc_list
            return rag_doc_list
        else:
            return list_documents(knowledge_base, rag_doc.limit_result)


def process_doc_task(doc_task: DocTask):
    """处理单个文档任务，根据文件类型调用相应的处理函数"""
    try:
        lock = redis_db.lock(f"doc_task_{doc_task.id}")
        with lock:
            with next(get_session()) as session:
                # curr_doc_task = session.exec(select(DocTask).where(DocTask.id == doc_task.id).with_for_update()).one()
                curr_doc_task = session.exec(select(DocTask).where(DocTask.id == doc_task.id)).one()
                if curr_doc_task.success_file >= curr_doc_task.total_file:
                    return curr_doc_task
                session.exec(sqldelete(RagDocuments).where(RagDocuments.doc_task_id == curr_doc_task.id))
                doc_file = session.get(DocFile, doc_task.file_id)
                file_name = os.path.basename(doc_file.file_path)
                file_suffix = file_name.split('.')[-1].lower()
                file_suffix = file_suffix.strip()
                # session.refresh(doc_file)
                doc_file = session.query(DocFile).get(doc_file.id)
            logger.info(f"解析文件 task {doc_task.id} {file_name}")
            if file_suffix in ['doc', 'docx', ]:
                process_word_doc_task(doc_task)
            elif file_suffix in ['xls', 'xlsx', 'csv', ]:
                process_excel_doc_task(doc_task)
            elif file_suffix in ['txt', ]:
                process_txt_doc_task(doc_task)
            elif file_suffix in ['md', ]:
                process_markdown_doc_task(doc_task)
            elif file_suffix in ['pdf', ]:
                logger.info("===================== pdf parse =====================")
                process_pdf_doc_task(doc_task)
            elif file_suffix in ['png', 'jpeg', 'jpg', ]:
                process_image_doc_task(doc_task)
    except Exception as e:
        with next(get_session()) as session:
            doc_task = session.get(DocTask, doc_task.id)
            doc_task.update_time = datetime.datetime.now()
            session.commit()
            traceback.print_exc()


def add_documents_by_doc_task(doc_task: DocTask, doc_content: str) -> List[RagDocuments]:
    """根据文档任务和内容创建文档片段并添加到向量数据库"""
    chunk_size = doc_task.chunk_size
    with next(get_session()) as session:
        doc_file = session.get(DocFile, doc_task.file_id)
    file_name_split = os.path.basename(doc_file.file_path).split(".")
    title = ".".join(file_name_split[0:len(file_name_split) - 1])
    rag_docs = []
    logger.info(
        "========================================= add_documents_by_doc_task =========================================")
    logger.info(f"============ {len(doc_content)} chunk_size {chunk_size} ")
    logger.info(
        "========================================= add_documents_by_doc_task =========================================")
    for i in range(0, len(doc_content), chunk_size):
        curr_ctn = doc_content[i:i + chunk_size]
        curr_title = title
        knowledge_base = get_knowledge_base(doc_task.doc_knowledge_base_id)
        logger.info("================================== add_documents_by_doc_task ==================================")
        logger.info(curr_title)
        logger.info(curr_ctn)
        logger.info("================================== add_documents_by_doc_task ==================================")
        rag_docs.append(RagDocuments(
            doc_title=curr_title,
            doc_content=f"{curr_title}\n\n{curr_ctn}",
            doc_file_id=doc_file.id,
            doc_task_id=doc_task.id,
            doc_file_name=os.path.basename(doc_file.file_path),
            doc_knowledge_base_id=doc_task.doc_knowledge_base_id,
            embedding_model_id=knowledge_base.embedding_model_id,
        ))
    with next(get_session()) as session:
        if len(rag_docs) > 0:
            session.add_all(rag_docs)
            knowledge_base = get_knowledge_base(doc_task.doc_knowledge_base_id)
            add_rag_doc_list_by_mysqldata(rag_doc_list=rag_docs,
                                          knowledge_base=knowledge_base)
        session.commit()
    return rag_docs


def process_word_doc_task(doc_task: DocTask):
    """处理Word文档任务，解析并分块存储文档内容"""
    with next(get_session()) as session:
        doc_file = session.get(DocFile, doc_task.file_id)
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.total_file = 1
        doc_task.success_file = 0
        doc_task.remaining_file = 1
        doc_task.curr_file_progress = 10
        doc_task.curr_filename = os.path.basename(doc_file.file_path)
        session.commit()
        doc_file = session.query(DocFile).get(doc_task.file_id)
        doc_task = session.query(DocTask).get(doc_task.id)
        # session.refresh(doc_task)
        # session.refresh(doc_file)
    logger.info("================================ book_chunk_ctn ================================")
    book_chunk_ctn = book_chunk(doc_file.file_path)
    logger.info(book_chunk_ctn)
    logger.info("================================ book_chunk_ctn ================================")
    book_ctn_list = [str(e.get("content_with_weight")) for e in book_chunk_ctn]
    book_word_ctn = "\n".join(book_ctn_list)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.curr_file_progress = 60
        session.commit()
        doc_task = session.query(DocTask).get(doc_task.id)
    add_documents_by_doc_task(doc_task=doc_task, doc_content=book_word_ctn)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.success_file = 1
        doc_task.curr_file_progress = 100
        session.commit()
        doc_task = session.query(DocTask).get(doc_task.id)
    clear_folder(Path(doc_file.file_path).parent)


def recognition_excel_file(file_path) -> List[str]:
    """识别并解析Excel文件内容"""
    with open(file_path, 'br', ) as f:
        table_chunk_ctn = table_chunk(os.path.basename(file_path), binary=f.read())
    table_ctn_list = [str(e.get("content_with_weight")) for e in table_chunk_ctn]
    return table_ctn_list


def process_excel_doc_task(doc_task: DocTask):
    """处理Excel文档任务，解析表格数据并分块存储"""
    with next(get_session()) as session:
        doc_file = session.query(DocFile).get(doc_task.file_id)
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.total_file = 1
        doc_task.success_file = 0
        doc_task.remaining_file = 1
        doc_task.curr_file_progress = 10
        doc_task.curr_filename = os.path.basename(doc_file.file_path)
        session.commit()
        doc_task = session.query(DocTask).get(doc_task.id)
    table_ctn_list = recognition_excel_file(doc_file.file_path)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.curr_file_progress = 60
        session.commit()
        doc_task = session.query(DocTask).get(doc_task.id)
    for table_ctn in table_ctn_list:
        add_documents_by_doc_task(doc_task=doc_task, doc_content=table_ctn)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.success_file = 1
        doc_task.curr_file_progress = 100
        session.commit()
        doc_task = session.query(DocTask).get(doc_task.id)
    clear_folder(Path(doc_file.file_path).parent)


def process_txt_doc_task(doc_task: DocTask):
    """处理文本文档任务，读取txt文件内容并分块存储"""
    with next(get_session()) as session:
        doc_file = session.query(DocFile).get(doc_task.file_id)
        # doc_file = session.get(DocFile, doc_task.file_id)
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.total_file = 1
        doc_task.success_file = 0
        doc_task.remaining_file = 1
        doc_task.curr_file_progress = 10
        doc_task.curr_filename = os.path.basename(doc_file.file_path)
        session.commit()
        # session.refresh(doc_task)
        doc_task = session.query(DocTask).get(doc_task.id)
    with open(doc_file.file_path, 'r+', encoding='utf-8') as file:
        book_chunk_ctn = file.read()
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.curr_file_progress = 60
        session.commit()
        # session.refresh(doc_task)
        # doc_file = session.query(DocFile).get(doc_task.file_id)
        doc_task = session.query(DocTask).get(doc_task.id)
    add_documents_by_doc_task(doc_task=doc_task, doc_content=book_chunk_ctn)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.success_file = 1
        doc_task.curr_file_progress = 100
        session.commit()
        # session.refresh(doc_task)
        # doc_file = session.query(DocFile).get(doc_task.file_id)
        doc_task = session.query(DocTask).get(doc_task.id)
    clear_folder(Path(doc_file.file_path).parent)


def process_markdown_doc_task(doc_task: DocTask):
    """处理Markdown文档任务，读取md文件内容并分块存储"""
    with next(get_session()) as session:
        doc_file = session.get(DocFile, doc_task.file_id)
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.total_file = 1
        doc_task.success_file = 0
        doc_task.remaining_file = 1
        doc_task.curr_file_progress = 10
        doc_task.curr_filename = os.path.basename(doc_file.file_path)
        session.commit()
        # session.refresh(doc_task)
        doc_file = session.query(DocFile).get(doc_task.file_id)
        doc_task = session.query(DocTask).get(doc_task.id)
    try:
        with open(doc_file.file_path, 'r+', encoding='utf-8') as file:
            book_chunk_ctn = file.read()
    except Exception as e:
        traceback.print_exc()
        book_chunk_ctn = ""
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.curr_file_progress = 60
        session.commit()
        # session.refresh(doc_task)
        doc_task = session.query(DocTask).get(doc_task.id)
    add_documents_by_doc_task(doc_task=doc_task, doc_content=book_chunk_ctn)
    with next(get_session()) as session:
        doc_file = session.get(DocFile, doc_task.file_id)
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.success_file = 1
        doc_task.curr_file_progress = 100
        session.commit()
        session.refresh(doc_task)
        clear_folder(Path(doc_file.file_path).parent)


def process_image_doc_task(doc_task: DocTask):
    with next(get_session()) as session:
        doc_file = session.get(DocFile, doc_task.file_id)
        file_path = doc_file.file_path
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.total_file = 1
        doc_task.success_file = 0
        doc_task.remaining_file = 1
        doc_task.curr_file_progress = 10
        doc_task.curr_filename = os.path.basename(file_path)
        session.commit()
        doc_task = session.query(DocTask).get(doc_task.id)
        # session.refresh(doc_task)
    ocr_txt = recognition_image_content(file_path)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.curr_file_progress = 60
        session.commit()
        # session.refresh(doc_task)
        doc_task = session.query(DocTask).get(doc_task.id)
    add_documents_by_doc_task(doc_task=doc_task, doc_content=ocr_txt)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.success_file = 1
        doc_task.curr_file_progress = 100
        doc_task.update_time = datetime.datetime.now()
        session.commit()
        session.refresh(doc_task)
        doc_task = session.query(DocTask).get(doc_task.id)
        # clear_folder(Path(doc_file.file_path).parent)


def recognition_pdf_file(file_path, uncompress_path, ) -> str:
    parse_pdf_result = mineru_parse_pdf_doc(
        file_path=file_path,
        output_dir=uncompress_path,
        backend="vlm-sglang-client",
        server_url=MINERU_SERVER_URL)
    pdf_md_path = parse_pdf_result["dump_md"]
    pdf_content = ""
    with open(pdf_md_path, 'r', encoding='utf-8') as f:
        pdf_content = f.read()
    return pdf_content


def process_pdf_doc_task(doc_task: DocTask):
    logger.info("===================== 解析pdf文件 =====================")
    with next(get_session()) as session:
        doc_file = session.get(DocFile, doc_task.file_id)
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.total_file = 1
        doc_task.success_file = 0
        doc_task.remaining_file = 1
        doc_task.curr_file_progress = 10
        doc_task.curr_filename = os.path.basename(doc_file.file_path)
        session.commit()
        doc_file = session.query(DocFile).get(doc_task.file_id)
        doc_task = session.query(DocTask).get(doc_task.id)
    pdf_content = recognition_pdf_file(doc_file.file_path, doc_file.uncompress_path)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.curr_file_progress = 70
        session.commit()
        doc_task = session.query(DocTask).get(doc_task.id)
    add_documents_by_doc_task(doc_task=doc_task, doc_content=pdf_content)
    with next(get_session()) as session:
        doc_task = session.get(DocTask, doc_task.id)
        doc_task.success_file = 1
        doc_task.curr_file_progress = 100
        doc_task.update_time = datetime.datetime.now()
        session.commit()
        session.refresh(doc_task)
        clear_folder(Path(doc_file.file_path).parent)
    logger.info("===================== 解析pdf文件 =====================")


def recognition_image_content(file_path) -> str:
    file_base64, file_type = file_to_base64(file_path)
    ocr_resp = openai_chat_with_message(
        chat_client=def_vlm_client,
        model_name=VLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """你是一个强大的多模态人工智能系统，特别擅长精确分析和理解图像内容。当前核心任务是从用户提供的图像中识别并提取所有可见的文字信息。

    **能力要求：**
    1.  **OCR 专家：** 以极高的准确度识别图像中的印刷体和清晰的手写体文字。
    2.  **内容完整性：** 提取图像中出现的**所有**文字内容，包括但不限于标题、正文、标签、说明、注释、数字、符号（如 $, €, %, &, +, -, =, @）、标点符号（., ,, ;, :, !, ?, ", ', (, ), [, ], {, }）以及数学公式、化学式中的特殊字符。
    3.  **格式保留：** 尽可能保留原始文本的**基本格式**：
        *   识别明显的**换行符**和**段落分隔**。
        *   识别**项目符号列表**（如 •, -, *）或**编号列表**并保留其结构。
        *   识别**表格结构**（如果清晰可辨），尽量以表格形式或分隔符（如 | ）呈现行列关系。
        *   识别**多列文本**布局并保持列内文本的顺序。
    4.  **非文本处理：** 忽略纯粹的装饰性图形、Logo（除非包含文字）、图片、背景图案等非文本元素。专注于文字内容本身。
    5.  **清晰输出：** 将提取到的所有文字内容以清晰、连贯、易读的纯文本格式输出。不要在输出中添加任何解释、分析或额外的评论（除非用户特别要求）。只输出图像中包含的文字本身。
    6.  **不确定性处理：** 对于模糊不清、遮挡严重或无法确定的内容，在相应位置用占位符 `[?]` 或 `[无法识别]` 明确标注，而不是猜测或省略。
    7.  **语言处理：** 识别文本的语言（如果混合，按原文输出），不进行翻译（除非用户额外要求）。

    **请严格遵循以上要求执行文字识别任务。**
    """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
    请分析我提供的这张图片，并执行以下任务：

    **核心任务：精确提取图片中的所有文字**

    **具体要求：**
    1.  **完整输出：** 提取图片中出现的**每一个字、字母、数字、符号和标点**。不要遗漏任何可见的文字信息。
    2.  **保持结构与顺序：**
        *   按照文本在图片中出现的**自然阅读顺序**（通常从左到右，从上到下）输出。
        *   保留明显的**换行**和**段落分隔**。用空行表示段落分隔。
        *   如果存在**项目符号列表**或**编号列表**，请保留项目符号/编号和缩进结构。
        *   如果存在**表格**，请尽力识别行列结构。输出时可以使用 `|` 分隔列，用 `---` 分隔表头（如果适用），或者清晰地标注行和列。如果表格结构复杂，优先保证文本内容的完整性和顺序。
        *   如果文本是**多列**布局（如报纸、杂志），请按列提取，一列结束后再提取下一列。
    3.  **特殊内容处理：**
        *   **数学公式/化学式：** 准确识别并提取公式中的所有符号、字母、数字、上下标（用 `_` 表示下标，`^` 表示上标，如 H_2O, E=mc^2）等。保持公式的逻辑结构。
        *   **代码片段：** 按原样提取，保留缩进和特殊符号。
        *   **清晰手写体：** 如果手写清晰可辨，请尽力识别并提取。
    4.  **忽略内容：** 完全忽略图片、图标、Logo（除非它们内部含有文字）、纯装饰性线条或图案。只关注文字元素。
    5.  **输出格式：**
        *   输出**纯文本**。
        *   不要添加任何额外的标题（如 “提取的文字：”）、总结或解释。
        *   不确定的内容用 `[?]` 或 `[无法识别]` 明确标注。
        *   如果图片中完全没有文字，请输出：`[未检测到文字内容]`。

    **现在请分析以下图片：**
    """},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{file_type};base64,{file_base64}"}
                    }
                ]
            },
        ],
    )
    ocr_txt = remove_think_tag(ocr_resp.resp_text)
    return ocr_txt


def clear_folder(folder_path: str | Path) -> None:
    """
    清空指定文件夹中的所有文件和子文件夹，但保留文件夹本身

    参数:
        folder_path: 要清空的文件夹路径
    """
    folder_path = Path(folder_path)

    # 确保文件夹存在
    if not folder_path.exists() or not folder_path.is_dir():
        # raise ValueError(f"路径不存在或不是有效的文件夹: {folder_path}")
        return False

    # 遍历文件夹中的所有内容
    for item in folder_path.iterdir():
        try:
            if item.is_file():
                item.unlink()  # 删除文件
            elif item.is_dir():
                shutil.rmtree(item)  # 递归删除子文件夹
        except Exception as e:
            print(f"无法删除 {item}: {e}")


def file_to_base64(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    读取文件并转换为Base64编码字符串

    :param file_path: 文件路径
    :return: 元组 (base64编码字符串, MIME类型)，出错时返回 (None, None)
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return None, None

        # 检查是否为文件
        if not os.path.isfile(file_path):
            print(f"错误: 不是有效的文件 - {file_path}")
            return None, None

        # 获取文件MIME类型（简单判断，更精确可使用python-magic库）
        file_ext = os.path.splitext(file_path)[-1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        mime_type = mime_types.get(file_ext, 'application/octet-stream')

        # 读取文件并转换为base64
        with open(file_path, 'rb') as file:
            file_content = file.read()
            base64_str = base64.b64encode(file_content).decode('utf-8')

        return base64_str, mime_type

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None, None


def process_all_doc_tasks():
    """批量处理所有待处理的文档任务"""
    try:
        with next(get_session()) as session:
            sql = """
            select dt.*
            from doc_task dt
            where dt.success_file < dt.total_file
            order by timestampdiff(second, dt.create_time, dt.update_time) asc
            limit 10
            """
            doc_task_result = session.exec(statement=sqltext(sql), params={}).all()
            unprocess_doc_tasks = [DocTask(**dtr._asdict()) for dtr in doc_task_result]
            for doc_task in unprocess_doc_tasks:
                process_doc_task(doc_task)
    except Exception as e:
        traceback.print_exc()
