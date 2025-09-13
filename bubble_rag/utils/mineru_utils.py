import copy
import json
import os
from pathlib import Path

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from os import path as os_path


def mineru_do_parse(
        output_dir,  # Output directory for storing parsing results
        pdf_file_name: str,  # List of PDF file names to be parsed
        pdf_bytes: bytes,  # List of PDF bytes to be parsed
        backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
        server_url=None,  # Server URL for vlm-sglang-client backend
        f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
        f_dump_md=True,  # Whether to dump markdown files
        f_dump_middle_json=True,  # Whether to dump middle JSON files
        f_dump_model_output=True,  # Whether to dump model output files
        f_dump_orig_pdf=True,  # Whether to dump original PDF files
        f_dump_content_list=True,  # Whether to dump content list files
        f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    if backend.startswith("vlm-"):
        backend = backend[4:]

    f_draw_span_bbox = False
    parse_method = "vlm"
    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend,
                                                server_url=server_url)

    pdf_info = middle_json["pdf_info"]

    result_path = {}

    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )
        result_path['orig_pdf'] = os_path.join(local_md_dir, f"{pdf_file_name}_origin.pdf")

    if f_dump_md:
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )
        result_path['dump_md'] = os_path.join(local_md_dir, f"{pdf_file_name}.md")

    if f_dump_content_list:
        image_dir = str(os.path.basename(local_image_dir))
        content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )
        result_path['content_list_json'] = os_path.join(local_md_dir, f"{pdf_file_name}_content_list.json")

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )
        result_path['middle_json'] = os_path.join(local_md_dir, f"{pdf_file_name}_middle.json")

    if f_dump_model_output:
        model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
        md_writer.write_string(
            f"{pdf_file_name}_model_output.txt",
            model_output,
        )
        result_path['model_output'] = os_path.join(local_md_dir, f"{pdf_file_name}_model_output.txt")

    logger.info(f"local output dir is {local_md_dir}")
    return result_path


def mineru_parse_pdf_doc(
        file_path: Path,
        output_dir,
        backend="pipeline",
        server_url=None,
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """
    try:
        file_name = str(Path(file_path).stem)
        pdf_bytes = read_fn(file_path)
        return mineru_do_parse(
            output_dir=output_dir,
            pdf_file_name=file_name,
            pdf_bytes=pdf_bytes,
            backend=backend,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)
