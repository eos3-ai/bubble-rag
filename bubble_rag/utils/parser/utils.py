from bubble_rag.utils.nlp import find_codec


def get_text(fnm: str, binary=None) -> str:
    """从文件名或二进制数据中读取文本内容"""
    txt = ""
    if binary:
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
    else:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                txt += line
    return txt
