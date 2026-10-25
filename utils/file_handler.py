import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utils.file_type_detector import PhysicalFileTypeDetector



# md5去重是将字符串生成一个32位十六进制字符串，不同内容md5不同记录，相同内容md5相同不记录
# md5也可以进行文件去重，比较文件内容是否相同
def get_file_md5_hex(filepath: str) -> str | None: # 获取文件的md5的十六进制值
    # 不存在
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    # 不是单独文件
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()

    # 分片加载计算md5，防止单独文件过大，占用内存少
    chunk_size = 4096 # 4kb分片，避免文件过大爆内存
    try:
        # rb 以二进制读取
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                # 把当前读取到的这块字节数据加入 MD5 的计算过程
                md5_obj.update(chunk)
                """
                chunk = f.read(chunk_size)
                while chunk:
                    md5_obj.update(chunk)
                    chunk = f.read(chunk_size)
                """
            # 获取当前已经计算出的 MD5 摘要，并转换成十六进制字符串
            md5_hex = md5_obj.hexdigest()
            return md5_hex
        
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None 

    


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]): # 返回文件夹内的文件列表（允许的文件后缀）
    files = []

    # 不是文件夹
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return ()

    normalized_types = {
        type_name.lower().lstrip(".") for type_name in allowed_types
    }

    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if not os.path.isfile(file_path):
            continue

        suffix = os.path.splitext(f)[1].lower().lstrip(".")
        detected_type = PhysicalFileTypeDetector.detect(file_path).value
        if suffix in normalized_types or detected_type in normalized_types:
            files.append(file_path)

    # 记录好以后不允许修改
    return tuple(files)


def pdf_loader(filepath: str, passwd = None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()




def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath).load()
