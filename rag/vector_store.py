import os
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"])
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})


    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的md5做去重
        """

        def check_md5_hex(md5_for_check: str) -> bool:
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 文件不存在创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding='utf-8').close()
                return False # md5没处理过
            
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding='utf-8') as f:
                for line in f:
                    # 去掉每一行开头和结尾的空白字符
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding='utf-8') as f:
                f.write(md5_for_check + '\n')

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_file_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_file_path:
            # 获取文件的md5
            md5_hex = get_file_md5_hex(path)

            # 文件不存在、不是文件或读取失败时无法计算 MD5，跳过该文件。
            if md5_hex is None:
                logger.warning(f"[加载知识库]{path}无法计算 MD5，跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在于知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效内容，跳过")
                    continue

                # 将内容存入向量库中
                batch_size = chroma_conf["batch_size"]

                for start in range(0, len(split_document), batch_size):
                    batch = split_document[start:start + batch_size]
                    self.vector_store.add_documents(batch)

                # 记录这个已经处理好的文件的md5值
                save_md5_hex(md5_hex)
                logger.info(f"[加载数据库]{path}内容加载成功")

            except Exception as e:
                # exc_info=True会记录详细的报错堆栈，如果为False仅记录信息报错本身
                logger.error(f"[加载数据库]{path}加载失败：{str(e)}", exc_info=True)

if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")

    for r in res:
        print(r.page_content)
        print("-" * 20)