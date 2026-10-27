import os
import sys
from pathlib import Path
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.file_handler import listdir_with_allowed_type, get_file_md5_hex
from utils.document_reader import DocumentReader
from rag.semantic_parser import SemanticParserService
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

        self.document_reader = DocumentReader(
            chroma_conf.get("semantic_type_mapping", {})
        )
        self.semantic_parser = SemanticParserService(
            fallback_chunk_size=chroma_conf["chunk_size"],
            fallback_chunk_overlap=chroma_conf["chunk_overlap"],
            max_chunk_size=chroma_conf.get("semantic_max_chunk_size", 1000),
            separators=chroma_conf["separators"],
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    @staticmethod
    def _prepare_chunk_ids(
        chunks: list[Document],
        source_path: str,
        file_md5: str,
    ) -> list[str]:
        """校验 chunk 内容并为没有 ID 的兜底分片生成稳定 ID。"""
        source_name = os.path.basename(source_path)
        chunk_ids: list[str] = []

        for index, chunk in enumerate(chunks):
            if not chunk.page_content.strip():
                raise ValueError(f"文件{source_path}存在空 chunk，无法进行 Embedding")

            chunk_id = chunk.metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                chunk_id = f"{source_name}_{file_md5[:12]}_{index}"
                chunk.metadata["chunk_id"] = chunk_id

            chunk_ids.append(chunk_id)

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f"文件{source_path}生成了重复的 chunk_id，停止写入")

        return chunk_ids

    def _embed_and_store(
        self,
        chunks: list[Document],
        chunk_ids: list[str],
        source_path: str,
    ) -> None:
        """分批调用 Embedding 并将向量写入 Chroma。"""
        batch_size = chroma_conf["batch_size"]

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            batch_ids = chunk_ids[start:start + batch_size]

            # Chroma 会使用初始化时传入的 DashScopeEmbeddings，
            # 自动调用 embed_documents，再将向量和原文写入向量库。
            self.vector_store.add_documents(batch, ids=batch_ids)
            logger.info(
                f"[Embedding] {source_path}已处理第{start + 1}-"
                f"{start + len(batch)}个 chunk"
            )


    def load_document(self, strict: bool = False):
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

        allowed_file_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        if strict and not allowed_file_path:
            raise RuntimeError("知识目录中没有可导入的文件，请检查数据挂载")

        for path in allowed_file_path:
            # 获取文件的md5
            md5_hex = get_file_md5_hex(path)

            # 文件不存在、不是文件或读取失败时无法计算 MD5，跳过该文件。
            if md5_hex is None:
                if strict:
                    raise RuntimeError(f"无法读取知识文件：{os.path.basename(path)}")
                logger.warning(f"[加载知识库]{path}无法计算 MD5，跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在于知识库内，跳过")
                continue

            try:
                documents: list[Document] = self.document_reader.read(path)

                if not documents:
                    if strict:
                        raise ValueError("文档没有有效文本内容")
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.semantic_parser.parse(documents)

                if not split_document:
                    if strict:
                        raise ValueError("文档没有生成有效分片")
                    logger.warning(f"[加载知识库]{path}分片后没有有效内容，跳过")
                    continue

                chunk_ids = self._prepare_chunk_ids(
                    split_document,
                    path,
                    md5_hex,
                )
                self._embed_and_store(split_document, chunk_ids, path)

                # 记录这个已经处理好的文件的md5值
                save_md5_hex(md5_hex)
                logger.info(f"[加载数据库]{path}内容加载成功")

            except Exception as e:
                if strict:
                    # 部署初始化失败必须返回非零状态，且不回显第三方异常中的凭据。
                    raise RuntimeError(
                        f"知识导入失败：{os.path.basename(path)}（{type(e).__name__}）"
                    ) from None
                # exc_info=True会记录详细的报错堆栈，如果为False仅记录信息报错本身
                logger.error(f"[加载数据库]{path}加载失败：{str(e)}", exc_info=True)


def test_embedding_and_chroma(
    test_file: str | Path | None = None,
    max_chunks: int = 5,
) -> None:
    """测试少量 chunk 的 Embedding 和 Chroma 写入结果。

    该测试会真实调用 DashScope Embedding，并把结果写入配置中的 Chroma。
    默认只处理维护保养文档的前 5 个 chunk，避免一次测试提交整个知识库。
    """
    project_root = Path(__file__).resolve().parents[1]
    file_path = Path(test_file) if test_file else project_root / "data" / "维护保养.txt"
    if not file_path.is_absolute():
        file_path = project_root / file_path

    service = VectorStoreService()
    file_md5 = get_file_md5_hex(str(file_path))
    if file_md5 is None:
        raise ValueError(f"无法计算测试文件 MD5：{file_path}")

    documents = service.document_reader.read(file_path)
    chunks = service.semantic_parser.parse(documents)
    chunks = chunks[:max_chunks]
    if not chunks:
        raise ValueError(f"测试文件没有可用 chunk：{file_path}")

    chunk_ids = service._prepare_chunk_ids(chunks, str(file_path), file_md5)
    service._embed_and_store(chunks, chunk_ids, str(file_path))

    # 从 Chroma 读回刚写入的第一条记录，验证向量、原文和元数据都已保存。
    stored = service.vector_store.get(
        ids=[chunk_ids[0]],
        include=["embeddings", "documents", "metadatas"],
    )
    # Chroma 返回的 embeddings 可能是 NumPy 数组，不能使用 `or []` 判断真假。
    embeddings = stored.get("embeddings")
    if embeddings is None:
        embeddings = []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    print("=" * 100)
    print(f"测试文件：{file_path.name}")
    print(f"本次写入 chunk 数量：{len(chunks)}")
    print(f"Chroma 返回记录数量：{len(documents)}")
    if len(embeddings) > 0:
        print(f"Embedding 向量维度：{len(embeddings[0])}")
        print(f"Embedding 前 8 个值：{embeddings[0][:8]}")
    if documents:
        print(f"Chroma 中保存的原文：{documents[0]}")
    if metadatas:
        print(f"Chroma 中保存的元数据：{metadatas[0]}")

    # 使用同一个 Chroma 集合做一次检索，确认写入的向量可被查询。
    results = service.get_retriever().invoke("机器人日常应该如何维护？")
    print("-" * 100)
    print("检索结果：")
    for index, result in enumerate(results, start=1):
        print(f"结果 {index}：{result.page_content}")
        print(f"元数据：{result.metadata}")

def _self_test_ingestion():
    """在本文件内验证部署导入的失败退出和 MD5 重跑行为，不调用模型。"""
    import tempfile
    from unittest.mock import Mock, patch

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "知识.txt"
        source.write_text("离线知识", encoding="utf-8")
        digest = get_file_md5_hex(str(source))
        manifest = Path(directory) / "md5.text"
        service = VectorStoreService.__new__(VectorStoreService)
        service.document_reader = Mock()
        service.document_reader.read.return_value = [Document(page_content="知识")]
        service.semantic_parser = Mock()
        service.semantic_parser.parse.return_value = [Document(page_content="知识")]
        service._embed_and_store = Mock(side_effect=ValueError("private-upstream-message"))
        with patch.dict(chroma_conf, {"md5_hex_store": str(manifest)}), patch(
            __name__ + ".listdir_with_allowed_type", return_value=[str(source)]
        ) as list_files:
            try:
                service.load_document(strict=True)
            except RuntimeError as error:
                assert "private-upstream-message" not in str(error)
            else:
                raise AssertionError("失败导入应抛出异常")
            assert digest not in manifest.read_text()
            service._embed_and_store.side_effect = None
            service.load_document(strict=True)
            assert digest in manifest.read_text()
            count = service._embed_and_store.call_count
            service.load_document(strict=True)
            assert service._embed_and_store.call_count == count
            list_files.return_value = []
            try:
                service.load_document(strict=True)
            except RuntimeError:
                pass
            else:
                raise AssertionError("空知识目录应抛出异常")
    print("知识导入离线验证通过：失败抛出、失败不记 MD5、成功重跑跳过、空目录检查。")


if __name__ == '__main__':
    if "--self-test" in sys.argv:
        _self_test_ingestion()
    elif "--ingest-only" in sys.argv:
        try:
            vs = VectorStoreService()
            vs.load_document(strict=True)
            count = vs.vector_store._collection.count()
            if count == 0:
                raise RuntimeError("向量库为空，请检查 MD5 记录是否与向量库配套")
        except Exception as error:
            # 仅输出自有安全错误消息，不回显供应商异常信息。
            message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            print(f"知识库初始化失败：{message}", file=sys.stderr)
            sys.exit(1)
        print(f"知识库初始化完成，共 {count} 个文档块。")
    elif "--test-embedding" in sys.argv:
        test_embedding_and_chroma()
    else:
        vs = VectorStoreService()

        vs.load_document()

        retriever = vs.get_retriever()

        res = retriever.invoke("迷路")

        for r in res:
            print(r.page_content)
            print("-" * 20)
