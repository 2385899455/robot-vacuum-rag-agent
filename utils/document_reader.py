"""读取支持的源文件，并统一处理 LangChain Document。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from langchain_core.documents import Document

from utils.file_handler import pdf_loader, txt_loader
from utils.file_type_detector import (
    PhysicalFileType,
    PhysicalFileTypeDetector,
)
from utils.logger_handler import logger
from utils.semantic_type_detector import (
    SemanticTypeDetector,
    SemanticTypeResult,
)


DocumentLoader = Callable[[str], list[Document]]


class DocumentReader:
    """识别物理类型 -> 选择Loader -> 读取文件 -> 统一Document元数据"""

    def __init__(self, semantic_type_mapping: Mapping[str, str] | None = None) -> None:
        self._loaders: dict[PhysicalFileType, DocumentLoader] = {
            PhysicalFileType.TXT: txt_loader,
            PhysicalFileType.PDF: pdf_loader,
        }
        self._semantic_type_detector = SemanticTypeDetector(semantic_type_mapping)

    def read(self, path: str | Path) -> list[Document]:
        """读取一个支持的文件，并返回统一处理后的 Document 列表。

        无法识别或不支持的物理文件类型会跳过，并记录到日志中，避免单个不支持的文件
        中断整个知识库导入流程。Loader 本身产生的异常会继续向上抛出，由向量库导入
        流程统一记录完整的错误信息。
        """
        file_path = Path(path)
        physical_type = PhysicalFileTypeDetector.detect(file_path)
        # 字典：物理文件类型 -> 对应Loader函数
        loader = self._loaders.get(physical_type)

        if loader is None:
            logger.warning(
                f"[文档读取]{file_path}无法识别或不支持物理文件类型，跳过"
            )
            return []

        # Loader加载的内容
        # Loader返回Document列表，给个Document包含page_content和metadata这两部分
        documents = loader(str(file_path))
        semantic_result = self._semantic_type_detector.detect(file_path, documents)
        self._normalize_metadata(
            documents,
            file_path,
            physical_type,
            semantic_result,
        )
        return documents

    # @staticmethod 表明方法不依赖self
    @staticmethod
    def _normalize_metadata(
        documents: list[Document],
        file_path: Path,
        physical_type: PhysicalFileType,
        semantic_result: SemanticTypeResult,
    ) -> None:
        """为所有读取结果补充统一的元数据。"""
        for index, document in enumerate(documents):
            # 如果 Loader 已经提供 source，则保留原值；同时补充稳定的文件名和绝对路径，
            # 方便后续定位文档来源。
            document.metadata.setdefault("source", str(file_path))
            document.metadata.update({
                "source_name": file_path.name,
                "source_path": str(file_path.resolve()),
                "physical_type": physical_type.value,
                "document_index": index,
                "semantic_type": semantic_result.semantic_type.value,
                "semantic_type_source": semantic_result.source,
                "semantic_type_confidence": semantic_result.confidence,
                "semantic_type_reason": semantic_result.reason,
            })


def test_document_reader() -> None:
    """测试 DocumentReader，并打印 TXT/PDF 的读取结果。"""
    from utils.config_handler import chroma_conf

    project_root = Path(__file__).resolve().parents[1]
    test_files = (
        project_root / "data" / "维护保养.txt",
        project_root / "data" / "扫地机器人100问.pdf",
    )
    reader = DocumentReader(chroma_conf.get("semantic_type_mapping", {}))

    for file_path in test_files:
        print("=" * 80)
        print(f"测试文件：{file_path.name}")

        documents = reader.read(file_path)
        print(f"Document 数量：{len(documents)}")

        for index, document in enumerate(documents):
            print("-" * 80)
            print(f"Document 索引：{index}")
            print(f"Document 类型：{type(document).__name__}")
            print(f"元数据：{document.metadata}")
            print("正文预览：")
            print(document.page_content[:500])


if __name__ == "__main__":
    test_document_reader()
