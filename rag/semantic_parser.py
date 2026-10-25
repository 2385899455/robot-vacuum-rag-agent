"""按照语义类型把原始 Document 切分为可独立检索的知识块。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.semantic_type_detector import SemanticType

# 匹配markdown标题 # 标题 ## 标题 ### 标题
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
# 识别数字编号条目 1. 2、 10.
_ENTRY_RE = re.compile(r"^\s*(\d+)[.、]\s*(.*?)\s*$")
# 识别故障条目的字段
_FAULT_FIELD_RE = re.compile(r"(故障现象|检测|修复)\s*[：:]")


@dataclass
class _SourceLine:
    """原始文档中的一行文本，以及这一行文本的来源信息"""

    text: str
    page: int | None
    metadata: dict


@dataclass
class _Entry:
    """从原始文档中识别出的编号条目。"""

    entry_no: int
    section: str
    lines: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_metadata: dict = field(default_factory=dict)


    def add_page(self, page: int | None) -> None:
        """向条目中加入一个来源页码，如果是txt文件或者没有页码就不处理"""
        if page is None:
            return

        if self.page_start is None:
            self.page_start = page
        else:
            self.page_start = min(self.page_start, page)

        if self.page_end is None:
            self.page_end = page
        else:
            self.page_end = max(self.page_end, page)


class SemanticParser(ABC):
    """所有语义解析器都遵循的统一接口。"""

    chunk_type: SemanticType

    @abstractmethod
    def parse(self, documents: Sequence[Document]) -> list[Document]:
        """所有解析器都要实现：把原始 Document 列表转换为语义 chunk。"""


class _NumberedEntryParser(SemanticParser):
    """
    处理章节标题和数字编号条目的公共逻辑。
    负责处理所有类型都共有的逻辑：
    - 识别标题；
    - 识别编号；
    - 合并续行；
    - 记录页码；
    - 创建 chunk；
    - 创建 chunk_id。
    """

    def parse(self, documents: Sequence[Document]) -> list[Document]:
        # 从原始Doucument中识别完整_Entry条目
        entries = self._collect_entries(documents)
        # 把每一个_Entry转换成一个Document
        return [self._build_chunk(entry) for entry in entries]

    # 把一个完整条目转换为最终 chunk
    def _build_chunk(self, entry: _Entry) -> Document:
        # 生成正文
        page_content = self._render_content(entry)
        metadata = dict(entry.source_metadata)
        metadata.update({
            "section": entry.section,
            "entry_no": entry.entry_no,
            "chunk_type": self.chunk_type.value,
            "chunk_id": self._build_chunk_id(entry),
        })

        # 如果来自PDF保存页码范围
        if entry.page_start is not None:
            metadata["page_start"] = entry.page_start
            metadata["page_end"] = entry.page_end

        return Document(page_content=page_content, metadata=metadata)


    # 后续继承的类每个具体解析器都必须自己定义“如何把 _Entry 变成正文”
    @abstractmethod
    def _render_content(self, entry: _Entry) -> str:
        """将一个条目渲染为适合 Embedding 的正文。"""


    # 从原始页面中提取完整条目
    def _collect_entries(self, documents: Sequence[Document]) -> list[_Entry]:
        # 先把原始Document拆成_SourceLine，每一行保留页码来源信息
        lines = list(self._iter_source_lines(documents))
        entries: list[_Entry] = []
        current_entry: _Entry | None = None
        current_section = ""

        # 删除行两侧空格
        for source_line in lines:
            text = source_line.text.strip()
            # 跳过空行
            if not text:
                continue

            # 判断是否是标题
            heading_match = _HEADING_RE.match(text)
            if heading_match:
                # 如果标题前正在构建条目，先保存条目
                if current_entry is not None:
                    entries.append(current_entry)
                    current_entry = None
                # 标题内容保存为当前章节
                current_section = self._clean_text(heading_match.group(1))
                continue

            # 判断是否是新编号条目
            entry_match = _ENTRY_RE.match(text)
            if entry_match:
                # 保存上一个条目
                if current_entry is not None:
                    entries.append(current_entry)

                section = current_section or self._default_section(source_line)
                current_entry = _Entry(
                    entry_no=int(entry_match.group(1)),
                    section=section,
                    lines=[],
                    source_metadata=dict(source_line.metadata),
                )
                current_entry.add_page(source_line.page)

                content = entry_match.group(2).strip()
                if content:
                    current_entry.lines.append(content)
                continue

            # 处理续行：一个序号下不同行
            if current_entry is not None:
                current_entry.lines.append(text)
                current_entry.add_page(source_line.page)
            # 没有条目也没有标题，把普通文本当成标题或默认章节
            elif not current_section:
                current_section = self._clean_text(text)
        # 最后一个条目保存
        if current_entry is not None:
            entries.append(current_entry)

        return entries

    # 将页面展开成文本行
    @staticmethod
    def _iter_source_lines(
        documents: Sequence[Document],
    ) -> Iterable[_SourceLine]:
        """
        input: Loader生成的Document列表
        output: _SourceLine对象
        """
        for document in documents:
            page = _get_page_number(document)
            # 按换行拆分正文
            for line in document.page_content.splitlines():
                yield _SourceLine(
                    text=line,
                    page=page,
                    metadata=dict(document.metadata),
                )


    # 当文档没有明确章节标题时，用文件名作为默认章节
    @staticmethod
    def _default_section(source_line: _SourceLine) -> str:
        source_name = source_line.metadata.get("source_name", "未命名文档")
        return Path(str(source_name)).stem

    # 清理标题和文本两侧的格式符号
    @staticmethod
    def _clean_text(text: str) -> str:
        return text.strip().strip("*_ ")

    # 生成每个 chunk 的唯一 ID
    def _build_chunk_id(self, entry: _Entry) -> str:
        source_name = str(
            entry.source_metadata.get("source_name", "未命名文档")
        )
        source_part = self._safe_id(source_name)
        section_part = self._safe_id(entry.section)
        return f"{source_part}_{section_part}_{entry.entry_no}"

    # 文本转换成相对安全的 ID 片段
    @staticmethod
    def _safe_id(value: str) -> str:
        value = re.sub(r"\s+", "_", value.strip())
        value = re.sub(r"[^\w\u4e00-\u9fff.-]", "_", value)
        return value.strip("_") or "unknown"


class QaParser(_NumberedEntryParser):
    """将问题和对应答案切分到同一个 chunk。"""

    chunk_type = SemanticType.QA

    def _render_content(self, entry: _Entry) -> str:
        """
        input: _Entry对象
        output: str正文

        把条目的第一行当成问题，剩余行作为答案，并清理列表符号 - * ·
        """
        question = self._clean_text(entry.lines[0]) if entry.lines else ""
        answer_lines = [self._clean_answer(line) for line in entry.lines[1:]]
        # 对于多行结果进行拼接
        answer = "\n".join(line for line in answer_lines if line).strip()
        return f"主题：{entry.section}\n问题：{question}\n答案：{answer}".strip()

    @staticmethod
    def _clean_answer(text: str) -> str:
        return re.sub(r"^\s*[-*•]\s*", "", text).strip()


class MaintenanceParser(_NumberedEntryParser):
    """将每条维护建议单独生成一个 chunk。"""

    chunk_type = SemanticType.MAINTENANCE

    def _render_content(self, entry: _Entry) -> str:
        advice = "\n".join(entry.lines).strip()
        return f"主题：{entry.section}\n建议：{advice}".strip()


class SelectionParser(_NumberedEntryParser):
    """将每条选购建议单独生成一个 chunk。"""

    chunk_type = SemanticType.SELECTION

    def _render_content(self, entry: _Entry) -> str:
        advice = "\n".join(entry.lines).strip()
        return f"主题：{entry.section}\n建议：{advice}".strip()


class FaultParser(_NumberedEntryParser):
    """将故障现象、检测和修复保存在同一个 chunk。"""

    chunk_type = SemanticType.FAULT

    def _render_content(self, entry: _Entry) -> str:
        raw_text = "\n".join(entry.lines).strip()
        fields = self._extract_fault_fields(raw_text)

        if not fields:
            return f"主题：{entry.section}\n故障内容：{raw_text}".strip()

        content = [f"主题：{entry.section}"]
        # 从原始文本中提取三个字段
        for field_name in ("故障现象", "检测", "修复"):
            if field_name in fields:
                content.append(f"{field_name}：{fields[field_name]}")
        return "\n".join(content)

    # 提取字段并生成字典
    @staticmethod
    def _extract_fault_fields(text: str) -> dict[str, str]:
        # 查找所有字段标签
        matches = list(_FAULT_FIELD_RE.finditer(text))
        fields: dict[str, str] = {}

        # 两个matches之间的为字段值
        for index, match in enumerate(matches):
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[value_start:value_end].strip(" ;；\n")
            # 把值放入字典
            fields[match.group(1)] = value

        return fields


def _get_page_number(document: Document) -> int | None:
    """从 PDF 元数据中取得从 1 开始的页码。"""
    page_label = document.metadata.get("page_label")
    if page_label is not None:
        try:
            return int(str(page_label))
        except ValueError:
            pass

    page = document.metadata.get("page")
    if isinstance(page, int):
        return page + 1

    return None


class SemanticParserService:
    """根据 semantic_type 选择解析器，并处理超长或未知文档。"""

    def __init__(
        self,
        fallback_chunk_size: int,
        fallback_chunk_overlap: int,
        max_chunk_size: int,
        separators: list[str],
    ) -> None:
        self._parsers: dict[SemanticType, SemanticParser] = {
            SemanticType.QA: QaParser(),
            SemanticType.FAULT: FaultParser(),
            SemanticType.MAINTENANCE: MaintenanceParser(),
            SemanticType.SELECTION: SelectionParser(),
        }
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
            separators=separators,
            length_function=len,
        )
        self._overflow_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            separators=separators,
            length_function=len,
        )
        self._max_chunk_size = max_chunk_size

    def parse(self, documents: Sequence[Document]) -> list[Document]:
        """按照文档的 semantic_type 生成最终 chunk。"""
        if not documents:
            return []

        semantic_type = self._get_semantic_type(documents[0])
        parser = self._parsers.get(semantic_type)

        if parser is None:
            return self._fallback_splitter.split_documents(list(documents))

        chunks = parser.parse(documents)
        if not chunks:
            return self._fallback_splitter.split_documents(list(documents))

        return self._split_oversized_chunks(chunks)

    @staticmethod
    def _get_semantic_type(document: Document) -> SemanticType:
        value = document.metadata.get("semantic_type", SemanticType.UNKNOWN.value)
        try:
            return SemanticType(str(value))
        except ValueError:
            return SemanticType.UNKNOWN

    # 超长条目处理
    def _split_oversized_chunks(self, chunks: list[Document]) -> list[Document]:
        result: list[Document] = []

        for chunk in chunks:
            if len(chunk.page_content) <= self._max_chunk_size:
                result.append(chunk)
                continue

            parent_chunk_id = chunk.metadata.get("chunk_id")
            # 只有超长条目才使用固定长度切分
            children = self._overflow_splitter.split_documents([chunk])
            # 给超长条目的子块编号，同时保存父子关系
            for index, child in enumerate(children, start=1):
                child.metadata.update({
                    "parent_chunk_id": parent_chunk_id,
                    "chunk_id": f"{parent_chunk_id}_{index}",
                })
                result.append(child)

        return result


def test_semantic_parser() -> None:
    """测试语义切分结果，只读取文件和切分 chunk，不调用 Embedding。"""
    from utils.config_handler import chroma_conf
    from utils.document_reader import DocumentReader

    project_root = Path(__file__).resolve().parents[1]
    test_files = (
        project_root / "data" / "维护保养.txt",
        project_root / "data" / "扫地机器人100问.pdf",
    )

    reader = DocumentReader(chroma_conf.get("semantic_type_mapping", {}))
    parser = SemanticParserService(
        fallback_chunk_size=chroma_conf["chunk_size"],
        fallback_chunk_overlap=chroma_conf["chunk_overlap"],
        max_chunk_size=chroma_conf.get("semantic_max_chunk_size", 1000),
        separators=chroma_conf["separators"],
    )

    for file_path in test_files:
        print("=" * 100)
        print(f"测试文件：{file_path.name}")

        # 先完成物理类型识别、Loader 加载和语义类型识别。
        documents = reader.read(file_path)
        # 再按照语义类型生成最终可用于 Embedding 的 chunk。
        chunks = parser.parse(documents)

        print(f"原始 Document 数量：{len(documents)}")
        print(f"最终 chunk 数量：{len(chunks)}")
        if documents:
            print(f"语义类型：{documents[0].metadata.get('semantic_type')}")

        # 默认只打印前 5 个，避免一次性输出整个文件的所有 chunk。
        for index, chunk in enumerate(chunks[:5], start=1):
            print("-" * 100)
            print(f"chunk {index}")
            print(f"正文长度：{len(chunk.page_content)}")
            print(f"元数据：{chunk.metadata}")
            print("正文：")
            print(chunk.page_content)


if __name__ == "__main__":
    test_semantic_parser()
