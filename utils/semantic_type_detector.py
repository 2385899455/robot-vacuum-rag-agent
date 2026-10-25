"""
知识文档语义类型识别。
配置映射 -> 文件名规则 -> 正文规则 -> unknown
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from langchain_core.documents import Document


class SemanticType(str, Enum):
    """当前知识库支持的语义类型。"""

    QA = "qa" # qa问答格式
    FAULT = "fault" # 故障问题
    MAINTENANCE = "maintenance" # 维护问题
    SELECTION = "selection" # 选购问题
    UNKNOWN = "unknown" 


@dataclass(frozen=True)
class SemanticTypeResult:
    """语义类型识别结果。"""

    semantic_type: SemanticType
    source: str
    confidence: float # 结果的置信度，按照规则算出来的，不是通过大模型
    reason: str # 识别原因


class SemanticTypeDetector:
    """按照配置、文件名、正文结构和关键词识别语义类型。"""

    # 文件名识别规则 (语义类型, (关键词1, 关键词2)),
    _FILE_NAME_RULES: tuple[tuple[SemanticType, tuple[str, ...]], ...] = (
        (SemanticType.FAULT, ("故障", "排除")),
        (SemanticType.MAINTENANCE, ("维护", "保养")),
        (SemanticType.SELECTION, ("选购",)),
        (SemanticType.QA, ("问答", "常见问题", "100问", "问题及解答")),
    )

    # 正文关键词规则
    # 每个关键词后面有一个分数，如果正文包含这一关键词就给相应类别加分
    _CONTENT_RULES: tuple[tuple[SemanticType, tuple[tuple[str, int], ...]], ...] = (
        (
            SemanticType.FAULT,
            (("故障现象", 3), ("检测", 3), ("修复", 3)),
        ),
        (
            SemanticType.QA,
            (("问题", 2), ("答案", 2), ("？", 1), ("?", 1)),
        ),
        (
            SemanticType.MAINTENANCE,
            (("维护", 2), ("保养", 2), ("每日", 1), ("每周", 1), ("每月", 1)),
        ),
        (
            SemanticType.SELECTION,
            (("选购", 2), ("购买", 2), ("优先选择", 2), ("适合", 1), ("参数", 1)),
        ),
    )

    def __init__(self, mapping: Mapping[str, str] | None = None) -> None:
        """保存文件名到语义类型的显式映射。"""
        self._mapping = self._normalize_mapping(mapping or {})

    def detect(
        self,
        path: str | Path,
        documents: Sequence[Document],
    ) -> SemanticTypeResult:
        """按照三层规则识别一个文件的语义类型。"""
        file_path = Path(path)
        file_name = file_path.name

        # 完整文件名称直接匹配
        configured_type = self._mapping.get(file_name.casefold())
        if configured_type is not None:
            return SemanticTypeResult(
                semantic_type=configured_type,
                source="config",
                confidence=1.0,
                reason="命中文件名语义类型配置",
            )

        # 文档名关键词匹配
        # 文件名中包含任意关键词直接返回
        file_name_type = self._detect_by_file_name(file_name)
        if file_name_type is not None:
            return SemanticTypeResult(
                semantic_type=file_name_type,
                source="filename",
                confidence=0.95,
                reason="命中文件名规则",
            )

        # 正文关键词匹配
        # 把所有 Document 的正文拼接成一个字符串
        # 最高分大于等于2分且严格高于第二名
        text = "\n".join(document.page_content for document in documents)
        content_result = self._detect_by_content(text)
        if content_result is not None:
            return content_result

        return SemanticTypeResult(
            semantic_type=SemanticType.UNKNOWN,
            source="unknown",
            confidence=0.0,
            reason="没有命中配置、文件名或正文规则",
        )

    @staticmethod
    def _normalize_mapping(mapping: Mapping[str, str]) -> dict[str, SemanticType]:
        """
        输入：
        {
            "维护保养.txt": "maintenance"
        }
        输出：
        {
            "维护保养.txt": SemanticType.MAINTENANCE
        }
        """
        normalized: dict[str, SemanticType] = {}

        for file_name, type_name in mapping.items():
            try:
                normalized[str(file_name).casefold()] = SemanticType(str(type_name))
            except ValueError:
                # 配置中的未知类型不应阻止其他文件继续导入。
                continue

        return normalized

    
    @classmethod
    # 文件名中是否有关键词
    def _detect_by_file_name(cls, file_name: str) -> SemanticType | None:
        name = file_name.casefold()

        for semantic_type, keywords in cls._FILE_NAME_RULES:
            if any(keyword.casefold() in name for keyword in keywords):
                return semantic_type

        return None

    @classmethod
    # 判断整篇文章是否出现过关键词，每个关键词最多加一分，不会因为同一个词重复出现而重复加分
    def _detect_by_content(cls, text: str) -> SemanticTypeResult | None:
        scores: dict[SemanticType, int] = {}

        for semantic_type, rules in cls._CONTENT_RULES:
            scores[semantic_type] = sum(
                weight for keyword, weight in rules if keyword in text
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < 2:
            return None

        best_type, best_score = ranked[0]
        if len(ranked) > 1 and best_score == ranked[1][1]:
            return None

        confidence = min(0.9, 0.5 + best_score * 0.1)
        return SemanticTypeResult(
            semantic_type=best_type,
            source="content",
            confidence=confidence,
            reason=f"正文规则得分最高：{best_score}",
        )
