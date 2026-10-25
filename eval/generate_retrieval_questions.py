"""根据当前知识库生成初始检索测试集。

生成后的 JSONL 是评测输入，证据使用正文文本而不是 entry_no，
因此固定长度切分和语义切分可以使用同一套标注。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.semantic_parser import SemanticParserService
from utils.config_handler import chroma_conf
from utils.document_reader import DocumentReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "eval" / "retrieval_questions.jsonl"


def _field_value(text: str, field: str) -> str:
    """从语义 chunk 中读取字段值。"""
    for line in text.splitlines():
        value = line.strip()
        for prefix in (f"{field}：", f"{field}:"):
            if value.startswith(prefix):
                return value[len(prefix):].strip()
    return ""


def _short_topic(text: str, fallback: str) -> str:
    """取证据文本前面的短主题，用于生成自然的问题。"""
    value = re.split(r"[：:，,。；;]", text, maxsplit=1)[0].strip()
    return value[:24] or fallback


def _parser() -> tuple[DocumentReader, SemanticParserService]:
    reader = DocumentReader(chroma_conf.get("semantic_type_mapping", {}))
    parser = SemanticParserService(
        fallback_chunk_size=chroma_conf["chunk_size"],
        fallback_chunk_overlap=chroma_conf["chunk_overlap"],
        max_chunk_size=chroma_conf.get("semantic_max_chunk_size", 1000),
        separators=chroma_conf["separators"],
    )
    return reader, parser


def _make_questions(
    file_name: str,
    count: int,
    category: str,
    field: str,
    question_template: str,
    reader: DocumentReader,
    parser: SemanticParserService,
) -> list[dict]:
    file_path = PROJECT_ROOT / "data" / file_name
    chunks = parser.parse(reader.read(file_path))
    questions: list[dict] = []

    for chunk in chunks:
        if len(questions) >= count:
            break

        evidence_text = _field_value(chunk.page_content, field)
        if not evidence_text:
            continue

        if field == "故障现象":
            topic = evidence_text
        elif field == "答案":
            # 问答文档使用原始问题作为查询，答案正文作为正确证据。
            topic = _field_value(chunk.page_content, "问题") or evidence_text
        elif field == "建议":
            topic = _short_topic(evidence_text, category)
        else:
            topic = evidence_text

        questions.append({
            "id": f"{category}_{len(questions) + 1:02d}",
            "query": question_template.format(topic=topic),
            "category": category,
            "should_retrieve": True,
            "gold_evidence": [{
                "source_name": file_name,
                "evidence_text": evidence_text,
            }],
        })

    if len(questions) != count:
        raise ValueError(
            f"{file_name}只生成了{len(questions)}条测试问题，期望{count}条"
        )
    return questions


def generate_questions() -> list[dict]:
    reader, parser = _parser()
    questions: list[dict] = []
    questions.extend(_make_questions(
        "故障排除.txt",
        20,
        "fault",
        "故障现象",
        "机器人出现“{topic}”时，应该如何排查和处理？",
        reader,
        parser,
    ))
    questions.extend(_make_questions(
        "维护保养.txt",
        20,
        "maintenance",
        "建议",
        "机器人在“{topic}”方面应该如何维护？",
        reader,
        parser,
    ))
    questions.extend(_make_questions(
        "选购指南.txt",
        20,
        "selection",
        "建议",
        "购买扫地机器人时，关于“{topic}”应该怎么选择？",
        reader,
        parser,
    ))
    questions.extend(_make_questions(
        "扫拖一体机器人100问.txt",
        20,
        "qa_txt",
        "答案",
        "请问，{topic}",
        reader,
        parser,
    ))
    questions.extend(_make_questions(
        "扫地机器人100问.pdf",
        19,
        "qa_pdf",
        "答案",
        "请问，{topic}",
        reader,
        parser,
    ))
    questions.append({
        "id": "unknown_001",
        "query": "这款扫地机器人可以在水里游泳吗？",
        "category": "unknown",
        "should_retrieve": False,
        "gold_evidence": [],
    })
    return questions


if __name__ == "__main__":
    questions = generate_questions()
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(json.dumps(question, ensure_ascii=False) + "\n")
    print(f"已生成 {len(questions)} 条测试问题：{OUTPUT_PATH}")
