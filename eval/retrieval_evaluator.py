"""比较固定长度切分和语义切分的向量检索效果。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from rag.semantic_parser import SemanticParserService
from utils.config_handler import chroma_conf
from utils.document_reader import DocumentReader
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type
from utils.path_tool import get_abs_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "retrieval_questions.jsonl"
EVAL_ROOT = PROJECT_ROOT / "chroma_eval"
TOP_K = 5
FIXED_COLLECTION = "agent_eval_fixed"
SEMANTIC_COLLECTION = "agent_eval_semantic"


def _normalise_text(text: str) -> str:
    """去除空白，方便比较固定切分块和语义条目的文本重叠。"""
    return re.sub(r"\s+", "", text)


def _unit_key(source_name: str, section: str, entry_no: int) -> str:
    """生成与切分策略无关的稳定知识单元 ID。"""
    return f"{source_name}|{section}|{entry_no}"


def _chunk_unit_key(chunk: Document) -> str | None:
    source_name = chunk.metadata.get("source_name")
    section = chunk.metadata.get("section")
    entry_no = chunk.metadata.get("entry_no")
    if source_name is None or section is None or entry_no is None:
        return None
    return _unit_key(str(source_name), str(section), int(entry_no))


def _semantic_payload_lines(chunk: Document) -> list[str]:
    """取得语义 chunk 中来自原文的内容行，忽略主题和字段标签。"""
    lines: list[str] = []
    labels = ("主题", "问题", "答案", "建议", "故障现象", "检测", "修复")
    for line in chunk.page_content.splitlines():
        value = line.strip()
        is_theme = False
        for label in labels:
            prefix = f"{label}："
            if value.startswith(prefix):
                is_theme = label == "主题"
                value = value[len(prefix):].strip()
                break
            prefix = f"{label}:"
            if value.startswith(prefix):
                is_theme = label == "主题"
                value = value[len(prefix):].strip()
                break
        if is_theme:
            continue
        if len(_normalise_text(value)) >= 8:
            lines.append(value)
    return lines


def _matches_unit(fixed_text: str, semantic_chunk: Document) -> bool:
    """判断固定长度 chunk 是否包含某个语义知识单元的正文。"""
    fixed_normalised = _normalise_text(fixed_text)
    for line in _semantic_payload_lines(semantic_chunk):
        line_normalised = _normalise_text(line)
        if line_normalised in fixed_normalised:
            return True

        # 如果固定长度边界截断了正文，则用最长公共片段进行近似匹配。
        match = SequenceMatcher(None, fixed_normalised, line_normalised).find_longest_match(
            0,
            len(fixed_normalised),
            0,
            len(line_normalised),
        )
        if match.size >= 12 and match.size / len(line_normalised) >= 0.35:
            return True
    return False


def _annotate_semantic_chunks(chunks: list[Document]) -> None:
    """给语义 chunk 补充与切分策略无关的知识单元 ID。"""
    for chunk in chunks:
        unit_id = _chunk_unit_key(chunk)
        if unit_id is not None:
            chunk.metadata["knowledge_unit_id"] = unit_id


def _annotate_fixed_chunks(
    fixed_chunks: list[Document],
    semantic_chunks: list[Document],
) -> None:
    """为固定长度 chunk 标记它覆盖到的语义知识单元。"""
    for fixed_chunk in fixed_chunks:
        unit_ids: list[str] = []
        for semantic_chunk in semantic_chunks:
            unit_id = _chunk_unit_key(semantic_chunk)
            if unit_id is not None and _matches_unit(fixed_chunk.page_content, semantic_chunk):
                unit_ids.append(unit_id)
        fixed_chunk.metadata["knowledge_unit_ids"] = json.dumps(
            sorted(set(unit_ids)),
            ensure_ascii=False,
        )


def _serializable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """确保 Chroma 元数据只包含可存储的标量值。"""
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _build_chunks(
    reader: DocumentReader,
    semantic_parser: SemanticParserService,
    fixed_splitter: RecursiveCharacterTextSplitter,
    strategy: str,
) -> tuple[list[Document], list[str]]:
    """按照指定策略切分全部知识文件，并返回 chunk 和稳定 ID。"""
    paths = listdir_with_allowed_type(
        get_abs_path(chroma_conf["data_path"]),
        tuple(chroma_conf["allow_knowledge_file_type"]),
    )
    all_chunks: list[Document] = []
    all_ids: list[str] = []

    for path in paths:
        file_path = Path(path)
        file_md5 = get_file_md5_hex(path)
        if file_md5 is None:
            continue

        documents = reader.read(file_path)
        if not documents:
            continue

        semantic_chunks = semantic_parser.parse(documents)
        if strategy == "semantic":
            chunks = semantic_chunks
            _annotate_semantic_chunks(chunks)
        else:
            chunks = fixed_splitter.split_documents(documents)
            _annotate_fixed_chunks(chunks, semantic_chunks)

        for index, chunk in enumerate(chunks):
            chunk.metadata = _serializable_metadata(chunk.metadata)
            chunk_id = f"{strategy}_{file_path.name}_{file_md5[:12]}_{index}"
            chunk.metadata["chunk_id"] = chunk_id
            all_chunks.append(chunk)
            all_ids.append(chunk_id)

    return all_chunks, all_ids


def _create_store(strategy: str, rebuild: bool) -> Chroma:
    if strategy == "fixed":
        collection_name = FIXED_COLLECTION
    else:
        collection_name = SEMANTIC_COLLECTION

    persist_directory = EVAL_ROOT / strategy
    if rebuild and persist_directory.exists():
        shutil.rmtree(persist_directory)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embed_model,
        persist_directory=str(persist_directory),
    )


def build_store(strategy: str, rebuild: bool = False) -> Chroma:
    """构建一个评测用 Chroma collection。"""
    store = _create_store(strategy, rebuild)
    if not rebuild and store._collection.count() > 0:
        print(f"[{strategy}] 使用已有评测向量库，共 {store._collection.count()} 条记录")
        return store
    if not rebuild and store._collection.count() == 0:
        raise RuntimeError(f"[{strategy}] 评测向量库为空，请先执行 --build")

    reader = DocumentReader(chroma_conf.get("semantic_type_mapping", {}))
    semantic_parser = SemanticParserService(
        fallback_chunk_size=chroma_conf["chunk_size"],
        fallback_chunk_overlap=chroma_conf["chunk_overlap"],
        max_chunk_size=chroma_conf.get("semantic_max_chunk_size", 1000),
        separators=chroma_conf["separators"],
    )
    fixed_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chroma_conf["chunk_size"],
        chunk_overlap=chroma_conf["chunk_overlap"],
        separators=chroma_conf["separators"],
        length_function=len,
    )
    chunks, ids = _build_chunks(reader, semantic_parser, fixed_splitter, strategy)
    batch_size = chroma_conf.get("batch_size", 20)
    for start in range(0, len(chunks), batch_size):
        store.add_documents(
            chunks[start:start + batch_size],
            ids=ids[start:start + batch_size],
        )

    print(f"[{strategy}] 已写入 {len(chunks)} 个 chunk")
    return store


def load_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def _gold_evidence(question: dict[str, Any]) -> list[dict[str, str]]:
    """读取与切分方式无关的证据文本标注。"""
    return [
        {
            "source_name": str(evidence["source_name"]),
            "evidence_text": str(evidence["evidence_text"]),
        }
        for evidence in question.get("gold_evidence", [])
    ]


def _result_unit_ids(document: Document) -> set[str]:
    unit_ids: set[str] = set()
    unit_id = document.metadata.get("knowledge_unit_id")
    if unit_id:
        unit_ids.add(str(unit_id))

    raw_unit_ids = document.metadata.get("knowledge_unit_ids", "[]")
    try:
        unit_ids.update(str(value) for value in json.loads(raw_unit_ids))
    except (TypeError, json.JSONDecodeError):
        pass
    return unit_ids


def _matched_evidence(
    evidence: list[dict[str, str]],
    documents: list[Document],
) -> list[dict[str, str]]:
    """判断已检索内容是否包含标注的证据文本。

    按来源文件合并 Top-k 正文，因此证据文本即使被固定长度边界拆开，
    只要相关片段都进入 Top-k，仍然可以被识别为召回成功。
    """
    source_texts: dict[str, str] = {}
    for document in documents:
        source_name = str(document.metadata.get("source_name", ""))
        source_texts[source_name] = source_texts.get(source_name, "")
        source_texts[source_name] += _normalise_text(document.page_content)

    matched: list[dict[str, str]] = []
    for item in evidence:
        source_name = item["source_name"]
        evidence_text = _normalise_text(item["evidence_text"])
        if evidence_text and evidence_text in source_texts.get(source_name, ""):
            matched.append(item)
    return matched


def _evaluate_question(
    store: Chroma,
    question: dict[str, Any],
    top_k: int,
    relevance_threshold: float,
) -> dict[str, Any]:
    scored_results = store.similarity_search_with_relevance_scores(
        question["query"],
        k=top_k,
    )
    gold_evidence = _gold_evidence(question)
    matched_ranks: list[int] = []
    matched_evidence: list[dict[str, str]] = []
    serialized_results: list[dict[str, Any]] = []
    retrieved_documents: list[Document] = []

    for rank, (document, score) in enumerate(scored_results, start=1):
        retrieved_documents.append(document)
        result_ids = _result_unit_ids(document)
        cumulative_matches = _matched_evidence(gold_evidence, retrieved_documents)
        if len(cumulative_matches) > len(matched_evidence):
            matched_ranks.append(rank)
            matched_evidence = cumulative_matches
        serialized_results.append({
            "rank": rank,
            "score": float(score),
            "source_name": document.metadata.get("source_name"),
            "knowledge_unit_ids": sorted(result_ids),
            "chunk_id": document.metadata.get("chunk_id"),
            "page_content": document.page_content,
        })

    first_rank = min(matched_ranks) if matched_ranks else None
    if gold_evidence:
        recall = len(matched_evidence) / len(gold_evidence)
        hit_at_1 = int(bool(matched_ranks and matched_ranks[0] == 1))
        hit_at_k = int(bool(matched_ranks))
        mrr = 1 / first_rank if first_rank else 0.0
    else:
        recall = None
        hit_at_1 = None
        hit_at_k = None
        mrr = None

    top1_score = serialized_results[0]["score"] if serialized_results else None
    return {
        "id": question["id"],
        "query": question["query"],
        "category": question.get("category"),
        "should_retrieve": question.get("should_retrieve", True),
        "gold_evidence": gold_evidence,
        "matched_evidence": matched_evidence,
        "results": serialized_results,
        "hit_at_1": hit_at_1,
        f"hit_at_{top_k}": hit_at_k,
        f"recall_at_{top_k}": recall,
        f"mrr_at_{top_k}": mrr,
        "first_relevant_rank": first_rank,
        "unanswerable_top1_above_threshold": (
            bool(
                not gold_evidence
                and top1_score is not None
                and top1_score >= relevance_threshold
            )
        ),
    }


def evaluate_store(
    store: Chroma,
    strategy: str,
    questions: Iterable[dict[str, Any]],
    top_k: int,
    relevance_threshold: float,
) -> list[dict[str, Any]]:
    results = [
        _evaluate_question(store, question, top_k, relevance_threshold)
        for question in questions
    ]
    answerable = [item for item in results if item["gold_evidence"]]
    unanswerable = [item for item in results if not item["gold_evidence"]]

    def average(field: str) -> float:
        values = [item[field] for item in answerable if item[field] is not None]
        return sum(values) / len(values) if values else 0.0

    summary = {
        "strategy": strategy,
        "question_count": len(results),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "hit_at_1": average("hit_at_1"),
        f"hit_at_{top_k}": average(f"hit_at_{top_k}"),
        f"recall_at_{top_k}": average(f"recall_at_{top_k}"),
        f"mrr_at_{top_k}": average(f"mrr_at_{top_k}"),
        "unanswerable_above_threshold": sum(
            item["unanswerable_top1_above_threshold"] for item in unanswerable
        ),
    }
    output = {"summary": summary, "questions": results}
    output_path = PROJECT_ROOT / "eval" / f"{strategy}_results.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[{strategy}] 详细结果已保存：{output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="比较固定长度切分和语义切分的召回效果")
    parser.add_argument("--build", action="store_true", help="重建两个评测向量库")
    parser.add_argument("--evaluate", action="store_true", help="评测两个已有向量库")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="统计前 k 个检索结果")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.7,
        help="无答案问题的相似度分数阈值",
    )
    args = parser.parse_args()

    if not args.build and not args.evaluate:
        parser.error("请至少指定 --build 或 --evaluate")

    stores: dict[str, Chroma] = {}
    if args.build:
        stores["fixed"] = build_store("fixed", rebuild=True)
        stores["semantic"] = build_store("semantic", rebuild=True)
    elif args.evaluate:
        stores["fixed"] = _create_store("fixed", rebuild=False)
        stores["semantic"] = _create_store("semantic", rebuild=False)

    if args.evaluate or args.build:
        questions = load_questions()
        for strategy, store in stores.items():
            evaluate_store(
                store,
                strategy,
                questions,
                args.top_k,
                args.relevance_threshold,
            )


if __name__ == "__main__":
    main()
