"""Agent 轨迹评测工具。

该模块读取 ``agent_questions.jsonl``，逐题执行 Agent，并根据测试集中的
期望工具、工具顺序、参数规则和答案关键词计算评测指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any


DEFAULT_QUESTION_FILE = Path(__file__).with_name("agent_questions.jsonl")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("agent_results.json")
DEFAULT_DESCRIPTION_BASELINE = Path(__file__).with_name("tool_descriptions_before.json")
DEFAULT_COMPARISON_FILE = Path(__file__).with_name("tool_description_comparison.json")


def load_questions(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 测试集。"""
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                question = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"测试集第 {line_number} 行不是有效 JSON: {error}") from error
            questions.append(question)
    return questions


def _normalize(value: Any) -> str:
    """将参数或返回值转换成适合比较的文本。"""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip().casefold()


def _actual_tool_names(trace: dict[str, Any]) -> list[str]:
    return [
        str(call.get("name", ""))
        for call in trace.get("tool_calls", [])
        if call.get("name")
    ]


def _find_tool_record(trace: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    for call in trace.get("tool_calls", []):
        if call.get("name") == tool_name:
            return call
    return None


def _result_for_tool(trace: dict[str, Any], tool_name: str) -> str:
    record = _find_tool_record(trace, tool_name)
    if record is None:
        return ""
    return _normalize(record.get("result"))


def check_argument_rules(
    question: dict[str, Any], trace: dict[str, Any]
) -> tuple[bool, list[str]]:
    """检查测试集声明的工具参数规则。"""
    errors: list[str] = []
    rules_by_tool = question.get("tool_arg_rules", {}) or {}

    for tool_name, rules in rules_by_tool.items():
        record = _find_tool_record(trace, tool_name)
        if record is None:
            errors.append(f"未找到工具调用: {tool_name}")
            continue

        args = record.get("args", {})
        if not isinstance(args, dict):
            args = {}

        for expected in rules.get("query_contains", []) or []:
            actual_query = _normalize(args.get("query"))
            if _normalize(expected) not in actual_query:
                errors.append(
                    f"{tool_name}.query 不包含 {expected!r}，实际为 {args.get('query')!r}"
                )

        if "city_equals" in rules:
            actual_city = _normalize(args.get("city"))
            expected_city = _normalize(rules["city_equals"])
            if actual_city != expected_city:
                errors.append(
                    f"{tool_name}.city 应为 {rules['city_equals']!r}，实际为 {args.get('city')!r}"
                )

        if "month_equals" in rules:
            actual_month = _normalize(args.get("month"))
            expected_month = _normalize(rules["month_equals"])
            if actual_month != expected_month:
                errors.append(
                    f"{tool_name}.month 应为 {rules['month_equals']!r}，实际为 {args.get('month')!r}"
                )

        for arg_name, source_tool in (
            ("city", rules.get("city_from_tool")),
            ("user_id", rules.get("user_id_from_tool")),
            ("month", rules.get("month_from_tool")),
        ):
            if not source_tool:
                continue
            expected_value = _result_for_tool(trace, source_tool)
            actual_value = _normalize(args.get(arg_name))
            if not expected_value or actual_value != expected_value:
                errors.append(
                    f"{tool_name}.{arg_name} 未正确使用 {source_tool} 的返回值，"
                    f"实际为 {args.get(arg_name)!r}，期望为 {expected_value!r}"
                )

    return not errors, errors


def _keyword_score(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    if not keywords:
        return 1.0, []
    normalized_answer = _normalize(answer)
    missing = [keyword for keyword in keywords if _normalize(keyword) not in normalized_answer]
    return (len(keywords) - len(missing)) / len(keywords), missing


def evaluate_trace(question: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """根据一条测试数据和一次执行轨迹生成逐题评测结果。"""
    expected_tools = question.get("expected_tools", []) or []
    expected_order = question.get("expected_order", expected_tools) or []
    forbidden_tools = set(question.get("forbidden_tools", []) or [])
    actual_tools = _actual_tool_names(trace)

    selection_exact = actual_tools == expected_tools
    order_correct = actual_tools == expected_order
    order_preserved = all(
        expected_tool in actual_tools[index:]
        for index, expected_tool in enumerate(expected_order)
    ) if expected_order else not actual_tools
    forbidden_used = sorted(forbidden_tools.intersection(actual_tools))
    argument_correct, argument_errors = check_argument_rules(question, trace)
    keyword_score, missing_keywords = _keyword_score(
        trace.get("final_answer", ""),
        question.get("answer_keywords", []) or [],
    )
    max_tool_calls = question.get("max_tool_calls")
    within_limit = (
        max_tool_calls is None
        or trace.get("tool_call_count", len(actual_tools)) <= max_tool_calls
    )
    error = trace.get("error")

    passed = all(
        [
            not error,
            selection_exact,
            argument_correct,
            not forbidden_used,
            keyword_score == 1.0,
            within_limit,
        ]
    )

    return {
        "id": question.get("id", ""),
        "category": question.get("category", ""),
        "query": question.get("query", ""),
        "passed": passed,
        "actual_tools": actual_tools,
        "expected_tools": expected_tools,
        "selection_exact": selection_exact,
        "order_correct": order_correct,
        "order_preserved": order_preserved,
        "forbidden_tools_used": forbidden_used,
        "argument_correct": argument_correct,
        "argument_errors": argument_errors,
        "answer_keyword_score": round(keyword_score, 4),
        "missing_answer_keywords": missing_keywords,
        "within_tool_call_limit": within_limit,
        "error": error,
        "tool_call_count": trace.get("tool_call_count", len(actual_tools)),
        "elapsed_seconds": trace.get("elapsed_seconds", 0.0),
        "trace": trace,
    }


def _average(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return round(sum(float(bool(result.get(key))) for result in results) / len(results), 4)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总总体指标。"""
    count = len(results)
    if count == 0:
        return {
            "question_count": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "tool_selection_accuracy": 0.0,
            "tool_order_accuracy": 0.0,
            "order_preserved_rate": 0.0,
            "argument_accuracy": 0.0,
            "answer_keyword_hit_rate": 0.0,
            "forbidden_tool_rate": 0.0,
            "within_tool_call_limit_rate": 0.0,
            "error_rate": 0.0,
            "average_tool_calls": 0.0,
            "average_elapsed_seconds": 0.0,
        }

    return {
        "question_count": count,
        "passed": sum(bool(result["passed"]) for result in results),
        "pass_rate": round(sum(bool(result["passed"]) for result in results) / count, 4),
        "tool_selection_accuracy": _average(results, "selection_exact"),
        "tool_order_accuracy": _average(results, "order_correct"),
        "order_preserved_rate": _average(results, "order_preserved"),
        "argument_accuracy": _average(results, "argument_correct"),
        "answer_keyword_hit_rate": round(
            sum(result["answer_keyword_score"] for result in results) / count,
            4,
        ),
        "forbidden_tool_rate": round(
            sum(not result["forbidden_tools_used"] for result in results) / count,
            4,
        ),
        "within_tool_call_limit_rate": _average(results, "within_tool_call_limit"),
        "error_rate": round(sum(bool(result["error"]) for result in results) / count, 4),
        "average_tool_calls": round(
            sum(result["tool_call_count"] for result in results) / count,
            3,
        ),
        "average_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / count,
            3,
        ),
    }


def summarize_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按测试类别分别汇总，便于定位 Agent 在哪类问题上表现较差。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["category"], []).append(result)
    return {
        category: summarize_results(category_results)
        for category, category_results in sorted(grouped.items())
    }


def run_evaluation(
    question_file: Path = DEFAULT_QUESTION_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    limit: int | None = None,
) -> dict[str, Any]:
    """运行完整评测并保存 JSON 结果。"""
    questions = load_questions(question_file)
    if limit is not None:
        questions = questions[:limit]

    # 必须在导入 ReactAgent 前设置，agent_tools 会读取这个环境变量。
    os.environ["AGENT_EVAL_MODE"] = "1"
    from agent.react_agent import ReactAgent

    agent = ReactAgent()
    results: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for index, question in enumerate(questions, 1):
        query = question["query"]
        print(f"[{index}/{len(questions)}] {question.get('id', '')}: {query}", flush=True)
        trace = agent.execute_with_trace(query)
        result = evaluate_trace(question, trace)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {status} tools={result['actual_tools']} "
            f"answer={result['answer_keyword_score']:.0%} "
            f"time={result['elapsed_seconds']:.3f}s",
            flush=True,
        )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "question_file": str(question_file),
        "evaluation_elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "metrics": summarize_results(results),
        "metrics_by_category": summarize_by_category(results),
        "results": results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _percent(value: float) -> str:
    return f"{value:.2%}"


def print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    print("\n===== Agent 评测结果 =====")
    print(f"通过率:             {_percent(metrics['pass_rate'])}")
    print(f"工具选择准确率:     {_percent(metrics['tool_selection_accuracy'])}")
    print(f"工具顺序准确率:     {_percent(metrics['tool_order_accuracy'])}")
    print(f"工具参数准确率:     {_percent(metrics['argument_accuracy'])}")
    print(f"答案关键词命中率:   {_percent(metrics['answer_keyword_hit_rate'])}")
    print(f"禁止工具未调用率:   {_percent(metrics['forbidden_tool_rate'])}")
    print(f"调用次数限制通过率: {_percent(metrics['within_tool_call_limit_rate'])}")
    print(f"执行错误率:         {_percent(metrics['error_rate'])}")
    print(f"平均工具调用次数:   {metrics['average_tool_calls']:.3f}")
    print(f"平均执行耗时:       {metrics['average_elapsed_seconds']:.3f}s")


def run_description_comparison(
    question_file: Path,
    baseline_file: Path,
    output_file: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """逐题交替运行前后版本，共享路由，持续保存进度；不修改工具业务实现。"""
    questions = load_questions(question_file)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise ValueError("对比实验至少需要一道题")
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError("基线描述文件必须为工具名称到描述的 JSON 对象")

    os.environ["AGENT_EVAL_MODE"] = "1"
    from agent.react_agent import ReactAgent
    from agent.routing import IntentRouter
    from agent.tools import agent_tools
    from utils.config_handler import agent_conf, rag_conf

    # 即使调用方已导入过工具模块，也使用相同的评测城市、用户和月份。
    agent_tools.EVAL_MODE = True
    current = {name: getattr(agent_tools, name).description for name in baseline}
    route_model = IntentRouter(mode=(agent_conf.get("router") or {}).get("mode", "hybrid"))
    decisions = {}

    class SharedRouter:
        def route(self, query):
            # 路由已在两组执行前计算；两组耗时均排除这次公共分类开销。
            return replace(decisions[query], elapsed_seconds=0.0)

    agents = {
        "before": ReactAgent(router=SharedRouter(), tool_descriptions=baseline),
        "after": ReactAgent(router=SharedRouter(), tool_descriptions=current),
    }
    results = {"before": [], "after": []}
    project_root = Path(__file__).resolve().parent.parent
    fingerprint_files = [
        "agent/routing.py", "agent/tools/agent_tools.py", "agent/tools/middleware.py",
        "prompts/main_prompt.txt", "prompts/report_prompt.txt", "prompts/rag_summarize.txt",
        "data/external/records.csv", "config/chroma.yml",
    ]
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "complete": False,
        "question_file": str(question_file),
        "baseline_file": str(baseline_file),
        "planned_question_count": len(questions),
        "controls": {
            "shared_routing": True,
            "alternating_execution_order": True,
            "eval_mode": True,
            "latency_excludes_shared_routing": True,
            "repeats": 1,
            "model": {key: value for key, value in rag_conf.get("chat_model", {}).items()
                      if key in {"name", "provider", "temperature", "use_responses_api"}},
            "file_sha256": {name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
                            for name in fingerprint_files},
        },
        "descriptions": {"before": baseline, "after": current},
        "questions": questions,
        "routing": {},
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    def save_progress():
        report["evaluation_elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        for version in results:
            report[version] = {
                "metrics": summarize_results(results[version]),
                "metrics_by_category": summarize_by_category(results[version]),
                "results": results[version],
            }
        # 中途退出仍保留已完成的逐题记录。
        temporary = output_file.with_suffix(output_file.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_file)

    save_progress()
    for index, question in enumerate(questions, 1):
        query = question["query"]
        if query not in decisions:
            decisions[query] = route_model.route(query)
        report["routing"][question["id"]] = decisions[query].to_dict()
        order = ("before", "after") if index % 2 else ("after", "before")
        for version in order:
            print(f"[{index}/{len(questions)}] {version} {question['id']}: {query}", flush=True)
            trace = agents[version].execute_with_trace(query)
            result = evaluate_trace(question, trace)
            results[version].append(result)
            print(
                f"  {'PASS' if result['passed'] else 'FAIL'} tools={result['actual_tools']} "
                f"answer={result['answer_keyword_score']:.0%} time={result['elapsed_seconds']:.3f}s",
                flush=True,
            )
            save_progress()

    report["complete"] = True
    report["metric_delta_after_minus_before"] = {
        key: round(report["after"]["metrics"][key] - value, 4)
        for key, value in report["before"]["metrics"].items()
    }
    report["changed_cases"] = [
        {"id": before["id"], "before_passed": before["passed"], "after_passed": after["passed"],
         "before_tools": before["actual_tools"], "after_tools": after["actual_tools"],
         "before_missing_keywords": before["missing_answer_keywords"],
         "after_missing_keywords": after["missing_answer_keywords"]}
        for before, after in zip(results["before"], results["after"])
        if before["passed"] != after["passed"] or before["actual_tools"] != after["actual_tools"]
    ]
    save_progress()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 Agent 的工具调用轨迹和最终答案")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTION_FILE,
        help="JSONL 测试集路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="评测结果 JSON 路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前 N 条测试，用于先验证流程",
    )
    parser.add_argument(
        "--compare-tool-descriptions",
        nargs="?",
        const=DEFAULT_DESCRIPTION_BASELINE,
        type=Path,
        help="比较旧版描述 JSON 与当前工具描述，可省略路径使用内置基线",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正整数")

    if args.compare_tool_descriptions is not None:
        output = args.output or DEFAULT_COMPARISON_FILE
        comparison = run_description_comparison(
            args.questions, args.compare_tool_descriptions, output, args.limit,
        )
        for version in ("before", "after"):
            print(f"\n===== 工具描述版本：{version} =====")
            print_summary(comparison[version])
        print(f"\n通过状态或工具序列变化的题数：{len(comparison['changed_cases'])}")
        print(f"完整对比结果已保存到: {output}")
        return

    output = args.output or DEFAULT_OUTPUT_FILE
    report = run_evaluation(
        question_file=args.questions,
        output_file=output,
        limit=args.limit,
    )
    print_summary(report)
    print(f"\n详细结果已保存到: {output}")


if __name__ == "__main__":
    main()
