import json
import sys
import time

from langchain.agents import create_agent
from agent.tools.middleware import (
    build_runtime_context,
    enforce_sequential_report_tools,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from model.factory import chat_model
from agent.routing import INTENT_BUDGETS, IntentRouter, TOOLS_BY_INTENT
from utils.config_handler import agent_conf
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import *



# 规定每个场景可以使用哪些工具
class ReactAgent:
    def __init__(
        self,
        router: IntentRouter | None = None,
        tool_descriptions: dict[str, str] | None = None,
    ):
        self.router = router if router is not None else IntentRouter(
            mode=(agent_conf.get("router") or {}).get("mode", "hybrid")
        )
        all_tools = {
            "rag_summarize": rag_summarize,
            "get_weather": get_weather,
            "get_user_location": get_user_location,
            "get_user_id": get_user_id,
            "get_current_month": get_current_month,
            "fetch_external_data": fetch_external_data,
            "fill_context_for_report": fill_context_for_report,
        }
        if tool_descriptions is not None:
            if set(tool_descriptions) != set(all_tools):
                raise ValueError("对比描述必须完整覆盖当前七个工具")
            if any(not isinstance(value, str) or not value.strip()
                   for value in tool_descriptions.values()):
                raise ValueError("工具描述必须是非空字符串")
            # 为实验复制工具，避免修改全局对象或影响另一组 Agent。
            all_tools = {
                name: tool.model_copy(update={"description": tool_descriptions[name]})
                for name, tool in all_tools.items()
            }
        self.agents = {}
        for intent, tool_names in TOOLS_BY_INTENT.items():
            self.agents[intent] = create_agent(
                model=chat_model,
                system_prompt=load_system_prompts(),
                tools=[all_tools[name] for name in tool_names],
                middleware=[
                    monitor_tool,
                    enforce_sequential_report_tools,
                    log_before_model,
                    report_prompt_switch,
                ],
            )

    def _agent_for_query(self, query: str):
        return self.agents[self.router.route(query).intent]


    def execute_stream(self, query: str, *, client_ip: str | None = None):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        decision = self.router.route(query)
        intent = decision.intent
        context = build_runtime_context(query, intent, client_ip=client_ip)
        agent = self.agents[intent]
        for chunk in agent.stream(input_dict, stream_mode="values", context=context):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

    @staticmethod
    def _to_jsonable(value):
        """把消息中的参数或返回值转换成可写入 JSON 的内容。"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [ReactAgent._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): ReactAgent._to_jsonable(item)
                for key, item in value.items()
            }
        return str(value)

    @staticmethod
    def _message_text(message) -> str:
        """提取消息正文，兼容字符串和内容块列表。"""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    text_parts.append(str(block["text"]))
                elif isinstance(block, str):
                    text_parts.append(block)
            return "".join(text_parts).strip()
        return str(content).strip() if content else ""

    @staticmethod
    def _append_tool_call(trace: dict, message, node_name: str, pending_calls: dict) -> None:
        """记录模型发出的工具调用，并保存 call_id 方便后续匹配返回值。"""
        for tool_call in getattr(message, "tool_calls", []) or []:
            call_id = tool_call.get("id") or f"tool_call_{len(trace['tool_calls']) + 1}"
            record = {
                "name": tool_call.get("name", ""),
                "args": ReactAgent._to_jsonable(tool_call.get("args", {})),
                "result": None,
                "node": node_name,
                "call_id": call_id,
            }
            trace["tool_calls"].append(record)
            pending_calls[call_id] = record

    @staticmethod
    def _append_tool_result(trace: dict, message, pending_calls: dict) -> None:
        """把工具返回值写回对应的工具调用记录。"""
        call_id = getattr(message, "tool_call_id", None)
        record = pending_calls.get(call_id) if call_id else None

        if record is None:
            # 少数模型或适配器可能不返回 tool_call_id，此时按工具名兜底匹配。
            tool_name = getattr(message, "name", "") or ""
            for candidate in reversed(trace["tool_calls"]):
                if candidate["name"] == tool_name and candidate["result"] is None:
                    record = candidate
                    break

        if record is None:
            record = {
                "name": getattr(message, "name", "") or "",
                "args": {},
                "result": None,
                "node": "tools",
                "call_id": call_id,
            }
            trace["tool_calls"].append(record)

        record["result"] = ReactAgent._to_jsonable(message.content)

    def execute_with_trace(self, query: str, *, client_ip: str | None = None) -> dict:
        """执行一次 Agent，并返回可用于评测的完整执行记录。"""
        trace = {
            "query": query,
            "node_sequence": [],
            "tool_calls": [],
            "final_answer": "",
            "tool_call_count": 0,
            "elapsed_seconds": 0.0,
        }
        pending_calls = {}
        start_time = time.perf_counter()
        decision = self.router.route(query)
        trace["routing"] = decision.to_dict()
        intent = decision.intent
        trace["intent"] = intent
        trace["max_tool_calls"] = INTENT_BUDGETS[intent]
        context = build_runtime_context(query, intent, client_ip=client_ip)
        agent = self.agents[intent]
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        try:
            for update in agent.stream(
                input_dict,
                stream_mode="updates",
                context=context,
            ):
                if not isinstance(update, dict):
                    continue

                for node_name, state_update in update.items():
                    trace["node_sequence"].append(str(node_name))
                    messages = []
                    if isinstance(state_update, dict):
                        messages = state_update.get("messages", []) or []

                    for message in messages:
                        if getattr(message, "tool_calls", None):
                            self._append_tool_call(
                                trace,
                                message,
                                str(node_name),
                                pending_calls,
                            )

                        message_type = getattr(message, "type", "")
                        if message_type == "tool":
                            self._append_tool_result(
                                trace,
                                message,
                                pending_calls,
                            )
                        elif message_type in {"ai", "assistant"}:
                            content = self._message_text(message)
                            if content and not getattr(message, "tool_calls", None):
                                trace["final_answer"] = content
        except Exception as error:
            trace["error"] = f"{type(error).__name__}: {error}"
        finally:
            trace["tool_call_count"] = len(trace["tool_calls"])
            trace["elapsed_seconds"] = round(time.perf_counter() - start_time, 3)

        return trace

    def execute_with_trace_text(self, query: str) -> str:
        """以格式化 JSON 文本返回 Agent 执行记录。"""
        return json.dumps(
            self.execute_with_trace(query),
            ensure_ascii=False,
            indent=2,
        )

if __name__ == "__main__":
    agent = ReactAgent()

    if "--trace" in sys.argv:
        print(agent.execute_with_trace_text("给我生成一个使用报告"))
    else:
        for chunk in agent.execute_stream("给我生成一个使用报告"):
            print(chunk, end="", flush=True)
