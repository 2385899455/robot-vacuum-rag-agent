from typing import Any, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


from agent.routing import INTENT_BUDGETS, classify_intent


def build_runtime_context(query: str, intent: str | None = None) -> dict[str, Any]:
    """构造一次 Agent 运行的路由和工具预算上下文。"""
    resolved_intent = intent or classify_intent(query) or "direct"
    return {
        "report": False,
        "intent": resolved_intent,
        "max_tool_calls": INTENT_BUDGETS[resolved_intent],
        "tool_call_count": 0,
        "tool_history": [],
    }


def _blocked_tool_result(request: ToolCallRequest, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=request.tool_call.get("id", ""),
        name=request.tool_call.get("name", ""),
    )


@wrap_model_call
def enforce_sequential_report_tools(request: ModelRequest, handler):
    """报告流程禁止并行工具调用，保证用户 ID 和月份先后可追踪。"""
    if request.runtime.context.get("intent") == "report":
        settings = dict(request.model_settings)
        settings["parallel_tool_calls"] = False
        return handler(request.override(model_settings=settings))
    return handler(request)



@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest, # 请求的数据封装
        handler: Callable[[ToolCallRequest], ToolMessage | Command] # 执行的函数本身
) -> ToolMessage | Command: # 工具执行的监控
    name = request.tool_call["name"]
    context = request.runtime.context
    history = context.setdefault("tool_history", [])
    call_count = int(context.get("tool_call_count", 0))
    max_tool_calls = int(context.get("max_tool_calls", 99))

    logger.info(f"[tool monitor]执行工具：{name}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    # 预算统计的是尝试次数，避免工具失败或被拒绝后无限重试。
    if call_count >= max_tool_calls:
        context["tool_call_count"] = call_count + 1
        return _blocked_tool_result(
            request,
            "当前场景已达到工具调用上限，请使用已有信息直接回答用户。",
        )

    # RAG 是一次性检索工具；重复检索只会增加延迟，不能提高流程可靠性。
    if name == "rag_summarize" and name in history:
        context["tool_call_count"] = call_count + 1
        return _blocked_tool_result(
            request,
            "rag_summarize 已经调用过一次，禁止重复调用，请直接生成最终回答。",
        )

    # 报告必须先完成上下文切换，才能读取外部数据。
    if name == "fetch_external_data" and "fill_context_for_report" not in history:
        context["tool_call_count"] = call_count + 1
        return _blocked_tool_result(
            request,
            "调用 fetch_external_data 前必须先调用 fill_context_for_report。",
        )

    context["tool_call_count"] = call_count + 1
    history.append(name)

    try:
        result =  handler(request)
        logger.info(f"[tool monitor]工具{name}调用成功")

        if name == "fill_context_for_report":
            context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{name}调用失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState, # 整个Agent智能体的状态记录
        runtime: Runtime, # 记录整个执行过程中的状态信息
): # 模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None

@dynamic_prompt # 每一次在生成提示词之前调用此函数
def report_prompt_switch(request: ModelRequest): # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report: # 是报告生成场景
        return load_report_prompts()

    return load_system_prompts()
