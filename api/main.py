"""本地运行：python -m api.main；离线验证：python -m api.main --self-test。"""

import argparse
import logging
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.concurrency import run_in_threadpool


logger = logging.getLogger(__name__)

# 规定前端可以发送什么，描述请求格式
class ChatRequest(BaseModel):
    """去掉首尾空格后，问题必须为 1–2000 字符的字符串。"""

    model_config = ConfigDict(extra="forbid")
    message: Annotated[
        str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2000)
    ]

# 规定后端成功时返回什么
class ChatResponse(BaseModel):
    answer: str

# 健康检查接口，与聊天答案无关
class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


def _create_agent():
    # 模型配置、向量库等依赖仅在服务启动时加载，离线验证无需真实凭据。
    from agent.react_agent import ReactAgent

    return ReactAgent()

# 创建FastAPI应用
def create_app(agent_factory: Callable | None = None) -> FastAPI:
    factory = agent_factory if agent_factory is not None else _create_agent

    # 管理服务启动和关闭
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # 创建agent
        application.state.agent = await run_in_threadpool(factory)
        # 创建聊天请求使用的锁
        application.state.agent_lock = Lock()
        try:
            yield
        finally:
            application.state.agent = None

    # 返回FastAPI对象
    application = FastAPI(
        title="扫地机器人 Agent API",
        description="单次问答接口。部分工具使用模拟数据，当前不提供多轮对话记忆。",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        """检查 Agent 是否完成本地初始化，不调用或检测远端模型服务。"""
        ready = getattr(request.app.state, "agent", None) is not None
        if not ready:
            raise HTTPException(status_code=503, detail="Agent 尚未准备就绪")
        return HealthResponse(status="ok", agent_ready=True)

    # 连接agent的接口
    @application.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request):
        """执行单次问答，仅返回最终答案；同步 Agent 调用在线程池中执行。"""
        # 先获取agent
        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent 尚未准备就绪")
        # 初版复用一个 Agent，避免并发操作共享缓存和延迟初始化的路由模型。
        lock = request.app.state.agent_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=503, detail="Agent 正忙，请稍后重试", headers={"Retry-After": "5"}
            )
        try:
            try:
                # 只读取 ASGI 客户端地址。代理头由 Uvicorn 的可信代理规则处理，
                # 不直接相信浏览器提交的 X-Forwarded-For / X-Real-IP。
                from agent.tools.mcp_maps import public_ipv4
                client_ip = public_ipv4(request.client.host if request.client else None)
                trace = agent.execute_with_trace(body.message, client_ip=client_ip)
            except Exception as error:
                # 异常正文可能包含服务地址、凭据等，不直接记录或返回它。
                logger.error("Agent 执行异常，类型：%s", type(error).__name__)
                raise HTTPException(status_code=500, detail="问答执行失败，请稍后重试") from None

            if not isinstance(trace, dict) or trace.get("error"):
                logger.error("Agent 返回失败状态")
                raise HTTPException(status_code=502, detail="答案生成失败，请稍后重试")
            answer = trace.get("final_answer")
            if not isinstance(answer, str) or not answer.strip():
                logger.error("Agent 未返回有效答案")
                raise HTTPException(status_code=502, detail="未生成有效答案，请稍后重试")
            return ChatResponse(answer=answer.strip())
        finally:
            lock.release()

    return application


app = create_app()


def _self_test():
    """在本文件中验证接口边界，不调用模型、不新建测试文件。"""
    from fastapi.testclient import TestClient

    class FakeAgent:
        def __init__(self):
            self.queries = []
            self.client_ips = []
            self.result = {"final_answer": " 测试答案 ", "tool_calls": [{"private": "hidden"}]}
            self.error = None

        def execute_with_trace(self, query, *, client_ip=None):
            self.queries.append(query)
            self.client_ips.append(client_ip)
            if self.error:
                raise self.error
            return self.result

    fake = FakeAgent()
    initialized = []

    def factory():
        initialized.append(True)
        return fake

    application = create_app(factory)
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok", "agent_ready": True}
        assert fake.queries == []
        for payload in ({}, {"message": ""}, {"message": " \n "}, {"message": 123},
                        {"message": None}, {"message": "字" * 2001},
                        {"message": "你好", "unexpected": True}):
            assert client.post("/api/chat", json=payload).status_code == 422
        assert client.post("/api/chat", content="{bad", headers={"Content-Type": "application/json"}).status_code == 422
        assert fake.queries == []
        for query in ("  你好  ", "第二个问题"):
            response = client.post("/api/chat", json={"message": query})
            assert response.status_code == 200
            assert response.json() == {"answer": "测试答案"}
        assert fake.queries == ["你好", "第二个问题"]
        assert fake.client_ips == [None, None]
        assert len(initialized) == 1

        fake.result = {"error": "secret-service-details", "final_answer": "不应返回的部分答案"}
        response = client.post("/api/chat", json={"message": "失败"})
        assert response.status_code == 502 and "secret" not in response.text
        for answer in ("", " \n ", None, []):
            fake.result = {"final_answer": answer}
            assert client.post("/api/chat", json={"message": "空结果"}).status_code == 502
        fake.error = RuntimeError("secret-key")
        response = client.post("/api/chat", json={"message": "异常"})
        assert response.status_code == 500 and "secret" not in response.text
        fake.error = None
        fake.result = {"final_answer": "恢复成功"}
        assert client.post("/api/chat", json={"message": "恢复"}).status_code == 200

        with application.state.agent_lock:
            previous = len(fake.queries)
            response = client.post("/api/chat", json={"message": "忙碌"})
            assert response.status_code == 503 and response.headers["Retry-After"] == "5"
            assert len(fake.queries) == previous
            assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert "/api/chat" in client.get("/openapi.json").json()["paths"]
        # 原始客户端不能通过伪造代理头修改定位 IP。
        with TestClient(create_app(lambda: fake), client=("8.8.8.8", 50000)) as public_client:
            public_client.post("/api/chat", json={"message": "天气"}, headers={"X-Forwarded-For": "1.1.1.1"})
            assert fake.client_ips[-1] == "8.8.8.8"
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
        for peer, expected in (("172.22.0.3", "1.1.1.1"), ("8.8.8.8", "8.8.8.8")):
            proxied = ProxyHeadersMiddleware(create_app(lambda: fake), trusted_hosts=["172.22.0.3"])
            with TestClient(proxied, client=(peer, 50000)) as proxy_client:
                proxy_client.post("/api/chat", json={"message": "天气"}, headers={"X-Forwarded-For": "1.1.1.1"})
                assert fake.client_ips[-1] == expected
    assert application.state.agent is None
    print("接口离线验证通过：输入校验、实例复用、成功响应、错误处理、繁忙状态及接口文档。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 Agent 本地 HTTP 服务")
    parser.add_argument("--self-test", action="store_true", help="运行本文件内的离线验证")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
