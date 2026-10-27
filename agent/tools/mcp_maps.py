"""高德官方 MCP 适配层；只允许天气和 IP 定位两个远程工具。"""

import asyncio
import ipaddress
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from urllib.parse import quote, urlencode

from mcp import Client


SOURCE = "高德地图 MCP"
TOOL_NAMES = {"maps_weather", "maps_ip_location"}
FORECAST_FIELDS = (
    "date", "week", "dayweather", "nightweather", "daytemp", "nighttemp",
    "daywind", "nightwind", "daypower", "nightpower", "daytemp_float", "nighttemp_float",
)


def public_ipv4(value: str | None) -> str | None:
    """只接受请求上下文中的公网 IPv4；不查询服务器出口 IP。"""
    try:
        address = ipaddress.ip_address(value or "")
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if (isinstance(address, ipaddress.IPv4Address) and address.is_global
                and not address.is_multicast and not address.is_reserved):
            return str(address)
    except ValueError:
        pass
    return None


def _error(code: str, message: str) -> dict:
    return {"ok": False, "source": SOURCE, "code": code, "message": message}


class _RedactKey(logging.Filter):
    """第三方 SDK 的异常日志也不能回显 URL 中的高德 Key。"""

    def filter(self, record):
        key = os.getenv("AMAP_MAPS_API_KEY", "").strip()
        if not key:
            return True
        def redact(text):
            return text.replace(key, "[REDACTED]").replace(quote(key, safe=""), "[REDACTED]")
        record.msg = redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


_REDACT = _RedactKey()


def _protect_sdk_logs():
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.startswith(("mcp", "httpx2", "httpcore")):
            if _REDACT not in logger.filters:
                logger.addFilter(_REDACT)


async def _remote_call(name: str, arguments: dict, key: str, timeout: float):
    # 拼接高德服务地址
    # endpoint 固定在官方域名，Key 只由后端环境变量注入。
    url = "https://mcp.amap.com/mcp?" + urlencode({"key": key})
    _protect_sdk_logs()
    # 设置整段操作的超时限制
    async with asyncio.timeout(timeout):
        # Client 自动管理连接与初始化；legacy 保持原有 initialize 握手方式。
        async with Client(url, mode="legacy") as client:
            return await client.call_tool(name, arguments)


def _decode_result(result) -> dict:
    if result.is_error:
        raise ValueError("MCP 工具返回错误")
    data = result.structured_content
    if data is None:
        text = "".join(block.text for block in result.content if block.type == "text")
        if len(text) > 100_000:
            raise ValueError("MCP 返回值过长")
        data = json.loads(text)
    if not isinstance(data, dict) or data.get("status") in ("0", 0):
        raise ValueError("MCP 返回值无效")
    return data


def _call(name: str, arguments: dict) -> dict:
    # 检查工具名称是否允许
    if name not in TOOL_NAMES:
        return _error("TOOL_NOT_ALLOWED", "不支持的地图工具。")

    # 读取高德地图Key
    key = os.getenv("AMAP_MAPS_API_KEY", "").strip()
    if not key:
        return _error("NOT_CONFIGURED", "天气与定位服务尚未配置高德 Key，暂时无法查询。")

    # 是否超时
    try:
        timeout = float(os.getenv("AMAP_MCP_TIMEOUT", "20"))
        if not 1 <= timeout <= 60:
            raise ValueError
    except ValueError:
        return _error("INVALID_CONFIG", "地图服务超时配置错误，应为 1–60 秒。")

    # 执行远程调用，并解析结果
    try:
        # 建立MCP通信
        result = asyncio.run(_remote_call(name, arguments, key, timeout))
        data = _decode_result(result)
        return {"ok": True, "data": data}
    except Exception:
        # 不传播原始异常、URL 或供应商错误正文，避免泄露 Key。
        return _error("MCP_UNAVAILABLE", "地图服务调用失败或返回无效数据，请稍后重试；不能据此推测天气或城市。")


def _fields(data: dict, fields) -> dict:
    return {key: value for key in fields
            if isinstance((value := data.get(key)), (str, int, float))
            and not isinstance(value, bool) and str(value).strip() and len(str(value)) <= 200}


"""
# 1. 调用 weather，获得 Python 字典
result = weather(city)
# 2. 将字典转换为 JSON 字符串
result_json = json.dumps(result, ensure_ascii=False)
# 3. 将字符串作为工具执行结果返回
return result_json
"""
def weather(city: str) -> dict:
    # 清理输入内容，去掉首位空格
    city = city.strip()
    # 拒绝空字符串，模糊字符串
    if not city or len(city) > 80 or city in {"我这里", "当地", "未知", "用户所在城市"}:
        return _error("CITY_REQUIRED", "请提供明确的城市名称或行政区划编码。")
    
    # 调用高德地图MCP
    result = _call("maps_weather", {"city": city})
    if not result["ok"]:
        return result

    # 这里MCP会有两种返回格式，要都考虑
    data = result["data"]
    forecasts = data.get("forecasts")
    # 兼容官方 MCP 的 {city, forecasts: casts} 和 Web 服务原始 forecasts 结构。
    if isinstance(forecasts, list) and forecasts and isinstance(forecasts[0], dict):
        if "casts" in forecasts[0]:
            data = forecasts[0]
            forecasts = data.get("casts")
    if not isinstance(forecasts, list):
        return _error("NO_WEATHER_DATA", "未取得有效天气预报，请核对城市；不提供猜测数据。")

    # days是每天的天气预报组成的列表
    days = [_fields(item, FORECAST_FIELDS) for item in forecasts[:7] if isinstance(item, dict)]
    # 去掉缺少必要信息的预报
    days = [day for day in days if day.get("date") and (day.get("dayweather") or day.get("nightweather"))]
    
    location = _fields(data, ("city", "province", "adcode", "reporttime"))
    if not days or not location.get("city"):
        return _error("NO_WEATHER_DATA", "未取得有效天气预报，请核对城市；不提供猜测数据。")
    return {
        "ok": True, "source": SOURCE, "kind": "forecast", "requested_city": city,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "data": {**location, "forecasts": days},
        "limitations": "这是日期对应的天气预报，不是实时观测；查询时间不等于数据发布时间。未提供湿度、AQI或降雨概率时不得补写。",
    }


def locate(client_ip: str | None) -> dict:
    ip = public_ipv4(client_ip)
    if not ip:
        return _error("LOCATION_UNAVAILABLE", "无法取得访问者的公网 IPv4，不能定位。请告诉我你所在的城市；不会使用服务器 IP 代替。")
    result = _call("maps_ip_location", {"ip": ip})
    if not result["ok"]:
        return result
    data = _fields(result["data"], ("city", "province", "adcode"))
    if not data.get("city"):
        return _error("CITY_NOT_FOUND", "该 IP 未返回明确城市，请直接告诉我你所在的城市。")
    return {
        "ok": True, "source": SOURCE, "method": "ip", "approximate": True,
        "data": data,
        "limitations": "仅为 IP 推测城市，可能受 VPN、代理及运营商出口影响；用户提供的城市优先。",
    }


def _self_test():
    """在本文件内离线验证参数、安全边界和返回值，不需要真实高德 Key。"""
    from unittest.mock import AsyncMock, patch
    from mcp.types import CallToolResult, TextContent

    for value in (None, "testclient", "127.0.0.1", "10.1.2.3", "192.168.1.2", "172.20.0.1", "100.64.0.1", "::1", "224.0.0.1", "203.0.113.1"):
        assert public_ipv4(value) is None
    assert public_ipv4("8.8.8.8") == "8.8.8.8"
    assert public_ipv4("::ffff:8.8.8.8") == "8.8.8.8"
    fixture = {"city": "北京市", "forecasts": [{"date": "2026-10-27", "dayweather": "晴", "daytemp": "18"}]}
    text_result = CallToolResult(content=[TextContent(type="text", text=json.dumps(fixture))])
    assert _decode_result(text_result) == fixture
    assert _decode_result(CallToolResult(content=[], structured_content=fixture)) == fixture
    with patch.dict(os.environ, {"AMAP_MAPS_API_KEY": "", "AMAP_MCP_TIMEOUT": "20"}):
        assert weather("北京")["code"] == "NOT_CONFIGURED"
    with patch.dict(os.environ, {"AMAP_MAPS_API_KEY": "private-test-key", "AMAP_MCP_TIMEOUT": "20"}), patch(
        __name__ + "._remote_call", new_callable=AsyncMock, return_value=text_result
    ) as remote:
        result = weather("北京")
        assert result["ok"] and result["kind"] == "forecast"
        assert "AQI" not in result["data"]
        assert remote.call_args.args[:2] == ("maps_weather", {"city": "北京"})
        before = remote.call_count
        assert locate("127.0.0.1")["code"] == "LOCATION_UNAVAILABLE"
        assert remote.call_count == before
        remote.return_value = CallToolResult(content=[], structured_content={"city": "北京市", "province": "北京市", "adcode": "110000", "ip": "8.8.8.8"})
        location = locate("8.8.8.8")
        assert location["ok"] and location["approximate"]
        assert "8.8.8.8" not in json.dumps(location)
        assert remote.call_args.args[:2] == ("maps_ip_location", {"ip": "8.8.8.8"})
        remote.return_value = CallToolResult(content=[], structured_content={"city": [], "province": []})
        assert not locate("8.8.8.8")["ok"]
        remote.return_value = CallToolResult(content=[], structured_content={"city": "北京", "forecasts": []})
        assert not weather("北京")["ok"]
        remote.return_value = CallToolResult(content=[TextContent(type="text", text="private-test-key")], is_error=True)
        assert "private-test-key" not in json.dumps(weather("北京"))
        remote.side_effect = TimeoutError("https://mcp.amap.com/mcp?key=private-test-key")
        assert "private-test-key" not in json.dumps(weather("北京"))
        record = logging.LogRecord("mcp", logging.ERROR, "", 0, "URL %s", ("?key=private-test-key",), None)
        _REDACT.filter(record)
        assert "private-test-key" not in record.getMessage()
    print("MCP 适配验证通过：参数、结果解析、天气预报边界、IP 过滤、异常脱敏和缺少配置。")


def _self_test_transport():
    """启动临时本地 MCP，验证真实 HTTP 握手和 tools/call；不访问高德。"""
    import socket
    import threading
    import time
    from unittest.mock import patch
    import uvicorn
    from mcp.server import MCPServer

    server = MCPServer("local-maps-check")

    @server.tool()
    def maps_weather(city: str) -> dict:
        return {"city": city, "forecasts": [{"date": "2026-10-27", "dayweather": "晴"}]}

    @server.tool()
    def maps_ip_location(ip: str) -> dict:
        assert ip == "8.8.8.8"
        return {"city": "北京市", "adcode": "110000"}

    app = server.streamable_http_app(json_response=True, stateless_http=True)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    runner = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=lambda: runner.run(sockets=[sock]), daemon=True)
    thread.start()
    original_client = Client

    def local_client(url, **kwargs):
        assert url.startswith("https://mcp.amap.com/mcp?key=")
        return original_client(f"http://127.0.0.1:{port}/mcp", **kwargs)

    try:
        deadline = time.monotonic() + 10
        while not runner.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert runner.started, "本地 MCP 未能启动"
        with patch.dict(os.environ, {"AMAP_MAPS_API_KEY": "local-test-key", "AMAP_MCP_TIMEOUT": "5"}), patch(
            __name__ + ".Client", side_effect=local_client
        ):
            assert weather("北京")["ok"]
            assert locate("8.8.8.8")["data"]["city"] == "北京市"
    finally:
        runner.should_exit = True
        thread.join(timeout=5)
        sock.close()
    assert not thread.is_alive(), "本地 MCP 未正常关闭"
    print("MCP HTTP 协议验证通过：initialize、tools/call、结构化返回和连接关闭。")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    from utils.path_tool import get_abs_path

    load_dotenv(get_abs_path(".env"), override=False, interpolate=False)
    parser = argparse.ArgumentParser(description="离线验证或使用真实高德 MCP 查询")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-transport", action="store_true")
    parser.add_argument("--city", help="真实查询指定城市天气，需要高德 Key")
    parser.add_argument("--ip", help="真实查询指定公网 IPv4 的推测城市，需要高德 Key")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    elif args.self_test_transport:
        _self_test_transport()
    elif args.city or args.ip:
        output = weather(args.city) if args.city else locate(args.ip)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        raise SystemExit(0 if output["ok"] else 1)
    else:
        parser.print_help()
