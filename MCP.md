# 高德 MCP 天气与 IP 定位

项目使用高德官方 Streamable HTTP MCP：`https://mcp.amap.com/mcp`。后端通过 MCP Python SDK 初始化会话并执行工具调用；这是 MCP 客户端接入，没有把普通 HTTP 天气查询伪装成 MCP。

| Agent 工具 | 高德 MCP 工具 | 数据边界 |
| --- | --- | --- |
| `get_weather(city)` | `maps_weather(city)` | 按日期的天气预报，不能等同于实时观测 |
| `get_user_location()` | `maps_ip_location(ip)` | 公网 IPv4 的推测城市，不是精确定位 |

仅这两个远程工具可被适配层调用。现有场景路由、工具名称和调用预算继续生效，其他高德工具没有开放给 Agent。

## 配置

在高德开放平台创建应用，再添加服务平台为 **Web 服务** 的 Key。填写项目根目录的私有 `.env`：

```dotenv
AMAP_MAPS_API_KEY=填写你自己的高德Key
AMAP_MCP_TIMEOUT=20
```

Key 只放在后端，不放入 Vue、不提交 Git，也不需要发送到聊天中。`AMAP_MCP_TIMEOUT` 限制整次 MCP 会话（含初始化与工具调用），允许 1–60 秒。不自动重试失败调用，避免超出场景预算和拖长响应。

本地 Python 模式需安装新增依赖，并重启后端：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m api.main
```

Docker 首次更新代码：

```bash
docker compose up --build -d
```

仅修改 `.env` 中 Key 时无需重新构建镜像，但需要重新创建后端容器：

```bash
docker compose up -d --force-recreate api
```

未设置 Key 时服务仍可启动，RAG 和其他工具仍可使用；天气/公网 IP 定位会返回 `NOT_CONFIGURED`，不退回模拟数据。

## 访问者 IP 从哪里来

```text
访问者 → Nginx 观察到的 remote_addr
       → 覆盖 X-Forwarded-For
       → Uvicorn 处理可信代理头
       → FastAPI request.client.host
       → 每次 Agent 请求独立的 runtime.context.client_ip
       → get_user_location → 高德 maps_ip_location
```

IP 是框架注入参数，不属于模型可填写的工具参数，也不接受聊天 JSON 中自行添加的 `client_ip`。不查询服务器的出口 IP，不从全局变量缓存用户位置。高德返回的 IP、矩形边界等多余字段不传给模型，只保留省、市、行政区编码及近似定位说明。

当前 Compose 的 API **没有发布宿主机端口**，只由内部 Nginx 访问，因此设置 `FORWARDED_ALLOW_IPS=*`，同时让 Nginx 用自己的 `remote_addr` 覆盖转发链。若以后直接对外发布 API 端口，必须收紧可信代理列表；不要沿用 `*` 接受任意代理头。若前面再加 CDN 或另一层反向代理，需要单独配置可信来源，不能直接信任浏览器的头信息。

本机访问 `127.0.0.1:8080` 时，后端通常只能看到回环或 Docker 内网地址，无法通过它定位。此时工具返回 `LOCATION_UNAVAILABLE`，Agent 会请你提供城市，例如“北京今天的天气怎么样”。不会拿服务器地址或随机城市代替。

IPv6、内网、回环、保留地址，以及未返回城市的公网 IP 都会明确失败。公网 IPv4 查询成功也只代表 IP 推测城市，VPN、代理及运营商出口可能导致城市不准确；用户主动提供的城市优先。

## 返回值与错误

正常模式返回 JSON 字符串：

- 成功：`ok=true`，包含 `source`、`data` 和能力限制。天气标记 `kind=forecast`；定位标记 `approximate=true`。
- 失败：`ok=false`，包含 `code`、`message`，不回显原始异常或带 Key 的 URL。
- 天气只保留供应商实际返回的日期、天气、温度、风向风力等字段；不补造湿度、AQI 或降雨概率。`queried_at` 只是查询时间，不是观测时间。
- 定位失败后，工具中间件会阻止未知城市场景继续调用天气工具，要求先询问用户城市。

## 验证

下列验证都放在对应源码文件的 `__main__` 入口，没有新建独立测试文件：

```bash
.venv/bin/python -m agent.tools.mcp_maps --self-test
.venv/bin/python -m agent.tools.mcp_maps --self-test-transport
.venv/bin/python -m agent.tools.agent_tools --self-test
.venv/bin/python -m api.main --self-test
```

`--self-test-transport` 临时启动本地 MCP 服务，验证真正的 HTTP 初始化和 `tools/call`，不访问高德。它不能代替真实高德 Key 的线上验证。

填好 Key 后，先绕过 LLM 单独验证高德服务：

```bash
.venv/bin/python -m agent.tools.mcp_maps --city 北京
# 在容器里验证同一条调用：
docker compose exec api python -m agent.tools.mcp_maps --city 北京
```

IP 定位可以使用该模块的 `--ip` 参数传入你自己的公网 IPv4 进行手动检查。真实查询会把城市或 IP 发送给高德。

`AGENT_EVAL_MODE=1` 仅用于原有确定性评测，天气和位置仍返回固定样本以保持测试可比性。不要在正常服务中设置该变量；旧评测分数不能证明高德数据可用或定位准确。

官方参考：[高德 MCP 快速接入](https://developer.amap.com/api/mcp-server/gettingstarted)、[能力说明](https://developer.amap.com/api/mcp-server/summary)、[创建 Web 服务 Key](https://developer.amap.com/api/mcp-server/create-project-and-key)、[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)。
