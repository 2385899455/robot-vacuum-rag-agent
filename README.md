# 扫地机器人智能客服 Agent

这是一个基于 LangChain Agent、LangGraph Middleware、Chroma 和 Embedding 模型实现的扫地机器人智能客服示例项目。

项目面向学习和演示场景，包含两条主要能力链路：

- **RAG 问答**：读取扫地机器人相关的 TXT/PDF 文档，切分后写入 Chroma 向量数据库，再根据用户问题检索参考资料并生成中文回答。
- **ReAct Agent**：由模型根据用户意图选择工具，完成天气查询、知识库检索、用户数据查询和个人使用报告生成等流程。

## 功能概览

- 使用 `create_agent` 创建 Agent。
- 使用 `@tool` 将 Python 函数注册为 Agent 工具。
- 使用 `stream_mode="values"` 流式获取 Agent 消息状态。
- 使用语义解析器按知识条目切分知识库文档，固定长度切分只作为兜底和超长保护。
- 使用 DashScope Embeddings 批量生成向量，并使用稳定的 chunk ID 写入 Chroma。
- 使用 Chroma 持久化文档向量。
- 使用文件 MD5 对已经处理过的知识文件去重。
- 使用 Middleware 记录工具调用、模型调用日志。
- 使用 `runtime.context["report"]` 标记报告生成场景，并动态切换报告提示词。
- 使用 `data/external/records.csv` 模拟个人用户月度使用数据。

## 工作流程

### RAG 流程

```text
data/*.txt、data/*.pdf
        ↓
读取文档
        ↓
识别语义类型
        ↓
按问答、故障、维护或选购规则切分
        ↓
Embedding 模型生成向量
        ↓
写入 Chroma
        ↓
用户问题生成查询向量
        ↓
返回最多 k 条参考文档
        ↓
聊天模型根据参考资料生成回答
```

### Agent 流程

```text
用户问题
    ↓
关键词路由（明确命中则直接选择场景）
    ↓ 未命中或有歧义时
独立 LLM 分类（只返回场景标签）
    ↓
按场景选择工具集合和调用上限
    ↓
Agent 模型判断是否需要工具
    ↓
生成 tool_call
    ↓
LangChain 执行对应工具
    ↓
Middleware 记录和处理调用
    ↓
模型观察工具返回结果
    ↓
继续调用工具或生成最终回答
```

当用户要求生成个人使用报告时，系统提示词规定的流程是：

```text
get_user_id
    ↓
get_current_month
    ↓
fill_context_for_report
    ↓
fetch_external_data
    ↓
生成报告
```

`fill_context_for_report` 工具调用成功后，工具 Middleware 会将本次运行上下文中的 `report` 标记改为 `True`，动态提示词 Middleware 随后加载 `prompts/report_prompt.txt`。

### 关键词优先的混合路由

`agent/routing.py` 的 `classify_intent()` 先匹配规则：明确命中时返回场景；未命中、否定表达或规则有歧义时返回 `None`。`IntentRouter.route()` 只在返回 `None` 时调用分类 LLM。例如“主刷怎么清理”直接走关键词，“它老是在原地打转”交给 LLM。纯问候才直接回答，“你好，主刷不转了”仍归为知识检索。

分类模型只允许输出 `{"intent":"rag"}` 这样的 JSON。代码校验七种合法标签，工具白名单与次数仍由 `TOOLS_BY_INTENT` 和 `INTENT_BUDGETS` 决定。模型不可增加工具或修改预算。分类服务异常、超时或格式不合法时，回退为无工具的 `direct` 场景；此时可能无法完成原请求，需要从执行轨迹发现故障。

`config/agent.yml` 默认设置 `router.mode: hybrid`；改成 `keyword` 可禁用路由模型，未识别的问题进入 `direct`。路由模型使用独立实例，默认继承 `config/rag.yml` 的 `chat_model` 配置。可添加 `router_model` 区块覆盖 `name`、`provider`、`base_url`、`api_key`、`use_responses_api`、`timeout` 和 `max_retries`。默认请求超时为 20 秒且不重试；切换模型服务时需同时检查地址、密钥及 API 协议。

`execute_with_trace()` 的 `routing` 字段记录 `intent`、`source`（`keyword` / `llm` / `direct_fallback`）、分类耗时和异常类型。只有转交 LLM 的请求才增加分类 API 调用；这次调用不计入工具次数，但计入端到端耗时。

离线验证路由分支、异常降级与响应格式：

```bash
python3 -m unittest discover -s tests -v
```

关键词规则仍可能误判，LLM 不会复核已经明确命中的规则；目前也只按本次问题分类，不读取多轮对话历史。

## 环境要求

- Python 3.10 或更高版本
- 可用的聊天模型 API
- 可用的 Embedding 模型 API

建议使用项目根目录下的虚拟环境：

```bash
cd Agent项目工程
source .venv/bin/activate
```

Windows PowerShell 可以使用：

```powershell
cd Agent项目工程
.venv\Scripts\Activate.ps1
```

## 安装依赖

已固定当前项目的直接依赖版本，包含 Agent、向量库和 FastAPI：

```bash
python -m pip install -r requirements.txt
```

Docker 使用 Python 3.12；本机已有可用 `.venv` 时可继续使用。

## 本地配置

推荐复制环境变量模板，并填入你的聊天模型与 Embedding 服务配置：

```bash
cp .env.example .env
```

配置读取优先级：**进程环境变量 > 项目根目录 `.env` > `config/rag.yml`**。没有本地 YAML 时使用 `config/rag.example.yml` 作为默认模板。`.env` 与私有 `rag.yml` 都被 Git 忽略，Docker 构建也不会包含它们。

现有 `config/rag.yml` 仍然兼容；如果同一字段在 `.env` 中有非空值，应修改 `.env`。路由模型未设置的字段继续继承聊天模型。布尔值使用 `true` / `false`。API Key 不要加 `VITE_` 前缀或写入前端。

容器部署、数据持久化和排错请阅读 [本地 Docker 部署](DEPLOYMENT.md)。

## 运行项目

所有命令都应从 `Agent项目工程` 项目根目录执行，并使用与依赖安装相同的 Python 环境。

### 1. 构建或更新向量库

首次运行 RAG 或删除了本地 `chroma_db` 后，需要先执行：

```bash
python3 -m rag.vector_store
```

该命令会：

1. 读取 `config/chroma.yml` 中配置的数据目录。
2. 加载允许的 TXT/PDF 文件。
3. 使用 MD5 跳过已经处理过且内容未变化的文件。
4. 切分文档并调用 Embedding 模型。
5. 将文档块写入本地 `chroma_db`。

`chroma_db/` 和 `md5.text` 属于本地生成数据，默认不会提交到 Git。换一台机器或清空数据库后，需要重新执行该命令。

### 2. 测试 Embedding 和 Chroma 写入

确认 `config/rag.yml` 中的 DashScope API Key 有效后，可以只处理维护保养文档的前 5 个 chunk：

```bash
python3 -m rag.vector_store --test-embedding
```

该命令会真实调用 Embedding 模型，将向量、原文和元数据写入 Chroma，并打印向量维度、部分向量值以及一次检索结果。

### 3. 单独测试 RAG 服务

```bash
python3 -m rag.rag_service
```

该命令会使用内置测试问题进行一次检索和总结。它只负责查询已经存在的向量库，不会自动导入原始文档，所以必须先运行 `rag.vector_store`。

### 4. 对比固定长度切分和语义切分

评测问题位于 `eval/retrieval_questions.jsonl`，当前包含 100 条问题。需要根据当前知识库重新生成测试集时，可以执行：

```bash
python3 -m eval.generate_retrieval_questions
```

首次运行对比实验时，使用有效的 DashScope API Key 执行：

```bash
python3 -m eval.retrieval_evaluator --build
```

该命令会分别构建 `chroma_eval/fixed` 和 `chroma_eval/semantic` 两个向量库，并输出 `Hit@1`、`Hit@5`、`Recall@5` 和 `MRR@5`。详细结果保存到：

```text
eval/fixed_results.json
eval/semantic_results.json
```

如果评测向量库已经存在，只重新评测而不重新生成文档向量；每个测试问题仍会调用一次查询 Embedding：

```bash
python3 -m eval.retrieval_evaluator --evaluate
```

需要修改测试问题时，直接编辑 `eval/retrieval_questions.jsonl`，然后重新执行 `--evaluate`；如果修改了文档或切分配置，需要重新执行 `--build`。

### 5. 运行 Agent

```bash
python3 -m agent.react_agent
```

当前入口会创建 `ReactAgent`，以流式方式执行示例问题：

```text
给我生成一个使用报告
```

如果需要查看 Agent 的完整执行记录，可以使用：

```bash
python3 -m agent.react_agent --trace
```

该入口会以 JSON 文本输出用户问题、节点执行顺序、工具名称、工具参数、工具返回值、最终答案、工具调用次数和执行耗时。代码中也可以直接调用：

```python
trace = agent.execute_with_trace("小户型适合哪种扫地机器人？")
print(trace)
```

Agent 评测测试集位于 `eval/agent_questions.jsonl`，当前包含 31 条问题，覆盖直接回答、RAG 检索、天气查询、位置与天气组合查询、个人报告、多工具调用和异常边界场景。每条记录中的 `expected_tools` 和 `expected_order` 用于检查工具选择与执行顺序，`tool_arg_rules` 用于检查参数，`answer_keywords` 用于检查最终答案，`forbidden_tools` 用于检查不应调用的工具。

运行 Agent 评测：

```bash
# 先用 1 条问题验证模型、配置和评测流程
python3 -m eval.agent_evaluator --limit 1

# 运行完整测试集
python3 -m eval.agent_evaluator
```

评测结果会保存到 `eval/agent_results.json`，其中包含总体指标、按类别统计、逐题判定以及完整执行轨迹。

比较工具描述改动前后的效果：

```bash
# 先检查流程，前 3 题都是直接回答题，不能用于判断工具描述是否改善
python3 -m eval.agent_evaluator --compare-tool-descriptions --limit 1 --output eval/tool_description_smoke.json

# 完整实验：31 题，每题分别执行修改前后两个版本，共 62 次 Agent 执行
python3 -m eval.agent_evaluator --compare-tool-descriptions
```

基线描述保存在 `eval/tool_descriptions_before.json`，实验组读取当前工具对象的描述。
两组使用同一模型、系统提示词、工具实现、参数 schema 和评测数据；每题共享一次路由分类结果，逐题交替执行先后顺序。工具对象仅在内存中复制，不回退源文件。城市、用户 ID 和月份采用评测模式的固定值；RAG 内部模型输出仍可能有波动。

结果保存到 `eval/tool_description_comparison.json`，包含 `before` / `after` 总体和分类指标、完整轨迹、描述快照、公共路由结果、指标差值及变化题目。运行中逐步保存，`complete: true` 才表示整个实验完成。两组平均执行耗时均不包含公共路由分类耗时。

自定义基线或输出文件：

```bash
python3 -m eval.agent_evaluator --compare-tool-descriptions eval/tool_descriptions_before.json --output eval/tool_description_comparison_repeat.json
```

这是单轮对比，不代表统计显著的提升；需要时换输出文件重复实验。当前指标侧重工具流程和答案关键词，无法充分衡量事实准确性、模拟数据说明和表达质量。本次描述修改同时包含结构调整与能力说明纠正，结果不能单独归因于 `Args / Returns` 格式。

Agent 的核心调用也可以在代码中使用：

```python
from agent.react_agent import ReactAgent

agent = ReactAgent()

for chunk in agent.execute_stream("小户型适合哪种扫地机器人？"):
    print(chunk, end="", flush=True)
```

## 本地 HTTP 接口

在已有 Agent 依赖和有效 `config/rag.yml` 的虚拟环境中，安装接口依赖并启动：

```bash
python3 -m pip install -r requirements-api.txt
python3 -m api.main
```

默认监听 `127.0.0.1:8000`，用于本机开发。端口被占用时可指定 `--port 8001`。
也可以使用 `python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000` 启动。

- `GET /health`：返回 `{"status":"ok","agent_ready":true}`，只检查本地初始化，不探测远端模型。
- `POST /api/chat`：接收 `{"message":"你好"}`，成功返回 `{"answer":"..."}`。
- `/docs`：浏览器中的交互式接口文档，可展开聊天接口并使用 Try it out 发起请求。

请求示例：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好"}'
```

`message` 必须是字符串，去掉首尾空格后长度为 1–2000 字符，不接受额外字段。错误输入返回 HTTP 422；执行异常返回 500；Agent 报错或空答案返回 502；未就绪或忙碌返回 503。执行错误使用 `{"detail":"..."}` 响应，参数校验错误的 `detail` 是错误列表。聊天接口只返回最终答案，不返回内部轨迹或异常正文。

每个服务进程启动时初始化一次 Agent。当前每进程只执行一个聊天请求，忙碌时立即返回 503 和 `Retry-After: 5`，不会将所有聊天请求堆入队列。每次问答使用独立上下文，当前无多轮记忆。健康检查不受聊天执行锁影响。

这是供后续 Vue 页面连接的本地接口；当前未包含公网认证和访问配额。Vue 开发时可通过 Vite 将 `/api` 代理到此服务。天气与 IP 定位通过高德 MCP 接入，配置与边界见 [MCP.md](MCP.md)；个人使用记录仍为模拟数据。

验证代码放在 `api/main.py` 内，不调用真实模型：

```bash
python3 -m api.main --self-test
```

该验证覆盖请求校验、实例复用、成功响应、失败和空答案、繁忙状态及接口文档。真实模型链路可以通过 `/docs` 或上面的 curl 命令验证。

## Vue 本地聊天页面

前端位于 `frontend/`，使用 Vue 3 + Vite，依赖版本由 `package-lock.json` 固定。
在两个终端分别启动后端和前端：

```bash
# 终端一：Agent项目工程 根目录
.venv/bin/python -m api.main
```

```bash
# 终端二：Agent项目工程 根目录
cd frontend
npm ci
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。Vite 开发代理会把 `/api` 和 `/health` 转发到
`http://127.0.0.1:8000`。如果更改后端端口，需要同步修改 `frontend/vite.config.js`。
前端使用 Node.js 和 npm，不使用 Python 虚拟环境安装前端依赖。

页面支持快捷问题填入、Enter 发送、Shift + Enter 换行、中文输入法组合输入、等待提示、
失败后手动重试、清空对话和复制回答。Markdown 回答经过 HTML 清理后展示，支持报告表格。
问题限制为去除首尾空格后的 1–2000 个字符；发送期间禁止再次发送。

页面上的聊天记录只保留在当前页面，刷新后消失，每次只发送当前问题，后端仍为单次问答。
前端等待最多 180 秒，超时不会取消后端正在运行的 Agent，不自动重试以免重复调用。
右上角状态检查仅表示本地后端就绪，不保证远端模型服务可用。

前端生产构建检查：

```bash
cd frontend
npm run build
```

产物在 `frontend/dist/`。Vite 的代理用于本地开发，正式部署时需另行配置 `/api`、`/health`
的后端转发；`npm run dev` 不作为公网生产部署方式。

## 主要工具

工具定义位于 `agent/tools/agent_tools.py`：

| 工具 | 作用 | 当前实现 |
| --- | --- | --- |
| `rag_summarize` | 从 Chroma 检索扫地机器人资料并总结 | 可用的 RAG 流程 |
| `get_weather` | 获取城市天气预报 | 高德官方 MCP，需配置 Key |
| `get_user_location` | 根据访问者公网 IPv4 推测城市 | 高德 MCP；失败时请用户提供城市 |
| `get_user_id` | 获取用户 ID | 当前从示例 ID 中随机返回 |
| `get_current_month` | 获取月份 | 当前从示例月份中随机返回 |
| `fetch_external_data` | 查询用户指定月份的 CSV 使用记录 | 读取 `data/external/records.csv` |
| `fill_context_for_report` | 标记进入报告生成场景 | 由 Middleware 更新运行时上下文 |

天气与用户位置已改为高德 MCP 适配：天气是按日期的预报，位置是公网 IPv4 的推测城市。配置与验证见 [MCP.md](MCP.md)。用户 ID、月份和个人记录仍为演示数据。

## 目录结构

```text
Agent项目工程/
├── agent/
│   ├── react_agent.py          # Agent 创建和流式执行入口
│   └── tools/
│       ├── agent_tools.py      # Agent 工具定义
│       └── middleware.py       # 工具监控、日志和动态提示词
├── config/
│   ├── agent.yml               # 外部数据路径
│   ├── chroma.yml              # 向量库和切分配置
│   ├── prompts.yml             # 提示词路径配置
│   ├── rag.yml                 # 本地模型配置，不提交
│   └── rag.example.yml         # 无密钥配置示例
├── data/
│   ├── *.txt、*.pdf            # RAG 知识库原始文件
│   └── external/records.csv    # 模拟用户使用数据
├── model/
│   └── factory.py              # 聊天模型和 Embedding 模型初始化
├── prompts/
│   ├── main_prompt.txt         # Agent 普通系统提示词
│   ├── rag_summarize.txt       # RAG 总结提示词
│   └── report_prompt.txt       # 报告生成提示词
├── rag/
│   ├── vector_store.py         # 文档加载、切分、Embedding 和入库
│   ├── semantic_parser.py      # 按语义类型切分知识条目
│   └── rag_service.py          # 检索并调用模型总结
└── utils/
    ├── config_handler.py       # YAML 配置加载
    ├── file_handler.py         # 文件读取和 MD5 计算
    ├── logger_handler.py       # 日志配置
    ├── path_tool.py            # 路径处理
    └── prompt_loader.py        # 提示词加载
```

## 常见问题

### RAG 返回“未检索到与问题相关的参考资料”

先确认向量库已经构建：

```bash
python3 -m rag.vector_store
python3 -m rag.rag_service
```

`config/chroma.yml` 中的 `k: 3` 表示最多返回 3 个文档块；如果 Chroma 集合为空，即使 `k` 设置为 3，也会返回空列表。

### `ModuleNotFoundError: No module named ...`

不要直接在项目子目录中运行带有绝对导入的文件。请回到项目根目录，以模块方式运行：

```bash
python3 -m rag.vector_store
python3 -m rag.rag_service
python3 -m agent.react_agent
```

### `InvalidApiKey`

检查本地 `config/rag.yml` 中的 `api_key`、模型名称和接口地址。该文件不会从 GitHub 获取，需要在本地自行创建和配置。

## 当前限制

- 天气/IP 定位需要配置高德 MCP Key；本地内网地址无法定位，IP 推测城市可能不准确。用户 ID、月份及个人使用记录仍为模拟数据。
- RAG 数据库和 MD5 文件是本地生成数据，不随仓库发布。
- 模型和 Embedding 服务依赖外部 API，运行时需要网络和有效凭据。
- 已提供本地 Docker Compose 部署配置；公网认证、限流和 HTTPS 尚未配置。
