# 扫地机器人智能客服 Agent

这是一个基于 LangChain Agent、LangGraph Middleware、Chroma 和 Embedding 模型实现的扫地机器人智能客服示例项目。

项目面向学习和演示场景，包含两条主要能力链路：

- **RAG 问答**：读取扫地机器人相关的 TXT/PDF 文档，切分后写入 Chroma 向量数据库，再根据用户问题检索参考资料并生成中文回答。
- **ReAct Agent**：由模型根据用户意图选择工具，完成天气查询、知识库检索、用户数据查询和个人使用报告生成等流程。

## 功能概览

- 使用 `create_agent` 创建 Agent。
- 使用 `@tool` 将 Python 函数注册为 Agent 工具。
- 使用 `stream_mode="values"` 流式获取 Agent 消息状态。
- 使用 `RecursiveCharacterTextSplitter` 切分知识库文档。
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
RecursiveCharacterTextSplitter 切分
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
模型判断是否需要工具
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

项目当前未提交 `requirements.txt`，可以先安装代码中使用的主要依赖：

```bash
pip install \
  langchain \
  langchain-core \
  langchain-community \
  langchain-chroma \
  langchain-openai \
  langchain-text-splitters \
  pyyaml \
  pypdf \
  dashscope \
  httpx
```

如果项目已有可用的 `.venv`，优先在该虚拟环境中安装依赖。

## 本地配置

真实的模型配置文件是本地文件：

```text
config/rag.yml
```

该文件被 `.gitignore` 忽略，不会上传到 GitHub。首次从远程仓库获取项目时，可以复制示例配置：

```bash
cp config/rag.example.yml config/rag.yml
```

然后根据实际模型服务修改：

```yaml
chat_model_name: qwen3.7-max
embedding_model_name: qwen3.7-text-embedding
base_url: "你的 OpenAI 兼容接口地址"
api_key: "你的本地 API Key"
```

不要把真实 API Key 写入 README、示例配置或其他会提交到 Git 的文件。如果 Key 曾经进入过 Git 历史或被上传过，应立即在服务平台撤销并重新生成。

其他配置文件：

- `config/chroma.yml`：Chroma 集合名、持久化目录、检索数量 `k`、文档切分参数。
- `config/prompts.yml`：普通提示词、RAG 提示词和报告提示词的路径。
- `config/agent.yml`：外部 CSV 数据文件路径。

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

### 2. 单独测试 RAG 服务

```bash
python3 -m rag.rag_service
```

该命令会使用内置测试问题进行一次检索和总结。它只负责查询已经存在的向量库，不会自动导入原始文档，所以必须先运行 `rag.vector_store`。

### 3. 运行 Agent

```bash
python3 -m agent.react_agent
```

当前入口会创建 `ReactAgent`，以流式方式执行示例问题：

```text
给我生成一个使用报告
```

Agent 的核心调用也可以在代码中使用：

```python
from agent.react_agent import ReactAgent

agent = ReactAgent()

for chunk in agent.execute_stream("小户型适合哪种扫地机器人？"):
    print(chunk, end="", flush=True)
```

## 主要工具

工具定义位于 `agent/tools/agent_tools.py`：

| 工具 | 作用 | 当前实现 |
| --- | --- | --- |
| `rag_summarize` | 从 Chroma 检索扫地机器人资料并总结 | 可用的 RAG 流程 |
| `get_weather` | 获取城市天气信息 | 当前返回示例字符串 |
| `get_user_location` | 获取用户所在城市 | 当前从示例城市中随机返回 |
| `get_user_id` | 获取用户 ID | 当前从示例 ID 中随机返回 |
| `get_current_month` | 获取月份 | 当前从示例月份中随机返回 |
| `fetch_external_data` | 查询用户指定月份的 CSV 使用记录 | 读取 `data/external/records.csv` |
| `fill_context_for_report` | 标记进入报告生成场景 | 由 Middleware 更新运行时上下文 |

天气、用户位置、用户 ID 和月份工具目前主要用于演示 Agent 的工具调用流程，并不代表已经接入真实外部系统。

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

- 天气和用户相关工具仍是示例实现，尚未连接真实接口。
- RAG 数据库和 MD5 文件是本地生成数据，不随仓库发布。
- 模型和 Embedding 服务依赖外部 API，运行时需要网络和有效凭据。
- 项目目前没有统一的测试套件和生产部署配置。
