# 本地 Docker 部署

目标：通过 Compose 运行 Vue 构建产物、Nginx 和 FastAPI。本机访问地址为 `http://127.0.0.1:8080`，可以与开发模式的 8000 / 5173 端口同时存在。

```text
浏览器 → 127.0.0.1:8080 → Nginx
                         ├─ /、/assets/：Vue 静态文件
                         └─ /api/、/health：api:8000 → FastAPI → Agent
```

## 1. 准备配置

安装并启动 Docker Desktop。需要 Docker Compose 2.30 或更高版本（本项目使用 `env_file.format: raw`，避免对密钥中的美元符号进行变量展开）。所有 Compose 命令在项目根目录执行：

```bash
cd "/Users/a/Documents/vscode/Agent智能体/Agent项目工程"
docker info
docker compose version
```

首次获取项目且没有 `.env` 时，复制 `.env.example` 为 `.env`，填写模型名称、服务地址和密钥。已有 `.env` 时不要覆盖。

配置优先级：进程环境变量 > `.env` > 私有 `config/rag.yml` > 公开示例模板（后两者择一）。不要为值添加引号，也不要写行尾注释；特殊字符将作为原始值读取。聊天模型的 `CHAT_MODEL_USE_RESPONSES_API` 必须与供应商协议一致。路由模型连接字段留空时继承聊天模型。

Docker 不复制 `.env`、私有 `config/rag.yml`、本机虚拟环境、历史评测结果或数据库。模型配置只在运行时注入后端。前端构建不接收模型密钥。`data/` 以只读方式挂载，复制项目到服务器时也要带上该目录。

## 2. 构建并初始化知识库

```bash
docker compose config --quiet
docker compose build
docker compose run --rm --no-deps api python -m rag.vector_store --ingest-only
```

第三条命令真实调用 Embedding 服务，会产生相应调用费用；不调用聊天模型。成功时会输出文档块数量，失败时返回非零退出码。成功导入的文件会通过 MD5 跳过，失败后可修复配置并重跑。若只想先验证普通问候，可以暂时跳过导入，但知识检索将没有参考资料。

容器使用独立的 `agent_storage` 命名卷，首次为空，**不会自动复制本机 `chroma_db/` 中的测试向量**。向量库和 MD5 清单保存在同一卷，必须一起备份和恢复。已有向量后不要直接更换 Embedding 模型，否则维度或语义空间可能不匹配；应建立新的库并重新导入。

更新知识库时先停止服务，再执行导入命令，避免两个进程同时访问本地 Chroma。现有导入是增量追加，修改或删除源文件不会自动清理旧分片；需要完整刷新时应先备份，再建立新库。

## 3. 启动与验收

```bash
docker compose up --build -d
docker compose ps
curl --fail http://127.0.0.1:8080/health
```

打开 `http://127.0.0.1:8080`，发送“你好”，再发送一个知识库问题。也可以测试接口（会调用真实模型）：

```bash
curl --fail-with-body http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好"}'
```

`/health` 只表示 Agent 已完成本地初始化，不证明模型密钥可用或知识库已导入。Nginx 在 API 健康检查通过后启动。后端仅在容器网络监听 8000，没有映射宿主机端口；浏览器统一通过 8080 访问，所以不需要添加跨域放行规则。

目前保留单进程与单个 Agent 实例，同时来的第二个问答请求返回 503。代理与前端等待上限均为 180 秒，但超时或关闭网页不一定能取消正在执行的 Python/模型调用。`CHAT_MODEL_TIMEOUT` 限制单次模型请求，不代表整个 Agent 的总耗时。

## 4. 日常操作

```bash
# 查看日志（Ctrl+C 只退出查看）
docker compose logs --tail=100 -f

# 停止并移除本项目容器，命名卷仍保留
docker compose down

# 下次启动
docker compose up -d

# 修改代码后重建
docker compose up --build -d

# 修改 .env 后重新创建 API 容器以应用配置
docker compose up -d --force-recreate api
```

不要为了普通停止使用 `docker compose down -v`：`-v` 会删除向量库、MD5 清单和日志卷。

| 文件或数据 | 用途 |
| --- | --- |
| `requirements.txt` | 固定 Python 直接依赖；间接依赖由 pip 解析 |
| `Dockerfile` | 打包 Python 后端，使用非 root 用户运行 |
| `frontend/Dockerfile` | Node 构建 Vue，再将 dist 放入 Nginx 镜像 |
| `frontend/nginx.conf` | 静态资源、单页路由和 API 转发 |
| `compose.yml` | 服务启动顺序、端口、卷和健康检查 |
| `.env` | 本机私有配置，权限建议 600 |
| `data/` | 原始知识文件及模拟 CSV，只读挂载 |
| `agent_storage` 卷 | 向量库和 MD5 清单 |
| `agent_logs` 卷 | 应用日志；需按使用量定期归档清理 |

## 5. 常见问题

- 无法连接 Docker daemon：启动 Docker Desktop，等待 `docker info` 成功。
- 拉取基础镜像出现 `unknown type text/html`：检查 Docker Desktop 的镜像加速器配置；某些失效的代理返回了网页而不是镜像。不要修改业务代码来解决此问题。
- 8080 被占用：修改 `.env` 的 `WEB_PORT`，再执行 `docker compose up -d`，使用新的端口访问。
- API 不健康：查看 `docker compose logs --tail=100 api`，确认模型配置和数据目录；不要把完整环境变量或密钥贴到公开渠道。
- 网页返回 502：检查 API 健康状态及模型服务连接；HTTP 代理正常不等于第三方模型可访问。
- 普通问候成功但知识问题没有资料：检查是否已执行 `--ingest-only`，以及 Embedding 配置是否与建库时一致。

本配置只绑定本机回环地址，用于学习部署。公开给其他人使用前还需配置访问认证、请求限流和 HTTPS；当前天气、身份、月份和 CSV 仍是演示数据。

参考：[Compose 启动顺序与健康检查](https://docs.docker.com/compose/how-tos/startup-order/)、[Nginx 代理配置](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_pass)。
