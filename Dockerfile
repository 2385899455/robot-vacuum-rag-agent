# 本文件使用 Dockerfile 指令语法，由 docker compose build 等构建命令读取。
# FROM、COPY、RUN 等用于构建镜像；CMD 指定容器启动时默认执行的命令。
# COPY 保存构建时的文件快照，本地源码修改后需要重新构建镜像并更新容器。

# 选择包含 Python 3.12 的精简基础镜像。
# 明确使用 Docker Hub 官方 Registry，避免本机失效的镜像加速器返回 HTML。
FROM registry-1.docker.io/library/python:3.12-slim

# 设置后续构建步骤和容器运行时使用的环境变量：
# PYTHONDONTWRITEBYTECODE=1：不生成 Python 字节码缓存文件（.pyc）。
# PYTHONUNBUFFERED=1：不缓冲标准输出和错误输出，方便及时查看容器日志。
# PIP_NO_CACHE_DIR=1：禁用 pip 下载缓存，减少镜像中的缓存文件。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
# 设置镜像内的工作目录；目录不存在时自动创建。
# 后续 COPY 的相对目标路径，以及 RUN、CMD 的执行目录，都以 /app 为基准。
WORKDIR /app

# 从构建上下文（本项目为项目根目录）复制依赖清单到 /app/。
# 先安装依赖再复制源码：仅修改业务代码时，通常可以复用依赖安装缓存层。
COPY requirements.txt requirements-api.txt ./
# RUN 在构建阶段执行 Shell 命令，安装结果保存在镜像中。
# requirements.txt 会引用 requirements-api.txt，一并安装后端 HTTP 服务依赖。
# && 表示前一条命令成功后才执行后一条；pip check 检查已安装包的依赖兼容性。
RUN pip install -r requirements.txt && pip check

# COPY 的左侧为本地构建上下文中的路径，右侧为镜像中的目标路径。
# 复制 FastAPI 接口代码到 /app/api。
COPY api ./api
# 复制 Agent 编排、场景路由及工具代码到 /app/agent。
COPY agent ./agent
# 复制模型初始化与调用代码到 /app/model。
COPY model ./model
# 复制知识检索、文档处理与向量库代码到 /app/rag。
COPY rag ./rag
# 复制配置读取、日志及路径等公共工具到 /app/utils。
COPY utils ./utils
# 复制允许进入构建上下文的配置文件；私有 config/rag.yml 由 .dockerignore 排除。
# .env 同样不打包进镜像，运行时由 Compose 注入环境变量。
COPY config ./config
# 复制提示词文件到 /app/prompts。
COPY prompts ./prompts

# 在构建阶段创建固定 UID/GID 的普通用户 agent，不创建其主目录。
# 创建持久数据和日志目录，并将目录所有权交给 agent，使其运行时可以写入。
RUN groupadd --gid 10001 agent && useradd --uid 10001 --gid agent --no-create-home agent \
    && mkdir -p /app/storage /app/logs \
    && chown -R agent:agent /app/storage /app/logs
# 后续指令及容器默认启动进程使用普通用户 agent，避免以 root 身份运行应用。
USER agent
# 声明应用使用容器内的 8000 端口；此指令本身不会将端口发布到宿主机。
# 当前 Compose 通过内部网络让 Nginx 访问 api:8000。
EXPOSE 8000
# 容器启动时执行的默认命令（构建镜像时不会启动服务）。
# 使用 JSON 数组形式直接启动 Python；-m api.main 表示运行 api.main 模块。
# 监听容器内所有 IPv4 网络接口的 8000 端口，供 Nginx 容器访问。
# 保持单进程：Agent 缓存和忙碌锁目前只在单进程内协调。
CMD ["python", "-m", "api.main", "--host", "0.0.0.0", "--port", "8000"]
