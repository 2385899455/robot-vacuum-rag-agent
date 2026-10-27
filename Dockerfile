# 明确使用 Docker Hub 官方 Registry，避免本机失效的镜像加速器返回 HTML。
FROM registry-1.docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt && pip check

COPY api ./api
COPY agent ./agent
COPY model ./model
COPY rag ./rag
COPY utils ./utils
COPY config ./config
COPY prompts ./prompts

RUN groupadd --gid 10001 agent && useradd --uid 10001 --gid agent --no-create-home agent \
    && mkdir -p /app/storage /app/logs \
    && chown -R agent:agent /app/storage /app/logs
USER agent
EXPOSE 8000
# 保持单进程：Agent 缓存和忙碌锁目前只在单进程内协调。
CMD ["python", "-m", "api.main", "--host", "0.0.0.0", "--port", "8000"]
