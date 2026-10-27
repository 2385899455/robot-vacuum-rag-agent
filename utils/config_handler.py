"""加载项目配置：进程环境变量 > 根目录 .env > YAML 文件。"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.path_tool import get_abs_path


def _boolean(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise ValueError("布尔配置只能填写 true 或 false")
    return value.lower() == "true"


MODEL_FIELDS = {
    "name": str,
    "provider": str,
    "base_url": str,
    "api_key": str,
    "use_responses_api": _boolean,
    "check_embedding_ctx_length": _boolean,
    "timeout": float,
    "max_retries": int,
    "temperature": float,
}


def _read_yaml(path, encoding="utf-8"):
    with open(path, encoding=encoding) as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是键值映射：{path}")
    return config


def load_rag_config(config_path=None, encoding="utf-8"):
    # Docker 镜像不复制私有 rag.yml；从公开模板加载，再由环境变量覆盖。
    if config_path is None:
        local = Path(get_abs_path("config/rag.yml"))
        config_path = local if local.exists() else get_abs_path("config/rag.example.yml")
    config = _read_yaml(config_path, encoding)
    for section in ("chat_model", "embedding_model", "router_model"):
        values = dict(config.get(section) or {})
        for field, convert in MODEL_FIELDS.items():
            variable = f"{section}_{field}".upper()
            raw = os.getenv(variable)
            # 可选变量留空意味着继承 YAML 或聊天模型。
            if raw is not None and raw.strip():
                try:
                    values[field] = convert(raw)
                except ValueError:
                    raise ValueError(f"环境变量 {variable} 格式错误") from None
        config[section] = values
    return config


def load_chroma_config(config_path=None, encoding="utf-8"):
    config = _read_yaml(config_path or get_abs_path("config/chroma.yml"), encoding)
    for key in ("persist_directory", "md5_hex_store", "data_path", "collection_name"):
        value = os.getenv(f"CHROMA_{key.upper()}")
        if value:
            config[key] = value
    return config


def load_prompts_config(config_path=None, encoding="utf-8"):
    return _read_yaml(config_path or get_abs_path("config/prompts.yml"), encoding)


def load_agent_config(config_path=None, encoding="utf-8"):
    return _read_yaml(config_path or get_abs_path("config/agent.yml"), encoding)


def _self_test():
    """用临时配置离线验证环境覆盖，不输出真实连接信息。"""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yml"
        path.write_text("chat_model:\n  name: yaml-model\n  api_key: yaml-key\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            assert load_rag_config(path)["chat_model"]["name"] == "yaml-model"
        with patch.dict(os.environ, {
            "CHAT_MODEL_NAME": "env-model", "CHAT_MODEL_API_KEY": "env-key",
            "CHAT_MODEL_USE_RESPONSES_API": "false", "CHAT_MODEL_TIMEOUT": "30",
            "ROUTER_MODEL_NAME": "", "EMBEDDING_MODEL_CHECK_EMBEDDING_CTX_LENGTH": "true",
        }, clear=True):
            config = load_rag_config(path)
            assert config["chat_model"]["name"] == "env-model"
            assert config["chat_model"]["api_key"] == "env-key"
            assert config["chat_model"]["use_responses_api"] is False
            assert config["chat_model"]["timeout"] == 30.0
            assert "name" not in config["router_model"]
            assert config["embedding_model"]["check_embedding_ctx_length"] is True
        with patch.dict(os.environ, {"CHAT_MODEL_USE_RESPONSES_API": "invalid"}, clear=True):
            try:
                load_rag_config(path)
            except ValueError as error:
                assert "CHAT_MODEL_USE_RESPONSES_API" in str(error)
            else:
                raise AssertionError("无效布尔值没有被拒绝")
        with patch.dict(os.environ, {"CHROMA_PERSIST_DIRECTORY": "/state/chroma"}, clear=True):
            assert load_chroma_config()["persist_directory"] == "/state/chroma"
        env_path = Path(directory) / ".env"
        env_path.write_text("CHAT_MODEL_NAME=dotenv-model\n", encoding="utf-8")
        with patch.dict(os.environ, {"CHAT_MODEL_NAME": "process-model"}, clear=True):
            load_dotenv(env_path, override=False, interpolate=False)
            assert load_rag_config(path)["chat_model"]["name"] == "process-model"
    print("配置离线验证通过：环境覆盖、类型转换、可选继承、持久化路径及错误处理。")


load_dotenv(get_abs_path(".env"), override=False, interpolate=False)
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

if __name__ == "__main__":
    _self_test()
