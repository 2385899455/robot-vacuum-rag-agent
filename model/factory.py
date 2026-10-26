from abc import ABC, abstractmethod
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from utils.config_handler import rag_conf
from langchain.chat_models import init_chat_model



class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Embeddings | BaseChatModel | None:
        pass


class ChatModelFactory(BaseModelFactory):
    def __init__(self, overrides: dict | None = None):
        self.overrides = overrides or {}

    def generator(self) -> BaseChatModel:
        chat_conf = {**rag_conf.get("chat_model", {}), **self.overrides}
        kwargs = {
            "model": chat_conf.get("name", rag_conf.get("chat_model_name")),
            "model_provider": chat_conf.get(
                "provider", rag_conf.get("chat_provider", "openai")
            ),
            "base_url": chat_conf.get("base_url", rag_conf.get("base_url")),
            "api_key": chat_conf.get("api_key", rag_conf.get("api_key")),
        }
        # Some OpenAI-compatible gateways expose only the Responses API.  Keep
        # this opt-in so existing Chat Completions configurations remain valid.
        if "use_responses_api" in chat_conf:
            kwargs["use_responses_api"] = bool(chat_conf["use_responses_api"])
        for key in ("timeout", "max_retries", "temperature"):
            if key in chat_conf:
                kwargs[key] = chat_conf[key]
        model = init_chat_model(**kwargs)
        return model


class RouterModelFactory(ChatModelFactory):
    """独立路由实例；未配置的连接信息继承 chat_model。"""

    def __init__(self):
        super().__init__({
            "timeout": 20,
            "max_retries": 0,
            **(rag_conf.get("router_model") or {}),
        })

class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        embedding_conf = rag_conf.get("embedding_model", {})
        provider = embedding_conf.get(
            "provider", rag_conf.get("embedding_provider", "dashscope")
        )
        model = embedding_conf.get(
            "name", rag_conf.get("embedding_model_name")
        )
        api_key = embedding_conf.get("api_key", rag_conf.get("api_key"))
        base_url = embedding_conf.get("base_url", rag_conf.get("base_url"))

        # qwen 的 compatible-mode 地址也遵循 OpenAI Embeddings 协议。
        # 只有明确使用 DashScope 原生接口时，才走 DashScopeEmbeddings。
        if provider == "openai" or (
            provider == "qwen" and base_url
        ):
            kwargs = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
            }
            # Some compatible embedding gateways reject the token-id payload
            # used by OpenAIEmbeddings' length-safety path and require strings.
            if "check_embedding_ctx_length" in embedding_conf:
                kwargs["check_embedding_ctx_length"] = bool(
                    embedding_conf["check_embedding_ctx_length"]
                )
            return OpenAIEmbeddings(**kwargs)

        return DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
        )


chat_model: BaseChatModel = ChatModelFactory().generator()
embed_model: Embeddings = EmbeddingFactory().generator()
