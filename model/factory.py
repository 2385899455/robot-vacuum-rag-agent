from abc import ABC, abstractmethod
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from utils.config_handler import rag_conf
from langchain.chat_models import init_chat_model



class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Embeddings | BaseChatModel | None:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        model = init_chat_model(
            model = rag_conf["chat_model_name"],
            model_provider = "openai",
            base_url = rag_conf["base_url"],
            api_key = rag_conf["api_key"],
        )
        return model

class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=rag_conf["api_key"],
        )


chat_model: BaseChatModel = ChatModelFactory().generator()
embed_model: Embeddings = EmbeddingFactory().generator()
