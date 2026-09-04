from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

# 启动时自动从项目根目录的 .env 加载环境变量
load_dotenv()

from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """对话模型：走 OpenAI 兼容中转站。
        base_url 在 rag.yml 的 chat_base_url 配置，API key 走环境变量 OPENAI_API_KEY。"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "未设置环境变量 OPENAI_API_KEY，对话模型（{0}）需要它来调用 {1}"
                .format(rag_conf["chat_model_name"], rag_conf["chat_base_url"])
            )
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["chat_base_url"],
            api_key=api_key,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """Embedding 模型：走 SiliconFlow 的 OpenAI 兼容接口。
        base_url 在 rag.yml 的 embedding_base_url 配置，API key 走环境变量 SILICONFLOW_API_KEY。"""
        api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "未设置环境变量 SILICONFLOW_API_KEY，Embedding 模型（{0}）需要它来调用 {1}"
                .format(rag_conf["embedding_model_name"], rag_conf["embedding_base_url"])
            )
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            base_url=rag_conf["embedding_base_url"],
            api_key=api_key,
        )


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()
