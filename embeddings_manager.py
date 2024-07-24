from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from config import Config
import logging

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, config: Config):
        self.config = config
        self.embedding_function = OpenAIEmbeddings(
            model=config.embeddings_model,
            openai_api_key=config.openai_api_key
        )

    def create_vectorstore(self, documents: list[Document]) -> Chroma:
        logger.info("Creating vector store")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.config.vector_store_path
        )
        logger.info("Vector store created successfully")
        return vectorstore