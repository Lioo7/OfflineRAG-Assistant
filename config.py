import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Document processing
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embedding
    embeddings_model: str = "text-embedding-ada-002"

    # Question Answering
    qa_model: str = "gpt-3.5-turbo"

    # Vector Store
    vector_store_path: str = "chroma"

    # Retriever
    top_k: int = 5

    # API Key (Load from environment variable)
    openai_api_key: str = os.getenv("OPENAI_API_KEY")

    def __init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment.")