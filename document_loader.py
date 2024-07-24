from typing import List
from pathlib import Path
import logging
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, paths: List[str]) -> List[Document]:
        documents = []
        for path in paths:
            path = Path(path.strip())
            if path.is_file():
                if path.suffix.lower() == '.pdf':
                    # Load PDF file
                    with fitz.open(path) as doc:
                        content = ""
                        for page in doc:
                            content += page.get_text()
                    doc = Document(page_content=content, metadata={"source": str(path)})
                else:
                    # Load text file
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        doc = Document(page_content=content, metadata={"source": str(path)})
                documents.append(doc)
            else:
                logger.warning(f"Invalid path: {path}")
        
        logger.info(f"Loaded {len(documents)} documents")
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        return split_docs
