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
                documents.extend(self._load_file(path))
            elif path.is_dir():
                for file_path in path.rglob('*.*'):  # Recursively get all files
                    documents.extend(self._load_file(file_path))
            else:
                logger.warning(f"Invalid path: {path}")
        
        if not documents:
            logger.warning("No documents found to process.")
            return []
        
        logger.info(f"Loaded {len(documents)} documents")
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        return split_docs

    def _load_file(self, path: Path) -> List[Document]:
        documents = []
        try:
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
        except UnicodeDecodeError:
            logger.warning(f"Skipping non-text or unsupported file: {path}")
        except Exception as e:
            logger.error(f"Error loading file {path}: {str(e)}")
        return documents
