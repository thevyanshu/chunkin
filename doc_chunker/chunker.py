import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    _EXTENSION_LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _get_loader(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        loader_class = self._EXTENSION_LOADER_MAP.get(ext)
        if not loader_class:
            supported = ", ".join(self._EXTENSION_LOADER_MAP.keys())
            raise ValueError(f"Unsupported file type: {ext}. Supported: {supported}")
        return loader_class(file_path)

    def create_chunks(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = self._get_loader(file_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source_file"] = os.path.basename(file_path)

        return chunks

    @classmethod
    def supported_formats(cls) -> List[str]:
        return list(cls._EXTENSION_LOADER_MAP.keys())
