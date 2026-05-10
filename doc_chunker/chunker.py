import os
from pathlib import Path
from typing import List, Literal, Optional

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
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)


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

    _CHUNKING_STRATEGIES = {
        "recursive": RecursiveCharacterTextSplitter,
        "character": CharacterTextSplitter,
        "markdown": MarkdownTextSplitter,
        "markdown_headers": MarkdownHeaderTextSplitter,
        "html_headers": HTMLHeaderTextSplitter,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: Literal[
            "recursive", "character", "markdown", "markdown_headers", "html_headers"
        ] = "recursive",
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
        keep_separator: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.text_splitter = self._create_splitter(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            is_separator_regex=is_separator_regex,
            keep_separator=keep_separator,
        )

    def _create_splitter(
        self,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]],
        is_separator_regex: bool,
        keep_separator: bool,
    ):
        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators or ["\n\n", "\n", " ", ""],
                is_separator_regex=is_separator_regex,
                keep_separator=keep_separator,
            )
        elif strategy == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separators[0] if separators else "\n",
                is_separator_regex=is_separator_regex,
            )
        elif strategy == "markdown":
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                is_separator_regex=is_separator_regex,
                keep_separator=keep_separator,
            )
        elif strategy == "markdown_headers":
            return MarkdownHeaderTextSplitter(separators=separators)
        elif strategy == "html_headers":
            return HTMLHeaderTextSplitter(separators=separators)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

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
            chunk.metadata["chunking_strategy"] = self.strategy

        return chunks

    @classmethod
    def supported_formats(cls) -> List[str]:
        return list(cls._EXTENSION_LOADER_MAP.keys())

    @classmethod
    def supported_strategies(cls) -> List[str]:
        return list(cls._CHUNKING_STRATEGIES.keys())
