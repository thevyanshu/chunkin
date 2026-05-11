import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Iterator

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
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
)

try:
    from langchain_experimental.text_splitter import SemanticChunker
    _SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SemanticChunker = None
    _SEMANTIC_CHUNKER_AVAILABLE = False


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
        "semantic": None,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: Literal[
            "recursive", "character", "markdown", "markdown_headers", "html_headers", "semantic"
        ] = "recursive",
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
        keep_separator: bool = True,
        embeddings: Optional[Embeddings] = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: int = 95,
        min_chunk_size: int = 0,
        buffer_size: int = 1,
        add_start_index: bool = False,
        nb_suffix: int = 1,
        output_dir: Optional[str] = None,
        save_chunks: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
        self.buffer_size = buffer_size
        self.add_start_index = add_start_index
        self.nb_suffix = nb_suffix
        self.output_dir = output_dir
        self.save_chunks = save_chunks
        self._chunks: Dict[str, List[Document]] = {}

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}). "
                "Otherwise chunking may produce unexpected results."
            )

        if strategy == "semantic" and embeddings is None:
            raise ValueError(
                "embeddings parameter is required for semantic chunking strategy. "
                "Example: embeddings=OpenAIEmbeddings()"
            )

        self.text_splitter = self._create_splitter(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            is_separator_regex=is_separator_regex,
            keep_separator=keep_separator,
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
            buffer_size=buffer_size,
            add_start_index=add_start_index,
            nb_suffix=nb_suffix,
        )

    def _create_splitter(
        self,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]],
        is_separator_regex: bool,
        keep_separator: bool,
        embeddings: Optional[Embeddings],
        breakpoint_threshold_type: str,
        breakpoint_threshold_amount: int,
        min_chunk_size: int,
        buffer_size: int,
        add_start_index: bool,
        nb_suffix: int,
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
            separator = separators[0] if separators and len(separators) > 0 else "\n"
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator,
                is_separator_regex=is_separator_regex,
            )
        elif strategy == "markdown":
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif strategy == "markdown_headers":
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=separators or [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )
        elif strategy == "html_headers":
            return HTMLHeaderTextSplitter(
                headers_to_split_on=separators or [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            )
        elif strategy == "semantic":
            if not _SEMANTIC_CHUNKER_AVAILABLE:
                raise ImportError(
                    "langchain_experimental is required for semantic chunking. "
                    "Install with: pip install langchain-experimental"
                )
            return SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
                min_chunk_size=min_chunk_size,
                buffer_size=buffer_size,
                add_start_index=add_start_index,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_loader(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        loader_class = self._EXTENSION_LOADER_MAP.get(ext)
        if not loader_class:
            supported = ", ".join(self._EXTENSION_LOADER_MAP.keys())
            raise ValueError(f"Unsupported file type: {ext}. Supported: {supported}")
        return loader_class(file_path)

    def _get_output_path(self, file_path: str) -> Path:
        file_stem = Path(file_path).stem
        return Path(self.output_dir) / f"{file_stem}_chunks.json"

    def _save_chunks_to_file(self, file_path: str, chunks: List[Document]) -> Optional[str]:
        if not self.save_chunks:
            return None

        if self.output_dir:
            output_path = self._get_output_path(file_path)
        else:
            output_path = Path(f"{Path(file_path).stem}_chunks.json")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

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

        self._chunks[file_path] = chunks
        self._save_chunks_to_file(file_path, chunks)

        return chunks

    def get_chunks(self, file_path: Optional[str] = None) -> Union[List[Document], Dict[str, List[Document]]]:
        if file_path:
            return self._chunks.get(file_path, [])
        return self._chunks.copy()

    def list_chunks(self) -> Dict[str, int]:
        return {file_path: len(chunks) for file_path, chunks in self._chunks.items()}

    def batch_chunks(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> Dict[str, List[Document]]:
        if not os.path.exists(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")

        if extensions is None:
            extensions = list(self._EXTENSION_LOADER_MAP.keys())

        extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]

        if not extensions:
            raise ValueError("extensions cannot be empty. Provide at least one valid extension.")

        valid_extensions = set(self._EXTENSION_LOADER_MAP.keys())
        invalid = set(extensions) - valid_extensions
        if invalid:
            raise ValueError(f"Unsupported extensions: {invalid}. Supported: {valid_extensions}")

        all_chunks: Dict[str, List[Document]] = {}
        pattern = "**/*" if recursive else "*"

        for file_path in Path(directory).glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    chunks = self.create_chunks(str(file_path))
                    all_chunks[str(file_path)] = chunks
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {e}")

        return all_chunks

    def batch_chunks_stream(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> Iterator[tuple[str, List[Document]]]:
        if not os.path.exists(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")

        if extensions is None:
            extensions = list(self._EXTENSION_LOADER_MAP.keys())

        extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]

        if not extensions:
            raise ValueError("extensions cannot be empty. Provide at least one valid extension.")

        pattern = "**/*" if recursive else "*"

        for file_path in Path(directory).glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    chunks = self.create_chunks(str(file_path))
                    yield str(file_path), chunks
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {e}")

    @classmethod
    def supported_formats(cls) -> List[str]:
        return list(cls._EXTENSION_LOADER_MAP.keys())

    @classmethod
    def supported_strategies(cls) -> List[str]:
        return list(cls._CHUNKING_STRATEGIES.keys())
