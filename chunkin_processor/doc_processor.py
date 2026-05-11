import os
from typing import Any, Dict, List, Literal, Optional, Union, Iterator

from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer, VectorStoreType
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class DocProcessorError(Exception):
    """Custom exception for DocProcessor errors."""
    pass


class DocProcessor:
    def __init__(
        self,
        embeddings: Embeddings,
        vector_store_type: str = "faiss",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: Literal[
            "recursive", "character", "markdown", "markdown_headers", "html_headers", "semantic"
        ] = "recursive",
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
        keep_separator: bool = True,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: int = 95,
        min_chunk_size: int = 0,
        buffer_size: int = 1,
        add_start_index: bool = False,
        nb_suffix: int = 1,
        output_dir: Optional[str] = None,
        save_chunks: bool = False,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        connection_string: Optional[str] = None,
        index_name: Optional[str] = None,
        continue_on_error: bool = False,
        **kwargs,
    ):
        self.embeddings = embeddings
        self.vector_store_type = vector_store_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.separators = separators
        self.is_separator_regex = is_separator_regex
        self.keep_separator = keep_separator
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
        self.buffer_size = buffer_size
        self.add_start_index = add_start_index
        self.nb_suffix = nb_suffix
        self.output_dir = output_dir
        self.save_chunks = save_chunks
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.connection_string = connection_string
        self.index_name = index_name
        self.continue_on_error = continue_on_error
        self.kwargs = kwargs
        self._errors: List[tuple[str, Exception]] = []

        self._chunker = self._create_chunker()
        self._indexer = self._create_indexer()

    def _create_chunker(self) -> DocumentChunker:
        return DocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            strategy=self.chunk_strategy,
            separators=self.separators,
            is_separator_regex=self.is_separator_regex,
            keep_separator=self.keep_separator,
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            min_chunk_size=self.min_chunk_size,
            buffer_size=self.buffer_size,
            add_start_index=self.add_start_index,
            nb_suffix=self.nb_suffix,
            output_dir=self.output_dir,
            save_chunks=self.save_chunks,
        )

    def _create_indexer(self) -> DocIndexer:
        return DocIndexer(
            vector_store_type=self.vector_store_type,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            connection_string=self.connection_string,
            index_name=self.index_name,
            **self.kwargs,
        )

    def process_file(self, file_path: str) -> List[Document]:
        chunks = self._chunker.create_chunks(file_path)
        self._indexer.index_documents(chunks)
        return chunks

    def process_files(self, file_paths: List[str]) -> Dict[str, List[Document]]:
        results = {}
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                results[file_path] = chunks
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return results

    def process_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> Dict[str, List[Document]]:
        all_chunks = self._chunker.batch_chunks(
            directory=directory,
            extensions=extensions,
            recursive=recursive,
        )

        for file_path, chunks in all_chunks.items():
            try:
                self._indexer.index_documents(chunks)
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")

        return all_chunks

    def process_directory_stream(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> Iterator[tuple[str, List[Document]]]:
        self._errors = []
        for file_path, chunks in self._chunker.batch_chunks_stream(
            directory=directory,
            extensions=extensions,
            recursive=recursive,
        ):
            try:
                self._indexer.index_documents(chunks)
                yield file_path, chunks
            except Exception as e:
                if self.continue_on_error:
                    self._errors.append((file_path, e))
                else:
                    raise DocProcessorError(
                        f"Failed to index document '{file_path}': {e}. "
                        "Set continue_on_error=True to skip failed files."
                    ) from e

    def get_errors(self) -> List[tuple[str, Exception]]:
        """Return list of errors that occurred during processing."""
        return self._errors.copy()

    def clear_errors(self) -> None:
        """Clear the error history."""
        self._errors = []

    def search(self, query: str, k: int = 4, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return self._indexer.search(query, k=k, filters=filters)

    def search_with_score(
        self, query: str, k: int = 4, filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        return self._indexer.search_with_score(query, k=k, filters=filters)

    def delete(self, ids: Optional[List[str]] = None) -> None:
        self._indexer.delete(ids=ids)

    def save(self, directory: Optional[str] = None) -> str:
        return self._indexer.save(directory=directory)

    def load(self, directory: str) -> None:
        self._indexer.load(directory=directory)

    @property
    def chunker(self) -> DocumentChunker:
        return self._chunker

    @property
    def indexer(self) -> DocIndexer:
        return self._indexer

    @property
    def indexed_count(self) -> int:
        return self._indexer.indexed_count

    @property
    def chunks(self) -> Dict[str, List[Document]]:
        return self._chunker.get_chunks()
