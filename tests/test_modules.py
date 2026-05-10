import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch


class TestDocumentChunker:
    def test_init_default_values(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.strategy == "recursive"

    def test_init_custom_values(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(
            chunk_size=500,
            chunk_overlap=50,
            strategy="character"
        )
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert chunker.strategy == "character"

    def test_supported_formats(self):
        from chunkin import DocumentChunker
        formats = DocumentChunker.supported_formats()
        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".txt" in formats
        assert ".csv" in formats

    def test_supported_strategies(self):
        from chunkin import DocumentChunker
        strategies = DocumentChunker.supported_strategies()
        assert "recursive" in strategies
        assert "character" in strategies
        assert "semantic" in strategies
        assert "markdown" in strategies

    def test_get_chunks_empty(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        chunks = chunker.get_chunks()
        assert chunks == {}

    def test_list_chunks_empty(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        summary = chunker.list_chunks()
        assert summary == {}

    def test_invalid_file_path(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with pytest.raises(FileNotFoundError):
            chunker.create_chunks("nonexistent_file.pdf")

    def test_invalid_strategy(self):
        from chunkin import DocumentChunker
        with pytest.raises(ValueError):
            chunker = DocumentChunker(strategy="invalid_strategy")


class TestDocIndexer:
    def test_init_requires_embeddings(self):
        from chunkin_indexer import DocIndexer
        with pytest.raises(ValueError, match="embeddings parameter is required"):
            DocIndexer()

    def test_supported_stores(self):
        from chunkin_indexer import DocIndexer
        stores = DocIndexer.supported_stores()
        assert "faiss" in stores
        assert "chroma" in stores
        assert "in_memory" in stores

    def test_init_faiss_with_mock_embeddings(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(vector_store_type="faiss", embeddings=mock_embeddings)
        assert indexer.vector_store_type == "faiss"
        assert indexer.indexed_count == 0

    def test_init_in_memory_with_mock_embeddings(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(vector_store_type="in_memory", embeddings=mock_embeddings)
        assert indexer.vector_store_type == "in_memory"


class TestDocProcessor:
    def test_init_requires_embeddings(self):
        from chunkin_processor import DocProcessor
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        with pytest.raises(Exception):
            DocProcessor(vector_store_type="in_memory")

    def test_init_with_mock_embeddings(self):
        from chunkin_processor import DocProcessor
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory"
        )
        assert processor.chunk_size == 1000
        assert processor.vector_store_type == "in_memory"

    def test_init_custom_chunk_config(self):
        from chunkin_processor import DocProcessor
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory",
            chunk_size=500,
            chunk_overlap=50,
            chunk_strategy="character"
        )
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
        assert processor.chunk_strategy == "character"

    def test_search_returns_list(self):
        from chunkin_processor import DocProcessor
        from langchain_core.documents import Document
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory"
        )
        results = processor.search("test query")
        assert isinstance(results, list)

    def test_indexed_count_initial_zero(self):
        from chunkin_processor import DocProcessor
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory"
        )
        assert processor.indexed_count == 0


class TestVectorStoreType:
    def test_vector_store_type_enum_values(self):
        from chunkin_indexer import VectorStoreType
        assert VectorStoreType.FAISS.value == "faiss"
        assert VectorStoreType.CHROMA.value == "chroma"
        assert VectorStoreType.IN_MEMORY.value == "in_memory"

    def test_vector_store_type_count(self):
        from chunkin_indexer import VectorStoreType
        assert len(VectorStoreType) > 30


class TestChunkingStrategies:
    def test_recursive_strategy(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(strategy="recursive")
        assert chunker.strategy == "recursive"

    def test_character_strategy(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(strategy="character")
        assert chunker.strategy == "character"

    def test_markdown_strategy(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(strategy="markdown")
        assert chunker.strategy == "markdown"

    def test_markdown_headers_strategy(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(strategy="markdown_headers")
        assert chunker.strategy == "markdown_headers"

    def test_html_headers_strategy(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(strategy="html_headers")
        assert chunker.strategy == "html_headers"


class TestOutputConfig:
    def test_output_dir_parameter(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(output_dir="chunks")
        assert chunker.output_dir == "chunks"

    def test_save_chunks_parameter(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(save_chunks=False)
        assert chunker.save_chunks is False

    def test_persist_directory_parameter(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(
            vector_store_type="chroma",
            embeddings=mock_embeddings,
            persist_directory="./db"
        )
        assert indexer.persist_directory == "./db"


class TestMetadataHandling:
    def test_chunks_have_metadata_after_processing(self):
        from chunkin import DocumentChunker
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        chunker = DocumentChunker()
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Test content", metadata={"source": "test.txt"})
        ]
        chunker._get_loader = MagicMock(return_value=mock_loader)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            chunks = chunker.create_chunks(temp_path)
            assert len(chunks) > 0
            assert "chunk_index" in chunks[0].metadata
            assert "source_file" in chunks[0].metadata
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
