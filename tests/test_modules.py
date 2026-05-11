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

    def test_init_with_separators(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(
            separators=["\n\n", "\n", " "],
            is_separator_regex=False
        )
        assert chunker.text_splitter is not None

    def test_init_with_optional_params(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            min_chunk_size=100,
            buffer_size=2,
            nb_suffix=3
        )
        assert chunker.add_start_index is True
        assert chunker.min_chunk_size == 100
        assert chunker.buffer_size == 2
        assert chunker.nb_suffix == 3

    def test_chunk_overlap_validation(self):
        from chunkin import DocumentChunker
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentChunker(chunk_size=100, chunk_overlap=150)

    def test_semantic_strategy_requires_embeddings(self):
        from chunkin import DocumentChunker
        with pytest.raises(ValueError, match="embeddings parameter is required"):
            DocumentChunker(strategy="semantic")

    def test_supported_formats(self):
        from chunkin import DocumentChunker
        formats = DocumentChunker.supported_formats()
        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".txt" in formats
        assert ".csv" in formats
        assert ".md" in formats
        assert ".xlsx" in formats
        assert ".pptx" in formats
        assert len(formats) == 10

    def test_supported_strategies(self):
        from chunkin import DocumentChunker
        strategies = DocumentChunker.supported_strategies()
        assert "recursive" in strategies
        assert "character" in strategies
        assert "semantic" in strategies
        assert "markdown" in strategies
        assert "markdown_headers" in strategies
        assert "html_headers" in strategies
        assert len(strategies) == 6

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

    def test_get_chunks_for_specific_file(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        chunks = chunker.get_chunks("nonexistent.txt")
        assert chunks == []

    def test_invalid_file_path(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with pytest.raises(FileNotFoundError):
            chunker.create_chunks("nonexistent_file.pdf")

    def test_invalid_strategy(self):
        from chunkin import DocumentChunker
        with pytest.raises(ValueError):
            DocumentChunker(strategy="invalid_strategy")

    def test_unsupported_file_extension(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode='w') as f:
            f.write("test content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                chunker.create_chunks(temp_path)
        finally:
            os.unlink(temp_path)

    def test_invalid_batch_directory(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with pytest.raises(NotADirectoryError):
            chunker.batch_chunks("nonexistent_directory")

    def test_batch_chunks_empty_extensions(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="extensions cannot be empty"):
                chunker.batch_chunks(tmpdir, extensions=[])

    def test_batch_chunks_invalid_extensions(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported extensions"):
                chunker.batch_chunks(tmpdir, extensions=[".xyz", ".abc"])

    def test_batch_chunks_stream_empty_directory(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(chunker.batch_chunks_stream(tmpdir))
            assert len(results) == 0

    def test_save_chunks_disabled(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(save_chunks=False)
        assert chunker.save_chunks is False

    def test_output_dir_parameter(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker(output_dir="chunks")
        assert chunker.output_dir == "chunks"


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
        assert "pinecone" in stores
        assert "weaviate" in stores
        assert "qdrant" in stores

    def test_supported_stores_count(self):
        from chunkin_indexer import DocIndexer
        stores = DocIndexer.supported_stores()
        assert len(stores) >= 40

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

    def test_init_with_collection_name(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(
            vector_store_type="in_memory",
            embeddings=mock_embeddings,
            collection_name="test_collection"
        )
        assert indexer.collection_name == "test_collection"

    def test_init_with_persist_directory(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(
            vector_store_type="chroma",
            embeddings=mock_embeddings,
            persist_directory="./test_db"
        )
        assert indexer.persist_directory == "./test_db"

    def test_init_with_kwargs(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(
            vector_store_type="in_memory",
            embeddings=mock_embeddings,
            custom_param="test"
        )
        assert indexer.kwargs.get("custom_param") == "test"

    def test_invalid_vector_store_type(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        with pytest.raises(ValueError, match="Unknown vector store type"):
            DocIndexer(vector_store_type="invalid_store", embeddings=mock_embeddings)

    def test_index_documents_empty_list(self):
        from chunkin_indexer import DocIndexer
        from langchain_core.documents import Document
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(vector_store_type="in_memory", embeddings=mock_embeddings)
        result = indexer.index_documents([])
        assert result == 0
        assert indexer.indexed_count == 0

    def test_index_documents(self):
        from chunkin_indexer import DocIndexer
        from langchain_core.documents import Document
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(vector_store_type="in_memory", embeddings=mock_embeddings)

        docs = [
            Document(page_content="Content 1", metadata={"source": "test1.txt"}),
            Document(page_content="Content 2", metadata={"source": "test2.txt"}),
        ]

        count = indexer.index_documents(docs)
        assert count == 2
        assert indexer.indexed_count == 2

    def test_search(self):
        from chunkin_indexer import DocIndexer
        from langchain_core.documents import Document
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(vector_store_type="in_memory", embeddings=mock_embeddings)

        docs = [
            Document(page_content="Python is great", metadata={"source": "test.txt"}),
        ]
        indexer.index_documents(docs)

        results = indexer.search("Python", k=1)
        assert isinstance(results, list)

    def test_search_with_score(self):
        from chunkin_indexer import DocIndexer
        from langchain_core.documents import Document
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(vector_store_type="in_memory", embeddings=mock_embeddings)

        docs = [
            Document(page_content="Python is great", metadata={"source": "test.txt"}),
        ]
        indexer.index_documents(docs)

        results = indexer.search_with_score("Python", k=1)
        assert isinstance(results, list)

    def test_delete_requires_ids(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        indexer = DocIndexer(vector_store_type="in_memory", embeddings=mock_embeddings)

        with pytest.raises(ValueError, match="ids parameter is required"):
            indexer.delete()

        with pytest.raises(ValueError, match="ids parameter is required"):
            indexer.delete(ids=None)

        with pytest.raises(ValueError, match="ids parameter is required"):
            indexer.delete(ids=[])

        indexer.delete(ids=["test_id"])
        print("   [PASS] Delete requires explicit ids")


class TestDocProcessor:
    def test_init_requires_embeddings(self):
        from chunkin_processor import DocProcessor
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

    def test_init_with_all_strategies(self):
        from chunkin_processor import DocProcessor
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        for strategy in ["recursive", "character", "markdown", "markdown_headers", "html_headers"]:
            processor = DocProcessor(
                embeddings=mock_embeddings,
                vector_store_type="in_memory",
                chunk_strategy=strategy
            )
            assert processor.chunk_strategy == strategy

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

    def test_chunker_property(self):
        from chunkin_processor import DocProcessor
        from chunkin import DocumentChunker
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory"
        )
        assert isinstance(processor.chunker, DocumentChunker)

    def test_indexer_property(self):
        from chunkin_processor import DocProcessor
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory"
        )
        assert isinstance(processor.indexer, DocIndexer)

    def test_chunks_property(self):
        from chunkin_processor import DocProcessor
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory"
        )
        assert isinstance(processor.chunks, dict)

    def test_delete_method(self):
        from chunkin_processor import DocProcessor
        from chunkin_processor.doc_processor import DocProcessorError
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory"
        )

        with pytest.raises(ValueError, match="ids parameter is required"):
            processor.delete()

        processor.delete(ids=["test_id"])
        print("   [PASS] Delete method works with ids")

    def test_process_directory_stream_error_handling(self):
        from chunkin_processor import DocProcessor
        from chunkin_processor.doc_processor import DocProcessorError
        import tempfile

        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory",
            continue_on_error=False
        )

        with pytest.raises((DocProcessorError, NotADirectoryError)):
            list(processor.process_directory_stream("/nonexistent/path"))

        processor2 = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory",
            continue_on_error=True
        )

        errors = processor2.get_errors()
        assert errors == []
        print("   [PASS] Error handling works")
        processor.delete(ids=["test_id"])


class TestVectorStoreType:
    def test_vector_store_type_enum_values(self):
        from chunkin_indexer import VectorStoreType
        assert VectorStoreType.FAISS.value == "faiss"
        assert VectorStoreType.CHROMA.value == "chroma"
        assert VectorStoreType.IN_MEMORY.value == "in_memory"
        assert VectorStoreType.PINECONE.value == "pinecone"
        assert VectorStoreType.WEAVIATE.value == "weaviate"

    def test_vector_store_type_count(self):
        from chunkin_indexer import VectorStoreType
        assert len(VectorStoreType) > 30

    def test_vector_store_types_coverage(self):
        from chunkin_indexer import VectorStoreType
        expected_stores = [
            "faiss", "chroma", "milvus", "lancedb", "pinecone",
            "qdrant", "weaviate", "mongodb", "pgvector", "astra_db",
            "elasticsearch", "neo4j", "azure_ai_search", "opensearch"
        ]
        for store_name in expected_stores:
            store = VectorStoreType[store_name.upper().replace("_", "_")]
            assert store.value == store_name


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

    def test_semantic_strategy_with_embeddings(self):
        from chunkin import DocumentChunker
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        chunker = DocumentChunker(strategy="semantic", embeddings=mock_embeddings)
        assert chunker.strategy == "semantic"

    def test_all_strategy_params(self):
        from chunkin import DocumentChunker
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        chunker = DocumentChunker(
            strategy="semantic",
            embeddings=mock_embeddings,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=80,
            min_chunk_size=50,
            buffer_size=2,
            add_start_index=True
        )

        assert chunker.breakpoint_threshold_type == "standard_deviation"
        assert chunker.breakpoint_threshold_amount == 80
        assert chunker.min_chunk_size == 50
        assert chunker.buffer_size == 2
        assert chunker.add_start_index is True


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
            assert "chunking_strategy" in chunks[0].metadata
        finally:
            os.unlink(temp_path)

    def test_chunk_metadata_values(self):
        from chunkin import DocumentChunker
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        chunker = DocumentChunker(strategy="character")
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
            assert chunks[0].metadata["chunk_index"] == 0
            assert chunks[0].metadata["source_file"] == os.path.basename(temp_path)
            assert chunks[0].metadata["chunking_strategy"] == "character"
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    def test_batch_chunks_stream_with_invalid_dir(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with pytest.raises(NotADirectoryError):
            list(chunker.batch_chunks_stream("nonexistent_dir"))

    def test_batch_chunks_with_empty_directory(self):
        from chunkin import DocumentChunker
        chunker = DocumentChunker()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = chunker.batch_chunks(tmpdir)
            assert result == {}

    def test_multiple_indexer_properties(self):
        from chunkin_indexer import DocIndexer
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        indexer = DocIndexer(
            vector_store_type="in_memory",
            embeddings=mock_embeddings,
            collection_name="test",
            connection_string="test_conn",
            index_name="test_index"
        )

        assert indexer.collection_name == "test"
        assert indexer.connection_string == "test_conn"
        assert indexer.index_name == "test_index"

    def test_processor_with_output_dir(self):
        from chunkin_processor import DocProcessor
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        processor = DocProcessor(
            embeddings=mock_embeddings,
            vector_store_type="in_memory",
            output_dir="./test_output",
            save_chunks=True
        )

        assert processor.output_dir == "./test_output"
        assert processor.save_chunks is True


class TestRealDocumentProcessing:
    def test_create_chunks_from_text_file(self):
        from chunkin import DocumentChunker
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="This is a test document. " * 20, metadata={"source": "test.txt"})
        ]
        chunker._get_loader = MagicMock(return_value=mock_loader)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document. " * 20)
            temp_path = f.name

        try:
            chunks = chunker.create_chunks(temp_path)
            assert len(chunks) > 0
            assert all(hasattr(c, 'page_content') for c in chunks)
            assert all(hasattr(c, 'metadata') for c in chunks)
        finally:
            os.unlink(temp_path)

    def test_list_chunks_after_processing(self):
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
            summary = chunker.list_chunks()
            assert temp_path in summary
            assert summary[temp_path] == len(chunks)
        finally:
            os.unlink(temp_path)

    def test_get_all_chunks(self):
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
            all_chunks = chunker.get_chunks()
            assert temp_path in all_chunks
            assert len(all_chunks[temp_path]) == len(chunks)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
