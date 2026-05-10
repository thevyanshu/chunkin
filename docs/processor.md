# Doc Processor

A unified module that combines chunking and indexing into a single class. Built on top of [LangChain](https://python.langchain.com/).

## Features

- **Single initialization** for both chunking and indexing
- **All chunking strategies** from [LangChain text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- **50+ vector stores** from [LangChain vector store integrations](https://python.langchain.com/docs/integrations/vectorstores/)
- **End-to-end processing**: file/directory to indexed chunks
- **Search-ready**: immediate similarity search after processing

## Quick Start

```python
from chunkin_processor import DocProcessor
from langchain_openai import OpenAIEmbeddings

processor = DocProcessor(
    embeddings=OpenAIEmbeddings(),
    vector_store_type="faiss",
    chunk_size=500,
    chunk_strategy="recursive",
)

# Single file
chunks = processor.process_file("document.pdf")

# Search
results = processor.search("your query", k=3)
```

## LangChain Integration

DocProcessor combines LangChain components:

- **Document Loaders**: Load documents from various formats
- **Text Splitters**: Split documents into chunks using different strategies
- **Vector Stores**: Store and search chunk embeddings

```python
from chunkin_processor import DocProcessor
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# All LangChain-compatible
processor = DocProcessor(
    embeddings=OpenAIEmbeddings(),
    vector_store_type="chroma",
    persist_directory="./vector_db",
)
```

## Configuration Options

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Target chunk size |
| `chunk_overlap` | 200 | Overlap between chunks |
| `chunk_strategy` | "recursive" | Chunking strategy |
| `separators` | None | Custom separators |
| `breakpoint_threshold_type` | "percentile" | Semantic breakpoint method |
| `breakpoint_threshold_amount` | 95 | Semantic threshold value |

### Indexing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vector_store_type` | "faiss" | Vector store type |
| `collection_name` | "documents" | Collection/table name |
| `persist_directory` | None | Local persistence path |
| `connection_string` | None | Database connection string |
| `index_name` | None | Index name (varies by store) |

### Strategies

**Chunking strategies** (from [LangChain text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)):
- `recursive` - RecursiveCharacterTextSplitter (default)
- `character` - CharacterTextSplitter
- `markdown` - MarkdownTextSplitter
- `markdown_headers` - MarkdownHeaderTextSplitter
- `html_headers` - HTMLHeaderTextSplitter
- `semantic` - SemanticChunker (requires embeddings)

## Methods

### process_file(file_path: str) -> List[Document]

Process a single file, chunk it, and index to vector store.

```python
chunks = processor.process_file("document.pdf")
```

### process_files(file_paths: List[str]) -> Dict[str, List[Document]]

Process multiple files.

```python
results = processor.process_files(["doc1.pdf", "doc2.docx"])
```

### process_directory(directory: str, extensions, recursive) -> Dict

Process all documents in a directory.

```python
all_chunks = processor.process_directory(
    "path/to/docs",
    extensions=[".pdf", ".docx"],
    recursive=True,
)
```

### process_directory_stream(directory, extensions, recursive) -> Iterator

Memory-efficient directory processing.

```python
for file_path, chunks in processor.process_directory_stream("docs"):
    print(f"Processed: {file_path}")
```

### search(query, k, filter) -> List[Document]

Similarity search.

```python
results = processor.search("your query", k=5)
```

### Properties

```python
processor.indexed_count  # Total documents indexed
processor.chunker        # Access underlying DocumentChunker
processor.indexer       # Access underlying DocIndexer
processor.chunks        # All stored chunks
```

## See Also

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)