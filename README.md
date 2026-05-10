# Document Chunker & Indexer

A Python library for processing documents and indexing them into vector stores.

## Modules

| Module | Description |
|--------|-------------|
| [doc_chunker](doc_chunker/) | Chunk documents (PDF, DOCX, TXT, MD, CSV, XLSX, PPT) |
| [doc_indexer](doc_indexer/) | Index chunks to 50+ vector stores |
| [doc_processor](doc_processor/) | Unified end-to-end processing |

## Quick Start

```python
from doc_processor import DocProcessor
from langchain_openai import OpenAIEmbeddings

processor = DocProcessor(
    embeddings=OpenAIEmbeddings(),
    vector_store_type="faiss",
    chunk_size=500,
)

processor.process_file("document.pdf")
results = processor.search("your query", k=3)
```

## Installation

```bash
# Core dependencies
pip install langchain langchain-text-splitters langchain-community pypdf openpyxl

# For specific features
pip install -r doc_chunker/requirements.txt
pip install -r doc_indexer/requirements.txt
pip install -r doc_processor/requirements.txt
```

## Documentation

- [Overview](docs/index.md)
- [Installation](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api.md)
- [Chunking Strategies](docs/strategies.md)
- [Vector Stores](docs/indexer.md)
- [Doc Processor](docs/processor.md)

## Supported Formats

| Format | Extensions |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx`, `.doc` |
| Text | `.txt` |
| Markdown | `.md` |
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| PowerPoint | `.pptx`, `.ppt` |

## Supported Vector Stores

### Local (No External Service)
- FAISS, Chroma, Milvus, LanceDB, LambdaDB, Deep Lake, Annoy

### Amazon AWS
- OpenSearch, Valkey, DocumentDB

### Microsoft Azure
- Azure AI Search, Azure Cosmos DB, Azure Cosmos DB NoSQL

### Google Cloud
- Databricks Vector Search, Vertex AI Vector Search, BigQuery, AlloyDB

### Other
- Qdrant, Weaviate, Pinecone, MongoDB Atlas, PGVector, Astra DB,
- Elasticsearch, Oracle, Neo4j, SingleStore, Supabase, MyScale,
- Zilliz, Marqo, Vectara, Meilisearch, Typesense, and more...

See [docs/indexer.md](docs/indexer.md) for full list.

## Supported Chunking Strategies

| Strategy | Description |
|----------|-------------|
| `recursive` | Recursively splits by paragraphs, sentences, words |
| `character` | Simple character-based splitting |
| `markdown` | Markdown-aware splitting |
| `markdown_headers` | Split by markdown headers |
| `html_headers` | Split by HTML header tags |
| `semantic` | Embedding-based semantic splitting |

See [docs/strategies.md](docs/strategies.md) for details.

## Project Structure

```
Indexer/
├── doc_chunker/          # Document chunking module
│   ├── chunker.py        # Main DocumentChunker class
│   └── examples/         # Usage examples
├── doc_indexer/          # Vector store indexing module
│   ├── indexer.py        # Main DocIndexer class
│   └── examples/         # Usage examples
├── doc_processor/         # Unified module
│   ├── doc_processor.py  # Main DocProcessor class
│   └── examples/         # Usage examples
├── docs/                 # MkDocs documentation
│   ├── index.md         # Home
│   ├── usage.md         # Usage guide
│   ├── api.md           # API reference
│   ├── strategies.md    # Chunking strategies
│   ├── indexer.md       # Vector stores
│   └── processor.md     # Doc Processor
├── mkdocs.yml           # MkDocs config
└── README.md
```

## Development

```bash
# Install all dependencies
pip install -r doc_chunker/requirements.txt
pip install -r doc_indexer/requirements.txt
pip install -r doc_processor/requirements.txt

# Serve docs locally
cd docs && pip install -r requirements.txt && mkdocs serve
```
