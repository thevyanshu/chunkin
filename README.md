# Chunkin

A Python library for document chunking and indexing into vector stores.

## Modules

| Module | Description |
|--------|-------------|
| `chunkin` | Document chunking (PDF, DOCX, TXT, MD, CSV, XLSX, PPT) |
| `chunkin_indexer` | Index chunks to 50+ vector stores |
| `chunkin_processor` | Unified end-to-end processing |

## Quick Start

```python
from chunkin_processor import DocProcessor
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
# Core only
pip install chunkin

# With OpenAI + FAISS (recommended)
pip install chunkin[core]

# With semantic chunking
pip install chunkin[semantic]

# Local vector stores (Chroma, Milvus, LanceDB, etc.)
pip install chunkin[local]

# Specific cloud providers
pip install chunkin[aws]     # Amazon AWS
pip install chunkin[azure]   # Microsoft Azure
pip install chunkin[gcp]     # Google Cloud

# All vector stores
pip install chunkin[all]
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
FAISS, Chroma, Milvus, LanceDB, LambdaDB, Deep Lake, Annoy

### Amazon AWS
OpenSearch, Valkey, DocumentDB

### Microsoft Azure
Azure AI Search, Azure Cosmos DB, Azure Cosmos DB NoSQL

### Google Cloud
Databricks Vector Search, Vertex AI Vector Search, BigQuery, AlloyDB

### Other
Qdrant, Weaviate, Pinecone, MongoDB Atlas, PGVector, Astra DB,
Elasticsearch, Oracle, Neo4j, SingleStore, Supabase, MyScale,
Zilliz, Marqo, Vectara, Meilisearch, Typesense, and more...

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
chunkin/
├── chunkin/                 # Document chunking module
│   ├── doc_chunker.py      # DocumentChunker class
│   └── examples/          # Usage examples
├── chunkin_indexer/         # Vector store indexing module
│   ├── doc_indexer.py     # DocIndexer class
│   └── examples/          # Usage examples
├── chunkin_processor/      # Unified module
│   ├── doc_processor.py   # DocProcessor class
│   └── examples/          # Usage examples
├── docs/                   # MkDocs documentation
├── pyproject.toml          # Package configuration
└── README.md
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Build package
python -m build

# Serve docs locally
cd docs && pip install -r requirements.txt && mkdocs serve
```

## License

MIT License