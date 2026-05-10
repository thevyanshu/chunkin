# Installation

## Requirements

Chunkin is built on top of [LangChain](https://python.langchain.com/) - a framework for developing applications powered by language models. It uses LangChain's document loaders, text splitters, and vector store integrations.

### Core Dependencies

```bash
pip install langchain-text-splitters langchain-community langchain-experimental
```

### Document Format Support

```bash
# PDF support
pip install pypdf

# Excel support
pip install openpyxl

# Word, PowerPoint, Markdown support (included in langchain-community)
```

### Optional Dependencies by Extra

```bash
# OpenAI + FAISS (recommended for quick start)
pip install "chunkin[core]"

# With semantic chunking (requires OpenAI or similar)
pip install "chunkin[semantic]"
pip install langchain-openai

# Local vector stores (Chroma, Milvus, LanceDB, etc.)
pip install "chunkin[local]"

# Cloud providers
pip install "chunkin[aws]"     # OpenSearch, Valkey, DocumentDB
pip install "chunkin[azure]"   # Azure AI Search, Cosmos DB
pip install "chunkin[gcp]"     # BigQuery, Vertex AI, Databricks

# All vector stores
pip install "chunkin[all]"
```

## Environment Variables

For cloud providers and semantic chunking:

```bash
# OpenAI (for semantic chunking)
export OPENAI_API_KEY="your-api-key"

# Azure AI Search
export AZURE_AI_SEARCH_API_KEY="your-key"
export AZURE_AI_SEARCH_ENDPOINT="your-endpoint"

# Pinecone
export PINECONE_API_KEY="your-key"

# Weaviate
export WEAVIATE_URL="your-url"
```

## LangChain Integration

Chunkin is designed to work seamlessly with LangChain:

```python
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Use with LangChain components
embeddings = OpenAIEmbeddings()
chunker = DocumentChunker()
indexer = DocIndexer(vector_store_type="faiss", embeddings=embeddings)
```

For more information about LangChain:
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)