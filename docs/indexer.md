# Doc Indexer

A Python module for indexing chunked documents into various vector stores.

## Features

- **35+ vector store support** including local and cloud options
- **Local-first options**: FAISS, Chroma, Milvus, LanceDB, LambdaDB (no external service required)
- **Unified interface**: Same API for all vector stores
- **Search**: Similarity search with optional metadata filtering

## Quick Start

```python
from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings

# Chunk documents
chunker = DocumentChunker()
chunks = chunker.create_chunks("document.pdf")

# Index to vector store
embeddings = OpenAIEmbeddings()
indexer = DocIndexer(vector_store_type="faiss", embeddings=embeddings)
indexer.index_documents(chunks)

# Search
results = indexer.search("your query", k=3)
```

## Supported Vector Stores

### Local (No External Service)

| Store | Type | Persistence |
|-------|------|-------------|
| FAISS | In-memory + file | ✅ Save/load |
| Chroma | Local DB | ✅ Auto-persist |
| Milvus | SQLite | ✅ File-based |
| LanceDB | Local DB | ✅ Auto-persist |
| LambdaDB | Local DB | ✅ Auto-persist |
| Annoy | File-based | ✅ Save |

### Cloud Services

| Store | Provider | Credentials |
|-------|----------|-------------|
| Azure AI Search | Microsoft | `AZURE_AI_SEARCH_API_KEY`, `AZURE_AI_SEARCH_ENDPOINT` |
| Azure Cosmos DB | Microsoft | `AZURE_COSMOS_CONNECTION_STRING` |
| MongoDB Atlas | MongoDB | `MONGODB_ATLAS_CONNECTION_STRING` |
| PGVector | PostgreSQL | `POSTGRES_CONNECTION_STRING` |
| Pinecone | Pinecone | `PINECONE_API_KEY` |
| Weaviate | Weaviate | `WEAVIATE_URL`, `WEAVIATE_API_KEY` |
| Qdrant | Qdrant | `QDRANT_URL` |
| Astra DB | DataStax | `ASTRA_DB_API_ENDPOINT`, `ASTRA_DB_APPLICATION_TOKEN` |
| Elasticsearch | Elastic | `ELASTICSEARCH_URL` |
| OpenSearch | AWS | `OPENSEARCH_URL` |
| Oracle | Oracle | `ORACLE_DSN`, `ORACLE_USERNAME`, `ORACLE_PASSWORD` |
| Turbopuffer | Turbopuffer | `TURBOPUFFER_API_KEY` |
| Valkey | Valkey Labs | `VALKEY_URL` |
| CockroachDB | CockroachDB | `COCKROACHDB_CONNECTION_STRING` |
| Clickhouse | ClickHouse | `CLICKHOUSE_HOST`, etc. |
| Couchbase | Couchbase | `COUCHBASE_CONNECTION_STRING` |
| Neo4j | Neo4j | `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` |
| SingleStore | SingleStore | `SINGLESTORE_CONNECTION_STRING` |
| Supabase | Supabase | `SUPABASE_CONNECTION_STRING` |
| MyScale | MyScale | `MYSCALE_HOST`, etc. |
| Zilliz | Zilliz | `ZILLIZ_URI`, `ZILLIZ_TOKEN` |
| Meilisearch | Meilisearch | `MEILISEARCH_URL` |
| Typesense | Typesense | `TYPESENSE_HOST` |
| Databricks | Databricks | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` |
