# Doc Indexer

A Python module for indexing chunked documents into various vector stores. Built on [LangChain's vector store integrations](https://python.langchain.com/docs/modules/data_connection/vectorstores/).

## Features

- **50+ vector store support** including local and cloud options
- **Local-first options**: FAISS, Chroma, Milvus, LanceDB, LambdaDB, Deep Lake (no external service required)
- **Unified interface**: Same API for all vector stores using [LangChain's VectorStore interface](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- **Search**: Similarity search with optional metadata filtering

## Quick Start

```python
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
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

## LangChain Integration

DocIndexer uses [LangChain vector stores](https://python.langchain.com/docs/integrations/vectorstores/) for all vector store implementations. This provides:

- Consistent API across all vector stores
- Access to the latest vector store features from LangChain
- Easy swapping between vector stores without code changes

## Supported Vector Stores

### Local (No External Service)

| Store | Type | LangChain Package |
|-------|------|-------------------|
| FAISS | In-memory + file | langchain-community |
| Chroma | Local DB | langchain-chroma |
| Milvus | SQLite | langchain-milvus |
| LanceDB | Local DB | langchain-lancedb |
| LambdaDB | Local DB | langchain-lambdadb |
| Deep Lake | Local DB | langchain-deeplake |
| Annoy | File-based | langchain-community |
| InMemory | In-memory | langchain-core |

### Amazon Web Services (AWS)

| Store | Type | Credentials |
|-------|------|-------------|
| OpenSearch | Search | `OPENSEARCH_URL` |
| Valkey | Redis | `VALKEY_URL` |
| DocumentDB | MongoDB | `DOCUMENT_DB_HOST` |

### Microsoft Azure

| Store | Type | Credentials |
|-------|------|-------------|
| Azure AI Search | Search | `AZURE_AI_SEARCH_API_KEY`, `AZURE_AI_SEARCH_ENDPOINT` |
| Azure Cosmos DB (Mongo vCore) | NoSQL | `AZURE_COSMOS_CONNECTION_STRING` |
| Azure Cosmos DB NoSQL | NoSQL | `AZURE_COSMOS_NOSQL_ENDPOINT`, `AZURE_COSMOS_NOSQL_TOKEN` |

### Google Cloud

| Store | Type | Credentials |
|-------|------|-------------|
| Databricks Vector Search | Search | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` |
| Vertex AI Vector Search | Search | `GCP_PROJECT`, `VERTEX_AI_INDEX_ID` |
| BigQuery Vector Search | Search | `GCP_PROJECT`, `BIGQUERY_DATASET` |
| AlloyDB Vector Search | Search | `ALLOYDB_CLUSTER_ID`, `GCP_PROJECT` |

### Other Cloud/Database

| Store | Provider | Credentials |
|-------|----------|-------------|
| Qdrant | Qdrant | `QDRANT_URL` |
| Weaviate | Weaviate | `WEAVIATE_URL` |
| Pinecone | Pinecone | `PINECONE_API_KEY` |
| MongoDB Atlas | MongoDB | `MONGODB_ATLAS_CONNECTION_STRING` |
| PGVector | PostgreSQL | `POSTGRES_CONNECTION_STRING` |
| Astra DB | DataStax | `ASTRA_DB_API_ENDPOINT`, `ASTRA_DB_APPLICATION_TOKEN` |
| Elasticsearch | Elastic | `ELASTICSEARCH_URL` |
| Oracle | Oracle | `ORACLE_DSN`, `ORACLE_USERNAME`, `ORACLE_PASSWORD` |
| Turbopuffer | Turbopuffer | `TURBOPUFFER_API_KEY` |
| CockroachDB | CockroachDB | `COCKROACHDB_CONNECTION_STRING` |
| Clickhouse | ClickHouse | `CLICKHOUSE_HOST` |
| Couchbase | Couchbase | `COUCHBASE_CONNECTION_STRING` |
| Neo4j | Neo4j | `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` |
| SingleStore | SingleStore | `SINGLESTORE_CONNECTION_STRING` |
| Supabase | Supabase | `SUPABASE_CONNECTION_STRING` |
| MyScale | MyScale | `MYSCALE_HOST` |
| Zilliz | Zilliz | `ZILLIZ_URI`, `ZILLIZ_TOKEN` |
| Marqo | Marqo | `MARQO_URL` |
| Vectara | Vectara | `VECTARA_CUSTOMER_ID`, `VECTARA_API_KEY` |
| Epsilla | Epsilla | `EPSILLA_HOST` |
| Meilisearch | Meilisearch | `MEILISEARCH_URL` |
| Typesense | Typesense | `TYPESENSE_HOST` |
| Timescale | Timescale | `TIMESCALE_CONNECTION_STRING` |
| TileDB | TileDB | `TILEDB_URI` |
| StarRocks | StarRocks | `STARROCKS_HOST` |
| DingoDB | DingoDB | `DINGO_URL` |

## All Supported Stores

```python
from chunkin_indexer import DocIndexer
print(DocIndexer.supported_stores())
# Returns list of all 50+ supported vector store types
```

## Further Reading

- [LangChain Vector Store Integrations](https://python.langchain.com/docs/integrations/vectorstores/)
- [LangChain VectorStore Class](https://python.langchain.com/docs/modules/data_connection/vectorstores/)