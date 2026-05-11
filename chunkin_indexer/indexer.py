import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreType(Enum):
    # Local stores
    FAISS = "faiss"
    CHROMA = "chroma"
    MILVUS = "milvus"
    LANCEDB = "lancedb"
    LAMBDA_DB = "lambdadb"
    ANNOY = "annoy"
    DEEP_LAKE = "deep_lake"
    IN_MEMORY = "in_memory"

    # Amazon AWS
    OPENSEARCH = "opensearch"
    VALKEY = "valkey"
    DOCUMENT_DB = "document_db"

    # Microsoft Azure
    AZURE_AI_SEARCH = "azure_ai_search"
    AZURE_COSMOS = "azure_cosmos"
    AZURE_COSMOS_NOSQL = "azure_cosmos_nosql"

    # Google Cloud
    DATABRICKS = "databricks"
    VERTEX_AI = "vertex_ai"
    BIGQUERY = "bigquery"
    ALLOYDB = "alloydb"

    # Other cloud/database
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    MONGODB = "mongodb"
    PGVECTOR = "pgvector"
    ASTRA_DB = "astra_db"
    ELASTICSEARCH = "elasticsearch"
    ORACLE = "oracle"
    TURBOPUFFER = "turbopuffer"
    COCKROACHDB = "cockroachdb"
    CLICKHOUSE = "clickhouse"
    COUCHBASE = "couchbase"
    NEO4J = "neo4j"
    SINGLESTORE = "singlestore"
    SUPABASE = "supabase"
    MYSCALE = "myscale"
    ZILLIZ = "zilliz"
    MARQO = "marqo"
    VECTARA = "vectara"
    EPSILLA = "epsilla"
    MEILISEARCH = "meilisearch"
    TYPESENSE = "typesense"
    TIMESCALE = "timescale"
    TILEDB = "tiledb"
    STARROCKS = "starrocks"
    DINGO_DB = "dingo_db"


class DocIndexer:
    _VECTOR_STORE_TYPES = {vs.value: vs for vs in VectorStoreType}

    def __init__(
        self,
        vector_store_type: Literal[
            # Local
            "faiss", "chroma", "milvus", "lancedb", "lambdadb", "annoy", "deep_lake", "in_memory",
            # Amazon AWS
            "opensearch", "valkey", "document_db",
            # Microsoft Azure
            "azure_ai_search", "azure_cosmos", "azure_cosmos_nosql",
            # Google Cloud
            "databricks", "vertex_ai", "bigquery", "alloydb",
            # Other
            "qdrant", "weaviate", "pinecone", "mongodb", "pgvector", "astra_db",
            "elasticsearch", "oracle", "turbopuffer", "cockroachdb", "clickhouse",
            "couchbase", "neo4j", "singlestore", "supabase", "myscale", "zilliz",
            "marqo", "vectara", "epsilla", "meilisearch", "typesense", "timescale",
            "tiledb", "starrocks", "dingo_db"
        ] = "faiss",
        embeddings: Optional[Embeddings] = None,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        connection_string: Optional[str] = None,
        index_name: Optional[str] = None,
        validate_connection: bool = False,
        **kwargs,
    ):
        self.vector_store_type = vector_store_type
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.connection_string = connection_string
        self.index_name = index_name
        self.kwargs = kwargs
        self._vector_store = None
        self._indexed_count = 0
        self._connection = None
        self._connection_validated = False

        if embeddings is None:
            raise ValueError("embeddings parameter is required")

        self._init_vector_store()

        if validate_connection:
            self._validate_connection()

    def _validate_connection(self) -> bool:
        """Validate the vector store connection after initialization."""
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized")

        try:
            if hasattr(self._vector_store, 'index') and self._vector_store.index is not None:
                pass
            elif hasattr(self._vector_store, ' similarity_search'):
                pass
            return True
        except Exception as e:
            raise ConnectionError(
                f"Failed to validate connection for {self.vector_store_type}: {e}. "
                "Check your credentials and network connectivity."
            ) from e

    def _init_vector_store(self):
        # Local stores
        if self.vector_store_type == "faiss":
            self._init_faiss()
        elif self.vector_store_type == "chroma":
            self._init_chroma()
        elif self.vector_store_type == "milvus":
            self._init_milvus()
        elif self.vector_store_type == "lancedb":
            self._init_lancedb()
        elif self.vector_store_type == "lambdadb":
            self._init_lambdadb()
        elif self.vector_store_type == "annoy":
            self._init_annoy()
        elif self.vector_store_type == "deep_lake":
            self._init_deep_lake()
        elif self.vector_store_type == "in_memory":
            self._init_in_memory()

        # Amazon AWS
        elif self.vector_store_type == "opensearch":
            self._init_opensearch()
        elif self.vector_store_type == "valkey":
            self._init_valkey()
        elif self.vector_store_type == "document_db":
            self._init_document_db()

        # Microsoft Azure
        elif self.vector_store_type == "azure_ai_search":
            self._init_azure_ai_search()
        elif self.vector_store_type == "azure_cosmos":
            self._init_azure_cosmos()
        elif self.vector_store_type == "azure_cosmos_nosql":
            self._init_azure_cosmos_nosql()

        # Google Cloud
        elif self.vector_store_type == "databricks":
            self._init_databricks()
        elif self.vector_store_type == "vertex_ai":
            self._init_vertex_ai()
        elif self.vector_store_type == "bigquery":
            self._init_bigquery()
        elif self.vector_store_type == "alloydb":
            self._init_alloydb()

        # Other cloud/database
        elif self.vector_store_type == "qdrant":
            self._init_qdrant()
        elif self.vector_store_type == "weaviate":
            self._init_weaviate()
        elif self.vector_store_type == "pinecone":
            self._init_pinecone()
        elif self.vector_store_type == "mongodb":
            self._init_mongodb()
        elif self.vector_store_type == "pgvector":
            self._init_pgvector()
        elif self.vector_store_type == "astra_db":
            self._init_astra_db()
        elif self.vector_store_type == "elasticsearch":
            self._init_elasticsearch()
        elif self.vector_store_type == "oracle":
            self._init_oracle()
        elif self.vector_store_type == "turbopuffer":
            self._init_turbopuffer()
        elif self.vector_store_type == "cockroachdb":
            self._init_cockroachdb()
        elif self.vector_store_type == "clickhouse":
            self._init_clickhouse()
        elif self.vector_store_type == "couchbase":
            self._init_couchbase()
        elif self.vector_store_type == "neo4j":
            self._init_neo4j()
        elif self.vector_store_type == "singlestore":
            self._init_singlestore()
        elif self.vector_store_type == "supabase":
            self._init_supabase()
        elif self.vector_store_type == "myscale":
            self._init_myscale()
        elif self.vector_store_type == "zilliz":
            self._init_zilliz()
        elif self.vector_store_type == "marqo":
            self._init_marqo()
        elif self.vector_store_type == "vectara":
            self._init_vectara()
        elif self.vector_store_type == "epsilla":
            self._init_epsilla()
        elif self.vector_store_type == "meilisearch":
            self._init_meilisearch()
        elif self.vector_store_type == "typesense":
            self._init_typesense()
        elif self.vector_store_type == "timescale":
            self._init_timescale()
        elif self.vector_store_type == "tiledb":
            self._init_tiledb()
        elif self.vector_store_type == "starrocks":
            self._init_starrocks()
        elif self.vector_store_type == "dingo_db":
            self._init_dingo_db()
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_store_type}")

    def _init_faiss(self):
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        try:
            embedding_dim = len(self.embeddings.embed_query("hello world"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize embeddings for FAISS: {e}. "
                "Ensure your embeddings model is properly configured and can embed queries."
            ) from e

        index = faiss.IndexFlatL2(embedding_dim)

        self._vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def _init_chroma(self):
        from langchain_chroma import Chroma

        persist_dir = self.persist_directory or "./chroma_db"
        os.makedirs(persist_dir, exist_ok=True)

        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

    def _init_milvus(self):
        from langchain_milvus import Milvus

        uri = self.persist_directory or "./milvus.db"

        self._vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": uri},
            index_params={"index_type": "FLAT", "metric_type": "L2"},
        )

    def _init_lancedb(self):
        from langchain_lancedb import LanceDBVectorStore
        import lancedb

        persist_dir = self.persist_directory or "./lancedb"
        os.makedirs(persist_dir, exist_ok=True)

        db = lancedb.connect(persist_dir)
        self._vector_store = LanceDBVectorStore(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection=db,
        )

    def _init_annoy(self):
        from langchain_community.vectorstores import Annoy

        persist_dir = self.persist_directory or "./annoy_index"
        os.makedirs(persist_dir, exist_ok=True)

        self._vector_store = Annoy(
            embedding=self.embeddings,
            metric=self.kwargs.get("metric", "angular"),
        )

    def _init_qdrant(self):
        from langchain_qdrant import QdrantVectorStore

        url = self.kwargs.get("url") or os.getenv("QDRANT_URL")
        location = self.kwargs.get("location") or os.getenv("QDRANT_LOCATION")

        if not url and not location:
            raise ValueError("url or location parameter required")

        self._vector_store = QdrantVectorStore.from_documents(
            documents=[],
            embedding=self.embeddings,
            collection_name=self.collection_name,
            url=url,
            location=location,
        )

    def _init_weaviate(self):
        from langchain_weaviate import WeaviateVectorStore

        url = self.kwargs.get("url") or os.getenv("WEAVIATE_URL")
        api_key = self.kwargs.get("api_key") or os.getenv("WEAVIATE_API_KEY")

        if not url:
            raise ValueError("url parameter or WEAVIATE_URL required")

        self._vector_store = WeaviateVectorStore(
            embedding=self.embeddings,
            index_name=self.index_name or self.collection_name,
            url=url,
            by_text=False,
            accurate=True,
        )

    def _init_pinecone(self):
        from langchain_pinecone import PineconeVectorStore

        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")

        if not api_key:
            raise ValueError("PINECONE_API_KEY required")

        self._vector_store = PineconeVectorStore(
            embedding=self.embeddings,
            index_name=self.index_name or self.collection_name,
            pinecone_api_key=api_key,
            environment=environment,
        )

    def _init_mongodb(self):
        from langchain_mongodb import MongoDBAtlasVectorSearch

        connection_string = self.connection_string or os.getenv("MONGODB_ATLAS_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("connection_string or MONGODB_ATLAS_CONNECTION_STRING required")

        self._vector_store = MongoDBAtlasVectorSearch(
            embedding=self.embeddings,
            collection=None,
            index_name=self.index_name or "vector_index",
        )

    def _init_pgvector(self):
        from langchain_postgres import PGVector

        connection_string = self.connection_string or os.getenv("POSTGRES_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("connection_string or POSTGRES_CONNECTION_STRING required")

        self._vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=connection_string,
        )

    def _init_astra_db(self):
        from langchain_astradb import AstraDBVectorStore

        api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

        if not api_endpoint or not token:
            raise ValueError("ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN required")

        self._vector_store = AstraDBVectorStore(
            embedding=self.embeddings,
            api_endpoint=api_endpoint,
            collection_name=self.collection_name,
            token=token,
        )

    def _init_elasticsearch(self):
        from langchain_elasticsearch import ElasticsearchStore

        url = self.kwargs.get("url") or os.getenv("ELASTICSEARCH_URL") or "http://localhost:9200"

        self._vector_store = ElasticsearchStore(
            index_name=self.index_name or self.collection_name,
            embedding=self.embeddings,
            es_url=url,
        )

    def _init_opensearch(self):
        from langchain_community.vectorstores import OpenSearchVectorSearch

        url = self.kwargs.get("url") or os.getenv("OPENSEARCH_URL")

        if not url:
            raise ValueError("url parameter or OPENSEARCH_URL required")

        self._vector_store = OpenSearchVectorSearch(
            embedding_function=self.embeddings,
            opensearch_url=url,
            index_name=self.index_name or self.collection_name,
        )

    def _init_azure_ai_search(self):
        from langchain_azure_ai.vectorstores import AzureAISearchVectorSearch

        api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError("AZURE_AI_SEARCH_API_KEY and AZURE_AI_SEARCH_ENDPOINT required")

        self._vector_store = AzureAISearchVectorSearch(
            embedding=self.embeddings,
            index_name=self.index_name or self.collection_name,
            api_key=api_key,
            azure_search_endpoint=endpoint,
        )

    def _init_azure_cosmos(self):
        from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
            AzureCosmosDBMongoVCoreVectorSearch,
        )

        connection_string = self.connection_string or os.getenv("AZURE_COSMOS_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("connection_string or AZURE_COSMOS_CONNECTION_STRING required")

        self._vector_store = AzureCosmosDBMongoVCoreVectorSearch(
            embedding_function=self.embeddings,
            connection_string=connection_string,
            collection_name=self.collection_name,
            index_name=self.index_name,
        )

    def _init_oracle(self):
        from langchain_oracledb.vectorstores import OracleVS
        from langchain_community.vectorstores.utils import DistanceStrategy

        dsn = self.kwargs.get("dsn") or os.getenv("ORACLE_DSN")
        username = self.kwargs.get("username") or os.getenv("ORACLE_USERNAME")
        password = self.kwargs.get("password") or os.getenv("ORACLE_PASSWORD")

        if not all([dsn, username, password]):
            raise ValueError("Oracle: dsn, username, password required (or env vars)")

        import oracledb
        self._connection = oracledb.connect(user=username, password=password, dsn=dsn)

        self._vector_store = OracleVS(
            client=self._connection,
            embedding_function=self.embeddings,
            table_name=self.collection_name,
            distance_strategy=self.kwargs.get("distance_strategy", DistanceStrategy.EUCLIDEAN_DISTANCE),
        )

    def _init_turbopuffer(self):
        from langchain_turbopuffer import TurbopufferVectorStore

        api_key = os.getenv("TURBOPUFFER_API_KEY")

        if not api_key:
            raise ValueError("TURBOPUFFER_API_KEY required")

        from turbopuffer import Turbopuffer
        tpuf = Turbopuffer(api_key=api_key, region=self.kwargs.get("region", "gcp-us-central1"))
        ns = tpuf.namespace(self.collection_name)

        self._vector_store = TurbopufferVectorStore(
            embedding=self.embeddings,
            namespace=ns,
        )

    def _init_valkey(self):
        from langchain_aws.vectorstores import ValkeyVectorStore

        url = self.kwargs.get("valkey_url") or os.getenv("VALKEY_URL") or "valkey://localhost:6379"

        self._vector_store = ValkeyVectorStore(
            embedding=self.embeddings,
            valkey_url=url,
            index_name=self.index_name or self.collection_name,
        )

    def _init_cockroachdb(self):
        from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

        connection_string = self.connection_string or os.getenv("COCKROACHDB_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("connection_string or COCKROACHDB_CONNECTION_STRING required")

        engine = CockroachDBEngine.from_connection_string(connection_string)

        self._vector_store = AsyncCockroachDBVectorStore(
            engine=engine,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
        )

    def _init_clickhouse(self):
        from langchain_community.vectorstores import ClickhouseVector

        host = self.kwargs.get("host") or os.getenv("CLICKHOUSE_HOST", "localhost")
        port = self.kwargs.get("port") or int(os.getenv("CLICKHOUSE_PORT", "8123"))
        username = self.kwargs.get("username") or os.getenv("CLICKHOUSE_USERNAME", "default")
        password = self.kwargs.get("password") or os.getenv("CLICKHOUSE_PASSWORD", "")

        self._vector_store = ClickhouseVector(
            embedding=self.embeddings,
            host=host,
            port=port,
            username=username,
            password=password,
            index=self.index_name or self.collection_name,
        )

    def _init_couchbase(self):
        from langchain_couchbase.vectorstores import CouchbaseVectorSearch

        connection_string = self.connection_string or os.getenv("COUCHBASE_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("connection_string or COUCHBASE_CONNECTION_STRING required")

        bucket_name = self.kwargs.get("bucket_name") or os.getenv("COUCHBASE_BUCKET_NAME", "default")
        scope_name = self.kwargs.get("scope_name") or os.getenv("COUCHBASE_SCOPE_NAME", "_default")
        collection_name = self.kwargs.get("collection_name") or os.getenv("COUCHBASE_COLLECTION_NAME", "_default")

        self._vector_store = CouchbaseVectorSearch(
            embedding=self.embeddings,
            connection_string=connection_string,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            index_name=self.index_name or self.collection_name,
        )

    def _init_neo4j(self):
        from langchain_neo4j import Neo4jVectorStore

        url = self.kwargs.get("url") or os.getenv("NEO4J_URL")
        username = self.kwargs.get("username") or os.getenv("NEO4J_USERNAME")
        password = self.kwargs.get("password") or os.getenv("NEO4J_PASSWORD")

        if not all([url, username, password]):
            raise ValueError("Neo4j: url, username, password required (or env vars)")

        self._vector_store = Neo4jVectorStore(
            embedding=self.embeddings,
            url=url,
            username=username,
            password=password,
            index_name=self.index_name or self.collection_name,
        )

    def _init_singlestore(self):
        from langchain_singlestore import SingleStoreVectorStore

        connection_string = self.connection_string or os.getenv("SINGLESTORE_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("connection_string or SINGLESTORE_CONNECTION_STRING required")

        self._vector_store = SingleStoreVectorStore(
            embedding=self.embeddings,
            connection_string=connection_string,
            table_name=self.collection_name,
        )

    def _init_supabase(self):
        from langchain_supabase import SupabaseVectorStore

        connection_string = self.connection_string or os.getenv("SUPABASE_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("connection_string or SUPABASE_CONNECTION_STRING required")

        self._vector_store = SupabaseVectorStore(
            embedding=self.embeddings,
            connection_string=connection_string,
            table_name=self.collection_name,
        )

    def _init_myscale(self):
        from langchain_myscale import MyScaleVectorStore

        host = self.kwargs.get("host") or os.getenv("MYSCALE_HOST")
        port = self.kwargs.get("port") or int(os.getenv("MYSCALE_PORT", "8443"))
        username = self.kwargs.get("username") or os.getenv("MYSCALE_USERNAME")
        password = self.kwargs.get("password") or os.getenv("MYSCALE_PASSWORD")

        if not all([host, username, password]):
            raise ValueError("MyScale: host, username, password required (or env vars)")

        self._vector_store = MyScaleVectorStore(
            embedding=self.embeddings,
            host=host,
            port=port,
            username=username,
            password=password,
            index_name=self.index_name or self.collection_name,
        )

    def _init_zilliz(self):
        from langchain_zilliz import ZillizVectorStore

        uri = self.kwargs.get("uri") or os.getenv("ZILLIZ_URI")
        token = self.kwargs.get("token") or os.getenv("ZILLIZ_TOKEN")

        if not all([uri, token]):
            raise ValueError("Zilliz: uri, token required (or env vars)")

        self._vector_store = ZillizVectorStore(
            embedding=self.embeddings,
            connection_args={"uri": uri, "token": token},
            collection_name=self.collection_name,
        )

    def _init_meilisearch(self):
        from langchain_meilisearch import MeilisearchVectorStore

        url = self.kwargs.get("url") or os.getenv("MEILISEARCH_URL") or "http://localhost:7700"
        api_key = self.kwargs.get("api_key") or os.getenv("MEILISEARCH_API_KEY")

        self._vector_store = MeilisearchVectorStore(
            embedding=self.embeddings,
            url=url,
            api_key=api_key,
            index_name=self.index_name or self.collection_name,
        )

    def _init_typesense(self):
        from langchain_typesense import TypesenseVectorStore

        host = self.kwargs.get("host") or os.getenv("TYPESENSE_HOST")
        port = self.kwargs.get("port") or os.getenv("TYPESENSE_PORT", "8108")
        protocol = self.kwargs.get("protocol") or os.getenv("TYPESENSE_PROTOCOL", "http")
        api_key = self.kwargs.get("api_key") or os.getenv("TYPESENSE_API_KEY")

        if not host:
            raise ValueError("Typesense: host required (or TYPESENSE_HOST env var)")

        self._vector_store = TypesenseVectorStore(
            embedding=self.embeddings,
            host=host,
            port=int(port),
            protocol=protocol,
            api_key=api_key,
            collection_name=self.collection_name,
        )

    def _init_databricks(self):
        from langchain_databricks import DatabricksVectorSearch

        host = self.kwargs.get("host") or os.getenv("DATABRICKS_HOST")
        token = self.kwargs.get("token") or os.getenv("DATABRICKS_TOKEN")

        if not all([host, token]):
            raise ValueError("Databricks: host, token required (or env vars)")

        endpoint = self.kwargs.get("endpoint") or os.getenv("DATABRICKS_ENDPOINT", f"https://{host}/vector_search")

        self._vector_store = DatabricksVectorSearch(
            embedding=self.embeddings,
            host=host,
            token=token,
            index_name=self.index_name or self.collection_name,
            endpoint=endpoint,
        )

    def _init_lambdadb(self):
        from langchain_lambdadb import LambdaDBVectorStore

        persist_dir = self.persist_directory or "./lambdadb"
        os.makedirs(persist_dir, exist_ok=True)

        self._vector_store = LambdaDBVectorStore(
            embedding=self.embeddings,
            directory=persist_dir,
            collection_name=self.collection_name,
        )

    def _init_in_memory(self):
        from langchain_core.vectorstores import InMemoryVectorStore

        self._vector_store = InMemoryVectorStore(embedding=self.embeddings)

    def _init_deep_lake(self):
        from langchain_deeplake import DeepLake

        persist_dir = self.persist_directory or "./deeplake"
        os.makedirs(persist_dir, exist_ok=True)

        self._vector_store = DeepLake(
            embedding=self.embeddings,
            path=persist_dir,
            collection_name=self.collection_name,
        )

    def _init_document_db(self):
        from langchain_aws import AmazonDocumentdbVectorSearch

        host = self.kwargs.get("host") or os.getenv("DOCUMENT_DB_HOST")
        port = self.kwargs.get("port") or int(os.getenv("DOCUMENT_DB_PORT", "27017"))
        collection = self.kwargs.get("collection")
        index_name = self.index_name or self.collection_name

        if not host:
            raise ValueError("DOCUMENT_DB_HOST required")

        self._vector_store = AmazonDocumentdbVectorSearch(
            embedding=self.embeddings,
            collection_name=collection,
            host=host,
            port=port,
            index_name=index_name,
        )

    def _init_azure_cosmos_nosql(self):
        from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch

        api_endpoint = self.kwargs.get("api_endpoint") or os.getenv("AZURE_COSMOS_NOSQL_ENDPOINT")
        token = self.kwargs.get("token") or os.getenv("AZURE_COSMOS_NOSQL_TOKEN")
        database_name = self.kwargs.get("database_name") or "vectors"
        container_name = self.kwargs.get("container_name") or self.collection_name

        if not api_endpoint or not token:
            raise ValueError("AZURE_COSMOS_NOSQL_ENDPOINT and AZURE_COSMOS_NOSQL_TOKEN required")

        self._vector_store = AzureCosmosDBNoSqlVectorSearch(
            embedding=self.embeddings,
            api_endpoint=api_endpoint,
            token=token,
            database_name=database_name,
            container_name=container_name,
        )

    def _init_vertex_ai(self):
        from langchain_google_vertexai import VertexAIVectorSearch

        project = self.kwargs.get("project") or os.getenv("GCP_PROJECT")
        region = self.kwargs.get("region") or os.getenv("GCP_REGION", "us-central1")
        index_id = self.kwargs.get("index_id") or os.getenv("VERTEX_AI_INDEX_ID")

        if not project or not index_id:
            raise ValueError("GCP_PROJECT and VERTEX_AI_INDEX_ID required")

        self._vector_store = VertexAIVectorSearch(
            project=project,
            region=region,
            index_id=index_id,
            embedding=self.embeddings,
        )

    def _init_bigquery(self):
        from langchain_google_community import BigQueryVectorSearch

        project = self.kwargs.get("project") or os.getenv("GCP_PROJECT")
        dataset = self.kwargs.get("dataset") or os.getenv("BIGQUERY_DATASET")
        table_name = self.collection_name

        if not project or not dataset:
            raise ValueError("GCP_PROJECT and BIGQUERY_DATASET required")

        self._vector_store = BigQueryVectorSearch(
            project=project,
            dataset=dataset,
            table_name=table_name,
            embedding=self.embeddings,
        )

    def _init_alloydb(self):
        from langchain_google_community import AlloyDBVectorSearch

        cluster_id = self.kwargs.get("cluster_id") or os.getenv("ALLOYDB_CLUSTER_ID")
        project_id = self.kwargs.get("project_id") or os.getenv("GCP_PROJECT")
        region = self.kwargs.get("region") or os.getenv("GCP_REGION", "us-central1")
        database = self.kwargs.get("database") or "default"
        table_name = self.collection_name

        if not cluster_id or not project_id:
            raise ValueError("ALLOYDB_CLUSTER_ID and GCP_PROJECT required")

        self._vector_store = AlloyDBVectorSearch(
            cluster_id=cluster_id,
            project_id=project_id,
            region=region,
            database=database,
            table_name=table_name,
            embedding=self.embeddings,
        )

    def _init_marqo(self):
        from langchain_marqo import Marqo

        url = self.kwargs.get("url") or os.getenv("MARQO_URL")
        api_key = self.kwargs.get("api_key") or os.getenv("MARQO_API_KEY")

        if not url:
            raise ValueError("MARQO_URL required")

        mq = Marqo(url=url, api_key=api_key)

        self._vector_store = self._create_marqo_store(mq)

    def _create_marqo_store(self, marqo_client):
        from langchain_marqo import MarqoVectorStore

        return MarqoVectorStore(
            client=marqo_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _init_vectara(self):
        from langchain_vectara import VectaraVectorStore

        api_key = self.kwargs.get("api_key") or os.getenv("VECTARA_API_KEY")
        customer_id = self.kwargs.get("customer_id") or os.getenv("VECTARA_CUSTOMER_ID")

        if not customer_id or not api_key:
            raise ValueError("VECTARA_CUSTOMER_ID and VECTARA_API_KEY required")

        self._vector_store = VectaraVectorStore(
            vectara_customer_id=customer_id,
            vectara_corpus_id=self.collection_name,
            vectara_api_key=api_key,
        )

    def _init_epsilla(self):
        from langchain_epsilla import EpsillaVectorStore

        host = self.kwargs.get("host") or os.getenv("EPSILLA_HOST", "localhost")
        port = self.kwargs.get("port") or int(os.getenv("EPSILLA_PORT", "37000"))

        self._vector_store = EpsillaVectorStore(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            host=host,
            port=port,
        )

    def _init_timescale(self):
        from langchain_timescalevector import TimescaleVector

        connection_string = self.connection_string or os.getenv("TIMESCALE_CONNECTION_STRING")
        table_name = self.collection_name

        if not connection_string:
            raise ValueError("TIMESCALE_CONNECTION_STRING required")

        self._vector_store = TimescaleVector(
            embedding=self.embeddings,
            connection_string=connection_string,
            table_name=table_name,
        )

    def _init_tiledb(self):
        from langchain_tiledb import TileDBVectorStore

        uri = self.kwargs.get("uri") or os.getenv("TILEDB_URI") or "./tiledb"

        self._vector_store = TileDBVectorStore(
            embedding=self.embeddings,
            uri=uri,
            index_name=self.index_name or self.collection_name,
        )

    def _init_starrocks(self):
        from langchain_starrocks import StarRocksVectorSearch

        host = self.kwargs.get("host") or os.getenv("STARROCKS_HOST")
        port = self.kwargs.get("port") or int(os.getenv("STARROCKS_PORT", "9030"))
        user = self.kwargs.get("user") or os.getenv("STARROCKS_USER", "root")
        password = self.kwargs.get("password") or os.getenv("STARROCKS_PASSWORD", "")

        if not host:
            raise ValueError("STARROCKS_HOST required")

        self._vector_store = StarRocksVectorSearch(
            embedding=self.embeddings,
            host=host,
            port=port,
            user=user,
            password=password,
            table_name=self.collection_name,
        )

    def _init_dingo_db(self):
        from langchain_dingo import DingoDB

        url = self.kwargs.get("url") or os.getenv("DINGO_URL")
        user = self.kwargs.get("user") or os.getenv("DINGO_USER", "root")
        password = self.kwargs.get("password") or os.getenv("DINGO_PASSWORD", "")

        if not url:
            raise ValueError("DINGO_URL required")

        self._vector_store = DingoDB(
            embedding=self.embeddings,
            index_name=self.index_name or self.collection_name,
            url=url,
            user=user,
            password=password,
        )

    def index_documents(self, documents: List[Document], **kwargs) -> int:
        if not documents:
            return 0

        self._vector_store.add_documents(documents=documents, **kwargs)
        self._indexed_count += len(documents)
        return len(documents)

    def search(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        return self._vector_store.similarity_search(
            query,
            k=k,
            filter=filters,
            **kwargs,
        )

    def search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[tuple[Document, float]]:
        return self._vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filters,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        if ids is None or len(ids) == 0:
            raise ValueError(
                "ids parameter is required for delete operation. "
                "Pass a list of document IDs to delete specific documents. "
                "To delete all documents, you must reinitialize the vector store."
            )
        self._vector_store.delete(ids=ids, **kwargs)

    def save(self, directory: Optional[str] = None) -> str:
        if self.vector_store_type == "faiss":
            dir_path = directory or self.persist_directory or "./faiss_index"
            os.makedirs(dir_path, exist_ok=True)
            self._vector_store.save_local(dir_path)
            return dir_path
        elif self.vector_store_type == "chroma":
            return self.persist_directory or "./chroma_db"
        elif self.vector_store_type == "lancedb":
            return self.persist_directory or "./lancedb"
        else:
            raise NotImplementedError(
                f"Save not supported for {self.vector_store_type}. "
                "Supported: faiss, chroma, lancedb"
            )

    def load(self, directory: str, allow_dangerous_deserialization: bool = False) -> None:
        if self.vector_store_type == "faiss":
            from langchain_community.vectorstores import FAISS

            if not allow_dangerous_deserialization:
                import os
                index_file = os.path.join(directory, "index.faiss")
                if os.path.exists(index_file):
                    raise ValueError(
                        "Loading FAISS index with allow_dangerous_deserialization=False (default). "
                        "This is for security - FAISS uses pickle which can execute arbitrary code. "
                        "Only set to True if you trust the source of the index files. "
                        "To load, explicitly call: indexer.load(directory, allow_dangerous_deserialization=True)"
                    )

            self._vector_store = FAISS.load_local(
                directory,
                self.embeddings,
                allow_dangerous_deserialization=allow_dangerous_deserialization,
            )
        else:
            raise NotImplementedError(f"Load not supported for {self.vector_store_type}")

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def indexed_count(self) -> int:
        return self._indexed_count

    @classmethod
    def supported_stores(cls) -> List[str]:
        return [vs.value for vs in VectorStoreType]
