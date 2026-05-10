import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreType(Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    MILVUS = "milvus"
    LANCEDB = "lancedb"
    ANNOY = "annoy"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    MONGODB = "mongodb"
    PGVECTOR = "pgvector"
    ASTRA_DB = "astra_db"
    ELASTICSEARCH = "elasticsearch"
    OPENSEARCH = "opensearch"
    AZURE_AI_SEARCH = "azure_ai_search"
    AZURE_COSMOS = "azure_cosmos"
    ORACLE = "oracle"
    TURBOPUFFER = "turbopuffer"
    VALKEY = "valkey"
    COCKROACHDB = "cockroachdb"
    CLICKHOUSE = "clickhouse"
    COUCHBASE = "couchbase"
    NEO4J = "neo4j"
    SINGLESTORE = "singlestore"
    SUPABASE = "supabase"
    MYSCALE = "myscale"
    ZILLIZ = "zilliz"
    MEILISEARCH = "meilisearch"
    TYPESENSE = "typesense"
    DATABRICKS = "databricks"
    LAMBDA_DB = "lambdadb"
    IN_MEMORY = "in_memory"


class DocIndexer:
    _VECTOR_STORE_TYPES = {vs.value: vs for vs in VectorStoreType}

    def __init__(
        self,
        vector_store_type: Literal[
            "faiss", "chroma", "milvus", "lancedb", "annoy", "qdrant", "weaviate",
            "pinecone", "mongodb", "pgvector", "astra_db", "elasticsearch",
            "opensearch", "azure_ai_search", "azure_cosmos", "oracle",
            "turbopuffer", "valkey", "cockroachdb", "clickhouse", "couchbase",
            "neo4j", "singlestore", "supabase", "myscale", "zilliz",
            "meilisearch", "typesense", "databricks", "lambdadb", "in_memory"
        ] = "faiss",
        embeddings: Optional[Embeddings] = None,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        connection_string: Optional[str] = None,
        index_name: Optional[str] = None,
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

        if embeddings is None:
            raise ValueError("embeddings parameter is required")

        self._init_vector_store()

    def _init_vector_store(self):
        if self.vector_store_type == "faiss":
            self._init_faiss()
        elif self.vector_store_type == "chroma":
            self._init_chroma()
        elif self.vector_store_type == "milvus":
            self._init_milvus()
        elif self.vector_store_type == "lancedb":
            self._init_lancedb()
        elif self.vector_store_type == "annoy":
            self._init_annoy()
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
        elif self.vector_store_type == "opensearch":
            self._init_opensearch()
        elif self.vector_store_type == "azure_ai_search":
            self._init_azure_ai_search()
        elif self.vector_store_type == "azure_cosmos":
            self._init_azure_cosmos()
        elif self.vector_store_type == "oracle":
            self._init_oracle()
        elif self.vector_store_type == "turbopuffer":
            self._init_turbopuffer()
        elif self.vector_store_type == "valkey":
            self._init_valkey()
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
        elif self.vector_store_type == "meilisearch":
            self._init_meilisearch()
        elif self.vector_store_type == "typesense":
            self._init_typesense()
        elif self.vector_store_type == "databricks":
            self._init_databricks()
        elif self.vector_store_type == "lambdadb":
            self._init_lambdadb()
        elif self.vector_store_type == "in_memory":
            self._init_in_memory()
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_store_type}")

    def _init_faiss(self):
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        embedding_dim = len(self.embeddings.embed_query("hello world"))
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
        connection = oracledb.connect(user=username, password=password, dsn=dsn)

        self._vector_store = OracleVS(
            client=connection,
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
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        return self._vector_store.similarity_search(
            query,
            k=k,
            filter=filter,
            **kwargs,
        )

    def search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[tuple[Document, float]]:
        return self._vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        if ids:
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
        return ""

    def load(self, directory: str) -> None:
        if self.vector_store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            self._vector_store = FAISS.load_local(
                directory,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            raise NotImplementedError(f"Load not supported for {self.vector_store_type}")

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def indexed_count(self) -> int:
        return self._indexed_count

    @classmethod
    def supported_stores(cls) -> List[str]:
        return [vs.value for vs in VectorStoreType]
