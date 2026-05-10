import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreType(Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    MILVUS = "milvus"
    AZURE_AI_SEARCH = "azure_ai_search"
    AZURE_COSMOS = "azure_cosmos"
    MONGODB = "mongodb"
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    ASTRA_DB = "astra_db"
    ELASTICSEARCH = "elasticsearch"
    OPENSEARCH = "opensearch"
    IN_MEMORY = "in_memory"


class DocIndexer:
    _VECTOR_STORE_TYPES = {vs.value: vs for vs in VectorStoreType}

    def __init__(
        self,
        vector_store_type: Literal[
            "faiss", "chroma", "milvus", "azure_ai_search", "azure_cosmos",
            "mongodb", "pgvector", "pinecone", "weaviate", "qdrant",
            "astra_db", "elasticsearch", "opensearch", "in_memory"
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
        elif self.vector_store_type == "azure_ai_search":
            self._init_azure_ai_search()
        elif self.vector_store_type == "azure_cosmos":
            self._init_azure_cosmos()
        elif self.vector_store_type == "mongodb":
            self._init_mongodb()
        elif self.vector_store_type == "pgvector":
            self._init_pgvector()
        elif self.vector_store_type == "pinecone":
            self._init_pinecone()
        elif self.vector_store_type == "weaviate":
            self._init_weaviate()
        elif self.vector_store_type == "qdrant":
            self._init_qdrant()
        elif self.vector_store_type == "astra_db":
            self._init_astra_db()
        elif self.vector_store_type == "elasticsearch":
            self._init_elasticsearch()
        elif self.vector_store_type == "opensearch":
            self._init_opensearch()
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
            return self._persist_directory or "./chroma_db"
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
