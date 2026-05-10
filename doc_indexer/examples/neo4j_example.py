"""Example: Neo4j vector store indexing."""

import os
from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    os.environ["NEO4J_URL"] = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME", "neo4j")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "password")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="neo4j",
        embeddings=embeddings,
        collection_name="documents",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Neo4j")


if __name__ == "__main__":
    main()
