"""Example: PostgreSQL with PGVector indexing."""

import os
from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    os.environ["POSTGRES_CONNECTION_STRING"] = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg://user:password@localhost:5432/vector_db"
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="pgvector",
        embeddings=embeddings,
        collection_name="documents",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to PGVector")


if __name__ == "__main__":
    main()
