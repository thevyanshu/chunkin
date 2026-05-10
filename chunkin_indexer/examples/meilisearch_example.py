"""Example: Meilisearch indexing."""

import os
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    os.environ["MEILISEARCH_URL"] = os.getenv("MEILISEARCH_URL", "http://localhost:7700")
    os.environ["MEILISEARCH_API_KEY"] = os.getenv("MEILISEARCH_API_KEY", "")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="meilisearch",
        embeddings=embeddings,
        collection_name="documents",
        index_name="doc-index",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Meilisearch")


if __name__ == "__main__":
    main()
