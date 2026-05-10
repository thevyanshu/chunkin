"""Example: Pinecone vector store indexing."""

import os
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
    os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="pinecone",
        embeddings=embeddings,
        collection_name="my_documents",
        index_name="doc-index",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Pinecone")


if __name__ == "__main__":
    main()
