"""Example: Marqo vector store indexing."""

import os
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    os.environ["MARQO_URL"] = os.getenv("MARQO_URL", "http://localhost:8882")
    os.environ["MARQO_API_KEY"] = os.getenv("MARQO_API_KEY", "")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="marqo",
        embeddings=embeddings,
        collection_name="documents",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Marqo")


if __name__ == "__main__":
    main()
