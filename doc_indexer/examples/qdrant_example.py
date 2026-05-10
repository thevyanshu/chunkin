"""Example: Qdrant vector store indexing."""

from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="qdrant",
        embeddings=embeddings,
        collection_name="my_documents",
        url="http://localhost:6333",  # or use QDRANT_URL env var
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Qdrant")


if __name__ == "__main__":
    main()
