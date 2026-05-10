"""Example: Weaviate vector store indexing."""

from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="weaviate",
        embeddings=embeddings,
        collection_name="my_documents",
        url="http://localhost:8080",  # or use WEAVIATE_URL env var
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Weaviate")


if __name__ == "__main__":
    main()
