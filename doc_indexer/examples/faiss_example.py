"""Example: FAISS local vector store indexing."""

from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="faiss",
        embeddings=embeddings,
        persist_directory="./faiss_index",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents")

    results = indexer.search("What is the document about?", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
