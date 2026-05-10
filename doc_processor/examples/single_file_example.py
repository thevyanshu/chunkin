"""End-to-end example: chunk and index a single file."""

from doc_processor import DocProcessor
from langchain_openai import OpenAIEmbeddings


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    processor = DocProcessor(
        embeddings=embeddings,
        vector_store_type="faiss",
        chunk_size=500,
        chunk_overlap=50,
        chunk_strategy="recursive",
        persist_directory="./faiss_index",
    )

    chunks = processor.process_file("sample.pdf")
    print(f"Processed {len(chunks)} chunks")

    results = processor.search("What is the document about?", k=3)
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
