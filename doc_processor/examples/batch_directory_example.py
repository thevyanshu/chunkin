"""End-to-end example: chunk and index a directory."""

from doc_processor import DocProcessor
from langchain_openai import OpenAIEmbeddings


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    processor = DocProcessor(
        embeddings=embeddings,
        vector_store_type="chroma",
        chunk_size=500,
        chunk_overlap=50,
        chunk_strategy="recursive",
        persist_directory="./chroma_db",
    )

    all_chunks = processor.process_directory(
        "path/to/documents",
        extensions=[".pdf", ".docx", ".txt"],
        recursive=True,
    )

    print(f"Processed {len(all_chunks)} files")
    print(f"Total indexed: {processor.indexed_count} chunks")

    results = processor.search("search query", k=5)
    print(f"\nSearch results: {len(results)} documents")


if __name__ == "__main__":
    main()
