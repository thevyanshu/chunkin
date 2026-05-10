"""Example: Batch indexing documents with FAISS."""

from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="faiss",
        embeddings=embeddings,
        persist_directory="./faiss_batch_index",
    )

    all_chunks = chunker.batch_chunks(
        "path/to/documents",
        extensions=[".pdf", ".docx", ".txt"],
        recursive=True,
    )

    print(f"Processing {len(all_chunks)} files...")

    total_indexed = 0
    for file_path, chunks in all_chunks.items():
        indexed = indexer.index_documents(chunks)
        total_indexed += indexed
        print(f"  {file_path}: {indexed} chunks indexed")

    print(f"\nTotal indexed: {total_indexed} documents")

    results = indexer.search("search query here", k=5)
    print(f"\nSearch results: {len(results)} documents found")


if __name__ == "__main__":
    main()
