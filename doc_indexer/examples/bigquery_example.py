"""Example: BigQuery Vector Search indexing."""

import os
from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_google_community import GoogleGenerativeAIEmbeddings


def main():
    os.environ["GCP_PROJECT"] = os.getenv("GCP_PROJECT", "your-project-id")
    os.environ["BIGQUERY_DATASET"] = os.getenv("BIGQUERY_DATASET", "your_dataset")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="bigquery",
        embeddings=embeddings,
        collection_name="documents",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to BigQuery Vector Search")


if __name__ == "__main__":
    main()
