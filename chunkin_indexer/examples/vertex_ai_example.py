"""Example: Google Vertex AI Vector Search indexing."""

import os
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_google_vertexai import VertexAIEmbeddings


def main():
    os.environ["GCP_PROJECT"] = os.getenv("GCP_PROJECT", "your-project-id")
    os.environ["VERTEX_AI_INDEX_ID"] = os.getenv("VERTEX_AI_INDEX_ID", "your-index-id")

    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="vertex_ai",
        embeddings=embeddings,
        index_id="your-index-id",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Vertex AI Vector Search")


if __name__ == "__main__":
    main()
