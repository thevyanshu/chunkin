"""Example: Elasticsearch indexing."""

import os
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings


def main():
    os.environ["ELASTICSEARCH_URL"] = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="elasticsearch",
        embeddings=embeddings,
        collection_name="documents",
        index_name="doc-index",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Elasticsearch")


if __name__ == "__main__":
    main()
