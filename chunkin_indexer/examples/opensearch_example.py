"""Example: Amazon OpenSearch indexing."""

import os
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_aws import BedrockEmbeddings


def main():
    os.environ["OPENSEARCH_URL"] = os.getenv("OPENSEARCH_URL", "https://your-opensearch.us-east-1.es.amazonaws.com")

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="opensearch",
        embeddings=embeddings,
        index_name="documents",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Amazon OpenSearch")


if __name__ == "__main__":
    main()
