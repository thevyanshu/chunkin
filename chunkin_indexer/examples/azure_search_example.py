"""Example: Azure AI Search indexing."""

import os
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import AzureOpenAIEmbeddings


def main():
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    os.environ["AZURE_AI_SEARCH_API_KEY"] = os.getenv("AZURE_AI_SEARCH_API_KEY", "")
    os.environ["AZURE_AI_SEARCH_ENDPOINT"] = os.getenv("AZURE_AI_SEARCH_ENDPOINT", "")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    indexer = DocIndexer(
        vector_store_type="azure_ai_search",
        embeddings=embeddings,
        collection_name="my_documents",
        index_name="doc-index",
    )

    chunks = chunker.create_chunks("sample.pdf")
    print(f"Created {len(chunks)} chunks")

    indexed = indexer.index_documents(chunks)
    print(f"Indexed {indexed} documents to Azure AI Search")


if __name__ == "__main__":
    main()
