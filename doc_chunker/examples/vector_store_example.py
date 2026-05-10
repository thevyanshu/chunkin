"""Example showing how to send chunks to a vector store (Azure AI Search)."""

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from doc_chunker import DocumentChunker
import os


def index_to_azure(file_path: str):
    azure_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_key = os.getenv("AZURE_SEARCH_API_KEY")

    if not azure_endpoint or not azure_key:
        print("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY environment variables")
        return

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
        api_key=azure_key,
    )

    vector_store = AzureSearch(
        azure_search_endpoint=azure_endpoint,
        azure_search_key=azure_key,
        index_name="document-chunks",
        embedding_function=embeddings.embed_query,
    )

    chunker = DocumentChunker()
    chunks = chunker.create_chunks(file_path)

    print(f"Indexing {len(chunks)} chunks from {file_path}...")
    vector_store.add_documents(documents=chunks)
    print("Done!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vector_store_example.py <file_path>")
    else:
        index_to_azure(sys.argv[1])
