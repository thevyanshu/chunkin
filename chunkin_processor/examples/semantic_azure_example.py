"""End-to-end example: semantic chunking with Azure AI Search."""

import os
from chunkin_processor import DocProcessor
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

    processor = DocProcessor(
        embeddings=embeddings,
        vector_store_type="azure_ai_search",
        chunk_size=1000,
        chunk_overlap=200,
        chunk_strategy="semantic",
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        collection_name="documents",
        index_name="doc-index",
    )

    chunks = processor.process_file("sample.pdf")
    print(f"Processed {len(chunks)} chunks with semantic chunking")

    results = processor.search("query here", k=5)
    print(f"\nFound {len(results)} results from Azure AI Search")


if __name__ == "__main__":
    main()
