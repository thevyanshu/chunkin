# Chunkin v0.1.0

Initial release of **chunkin** - A Python library for document chunking and indexing into vector stores, built on LangChain.

## Features

- **8 document formats**: PDF, DOCX, TXT, MD, CSV, XLSX, PPT
- **6 chunking strategies**: recursive, character, markdown, markdown_headers, html_headers, semantic
- **50+ vector stores**: FAISS, Chroma, Pinecone, Weaviate, Qdrant, Azure AI Search, and more
- **Modular extras**: Install only the vector stores you need

## Installation

```bash
pip install chunkin
pip install chunkin[core]  # OpenAI + FAISS
pip install chunkin[all]   # All vector stores
```

## Quick Start

```python
from chunkin_processor import DocProcessor
from langchain_openai import OpenAIEmbeddings

processor = DocProcessor(
    embeddings=OpenAIEmbeddings(),
    vector_store_type="faiss",
    chunk_size=500,
)

processor.process_file("document.pdf")
results = processor.search("your query", k=3)
```

## Documentation

- [Overview](https://thevyanshu.github.io/chunkin/)
- [Installation](https://thevyanshu.github.io/chunkin/installation/)
- [Usage Guide](https://thevyanshu.github.io/chunkin/usage/)
- [API Reference](https://thevyanshu.github.io/chunkin/api/)
- [Chunking Strategies](https://thevyanshu.github.io/chunkin/strategies/)

## Built on LangChain

Chunkin leverages LangChain for document loading, text splitting, and vector store integrations. Learn more at [python.langchain.com](https://python.langchain.com/).