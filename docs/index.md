# Document Chunker & Indexer

A Python module for processing and chunking various document formats, and indexing them into vector stores.

## Modules

### [Document Chunker](usage.md)
Process documents and create chunks for vector store indexing.
- **8 formats**: PDF, DOCX, TXT, MD, CSV, XLSX, PPT
- **6 strategies**: recursive, character, markdown, markdown_headers, html_headers, semantic
- **Batch processing** with directory support

### [Doc Indexer](indexer.md)
Index chunks into various vector stores and perform similarity search.
- **50+ vector stores**: Local, AWS, Azure, Google Cloud, and more
- **Unified API** for all vector stores
- **Search** with metadata filtering

## Quick Start

```python
from doc_chunker import DocumentChunker
from doc_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings

# Chunk documents
chunker = DocumentChunker()
chunks = chunker.create_chunks("document.pdf")

# Index to vector store
embeddings = OpenAIEmbeddings()
indexer = DocIndexer(vector_store_type="faiss", embeddings=embeddings)
indexer.index_documents(chunks)

# Search
results = indexer.search("your query", k=3)
```

## Supported Formats

| Format | Extensions | Default Metadata |
|--------|------------|-----------------|
| PDF | `.pdf` | `source`, `page` |
| Word | `.docx`, `.doc` | `source` |
| Text | `.txt` | `source` |
| Markdown | `.md` | `source` |
| CSV | `.csv` | `source` |
| Excel | `.xlsx`, `.xls` | `source` |
| PowerPoint | `.pptx`, `.ppt` | `source` |
