# Chunkin - Document Chunker & Indexer

A Python library for processing and chunking various document formats, and indexing them into vector stores. Built on [LangChain](https://python.langchain.com/).

## Built on LangChain

Chunkin leverages [LangChain](https://python.langchain.com/) for:

- **Document Loading**: [LangChain document loaders](https://python.langchain.com/docs/integrations/document_loaders/) for 8 formats
- **Text Splitting**: [LangChain text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) with 6 strategies
- **Vector Storage**: [LangChain vector store integrations](https://python.langchain.com/docs/integrations/vectorstores/) for 50+ stores

## Modules

### [Document Chunker](usage.md)
Process documents and create chunks for vector store indexing.
- **8 formats**: PDF, DOCX, TXT, MD, CSV, XLSX, PPT
- **6 strategies**: recursive, character, markdown, markdown_headers, html_headers, semantic
- **Batch processing** with directory support

### [Doc Indexer](indexer.md)
Index chunks into various vector stores and perform similarity search.
- **50+ vector stores**: Local, AWS, Azure, Google Cloud, and more
- **Unified API** for all vector stores using [LangChain's VectorStore interface](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- **Search** with metadata filtering

### [Doc Processor](processor.md)
Unified end-to-end processing combining chunking and indexing.
- **Single initialization** for both modules
- **All chunking + indexing options**
- **process_file/directory()** for easy workflows

## Quick Start

```python
from chunkin_processor import DocProcessor
from langchain_openai import OpenAIEmbeddings

processor = DocProcessor(
    embeddings=OpenAIEmbeddings(),
    vector_store_type="faiss",
    chunk_size=500,
)

# Process file
processor.process_file("document.pdf")

# Search
results = processor.search("your query", k=3)
```

## Supported Formats

Uses [LangChain document loaders](https://python.langchain.com/docs/integrations/document_loaders/):

| Format | Extensions | Default Metadata |
|--------|------------|-----------------|
| PDF | `.pdf` | `source`, `page` |
| Word | `.docx`, `.doc` | `source` |
| Text | `.txt` | `source` |
| Markdown | `.md` | `source` |
| CSV | `.csv` | `source` |
| Excel | `.xlsx`, `.xls` | `source` |
| PowerPoint | `.pptx`, `.ppt` | `source` |

## Installation

```bash
# Core only
pip install chunkin

# With OpenAI + FAISS
pip install chunkin[core]

# With semantic chunking
pip install chunkin[semantic]

# Local vector stores
pip install chunkin[local]

# All vector stores
pip install chunkin[all]
```

## LangChain Integration

Chunkin is designed to work seamlessly with other LangChain components:

```python
from chunkin import DocumentChunker
from chunkin_indexer import DocIndexer
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# All components work together
embeddings = OpenAIEmbeddings()
chunker = DocumentChunker()
indexer = DocIndexer(vector_store_type="faiss", embeddings=embeddings)
```

## Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)