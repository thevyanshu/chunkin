# Document Chunker

A Python module for processing and chunking various document formats for vector store indexing.

## Features

- Support for multiple document formats: PDF, DOCX, TXT, MD, CSV, XLSX, PPT
- Multiple chunking strategies: recursive, character, markdown, headers
- Simple API: initialize class, call `create_chunks()`

## Quick Start

```python
from doc_chunker import DocumentChunker

chunker = DocumentChunker()
chunks = chunker.create_chunks("document.pdf")
```
