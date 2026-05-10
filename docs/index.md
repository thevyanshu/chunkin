# Document Chunker

A Python module for processing and chunking various document formats for vector store indexing.

## Features

- **Multi-format support**: PDF, DOCX, TXT, MD, CSV, XLSX, PPT
- **6 Chunking strategies**: recursive, character, markdown, markdown_headers, html_headers, semantic
- **Batch processing**: Process all documents in a directory
- **Streaming**: Memory-efficient batch processing
- **Output management**: Save chunks to JSON, configurable output directory
- **Internal store**: Access chunks after processing

## Quick Start

```python
from doc_chunker import DocumentChunker

# Single file
chunker = DocumentChunker(output_dir="chunks")
chunks = chunker.create_chunks("document.pdf")

# Batch processing
all_chunks = chunker.batch_chunks("path/to/documents")
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
