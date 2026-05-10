# Usage

## Basic Usage

```python
from doc_chunker import DocumentChunker

chunker = DocumentChunker()
chunks = chunker.create_chunks("path/to/document.pdf")

print(f"Created {len(chunks)} chunks")
for chunk in chunks:
    print(chunk.page_content[:100])
```

## Custom Chunk Size

```python
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.create_chunks("document.pdf")
```

## Choosing a Strategy

```python
# Recursive (default)
chunker = DocumentChunker(strategy="recursive")

# Character-based
chunker = DocumentChunker(strategy="character", chunk_size=300)

# Markdown-aware
chunker = DocumentChunker(strategy="markdown", chunk_size=800)
```

## Supported Formats

| Format | Extension | Loader |
|--------|-----------|--------|
| PDF | `.pdf` | PyPDFLoader |
| Word | `.docx`, `.doc` | UnstructuredWordDocumentLoader |
| Text | `.txt` | TextLoader |
| Markdown | `.md` | UnstructuredMarkdownLoader |
| CSV | `.csv` | CSVLoader |
| Excel | `.xlsx`, `.xls` | UnstructuredExcelLoader |
| PowerPoint | `.pptx`, `.ppt` | UnstructuredPowerPointLoader |
