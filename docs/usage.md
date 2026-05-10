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

# Semantic chunking
from langchain_openai import OpenAIEmbeddings

chunker = DocumentChunker(
    strategy="semantic",
    embeddings=OpenAIEmbeddings(),
)
```

## Output Directory & Saving

By default, chunks are saved as JSON files. Set `output_dir` to specify a directory:

```python
chunker = DocumentChunker(output_dir="chunks")
chunks = chunker.create_chunks("document.pdf")
# Saves to: chunks/document_chunks.json
```

For single files without `output_dir`, saves to working directory:

```python
chunker = DocumentChunker(save_chunks=False)  # Don't save, just return chunks
```

## Retrieving Chunks

After processing, access chunks from the internal store:

```python
# Get all chunks
all_chunks = chunker.get_chunks()

# Get chunks for specific file
pdf_chunks = chunker.get_chunks("document.pdf")

# List all processed files and their chunk counts
chunk_summary = chunker.list_chunks()
# {'document.pdf': 15, 'report.docx': 8}
```

## Batch Processing

Process all documents in a directory:

```python
chunker = DocumentChunker(output_dir="chunks")

# Get all chunks as a dictionary
all_chunks = chunker.batch_chunks("path/to/documents")
for file_path, chunks in all_chunks.items():
    print(f"{file_path}: {len(chunks)} chunks")
# Saves to: chunks/document_chunks.json, chunks/report_chunks.json, etc.
```

Filter by specific extensions:

```python
all_chunks = chunker.batch_chunks(
    "path/to/documents",
    extensions=[".pdf", ".docx"],
)
```

Recursive processing (including subdirectories):

```python
all_chunks = chunker.batch_chunks(
    "path/to/documents",
    extensions=[".pdf", ".docx", ".txt"],
    recursive=True,
)
```

Stream processing for large directories:

```python
chunker = DocumentChunker(output_dir="chunks")

for file_path, chunks in chunker.batch_chunks_stream("path/to/documents"):
    print(f"Processing: {file_path}")
    # Process chunks here
```

## Chunk Metadata

Each chunk includes metadata:

```python
{
    "source": "path/to/document.pdf",  # from loader
    "page": 0,                          # from loader (PDF only)
    "chunk_index": 0,                    # added by DocumentChunker
    "source_file": "document.pdf",      # added by DocumentChunker
    "chunking_strategy": "recursive"    # added by DocumentChunker
}
```

Access metadata:

```python
chunks = chunker.create_chunks("document.pdf")
for chunk in chunks:
    print(f"Chunk {chunk.metadata['chunk_index']}: {len(chunk.page_content)} chars")
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
