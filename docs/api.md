# API Reference

## DocumentChunker

```python
class DocumentChunker:
```

### Constructor

```python
DocumentChunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    separators: Optional[List[str]] = None,
    is_separator_regex: bool = False,
    keep_separator: bool = True,
    embeddings: Optional[Embeddings] = None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 95,
    min_chunk_size: int = 0,
    buffer_size: int = 1,
    add_start_index: bool = False,
    nb_suffix: int = 1,
    output_dir: Optional[str] = None,
    save_chunks: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1000 | Target size of each chunk |
| `chunk_overlap` | int | 200 | Overlap between chunks |
| `strategy` | str | "recursive" | Chunking strategy |
| `separators` | List[str] | None | Custom separators |
| `is_separator_regex` | bool | False | Treat separators as regex |
| `keep_separator` | bool | True | Include separator in chunks |
| `embeddings` | Embeddings | None | Required for semantic strategy |
| `breakpoint_threshold_type` | str | "percentile" | Semantic breakpoint method |
| `breakpoint_threshold_amount` | int | 95 | Threshold for semantic splits |
| `buffer_size` | int | 1 | Sentence overlap for semantic |
| `min_chunk_size` | int | 0 | Minimum chunk size |
| `add_start_index` | bool | False | Add start index to metadata |
| `nb_suffix` | int | 1 | Number suffix for splits |
| `output_dir` | str | None | Directory to save chunk JSON files |
| `save_chunks` | bool | True | Whether to save chunks to JSON |

### Methods

#### create_chunks(file_path: str) -> List[Document]

Loads a document, chunks it, saves to JSON, and stores in internal collection.

```python
chunks = chunker.create_chunks("document.pdf")
```

#### get_chunks(file_path: Optional[str] = None) -> Union[List[Document], Dict[str, List[Document]]]

Retrieve chunks from the internal store.

```python
# All chunks
all_chunks = chunker.get_chunks()

# Chunks for specific file
pdf_chunks = chunker.get_chunks("document.pdf")
```

#### list_chunks() -> Dict[str, int]

Returns a summary of all processed files with their chunk counts.

```python
summary = chunker.list_chunks()
# {'document.pdf': 15, 'report.docx': 8}
```

#### batch_chunks(directory: str, extensions: Optional[List[str]] = None, recursive: bool = False) -> Dict[str, List[Document]]

Process all documents in a directory.

```python
all_chunks = chunker.batch_chunks("path/to/documents")
```

#### batch_chunks_stream(directory: str, extensions: Optional[List[str]] = None, recursive: bool = False) -> Iterator[tuple[str, List[Document]]]

Stream process documents for memory-efficient batch processing.

```python
for file_path, chunks in chunker.batch_chunks_stream("path/to/documents"):
    print(f"Processed: {file_path} ({len(chunks)} chunks)")
```

#### supported_formats() -> List[str]

Returns list of supported file extensions.

```python
formats = DocumentChunker.supported_formats()
# ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pptx', '.ppt']
```

#### supported_strategies() -> List[str]

Returns list of available chunking strategies.

```python
strategies = DocumentChunker.supported_strategies()
# ['recursive', 'character', 'markdown', 'markdown_headers', 'html_headers', 'semantic']
```

### Chunk Metadata

Each chunk contains:

**From document loaders:**
- `source` - full path to source file
- `page` - page number (PDF only)

**Added by DocumentChunker:**
- `chunk_index` - index of chunk within source file
- `source_file` - filename only
- `chunking_strategy` - strategy used

### Output Format

Chunks are saved as JSON:

```json
[
  {
    "content": "Chunk text content...",
    "metadata": {
      "source": "/path/to/document.pdf",
      "page": 0,
      "chunk_index": 0,
      "source_file": "document.pdf",
      "chunking_strategy": "recursive"
    }
  },
  {
    "content": "Next chunk...",
    "metadata": {
      "source": "/path/to/document.pdf",
      "page": 0,
      "chunk_index": 1,
      "source_file": "document.pdf",
      "chunking_strategy": "recursive"
    }
  }
]
```
