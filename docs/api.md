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

### Methods

#### create_chunks(file_path: str) -> List[Document]

Loads a document and returns chunked documents.

```python
chunks = chunker.create_chunks("document.pdf")
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
# ['recursive', 'character', 'markdown', 'markdown_headers', 'html_headers']
```
