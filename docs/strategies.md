# Chunking Strategies

## recursive (Default)

`RecursiveCharacterTextSplitter` - Recursively splits text using hierarchy of separators.

Default separators: `["\n\n", "\n", " ", ""]`

Best for: General purpose text chunking with good context preservation.

```python
chunker = DocumentChunker(strategy="recursive")
```

## character

`CharacterTextSplitter` - Simple character-based splitting.

Best for: Simple use cases, consistent chunk sizes.

```python
chunker = DocumentChunker(strategy="character", chunk_size=500)
```

## markdown

`MarkdownTextSplitter` - Splits markdown preserving structure.

Best for: Markdown documents where header hierarchy should be respected.

```python
chunker = DocumentChunker(strategy="markdown", chunk_size=800)
```

## markdown_headers

`MarkdownHeaderTextSplitter` - Splits by markdown headers only.

Best for: When you want to group content by section headers.

```python
chunker = DocumentChunker(strategy="markdown_headers")
```

## html_headers

`HTMLHeaderTextSplitter` - Splits HTML by header tags.

Best for: HTML documents, web page content.

```python
chunker = DocumentChunker(strategy="html_headers")
```

## semantic

`SemanticChunker` - Uses embeddings to split text based on semantic similarity.

Best for: When you want chunks that maintain semantic coherence, better for RAG retrieval quality.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
chunker = DocumentChunker(
    strategy="semantic",
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
```

### Semantic Chunking Parameters

| Parameter | Default | Description |
|-----------|--------|-------------|
| `embeddings` | required | Embedding model (e.g., OpenAIEmbeddings) |
| `breakpoint_threshold_type` | `"percentile"` | Method for determining breakpoints |
| `breakpoint_threshold_amount` | `95` | Threshold value |
| `buffer_size` | `1` | Number of sentences to overlap |
| `min_chunk_size` | `0` | Minimum chunk size in characters |
| `add_start_index` | `False` | Add start index to metadata |
| `nb_suffix` | `1` | Number suffix for splits |

### Breakpoint Threshold Types

| Type | Description |
|------|-------------|
| `percentile` | Splits at the nth percentile of distances |
| `standard_deviation` | Splits at n standard deviations above mean |
| `interquartile` | Splits using interquartile range |
| `gradient` | Uses gradient of distance array (good for legal/medical texts) |

### Choosing a Threshold

- **Higher threshold** (e.g., 95-99): Fewer, larger chunks - less sensitive to meaning shifts
- **Lower threshold** (e.g., 70-80): More, smaller chunks - more sensitive to meaning shifts

```python
# More sensitive (more chunks)
chunker = DocumentChunker(
    strategy="semantic",
    embeddings=embeddings,
    breakpoint_threshold_amount=80,
)

# Less sensitive (fewer chunks)
chunker = DocumentChunker(
    strategy="semantic",
    embeddings=embeddings,
    breakpoint_threshold_amount=99,
)
```
