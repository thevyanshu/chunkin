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
