"""Workable example for DocumentChunker module."""

from doc_chunker import DocumentChunker


def main():
    test_files = [
        "sample.pdf",
        "sample.docx",
        "sample.txt",
        "sample.md",
        "sample.csv",
        "sample.xlsx",
        "sample.pptx",
    ]

    print("=" * 60)
    print("Basic Single File Chunking")
    print("=" * 60)

    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

    for file_path in test_files:
        try:
            chunks = chunker.create_chunks(file_path)
            print(f"\n{file_path}: {len(chunks)} chunks")
        except FileNotFoundError:
            print(f"\n{file_path}: not found (skip)")
        except Exception as e:
            print(f"\n{file_path}: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Listing All Chunks")
    print("=" * 60)
    print(chunker.list_chunks())

    print("\n" + "=" * 60)
    print("Getting Specific Chunks")
    print("=" * 60)
    chunks = chunker.get_chunks("sample.pdf")
    if chunks:
        print(f"sample.pdf: {len(chunks)} chunks")


def with_output_dir():
    print("\n" + "=" * 60)
    print("Chunking with Output Directory")
    print("=" * 60)

    chunker = DocumentChunker(output_dir="chunks")

    try:
        chunks = chunker.create_chunks("sample.pdf")
        print(f"Created {len(chunks)} chunks")
        print(f"Chunks saved to: chunks/sample_chunks.json")
    except FileNotFoundError:
        print("sample.pdf not found")
    except Exception as e:
        print(f"Error: {e}")


def batch_example():
    print("\n" + "=" * 60)
    print("Batch Chunking with Output Directory")
    print("=" * 60)

    chunker = DocumentChunker(output_dir="chunks")

    try:
        all_chunks = chunker.batch_chunks(
            "path/to/documents",
            extensions=[".pdf", ".docx"],
            recursive=True,
        )
        print(f"Processed {len(all_chunks)} files")
        print("\nSummary:")
        print(chunker.list_chunks())
    except NotADirectoryError:
        print("Directory not found")


def semantic_example():
    print("\n" + "=" * 60)
    print("Semantic Chunking Example")
    print("=" * 60)

    try:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()

        semantic_chunker = DocumentChunker(
            strategy="semantic",
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

        chunks = semantic_chunker.create_chunks("sample.pdf")
        print(f"Total semantic chunks: {len(chunks)}")

    except FileNotFoundError:
        print("sample.pdf not found")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
    with_output_dir()
    batch_example()
    semantic_example()
