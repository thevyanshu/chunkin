"""Workable example for DocumentChunker module."""

from doc_chunker import DocumentChunker


def main():
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

    test_files = [
        "sample.pdf",
        "sample.docx",
        "sample.txt",
        "sample.md",
        "sample.csv",
        "sample.xlsx",
        "sample.pptx",
    ]

    for file_path in test_files:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print("=" * 60)

        try:
            chunks = chunker.create_chunks(file_path)
            print(f"Total chunks created: {len(chunks)}")

            for i, chunk in enumerate(chunks[:3]):
                print(f"\n--- Chunk {i} ---")
                print(f"Content ({len(chunk.page_content)} chars): {chunk.page_content[:200]}...")
                print(f"Metadata: {chunk.metadata}")

            if len(chunks) > 3:
                print(f"\n... and {len(chunks) - 3} more chunks")

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Add a sample file to test this format.")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
