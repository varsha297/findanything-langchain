import sys

from app.ingestion.loader import (
    load_pdf,
)

from app.ingestion.splitter import (
    split_documents,
)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m app.ingestion.test_pipeline <pdf-path>"
        )

    file_path = sys.argv[1]

    print(
        f"Loading PDF: {file_path}"
    )

    documents = load_pdf(
        file_path
    )

    print(
        f"Loaded pages: {len(documents)}"
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Generated chunks: {len(chunks)}"
    )

    for index, chunk in enumerate(
        chunks[:3]
    ):
        print()
        print(
            f"--- Chunk {index} ---"
        )

        print(
            "Page:",
            chunk.metadata.get(
                "page"
            ),
        )

        print(
            "Characters:",
            len(
                chunk.page_content
            ),
        )

        print(
            chunk.page_content[:500]
        )


if __name__ == "__main__":
    main()