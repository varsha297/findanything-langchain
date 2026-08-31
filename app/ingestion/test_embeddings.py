import sys

from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.indexer import generate_embeddings


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python3 -m app.ingestion.test_embeddings <pdf-path>"
        )

    file_path = sys.argv[1]

    print(
        f"Loading PDF: {file_path}"
    )

    documents = load_pdf(
        file_path
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Generated chunks: {len(chunks)}"
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = generate_embeddings(
        texts
    )

    print(
        f"Generated embeddings: {len(embeddings)}"
    )

    if embeddings:
        print(
            f"Embedding dimensions: {len(embeddings[0])}"
        )


if __name__ == "__main__":
    main()