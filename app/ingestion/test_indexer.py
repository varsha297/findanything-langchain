import sys

from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.indexer import generate_embeddings
from app.database.postgres import insert_document_chunks


def main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: "
            "python3 -m app.ingestion.test_indexer "
            "<pdf-path> <document-id> <owner-id>"
        )

    file_path = sys.argv[1]
    document_id = sys.argv[2]
    owner_id = sys.argv[3]

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

    print(
        f"Embedding dimensions: "
        f"{len(embeddings[0])}"
    )

    insert_document_chunks(
        document_id=document_id,
        owner_id=owner_id,
        chunks=chunks,
        embeddings=embeddings,
    )

    print(
        "LangChain chunks stored successfully."
    )


if __name__ == "__main__":
    main()