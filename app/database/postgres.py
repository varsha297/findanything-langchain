import psycopg

from app.config import DATABASE_URL


def get_connection():
    return psycopg.connect(DATABASE_URL)


def insert_document_chunks(
    document_id: str,
    owner_id: str,
    chunks,
    embeddings: list[list[float]],
):
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings."
        )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            for index, (chunk, embedding) in enumerate(
                zip(chunks, embeddings)
            ):
                page_number = chunk.metadata.get("page")

                metadata = {
                    "source": chunk.metadata.get("source"),
                    "page": page_number,
                }

                cursor.execute(
                    """
                    INSERT INTO langchain_document_chunks (
                        document_id,
                        owner_id,
                        chunk_index,
                        content,
                        metadata,
                        embedding,
                        embedding_model
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::vector,
                        %s
                    )
                    ON CONFLICT (
                        document_id,
                        chunk_index
                    )
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model
                    """,
                    (
                        document_id,
                        owner_id,
                        index,
                        chunk.page_content,
                        __import__("json").dumps(metadata),
                        str(embedding),
                        "text-embedding-3-small",
                    ),
                )

        connection.commit()