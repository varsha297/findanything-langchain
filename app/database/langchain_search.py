from typing import Any

import psycopg

from app.config import DATABASE_URL


def search_similar_chunks(
    question_embedding: list[float],
    limit: int = 5,
) -> list[dict[str, Any]]:

    embedding_text = str(
        question_embedding
    )

    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    document_id,
                    chunk_index,
                    content,
                    metadata,
                    embedding <=> %s::vector AS distance
                FROM langchain_document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    embedding_text,
                    embedding_text,
                    limit,
                ),
            )

            rows = cursor.fetchall()

    results = []

    for row in rows:
        (
            chunk_id,
            document_id,
            chunk_index,
            content,
            metadata,
            distance,
        ) = row

        similarity = 1 - float(distance)

        results.append(
            {
                "chunkId": str(chunk_id),
                "documentId": str(document_id),
                "chunkIndex": chunk_index,
                "content": content,
                "metadata": metadata,
                "similarity": similarity,
            }
        )

    return results