import sys

from app.database.langchain_search import (
    search_similar_chunks,
)
from app.retrieval.question_embedding import (
    embed_question,
)


def main():

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: "
            "python3 -m app.retrieval.test_search "
            "\"your question\""
        )

    question = " ".join(
        sys.argv[1:]
    )

    print(
        "Question:",
        question,
    )

    question_embedding = embed_question(
        question
    )

    print(
        "Question embedding dimensions:",
        len(question_embedding),
    )

    results = search_similar_chunks(
        question_embedding,
        limit=5,
    )

    print(
        f"Retrieved chunks: {len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            "Chunk ID:",
            result["chunkId"],
        )

        print(
            "Document ID:",
            result["documentId"],
        )

        print(
            "Chunk index:",
            result["chunkIndex"],
        )

        print(
            "Similarity:",
            result["similarity"],
        )

        print(
            "Content:"
        )

        print(
            result["content"][:500]
        )


if __name__ == "__main__":
    main()