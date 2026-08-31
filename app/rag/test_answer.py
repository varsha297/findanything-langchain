import sys

from app.rag.answer_service import answer_question


def main():

    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python3 -m app.rag.test_answer "question"'
        )

    question = " ".join(
        sys.argv[1:]
    )

    print(
        "Question:",
        question,
    )

    result = answer_question(
        question
    )

    print(
        "\n================ ANSWER ================\n"
    )

    print(
        result["answer"]
    )

    print(
        "\n================ SOURCES ================\n"
    )

    for source in result["sources"]:
        print(
            {
                "chunkId": source["chunkId"],
                "chunkIndex": source["chunkIndex"],
                "similarity": source["similarity"],
            }
        )


if __name__ == "__main__":
    main()