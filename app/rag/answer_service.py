from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY
from app.database.langchain_search import search_similar_chunks
from app.retrieval.question_embedding import embed_question


llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
)


def answer_question(
    question: str,
    limit: int = 5,
) -> dict:

    # -----------------------------------------
    # 1. Convert question into an embedding
    # -----------------------------------------

    question_embedding = embed_question(
        question
    )

    # -----------------------------------------
    # 2. Retrieve relevant chunks
    # -----------------------------------------

    chunks = search_similar_chunks(
        question_embedding,
        limit=limit,
    )

    if not chunks:
        return {
            "answer": (
                "I could not find enough "
                "information in the provided documents."
            ),
            "sources": [],
        }

    # -----------------------------------------
    # 3. Build context
    # -----------------------------------------

    context_parts = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        context_parts.append(
            f"""
SOURCE {index}

Document ID:
{chunk["documentId"]}

Chunk:
{chunk["chunkIndex"]}

Similarity:
{chunk["similarity"]}

Content:
{chunk["content"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    # -----------------------------------------
    # 4. Ask the LLM
    # -----------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY
the information contained in the provided
document context.

If the context does not contain enough
information to answer the question, say:

"I could not find enough information
in the provided documents."

Do not invent information.

Question:
{question}

Document context:
{context}
"""

    response = llm.invoke(
        prompt
    )

    answer = response.content

    # -----------------------------------------
    # 5. Return answer + sources
    # -----------------------------------------

    sources = [
        {
            "chunkId": chunk["chunkId"],
            "documentId": chunk["documentId"],
            "chunkIndex": chunk["chunkIndex"],
            "similarity": chunk["similarity"],
            "metadata": chunk["metadata"],
        }
        for chunk in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
    }