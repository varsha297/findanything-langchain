from langchain_openai import OpenAIEmbeddings

from app.config import OPENAI_API_KEY


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)


def embed_question(
    question: str,
) -> list[float]:

    return embedding_model.embed_query(
        question
    )