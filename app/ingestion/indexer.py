from langchain_openai import OpenAIEmbeddings

from app.config import OPENAI_API_KEY


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []

    return embedding_model.embed_documents(
        texts
    )