import os

from dotenv import load_dotenv


load_dotenv()


PORT = int(
    os.getenv("PORT", "4100")
)

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required."
    )


if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is required."
    )