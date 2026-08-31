from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdf(
    file_path: str,
) -> list[Document]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    loader = PyMuPDFLoader(
        str(path)
    )

    documents = loader.load()

    return documents