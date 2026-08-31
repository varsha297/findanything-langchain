from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import PORT
from app.database.postgres import insert_document_chunks
from app.ingestion.indexer import generate_embeddings
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.storage.s3 import download_s3_object


app = FastAPI(
    title="FindAnything LangChain Service",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST MODELS
# ============================================================


class ProcessDocumentRequest(BaseModel):
    documentId: str
    ownerId: str
    s3Bucket: str
    s3Key: str
    originalName: str


class AskRequest(BaseModel):
    question: str
    limit: int = 5


# ============================================================
# HEALTH
# ============================================================


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "findanything-langchain",
    }


# ============================================================
# DOCUMENT PROCESSING
# ============================================================


@app.post("/process")
def process_document(
    request: ProcessDocumentRequest,
):
    temporary_file_path = None

    try:
        print(
            "Processing document with LangChain.",
            {
                "documentId": request.documentId,
                "bucket": request.s3Bucket,
                "key": request.s3Key,
                "originalName": request.originalName,
            },
        )

        # ----------------------------------------------------
        # 1. Download original document from S3
        # ----------------------------------------------------

        temporary_file_path = download_s3_object(
            bucket=request.s3Bucket,
            key=request.s3Key,
            original_name=request.originalName,
        )

        print(
            "Document downloaded from S3.",
            {
                "documentId": request.documentId,
                "temporaryFile": temporary_file_path,
            },
        )

        # ----------------------------------------------------
        # 2. Extract document text
        # ----------------------------------------------------

        documents = load_pdf(
            temporary_file_path,
        )

        print(
            "Document loaded.",
            {
                "documentId": request.documentId,
                "pageCount": len(documents),
            },
        )

        # ----------------------------------------------------
        # 3. Split document into chunks
        # ----------------------------------------------------

        chunks = split_documents(
            documents,
        )

        print(
            "Document chunked.",
            {
                "documentId": request.documentId,
                "chunkCount": len(chunks),
            },
        )

        # ----------------------------------------------------
        # 4. Generate embeddings
        # ----------------------------------------------------

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = generate_embeddings(
            texts,
        )

        embedding_dimensions = (
            len(embeddings[0])
            if embeddings
            else 0
        )

        print(
            "Document embeddings generated.",
            {
                "documentId": request.documentId,
                "embeddingCount": len(embeddings),
                "dimensions": embedding_dimensions,
            },
        )

        # ----------------------------------------------------
        # 5. Store LangChain chunks + embeddings
        # ----------------------------------------------------

        insert_document_chunks(
            document_id=request.documentId,
            owner_id=request.ownerId,
            chunks=chunks,
            embeddings=embeddings,
        )

        print(
            "LangChain document indexed.",
            {
                "documentId": request.documentId,
                "chunkCount": len(chunks),
            },
        )

        # ----------------------------------------------------
        # 6. Return processing result
        # ----------------------------------------------------

        return {
            "status": "ready",
            "documentId": request.documentId,
            "chunkCount": len(chunks),
            "embeddingCount": len(embeddings),
            "embeddingDimensions": embedding_dimensions,
        }

    except Exception as error:
        print(
            "LangChain processing failed.",
            {
                "documentId": request.documentId,
                "error": str(error),
            },
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:
        # ----------------------------------------------------
        # 7. Delete temporary downloaded file
        # ----------------------------------------------------

        if temporary_file_path:
            Path(
                temporary_file_path,
            ).unlink(
                missing_ok=True,
            )

            print(
                "Temporary file cleaned up.",
                {
                    "documentId": request.documentId,
                },
            )


# ============================================================
# QUESTION ANSWERING
# ============================================================


@app.post("/ask")
def ask_question(
    request: AskRequest,
):
    try:
        from app.rag.answer_service import (
            answer_question,
        )

        question = request.question.strip()

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question is required.",
            )

        # ----------------------------------------------------
        # Validate retrieval limit
        # ----------------------------------------------------

        if request.limit < 1 or request.limit > 20:
            raise HTTPException(
                status_code=400,
                detail="limit must be between 1 and 20.",
            )

        print(
            "Processing LangChain question.",
            {
                "question": question,
                "limit": request.limit,
            },
        )

        # ----------------------------------------------------
        # Run LangChain RAG
        # ----------------------------------------------------

        result = answer_question(
            question=question,
            limit=request.limit,
        )

        print(
            "LangChain question completed.",
            {
                "sourceCount": len(
                    result["sources"],
                ),
            },
        )

        # ----------------------------------------------------
        # Return answer + sources
        # ----------------------------------------------------

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "LangChain question failed.",
            {
                "error": str(error),
            },
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# LOCAL DEVELOPMENT ENTRYPOINT
# ============================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )