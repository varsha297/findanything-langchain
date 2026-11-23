# FindMe Anything – LangChain RAG Service

A Python-based Retrieval-Augmented Generation (RAG) service built using **FastAPI, LangChain, OpenAI Embeddings, PostgreSQL and pgvector**.

FindMe Anything is a document intelligence application that allows users to upload documents and ask natural-language questions about their content.

This repository contains the Python LangChain implementation of the RAG pipeline. It is maintained separately from the main application's native Node.js RAG implementation so that both approaches can be developed, tested and compared independently.

---

## 1. Project Overview

### Problem Statement

Users often have information distributed across multiple documents, PDFs and text files. Finding specific information manually is time-consuming.

Traditional keyword search has limitations:

- Users need to know the exact words used in a document.
- It struggles with semantic questions and different ways of expressing the same idea.
- It can be difficult to retrieve relevant context across multiple documents.
- It does not naturally generate contextual answers.

FindMe Anything addresses these problems using Retrieval-Augmented Generation.

### Key Features

- Document ingestion and processing.
- Text extraction and chunking.
- Semantic embeddings using OpenAI.
- Vector storage using PostgreSQL and pgvector.
- Semantic similarity search.
- LangChain-based RAG implementation.
- Document and owner-level retrieval boundaries.
- FastAPI endpoints for document processing and question answering.
- Separate storage for LangChain-generated chunks.

---

## 2. Architecture

The LangChain implementation runs as a separate Python service alongside the main FindMe Anything application.

The main application handles the document upload workflow and document metadata. The Python service implements the LangChain-based processing and retrieval pipeline.

```text
                    User
                      |
                      v
               Next.js Frontend
                      |
                      v
                 Node.js API
                      |
              Document Upload
                      |
             +--------+--------+
             |                 |
             v                 v
          Amazon S3        PostgreSQL
       Original Files    Document Metadata
                              |
                              |
                 LangChain Processing Service
                       Python / FastAPI
                              |
                              v
                       Document Loading
                              |
                              v
                        Text Extraction
                              |
                              v
                         Text Splitting
                              |
                              v
                       OpenAI Embeddings
                              |
                              v
                     PostgreSQL + pgvector
                              |
                              v
                       Vector Retrieval
                              |
                              v
                      Retrieved Context
                              |
                              v
                         LLM Generation
                              |
                              v
                        Final Answer
```

The architecture separates document storage, metadata management and AI processing.

The LangChain service shares the PostgreSQL database with the main application but uses its own chunk-storage table.

---

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | REST API framework |
| LangChain | RAG orchestration and document processing |
| OpenAI API | Embeddings and LLM integration |
| text-embedding-3-small | Embedding model |
| PostgreSQL | Relational database |
| pgvector | Vector storage and similarity search |
| psycopg | PostgreSQL connectivity |
| Pydantic | Request validation and data modelling |
| Uvicorn | ASGI application server |

### Why LangChain?

LangChain provides reusable abstractions for building LLM applications.

It offers components for:

- Document loaders.
- Text splitters.
- Embedding model integrations.
- Vector-store integrations.
- Retrievers.
- Prompt templates.
- LLM integrations and orchestration.

Instead of implementing every component from scratch, LangChain provides reusable building blocks.

The project uses these abstractions to implement and evaluate a framework-based RAG pipeline.

---

## 4. Understanding the RAG Pipeline

The RAG pipeline has two major stages:

1. Document ingestion.
2. Question answering.

### A. Document Ingestion

Document ingestion happens before a user asks a question.

**Step 1: Document loading**

The document is made available to the Python processing service.

The document content is extracted into machine-readable text.

**Step 2: Text chunking**

Large documents are divided into smaller pieces called chunks.

Why do we chunk documents?

- LLM context windows are limited.
- Retrieving entire documents is inefficient.
- Smaller chunks can improve retrieval precision.
- Relevant sections can be retrieved independently.
- Chunk-level metadata allows more targeted retrieval.

**Step 3: Generate embeddings**

Each chunk is converted into a numerical vector using an embedding model.

Configured model:

`text-embedding-3-small`

Embedding dimensions:

`1536`

An embedding represents the semantic meaning of a piece of text in vector space.

**Step 4: Store chunks and embeddings**

The generated chunks and their embeddings are stored in PostgreSQL using pgvector.

Each chunk is associated with its document and owner.

### Ingestion Flow

```text
Original Document
       |
       v
Document Loader
       |
       v
Text Extraction
       |
       v
Text Splitter
       |
       v
Document Chunks
       |
       v
OpenAI Embeddings
       |
       v
1536-dimensional Vectors
       |
       v
PostgreSQL + pgvector
```

### B. Question Answering

When a user asks a question, the system performs semantic retrieval.

**Step 1: User question**

Example:

"What are the main responsibilities mentioned in my employment agreement?"

**Step 2: Generate query embedding**

The question is converted into a vector using the same embedding model used during ingestion.

Using the same embedding model ensures that the query and stored document chunks occupy a compatible vector space.

**Step 3: Similarity search**

The query embedding is compared against the stored chunk embeddings.

The most relevant chunks are retrieved using vector similarity search.

**Step 4: Context construction**

The retrieved chunks are assembled into context for the LLM.

**Step 5: LLM generation**

The LLM uses the retrieved context and the original question to generate a contextual answer.

### Retrieval Flow

```text
User Question
      |
      v
Query Embedding
      |
      v
Vector Similarity Search
      |
      v
Relevant Document Chunks
      |
      v
Context Construction
      |
      v
LLM
      |
      v
Final Answer
```

---

## 5. Database Architecture

The application uses PostgreSQL with the pgvector extension.

The LangChain service shares the main application's PostgreSQL database while maintaining a separate chunk-storage table.

### Database Tables

| Table | Responsibility |
|---|---|
| `documents` | Stores document metadata |
| `document_chunks` | Stores chunks generated by the native RAG implementation |
| `langchain_document_chunks` | Stores chunks generated by the LangChain implementation |

### Why Separate Chunk Tables?

The project intentionally maintains two RAG implementations.

**Native implementation**

`document_chunks`

**LangChain implementation**

`langchain_document_chunks`

This separation allows us to:

- Compare chunking strategies.
- Compare retrieval results.
- Evaluate embedding and retrieval quality.
- Experiment with LangChain independently.
- Avoid mixing records from different implementations.

Both implementations can reference the same document metadata.

### Vector Storage

The configured embedding model generates vectors with 1536 dimensions.

Conceptually, the database column is:

```sql
embedding vector(1536)
```

The pgvector extension allows PostgreSQL to store and compare these vectors.

### Cosine Distance Search

A similarity search can use pgvector's cosine distance operator:

```sql
SELECT
    content,
    embedding <=> $1::vector AS distance
FROM langchain_document_chunks
WHERE owner_id = $2
ORDER BY distance ASC
LIMIT 5;
```

Here:

- `$1` is the query embedding.
- `$2` is the owner identifier.
- `<=>` calculates cosine distance.
- Lower distance indicates greater similarity.
- `LIMIT 5` retrieves the five closest matching chunks.

This is an illustrative query. The actual query must match the repository's current schema and parameter conventions.

---

## 6. API Endpoints

The FastAPI service exposes two primary endpoints.

### POST `/process`

Processes a document through the LangChain ingestion pipeline.

Responsibilities:

- Accept document-processing information.
- Load and extract document content.
- Split content into chunks.
- Generate embeddings.
- Persist LangChain chunks and vectors in PostgreSQL.

### POST `/ask`

Accepts a natural-language question and performs RAG-based question answering.

Responsibilities:

- Validate the question.
- Generate query embeddings.
- Retrieve relevant document chunks.
- Construct contextual prompts.
- Invoke the configured LLM.
- Return the generated answer.

### API Documentation

When the service is running, FastAPI provides interactive API documentation at:

`http://localhost:8000/docs`

Use the interactive documentation to inspect the actual request and response schemas implemented in the repository.

---

## 7. Local Development Setup

### Prerequisites

Install:

- Python 3.11+
- PostgreSQL with pgvector
- OpenAI API key
- Git

The LangChain service uses the shared FindMe Anything PostgreSQL database.

### Step 1: Clone the repository

```bash
git clone <repository-url>

cd findanything-langchain
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

If the repository uses `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Configure environment variables

Create a `.env` file in the repository root.

Example:

```env
OPENAI_API_KEY=your-openai-api-key

DATABASE_URL=postgresql://agentops:agentops_dev_password@localhost:5433/agentops
```

The database URL above is for the local development database configuration.

Keep credentials out of Git. Add `.env` to `.gitignore`.

### Step 5: Start the FastAPI service

If the FastAPI application entry point is `main.py`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If the application is inside a Python package, use the actual module path.

### Step 6: Open API documentation

```text
http://localhost:8000/docs
```

---

## 8. Environment Configuration

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Authentication for OpenAI APIs |
| `DATABASE_URL` | PostgreSQL connection string |

### Embedding Configuration

| Configuration | Value |
|---|---|
| Embedding provider | OpenAI |
| Model | `text-embedding-3-small` |
| Dimensions | `1536` |

The embedding model and dimensions must remain consistent between document ingestion and query embedding.

---

## 9. LangChain vs Native RAG

An important objective of this project is to understand the difference between implementing RAG using native components and using LangChain.

| Area | Native RAG | LangChain |
|---|---|---|
| Text extraction | Custom implementation | Loader integrations |
| Chunking | Custom implementation | Text splitters |
| Embeddings | Direct API integration | Embedding abstractions |
| Vector retrieval | Custom SQL | Retriever/vector-store abstractions |
| Prompt construction | Custom implementation | Prompt templates |
| LLM integration | Direct API calls | Model integrations |
| Flexibility | Direct control over components | Reusable abstractions |
| Development effort | More implementation work | Faster experimentation |

Neither approach is automatically better.

**Native RAG**

- Greater control over individual components.
- Direct visibility into SQL and retrieval logic.
- More responsibility for implementing and maintaining integrations.

**LangChain RAG**

- Reusable components and integrations.
- Faster experimentation.
- Less custom orchestration code.
- Framework abstractions can introduce additional complexity.

The objective is to evaluate both implementations using the same underlying documents and compare their behavior.

---

## 10. Security Considerations

The application must enforce document ownership during retrieval.

Important considerations:

- Every document must be associated with an owner.
- Retrieval must be scoped to the authenticated owner.
- Users must never retrieve another user's document chunks.
- API keys must be stored securely.
- Database credentials must not be committed.
- Document access must be validated before processing.
- API request inputs must be validated.

Owner-level filtering must happen in the database query itself, not only in the frontend.

In production, owner identity should come from a trusted authentication context rather than relying on an arbitrary owner ID supplied by the client.

---

## 11. Current Implementation Status

The following components have been implemented and exercised during development.

- [x] Separate Python LangChain service.
- [x] FastAPI application.
- [x] PostgreSQL connectivity.
- [x] OpenAI embedding integration.
- [x] `text-embedding-3-small` configuration.
- [x] 1536-dimensional embeddings.
- [x] Separate `langchain_document_chunks` table.
- [x] Document processing endpoint (`/process`).
- [x] Question endpoint (`/ask`).
- [x] CORS middleware for browser access.
- [x] Sample document ingestion successfully exercised.

The repository should be checked to confirm the exact final state of the LLM answer-generation chain and its integration tests before marking end-to-end question answering as production-ready.

### Further Production-Hardening Work

- [ ] Authentication and authorization integration.
- [ ] Robust retry and timeout handling.
- [ ] Background processing for large documents.
- [ ] Retrieval evaluation and benchmarking.
- [ ] Hybrid search (keyword + vector).
- [ ] Reranking.
- [ ] Observability and structured logging.
- [ ] Automated unit and integration tests.
- [ ] Production deployment.

---

## 12. Future Improvements

### Retrieval Quality

- Experiment with different chunk sizes and overlaps.
- Evaluate top-K retrieval strategies.
- Implement metadata filtering.
- Introduce hybrid search.
- Experiment with reranking.
- Evaluate retrieval relevance.

### Production Scalability

- Move long-running ingestion work to background workers.
- Implement idempotent document processing.
- Add retry and failure handling.
- Add batch embedding generation.
- Introduce database connection pooling.
- Monitor embedding API usage and latency.

### Evaluation and Benchmarking

Compare the native and LangChain implementations using:

- Retrieval precision.
- Retrieval recall.
- Context relevance.
- Answer relevance.
- Response latency.
- Token consumption.
- Embedding cost.

The goal is to compare both implementations using the same questions and document set rather than relying on subjective answers.

---

## 13. Learning Objectives

This project demonstrates practical experience with:

1. Retrieval-Augmented Generation.
2. LangChain architecture.
3. Document loaders and text splitters.
4. Embedding generation.
5. Vector databases.
6. Semantic similarity search.
7. PostgreSQL and pgvector.
8. FastAPI backend development.
9. LLM integration.
10. AI application architecture.
11. Native versus framework-based RAG.
12. Production considerations for AI-powered applications.

---

## 14. Project Goal

The long-term goal is to build a scalable document intelligence platform where users can upload documents, search across their knowledge base and receive contextual answers using Generative AI.

The project also serves as a practical comparison between custom-built RAG systems and framework-based implementations using LangChain.

**Core idea: Upload once, process intelligently, and ask anything about your documents.**
