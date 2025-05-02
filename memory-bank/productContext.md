# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-04-22 23:21:21 - Log of updates made will be appended as footnotes to the end of this file.

*

## Project Goal

*   To build a Retrieval-Augmented Generation (RAG) pipeline that combines keyword search (BM25) and dense vector search (ChromaDB with Sentence Transformers) to retrieve relevant context for a Language Model (LLM) to generate an answer.

## Key Features

*   Hybrid Retrieval (BM25 and ChromaDB)
*   Fusion & Deduplication
*   Neural Reranking (CrossEncoder)
*   LLM Generation
*   Streamlit Interface (`app.py`)
*   Command-Line Interface (`cli.py`)

## Overall Architecture

*   The pipeline combines keyword search (BM25) and dense vector search (ChromaDB with Sentence Transformers) followed by a reranking step (Cross-Encoder) to retrieve relevant context for a Language Model (LLM) to generate an answer. The configuration is loaded from `.env` via `config.py`. The main RAG pipeline is defined in `rag_pipeline.py`, which uses custom retrievers defined in `retrievers.py` and utility functions defined in `utils.py`.

2025-04-23 10:18:09 - Memory Bank reviewed by user request. No changes to project scope, features, or architecture at this time.

---
2025-05-02 11:09:56 - Updated to reflect details of api.py and api_client.py as follows:

### API Server (`api.py`)
- Implements a LitServe-based REST API for the DSPy RAG pipeline.
- Loads all core components (embedder, reranker, LLM, retrievers, ChromaDB client) on startup via `setup()`.
- Handles both document indexing and retrieval automatically; re-indexes if ChromaDB is empty or out-of-sync.
- Exposes `/predict` endpoint for question answering (accepts JSON with `question` field; returns answer and context).
- Exposes `/upload_file` endpoint for uploading and indexing new documents (multipart/form-data); updates both ChromaDB and BM25 indices.
- Uses modular utility functions from `utils.py` for component loading, indexing, and pipeline creation.
- Supports toggling between context-only and LLM answer modes (via API payload).

### API Client (`api_client.py`)
- Command-line script to interact with the API server.
- Supports uploading a file and querying in a single call (sends file to `/upload_file`, then queries `/predict`).
- Supports toggling LLM answer generation with `--llm` flag.
- Demonstrates payload structure and expected responses for both endpoints.

[2025-05-02 11:09:56] - Added/clarified API and client details based on recent code and README review.