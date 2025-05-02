# Decision Log

This file records architectural and implementation decisions using a list format.
2025-04-23 10:21:10 - Confirmed architectural decisions: hybrid RAG pipeline (BM25 + ChromaDB), neural reranking (CrossEncoder), LLM generation (OpenRouter), modular retriever and pipeline classes, separation of CLI and Streamlit DBs, configuration via .env and Pydantic config.
2025-04-23 10:24:07 - Expanded architectural decision entry with rationale and implementation details.

## Decision

Adopt a hybrid Retrieval-Augmented Generation (RAG) pipeline combining BM25 keyword search and dense vector search (ChromaDB with Sentence Transformers), followed by neural reranking (CrossEncoder) and optional LLM answer generation (OpenRouter). Use modular retriever and pipeline classes, separate DBs for CLI and Streamlit, and configuration via .env and Pydantic config.

## Rationale

- Hybrid retrieval leverages the strengths of both keyword (BM25) and semantic (embedding) search for improved recall and precision.
- Neural reranking (CrossEncoder) ensures the most relevant results are prioritized for the LLM.
- Modular class-based architecture (DSPy modules) enables extensibility and testability.
- Separate ChromaDB directories for CLI and Streamlit prevent concurrency issues.
- Environment/configuration-driven design (via .env and Pydantic) allows for flexible deployment and reproducibility.
- Optional LLM step (OpenRouter) allows users to toggle between context-only and full answer generation modes.

## Implementation Details

- `RAGHybridFusedRerank` (DSPy Module) orchestrates the hybrid retrieval, fusion, reranking, and LLM generation steps.
- `ChromaRetriever` and `BM25Retriever` encapsulate vector and keyword search logic, respectively.
- `utils.py` provides functions to load components, create retrievers, and instantiate the pipeline.
- Configuration is loaded from `.env` using Pydantic (`config.py`).
- Streamlit app (`app.py`) and CLI (`cli.py`) use separate ChromaDB paths.
- Demo data and modular utilities facilitate unit and integration testing.