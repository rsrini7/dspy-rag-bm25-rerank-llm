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