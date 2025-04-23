# System Patterns *Optional*

This file documents recurring patterns and standards used in the project.
It is optional, but recommended to be updated as the project evolves.
2025-04-22 23:21:44 - Log of updates made.
2025-04-23 10:18:09 - Memory Bank reviewed. No new system patterns or standards identified in this session.
2025-04-23 10:21:10 - System patterns updated after deep codebase review.

## Coding Patterns
* Modular class-based design for retrievers and pipeline (DSPy modules)
* Use of Pydantic for configuration
* NLTK-based preprocessing for BM25
* Logging and error handling throughout utilities
* Separation of UI (Streamlit) and CLI logic

## Architectural Patterns
* Hybrid retrieval (BM25 + dense vector)
* Fusion and deduplication of results
* Neural reranking (CrossEncoder)
* LLM generation as optional, pluggable step
* Environment/configuration-driven pipeline
* Persistent vector DB separation for UI/CLI

## Testing Patterns
* Use of demo data in `data.py`
* Modular utilities and retrievers for unit testing
* Configurable pipeline for integration testing