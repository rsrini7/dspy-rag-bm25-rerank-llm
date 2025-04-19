# DSPy RAG Hybrid Pipeline with BM25, Embedding, Rerank and LLM

## Architecture Diagram

![Hybrid RAG Pipeline Architecture](assets/architecture.png)

## Architecture Flow Diagram

![Hybrid RAG Pipeline Architecture Flow](assets/architecture-flow.png)

## Architecture Flow

1.  **Query Input**: The user provides a query.
2.  **Hybrid Retrieval**:
    *   **Vector Retrieval**: The query is embedded using SentenceTransformer and searched in ChromaDB for similar documents.
    *   **Keyword Retrieval**: The query is tokenized (NLTK), and BM25Okapi retrieves relevant documents by keyword match.
3.  **Fusion & Deduplication**: Results from both retrieval methods are merged and deduplicated to form a single candidate set.
4.  **CrossEncoder Reranker**: The candidate documents are reranked for relevance to the query using a CrossEncoder model.
5.  **LLM Generation**: The top reranked context is sent to an LLM (via OpenRouter) to generate a final answer.
6.  **Answer Output**: The answer is returned to the user.

---

## Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built with DSPy. It combines keyword search (BM25) and dense vector search (ChromaDB with Sentence Transformers) followed by a reranking step (Cross-Encoder) to retrieve relevant context for a Language Model (LLM) to generate an answer. The project includes both a command-line interface (`main.py`) and an interactive Streamlit web application (`app.py`).

---

## Features

*   **Hybrid Retrieval**: Combines dense embedding (vector) search and BM25 keyword search.
*   **Fusion & Deduplication**: Merges results from both retrieval strategies.
*   **Neural Reranking**: Uses a CrossEncoder model to rerank the fused results for relevance.
*   **LLM Generation**: Generates answers using a large language model, conditioned on the top reranked context.
*   **Streamlit Interface**: Provides a web UI for interacting with the RAG pipeline.
*   **`src` Layout**: Organizes source code cleanly within a `src` directory.

---

## Step-by-Step Setup & Usage

### 1. Install [uv](https://github.com/astral-sh/uv) (if not already)

```bash
pip install uv
```

### 2. Sync Dependencies and Create Virtual Environment
This project uses `pyproject.toml` for dependency management. Run:
```bash
uv sync
```
This will create a `.venv` directory and install all dependencies.

### 3. Activate the Virtual Environment
- On **Unix/Mac**:
  ```bash
  source .venv/bin/activate
  ```
- On **Windows**:
  ```bash
  .venv\Scripts\activate
  ```

### 4. Prepare Environment Variables
Create a `.env` file (see `.env.example`) with the following variables:
```
OPENROUTER_API_KEY=your_openrouter_api_key
EMBEDDER_MODEL=all-mpnet-base-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
LLM_MODEL=openai/gpt-3.5-turbo
K_EMBEDDING_RETRIEVAL=10
K_BM25_RETRIEVAL=10
K_RERANK=3
CHROMA_DB_PATH=./chroma_db_dspy
CHROMA_COLLECTION_NAME=hybrid_rag_docs
```

### 5. Download NLTK Data (if needed)
The scripts (`main.py` and `app.py`) will automatically attempt to download 'punkt' and 'stopwords' via the `ensure_nltk_resources` function if they are not found in the standard NLTK data path.

### 6. Run the Application

You have two main ways to run the application:

*   **Command-Line Interface (main.py):**
    This script loads the default documents, indexes them, runs predefined queries, and shows the pipeline steps. Use `uv run` to execute it within the managed environment:
    ```bash
    uv run python main.py
    ```

*   **Streamlit Web Interface (app.py):**
    This provides an interactive web UI. It attempts to load previously indexed data automatically. If none exists, or if you want to index new data, you can upload/paste text and index it via the sidebar. Use `uv run` to start the Streamlit server:
    ```bash
    uv run streamlit run app.py
    ```

### 7. What Happens Internally (Example: `main.py` execution)?
- **Load Components (`utils.load_components`)**: Loads configuration (`config.py`), initializes embedding (`SentenceTransformer`), reranking (`CrossEncoder`), and LLM (`dspy.LM via OpenRouter`) models. Initializes the ChromaDB client (`chromadb.PersistentClient`). Configures DSPy settings globally.
- **Index Data (`utils.index_chroma_data`, `utils.create_bm25_index`)**:
    - Indexes the default documents (`data.py`) into ChromaDB, creating vector embeddings.
    - Preprocesses documents (`bm25_utils.py`) and builds a BM25 keyword index (`rank_bm25`).
- **Create Retrievers (`utils.create_retrievers`)**: Sets up both Chroma (`ChromaRetriever`) and BM25 (`BM25Retriever`) retrievers for DSPy, using the indexed data.
- **Create RAG Pipeline (`utils.create_rag_pipeline`)**: Instantiates the main DSPy module (`RAGHybridFusedRerank`) which combines both retrievers, fuses and deduplicates results, reranks using the CrossEncoder, and prepares for answer generation with the LLM.
- **Run Queries (`main.py`)**: Executes the RAG pipeline with example questions and prints the results.
- **Streamlit App (`app.py`)**: Follows a similar flow but uses a separate ChromaDB path (`_st` suffix). It attempts to load existing data first. If new data is provided via the UI, it calls the same utility functions (`index_chroma_data`, `create_bm25_index`, `create_retrievers`, `create_rag_pipeline`) to process and index it before enabling search.

### 8. Customization
- Add your own default documents to the `DOCUMENTS` list in `src/dspy_rag_app/data.py`.
- Change retrieval, rerank parameters, model names, or paths via the `.env` file.
- Modify the core pipeline logic in `src/dspy_rag_app/rag_pipeline.py` or the utility functions in `src/dspy_rag_app/utils.py`.

### 9. Troubleshooting
- Ensure your `.env` is set up and API keys are valid.
- If you see NLTK errors, delete `.venv/nltk_data` and rerun.
- For LLM errors, check your OpenRouter API key and model name.

---

## Project Structure

```plaintext
/Users/srini/Ws/dspy-rag-bm25-rerank-llm/
├── .env                    # Local environment variables (ignored by git)
├── .env.example            # Example environment variable file
├── .git/                   # Git repository data
├── .gitignore              # Git ignore configuration
├── .venv/                  # Virtual environment created by uv
├── app.py                  # Streamlit application entry point (uses utils.py)
├── assets/                 # Directory for static assets (e.g., diagrams)
│   ├── .keep               # Placeholder file
│   ├── architecture.png
│   └── architecture-flow.png
├── main.py                 # Command-line script entry point (uses utils.py)
├── pyproject.toml          # Project metadata and dependencies (used by uv/setuptools)
├── README.md               # This guide
├── src/                    # Main source code directory
│   └── dspy_rag_app/       # The core Python package for the RAG application
│       ├── __init__.py     # Makes dspy_rag_app a Python package
│       ├── bm25_utils.py   # Utilities for BM25 indexing (preprocessing, NLTK)
│       ├── config.py       # Loads configuration settings (models, paths, keys) from .env
│       ├── data.py         # Defines or loads the default document corpus for main.py
│       ├── rag_pipeline.py # Defines the main RAG DSPy module/pipeline (RAGHybridFusedRerank)
│       ├── retrievers.py   # Custom DSPy retriever modules (ChromaRetriever, BM25Retriever)
│       └── utils.py        # Shared utility functions (load components, index data, create retrievers/pipeline) used by main.py and app.py
├── uv.lock                 # uv lock file for reproducible dependencies
└── chroma_db_dspy/         # Default ChromaDB storage location (for main.py, ignored by git)
└── chroma_db_dspy_st/      # Default ChromaDB storage location (for app.py, ignored by git)
```

---

## References

*   [DSPy](https://github.com/stanfordnlp/dspy)
*   [ChromaDB](https://www.trychroma.com/)
*   [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
*   [Sentence Transformers](https://www.sbert.net/)
*   [OpenRouter](https://openrouter.ai/)
*   [uv](https://github.com/astral-sh/uv)
*   [Streamlit](https://streamlit.io/)
