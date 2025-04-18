# DSPy RAG Hybrid Pipeline with BM25, Embedding, Rerank and LLM
---
## Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built with DSPy. It combines keyword search (BM25) and dense vector search (ChromaDB with Sentence Transformers) followed by a reranking step (Cross-Encoder) to retrieve relevant context for a Language Model (LLM) to generate an answer. The project includes both a command-line interface (`main.py`) with improved argument handling and an interactive Streamlit web application (`app.py`). ChromaDB telemetry is disabled, and separate database paths are used for the CLI and Streamlit app to prevent conflicts.

---
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

## Features

*   **Hybrid Retrieval**: Combines dense embedding (vector) search and BM25 keyword search.
*   **Fusion & Deduplication**: Merges results from both retrieval strategies.
*   **Neural Reranking**: Uses a CrossEncoder model to rerank the fused results for relevance.
*   **LLM Generation**: Generates answers using a large language model, conditioned on the top reranked context.
*   **Streamlit Interface**: Provides a web UI for interacting with the RAG pipeline.
*   **Command-Line Interface**: Offers a CLI (`main.py`) for indexing and querying, with clear argument requirements.
*   **Separate DB Paths**: Uses distinct ChromaDB directories for CLI (`./chroma_db_dspy`) and Streamlit (`./chroma_db_dspy_st`) modes.
*   **ChromaDB Telemetry Disabled**: Configured to prevent sending anonymized usage data.
*   **Logging Control**: Suppresses verbose `INFO` logs from underlying libraries like `httpx` and `LiteLLM`.
*   **`src` Layout**: Organizes source code cleanly within a `src` directory.

## Optional LLM Generation (Streamlit & CLI)

Both the Streamlit app and the CLI now support toggling LLM answer generation. You can choose to either generate an answer using a large language model (LLM), or just view the top reranked context (retrieved documents) without LLM generation.

### Streamlit Web App (`app.py`)

When you run the Streamlit app, you'll see a checkbox labeled **"Enable LLM Generation"** in the sidebar:

- **Checked:** The app will generate and display an LLM answer to your query (using the reranked context as input).
- **Unchecked:** The app will only display the reranked context (top documents), without generating an LLM answer.

This allows you to control LLM usage interactively during your session.

### Command-Line Interface (`main.py`)

The CLI provides a `--llm` flag to control LLM generation:

- **With `--llm`:**
  ```bash
  uv run python main.py --query "What is DSPy?" --llm
  ```
  The pipeline will generate and print the LLM answer.
- **Without `--llm`:**
  ```bash
  uv run python main.py --query "What is DSPy?"
  ```
  Only the reranked context (top documents) will be printed, with no LLM answer.

This makes it easy to use the pipeline for pure retrieval/rerank or full RAG with LLM, depending on your needs.

**Note:** LLM generation requires a valid API key and model configuration (see environment variables section below).

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
```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
EMBEDDER_MODEL=all-mpnet-base-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
LLM_MODEL=openai/gpt-3.5-turbo
K_EMBEDDING_RETRIEVAL=10
K_BM25_RETRIEVAL=10
K_RERANK=3
CHROMA_DB_PATH=./chroma_db_dspy         # Path for main.py
CHROMA_DB_PATH_ST=./chroma_db_dspy_st   # Path for app.py (Streamlit)
CHROMA_COLLECTION_NAME=hybrid_rag_docs
```

**Configuration Usage Update:**
All configuration variables (model names, retrieval parameters, ChromaDB paths, etc.) are now accessed via the `config` object in code. Do **not** import configuration values directly from `config.py`—instead, use `config.<VAR>` (e.g., `config.EMBEDDER_MODEL`, `config.K_BM25_RETRIEVAL`). This ensures a single source of truth and easier environment-driven configuration.

- The `.env` file is loaded on startup, and all config values are available as attributes of the `config` object in `src/dspy_rag_app/config.py`.
- If you add new config variables, update both `.env` and the `Config` class in `config.py`.

### 5. Download NLTK Data (if needed)
The scripts (`main.py` and `app.py`) will automatically attempt to download 'punkt' and 'stopwords' via the `ensure_nltk_resources` function if they are not found in the standard NLTK data path.

### 6. Run the Application

You have two main ways to run the application:

*   **Command-Line Interface (main.py):**
    This script requires a query (`--query`) to run. Optionally, you can provide a file (`--file`) containing documents (one per line) to index its content. **If `--file` is not provided, the script uses a default set of documents defined in `src/dspy_rag_app/data.py`.** If you provide a file, a query is mandatory. Use `uv run` to execute it within the managed environment:

    **Arguments:**
    *   `-f, --file FILE_PATH`: (Optional) Path to a `.txt` file containing documents (one per line). If provided, these documents will be indexed instead of the defaults. Requires `--query` to be specified as well.
    *   `-q, --query QUERY_TEXT`: (Required) Query string to run against the indexed documents (either default or from `--file`).
    *   `--llm`: (Optional) Enable LLM answer generation. If not set, only reranked context is shown.

    **Examples:**
    ```bash
    # Use default documents and run a specific query (reranked context only)
    uv run python main.py --query 'What is DSPy?'

    # Use default documents and get an LLM answer
    uv run python main.py --query 'What is DSPy?' --llm

    # Index 'my_docs.txt' and run a specific query (LLM answer)
    uv run python main.py --file path/to/my_docs.txt --query 'Summarize the file.' --llm
    ```

*   **Streamlit Web Interface (app.py):**
    This provides an interactive web UI. It attempts to load previously indexed data (from its specific path `chroma_db_dspy_st/`) automatically. If none exists, or if you want to index new data, you can upload/paste text and index it via the sidebar. Use `uv run` to start the Streamlit server:
    ```bash
    uv run streamlit run app.py
    ```

### 7. What Happens Internally (Example: `main.py` execution)?

When you run `uv run python main.py --query 'some query'`:

*   **Load Components (`utils.load_components`)**:
    *   Loads configuration from `.env` via <mcfile name="config.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/config.py"></mcfile>.
    *   Initializes the embedding model (`SentenceTransformer`), reranking model (`CrossEncoder`), and LLM (`dspy.LM` via OpenRouter).
    *   Initializes the ChromaDB client (`chromadb.PersistentClient`) using the CLI path (`CHROMA_DB_PATH`) and **disables telemetry**.
    *   Configures DSPy settings globally with the loaded LLM.
    *   Suppresses `INFO` level logs from `httpx` and `LiteLLM`.
*   **Parse Arguments (`main.py`)**:
    *   Processes command-line arguments (`--file`, `--query`).
    *   Determines the document source: uses the file specified by `--file` or defaults to documents in <mcfile name="data.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/data.py"></mcfile> if `--file` is omitted.
    *   Ensures a query is provided.
*   **Index Data (`utils.index_chroma_data`, `utils.create_bm25_index`)**:
    *   **ChromaDB Indexing**:
        *   Gets or creates the ChromaDB collection specified in `config.py`.
        *   Clears any existing documents from the collection (default behavior).
        *   Generates vector embeddings for the documents using the loaded embedder.
        *   Upserts the documents, embeddings, and IDs into the ChromaDB collection.
    *   **BM25 Indexing**:
        *   Preprocesses the documents for keyword search using functions in <mcfile name="bm25_utils.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/bm25_utils.py"></mcfile> (e.g., tokenization, stopword removal via NLTK).
        *   Builds a BM25 keyword index (`rank_bm25.BM25Okapi`) from the preprocessed documents.
*   **Create Retrievers (`utils.create_retrievers`)**:
    *   Instantiates the custom <mcsymbol name="ChromaRetriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="13" type="class"></mcsymbol> using the ChromaDB collection and embedder.
    *   Instantiates the custom <mcsymbol name="BM25Retriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="35" type="class"></mcsymbol> using the BM25 index and the original document corpus.
    *   Both retrievers are configured with their respective `k` values from `config.py`.
*   **Create RAG Pipeline (`utils.create_rag_pipeline`)**:
    *   Instantiates the main DSPy module, <mcsymbol name="RAGHybridFusedRerank" filename="rag_pipeline.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/rag_pipeline.py" startline="10" type="class"></mcsymbol>, defined in <mcfile name="rag_pipeline.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/rag_pipeline.py"></mcfile>.
    *   This pipeline module integrates:
        *   The vector retriever (<mcsymbol name="ChromaRetriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="13" type="class"></mcsymbol>).
        *   The keyword retriever (<mcsymbol name="BM25Retriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="35" type="class"></mcsymbol>).
        *   The reranking model (`CrossEncoder`).
        *   The LLM (`dspy.LM`).
    *   It's configured with the `rerank_k` value from `config.py`.
*   **Run Query (`main.py`)**:
    *   Calls the forward method of the <mcsymbol name="RAGHybridFusedRerank" filename="rag_pipeline.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/rag_pipeline.py" startline="10" type="class"></mcsymbol> instance with the user's query (`args.query`).
    *   Internally, the pipeline performs:
        1.  Retrieval using both vector and keyword retrievers.
        2.  Fusion and deduplication of results.
        3.  Reranking of fused results using the CrossEncoder.
        4.  Selection of the top `rerank_k` documents as context.
        5.  Generation of the final answer by the LLM using the context.
    *   Prints the generated answer to the console.

*   **Streamlit App (`app.py`)**: Follows a similar flow but uses the separate ChromaDB path (`_st` suffix). It attempts to load existing data first. If new data is provided via the UI, it calls the same utility functions (`index_chroma_data`, `create_bm25_index`, `create_retrievers`, `create_rag_pipeline`) to process and index it before enabling search. Suppresses `httpx` and `LiteLLM` info logs.

### 8. What Happens Internally (Example: `app.py` Streamlit execution)?

When you run `uv run streamlit run app.py`:

*   **Load Components (`utils.load_components`)**:
    *   Loads configuration from `.env` via <mcfile name="config.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/config.py"></mcfile>.
    *   Initializes the embedding model (`SentenceTransformer`), reranking model (`CrossEncoder`), and LLM (`dspy.LM` via OpenRouter).
    *   Initializes the ChromaDB client (`chromadb.PersistentClient`) using the Streamlit path (`CHROMA_DB_PATH_ST`) and **disables telemetry**.
    *   Configures DSPy settings globally with the loaded LLM.
    *   Suppresses `INFO` level logs from `httpx` and `LiteLLM`.
*   **Streamlit UI Initialization (`app.py`)**:
    *   Renders the web interface, including sidebar for data upload and main area for querying.
    *   Allows users to upload files or paste text to index new documents, or loads previously indexed data from the Streamlit ChromaDB path.
*   **Index Data (on user action)**:
    *   **ChromaDB Indexing**:
        *   Gets or creates the ChromaDB collection specified in `config.py` (using the Streamlit path).
        *   Clears any existing documents from the collection (default behavior).
        *   Generates vector embeddings for the documents using the loaded embedder.
        *   Upserts the documents, embeddings, and IDs into the ChromaDB collection.
    *   **BM25 Indexing**:
        *   Preprocesses the documents for keyword search using functions in <mcfile name="bm25_utils.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/bm25_utils.py"></mcfile> (e.g., tokenization, stopword removal via NLTK).
        *   Builds a BM25 keyword index (`rank_bm25.BM25Okapi`) from the preprocessed documents.
*   **Create Retrievers (`utils.create_retrievers`)**:
    *   Instantiates the custom <mcsymbol name="ChromaRetriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="13" type="class"></mcsymbol> using the ChromaDB collection and embedder.
    *   Instantiates the custom <mcsymbol name="BM25Retriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="35" type="class"></mcsymbol> using the BM25 index and the original document corpus.
    *   Both retrievers are configured with their respective `k` values from `config.py`.
*   **Create RAG Pipeline (`utils.create_rag_pipeline`)**:
    *   Instantiates the main DSPy module, <mcsymbol name="RAGHybridFusedRerank" filename="rag_pipeline.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/rag_pipeline.py" startline="10" type="class"></mcsymbol>, defined in <mcfile name="rag_pipeline.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/rag_pipeline.py"></mcfile>.
    *   This pipeline module integrates:
        *   The vector retriever (<mcsymbol name="ChromaRetriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="13" type="class"></mcsymbol>).
        *   The keyword retriever (<mcsymbol name="BM25Retriever" filename="retrievers.py" path="/Users/srini/Ws/dspy-rag-bm25-rerank-llm/src/dspy_rag_app/retrievers.py" startline="35" type="class"></mcsymbol>).
        *   The reranking model (`CrossEncoder`).
        *   The LLM (`dspy.LM`).
    *   It's configured with the `rerank_k` value from `config.py`.
*   **Run Query (on user action)**:
    *   When the user enters a query and clicks 'Search', the pipeline is executed with the query.
    *   Internally, the pipeline performs:
        1.  Retrieval using both vector and keyword retrievers.
        2.  Fusion and deduplication of results.
        3.  Reranking using the CrossEncoder.
        4.  Answer generation using the LLM.
    *   The answer is displayed in the Streamlit UI.

---
### 9. Customization
- Add your own default documents to the `DOCUMENTS` list in `src/dspy_rag_app/data.py`.
- Change retrieval, rerank parameters, model names, or paths via the `.env` file.
- Modify the core pipeline logic in `src/dspy_rag_app/rag_pipeline.py` or the utility functions in `src/dspy_rag_app/utils.py`.

### 10. Troubleshooting
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
