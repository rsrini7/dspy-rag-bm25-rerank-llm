# DSPy RAG Hybrid Pipeline with BM25, Embedding, and Rerank LLM

## Architecture Diagram

![Hybrid RAG Pipeline Architecture](assets/architecture.png)

## Architecture Flow Diagram

![Hybrid RAG Pipeline Architecture Flow](assets/architecture-flow.png)

## Architecture Flow

1. **Query Input**: The user provides a query.
2. **Hybrid Retrieval**:
   - **Vector Retrieval**: The query is embedded using SentenceTransformer and searched in ChromaDB for similar documents.
   - **Keyword Retrieval**: The query is tokenized (NLTK), and BM25Okapi retrieves relevant documents by keyword match.
3. **Fusion & Deduplication**: Results from both retrieval methods are merged and deduplicated to form a single candidate set.
4. **CrossEncoder Reranker**: The candidate documents are reranked for relevance to the query using a CrossEncoder model.
5. **LLM Generation**: The top reranked context is sent to an LLM (via OpenRouter) to generate a final answer.
6. **Answer Output**: The answer is returned to the user.

## Features
- **Hybrid Retrieval**: Combines dense embedding (vector) search and BM25 keyword search.
- **Fusion & Deduplication**: Merges results from both retrieval strategies.
- **Neural Reranking**: Uses a CrossEncoder model to rerank the fused results for relevance.
- **LLM Generation**: Generates answers using a large language model, conditioned on the top reranked context.

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
The script will automatically download 'punkt' and 'stopwords' if not present.

### 6. Run the Main Script
```bash
python main.py
```

### 7. What Happens Internally?
- **Environment & Models**: Loads config and initializes embedding, reranking, and LLM models.
- **ChromaDB**: Checks and (re)indexes documents into ChromaDB if needed.
- **BM25**: Preprocesses documents and builds a BM25 keyword index.
- **Retrievers**: Sets up both Chroma and BM25 retrievers for DSPy.
- **RAG Pipeline**: Combines both retrievers, fuses and deduplicates results, reranks using CrossEncoder, and generates an answer with the LLM.
- **Test Queries**: Runs two example queries to demonstrate the pipeline.

### 8. Customization
- Add your own documents to the `DOCUMENTS` list in `main.py`.
- Change retrieval or rerank parameters via the `.env` file.

### 9. Troubleshooting
- Ensure your `.env` is set up and API keys are valid.
- If you see NLTK errors, delete `.venv/nltk_data` and rerun.
- For LLM errors, check your OpenRouter API key and model name.

---

## Project Structure
- `main.py` - Main pipeline script
- `pyproject.toml` - Python dependencies (used by uv)
- `.env.example` - Example environment file
- `README.md` - This guide

---

## References
- [DSPy](https://github.com/stanfordnlp/dspy)
- [ChromaDB](https://www.trychroma.com/)
- [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenRouter](https://openrouter.ai/)
- [uv](https://github.com/astral-sh/uv)