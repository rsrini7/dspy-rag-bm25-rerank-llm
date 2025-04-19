import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

# --- Configuration ---

# Load environment variables from .env
load_dotenv()

# Environment Variables and Constants (from .env)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-mpnet-base-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
K_EMBEDDING_RETRIEVAL = int(os.getenv("K_EMBEDDING_RETRIEVAL", 10))
K_BM25_RETRIEVAL = int(os.getenv("K_BM25_RETRIEVAL", 10))
K_RERANK = int(os.getenv("K_RERANK", 3))
# --- ChromaDB Configuration ---
CHROMA_DB_PATH = "./chroma_db_dspy" # Default path for CLI
CHROMA_DB_PATH_ST = "./chroma_db_dspy_st" # Separate path for Streamlit
CHROMA_COLLECTION_NAME = "hybrid_rag_docs" # Collection name for both CLI and Streamlit (will be suffixed in app.py)

# --- Retrieval Configuration ---
K_EMBEDDING_RETRIEVAL = 5 # Number of documents to retrieve using embeddings

if not OPENROUTER_API_KEY:
    logging.warning("OPENROUTER_API_KEY environment variable not set. LLM calls will fail.")
