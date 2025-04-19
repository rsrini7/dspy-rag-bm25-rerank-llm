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
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_dspy")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hybrid_rag_docs")

if not OPENROUTER_API_KEY:
    logging.warning("OPENROUTER_API_KEY environment variable not set. LLM calls will fail.")
