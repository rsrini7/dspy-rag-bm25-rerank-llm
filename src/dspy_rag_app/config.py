import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
# Set httpx logger level to WARNING *before* importing modules that might use it
logging.getLogger("httpx").setLevel(logging.WARNING)
# Set LiteLLM logger level to WARNING (can stay here or move up too)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Load environment variables from .env
load_dotenv()

# Environment Variables and Constants (from .env)
from typing import Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    OPENROUTER_API_KEY: str
    EMBEDDER_MODEL: str
    RERANKER_MODEL: str
    LLM_MODEL: str
    K_EMBEDDING_RETRIEVAL: int
    K_BM25_RETRIEVAL: int
    K_RERANK: int
    CHROMA_DB_PATH: str
    CHROMA_DB_PATH_ST: str
    CHROMA_COLLECTION_NAME: str

# Load config from environment variables
config = Config(**{
    key: os.getenv(key)
    for key in Config.__annotations__.keys()
})

# --- Retrieval Configuration ---
config.K_EMBEDDING_RETRIEVAL = 5 # Number of documents to retrieve using embeddings

if not config.OPENROUTER_API_KEY:
    logging.warning("OPENROUTER_API_KEY environment variable not set. LLM calls will fail.")
