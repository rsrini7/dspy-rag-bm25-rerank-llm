import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

# --- Configuration ---

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

# Expose configuration values
CHROMA_COLLECTION_NAME = config.CHROMA_COLLECTION_NAME
CHROMA_DB_PATH = config.CHROMA_DB_PATH
CHROMA_DB_PATH_ST = config.CHROMA_DB_PATH_ST
OPENROUTER_API_KEY = config.OPENROUTER_API_KEY
EMBEDDER_MODEL = config.EMBEDDER_MODEL
RERANKER_MODEL = config.RERANKER_MODEL
LLM_MODEL = config.LLM_MODEL
K_EMBEDDING_RETRIEVAL = config.K_EMBEDDING_RETRIEVAL
K_BM25_RETRIEVAL = config.K_BM25_RETRIEVAL
K_RERANK = config.K_RERANK

# --- Retrieval Configuration ---
K_EMBEDDING_RETRIEVAL = 5 # Number of documents to retrieve using embeddings

if not config.OPENROUTER_API_KEY:
    logging.warning("OPENROUTER_API_KEY environment variable not set. LLM calls will fail.")
