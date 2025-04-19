import logging
import chromadb
import dspy
from sentence_transformers import SentenceTransformer, CrossEncoder
# Option 1: Relative import
from .config import (
    EMBEDDER_MODEL, RERANKER_MODEL, LLM_MODEL,
    CHROMA_DB_PATH, OPENROUTER_API_KEY
)
# Option 2: Absolute import
# from dspy_rag_app.config import ( ... )

logging.basicConfig(level=logging.INFO)

def load_components(streamlit_mode: bool = False):
    """
    Loads the embedder, reranker, Chroma client, and LLM.

    Args:
        streamlit_mode (bool): If True, appends '_st' to the ChromaDB path
                               to avoid conflicts when running main.py and app.py.

    Returns:
        tuple: (embedder, reranker, client, llm)
    """
    logging.info("--- Loading Components ---")

    # Load Embedder
    logging.info(f"Initializing Embedder: {EMBEDDER_MODEL}")
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    # Load Reranker
    logging.info(f"Initializing Reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)

    # Setup ChromaDB Client
    db_path = CHROMA_DB_PATH + ("_st" if streamlit_mode else "")
    logging.info(f"Initializing ChromaDB Client (Path: {db_path})")
    client = chromadb.PersistentClient(path=db_path)

    # Setup LLM (DSPy LM)
    llm = None
    if OPENROUTER_API_KEY:
        try:
            logging.info(f"Attempting to configure LLM: {LLM_MODEL} via OpenRouter")
            llm = dspy.LM(
                model=LLM_MODEL,
                api_key=OPENROUTER_API_KEY,
                api_base="https://openrouter.ai/api/v1",
                provider="openrouter",
                max_tokens=500 # Consider making this configurable if needed
            )
            logging.info("LLM configured successfully.")
        except Exception as e:
            logging.error(f"Failed to configure DSPy LM: {e}", exc_info=True)
            llm = None # Ensure llm is None if setup fails
    else:
        logging.warning("OpenRouter API Key not found. LLM features will be disabled.")

    # --- Initialize DSPy Settings HERE ---
    # Configure DSPy settings immediately after LLM is loaded (or determined to be None)
    # This ensures it happens only once when components are loaded.
    initialize_dspy(llm)
    # ------------------------------------

    logging.info("--- Components Loaded ---")
    return embedder, reranker, client, llm

def initialize_dspy(llm):
    """Configures DSPy settings with the loaded LLM."""
    if llm:
        # Check if settings are already configured to avoid the error,
        # although calling this from load_components should prevent redundant calls.
        if dspy.settings.lm is None or dspy.settings.lm != llm:
             dspy.settings.configure(lm=llm)
             logging.info(f"DSPy globally configured with LLM: {llm.kwargs.get('model', 'Unknown')}")
        else:
             logging.info(f"DSPy already configured with the correct LLM.")
        return True
    else:
        logging.warning("DSPy LM not provided or failed to load. DSPy global settings not configured.")
        # Ensure settings are cleared if no LLM
        if dspy.settings.lm:
             dspy.settings.configure(lm=None)
             logging.info("Cleared existing DSPy global LLM configuration.")
        return False