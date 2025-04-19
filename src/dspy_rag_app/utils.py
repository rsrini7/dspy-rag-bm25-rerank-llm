import logging
import chromadb
from chromadb.config import Settings # Import Settings
import dspy
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
# Relative import for project components
from .config import config
from .bm25_utils import BM25Processor
from .retrievers import ChromaRetriever, BM25Retriever
from .rag_pipeline import RAGHybridFusedRerank

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
    logging.info(f"Initializing Embedder: {config.EMBEDDER_MODEL}")
    embedder = SentenceTransformer(config.EMBEDDER_MODEL)

    # Load Reranker
    logging.info(f"Initializing Reranker: {config.RERANKER_MODEL}")
    reranker = CrossEncoder(config.RERANKER_MODEL)

    # Setup ChromaDB Client
    # Determine path based on mode (using CHROMA_DB_PATH_ST for streamlit)
    db_path = config.CHROMA_DB_PATH_ST if streamlit_mode else config.CHROMA_DB_PATH
    logging.info(f"Initializing ChromaDB Client (Path: {db_path})")
    # Initialize client with telemetry disabled using Settings
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False) # Disable telemetry
    )

    # Setup LLM (DSPy LM)
    llm = None
    if config.OPENROUTER_API_KEY:
        try:
            logging.info(f"Attempting to configure LLM: {config.LLM_MODEL} via OpenRouter")
            llm = dspy.LM(
                model=config.LLM_MODEL,
                api_key=config.OPENROUTER_API_KEY,
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

# --- NEW UTILITY FUNCTIONS ---

def index_chroma_data(client: chromadb.PersistentClient,
                      embedder: SentenceTransformer,
                      documents: list[str],
                      doc_ids: list[str],
                      collection_name: str,
                      clear_existing: bool = True):
    """Gets/Creates a Chroma collection, optionally clears it, and indexes data."""
    logging.info(f"Getting/Creating Chroma Collection: {collection_name}")
    collection = client.get_or_create_collection(collection_name)

    if clear_existing:
        existing_ids = collection.get(include=[])['ids']
        if existing_ids:
            logging.info(f"Clearing {len(existing_ids)} existing documents from collection '{collection_name}'.")
            collection.delete(ids=existing_ids)
        else:
            logging.info(f"Collection '{collection_name}' is already empty or new.")

    logging.info(f"Encoding and indexing {len(documents)} documents in ChromaDB collection '{collection_name}'...")
    embeddings = embedder.encode(documents, show_progress_bar=False) # Progress bar off for utils
    collection.upsert(
        documents=documents,
        embeddings=embeddings.tolist(),
        ids=doc_ids
    )
    logging.info(f"Successfully indexed {len(documents)} documents in ChromaDB.")
    return collection

def create_bm25_index(documents: list[str]) -> BM25Processor:
    """Preprocesses documents and creates a BM25Processor instance."""
    logging.info(f"Preprocessing {len(documents)} documents for BM25...")
    bm25_processor = BM25Processor(documents)
    logging.info("BM25Processor created.")
    return bm25_processor

def create_retrievers(collection: chromadb.Collection,
                      embedder: SentenceTransformer,
                      bm25_processor: BM25Processor,
                      corpus: list[str]) -> tuple[ChromaRetriever, BM25Retriever]:
    """Instantiates and returns Chroma and BM25 retrievers using config K values."""
    logging.info("Initializing retrievers...")
    chroma_retriever = ChromaRetriever(
        chroma_collection=collection,
        embed_model=embedder,
        k=config.K_EMBEDDING_RETRIEVAL # Use config value
    )
    bm25_retriever = BM25Retriever(
        bm25_processor=bm25_processor,
        corpus=corpus,
        k=config.K_BM25_RETRIEVAL # Use config value
    )
    logging.info("Retrievers initialized.")
    return chroma_retriever, bm25_retriever

def create_rag_pipeline(vector_retriever: ChromaRetriever,
                        keyword_retriever: BM25Retriever,
                        reranker_model: CrossEncoder,
                        llm: dspy.LM) -> RAGHybridFusedRerank | None:
    """Instantiates the RAG pipeline if LLM is available, using config K_RERANK."""
    if not llm:
        logging.warning("LLM not available, cannot create RAG pipeline.")
        return None

    logging.info("Initializing RAG pipeline...")
    rag_pipeline = RAGHybridFusedRerank(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        reranker_model=reranker_model,
        llm=llm,
        rerank_k=config.K_RERANK # Use config value
    )
    logging.info("RAG pipeline initialized.")
    return rag_pipeline