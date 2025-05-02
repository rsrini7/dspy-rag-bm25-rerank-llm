import logging
import chromadb
from chromadb.config import Settings # Import Settings
import dspy
from sentence_transformers import SentenceTransformer, CrossEncoder
# Relative import for project components
from dspy_rag_app.config import config
from dspy_rag_app.bm25_utils import BM25Processor
from dspy_rag_app.retrievers import ChromaRetriever, BM25Retriever
from dspy_rag_app.rag_pipeline import RAGHybridFusedRerank
import nltk
import os

def load_components(db_path: str = "chroma_db_default"):
    """
    Loads the embedder, reranker, Chroma client, and LLM.

    Args:
        db_path (str): Path to the ChromaDB directory.
    Returns:
        tuple: (embedder, reranker, client, llm)
    """

    # Load Embedder
    logging.info(f"Initializing Embedder: {config.EMBEDDER_MODEL}")
    embedder = SentenceTransformer(config.EMBEDDER_MODEL)

    # Load Reranker
    logging.info(f"Initializing Reranker: {config.RERANKER_MODEL}")
    reranker = CrossEncoder(config.RERANKER_MODEL)

    # Setup ChromaDB Client
    logging.info(f"Initializing ChromaDB Client (Path: {db_path})")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
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
                max_tokens=500
            )
            logging.info("LLM configured successfully.")
        except Exception as e:
            logging.error(f"Failed to configure DSPy LM: {e}", exc_info=True)
            llm = None
    else:
        logging.warning("OpenRouter API Key not found. LLM features will be disabled.")

    initialize_dspy(llm)
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
                        llm: dspy.LM) -> RAGHybridFusedRerank:
    """Instantiates the RAG pipeline, using config K_RERANK. Allows llm to be None for rerank-only mode."""
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

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    # Get absolute path to .venv/nltk_data relative to this file
    nltk_data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '.venv', 'nltk_data')
    )
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)