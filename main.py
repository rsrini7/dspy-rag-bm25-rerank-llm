# --- Imports ---
import dspy
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from config import *
from data import DOCUMENTS, DOCUMENT_IDS
from bm25_utils import preprocess_for_bm25, ensure_nltk_resources
from retrievers import ChromaRetriever, BM25Retriever
from rag_pipeline import RAGHybridFusedRerank
import logging
logging.basicConfig(level=logging.INFO)

# --- Model and Client Initialization ---

logging.info(f"Initializing Embedder: {EMBEDDER_MODEL}")
embedder = SentenceTransformer(EMBEDDER_MODEL)

logging.info(f"Initializing Reranker: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)

logging.info(f"Initializing ChromaDB Client (Path: {CHROMA_DB_PATH})")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
logging.info(f"Getting/Creating Chroma Collection: {CHROMA_COLLECTION_NAME}")
collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

# --- Data Preprocessing and Indexing ---

# Ensure NLTK resources for BM25
ensure_nltk_resources()

# Index data in Chroma if collection is empty or doesn't match
existing_ids = set(collection.get(include=[])['ids'])
if not existing_ids or existing_ids != set(DOCUMENT_IDS):
    logging.info("Collection is empty or different. Indexing documents in ChromaDB...")
    embeddings = embedder.encode(DOCUMENTS, show_progress_bar=True)
    collection.upsert(
        documents=DOCUMENTS,
        embeddings=embeddings.tolist(),
        ids=DOCUMENT_IDS
    )
    logging.info(f"Indexed {len(DOCUMENTS)} documents in ChromaDB.")
else:
    logging.info("Documents already indexed in ChromaDB.")

# Preprocess documents for BM25
logging.info("Preprocessing documents for BM25...")
tokenized_docs_bm25 = preprocess_for_bm25(DOCUMENTS)
bm25 = BM25Okapi(tokenized_docs_bm25)
logging.info("BM25 index created.")

# --- DSPy Configuration ---

if OPENROUTER_API_KEY:
    openrouter_lm = dspy.LM(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        api_base="https://openrouter.ai/api/v1",
        provider="openrouter",
        max_tokens=500
    )
    dspy.settings.configure(lm=openrouter_lm)
    logging.info(f"DSPy configured with LLM: {LLM_MODEL} via OpenRouter")
else:
    logging.warning("DSPy LM not configured due to missing API key.")
    openrouter_lm = None

# --- Main Execution & Testing ---

if __name__ == "__main__":
    if not dspy.settings.lm:
        logging.error("DSPy Language Model not configured. Exiting.")
    else:
        logging.info("--- Initializing RAG Pipeline ---")
        # Instantiate retrievers
        chroma_retriever = ChromaRetriever(collection, embedder, k=K_EMBEDDING_RETRIEVAL)
        bm25_retriever = BM25Retriever(bm25, DOCUMENTS, k=K_BM25_RETRIEVAL)

        # Instantiate the main RAG module
        rag_pipeline = RAGHybridFusedRerank(
            vector_retriever=chroma_retriever,
            keyword_retriever=bm25_retriever,
            reranker_model=reranker, # Pass the initialized CrossEncoder
            llm=dspy.settings.lm,     # Pass the configured LLM
            rerank_k=K_RERANK
        )

        logging.info("--- Running Test Query ---")
        test_question = "What is DSPy used for and how does it relate to LLMs?"
        logging.info(f"Question: {test_question}")

        # Run the pipeline
        response = rag_pipeline(question=test_question)
        logging.info(f"\nAnswer: {response.answer}")

        logging.info("--- Example with potentially no good context ---")
        test_question_2 = "Tell me about the history of cheese making."
        logging.info(f"Question: {test_question_2}")
        response2 = rag_pipeline(question=test_question_2)
        logging.info(f"\nAnswer: {response2.answer}")