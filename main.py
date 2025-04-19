# --- Imports ---
from rank_bm25 import BM25Okapi
# Update imports to use the package structure
from dspy_rag_app.config import (
    CHROMA_COLLECTION_NAME, K_EMBEDDING_RETRIEVAL, K_BM25_RETRIEVAL, K_RERANK
)
from dspy_rag_app.data import DOCUMENTS, DOCUMENT_IDS
from dspy_rag_app.bm25_utils import preprocess_for_bm25, ensure_nltk_resources
from dspy_rag_app.retrievers import ChromaRetriever, BM25Retriever
from dspy_rag_app.rag_pipeline import RAGHybridFusedRerank
from dspy_rag_app.utils import load_components, initialize_dspy
import logging

logging.basicConfig(level=logging.INFO)

# --- Ensure NLTK resources ---
# Use the imported function
ensure_nltk_resources()

# --- Load Components using Utils ---
# Use the imported function
embedder, reranker, client, llm = load_components()

# Initialize DSPy settings globally
# Use the imported function
dspy_initialized = initialize_dspy(llm)

# --- Data Preprocessing and Indexing (Specific to main.py) ---

# Get/Create Chroma Collection
logging.info(f"Getting/Creating Chroma Collection: {CHROMA_COLLECTION_NAME}")
# Use the client loaded via load_components
collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

# Index data in Chroma if collection is empty or doesn't match DOCUMENTS
existing_ids = set(collection.get(include=[])['ids'])
# Convert DOCUMENT_IDS to set for comparison
target_ids = set(DOCUMENT_IDS)

if not existing_ids or existing_ids != target_ids:
    logging.info("Collection is empty or different. Indexing documents in ChromaDB...")
    # Clear if necessary before upserting
    if existing_ids:
        logging.info(f"Clearing {len(existing_ids)} existing documents from collection.")
        collection.delete(ids=list(existing_ids))

    # Use the embedder loaded via load_components
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


# --- Main Execution & Testing ---

if __name__ == "__main__":
    # Check if DSPy was initialized successfully (meaning LLM was loaded)
    if not dspy_initialized:
        logging.error("DSPy Language Model not configured. Exiting.")
    else:
        logging.info("--- Initializing RAG Pipeline ---")
        # Instantiate retrievers using loaded components and config K values
        chroma_retriever = ChromaRetriever(collection, embedder, k=K_EMBEDDING_RETRIEVAL)
        bm25_retriever = BM25Retriever(bm25, DOCUMENTS, k=K_BM25_RETRIEVAL)

        # Instantiate the main RAG module using loaded components
        rag_pipeline = RAGHybridFusedRerank(
            vector_retriever=chroma_retriever,
            keyword_retriever=bm25_retriever,
            reranker_model=reranker, # Pass the loaded reranker
            llm=llm,                 # Pass the loaded llm instance
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