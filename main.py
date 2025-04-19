# Update imports to use the package structure
from dspy_rag_app.config import CHROMA_COLLECTION_NAME
from dspy_rag_app.data import DOCUMENTS, DOCUMENT_IDS
from dspy_rag_app.bm25_utils import ensure_nltk_resources
# Import utility functions
from dspy_rag_app.utils import (
    load_components,
    index_chroma_data, create_bm25_index, create_retrievers, create_rag_pipeline
)
import logging

logging.basicConfig(level=logging.INFO)

# --- Ensure NLTK resources ---
ensure_nltk_resources()

# --- Load Components using Utils ---
embedder, reranker, client, llm = load_components() # streamlit_mode=False (default)

# DSPy settings are initialized within load_components now.

# --- Data Preprocessing and Indexing ---

# Use the utility function to index data in Chroma
collection = index_chroma_data(
    client=client,
    embedder=embedder,
    documents=DOCUMENTS,
    doc_ids=DOCUMENT_IDS,
    collection_name=CHROMA_COLLECTION_NAME,
    clear_existing=True # Ensure main.py always reflects DOCUMENTS
)

# Use the utility function to create the BM25 index
bm25_index = create_bm25_index(DOCUMENTS)


# --- Main Execution & Testing ---

if __name__ == "__main__":
    # Check if LLM was loaded successfully (DSPy initialization happens in load_components)
    if not llm: # Check if llm object exists
        logging.error("DSPy Language Model not configured. Cannot run RAG pipeline. Exiting.")
    elif not collection or not bm25_index:
         logging.error("Index creation failed. Cannot run RAG pipeline. Exiting.")
    else:
        logging.info("--- Initializing RAG Pipeline Components ---")
        # Use utility function to create retrievers
        chroma_retriever, bm25_retriever = create_retrievers(
            collection=collection,
            embedder=embedder,
            bm25_index=bm25_index,
            corpus=DOCUMENTS
        )

        # Use utility function to create the RAG pipeline
        rag_pipeline = create_rag_pipeline(
            vector_retriever=chroma_retriever,
            keyword_retriever=bm25_retriever,
            reranker_model=reranker,
            llm=llm
        )

        # Check if pipeline creation was successful (it returns None if LLM is missing)
        if not rag_pipeline:
             logging.error("RAG pipeline creation failed (likely due to missing LLM). Exiting.")
        else:
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