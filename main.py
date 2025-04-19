# Update imports to use the package structure
from dspy_rag_app.config import CHROMA_COLLECTION_NAME
from dspy_rag_app.data import DOCUMENTS as DEFAULT_DOCUMENTS # Rename default import
from dspy_rag_app.bm25_utils import ensure_nltk_resources
# Import utility functions
from dspy_rag_app.utils import (
    load_components,
    index_chroma_data, create_bm25_index, create_retrievers, create_rag_pipeline
)
import logging
import argparse # Import argparse
import os # Import os for path validation

logging.basicConfig(level=logging.INFO)

# --- Function to load documents from file ---
def load_documents_from_file(filepath):
    """Reads a text file and returns a list of non-empty lines as documents."""
    if not os.path.exists(filepath):
        logging.error(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            docs = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(docs)} documents from {filepath}")
        return docs
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return None

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run DSPy RAG pipeline with optional file input and query.")
parser.add_argument(
    "-f", "--file",
    type=str,
    help="Path to a .txt file containing documents (one per line)."
)
parser.add_argument(
    "-q", "--query",
    type=str,
    help="Query string to run against the indexed documents."
)
args = parser.parse_args()

# --- Ensure NLTK resources ---
ensure_nltk_resources()

# --- Load Components using Utils ---
embedder, reranker, client, llm = load_components() # streamlit_mode=False (default)

# --- Determine Documents and IDs ---
documents_to_index = DEFAULT_DOCUMENTS
if args.file:
    loaded_docs = load_documents_from_file(args.file)
    if loaded_docs:
        documents_to_index = loaded_docs
    else:
        logging.warning("Failed to load documents from file, using default documents.")
        # Keep documents_to_index as DEFAULT_DOCUMENTS

# Generate IDs based on the final list of documents
document_ids = [f"doc_{i}" for i in range(len(documents_to_index))]

# --- Data Preprocessing and Indexing ---

# Use the utility function to index data in Chroma
# Use the determined documents and IDs
collection = index_chroma_data(
    client=client,
    embedder=embedder,
    documents=documents_to_index, # Use potentially loaded documents
    doc_ids=document_ids,         # Use generated IDs
    collection_name=CHROMA_COLLECTION_NAME,
    clear_existing=True # Clear collection before indexing new/default data
)

# Use the utility function to create the BM25 index
# Use the determined documents
bm25_index = create_bm25_index(documents_to_index)


# --- Main Execution & Testing ---

if __name__ == "__main__":
    # Check if LLM was loaded successfully
    if not llm:
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
            corpus=documents_to_index # Use the indexed documents for the corpus
        )

        # Use utility function to create the RAG pipeline
        rag_pipeline = create_rag_pipeline(
            vector_retriever=chroma_retriever,
            keyword_retriever=bm25_retriever,
            reranker_model=reranker,
            llm=llm
        )

        # Check if pipeline creation was successful
        if not rag_pipeline:
             logging.error("RAG pipeline creation failed (likely due to missing LLM). Exiting.")
        # --- Execute Query ---
        elif args.query:
            # If a query was provided via CLI
            logging.info(f"--- Running Provided Query ---")
            logging.info(f"Question: {args.query}")
            response = rag_pipeline(question=args.query)
            logging.info(f"\nAnswer: {response.answer}")
        else:
            # Otherwise, run default test questions
            logging.info("--- Running Default Test Queries ---")
            test_question_1 = "What is DSPy used for and how does it relate to LLMs?"
            logging.info(f"Question 1: {test_question_1}")
            response1 = rag_pipeline(question=test_question_1)
            logging.info(f"\nAnswer 1: {response1.answer}")

            logging.info("\n--- Example with potentially no good context ---")
            test_question_2 = "Tell me about the history of cheese making."
            logging.info(f"Question 2: {test_question_2}")
            response2 = rag_pipeline(question=test_question_2)
            logging.info(f"\nAnswer 2: {response2.answer}")