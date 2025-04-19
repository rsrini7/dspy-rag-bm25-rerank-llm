import logging
import argparse # Import argparse
import os # Import os for path validation

# Configure logging ASAP
logging.basicConfig(level=logging.INFO)
# Set httpx logger level to WARNING *before* importing modules that might use it
logging.getLogger("httpx").setLevel(logging.WARNING)
# Set LiteLLM logger level to WARNING (can stay here or move up too)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# --- Now import project modules ---
# Update imports to use the package structure
from dspy_rag_app.config import CHROMA_COLLECTION_NAME
from dspy_rag_app.data import DOCUMENTS as DEFAULT_DOCUMENTS # Rename default import
from dspy_rag_app.bm25_utils import ensure_nltk_resources
# Import utility functions
from dspy_rag_app.utils import (
    load_components,
    index_chroma_data, create_bm25_index, create_retrievers, create_rag_pipeline
)

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

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Check argument combinations FIRST ---

    # Case 1: No arguments provided
    if not args.file and not args.query:
        print("\nUsage Error: Please provide a query (--query), or both a file (--file) and a query (--query).")
        print("\nOptions:")
        print("  -f, --file FILE_PATH   : Path to a .txt file containing documents (one per line).")
        print("                           If provided, these documents will be indexed.")
        print("                           Requires --query to be specified as well.")
        print("  -q, --query QUERY_TEXT : Query string to run against the indexed documents.")
        print("                           Can be used alone (with default documents) or with --file.")
        print("\nExamples:")
        print("  # Use default documents and run a specific query:")
        print("  uv run python main.py --query 'What is DSPy?'")
        print("\n  # Index 'my_docs.txt' and run a specific query:")
        print("  uv run python main.py --file path/to/my_docs.txt --query 'Summarize the file.'")
        print("\nNote: Running without arguments shows this help message.")
        exit(1) # Exit if no arguments are given

    # Case 2: --file provided, but --query is missing
    elif args.file and not args.query:
        print("\nUsage Error: When providing a --file, you must also provide a --query.")
        print("\nExample:")
        print("  uv run python main.py --file path/to/my_docs.txt --query 'Summarize the file.'")
        exit(1) # Exit because --query is mandatory with --file

    # --- Proceed only if arguments are valid (either --query alone, or --file and --query together) ---

    # --- Ensure NLTK resources ---
    ensure_nltk_resources()

    # --- Load Components using Utils ---
    logging.info("--- Loading Components ---")
    embedder, reranker, client, llm = load_components() # streamlit_mode=False (default)
    if not llm:
        logging.error("DSPy Language Model not configured. Cannot run RAG pipeline. Exiting.")
        exit(1) # Exit if LLM failed

    # --- Determine Documents and IDs ---
    documents_to_index = DEFAULT_DOCUMENTS
    if args.file:
        logging.info(f"Attempting to load documents from: {args.file}")
        loaded_docs = load_documents_from_file(args.file)
        if loaded_docs:
            documents_to_index = loaded_docs
        else:
            # Error logged in load_documents_from_file, exit if file load failed
            logging.error(f"Exiting due to failure loading file: {args.file}")
            exit(1)
    else:
        logging.info("No file provided, using default documents.")

    # Generate IDs based on the final list of documents
    document_ids = [f"doc_{i}" for i in range(len(documents_to_index))]

    # --- Data Preprocessing and Indexing ---
    logging.info("--- Starting Data Indexing ---")
    # Use the utility function to index data in Chroma
    collection = index_chroma_data(
        client=client,
        embedder=embedder,
        documents=documents_to_index,
        doc_ids=document_ids,
        collection_name=CHROMA_COLLECTION_NAME,
        clear_existing=True
    )

    # Use the utility function to create the BM25 index
    bm25_index = create_bm25_index(documents_to_index)

    if not collection or not bm25_index:
         logging.error("Index creation failed. Cannot run RAG pipeline. Exiting.")
         exit(1) # Exit if indexing failed

    # --- Initialize and Run Pipeline ---
    logging.info("--- Initializing RAG Pipeline Components ---")
    # Use utility function to create retrievers
    chroma_retriever, bm25_retriever = create_retrievers(
        collection=collection,
        embedder=embedder,
        bm25_index=bm25_index,
        corpus=documents_to_index
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
         exit(1) # Exit if pipeline failed

    # --- Execute Query ---
    # At this point, args.query is guaranteed to be non-empty because of the checks above.
    logging.info(f"--- Running Provided Query ---")
    logging.info(f"Question: {args.query}")
    response = rag_pipeline(question=args.query)
    logging.info(f"\nAnswer: {response.answer}")

    # The 'else' block for default queries is removed as it's no longer reachable
    # due to the mandatory --query check when --file is present.