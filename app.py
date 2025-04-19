import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import logging
import io # For handling file uploads

# --- Import from project files (Update these) ---
from dspy_rag_app.config import ( 
    K_EMBEDDING_RETRIEVAL, K_BM25_RETRIEVAL, K_RERANK, CHROMA_COLLECTION_NAME
)
from dspy_rag_app.bm25_utils import ensure_nltk_resources, preprocess_for_bm25 # Import from bm25_utils.py
from dspy_rag_app.retrievers import ChromaRetriever, BM25Retriever # Import from retrievers.py
from dspy_rag_app.rag_pipeline import RAGHybridFusedRerank # Import from rag_pipeline.py
from dspy_rag_app.utils import load_components, initialize_dspy # Import the new utility functions

logging.basicConfig(level=logging.INFO) # Configure logging

# --- Ensure NLTK resources are available ---
# Use the imported function
ensure_nltk_resources()

# --- Cached Resource Loading using Utils ---

@st.cache_resource # Use cache_resource for expensive loads like models/clients
def cached_load_components():
    """Cached function to load components using the utility function for Streamlit."""
    logging.info("Executing cached_load_components...")
    # Calls the utility function with streamlit_mode=True
    # Use the imported function
    embedder, reranker, client, llm = load_components(streamlit_mode=True)
    # Display LLM status in the UI after loading (can be helpful)
    if not llm:
         st.warning("LLM configuration failed or API key missing. Search will be limited.")
    return embedder, reranker, client, llm

@st.cache_data # Cache the preprocessing function result based on input docs
def get_tokenized_docs(docs):
    """Preprocesses documents for BM25 using the utility function."""
    logging.info(f"Preprocessing {len(docs)} documents for BM25...")
    # Use the imported function
    tokenized = preprocess_for_bm25(docs)
    logging.info("BM25 preprocessing done.")
    return tokenized

# --- Attempt to Load Existing Data ---
def try_load_existing_data(client, embedder, reranker, llm):
    """Checks ChromaDB for existing data and loads it into session state if found."""
    try:
        collection_name = CHROMA_COLLECTION_NAME + "_st"
        logging.info(f"Checking for existing data in collection: {collection_name}")
        # Attempt to get the collection; this might raise an exception if it doesn't exist
        collection = client.get_collection(name=collection_name)

        if collection.count() > 0:
            logging.info(f"Found {collection.count()} documents in existing collection. Loading...")
            with st.spinner("Loading existing indexed data..."):
                # Retrieve documents and IDs
                results = collection.get(include=['documents']) # Only need documents for BM25
                st.session_state.documents = results['documents']
                st.session_state.doc_ids = results['ids']
                st.session_state.collection = collection # Store collection handle

                # Rebuild BM25 index
                tokenized_docs = get_tokenized_docs(tuple(st.session_state.documents))
                st.session_state.bm25_index = BM25Okapi(tokenized_docs)
                logging.info("BM25 index rebuilt from loaded data.")

                # Instantiate Retrievers
                chroma_retriever = ChromaRetriever(
                    chroma_collection=st.session_state.collection,
                    embed_model=embedder,
                    k=K_EMBEDDING_RETRIEVAL
                )
                bm25_retriever = BM25Retriever(
                    bm25_index=st.session_state.bm25_index,
                    corpus=st.session_state.documents,
                    k=K_BM25_RETRIEVAL
                )

                # Instantiate RAG pipeline if LLM is available
                if llm:
                    st.session_state.rag_pipeline = RAGHybridFusedRerank(
                        vector_retriever=chroma_retriever,
                        keyword_retriever=bm25_retriever,
                        reranker_model=reranker,
                        llm=llm,
                        rerank_k=K_RERANK
                    )
                    logging.info("RAG pipeline initialized with loaded data.")
                else:
                    st.session_state.rag_pipeline = None
                    logging.warning("LLM not available, RAG pipeline not initialized.")

                st.session_state.data_indexed = True
                logging.info("Existing data loaded successfully.")
                st.success(f"Loaded {len(st.session_state.documents)} documents from existing database.") # User feedback
                return True # Indicate success
        else:
            logging.info("Existing collection found but is empty.")
            return False
    except Exception as e:
        # Handle cases where the collection doesn't exist or other errors occur
        logging.info(f"Could not load existing data (may be first run or an error): {e}")
        return False

# --- Streamlit App ---

st.title("📄 RAG Pipeline Interface")

# --- Initialization ---
embedder, reranker, client, llm = cached_load_components()

# --- Session State Initialization ---
# Initialize keys FIRST to ensure they exist
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'doc_ids' not in st.session_state:
    st.session_state.doc_ids = []
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'bm25_index' not in st.session_state:
    st.session_state.bm25_index = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'data_indexed' not in st.session_state:
    st.session_state.data_indexed = False # Default to False

# --- Attempt Auto-Load ---
# Try loading ONLY if data hasn't been marked as indexed in this specific session run yet
if not st.session_state.data_indexed:
    try_load_existing_data(client, embedder, reranker, llm)

# --- Sidebar for Data Management ---
with st.sidebar:
    st.header("Data Loading & Indexing")

    data_source = st.radio("Select Data Source", ("Upload Files", "Paste Text"))

    uploaded_files = None
    pasted_text = ""

    if data_source == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload text files (.txt)", type=["txt"], accept_multiple_files=True
        )
    else:
        pasted_text = st.text_area("Paste text (one document per line)", height=200)

    # The "Load and Index Data" button now primarily handles *new* uploads or overwrites
    if st.button("Load and Index New Data"): # Changed button text slightly for clarity
        docs_to_index = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                docs_to_index.extend(stringio.read().splitlines()) # Simple split by line
        elif pasted_text:
            docs_to_index = pasted_text.strip().split('\n')

        if docs_to_index:
            with st.spinner("Processing and indexing new data..."): # Updated spinner text
                try:
                    st.session_state.documents = [doc for doc in docs_to_index if doc.strip()]
                    st.session_state.doc_ids = [f"doc_{i}" for i in range(len(st.session_state.documents))]

                    # Get/Create Chroma Collection using config name + _st
                    collection_name = CHROMA_COLLECTION_NAME + "_st"
                    st.session_state.collection = client.get_or_create_collection(collection_name)
                    logging.info(f"Using Chroma Collection: {collection_name}")

                    # Clear existing data before adding new data
                    logging.info(f"Clearing existing data from collection '{collection_name}' before indexing new data...")
                    existing_ids = st.session_state.collection.get(include=[])['ids']
                    if existing_ids:
                        st.session_state.collection.delete(ids=existing_ids)

                    # Index in Chroma
                    logging.info("Encoding and indexing in ChromaDB...")
                    embeddings = embedder.encode(st.session_state.documents, show_progress_bar=False)
                    st.session_state.collection.upsert(
                        documents=st.session_state.documents,
                        embeddings=embeddings.tolist(),
                        ids=st.session_state.doc_ids
                    )
                    logging.info(f"Indexed {len(st.session_state.documents)} new documents in ChromaDB.")

                    # Create BM25 Index
                    tokenized_docs = get_tokenized_docs(tuple(st.session_state.documents))
                    st.session_state.bm25_index = BM25Okapi(tokenized_docs)
                    logging.info("BM25 index created for new data.")

                    # Instantiate Retrievers
                    chroma_retriever = ChromaRetriever(
                        chroma_collection=st.session_state.collection,
                        embed_model=embedder,
                        k=K_EMBEDDING_RETRIEVAL
                    )
                    bm25_retriever = BM25Retriever(
                        bm25_index=st.session_state.bm25_index,
                        corpus=st.session_state.documents,
                        k=K_BM25_RETRIEVAL
                    )

                    # Initialize RAG pipeline if LLM available
                    if llm:
                        st.session_state.rag_pipeline = RAGHybridFusedRerank(
                            vector_retriever=chroma_retriever,
                            keyword_retriever=bm25_retriever,
                            reranker_model=reranker,
                            llm=llm,
                            rerank_k=K_RERANK
                        )
                        st.session_state.data_indexed = True
                        st.success(f"Successfully indexed {len(st.session_state.documents)} new documents! RAG Pipeline is ready.")
                    else:
                        st.session_state.rag_pipeline = None
                        st.session_state.data_indexed = True
                        st.info("New data indexing complete. Search available but final answer generation requires LLM.")

                except Exception as e:
                    st.error(f"An error occurred during indexing: {e}")
                    logging.error(f"Indexing error: {e}", exc_info=True)
                    st.session_state.data_indexed = False
                    st.session_state.rag_pipeline = None
        else:
            st.warning("No documents provided to index.")

    # --- Sidebar Status Update ---
    # This part remains the same, reflecting the current state of data_indexed
    if st.session_state.data_indexed:
        st.sidebar.success(f"{len(st.session_state.documents)} documents indexed.")
        if not st.session_state.rag_pipeline:
             st.sidebar.info("LLM not configured. Search limited.")
    else:
        st.sidebar.info("Load data using the options above or data might load automatically if previously indexed.") # Updated info text

# --- Main Search Area ---
st.header("🔍 Search")

if not st.session_state.data_indexed:
    st.warning("Please load and index data using the sidebar, or wait for automatic loading if data exists.") # Updated warning
elif not st.session_state.rag_pipeline:
     st.warning("RAG Pipeline not initialized (LLM configuration failed or missing). Cannot generate answers.")
else:
    question = st.text_input("Enter your question:")

    if st.button("Search"):
        if question:
            with st.spinner("Searching..."):
                try:
                    # Use the stored RAG pipeline instance
                    response = st.session_state.rag_pipeline(question=question)
                    answer = response.answer

                    st.subheader("Answer:")
                    st.markdown(answer)

                    if hasattr(response, 'context') and response.context:
                         with st.expander("Show Final Context Used for Answer"):
                             st.write(response.context)

                except Exception as e:
                    st.error(f"An error occurred during search: {e}")
                    logging.error(f"Search error: {e}", exc_info=True)
        else:
            st.warning("Please enter a question.")