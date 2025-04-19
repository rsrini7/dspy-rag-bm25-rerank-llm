import streamlit as st
import io # For handling file uploads
import logging # Import logging

# --- Import from project files ---
from dspy_rag_app.config import config
from dspy_rag_app.bm25_utils import ensure_nltk_resources
# Import utility functions
from dspy_rag_app.utils import (
    load_components,
    index_chroma_data, create_bm25_index, create_retrievers, create_rag_pipeline
)
from rank_bm25 import BM25Okapi # Keep for cached function type hint

# --- Ensure NLTK resources are available ---
ensure_nltk_resources()

# --- Cached Resource Loading using Utils ---
@st.cache_resource
def cached_load_components():
    """Cached function to load components using the utility function for Streamlit."""
    logging.info("Executing cached_load_components...")
    embedder, reranker, client, llm = load_components(streamlit_mode=True)
    if not llm:
         st.warning("LLM configuration failed or API key missing. Search will be limited.")
    return embedder, reranker, client, llm

# Keep caching for BM25 index creation as preprocessing might be slow
@st.cache_data
def get_or_create_bm25_index_cached(docs_tuple: tuple[str]) -> BM25Okapi:
    """Cached function to preprocess documents and create BM25 index."""
    # Convert tuple back to list for processing
    documents = list(docs_tuple)
    # Use the utility function internally, but cache the result
    # Note: This uses the non-cached preprocess_for_bm25 inside create_bm25_index
    # If preprocess_for_bm25 itself is very slow, it could also be cached separately.
    return create_bm25_index(documents)


# --- Attempt to Load Existing Data ---
def try_load_existing_data(client, embedder, reranker, llm):
    """Checks ChromaDB for existing data and loads it into session state if found."""
    try:
        collection_name = config.CHROMA_COLLECTION_NAME + "_st"
        logging.info(f"Checking for existing data in collection: {collection_name}")
        collection = client.get_collection(name=collection_name) # Use get_collection

        if collection.count() > 0:
            logging.info(f"Found {collection.count()} documents in existing collection. Loading...")
            with st.spinner("Loading existing indexed data..."):
                # Retrieve documents and IDs
                results = collection.get(include=['documents'])
                st.session_state.documents = results['documents']
                st.session_state.doc_ids = results['ids']
                st.session_state.collection = collection # Store collection handle

                # Rebuild BM25 index using the cached function
                st.session_state.bm25_index = get_or_create_bm25_index_cached(tuple(st.session_state.documents))
                logging.info("BM25 index rebuilt from loaded data.")

                # Instantiate Retrievers using utility function
                chroma_retriever, bm25_retriever = create_retrievers(
                    collection=st.session_state.collection,
                    embedder=embedder,
                    bm25_processor=st.session_state.bm25_index,
                    corpus=st.session_state.documents
                )

                # Instantiate RAG pipeline using utility function
                st.session_state.rag_pipeline = create_rag_pipeline(
                    vector_retriever=chroma_retriever,
                    keyword_retriever=bm25_retriever,
                    reranker_model=reranker,
                    llm=llm
                )

                st.session_state.data_indexed = True
                logging.info("Existing data loaded successfully.")
                st.success(f"Loaded {len(st.session_state.documents)} documents from existing database.")
                return True
        else:
            logging.info("Existing collection found but is empty.")
            return False
    except Exception as e:
        logging.info(f"Could not load existing data (may be first run or an error): {e}")
        return False

# --- Streamlit App ---
st.title("📄 RAG Pipeline Interface")

# Add project description below the title
st.markdown("""
This application demonstrates a Retrieval-Augmented Generation (RAG) pipeline built with DSPy.
It combines keyword search (BM25) and dense vector search (ChromaDB with Sentence Transformers)
followed by a reranking step (Cross-Encoder) to retrieve relevant context for a Language Model (LLM)
to generate an answer.

Use the sidebar to load and index your text data before asking questions.
""")

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

    if st.button("Load and Index New Data"):
        docs_to_index = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                docs_to_index.extend(stringio.read().splitlines()) # Simple split by line
        elif pasted_text:
            docs_to_index = pasted_text.strip().split('\n')

        if docs_to_index:
            with st.spinner("Processing and indexing new data..."):
                try:
                    st.session_state.documents = [doc for doc in docs_to_index if doc.strip()]
                    st.session_state.doc_ids = [f"doc_{i}" for i in range(len(st.session_state.documents))]
                    collection_name = config.CHROMA_COLLECTION_NAME + "_st"

                    # Index in Chroma using utility function (clears by default)
                    st.session_state.collection = index_chroma_data(
                        client=client,
                        embedder=embedder,
                        documents=st.session_state.documents,
                        doc_ids=st.session_state.doc_ids,
                        collection_name=collection_name,
                        clear_existing=True # Explicitly clear when loading new data
                    )

                    # Create BM25 Index using cached function
                    st.session_state.bm25_index = get_or_create_bm25_index_cached(tuple(st.session_state.documents))

                    # Instantiate Retrievers using utility function
                    chroma_retriever, bm25_retriever = create_retrievers(
                        collection=st.session_state.collection,
                        embedder=embedder,
                        bm25_processor=st.session_state.bm25_index,
                        corpus=st.session_state.documents
                    )

                    # Initialize RAG pipeline using utility function
                    st.session_state.rag_pipeline = create_rag_pipeline(
                        vector_retriever=chroma_retriever,
                        keyword_retriever=bm25_retriever,
                        reranker_model=reranker,
                        llm=llm
                    )

                    # Update state based on whether pipeline was created
                    if st.session_state.rag_pipeline:
                        st.session_state.data_indexed = True
                        st.success(f"Successfully indexed {len(st.session_state.documents)} new documents! RAG Pipeline is ready.")
                    else:
                        # If pipeline failed (no LLM), still mark data as indexed for search
                        st.session_state.data_indexed = True
                        st.info("New data indexing complete. Search available but final answer generation requires LLM.")

                except Exception as e:
                    st.error(f"An error occurred during indexing: {e}")
                    logging.error(f"Indexing error: {e}", exc_info=True)
                    st.session_state.data_indexed = False
                    st.session_state.rag_pipeline = None # Ensure pipeline is None on error
        else:
            st.warning("No documents provided to index.")

    # --- Sidebar Status Update ---
    if st.session_state.data_indexed:
        st.sidebar.success(f"{len(st.session_state.documents)} documents indexed.")
        if not st.session_state.rag_pipeline:
             st.sidebar.info("LLM not configured. Search limited.")
    else:
        st.sidebar.info("Load data using the options above or data might load automatically if previously indexed.")


# --- Main Search Area ---
st.header("🔍 Search")

if not st.session_state.data_indexed:
    st.warning("Please load and index data using the sidebar, or wait for automatic loading if data exists.")
elif not st.session_state.rag_pipeline:
     st.warning("RAG Pipeline not initialized (LLM configuration failed or missing). Cannot generate answers.")
else:
    question = st.text_input("Enter your question:")
    enable_llm = st.checkbox("Enable LLM Generation", value=False, help="If unchecked, only reranked context will be shown; no LLM answer will be generated.")

    if st.button("Search"):
        if question:
            with st.spinner("Searching..."):
                try:
                    response = st.session_state.rag_pipeline(question=question, use_llm=enable_llm)
                    if enable_llm:
                        answer = response.answer
                        st.subheader("Answer:")
                        st.markdown(answer)
                    else:
                        # Only show reranked context if LLM is disabled
                        if hasattr(response, 'context') and response.context:
                            st.subheader("Reranked Results (Context)")
                            if isinstance(response.context, list):
                                st.write("\n\n---\n\n".join(response.context))
                            else:
                                st.write(response.context)
                except Exception as e:
                    st.error(f"An error occurred during search: {e}")
                    logging.error(f"Search error: {e}", exc_info=True)
        else:
            st.warning("Please enter a question.")