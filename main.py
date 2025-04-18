import dspy
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
from dotenv import load_dotenv

# --- Configuration ---

# Load environment variables from .env
load_dotenv()

# NLTK Data (Download only if necessary)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK resources (punkt, stopwords)...")
    nltk.download('punkt', quiet=True, download_dir='.venv/nltk_data')
    nltk.download('stopwords', quiet=True, download_dir='.venv/nltk_data')
    print("NLTK resources downloaded.")

# Environment Variables and Constants (from .env)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-mpnet-base-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
K_EMBEDDING_RETRIEVAL = int(os.getenv("K_EMBEDDING_RETRIEVAL", 10))
K_BM25_RETRIEVAL = int(os.getenv("K_BM25_RETRIEVAL", 10))
K_RERANK = int(os.getenv("K_RERANK", 3))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_dspy")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hybrid_rag_docs")

if not OPENROUTER_API_KEY:
    warnings.warn("OPENROUTER_API_KEY environment variable not set. LLM calls will fail.")

# --- Sample Data ---
DOCUMENTS = [
    "Python is a versatile high-level programming language widely used for web development, data science, artificial intelligence, and scripting.",
    "DSPy is a framework from Stanford NLP for algorithmically optimizing Language Model prompts and weights, especially for complex pipelines.",
    "Chroma is an open-source embedding database (vector store) designed to make it easy to build LLM apps by making knowledge, facts, and skills pluggable.",
    "BM25 (Okapi BM25) is a ranking function used by search engines to rank matching documents according to their relevance to a given search query. It is based on the probabilistic retrieval framework.",
    "Large Language Models (LLMs) like GPT-4 demonstrate powerful capabilities in text generation, translation, and question answering.",
    "Reranking is a crucial step in information retrieval systems to improve the relevance ordering of initially retrieved documents.",
    "Sentence Transformers is a Python library for state-of-the-art sentence, text, and image embeddings.",
    "Hybrid search combines keyword-based search (like BM25) and semantic vector search to leverage the strengths of both approaches."
]
DOCUMENT_IDS = [f"doc_{i}" for i in range(len(DOCUMENTS))]

# --- Model and Client Initialization ---

print(f"Initializing Embedder: {EMBEDDER_MODEL}")
embedder = SentenceTransformer(EMBEDDER_MODEL)

print(f"Initializing Reranker: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)

print(f"Initializing ChromaDB Client (Path: {CHROMA_DB_PATH})")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
print(f"Getting/Creating Chroma Collection: {CHROMA_COLLECTION_NAME}")
collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

# --- Data Preprocessing and Indexing ---

# Index data in Chroma if collection is empty or doesn't match
existing_ids = set(collection.get(include=[])['ids'])
if not existing_ids or existing_ids != set(DOCUMENT_IDS):
    print("Collection is empty or different. Indexing documents in ChromaDB...")
    embeddings = embedder.encode(DOCUMENTS, show_progress_bar=True)
    collection.upsert( # Use upsert to handle potential re-runs
        documents=DOCUMENTS,
        embeddings=embeddings.tolist(), # Chroma expects lists
        ids=DOCUMENT_IDS
    )
    print(f"Indexed {len(DOCUMENTS)} documents in ChromaDB.")
else:
    print("Documents already indexed in ChromaDB.")

# Preprocess documents for BM25
print("Preprocessing documents for BM25...")
stop_words = set(stopwords.words('english'))
tokenized_docs_bm25 = [
    [word.lower() for word in word_tokenize(doc) if word.isalnum() and word.lower() not in stop_words]
    for doc in DOCUMENTS
]
bm25 = BM25Okapi(tokenized_docs_bm25)
print("BM25 index created.")

# --- DSPy Component Definitions ---

class ChromaRetriever(dspy.Retrieve):
    """DSPy Retriever for ChromaDB using SentenceTransformer embeddings."""
    def __init__(self, chroma_collection, embed_model, k=3):
        self._collection = chroma_collection
        self._embedder = embed_model
        self._k = k
        super().__init__(k=k)

    def forward(self, query_or_queries, k=None):
        """Search ChromaDB for top k documents based on query embedding."""
        k = k if k is not None else self._k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        query_embeddings = self._embedder.encode(queries, show_progress_bar=False).tolist()

        # Perform the query
        results = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            include=['documents'] # Only fetch documents content
        )

        # Process results: Extract documents for each query
        final_results = []
        for docs in results['documents']:
            # Handle case where fewer than k docs are returned or query returns nothing
            if docs:
                 final_results.extend([dspy.Example(long_text=doc) for doc in docs])
            # You might want more sophisticated handling if matching results to queries in batch mode
        
        # For simplicity, assuming single query or flat list is acceptable
        # If strict per-query results are needed, return structure should be List[List[Example]]
        # print(f"Chroma retrieved: {[r.long_text[:50] + '...' for r in final_results]}") # Debug
        return final_results


class BM25Retriever(dspy.Retrieve):
    """DSPy Retriever using BM25Okapi for keyword search."""
    def __init__(self, bm25_index, corpus, k=3):
        self._bm25 = bm25_index
        self._corpus = corpus # The original list of document texts
        self._k = k
        super().__init__(k=k)

    def forward(self, query_or_queries, k=None):
        """Search corpus using BM25 for top k documents."""
        k = k if k is not None else self._k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries

        all_results = []
        for query in queries:
            # Preprocess the query
            tokenized_query = [
                word.lower() for word in word_tokenize(query) if word.isalnum() and word.lower() not in stop_words
            ]
            # Get scores for the *entire* corpus indexed by BM25
            scores = self._bm25.get_scores(tokenized_query)

            # Combine scores with indices, sort, and get top k indices
            indexed_scores = list(enumerate(scores))
            ranked_indices = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:k]

            # Get the actual documents using the indices
            # Filter out results with score 0 or less if needed
            top_docs = [self._corpus[idx] for idx, score in ranked_indices if score > 0]

            all_results.extend([dspy.Example(long_text=doc) for doc in top_docs])
            # As with Chroma, returning flat list for simplicity

        # print(f"BM25 retrieved: {[r.long_text[:50] + '...' for r in all_results]}") # Debug
        return all_results


class RAGHybridFusedRerank(dspy.Module):
    """
    DSPy Module implementing a RAG pipeline with:
    1. Hybrid Retrieval (Vector + BM25)
    2. Fusion (Combine + Deduplicate)
    3. Neural Reranking (CrossEncoder)
    4. LLM Generation
    """
    
    def __init__(self, vector_retriever, keyword_retriever, reranker_model, llm, rerank_k=3):
        super().__init__()
        self.vector_retrieve = vector_retriever
        self.keyword_retrieve = keyword_retriever
        # Wrap the CrossEncoder model for use in dspy.ReRank
        self.reranker = reranker_model
        self.generate = dspy.Predict(
            "context, question -> answer",
            # You can add instructions here:
            # "Instructions: Based *only* on the provided context, answer the question concisely. If the answer is not found in the context, state 'I could not find the answer in the provided documents.'\n---\nContext: {context}\nQuestion: {question}\nAnswer:"
        )
        self.llm = llm # Store the configured LLM
        self.rerank_k = rerank_k

    def rerank(self, query, documents):
        # Use CrossEncoder for reranking
        scores = self.reranker.predict([(query, doc) for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.rerank_k]]

    def forward(self, question):
        # 1. Retrieve from both sources
        vector_results = self.vector_retrieve(question)
        keyword_results = self.keyword_retrieve(question)

        # 2. Fuse the results: Combine and deduplicate
        # Use a dictionary to handle deduplication based on 'long_text'
        fused_docs = {doc['long_text']: doc for doc in vector_results + keyword_results}
        fused_list = list(fused_docs.values())

        # 3. Rerank using the self.rerank method
        reranked_docs = self.rerank(question, [doc['long_text'] for doc in fused_list])
        context = "\n".join(reranked_docs)

        # 4. Generate answer using LLM
        with dspy.settings.context(lm=self.llm):
            prediction = self.generate(question=question, context=context)
            return prediction

# --- DSPy Configuration ---

# Configure DSPy to use the chosen LLM via OpenRouter (Gemini or other)
if OPENROUTER_API_KEY:
    openrouter_lm = dspy.LM(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        api_base="https://openrouter.ai/api/v1",
        provider="openrouter",
        max_tokens=500
    )
    dspy.settings.configure(lm=openrouter_lm)
    print(f"DSPy configured with LLM: {LLM_MODEL} via OpenRouter")
else:
    # Fallback or error if no API key
    print("DSPy LM not configured due to missing API key.")
    # You might configure a local model here if available, or exit.
    # Example for a local HF model (if you had one loaded):
    # from dspy.models.hf import HFModel
    # local_llm = HFModel(model="path/to/your/local/model")
    # dspy.settings.configure(lm=local_llm)
    openrouter_lm = None # Ensure it's None if not configured


# --- Main Execution & Testing ---

if __name__ == "__main__":
    if not dspy.settings.lm:
        print("\nERROR: DSPy Language Model not configured. Exiting.")
    else:
        print("\n--- Initializing RAG Pipeline ---")
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

        print("\n--- Running Test Query ---")
        test_question = "What is DSPy used for and how does it relate to LLMs?"
        print(f"Question: {test_question}")

        # Run the pipeline
        response = rag_pipeline(question=test_question)

        print("\n--- Generated Answer ---")
        print(response.answer)

        # Optional: Inspect the full trace (if needed for debugging)
        # print("\n--- DSPy LM Trace ---")
        # try:
        #     dspy.settings.lm.inspect_history(n=1)
        # except Exception as e:
        #     print(f"Could not inspect history: {e}")

        print("\n--- Example with potentially no good context ---")
        test_question_2 = "Tell me about the history of cheese making."
        print(f"Question: {test_question_2}")
        response_2 = rag_pipeline(question=test_question_2)
        print("\n--- Generated Answer ---")
        print(response_2.answer) # Observe how the LLM handles irrelevant context based on prompt/defaults