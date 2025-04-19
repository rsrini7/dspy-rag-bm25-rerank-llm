import logging
logging.basicConfig(level=logging.INFO)

import dspy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class ChromaRetriever(dspy.Retrieve):
    """DSPy Retriever for ChromaDB using SentenceTransformer embeddings."""
    def __init__(self, chroma_collection, embed_model, k=3):
        self._collection = chroma_collection
        self._embedder = embed_model
        self._k = k
        super().__init__(k=k)

    def forward(self, query_or_queries, k=None):
        k = k if k is not None else self._k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        query_embeddings = self._embedder.encode(queries, show_progress_bar=False).tolist()
        results = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            include=['documents']
        )
        final_results = []
        for docs in results['documents']:
            if docs:
                final_results.extend([dspy.Example(long_text=doc) for doc in docs])
        return final_results

class BM25Retriever(dspy.Retrieve):
    """DSPy Retriever using BM25Okapi for keyword search."""
    def __init__(self, bm25_index, corpus, k=3):
        self._bm25 = bm25_index
        self._corpus = corpus
        self._k = k
        super().__init__(k=k)

    def forward(self, query_or_queries, k=None):
        k = k if k is not None else self._k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        stop_words = set(stopwords.words('english'))
        all_results = []
        for query in queries:
            tokenized_query = [
                word.lower() for word in word_tokenize(query) if word.isalnum() and word.lower() not in stop_words
            ]
            scores = self._bm25.get_scores(tokenized_query)
            indexed_scores = list(enumerate(scores))
            ranked_indices = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:k]
            top_docs = [self._corpus[idx] for idx, score in ranked_indices if score > 0]
            all_results.extend([dspy.Example(long_text=doc) for doc in top_docs])
        return all_results
