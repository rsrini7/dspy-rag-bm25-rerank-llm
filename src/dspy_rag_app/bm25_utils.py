from typing import List
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dspy_rag_app.nltk_utils import ensure_nltk_resources

class BM25Processor:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.tokenized_docs = self._preprocess_documents()
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _preprocess_documents(self) -> List[List[str]]:
        """Tokenize and preprocess documents"""
        ensure_nltk_resources()
        stop_words = set(stopwords.words('english'))
        tokenized_docs_bm25 = [
            [word.lower() for word in word_tokenize(doc) if word.isalnum() and word.lower() not in stop_words]
            for doc in self.documents  # Changed from 'documents' to 'self.documents'
        ]
        return tokenized_docs_bm25

    def _preprocess_query(self, query: str) -> List[str]:
        """Preprocess a query for BM25 scoring"""
        ensure_nltk_resources()
        stop_words = set(stopwords.words('english'))
        return [
            word.lower() for word in word_tokenize(query) 
            if word.isalnum() and word.lower() not in stop_words
        ]

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for a query"""
        tokenized_query = self._preprocess_query(query)
        return self.bm25.get_scores(tokenized_query)
