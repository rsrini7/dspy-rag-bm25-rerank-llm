import dspy
import logging
logging.basicConfig(level=logging.INFO)

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
        self.reranker = reranker_model
        self.generate = dspy.Predict(
            "context, question -> answer",
        )
        self.llm = llm
        self.rerank_k = rerank_k

    def rerank(self, query, documents):
        scores = self.reranker.predict([(query, doc) for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.rerank_k]]

    def forward(self, question):
        vector_results = self.vector_retrieve(question)
        keyword_results = self.keyword_retrieve(question)
        fused_docs = {doc['long_text']: doc for doc in vector_results + keyword_results}
        fused_list = list(fused_docs.values())
        reranked_docs = self.rerank(question, [doc['long_text'] for doc in fused_list])
        context = "\n".join(reranked_docs)
        with dspy.settings.context(lm=self.llm):
            prediction = self.generate(question=question, context=context)
            return prediction
