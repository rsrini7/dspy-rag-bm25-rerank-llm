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
