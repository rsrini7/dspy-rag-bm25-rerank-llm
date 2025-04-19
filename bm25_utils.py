import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK resources (punkt, stopwords)...")
        nltk.download('punkt', quiet=True, download_dir='.venv/nltk_data')
        nltk.download('stopwords', quiet=True, download_dir='.venv/nltk_data')
        print("NLTK resources downloaded.")

def preprocess_for_bm25(documents):
    stop_words = set(stopwords.words('english'))
    tokenized_docs_bm25 = [
        [word.lower() for word in word_tokenize(doc) if word.isalnum() and word.lower() not in stop_words]
        for doc in documents
    ]
    return tokenized_docs_bm25
