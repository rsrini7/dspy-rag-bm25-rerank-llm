import os
import nltk
import logging
logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)
    
def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    nltk_data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '.venv', 'nltk_data')
    )
    
    logger.info(f"Using NLTK data directory: {nltk_data_dir}")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    try:
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK resource 'punkt_tab' found.")
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK resource 'stopwords' found.")
        logger.info("All required NLTK resources are present. No download needed.")
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt_tab', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        logger.info("NLTK resources downloaded.")