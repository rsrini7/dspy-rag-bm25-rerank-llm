# Need to install litserve: pip install litserve
import litserve as ls # Import LitServe
import dspy
import os
import time
from dotenv import load_dotenv
from dspy_rag_app.config import config
from dspy_rag_app.nltk_utils import ensure_nltk_resources
from dspy_rag_app.data import DOCUMENTS, DOCUMENT_IDS # Rename default import
from dspy_rag_app.utils import load_components, create_retrievers, create_rag_pipeline
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- LitServe API Definition ---
class DSPyRAGAPI(ls.LitAPI):
    """LitAPI implementation to serve the DSPy RAG pipeline."""

    def setup(self, device):
        """
        Initialize models, retrievers, and the DSPy pipeline.
        This runs once per worker process.
        'device' argument is provided by LitServe (e.g., 'cuda:0', 'cpu')
        """
        print(f"[{os.getpid()}] Setting up DSPy RAG API on device: {device}...")
        start_time = time.time()

        # --- Load all core components using utility function ---
        self.embedder, self.reranker, self.client, self.llm = load_components(db_path=config.CHROMA_DB_PATH_API)
        self.collection = self.client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)

        # --- Data Preprocessing and Indexing (Ensure it runs if needed) ---
        existing_ids = set(self.collection.get(include=[])['ids'])
        if not existing_ids or existing_ids != set(DOCUMENT_IDS):
            print(f"[{os.getpid()}] Warning: Chroma collection empty or different. Indexing...")
            from dspy_rag_app.utils import index_chroma_data
            index_chroma_data(self.client, self.embedder, DOCUMENTS, DOCUMENT_IDS, config.CHROMA_COLLECTION_NAME, clear_existing=True)
            print(f"[{os.getpid()}] Indexed {len(DOCUMENTS)} documents in ChromaDB.")
        else:
            print(f"[{os.getpid()}] Documents already indexed in ChromaDB.")

        # --- BM25 Indexing using utility ---
        from dspy_rag_app.utils import create_bm25_index
        self.bm25 = create_bm25_index(DOCUMENTS)
        print(f"[{os.getpid()}] BM25 index created.")

        # --- DSPy Configuration handled in load_components ---
        if not self.llm:
            raise RuntimeError(f"[{os.getpid()}] LLM could not be configured. Aborting setup.")

        # --- Instantiate DSPy Retrievers using utility ---
        self.chroma_retriever, self.bm25_retriever = create_retrievers(self.collection, self.embedder, self.bm25, DOCUMENTS)

        # --- Instantiate DSPy RAG Pipeline using utility ---
        self.rag_pipeline = create_rag_pipeline(self.chroma_retriever, self.bm25_retriever, self.reranker, self.llm)

        end_time = time.time()
        print(f"[{os.getpid()}] Setup complete in {end_time - start_time:.2f} seconds.")


    def decode_request(self, request):
        """Parse the incoming request (expected JSON)."""
        try:
            if isinstance(request, dict):
                content = request
            elif hasattr(request, "json") and callable(request.json):
                content = request.json()
            else:
                raise ValueError("Request is not a dict and does not have a .json() method.")
            question = content.get("question")
            if not question or not isinstance(question, str):
                raise ValueError("'question' field missing or not a string in request JSON.")
            print(f"[{os.getpid()}] Decoded question: {question[:50]}...") # Log snippet
            return question # Return the essential data needed for predict
        except Exception as e:
            print(f"[{os.getpid()}] Error decoding request: {e}")
            raise ValueError(f"Bad Request: Could not decode JSON or missing 'question'. Error: {e}")


    def predict(self, decoded_request):
        """
        Run the DSPy pipeline with the decoded input.
        'decoded_request' is the output from decode_request (the question string).
        """
        question = decoded_request
        print(f"[{os.getpid()}] Running prediction for: {question[:50]}...")
        start_time = time.time()
        try:
             # Ensure the correct LM context is used if not relying solely on global dspy.settings
             # (Though configuring in setup should handle this for the worker)
             with dspy.settings.context(lm=self.llm):
                  prediction = self.rag_pipeline(question=question)

             end_time = time.time()
             print(f"[{os.getpid()}] Prediction finished in {end_time - start_time:.2f}s.")
             # The result 'prediction' is likely a dspy.Prediction object
             return prediction
        except Exception as e:
             print(f"[{os.getpid()}] Error during DSPy pipeline execution: {e}")
             # Raise an error that LitServe can catch and report
             raise ls.LitAPIStatusError(500, f"Internal Server Error during prediction: {e}")


    def encode_response(self, prediction):
        """Format the prediction output into the API response (JSON)."""
        # 'prediction' is the output from the predict method (the dspy.Prediction object)
        try:
            answer = prediction.answer # Access the 'answer' field
            print(f"[{os.getpid()}] Encoding answer: {answer[:50]}...") # Log snippet
            # Return a dictionary, LitServe will automatically convert to JSON
            return {"answer": answer}
        except AttributeError:
            # Handle cases where the prediction object might not have 'answer'
             print(f"[{os.getpid()}] Error encoding response: 'answer' field missing in prediction.")
             raise ls.LitAPIStatusError(500, "Internal Server Error: Could not format prediction.")
        except Exception as e:
             print(f"[{os.getpid()}] Error during response encoding: {e}")
             raise ls.LitAPIStatusError(500, f"Internal Server Error during encoding: {e}")

    async def upload_file(self, file):
        """
        Accept a file upload (multipart/form-data), process its contents, and update the document index.
        """
        from fastapi import UploadFile
        try:
            # Read file contents
            if isinstance(file, UploadFile):
                file_content = await file.read()
                filename = file.filename
            else:
                file_content = file.read()
                filename = getattr(file, 'name', 'uploaded_file')
            # Assume text file for now; adapt as needed for other formats
            text = file_content.decode("utf-8")
            # Here, split or parse text into documents as needed
            # For demonstration, treat the whole file as one document
            new_doc_id = f"uploaded_{filename}"
            logger.info(f"Uploaded file: {filename}")
            new_docs = [text]
            new_doc_ids = [new_doc_id]
            # Index the new document(s)
            from dspy_rag_app.utils import index_chroma_data
            index_chroma_data(self.client, self.embedder, new_docs, new_doc_ids, config.CHROMA_COLLECTION_NAME, clear_existing=False)
            # Update BM25 index as well
            self.bm25 = self.bm25 + new_docs
            # Optionally, update retrievers if needed
            self.chroma_retriever, self.bm25_retriever = create_retrievers(self.collection, self.embedder, self.bm25, DOCUMENTS + new_docs)
            self.rag_pipeline = create_rag_pipeline(self.chroma_retriever, self.bm25_retriever, self.reranker, self.llm)
            return {"status": "success", "message": f"File '{filename}' uploaded and indexed."}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# --- Main Execution: Run the LitServe Server ---
if __name__ == "__main__":
    # Ensure NLTK data is available before starting server setup
    ensure_nltk_resources()

    print("--- Starting LitServe Server ---")
    # Instantiate the LitAPI
    api = DSPyRAGAPI()

    # Run the LitServe server (remove FastAPI integration for now)
    server = ls.LitServer(api, accelerator="auto", max_batch_size=1, timeout=60)
    # --- Add custom upload_file endpoint to FastAPI app ---
    from fastapi import UploadFile, File
    from fastapi.responses import JSONResponse
    @server.app.post("/upload_file")
    async def upload_file_endpoint(file: UploadFile = File(...)):
        result = await api.upload_file(file)
        return JSONResponse(content=result)
    server.run(port=8001) # Choose a port