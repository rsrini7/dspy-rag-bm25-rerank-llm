import requests
import argparse

parser = argparse.ArgumentParser(description="DSPy RAG Client: Ask a question or upload a file.")
parser.add_argument("--question","-q", type=str, help="Question to ask the RAG API.")
parser.add_argument("--file","-f", type=str, help="Path to file to upload.")
parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation. If not set, only reranked context is shown.")
args = parser.parse_args()

# Both question and file must be provided together
if args.question and args.file:
    with open(args.file, "rb") as f:
        files = {"file": (args.file, f)}
        response = requests.post("http://127.0.0.1:8001/upload_file/", files=files)
    print(f"Status: {response.status_code}\nResponse:\n {response.text}")
    payload = {"question": args.question}
    if args.llm:
        payload["use_llm"] = True
    response = requests.post("http://127.0.0.1:8001/predict", json=payload)
    print(f"Status: {response.status_code}\nResponse:\n {response.text}")
else:
    print("Both --question and --file must be provided together. Using default question.")
    payload = {"question": "what is dspy ?"}
    if args.llm:
        payload["use_llm"] = True
    response = requests.post("http://127.0.0.1:8001/predict", json=payload)
    print(f"Status: {response.status_code}\nResponse:\n {response.text}")
