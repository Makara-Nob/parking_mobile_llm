import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "data")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "chroma_db")

# --- MODEL CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure directories exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
