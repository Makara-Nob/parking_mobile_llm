import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "data")

# --- VECTOR DB CONFIG (UPSTASH) ---
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

# --- MODEL CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure directories exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)
