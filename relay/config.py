"""Configuration parameters for The Relay pipeline, limiter, and mock stand-in."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

STANDIN_HOST = "127.0.0.1"
STANDIN_PORT = 8000
STANDIN_BASE_URL = f"http://{STANDIN_HOST}:{STANDIN_PORT}/v1"

RPM_LIMIT = 30
TPM_LIMIT = 6000
RPD_LIMIT = 14400
TPD_LIMIT = 500000
MAX_CONCURRENCY = 5

MAX_QUEUE_SIZE = 25
REQUEST_TIMEOUT_SEC = 12.0

RETRIEVAL_SETTINGS = {
    "setting_1": {
        "name": "Setting 1 (Compact)",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 3,
        "rerank": False,
        "description": "500 chars, k=3, no re-rank (Low token overhead)"
    },
    "setting_2": {
        "name": "Setting 2 (Balanced)",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "rerank": True,
        "description": "1000 chars, k=5, term re-ranking (Recommended)"
    },
    "setting_3": {
        "name": "Setting 3 (Heavy)",
        "chunk_size": 2000,
        "chunk_overlap": 300,
        "top_k": 8,
        "rerank": True,
        "description": "2000 chars, k=8, term re-ranking (High recall, high TPM risk)"
    }
}
DEFAULT_RETRIEVAL_SETTING = "setting_2"
