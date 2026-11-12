"""Configuration parameters for The Relay pipeline, limiter, and mock stand-in."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Groq Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

# Stand-in Server Configuration
STANDIN_HOST = "127.0.0.1"
STANDIN_PORT = 8000
STANDIN_BASE_URL = f"http://{STANDIN_HOST}:{STANDIN_PORT}/v1"

# Provider Limits (Calibrated via probe and official docs)
# Groq free tier baseline for llama-3.1-8b-instant:
RPM_LIMIT = 30           # Requests Per Minute
TPM_LIMIT = 6000         # Tokens Per Minute
RPD_LIMIT = 14400        # Requests Per Day
TPD_LIMIT = 500000       # Tokens Per Day
MAX_CONCURRENCY = 5      # Max concurrent active connections

# Queue and Admission Limits
MAX_QUEUE_SIZE = 25      # Bounded queue before load shedding (503)
REQUEST_TIMEOUT_SEC = 12.0  # Max waiting + execution deadline before dropping

# Retrieval Settings for comparison (Table 3)
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

