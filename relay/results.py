"""Authority module for writing results.csv according to the brief's exact specification."""
import csv
import threading
from pathlib import Path
from typing import Literal

RESULTS_FILE = Path("results.csv")

FIELDNAMES = [
    "request_id",
    "scenario",
    "source",            # 'real' | 'stand-in'
    "start_time",        # ISO timestamp or seconds float
    "first_token_ms",    # float / int ms
    "total_ms",          # float / int ms
    "status",            # 'ok' | 'rate_limited' | 'timeout' | 'error'
    "retries",           # int count
    "tokens_in",         # int count
    "tokens_out",        # int count
    "retrieval_setting", # e.g. 'setting_1', 'setting_2', 'setting_3'
    "answer_grade"       # 'correct' | 'wrong' | 'refused_correctly' | 'refused_wrongly'
]

_lock = threading.Lock()

def init_results_csv(overwrite: bool = False):
    """Initializes the CSV with headers if it does not exist or if overwrite is requested."""
    with _lock:
        if overwrite or not RESULTS_FILE.exists():
            with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

def log_result(
    request_id: str,
    scenario: str,
    source: Literal["real", "stand-in"],
    start_time: str,
    first_token_ms: float,
    total_ms: float,
    status: Literal["ok", "rate_limited", "timeout", "error"],
    retries: int,
    tokens_in: int,
    tokens_out: int,
    retrieval_setting: str,
    answer_grade: Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]
):
    """Thread-safe append of a single request outcome to results.csv."""
    with _lock:
        if not RESULTS_FILE.exists():
            init_results_csv()
            
        with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({
                "request_id": request_id,
                "scenario": scenario,
                "source": source,
                "start_time": start_time,
                "first_token_ms": round(first_token_ms, 1),
                "total_ms": round(total_ms, 1),
                "status": status,
                "retries": retries,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "retrieval_setting": retrieval_setting,
                "answer_grade": answer_grade
            })
