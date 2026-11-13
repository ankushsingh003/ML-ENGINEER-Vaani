"""
relay/trace.py
--------------
Single-authority for per-request lifecycle event tracing.
Writes one JSON object per line to results_trace.jsonl.
Every timestamp is time.perf_counter() relative to the
process start, plus an ISO wall-clock anchor at t=0.

Lifecycle events (in order):
  enqueued       – request entered the bounded queue
  dequeued       – a worker picked it up
  retrieval_done – BM25 (and optional re-retrieve) finished
  model_sent     – HTTP request dispatched to provider
  first_token    – first streamed byte received from provider
  completed      – stream closed, verify done
  shed           – rejected at admission (queue full)
  timed_out      – deadline expired (in queue or during exec)
  error          – unhandled exception

TTFT that the user experiences = first_token - enqueued
Queue wait                      = dequeued   - enqueued
Retrieval time                  = retrieval_done - dequeued
Model prefill                   = first_token    - model_sent
Streaming time                  = completed      - first_token
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

_PROCESS_START = time.perf_counter()
_WALL_ANCHOR = time.time()          # UTC epoch seconds at process start

TRACE_FILE = Path("results/results_trace.jsonl")
_lock = threading.Lock()


def _now_ms() -> float:
    """Monotonic milliseconds since process start."""
    return (time.perf_counter() - _PROCESS_START) * 1000.0


def _wall_iso() -> str:
    offset_s = (time.perf_counter() - _PROCESS_START)
    return time.strftime(
        "%Y-%m-%dT%H:%M:%S",
        time.gmtime(_WALL_ANCHOR + offset_s)
    ) + "Z"


def _append(record: dict) -> None:
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class RequestTrace:
    """
    Accumulates timestamps for one request then writes to trace file.
    Usage:
        t = RequestTrace(request_id="…", scenario="…", source="…",
                         queue_len_at_enqueue=N)
        t.mark_enqueued()
        …
        t.mark_first_token(tokens_in=N)
        t.finish(tokens_in=N, tokens_out=N, retries=N,
                 status="ok", answer_grade="correct",
                 retrieval_setting="setting_2")
    """

    def __init__(self, request_id: str, scenario: str,
                 source: str, queue_len_at_enqueue: int = 0):
        self.request_id = request_id
        self.scenario = scenario
        self.source = source
        self.queue_len_at_enqueue = queue_len_at_enqueue

        # Monotonic ms timestamps (None until marked)
        self.t_enqueued:        Optional[float] = None
        self.t_dequeued:        Optional[float] = None
        self.t_retrieval_done:  Optional[float] = None
        self.t_model_sent:      Optional[float] = None
        self.t_first_token:     Optional[float] = None
        self.t_completed:       Optional[float] = None

        self.wall_enqueued: str = ""

    # ------------------------------------------------------------------
    # Lifecycle markers
    # ------------------------------------------------------------------

    def mark_enqueued(self) -> None:
        self.t_enqueued = _now_ms()
        self.wall_enqueued = _wall_iso()

    def mark_dequeued(self) -> None:
        self.t_dequeued = _now_ms()

    def mark_retrieval_done(self) -> None:
        self.t_retrieval_done = _now_ms()

    def mark_model_sent(self) -> None:
        self.t_model_sent = _now_ms()

    def mark_first_token(self) -> None:
        self.t_first_token = _now_ms()

    # ------------------------------------------------------------------
    # Terminal events (shed / timeout / error / completed)
    # ------------------------------------------------------------------

    def _base_record(self) -> dict:
        return {
            "request_id":           self.request_id,
            "scenario":             self.scenario,
            "source":               self.source,
            "wall_enqueued":        self.wall_enqueued,
            "queue_len_at_enqueue": self.queue_len_at_enqueue,
            "t_enqueued_ms":        _r(self.t_enqueued),
            "t_completed_ms":       _r(self.t_completed),
        }

    def finish(self, *, tokens_in: int, tokens_out: int, retries: int,
               status: str, answer_grade: str, retrieval_setting: str,
               queue_wait_ms: Optional[float] = None,
               retrieval_ms: Optional[float] = None,
               prefill_ms: Optional[float] = None,
               streaming_ms: Optional[float] = None) -> dict:
        """
        Call at successful or errored completion.
        Returns the full record dict (also written to trace file).
        """
        self.t_completed = _now_ms()
        t0 = self.t_enqueued or self.t_completed

        # Derived intervals (all in ms, all positive)
        qw   = queue_wait_ms  if queue_wait_ms  is not None else _diff(self.t_enqueued, self.t_dequeued)
        retr = retrieval_ms   if retrieval_ms   is not None else _diff(self.t_dequeued, self.t_retrieval_done)
        pre  = prefill_ms     if prefill_ms     is not None else _diff(self.t_model_sent, self.t_first_token)
        strm = streaming_ms   if streaming_ms   is not None else _diff(self.t_first_token, self.t_completed)

        # True user TTFT = first_token from the user's perspective
        if self.t_first_token and self.t_enqueued:
            ttft_ms = self.t_first_token - self.t_enqueued
        elif self.t_completed and self.t_enqueued:
            ttft_ms = self.t_completed - self.t_enqueued   # no streaming
        else:
            ttft_ms = 0.0

        total_ms = max(0.0, (self.t_completed or t0) - t0)

        rec = self._base_record()
        rec.update({
            "status":            status,
            "retries":           retries,
            "tokens_in":         tokens_in,
            "tokens_out":        tokens_out,
            "retrieval_setting": retrieval_setting,
            "answer_grade":      answer_grade,
            # Timing breakdown
            "queue_wait_ms":     _r(qw),
            "retrieval_ms":      _r(retr),
            "prefill_ms":        _r(pre),
            "streaming_ms":      _r(strm),
            "ttft_ms":           _r(ttft_ms),
            "total_ms":          _r(total_ms),
        })
        _append(rec)
        return rec

    def shed(self) -> dict:
        """Request rejected at admission (queue full)."""
        self.t_completed = _now_ms()
        t0 = self.t_enqueued or self.t_completed
        total_ms = max(0.0, self.t_completed - t0)
        rec = self._base_record()
        rec.update({
            "status":            "shed",
            "retries":           0,
            "tokens_in":         0,
            "tokens_out":        0,
            "retrieval_setting": "n/a",
            "answer_grade":      "n/a",
            "queue_wait_ms":     _r(total_ms),
            "retrieval_ms":      0.0,
            "prefill_ms":        0.0,
            "streaming_ms":      0.0,
            "ttft_ms":           _r(total_ms),
            "total_ms":          _r(total_ms),
        })
        _append(rec)
        return rec

    def timed_out(self, retrieval_setting: str = "n/a") -> dict:
        """Request cancelled after deadline elapsed."""
        self.t_completed = _now_ms()
        t0 = self.t_enqueued or self.t_completed
        total_ms = max(0.0, self.t_completed - t0)
        qw = _diff(self.t_enqueued, self.t_dequeued)
        rec = self._base_record()
        rec.update({
            "status":            "timeout",
            "retries":           0,
            "tokens_in":         0,
            "tokens_out":        0,
            "retrieval_setting": retrieval_setting,
            "answer_grade":      "n/a",
            "queue_wait_ms":     _r(qw),
            "retrieval_ms":      _r(_diff(self.t_dequeued, self.t_retrieval_done)),
            "prefill_ms":        0.0,
            "streaming_ms":      0.0,
            "ttft_ms":           _r(total_ms),
            "total_ms":          _r(total_ms),
        })
        _append(rec)
        return rec


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    return max(0.0, b - a)


def _r(v: float) -> float:
    return round(v, 2)

