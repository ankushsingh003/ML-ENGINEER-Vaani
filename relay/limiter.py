"""
relay/limiter.py
----------------
Token-aware async rate limiter.

Design decisions:
- Uses threading.Lock (not asyncio.Lock) for data mutations so the
  module-level _State object is safe across multiple asyncio.run() calls.
- Waiting is done with asyncio.sleep() so the event loop stays unblocked.
- acquire_concurrency uses a simple asyncio.Semaphore created lazily per
  event loop via get_event_loop() context.
"""

import asyncio
import time
import threading
import logging
from collections import deque
from typing import Literal, Tuple

import tiktoken

from relay.config import (
    RPM_LIMIT, TPM_LIMIT, RPD_LIMIT, TPD_LIMIT, MAX_CONCURRENCY
)

log = logging.getLogger(__name__)

try:
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return max(1, len(_enc.encode(text)))
except Exception:
    def count_tokens(text: str) -> int:          # noqa: F811
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

class _State:
    def __init__(self) -> None:
        self._lock = threading.Lock()           # plain threading lock — safe across loops

        # Sliding-window deques: (monotonic_s, value)
        self._req_win: deque = deque()
        self._tok_win: deque = deque()

        # Daily counters
        self._day_requests: int = 0
        self._day_tokens:   int = 0
        self._day_date:     str = ""

        # 429 cooldown (monotonic seconds)
        self._cooldown_until: float = 0.0

        # Concurrency counter (semaphore-like, guarded by _lock)
        self._inflight: int = 0

    # ------------------------------------------------------------------
    # Sliding-window helpers — call under self._lock
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        cutoff = time.monotonic() - 60.0
        while self._req_win and self._req_win[0][0] < cutoff:
            self._req_win.popleft()
        while self._tok_win and self._tok_win[0][0] < cutoff:
            self._tok_win.popleft()

    def _curr_rpm(self) -> int:
        return len(self._req_win)

    def _curr_tpm(self) -> int:
        return sum(t for _, t in self._tok_win)

    def _seconds_until_rpm_slot(self) -> float:
        if len(self._req_win) < RPM_LIMIT:
            return 0.0
        return max(0.0, self._req_win[0][0] + 60.0 - time.monotonic())

    def _seconds_until_tpm_slot(self, tokens: int) -> float:
        if self._curr_tpm() + tokens <= TPM_LIMIT:
            return 0.0
        return max(0.0, self._tok_win[0][0] + 60.0 - time.monotonic())

    def _reset_daily_if_needed(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self._day_date:
            self._day_date = today
            self._day_requests = 0
            self._day_tokens = 0

    def _daily_exhausted(self, tokens: int) -> bool:
        return (self._day_requests >= RPD_LIMIT or
                self._day_tokens + tokens > TPD_LIMIT)

    def _reserve(self, tokens: int) -> None:
        now = time.monotonic()
        self._req_win.append((now, 1))
        self._tok_win.append((now, tokens))
        self._day_requests += 1
        self._day_tokens += tokens

    def adjust_actual_tokens(self, estimated: int,
                             actual_in: int, actual_out: int) -> None:
        diff = (actual_in + actual_out) - estimated
        if diff == 0 or not self._tok_win:
            return
        with self._lock:
            t, tok = self._tok_win[-1]
            self._tok_win[-1] = (t, max(0, tok + diff))
            self._day_tokens = max(0, self._day_tokens + diff)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "curr_rpm":     self._curr_rpm(),
                "rpm_limit":    RPM_LIMIT,
                "curr_tpm":     self._curr_tpm(),
                "tpm_limit":    TPM_LIMIT,
                "day_requests": self._day_requests,
                "rpd_limit":    RPD_LIMIT,
                "day_tokens":   self._day_tokens,
                "tpd_limit":    TPD_LIMIT,
                "inflight":     self._inflight,
                "max_inflight": MAX_CONCURRENCY,
                "cooldown_s":   max(0.0, self._cooldown_until - time.monotonic()),
            }

    # ------------------------------------------------------------------
    # Concurrency gate (threading-based, so asyncio.sleep() can wait)
    # ------------------------------------------------------------------

    def try_acquire_concurrency(self) -> bool:
        with self._lock:
            if self._inflight >= MAX_CONCURRENCY:
                return False
            self._inflight += 1
            return True

    def release_concurrency(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)


_state = _State()


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def acquire(
    estimated_tokens: int,
    deadline: float,          # absolute monotonic time
) -> Tuple[bool, str]:
    """
    Waits (via asyncio.sleep) until capacity is available or deadline expires.

    Returns:
        (True,  "NORMAL"|"THROTTLED")   – slot reserved; call release_tokens()
        (False, "DAILY_EXHAUSTED")      – unrecoverable today
        (False, "RATE_LIMITED")         – deadline passed before slot opened
    """
    # Quick daily check (non-blocking)
    with _state._lock:
        _state._reset_daily_if_needed()
        if _state._daily_exhausted(estimated_tokens):
            return False, "DAILY_EXHAUSTED"

    while True:
        now = time.monotonic()
        if now >= deadline:
            return False, "RATE_LIMITED"

        # Respect 429 global cooldown
        cooldown_left = _state._cooldown_until - now
        if cooldown_left > 0:
            sleep_s = min(cooldown_left, deadline - now)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            continue

        with _state._lock:
            _state._prune()

            if _state._daily_exhausted(estimated_tokens):
                return False, "DAILY_EXHAUSTED"

            wait = max(
                _state._seconds_until_rpm_slot(),
                _state._seconds_until_tpm_slot(estimated_tokens),
            )

            if wait <= 0:
                _state._reserve(estimated_tokens)
                rpm_ratio = _state._curr_rpm() / RPM_LIMIT
                tpm_ratio = _state._curr_tpm() / TPM_LIMIT
                mode = "THROTTLED" if max(rpm_ratio, tpm_ratio) > 0.75 else "NORMAL"
                return True, mode

        sleep_s = min(wait + 0.05, deadline - time.monotonic())
        if sleep_s <= 0:
            return False, "RATE_LIMITED"
        await asyncio.sleep(sleep_s)


async def acquire_concurrency(deadline: float) -> bool:
    """
    Waits until a concurrency slot opens or deadline expires.
    Uses polling + asyncio.sleep (no asyncio primitives stored globally).
    """
    while True:
        if _state.try_acquire_concurrency():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.05, remaining))


def release_concurrency() -> None:
    _state.release_concurrency()


def release_tokens(estimated: int, actual_in: int, actual_out: int) -> None:
    _state.adjust_actual_tokens(estimated, actual_in, actual_out)


def set_cooldown(retry_after_seconds: float) -> None:
    _state._cooldown_until = time.monotonic() + retry_after_seconds
    log.warning("429 cooldown set for %.1fs", retry_after_seconds)


def snapshot() -> dict:
    return _state.snapshot()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def force_daily_exhausted() -> None:
    with _state._lock:
        _state._day_requests = RPD_LIMIT


def reset_for_test() -> None:
    with _state._lock:
        _state._req_win.clear()
        _state._tok_win.clear()
        _state._day_requests = 0
        _state._day_tokens = 0
        _state._cooldown_until = 0.0
        _state._inflight = 0
