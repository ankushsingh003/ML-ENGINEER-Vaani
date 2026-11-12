"""Token-aware rate limiter, concurrency controller, and degradation ladder."""
import time
import asyncio
import threading
from typing import Tuple, Dict, Any
import tiktoken

from relay.config import RPM_LIMIT, TPM_LIMIT, RPD_LIMIT, TPD_LIMIT, MAX_CONCURRENCY

try:
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

class RateLimiter:
    def __init__(
        self,
        rpm_limit: int = RPM_LIMIT,
        tpm_limit: int = TPM_LIMIT,
        rpd_limit: int = RPD_LIMIT,
        tpd_limit: int = TPD_LIMIT,
        max_concurrency: int = MAX_CONCURRENCY
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.tpd_limit = tpd_limit
        self.max_concurrency = max_concurrency

        self._lock = threading.Lock()
        
        # Sliding windows (timestamps)
        self.req_window = []  # list of float timestamps
        self.tok_window = []  # list of (timestamp, token_count)
        
        # Daily counters
        self.daily_requests = 0
        self.daily_tokens = 0
        
        # Active in-flight concurrency
        self.in_flight = 0

    def get_status(self) -> Dict[str, Any]:
        """Returns snapshot of current capacity and utilization."""
        now = time.time()
        with self._lock:
            self._prune_windows(now)
            curr_rpm = len(self.req_window)
            curr_tpm = sum(t for _, t in self.tok_window)
            return {
                "curr_rpm": curr_rpm,
                "rpm_limit": self.rpm_limit,
                "curr_tpm": curr_tpm,
                "tpm_limit": self.tpm_limit,
                "daily_requests": self.daily_requests,
                "rpd_limit": self.rpd_limit,
                "daily_tokens": self.daily_tokens,
                "tpd_limit": self.tpd_limit,
                "in_flight": self.in_flight,
                "max_concurrency": self.max_concurrency
            }

    def _prune_windows(self, now: float):
        cutoff = now - 60.0
        self.req_window = [t for t in self.req_window if t > cutoff]
        self.tok_window = [(t, tok) for t, tok in self.tok_window if t > cutoff]

    def acquire(self, estimated_tokens: int = 500) -> Tuple[bool, str, float]:
        """
        Check if request can proceed.
        Returns: (allowed: bool, degradation_mode: str, wait_seconds: float)
        degradation_mode in ['NORMAL', 'THROTTLED', 'DAILY_EXHAUSTED', 'RATE_LIMITED']
        """
        now = time.time()
        with self._lock:
            self._prune_windows(now)
            
            # 1. Daily Budget Check
            if self.daily_requests >= self.rpd_limit or (self.daily_tokens + estimated_tokens) >= self.tpd_limit:
                return False, "DAILY_EXHAUSTED", 0.0

            # 2. Concurrency Check
            if self.in_flight >= self.max_concurrency:
                return False, "RATE_LIMITED", 0.5

            curr_rpm = len(self.req_window)
            curr_tpm = sum(t for _, t in self.tok_window)

            # 3. Minute Limit Check
            if curr_rpm >= self.rpm_limit or (curr_tpm + estimated_tokens) > self.tpm_limit:
                # Calculate time until earliest slot opens
                oldest_req = self.req_window[0] if self.req_window else now
                oldest_tok = self.tok_window[0][0] if self.tok_window else now
                wait_time = max(0.1, 60.0 - (now - min(oldest_req, oldest_tok)))
                return False, "RATE_LIMITED", wait_time

            # 4. Check for Throttle threshold (>75% capacity)
            mode = "NORMAL"
            if curr_rpm > (0.75 * self.rpm_limit) or (curr_tpm + estimated_tokens) > (0.75 * self.tpm_limit):
                mode = "THROTTLED"

            # Reserve slot
            self.in_flight += 1
            self.req_window.append(now)
            self.tok_window.append((now, estimated_tokens))
            self.daily_requests += 1
            self.daily_tokens += estimated_tokens

            return True, mode, 0.0

    def release(self, estimated_tokens: int, actual_tokens_in: int, actual_tokens_out: int):
        """Release concurrency slot and adjust actual tokens consumed."""
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            actual_total = actual_tokens_in + actual_tokens_out
            diff = actual_total - estimated_tokens
            if diff != 0:
                self.daily_tokens += diff
                now = time.time()
                # adjust recent entry
                if self.tok_window:
                    last_t, last_tok = self.tok_window[-1]
                    self.tok_window[-1] = (last_t, max(0, last_tok + diff))

# Global limiter instance
limiter = RateLimiter()
