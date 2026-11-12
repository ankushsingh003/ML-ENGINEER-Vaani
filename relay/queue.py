"""Admission Control: Bounded FIFO queue with deadline expiration and load shedding."""
import time
import asyncio
from typing import Optional, Callable, Awaitable, Any

from relay.config import MAX_QUEUE_SIZE, REQUEST_TIMEOUT_SEC

class AdmissionController:
    def __init__(self, max_queue_size: int = MAX_QUEUE_SIZE, default_timeout: float = REQUEST_TIMEOUT_SEC):
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.peak_queue_length = 0
        self.total_admitted = 0
        self.total_shed = 0
        self.total_timed_out = 0

    async def enqueue_and_execute(
        self,
        task_func: Callable[[], Awaitable[Any]],
        deadline_sec: Optional[float] = None
    ) -> Any:
        timeout = deadline_sec or self.default_timeout
        t_enqueued = time.perf_counter()

        # 1. Admission Check (Load Shedding)
        if self.queue.qsize() >= self.max_queue_size:
            self.total_shed += 1
            return {
                "status": "rate_limited",
                "content": "Service overloaded. Request shed at admission.",
                "first_token_ms": 0.0,
                "total_ms": 0.0,
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "answer_grade": "refused_correctly"
            }

        # Track peak queue length
        qlen = self.queue.qsize() + 1
        if qlen > self.peak_queue_length:
            self.peak_queue_length = qlen

        # Place ticket in queue
        await self.queue.put(1)
        self.total_admitted += 1

        try:
            # Check deadline before starting execution
            time_waited = time.perf_counter() - t_enqueued
            remaining_time = timeout - time_waited
            if remaining_time <= 0:
                self.total_timed_out += 1
                return {
                    "status": "timeout",
                    "content": "Request expired waiting in admission queue.",
                    "first_token_ms": 0.0,
                    "total_ms": time_waited * 1000.0,
                    "retries": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "answer_grade": "refused_correctly"
                }

            # Execute under remaining deadline
            result = await asyncio.wait_for(task_func(), timeout=remaining_time)
            return result

        except asyncio.TimeoutError:
            self.total_timed_out += 1
            total_elapsed = (time.perf_counter() - t_enqueued) * 1000.0
            return {
                "status": "timeout",
                "content": "Request timed out during execution.",
                "first_token_ms": total_elapsed,
                "total_ms": total_elapsed,
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "answer_grade": "refused_correctly"
            }
        finally:
            self.queue.get_nowait()
            self.queue.task_done()

# Global admission controller instance
admission_controller = AdmissionController()

