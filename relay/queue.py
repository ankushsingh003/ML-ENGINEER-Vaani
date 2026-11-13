"""
relay/queue.py
--------------
True FIFO admission queue with bounded capacity, worker pool, deadline enforcement,
and dynamic peak queue tracking.
"""

import asyncio
import time
import logging
from typing import Callable, Awaitable, Any, Optional

from relay.config import MAX_QUEUE_SIZE, REQUEST_TIMEOUT_SEC, MAX_CONCURRENCY
from relay.trace import RequestTrace

log = logging.getLogger(__name__)

class QueueItem:
    def __init__(
        self,
        task_fn: Callable[[], Awaitable[Any]],
        trace: RequestTrace,
        deadline_sec: float,
        future: asyncio.Future
    ):
        self.task_fn = task_fn
        self.trace = trace
        self.deadline_sec = deadline_sec
        self.t_enqueued = time.monotonic()
        self.future = future

class AdmissionController:
    def __init__(self, max_queue_size: int = MAX_QUEUE_SIZE, num_workers: int = MAX_CONCURRENCY):
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.queue: Optional[asyncio.Queue] = None
        self.workers: list[asyncio.Task] = []
        self.peak_queue_length = 0
        self.total_enqueued = 0
        self.total_shed = 0
        self.total_timed_out = 0
        self._running = False

    def _ensure_started(self):
        if not self._running or self.queue is None:
            self.queue = asyncio.Queue(maxsize=self.max_queue_size)
            self._running = True
            self.workers = [
                asyncio.create_task(self._worker_loop(i))
                for i in range(self.num_workers)
            ]

    async def _worker_loop(self, worker_id: int):
        while self._running:
            try:
                item: QueueItem = await self.queue.get()
            except asyncio.CancelledError:
                break

            now = time.monotonic()
            wait_time = now - item.t_enqueued

            if wait_time >= item.deadline_sec:
                self.total_timed_out += 1
                rec = item.trace.timed_out(retrieval_setting="n/a")
                result = {
                    "status": "timeout",
                    "answer": "Request timed out waiting in admission queue.",
                    "first_token_ms": rec["ttft_ms"],
                    "total_ms": rec["total_ms"],
                    "retries": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "answer_grade": "wrong"
                }
                if not item.future.done():
                    item.future.set_result(result)
                self.queue.task_done()
                continue

            item.trace.mark_dequeued()

            remaining_deadline = item.deadline_sec - wait_time
            try:
                res = await asyncio.wait_for(item.task_fn(), timeout=remaining_deadline)
                if not item.future.done():
                    item.future.set_result(res)
            except asyncio.TimeoutError:
                self.total_timed_out += 1
                rec = item.trace.timed_out(retrieval_setting="n/a")
                res = {
                    "status": "timeout",
                    "answer": "Request execution exceeded deadline.",
                    "first_token_ms": rec["ttft_ms"],
                    "total_ms": rec["total_ms"],
                    "retries": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "answer_grade": "wrong"
                }
                if not item.future.done():
                    item.future.set_result(res)
            except Exception as e:
                log.exception("Worker %d encountered error: %s", worker_id, e)
                if not item.future.done():
                    item.future.set_exception(e)
            finally:
                self.queue.task_done()

    async def submit(
        self,
        task_fn: Callable[[], Awaitable[Any]],
        trace: RequestTrace,
        deadline_sec: float = REQUEST_TIMEOUT_SEC
    ) -> Any:
        self._ensure_started()
        trace.mark_enqueued()

        current_qsize = self.queue.qsize()
        trace.queue_len_at_enqueue = current_qsize

        if current_qsize >= self.max_queue_size:
            self.total_shed += 1
            rec = trace.shed()
            return {
                "status": "rate_limited",
                "answer": "Service overloaded. Request shed at admission.",
                "first_token_ms": rec["ttft_ms"],
                "total_ms": rec["total_ms"],
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "answer_grade": "wrong"
            }

        new_qsize = current_qsize + 1
        if new_qsize > self.peak_queue_length:
            self.peak_queue_length = new_qsize

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = QueueItem(task_fn=task_fn, trace=trace, deadline_sec=deadline_sec, future=future)

        await self.queue.put(item)
        self.total_enqueued += 1

        return await future

    async def shutdown(self):
        self._running = False
        for w in self.workers:
            w.cancel()
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

admission_controller = AdmissionController()
