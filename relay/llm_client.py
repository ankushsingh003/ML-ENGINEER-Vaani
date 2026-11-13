"""
relay/llm_client.py
-------------------
LLM Client for Real Groq and Stand-in Model Service with:
- Streaming token consumption.
- TTFT and monotonic latency instrumentation.
- Provider retry handling strictly routed through limiter & concurrency controls.
- Automatic 429 global cooldown propagation.
"""

import time
import json
import random
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Tuple, Optional, List
import httpx

from relay.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, STANDIN_BASE_URL
from relay.limiter import (
    count_tokens,
    acquire,
    acquire_concurrency,
    release_concurrency,
    release_tokens,
    set_cooldown
)
from relay.trace import RequestTrace

log = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, source: str = "stand-in", max_retries: int = 2, base_url: Optional[str] = None):
        self.source = source  # "real" or "stand-in"
        self.max_retries = max_retries
        if base_url:
            self.base_url = base_url
        elif source == "real":
            self.base_url = GROQ_BASE_URL
        else:
            self.base_url = STANDIN_BASE_URL

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        trace: Optional[RequestTrace] = None,
        model: str = GROQ_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 512,
        deadline: Optional[float] = None
    ) -> Tuple[AsyncGenerator[str, None], Dict[str, Any]]:
        """
        Sends chat completion request with streaming.
        Returns (token_stream_generator, metrics_dict).
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.source == "real":
            headers["Authorization"] = f"Bearer {GROQ_API_KEY}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        # Estimate input tokens
        input_text = " ".join(m["content"] for m in messages)
        est_tokens_in = count_tokens(input_text)

        metrics = {
            "first_token_ms": 0.0,
            "total_ms": 0.0,
            "tokens_in": est_tokens_in,
            "tokens_out": 0,
            "status": "ok",
            "retries": 0,
            "error_msg": None,
            "headers": {}
        }

        async def _generator():
            retries = 0
            t_client_start = time.perf_counter()
            first_token_seen = False
            effective_deadline = deadline or (time.monotonic() + 12.0)

            while retries <= self.max_retries:
                # 1. Acquire concurrency slot
                concurrency_acquired = await acquire_concurrency(effective_deadline)
                if not concurrency_acquired:
                    metrics["status"] = "timeout"
                    metrics["error_msg"] = "Could not acquire concurrency slot before deadline."
                    yield "[Request timed out waiting for execution slot]"
                    return

                try:
                    remaining_timeout = max(0.5, effective_deadline - time.monotonic())
                    if trace:
                        trace.mark_model_sent()

                    async with httpx.AsyncClient(timeout=remaining_timeout) as client:
                        async with client.stream("POST", url, json=payload, headers=headers) as response:
                            metrics["headers"] = dict(response.headers)

                            # Handle 429 Rate Limit
                            if response.status_code == 429:
                                metrics["status"] = "rate_limited"
                                retry_after = float(response.headers.get("retry-after", 2.0))
                                set_cooldown(retry_after)
                                retries += 1
                                metrics["retries"] = retries
                                release_concurrency()

                                if retries > self.max_retries or time.monotonic() >= effective_deadline:
                                    yield f"[Rate limited: 429. Retry after {retry_after}s]"
                                    return

                                # Jittered backoff
                                sleep_time = min(retry_after + random.uniform(0.1, 0.4), effective_deadline - time.monotonic())
                                if sleep_time > 0:
                                    await asyncio.sleep(sleep_time)

                                # Re-acquire token allowance from limiter for the retry
                                ok, mode = await acquire(estimated_tokens=est_tokens_in, deadline=effective_deadline)
                                if not ok:
                                    metrics["status"] = "rate_limited"
                                    yield "[Rate limit backoff expired without available token capacity]"
                                    return
                                continue

                            # Handle non-200 errors
                            if response.status_code != 200:
                                body = await response.aread()
                                metrics["status"] = "error"
                                metrics["error_msg"] = f"HTTP {response.status_code}: {body.decode(errors='ignore')}"
                                retries += 1
                                metrics["retries"] = retries
                                release_concurrency()

                                if retries > self.max_retries or time.monotonic() >= effective_deadline:
                                    yield f"[Error: {metrics['error_msg']}]"
                                    return

                                sleep_time = min(0.5 * (2 ** retries) + random.uniform(0.05, 0.2), effective_deadline - time.monotonic())
                                if sleep_time > 0:
                                    await asyncio.sleep(sleep_time)
                                continue

                            # Successful 200 Stream
                            metrics["status"] = "ok"
                            out_tokens = 0

                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        if "x_groq" in chunk and "usage" in chunk["x_groq"]:
                                            u = chunk["x_groq"]["usage"]
                                            metrics["tokens_in"] = u.get("prompt_tokens", metrics["tokens_in"])
                                            metrics["tokens_out"] = u.get("completion_tokens", metrics["tokens_out"])

                                        choices = chunk.get("choices", [])
                                        if choices and "delta" in choices[0]:
                                            delta = choices[0]["delta"]
                                            content = delta.get("content", "")
                                            if content:
                                                if not first_token_seen:
                                                    first_token_seen = True
                                                    if trace:
                                                        trace.mark_first_token()
                                                    metrics["first_token_ms"] = (time.perf_counter() - t_client_start) * 1000.0
                                                out_tokens += 1
                                                yield content
                                    except json.JSONDecodeError:
                                        continue

                            metrics["total_ms"] = (time.perf_counter() - t_client_start) * 1000.0
                            if not first_token_seen:
                                metrics["first_token_ms"] = metrics["total_ms"]
                            if metrics["tokens_out"] == 0:
                                metrics["tokens_out"] = max(1, out_tokens)
                            return

                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    retries += 1
                    metrics["retries"] = retries
                    metrics["status"] = "timeout" if isinstance(e, httpx.TimeoutException) else "error"
                    metrics["error_msg"] = str(e)
                    if retries > self.max_retries or time.monotonic() >= effective_deadline:
                        yield f"[Request connection failed: {str(e)}]"
                        return
                    await asyncio.sleep(min(0.5 * (2 ** retries) + random.uniform(0.1, 0.3), max(0.1, effective_deadline - time.monotonic())))
                finally:
                    release_concurrency()

            metrics["total_ms"] = (time.perf_counter() - t_client_start) * 1000.0

        return _generator(), metrics
