"""LLM Client for Real Groq and Stand-in Model Service with streaming and retries."""
import time
import json
import random
import asyncio
from typing import AsyncGenerator, Dict, Any, Tuple, Optional, List
import httpx

from relay.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, STANDIN_BASE_URL
from relay.limiter import count_tokens

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
        model: str = GROQ_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 512,
        request_timeout: float = 10.0
    ) -> Tuple[AsyncGenerator[str, None], Dict[str, Any]]:
        """
        Sends chat completion request with streaming.
        Returns (token_stream_generator, metrics_dict)
        where metrics_dict is populated as the stream is consumed.
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
            t_start = time.perf_counter()
            first_token_seen = False

            while retries <= self.max_retries:
                try:
                    async with httpx.AsyncClient(timeout=request_timeout) as client:
                        req_t0 = time.perf_counter()
                        async with client.stream("POST", url, json=payload, headers=headers) as response:
                            metrics["headers"] = dict(response.headers)
                            
                            if response.status_code == 429:
                                metrics["status"] = "rate_limited"
                                retry_after = float(response.headers.get("retry-after", 1.0))
                                retries += 1
                                metrics["retries"] = retries
                                if retries > self.max_retries:
                                    yield f"[Rate limited: 429. Retry after {retry_after}s]"
                                    return
                                # Jittered backoff
                                sleep_time = retry_after + random.uniform(0.1, 0.5)
                                await asyncio.sleep(sleep_time)
                                continue

                            if response.status_code != 200:
                                body = await response.aread()
                                metrics["status"] = "error"
                                metrics["error_msg"] = f"HTTP {response.status_code}: {body.decode(errors='ignore')}"
                                retries += 1
                                metrics["retries"] = retries
                                if retries > self.max_retries:
                                    yield f"[Error: {metrics['error_msg']}]"
                                    return
                                await asyncio.sleep(0.5 * (2 ** retries) + random.uniform(0.05, 0.2))
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
                                        # Check Groq x_groq usage if present
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
                                                    metrics["first_token_ms"] = (time.perf_counter() - t_start) * 1000.0
                                                out_tokens += 1
                                                yield content
                                    except json.JSONDecodeError:
                                        continue

                            t_end = time.perf_counter()
                            metrics["total_ms"] = (t_end - t_start) * 1000.0
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
                    if retries > self.max_retries:
                        yield f"[Request timed out: {str(e)}]"
                        return
                    await asyncio.sleep(0.5 * (2 ** retries) + random.uniform(0.1, 0.3))

            metrics["total_ms"] = (time.perf_counter() - t_start) * 1000.0

        return _generator(), metrics
