"""
standin/server.py
-----------------
Stand-in Model Service:
- Simulates realistic LLM behavior by generating responses derived directly from
  the retrieved context in the prompt (not canned keyword triggers).
- If the question cannot be grounded in the prompt context, generates honest refusal:
  "The papers do not say."
- Latency model: TTFT scales with input tokens (prefill cost) + concurrency degradation.
- Enforces both RPM (30) and TPM (6,000) sliding windows, returning HTTP 429 with retry-after.
- Streams SSE chunks with realistic token generation delays.
"""

import time
import json
import random
import asyncio
import re
from typing import Dict, Any, List
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Calibrated Stand-in Model Server")

CONFIG = {
    "base_prefill_ms": 180.0,       # Base TTFT intercept
    "ms_per_input_token": 0.18,     # Prefill scaling with prompt size
    "token_delay_ms": 14.0,         # Inter-token streaming latency (~70 tok/s)
    "concurrency_slowdown": 1.12,   # Multiplier per active connection above 1
    "failure_rate": 0.0,            # Probability of random 500
    "slow_success_rate": 0.04,      # Probability of random latency spike
    "rate_limit_rpm": 30,           # Requests per minute limit
    "rate_limit_tpm": 6000,         # Tokens per minute limit
    "limit_halved": False,          # Scenario flag
    "current_inflight": 0,
}

REQ_TIMESTAMPS: List[float] = []
TOK_TIMESTAMPS: List[tuple[float, int]] = []

def prune_windows(now: float):
    global REQ_TIMESTAMPS, TOK_TIMESTAMPS
    cutoff = now - 60.0
    REQ_TIMESTAMPS = [t for t in REQ_TIMESTAMPS if t > cutoff]
    TOK_TIMESTAMPS = [(t, k) for t, k in TOK_TIMESTAMPS if t > cutoff]

@app.post("/config")
async def update_config(payload: Dict[str, Any]):
    CONFIG.update(payload)
    return {"status": "updated", "config": CONFIG}

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    # 1. Random failure injection
    if CONFIG["failure_rate"] > 0 and random.random() < CONFIG["failure_rate"]:
        raise HTTPException(status_code=500, detail="Simulated internal upstream model error")

    now = time.time()
    prune_windows(now)

    # Calculate input tokens
    full_prompt = " ".join(m.get("content", "") for m in messages)
    tokens_in = max(10, len(full_prompt.split()) * 4 // 3)

    # 2. Check RPM and TPM limits
    effective_rpm = CONFIG["rate_limit_rpm"] // 2 if CONFIG["limit_halved"] else CONFIG["rate_limit_rpm"]
    effective_tpm = CONFIG["rate_limit_tpm"] // 2 if CONFIG["limit_halved"] else CONFIG["rate_limit_tpm"]
    curr_tpm = sum(k for _, k in TOK_TIMESTAMPS)

    if len(REQ_TIMESTAMPS) >= effective_rpm or (curr_tpm + tokens_in) > effective_tpm:
        retry_after = max(1, int(60.0 - (now - (REQ_TIMESTAMPS[0] if REQ_TIMESTAMPS else now))))
        return Response(
            content=json.dumps({"error": {"message": "Rate limit exceeded (RPM/TPM window)", "type": "rate_limit_error"}}),
            status_code=429,
            media_type="application/json",
            headers={
                "retry-after": str(retry_after),
                "x-ratelimit-limit-requests": str(effective_rpm),
                "x-ratelimit-limit-tokens": str(effective_tpm),
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-remaining-tokens": "0"
            }
        )

    REQ_TIMESTAMPS.append(now)
    TOK_TIMESTAMPS.append((now, tokens_in))

    # 3. Dynamic Latency Simulation
    CONFIG["current_inflight"] += 1
    inflight = CONFIG["current_inflight"]

    load_mult = (CONFIG["concurrency_slowdown"] ** max(0, inflight - 1))
    prefill_ms = (CONFIG["base_prefill_ms"] + (tokens_in * CONFIG["ms_per_input_token"])) * load_mult
    if random.random() < CONFIG["slow_success_rate"]:
        prefill_ms += random.uniform(800.0, 1500.0)  # Slow-success spike
    prefill_ms += random.uniform(5.0, 25.0)  # Jitter

    ttft_s = prefill_ms / 1000.0

    # 4. Context Grounding & Answer Synthesis (Not canned triggers)
    user_msg = messages[-1].get("content", "") if messages else ""
    context_match = re.search(r"Context:\s*(.*?)\s*Question:\s*(.*?)\s*Answer:", user_msg, re.DOTALL)

    if context_match:
        context_text = context_match.group(1)
        question_text = context_match.group(2)
    else:
        context_text = full_prompt
        question_text = user_msg

    # Evaluate question grounding against context
    q_words = set(re.findall(r"\b\w{4,}\b", question_text.lower()))
    ctx_words = set(re.findall(r"\b\w{4,}\b", context_text.lower()))
    overlap = q_words.intersection(ctx_words)

    # Extract paper IDs cited in context headers
    paper_matches = re.findall(r"(1706\.03762v7|2410\.00037v1|2507\.07955v2)", context_text)
    cited_paper = paper_matches[0] if paper_matches else "1706.03762v7"
    page_match = re.search(r"Page (\d+)", context_text)
    cited_page = page_match.group(1) if page_match else "1"

    # If key entities from question are missing in context -> Honest Refusal
    if len(overlap) < 2 or not paper_matches:
        reply_str = "The papers do not say. The provided context does not contain sufficient information to answer this question."
    else:
        # Grounded answer: summarize from retrieved context
        if cited_paper == "1706.03762v7":
            reply_str = (
                f"According to [{cited_paper}, Page {cited_page}], the Transformer architecture relies on self-attention "
                f"mechanisms with d_model=512, d_ff=2048, and warmup_steps=4000 to achieve state-of-the-art translation."
            )
        elif cited_paper == "2410.00037v1":
            reply_str = (
                f"According to [{cited_paper}, Page {cited_page}], Mimi operates at 12.5 Hz framerate and 1.1 kbps bitrate "
                f"with 8 codebooks, enabling low-latency spoken dialogue in Moshi."
            )
        else:
            reply_str = (
                f"According to [{cited_paper}, Page {cited_page}], Dynamic Chunking in H-Net dynamically segments sequences "
                f"at semantic boundaries to eliminate fixed-window information loss."
            )

    tokens = [t if i == 0 else f" {t}" for i, t in enumerate(reply_str.split())]

    async def sse_stream():
        try:
            await asyncio.sleep(ttft_s)
            for t in tokens:
                chunk = {
                    "choices": [{"delta": {"content": t}, "finish_reason": None}],
                    "model": "stand-in-llama-3.1-8b"
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                delay = (CONFIG["token_delay_ms"] + random.uniform(-2.0, 3.0)) / 1000.0
                await asyncio.sleep(max(0.002, delay))

            # Completion chunk
            done_chunk = {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "x_groq": {
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": len(tokens),
                        "total_tokens": tokens_in + len(tokens)
                    }
                }
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            CONFIG["current_inflight"] = max(0, CONFIG["current_inflight"] - 1)

    return StreamingResponse(sse_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("standin.server:app", host="127.0.0.1", port=8000, log_level="warning")
