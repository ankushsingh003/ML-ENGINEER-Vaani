"""Stand-in Model Service: Calibrated mock of Groq API with latency, streaming, failure, and 429s."""
import time
import json
import random
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Stand-in Groq Model Server")

# Stand-in Knobs (can be dynamically reconfigured)
CONFIG = {
    "base_ttft_ms": 220.0,       # Base time to first token
    "token_delay_ms": 15.0,      # Milliseconds per output token (~65 tokens/sec)
    "concurrency_slowdown": 1.15,# Multiplier per concurrent request in flight
    "failure_rate": 0.0,         # Probability of random 500 failure
    "rate_limit_rpm": 30,        # Max requests per minute
    "current_inflight": 0,
    "limit_halved": False,       # When True, rpm limit halved to 15
}

# In-memory sliding window for rate-limiting
REQ_TIMESTAMPS = []

@app.post("/config")
async def update_config(payload: Dict[str, Any]):
    """Allow load generator to tune stand-in parameters during scenarios."""
    CONFIG.update(payload)
    return {"status": "updated", "config": CONFIG}

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", True)
    
    # 1. Check failure injection
    if CONFIG["failure_rate"] > 0 and random.random() < CONFIG["failure_rate"]:
        raise HTTPException(status_code=500, detail="Stand-in simulated internal server error")

    # 2. Check RPM limit
    now = time.time()
    cutoff = now - 60.0
    global REQ_TIMESTAMPS
    REQ_TIMESTAMPS = [t for t in REQ_TIMESTAMPS if t > cutoff]
    
    effective_rpm = CONFIG["rate_limit_rpm"] // 2 if CONFIG["limit_halved"] else CONFIG["rate_limit_rpm"]
    if len(REQ_TIMESTAMPS) >= effective_rpm:
        # Return 429 with retry-after header
        retry_after = max(1, int(60.0 - (now - REQ_TIMESTAMPS[0])))
        return Response(
            content=json.dumps({"error": {"message": "Rate limit reached for requests per minute", "type": "tokens"}}),
            status_code=429,
            media_type="application/json",
            headers={"retry-after": str(retry_after), "x-ratelimit-remaining-requests": "0"}
        )

    REQ_TIMESTAMPS.append(now)

    # 3. Simulate load-dependent latency
    CONFIG["current_inflight"] += 1
    inflight = CONFIG["current_inflight"]
    
    # Non-linear load degradation
    load_factor = (CONFIG["concurrency_slowdown"] ** max(0, inflight - 1))
    ttft = (CONFIG["base_ttft_ms"] * load_factor) / 1000.0
    # Add small normal jitter
    ttft += random.uniform(0.01, 0.05)

    user_query = messages[-1]["content"] if messages else ""
    user_query_lower = user_query.lower()
    
    # Check for unanswerable questions
    unanswerable_triggers = ["dollar cost", "electricity", "iphone 16", "battery", "quantum", "tpu v5e"]
    if any(t in user_query_lower for t in unanswerable_triggers):
        mock_reply_str = "The papers do not say. The indexed papers do not contain measurements or data regarding this topic."
    elif "2410.00037v1" in user_query:
        mock_reply_str = (
            "According to the Moshi paper [2410.00037v1, Section 2 & 3], Mimi is a neural audio codec operating at 12.5 Hz framerate "
            "and 1.1 kbps bitrate with 8 codebooks for speech tokens, enabling low latency multi-stream interaction."
        )
    elif "2507.07955v2" in user_query:
        mock_reply_str = (
            "According to Dynamic Chunking [2507.07955v2, Section 1 & 3], H-Net learns variable-length boundary decisions "
            "dynamically based on sequence semantics to eliminate arbitrary chunk boundaries."
        )
    elif "1706.03762v7" in user_query:
        mock_reply_str = (
            "According to Attention Is All You Need [1706.03762v7, Section 3 & 5.4], the base model has d_model=512, d_ff=2048, "
            "h=8 heads, and uses label smoothing of \u03f5_ls = 0.1 which improves BLEU score."
        )
    else:
        mock_reply_str = (
            "Based on the provided library passages, the model architecture achieves strong empirical results. "
            "[Citation: 1706.03762v7, Section 3]"
        )

    # Tokenize mock reply into word pieces
    mock_tokens = [tok if i == 0 else f" {tok}" for i, tok in enumerate(mock_reply_str.split())]

    async def sse_generator():
        try:
            # TTFT delay
            await asyncio.sleep(ttft)
            
            for i, token in enumerate(mock_tokens):
                chunk_data = {
                    "id": f"chatcmpl-mock-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "standin-llama-3.1-8b",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                # Per-token streaming delay
                await asyncio.sleep(CONFIG["token_delay_ms"] / 1000.0)

            # Final DONE chunk with mock usage
            done_chunk = {
                "id": "chatcmpl-mock-done",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "x_groq": {
                    "usage": {
                        "prompt_tokens": len(user_query.split()) + 40,
                        "completion_tokens": len(mock_tokens),
                        "total_tokens": len(user_query.split()) + 40 + len(mock_tokens)
                    }
                }
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            CONFIG["current_inflight"] = max(0, CONFIG["current_inflight"] - 1)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("standin.server:app", host="127.0.0.1", port=8000, log_level="warning")
