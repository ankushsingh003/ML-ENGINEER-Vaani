"""
standin/server.py
-----------------
Stand-in Model Service.
"""

import time
import json
import random
import asyncio
import re
from typing import Dict, Any, List, Tuple
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Calibrated Stand-in Model Server")

CONFIG = {
    "base_prefill_ms": 180.0,
    "ms_per_input_token": 0.18,
    "token_delay_ms": 14.0,
    "concurrency_slowdown": 1.12,
    "failure_rate": 0.0,
    "slow_success_rate": 0.04,
    "rate_limit_rpm": 30,
    "rate_limit_tpm": 6000,
    "limit_halved": False,
    "current_inflight": 0,
}

REQ_TIMESTAMPS: List[float] = []
TOK_TIMESTAMPS: List[Tuple[float, int]] = []

def prune_windows(now: float):
    global REQ_TIMESTAMPS, TOK_TIMESTAMPS
    cutoff = now - 60.0
    REQ_TIMESTAMPS = [t for t in REQ_TIMESTAMPS if t > cutoff]
    TOK_TIMESTAMPS = [(t, k) for t, k in TOK_TIMESTAMPS if t > cutoff]

@app.post("/config")
async def update_config(payload: Dict[str, Any]):
    CONFIG.update(payload)
    return {"status": "updated", "config": CONFIG}

ANSWER_TEMPLATES = [
    (
        [r"label smooth", r"smoothing.*0\.1", r"epsilon.*ls"],
        "1706.03762v7", "8",
        "According to [1706.03762v7, Page 8], label smoothing of value 0.1 "
        "(epsilon_ls=0.1) was used. This hurts perplexity but improves accuracy and BLEU score."
    ),
    (
        [r"d_model|d_ff|dmodel|dff|dimension.*base|base.*model.*dimension"],
        "1706.03762v7", "3",
        "According to [1706.03762v7, Page 3], the Transformer base model uses "
        "d_model=512 and d_ff=2048, with 8 attention heads and h=8 parallel attention layers."
    ),
    (
        [r"optimizer|warmup|warm.up|adam|learning.rate.schedule"],
        "1706.03762v7", "8",
        "According to [1706.03762v7, Page 8], training used the Adam optimizer "
        "with beta1=0.9, beta2=0.98, epsilon=1e-9, and warmup_steps=4000. "
        "The learning rate increased linearly for the first 4000 steps then decreased."
    ),
    (
        [r"mimi.*framerate|mimi.*bitrate|framerate.*mimi|bitrate.*mimi|mimi.*12\.5|mimi.*1\.1"],
        "2410.00037v1", "4",
        "According to [2410.00037v1, Page 4], Mimi is Moshi's neural audio codec operating "
        "at 12.5 Hz framerate and 1.1 kbps bitrate, encoding audio into discrete tokens "
        "that the LM can process in real time."
    ),
    (
        [r"codebook|how many.*codebook|codebook.*mimi|residual.*quanti"],
        "2410.00037v1", "4",
        "According to [2410.00037v1, Page 4], Mimi uses 8 codebooks (residual vector "
        "quantization), producing 8 discrete tokens per 80ms audio frame at 12.5 Hz."
    ),
    (
        [r"latency.*moshi|moshi.*latency|end.to.end.*latency|160|200.*ms|theoretical.*latency"],
        "2410.00037v1", "2",
        "According to [2410.00037v1, Page 2], Moshi targets a theoretical latency of 160ms "
        "(audio chunk size) with a measured overall end-to-end latency of approximately 200ms "
        "in real deployments."
    ),
    (
        [r"dynamic chunk|h.net.*chunk|chunk.*h.net|variable.*length.*chunk|semantic.*boundar"],
        "2507.07955v2", "2",
        "According to [2507.07955v2, Page 2], Dynamic Chunking in H-Net segments input "
        "sequences at variable-length semantic boundaries rather than fixed-window positions, "
        "eliminating information loss at chunk edges in hierarchical sequence modeling."
    ),
    (
        [r"differentiab|gradient.*chunk|soft.*boundar|gumbel|backprop.*chunk"],
        "2507.07955v2", "4",
        "According to [2507.07955v2, Page 4], Dynamic Chunking maintains differentiability "
        "via soft boundary weights using a Gumbel-softmax relaxation, allowing gradients to "
        "flow through the boundary selection decision during training."
    ),
    (
        [r"benchmark|pg19|wikitext|perplexity.*h.net|flops|evaluat"],
        "2507.07955v2", "5",
        "According to [2507.07955v2, Page 5], H-Net evaluates on PG19 and WikiText-103 "
        "language modeling benchmarks, reporting perplexity improvements over fixed-window "
        "baselines, and measures FLOPs efficiency at equivalent parameter counts."
    ),
    (
        [r"chunking.*moshi.*h.net|moshi.*chunking.*h.net|compare.*chunk|acoustic.*chunk|audio.*frame.*chunk"],
        "2410.00037v1", "3",
        "According to [2410.00037v1, Page 3] and [2507.07955v2, Page 2]: In Moshi, 'chunking' "
        "refers to fixed 80ms acoustic audio frames at 12.5 Hz that are streamed to the codec. "
        "In H-Net, 'Dynamic Chunking' refers to variable-length semantic text sequence "
        "segmentation at linguistic boundaries — a fundamentally different level of abstraction."
    ),
    (
        [r"helium|language model.*moshi|7b|7 billion|parameter.*count"],
        "2410.00037v1", "3",
        "According to [2410.00037v1, Page 3], Helium is the backbone language model in Moshi "
        "with 7B (7 billion) parameters, pre-trained on text before fine-tuning for joint "
        "speech-text generation."
    ),
    (
        [r"maximum.*path|path.*length|o\(1\)|o\(n\)|self.attention.*table|table.*1.*attention"],
        "1706.03762v7", "6",
        "According to [1706.03762v7, Page 6], Table 1 shows Self-Attention has maximum path "
        "length O(1) between any two positions, compared to O(n) for recurrent layers, "
        "O(log_k(n)) for convolutional, and O(n/k) for restricted self-attention."
    ),
]

PAPER_FALLBACK = {
    "1706.03762v7": (
        "According to [{paper_id}, Page {page}], the Transformer architecture uses "
        "d_model=512, d_ff=2048, h=8 attention heads, warmup_steps=4000, and "
        "label smoothing epsilon_ls=0.1 to achieve state-of-the-art translation quality."
    ),
    "2410.00037v1": (
        "According to [{paper_id}, Page {page}], Moshi is a real-time speech-text dialogue "
        "system using Mimi (12.5 Hz, 1.1 kbps, 8 codebooks) with Helium (7B parameters) "
        "achieving ~200ms end-to-end latency."
    ),
    "2507.07955v2": (
        "According to [{paper_id}, Page {page}], H-Net uses Dynamic Chunking to segment "
        "sequences at variable semantic boundaries via differentiable Gumbel-softmax, "
        "evaluated on PG19 and WikiText-103 benchmarks."
    ),
}

def synthesize_answer(question_text: str, context_text: str) -> str:
    q_lower = question_text.lower()
    ctx_lower = context_text.lower()

    for patterns, paper_id, page, answer in ANSWER_TEMPLATES:
        if any(re.search(p, q_lower) for p in patterns):
            if paper_id in context_text or not re.search(r"1706|2410|2507", context_text):
                return answer

    paper_matches = re.findall(r"(1706\.03762v7|2410\.00037v1|2507\.07955v2)", context_text)
    cited_paper = paper_matches[0] if paper_matches else None
    page_match = re.search(r"Page (\d+)", context_text)
    cited_page = page_match.group(1) if page_match else "1"

    if cited_paper and cited_paper in PAPER_FALLBACK:
        return PAPER_FALLBACK[cited_paper].format(paper_id=cited_paper, page=cited_page)

    return "The papers do not say. The provided context does not contain sufficient information to answer this question."

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    if CONFIG["failure_rate"] > 0 and random.random() < CONFIG["failure_rate"]:
        raise HTTPException(status_code=500, detail="Simulated internal upstream model error")

    now = time.time()
    prune_windows(now)

    full_prompt = " ".join(m.get("content", "") for m in messages)
    tokens_in = max(10, len(full_prompt.split()) * 4 // 3)

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

    CONFIG["current_inflight"] += 1
    inflight = CONFIG["current_inflight"]

    load_mult = (CONFIG["concurrency_slowdown"] ** max(0, inflight - 1))
    prefill_ms = (CONFIG["base_prefill_ms"] + (tokens_in * CONFIG["ms_per_input_token"])) * load_mult
    if random.random() < CONFIG["slow_success_rate"]:
        prefill_ms += random.uniform(800.0, 1500.0)
    prefill_ms += random.uniform(5.0, 25.0)

    ttft_s = prefill_ms / 1000.0

    user_msg = messages[-1].get("content", "") if messages else ""
    context_match = re.search(r"Context:\s*(.*?)\s*Question:\s*(.*?)\s*Answer:", user_msg, re.DOTALL)

    if context_match:
        context_text = context_match.group(1)
        question_text = context_match.group(2)
    else:
        context_text = full_prompt
        question_text = user_msg

    q_words = set(re.findall(r"\b\w{4,}\b", question_text.lower()))
    ctx_words = set(re.findall(r"\b\w{4,}\b", context_text.lower()))
    overlap = q_words.intersection(ctx_words)

    paper_matches = re.findall(r"(1706\.03762v7|2410\.00037v1|2507\.07955v2)", context_text)
    if len(overlap) < 2 or not paper_matches:
        reply_str = "The papers do not say. The provided context does not contain sufficient information to answer this question."
    else:
        reply_str = synthesize_answer(question_text, context_text)

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

    response_headers = {
        "x-ratelimit-limit-requests": str(effective_rpm),
        "x-ratelimit-limit-tokens": str(effective_tpm),
        "x-ratelimit-remaining-requests": str(max(0, effective_rpm - len(REQ_TIMESTAMPS))),
        "x-ratelimit-remaining-tokens": str(max(0, effective_tpm - curr_tpm - tokens_in)),
        "x-ratelimit-reset-requests": f"{max(0.5, 60.0 - (now - REQ_TIMESTAMPS[0])):.1f}s" if REQ_TIMESTAMPS else "60.0s",
    }

    return StreamingResponse(sse_stream(), media_type="text/event-stream", headers=response_headers)

if __name__ == "__main__":
    uvicorn.run("standin.server:app", host="127.0.0.1", port=8000, log_level="warning")
