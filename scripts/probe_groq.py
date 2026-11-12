"""Groq Probe Script: Measures latency distribution, provider rate limits, and 429 behavior."""
import os
import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
import httpx

# Ensure relay is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from relay.results import log_result, init_results_csv

PROBE_HEADERS_FILE = Path("results/probe_headers.jsonl")

SAMPLE_PROMPTS = [
    "What is the learning rate warmup schedule in Attention Is All You Need?",
    "Explain the Mimi neural audio codec framerate and bitrate in Moshi.",
    "How does Dynamic Chunking in H-Net prevent information loss?",
    "What are the self-attention layer complexity bounds in the Transformer?",
    "Summarize the multi-stream causal architecture of Moshi.",
    "Can you explain the difference between encoder and decoder attention in Transformers?",
    "What benchmark datasets were evaluated in Dynamic Chunking for sequence modeling?",
    "How many codebooks are used by Mimi in Moshi?"
]

async def send_single_probe(client: httpx.AsyncClient, req_id: int, prompt: str) -> dict:
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise research assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "stream": True
    }

    t0 = time.perf_counter()
    first_token_ms = 0.0
    first_token_seen = False
    tokens_out = 0
    tokens_in = len(prompt.split()) + 15
    retries = 0

    try:
        async with client.stream("POST", url, json=payload, headers=headers, timeout=12.0) as resp:
            resp_headers = dict(resp.headers)
            status_code = resp.status_code

            # Record ratelimit headers
            rl_headers = {k: v for k, v in resp_headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
            if rl_headers:
                PROBE_HEADERS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(PROBE_HEADERS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"req_id": req_id, "timestamp": time.time(), "status": status_code, "headers": rl_headers}) + "\n")

            if status_code == 429:
                total_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "req_id": f"probe_{req_id}",
                    "status": "rate_limited",
                    "first_token_ms": 0.0,
                    "total_ms": total_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": 0,
                    "retries": 0,
                    "headers": rl_headers
                }

            if status_code != 200:
                body = await resp.aread()
                total_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "req_id": f"probe_{req_id}",
                    "status": "error",
                    "first_token_ms": 0.0,
                    "total_ms": total_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": 0,
                    "retries": 0,
                    "headers": rl_headers
                }

            async for line in resp.aiter_lines():
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
                            tokens_in = u.get("prompt_tokens", tokens_in)
                            tokens_out = u.get("completion_tokens", tokens_out)

                        choices = chunk.get("choices", [])
                        if choices and "delta" in choices[0]:
                            content = choices[0]["delta"].get("content", "")
                            if content and not first_token_seen:
                                first_token_seen = True
                                first_token_ms = (time.perf_counter() - t0) * 1000.0
                            if content:
                                tokens_out += 1
                    except Exception:
                        pass

            total_ms = (time.perf_counter() - t0) * 1000.0
            if not first_token_seen:
                first_token_ms = total_ms

            return {
                "req_id": f"probe_{req_id}",
                "status": "ok",
                "first_token_ms": first_token_ms,
                "total_ms": total_ms,
                "tokens_in": tokens_in,
                "tokens_out": max(1, tokens_out),
                "retries": 0,
                "headers": rl_headers
            }

    except Exception as e:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "req_id": f"probe_{req_id}",
            "status": "error",
            "first_token_ms": 0.0,
            "total_ms": total_ms,
            "tokens_in": tokens_in,
            "tokens_out": 0,
            "retries": 0,
            "headers": {}
        }

async def run_probe(n: int = 10, rate_per_min: int = 15):
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable is not set!")
        print("Please set GROQ_API_KEY before running the probe.")
        return

    print(f"=== Starting Groq Probe (n={n}, rate={rate_per_min}/min) on model: {GROQ_MODEL} ===")
    delay_between = 60.0 / rate_per_min

    results = []
    init_results_csv()

    async with httpx.AsyncClient() as client:
        for i in range(n):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            print(f"Sending probe request {i+1}/{n}...")
            res = await send_single_probe(client, i + 1, prompt)
            results.append(res)
            print(f"  Req {i+1}: status={res['status']}, TTFT={res['first_token_ms']:.1f}ms, Total={res['total_ms']:.1f}ms")
            
            # Log to results.csv
            log_result(
                request_id=res["req_id"],
                scenario="probe_latency",
                source="real",
                start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                first_token_ms=res["first_token_ms"],
                total_ms=res["total_ms"],
                status=res["status"],
                retries=res["retries"],
                tokens_in=res["tokens_in"],
                tokens_out=res["tokens_out"],
                retrieval_setting="probe",
                answer_grade="correct" if res["status"] == "ok" else "wrong"
            )

            if i < n - 1:
                await asyncio.sleep(delay_between)

    # Compute quick stats
    ok_runs = [r for r in results if r["status"] == "ok"]
    if ok_runs:
        ttfts = sorted(r["first_token_ms"] for r in ok_runs)
        totals = sorted(r["total_ms"] for r in ok_runs)
        p50_ttft = ttfts[len(ttfts) // 2]
        p95_ttft = ttfts[int(len(ttfts) * 0.95)]
        p50_total = totals[len(totals) // 2]
        p95_total = totals[int(len(totals) * 0.95)]
        print("\n=== Probe Summary ===")
        print(f"Successful: {len(ok_runs)}/{n}")
        print(f"TTFT (ms): p50={p50_ttft:.1f}, p95={p95_ttft:.1f}")
        print(f"Total time (ms): p50={p50_total:.1f}, p95={p95_total:.1f}")
        print(f"Headers captured in: {PROBE_HEADERS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Number of probe requests")
    parser.add_argument("--rate", type=int, default=15, help="Requests per minute rate")
    args = parser.parse_args()
    asyncio.run(run_probe(n=args.n, rate_per_min=args.rate))

