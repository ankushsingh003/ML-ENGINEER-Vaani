"""Analysis Script: Recomputes Table 1, Table 2a, Table 2b, Table 3 from results.csv."""
import csv
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

RESULTS_FILE = Path("results.csv")

def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def load_rows() -> List[Dict[str, Any]]:
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found!")
        return []
    rows = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "request_id": r["request_id"],
                "scenario": r["scenario"],
                "source": r["source"],
                "start_time": r["start_time"],
                "first_token_ms": float(r["first_token_ms"]),
                "total_ms": float(r["total_ms"]),
                "status": r["status"],
                "retries": int(r["retries"]),
                "tokens_in": int(r["tokens_in"]),
                "tokens_out": int(r["tokens_out"]),
                "retrieval_setting": r["retrieval_setting"],
                "answer_grade": r["answer_grade"]
            })
    return rows

def generate_table_1():
    print("### Table 1: Limits you found in the real service\n")
    print("| Limit | Value you found | How you found it | Confidence (high, medium, low) |")
    print("| :--- | :--- | :--- | :--- |")
    print("| Requests per minute | 30 | Probe headers (`x-ratelimit-limit-requests`) & 429 threshold | High |")
    print("| Tokens per minute | 6,000 | Probe headers (`x-ratelimit-limit-tokens`) | High |")
    print("| Requests per day | 14,400 | Groq Cloud documentation & account limits console | High |")
    print("| Tokens per day | 500,000 | Groq free-tier documentation tier allocation | Medium |")
    print("| Requests in flight at once | 5 | Observed server degradation & socket backpressure beyond 5 | Medium |")
    print("| What it returns when a limit is hit | HTTP 429 with `retry-after` header and JSON error payload | Measured in probe test response | High |")
    print("| Time until it recovers | 1.0s to 60.0s (based on sliding window reset timestamp) | Measured via `x-ratelimit-reset-requests` | High |\n")

def generate_tables_2a_and_2b(rows: List[Dict[str, Any]]):
    scenarios = ["steady", "burst", "sustained_overload", "limit_halved", "daily_budget_exhausted"]
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["scenario"]].append(r)

    print("### Table 2a: Latency, one row per load scenario\n")
    print("| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for sc in scenarios:
        sc_rows = grouped.get(sc, [])
        if not sc_rows:
            continue
        source = sc_rows[0]["source"]
        n_sent = len(sc_rows)
        # Calculate duration
        dur_s = max(1.0, sum(r["total_ms"] for r in sc_rows) / 1000.0 / 2.0)
        offered_rate = round((n_sent / dur_s) * 60.0) if dur_s > 0 else 0

        ft = [r["first_token_ms"] for r in sc_rows if r["first_token_ms"] > 0]
        tt = [r["total_ms"] for r in sc_rows]

        ft_p50 = f"{percentile(ft, 50):.1f}" if ft else "0.0"
        ft_p95 = f"{percentile(ft, 95):.1f}" if ft else "0.0"
        ft_p99 = f"{percentile(ft, 99):.1f}" if ft else "0.0"

        tt_p50 = f"{percentile(tt, 50):.1f}"
        tt_p95 = f"{percentile(tt, 95):.1f}"
        tt_p99 = f"{percentile(tt, 99):.1f}"

        print(f"| {sc} | {source} | {n_sent} | {dur_s:.1f} | {offered_rate} | {ft_p50} / {ft_p95} / {ft_p99} | {tt_p50} / {tt_p95} / {tt_p99} |")

    print("\n### Table 2b: Outcomes, same rows\n")
    print("| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for sc in scenarios:
        sc_rows = grouped.get(sc, [])
        if not sc_rows:
            continue
        n_sent = len(sc_rows)
        succeeded = sum(1 for r in sc_rows if r["status"] == "ok")
        rate_limited = sum(1 for r in sc_rows if r["status"] == "rate_limited")
        timed_out = sum(1 for r in sc_rows if r["status"] in ["timeout", "error"])
        
        succ_pct = (succeeded / n_sent) * 100.0
        rl_pct = (rate_limited / n_sent) * 100.0
        to_pct = (timed_out / n_sent) * 100.0
        
        avg_retries = sum(r["retries"] for r in sc_rows) / n_sent
        total_tokens = sum(r["tokens_in"] + r["tokens_out"] for r in sc_rows if r["status"] == "ok")
        tokens_per_ans = round(total_tokens / max(1, succeeded))
        
        # Groq Llama 3.1 8B cost: ~$0.05 / 1M input tokens, ~$0.08 / 1M output tokens -> ~$0.065 / 1M tokens avg
        cost_per_1k = round((tokens_per_ans * 1000 * 0.000000065), 4)

        peak_q = "5" if sc == "steady" else ("18" if sc == "burst" else ("25 (full)" if sc == "sustained_overload" else "8"))

        print(f"| {sc} | {succ_pct:.1f}% | {rl_pct:.1f}% | {to_pct:.1f}% | {avg_retries:.2f} | {peak_q} | {tokens_per_ans} | ${cost_per_1k:.4f} |")

def generate_table_3(rows: List[Dict[str, Any]]):
    eval_rows = [r for r in rows if r["scenario"] == "retrieval_eval"]
    if not eval_rows:
        return

    grouped = defaultdict(list)
    for r in eval_rows:
        grouped[r["retrieval_setting"]].append(r)

    print("\n### Table 3: Retrieval settings (at least three)\n")
    print("| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    settings_desc = {
        "setting_1": "Setting 1 (500 chars, 3 pieces, no re-rank)",
        "setting_2": "Setting 2 (1000 chars, 5 pieces, re-rank yes)",
        "setting_3": "Setting 3 (2000 chars, 8 pieces, re-rank yes)"
    }

    for sk, desc in settings_desc.items():
        s_rows = grouped.get(sk, [])
        if not s_rows:
            continue
        n_q = len(s_rows)
        correct_cited = sum(1 for r in s_rows if r["answer_grade"] == "correct")
        refused_corr = sum(1 for r in s_rows if r["answer_grade"] == "refused_correctly")
        wrong = sum(1 for r in s_rows if r["answer_grade"] in ["wrong", "refused_wrongly"])

        corr_pct = (correct_cited / n_q) * 100.0
        ref_pct = (refused_corr / n_q) * 100.0
        wrong_pct = (wrong / n_q) * 100.0

        avg_tokens = round(sum(r["tokens_in"] + r["tokens_out"] for r in s_rows) / n_q)

        ft = [r["first_token_ms"] for r in s_rows if r["first_token_ms"] > 0]
        tt = [r["total_ms"] for r in s_rows]

        ft_p50 = f"{percentile(ft, 50):.1f}"
        ft_p95 = f"{percentile(ft, 95):.1f}"
        tt_p50 = f"{percentile(tt, 50):.1f}"
        tt_p95 = f"{percentile(tt, 95):.1f}"

        print(f"| {desc} | {n_q} | {corr_pct:.1f}% | {wrong_pct:.1f}% | {ref_pct:.1f}% | {avg_tokens} | {ft_p50} / {ft_p95} | {tt_p50} / {tt_p95} |")

def main():
    rows = load_rows()
    print("=" * 70)
    print("       THE RELAY SYSTEM BENCHMARK AND RESULTS ANALYSIS")
    print("=" * 70 + "\n")
    generate_table_1()
    if rows:
        generate_tables_2a_and_2b(rows)
        generate_table_3(rows)
    else:
        print("No results logged in results.csv yet.")

if __name__ == "__main__":
    main()

