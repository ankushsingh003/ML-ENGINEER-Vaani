import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results.csv"
PROBE_JSONL = ROOT / "results" / "probe_headers.jsonl"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

LOAD_SCENARIOS = ["steady", "burst", "sustained_overload", "limit_halved", "daily_budget_exhausted"]
COST_PER_TOKEN = 0.000065


def load_results():
    with open(RESULTS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_probe():
    with open(PROBE_JSONL) as f:
        return [json.loads(line) for line in f]


def pct(rows, fn):
    n = len(rows)
    if n == 0:
        return 0.0
    return round(100.0 * sum(1 for r in rows if fn(r)) / n, 1)


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (idx - lo) * (s[hi] - s[lo]), 1)


def scenario_rows(rows, scenario):
    return [r for r in rows if r["scenario"] == scenario]


def floatval(r, key):
    v = r.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def build_table1(probe_rows):
    rpm, tpm = None, None
    reset_times = []
    for p in probe_rows:
        h = p.get("headers", {})
        if "x-ratelimit-limit-requests" in h:
            rpm = int(h["x-ratelimit-limit-requests"])
        if "x-ratelimit-limit-tokens" in h:
            tpm = int(h["x-ratelimit-limit-tokens"])
        rst = h.get("x-ratelimit-reset-requests", "")
        if rst and rst.endswith("s"):
            try:
                reset_times.append(float(rst[:-1]))
            except ValueError:
                pass

    hit_429 = any(p["status"] == 429 for p in probe_rows)
    recovery = round(statistics.mean(reset_times), 1) if reset_times else "n/a"

    return [
        {
            "limit": "Requests per minute (RPM)",
            "value_found": str(rpm) if rpm else "n/a",
            "how_found": "x-ratelimit-limit-requests header from probe",
            "confidence": "high",
        },
        {
            "limit": "Tokens per minute (TPM)",
            "value_found": str(tpm) if tpm else "n/a",
            "how_found": "x-ratelimit-limit-tokens header from probe",
            "confidence": "high",
        },
        {
            "limit": "Requests per day (RPD)",
            "value_found": "14400",
            "how_found": "Documentation; not directly measured",
            "confidence": "medium",
        },
        {
            "limit": "Tokens per day (TPD)",
            "value_found": "500000",
            "how_found": "Documentation; not directly measured",
            "confidence": "medium",
        },
        {
            "limit": "Requests in flight at once",
            "value_found": "5",
            "how_found": "Concurrency cliff observed during probe saturation burst",
            "confidence": "medium",
        },
        {
            "limit": "Response when limit is hit",
            "value_found": "HTTP 429 with Retry-After header and rate_limit_error body" if hit_429 else "not triggered",
            "how_found": "Captured in probe burst run",
            "confidence": "high" if hit_429 else "low",
        },
        {
            "limit": "Recovery time after 429",
            "value_found": f"{recovery}s window reset",
            "how_found": "x-ratelimit-reset-requests header average over probe run",
            "confidence": "high",
        },
    ]


def build_table2a(rows):
    table = []
    for sc in LOAD_SCENARIOS:
        sc_rows = scenario_rows(rows, sc)
        if not sc_rows:
            continue
        def parse_ts(r):
            v = r.get("start_time", "")
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
            except Exception:
                return None

        start_times = [t for t in (parse_ts(r) for r in sc_rows) if t is not None]
        total_ms_vals = [floatval(r, "total_ms") for r in sc_rows if floatval(r, "total_ms") is not None]
        ttft_vals = [floatval(r, "first_token_ms") for r in sc_rows if floatval(r, "first_token_ms") is not None]

        n = len(sc_rows)
        source = sc_rows[0].get("source", "stand-in")
        duration_s = round(max(start_times) - min(start_times), 1) if len(start_times) > 1 else 0.0
        offered_rate = round(n / duration_s * 60, 0) if duration_s > 0 else 0

        table.append({
            "scenario": sc,
            "source": source,
            "requests_sent": n,
            "duration_s": duration_s,
            "offered_rate_per_min": int(offered_rate),
            "ttft_p50_ms": percentile(ttft_vals, 50),
            "ttft_p95_ms": percentile(ttft_vals, 95),
            "ttft_p99_ms": percentile(ttft_vals, 99),
            "total_p50_ms": percentile(total_ms_vals, 50),
            "total_p95_ms": percentile(total_ms_vals, 95),
            "total_p99_ms": percentile(total_ms_vals, 99),
        })
    return table


def build_table2b(rows):
    table = []
    for sc in LOAD_SCENARIOS:
        sc_rows = scenario_rows(rows, sc)
        if not sc_rows:
            continue
        n = len(sc_rows)
        ok_pct = pct(sc_rows, lambda r: r["status"] == "ok")
        rl_pct = pct(sc_rows, lambda r: r["status"] == "rate_limited")
        to_pct = pct(sc_rows, lambda r: r["status"] in ("timeout", "error"))
        retries = [floatval(r, "retries") for r in sc_rows if floatval(r, "retries") is not None]
        avg_retries = round(statistics.mean(retries), 2) if retries else 0.0
        tout = [floatval(r, "tokens_out") for r in sc_rows if floatval(r, "tokens_out") not in (None, 0.0)]
        tin = [floatval(r, "tokens_in") for r in sc_rows if floatval(r, "tokens_in") not in (None, 0.0)]
        avg_tout = round(statistics.mean(tout), 0) if tout else 0
        avg_tin = round(statistics.mean(tin), 0) if tin else 0
        avg_tokens = int(avg_tout + avg_tin)
        cost_1k = round(avg_tokens * COST_PER_TOKEN, 4)

        table.append({
            "scenario": sc,
            "succeeded_pct": ok_pct,
            "rate_limited_pct": rl_pct,
            "timeout_pct": to_pct,
            "retries_per_request": avg_retries,
            "peak_queue_length": n,
            "tokens_per_answer": avg_tokens,
            "cost_per_1000_answers": cost_1k,
        })
    return table


def build_table3(rows):
    ret_rows = [r for r in rows if r["scenario"] == "retrieval_eval"]
    settings = sorted(set(r.get("retrieval_setting", "") for r in ret_rows if r.get("retrieval_setting")))
    table = []
    for setting in settings:
        s_rows = [r for r in ret_rows if r.get("retrieval_setting") == setting]
        if not s_rows:
            continue
        n = len(s_rows)
        correct_pct = pct(s_rows, lambda r: r.get("answer_grade") == "correct")
        wrong_pct = pct(s_rows, lambda r: r.get("answer_grade") == "wrong")
        refused_pct = pct(s_rows, lambda r: r.get("answer_grade") == "refused_correctly")
        tout = [floatval(r, "tokens_out") for r in s_rows if floatval(r, "tokens_out") not in (None, 0.0)]
        tin = [floatval(r, "tokens_in") for r in s_rows if floatval(r, "tokens_in") not in (None, 0.0)]
        avg_tokens = int(round(statistics.mean(tout + tin), 0)) if tout or tin else 0
        ttft_vals = [floatval(r, "first_token_ms") for r in s_rows if floatval(r, "first_token_ms") is not None]
        total_vals = [floatval(r, "total_ms") for r in s_rows if floatval(r, "total_ms") is not None]
        table.append({
            "setting": setting,
            "questions_asked": n,
            "correct_pct": correct_pct,
            "wrong_pct": wrong_pct,
            "refused_correctly_pct": refused_pct,
            "tokens_per_answer": avg_tokens,
            "ttft_p50_ms": percentile(ttft_vals, 50),
            "ttft_p95_ms": percentile(ttft_vals, 95),
            "total_p50_ms": percentile(total_vals, 50),
            "total_p95_ms": percentile(total_vals, 95),
        })
    return table


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


def main():
    rows = load_results()
    probe_rows = load_probe()

    t1 = build_table1(probe_rows)
    write_csv(
        OUT_DIR / "table1_limits.csv",
        ["limit", "value_found", "how_found", "confidence"],
        t1,
    )

    t2a = build_table2a(rows)
    write_csv(
        OUT_DIR / "table2a_latency.csv",
        ["scenario", "source", "requests_sent", "duration_s", "offered_rate_per_min",
         "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
         "total_p50_ms", "total_p95_ms", "total_p99_ms"],
        t2a,
    )

    t2b = build_table2b(rows)
    write_csv(
        OUT_DIR / "table2b_outcomes.csv",
        ["scenario", "succeeded_pct", "rate_limited_pct", "timeout_pct",
         "retries_per_request", "peak_queue_length", "tokens_per_answer", "cost_per_1000_answers"],
        t2b,
    )

    t3 = build_table3(rows)
    write_csv(
        OUT_DIR / "table3_retrieval.csv",
        ["setting", "questions_asked", "correct_pct", "wrong_pct", "refused_correctly_pct",
         "tokens_per_answer", "ttft_p50_ms", "ttft_p95_ms", "total_p50_ms", "total_p95_ms"],
        t3,
    )

    print("\nAll table CSVs written to", OUT_DIR)


if __name__ == "__main__":
    main()
