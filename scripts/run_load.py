"""
scripts/run_load.py
-------------------
Benchmark orchestrator executing required load shapes and retrieval evaluations.
"""

import sys
import time
import json
import random
import asyncio
import argparse
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.config import STANDIN_BASE_URL, RETRIEVAL_SETTINGS
from relay.results import log_result, init_results_csv
from relay.pipeline import RelayPipeline
from relay.queue import admission_controller
from relay.limiter import reset_for_test, force_daily_exhausted
from relay.trace import RequestTrace
from relay.grading import grade_answer

QUESTIONS_FILE = Path("data/questions/questions.json")

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

async def configure_standin(config_overrides: dict):
    try:
        async with httpx.AsyncClient() as client:
            await client.post("http://127.0.0.1:8000/config", json=config_overrides, timeout=2.0)
    except Exception:
        pass

async def execute_request(
    req_id: str,
    scenario: str,
    source: str,
    question_data: dict,
    retrieval_setting: str = "setting_2",
    deadline_sec: float = 12.0
):
    trace = RequestTrace(request_id=req_id, scenario=scenario, source=source)
    pipeline = RelayPipeline(source=source, retrieval_setting=retrieval_setting)

    async def _run_pipeline():
        return await pipeline.execute(question_data["question"], trace=trace, deadline_sec=deadline_sec)

    outcome = await admission_controller.submit(
        task_fn=_run_pipeline,
        trace=trace,
        deadline_sec=deadline_sec
    )

    status = outcome.get("status", "ok")
    answer = outcome.get("answer", "")
    grade = grade_answer(question_data, answer, status)

    rec = trace.finish(
        tokens_in=outcome.get("tokens_in", 0),
        tokens_out=outcome.get("tokens_out", 0),
        retries=outcome.get("retries", 0),
        status=status,
        answer_grade=grade,
        retrieval_setting=retrieval_setting
    )

    log_result(
        request_id=req_id,
        scenario=scenario,
        source=source,
        start_time=rec["wall_enqueued"],
        first_token_ms=rec["ttft_ms"],
        total_ms=rec["total_ms"],
        status=status,
        retries=outcome.get("retries", 0),
        tokens_in=outcome.get("tokens_in", 0),
        tokens_out=outcome.get("tokens_out", 0),
        retrieval_setting=retrieval_setting,
        answer_grade=grade
    )

    return outcome, rec

async def run_scenario_steady(source: str = "stand-in", count: int = 15):
    print(f"\n>>> Running Scenario: STEADY (n={count}, source={source}) <<<")
    await configure_standin({"limit_halved": False, "failure_rate": 0.0})
    reset_for_test()
    questions = load_questions()
    for i in range(count):
        q = questions[i % len(questions)]
        res, rec = await execute_request(f"steady_{i+1}", "steady", source, q)
        print(f"Req steady_{i+1}: status={res.get('status')}, TTFT={rec['ttft_ms']:.1f}ms, Total={rec['total_ms']:.1f}ms")
        await asyncio.sleep(2.5)

async def run_scenario_burst(source: str = "stand-in", count: int = 20):
    print(f"\n>>> Running Scenario: BURST (n={count}, source={source}) <<<")
    await configure_standin({"limit_halved": False, "failure_rate": 0.02})
    reset_for_test()
    questions = load_questions()
    tasks = [
        execute_request(f"burst_{i+1}", "burst", source, questions[i % len(questions)])
        for i in range(count)
    ]
    results = await asyncio.gather(*tasks)
    print(f"Completed burst of {count} requests.")

async def run_scenario_overload(source: str = "stand-in", count: int = 25):
    print(f"\n>>> Running Scenario: SUSTAINED OVERLOAD (n={count}, source={source}) <<<")
    await configure_standin({"limit_halved": False, "failure_rate": 0.04})
    reset_for_test()
    questions = load_questions()
    tasks = []
    for i in range(count):
        q = questions[i % len(questions)]
        tasks.append(asyncio.create_task(execute_request(f"overload_{i+1}", "sustained_overload", source, q)))
        await asyncio.sleep(0.8)
    await asyncio.gather(*tasks)

async def run_scenario_limit_halved(source: str = "stand-in", count: int = 15):
    print(f"\n>>> Running Scenario: PER-MINUTE LIMIT HALVED (n={count}, source={source}) <<<")
    await configure_standin({"limit_halved": True, "failure_rate": 0.0})
    reset_for_test()
    questions = load_questions()
    for i in range(count):
        q = questions[i % len(questions)]
        res, rec = await execute_request(f"halved_{i+1}", "limit_halved", source, q)
        print(f"Req halved_{i+1}: status={res.get('status')}, TTFT={rec['ttft_ms']:.1f}ms, Total={rec['total_ms']:.1f}ms")
        await asyncio.sleep(1.8)

async def run_scenario_daily_exhausted(source: str = "stand-in", count: int = 10):
    print(f"\n>>> Running Scenario: DAILY BUDGET EXHAUSTED (n={count}, source={source}) <<<")
    reset_for_test()
    force_daily_exhausted()
    questions = load_questions()
    for i in range(count):
        q = questions[i % len(questions)]
        res, rec = await execute_request(f"daily_ex_{i+1}", "daily_budget_exhausted", source, q)
        print(f"Req daily_ex_{i+1}: status={res.get('status')}, TTFT={rec['ttft_ms']:.1f}ms, preview={str(res.get('answer',''))[:50]}...")
        await asyncio.sleep(0.3)
    reset_for_test()

async def run_retrieval_comparison(source: str = "stand-in"):
    print(f"\n>>> Running Retrieval Benchmark across 3 settings <<<")
    await configure_standin({"limit_halved": False, "failure_rate": 0.0})
    print("Waiting 65s for rate-limit windows to clear before retrieval eval...")
    await asyncio.sleep(65)
    questions = load_questions()
    for setting_key in ["setting_1", "setting_2", "setting_3"]:
        print(f"\n--- Testing {setting_key}: {RETRIEVAL_SETTINGS[setting_key]['description']} ---")
        reset_for_test()
        for i, q in enumerate(questions):
            res, rec = await execute_request(
                req_id=f"retrieval_{setting_key}_{i+1}",
                scenario="retrieval_eval",
                source=source,
                question_data=q,
                retrieval_setting=setting_key
            )
            print(f"[{setting_key}] Q{i+1}: status={res.get('status')}, grade={rec.get('answer_grade')}, tokens_in={rec.get('tokens_in')}, TTFT={rec['ttft_ms']:.1f}ms")
            await asyncio.sleep(2.2)
        print(f"Waiting 65s to clear rate-limit windows before next retrieval setting...")
        await asyncio.sleep(65)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["all", "steady", "burst", "overload", "halved", "daily", "retrieval"])
    parser.add_argument("--source", default="stand-in", choices=["stand-in", "real"])
    args = parser.parse_args()

    init_results_csv(overwrite=False)

    try:
        if args.scenario in ["all", "steady"]:
            await run_scenario_steady(source=args.source)
        if args.scenario in ["all", "burst"]:
            await run_scenario_burst(source="stand-in")
        if args.scenario in ["all", "overload"]:
            await run_scenario_overload(source="stand-in")
        if args.scenario in ["all", "halved"]:
            await run_scenario_limit_halved(source="stand-in")
        if args.scenario in ["all", "daily"]:
            await run_scenario_daily_exhausted(source="stand-in")
        if args.scenario in ["all", "retrieval"]:
            await run_retrieval_comparison(source=args.source)
    finally:
        await admission_controller.shutdown()

    print("\nBenchmark scenarios finished. Results recorded in results.csv and results/results_trace.jsonl.")

if __name__ == "__main__":
    asyncio.run(main())
