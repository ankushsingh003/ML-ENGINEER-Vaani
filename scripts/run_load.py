"""Load Generator: Runs required scenarios (steady, burst, overload, limit_halved, daily_budget_exhausted, and retrieval settings)."""
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
from relay.limiter import limiter

QUESTIONS_FILE = Path("data/questions/questions.json")

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

async def configure_standin(config_overrides: dict):
    """Update stand-in knobs via HTTP."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post("http://127.0.0.1:8000/config", json=config_overrides, timeout=2.0)
    except Exception:
        pass

async def execute_user_request(
    req_id: str,
    scenario: str,
    source: str,
    question_data: dict,
    retrieval_setting: str = "setting_2",
    deadline_sec: float = 12.0
):
    """Executes a single user request through admission queue and pipeline."""
    pipeline = RelayPipeline(source=source, retrieval_setting=retrieval_setting)
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    async def _run():
        return await pipeline.execute(question_data["question"], deadline_sec=deadline_sec)

    outcome = await admission_controller.enqueue_and_execute(_run, deadline_sec=deadline_sec)

    # Determine answer grade
    can_answer = question_data.get("can_answer", True)
    answer_text = str(outcome.get("answer", outcome.get("content", "")))
    status = outcome.get("status", "ok")

    if status != "ok":
        grade = "refused_correctly" if not can_answer else "wrong"
    else:
        if not can_answer:
            if "papers do not say" in answer_text.lower():
                grade = "refused_correctly"
            else:
                grade = "refused_wrongly"
        else:
            if "papers do not say" in answer_text.lower():
                grade = "refused_wrongly"
            else:
                grade = "correct"

    log_result(
        request_id=req_id,
        scenario=scenario,
        source=source,
        start_time=t_start,
        first_token_ms=outcome.get("first_token_ms", 0.0),
        total_ms=outcome.get("total_ms", 0.0),
        status=status,
        retries=outcome.get("retries", 0),
        tokens_in=outcome.get("tokens_in", 0),
        tokens_out=outcome.get("tokens_out", 0),
        retrieval_setting=retrieval_setting,
        answer_grade=grade
    )

    return outcome

# --- Scenarios ---

async def run_scenario_steady(source: str = "stand-in", count: int = 15):
    print(f"\n>>> Running Scenario: STEADY (n={count}, source={source}) <<<")
    await configure_standin({"concurrency_slowdown": 1.05, "failure_rate": 0.0, "limit_halved": False})
    questions = load_questions()
    for i in range(count):
        q = questions[i % len(questions)]
        res = await execute_user_request(f"steady_{i+1}", "steady", source, q)
        print(f"Req steady_{i+1}: status={res.get('status')}, TTFT={res.get('first_token_ms',0):.1f}ms, Total={res.get('total_ms',0):.1f}ms")
        await asyncio.sleep(2.5)  # ~24 req/min rate

async def run_scenario_burst(source: str = "stand-in", count: int = 20):
    print(f"\n>>> Running Scenario: BURST (n={count}, source={source}) <<<")
    await configure_standin({"concurrency_slowdown": 1.25, "failure_rate": 0.02, "limit_halved": False})
    questions = load_questions()
    tasks = [
        execute_user_request(f"burst_{i+1}", "burst", source, questions[i % len(questions)])
        for i in range(count)
    ]
    results = await asyncio.gather(*tasks)
    print(f"Completed burst of {count} requests.")

async def run_scenario_overload(source: str = "stand-in", count: int = 25):
    print(f"\n>>> Running Scenario: SUSTAINED OVERLOAD (n={count}, source={source}) <<<")
    await configure_standin({"concurrency_slowdown": 1.30, "failure_rate": 0.05, "limit_halved": False})
    questions = load_questions()
    # Offer 45 requests per minute (delay ~ 1.3s)
    for i in range(count):
        q = questions[i % len(questions)]
        asyncio.create_task(execute_user_request(f"overload_{i+1}", "sustained_overload", source, q))
        await asyncio.sleep(1.2)
    # Allow buffer to finish in-flight
    await asyncio.sleep(8.0)

async def run_scenario_limit_halved(source: str = "stand-in", count: int = 15):
    print(f"\n>>> Running Scenario: PER-MINUTE LIMIT HALVED (n={count}, source={source}) <<<")
    await configure_standin({"limit_halved": True, "concurrency_slowdown": 1.15})
    questions = load_questions()
    for i in range(count):
        q = questions[i % len(questions)]
        res = await execute_user_request(f"halved_{i+1}", "limit_halved", source, q)
        print(f"Req halved_{i+1}: status={res.get('status')}, TTFT={res.get('first_token_ms',0):.1f}ms")
        await asyncio.sleep(1.8)

async def run_scenario_daily_exhausted(source: str = "stand-in", count: int = 10):
    print(f"\n>>> Running Scenario: DAILY BUDGET EXHAUSTED (n={count}, source={source}) <<<")
    questions = load_questions()
    # Set limiter daily requests artificially to limit to trigger Heading 7 degradation
    limiter.daily_requests = limiter.rpd_limit
    for i in range(count):
        q = questions[i % len(questions)]
        res = await execute_user_request(f"daily_ex_{i+1}", "daily_budget_exhausted", source, q)
        print(f"Req daily_ex_{i+1}: status={res.get('status')}, answer preview={str(res.get('answer',''))[:60]}...")
        await asyncio.sleep(0.5)
    # Reset limiter
    limiter.daily_requests = 0

async def run_retrieval_comparison(source: str = "stand-in"):
    print(f"\n>>> Running Clean Retrieval Settings Benchmark (Table 3) across 3 settings <<<")
    await configure_standin({"concurrency_slowdown": 1.05, "failure_rate": 0.0, "limit_halved": False})
    questions = load_questions()
    for setting_key in ["setting_1", "setting_2", "setting_3"]:
        print(f"\n--- Testing {setting_key}: {RETRIEVAL_SETTINGS[setting_key]['description']} ---")
        for i, q in enumerate(questions):
            # Reset sliding window so retrieval evaluation is uninhibited by artificial rate limits
            limiter.req_window.clear()
            limiter.tok_window.clear()
            limiter.in_flight = 0
            
            res = await execute_user_request(
                req_id=f"retrieval_{setting_key}_{i+1}",
                scenario="retrieval_eval",
                source=source,
                question_data=q,
                retrieval_setting=setting_key
            )
            print(f"[{setting_key}] Q{i+1}: status={res.get('status')}, tokens_in={res.get('tokens_in')}, TTFT={res.get('first_token_ms',0):.1f}ms")
            await asyncio.sleep(0.1)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["all", "steady", "burst", "overload", "halved", "daily", "retrieval"])
    parser.add_argument("--source", default="stand-in", choices=["stand-in", "real"])
    args = parser.parse_args()

    init_results_csv(overwrite=False)

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

    print("\nAll requested load scenarios finished. Results recorded in results.csv.")

if __name__ == "__main__":
    asyncio.run(main())
