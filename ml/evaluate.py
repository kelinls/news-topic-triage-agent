"""Run paired AgentX and fixed-baseline evaluation over the six-topic holdout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from .train import CATEGORIES, fetch_data

SEED = 20261122
RESULTS = Path(__file__).resolve().parent / "results"


def test_cases() -> list[dict]:
    data = fetch_data("test")
    rng = random.Random(SEED)
    cases = []
    for topic, label in enumerate(CATEGORIES):
        pool = [i for i, target in enumerate(data.target)
                if target == topic and 200 <= len(data.data[i]) <= 1800]
        for position, index in enumerate(rng.sample(pool, 3), 1):
            cases.append((f"rep-{topic + 1}-{position}", "representative", index))
    cases.extend((f"hard-{i}", "challenge", index)
                 for i, index in enumerate([49, 53, 63], 1))
    cases.extend((f"short-{i}", "invalid", index)
                 for i, index in enumerate([5, 109, 156], 1))
    assert len({index for _, _, index in cases}) == 24
    return [{"id": case_id, "subset": subset, "index": index,
             "label": CATEGORIES[int(data.target[index])], "text": data.data[index]}
            for case_id, subset, index in cases]


def request(url: str, text: str, use_skill: bool | None) -> dict:
    payload = {"text": text}
    if use_skill is not None:
        payload["useSkill"] = use_skill
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=150) as response:
        return json.load(response)


def evaluate(mode: str, url: str, cases: list[dict], use_skill: bool | None, workers: int) -> list[dict]:
    def run(case: dict) -> dict:
        result = request(url, case["text"], use_skill)
        record = {key: case[key] for key in ("id", "subset", "index", "label")}
        record["text_sha256"] = hashlib.sha256(case["text"].encode()).hexdigest()
        record["result"] = result
        print(mode, case["id"], result["status"], result.get("label"), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(run, cases))
    (RESULTS / f"agentx_{mode}.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def summary(records: list[dict]) -> dict:
    outputs = [r["result"] for r in records]
    status = Counter(r["status"] for r in outputs)
    correct = sum(r["result"]["status"] == "auto_route" and
                  r["result"]["label"] == r["label"] for r in records)
    latencies = [r["elapsedMs"] for r in outputs if "elapsedMs" in r]
    return {
        "total": len(outputs), "auto": status["auto_route"], "correct": correct,
        "review": status["human_review"], "rejected": status["rejected"],
        "fallback": sum(bool(r.get("fallback")) for r in outputs),
        "skill_loaded": sum(bool(r.get("skillLoaded")) for r in outputs),
        "tool_errors": sum(not call["ok"] for r in outputs for call in r.get("toolTrace", [])),
        "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "errors": [
            {"id": r["id"], "label": r["label"], "predicted": r["result"]["label"]}
            for r in records if r["result"]["status"] == "auto_route"
            and r["result"]["label"] != r["label"]
        ],
        "challenge_routes": [r["result"]["status"] for r in records if r["subset"] == "challenge"],
        "invalid_rejected": sum(r["subset"] == "invalid" and r["result"]["status"] == "rejected"
                                for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8088/api/triage")
    parser.add_argument("--baseline", default="http://127.0.0.1:8091/baseline")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cases = test_cases()
    baseline = evaluate("baseline", args.baseline, cases, None, args.workers)
    with ThreadPoolExecutor(max_workers=2) as batches:
        full_run = batches.submit(evaluate, "full", args.api, cases, True, args.workers)
        no_skill_run = batches.submit(evaluate, "no_skill", args.api, cases, False, args.workers)
        full, no_skill = full_run.result(), no_skill_run.result()
    repeated_cases = [c for c in cases if c["subset"] == "representative" and not c["id"].endswith("-3")]
    repeat = evaluate("repeat", args.api, repeated_cases, True, args.workers)
    by_id = {r["id"]: r["result"] for r in full}
    comparison = [{"id": r["id"], "first_status": by_id[r["id"]]["status"],
                   "second_status": r["result"]["status"], "first_label": by_id[r["id"]]["label"],
                   "second_label": r["result"]["label"]} for r in repeat]
    witness = next(r["result"] for r in full if r["result"].get("model"))
    metrics = {
        "date": date.today().isoformat(), "seed": SEED, "model": witness["model"], "runtime": witness["runtime"],
        "baseline": summary(baseline), "full": summary(full), "no_skill": summary(no_skill),
        "repeat": {
            **summary(repeat),
            "status_agreements": sum(x["first_status"] == x["second_status"] for x in comparison),
            "label_agreements": sum(x["first_label"] == x["second_label"] for x in comparison),
            "comparisons": comparison,
        },
    }
    (RESULTS / "agentx_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
