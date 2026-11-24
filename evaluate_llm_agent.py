#!/usr/bin/env python3
"""Run reproducible LLM-agent, fixed-pipeline, ablation, and repeat evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.datasets import fetch_20newsgroups

from llm_agent import NewsTriageLLMAgent, load_artifact
from skill_registry import SkillRegistry


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "llm_agent_results"
PREDICTIONS_PATH = ROOT / "agent_results" / "test_predictions.csv"
CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.guns",
]
CATEGORY_SLUGS = {
    "alt.atheism": "atheism",
    "comp.graphics": "graphics",
    "rec.autos": "autos",
    "rec.sport.baseball": "baseball",
    "sci.med": "med",
    "talk.politics.guns": "guns",
}
SAMPLE_SEED = 20261122


def load_test_data():
    return fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
        download_if_missing=True,
    )


def load_prediction_rows() -> list[dict[str, str]]:
    with PREDICTIONS_PATH.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_cases(
    data,
    prediction_rows: list[dict[str, str]],
    fixed_agent=None,
) -> list[dict[str, Any]]:
    rng = random.Random(SAMPLE_SEED)
    used: set[int] = set()
    cases: list[dict[str, Any]] = []

    for class_index, label in enumerate(CATEGORIES):
        candidates = [
            index
            for index, target in enumerate(data.target)
            if int(target) == class_index and 200 <= len(data.data[index]) <= 1800
        ]
        selected = rng.sample(candidates, 3)
        for number, index in enumerate(selected, start=1):
            used.add(index)
            cases.append(
                {
                    "task_id": f"rep-{CATEGORY_SLUGS[label]}-{number}",
                    "subset": "representative",
                    "test_index": index,
                    "text": data.data[index],
                    "true_label": label,
                    "selection_note": "stratified random sample, seed=20261122",
                }
            )

    # Deterministic challenge cases from the fixed pipeline's confidently wrong routes.
    hard_indices = [49, 53, 63]
    for number, index in enumerate(hard_indices, start=1):
        if index in used:
            raise RuntimeError(f"challenge case {index} overlaps representative sample")
        used.add(index)
        truth = CATEGORIES[int(data.target[index])]
        cases.append(
            {
                "task_id": f"hard-{number:02d}",
                "subset": "hard_error",
                "test_index": index,
                "text": data.data[index],
                "true_label": truth,
                "selection_note": "fixed pipeline auto_route error; chosen to reveal failure mode",
            }
        )

    # Three official test-set inputs shorter than the 40-character gate.
    short_indices = [5, 109, 156]
    for number, index in enumerate(short_indices, start=1):
        if index in used or len(data.data[index]) >= 40:
            raise RuntimeError(f"invalid short-case selection: {index}")
        used.add(index)
        truth = CATEGORIES[int(data.target[index])]
        cases.append(
            {
                "task_id": f"short-{number:02d}",
                "subset": "invalid_input",
                "test_index": index,
                "text": data.data[index],
                "true_label": truth,
                "selection_note": "official test input below the validation threshold",
            }
        )

    for case in cases:
        row = prediction_rows[case["test_index"]]
        if fixed_agent is not None:
            fixed = fixed_agent.triage(case["text"])
            correct = None
            if fixed["status"] == "auto_route":
                correct = fixed["suggested_label"] == case["true_label"]
            case["fixed_pipeline"] = {
                "status": fixed["status"],
                "label": fixed["suggested_label"],
                "confidence": fixed["confidence"],
                "neighbor_agreement": fixed["neighbor_agreement"],
                "correct": correct,
            }
        else:
            case["fixed_pipeline"] = {
                "status": row["routing_status"],
                "label": row["predicted_label"],
                "confidence": float(row["confidence"]),
                "neighbor_agreement": float(row["neighbor_agreement"]),
                "correct": row["correct"] == "True",
            }
        case["text_sha256"] = hashlib.sha256(case["text"].encode("utf-8")).hexdigest()
    return cases


def safe_correct(case: dict[str, Any], status: str, label: str | None) -> bool | None:
    if case["subset"] == "invalid_input":
        return status == "rejected"
    if status != "auto_route":
        return None
    return label == case["true_label"]


def summarize_run(
    mode: str,
    records: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    completed = [record for record in records if not record["result"].get("fallback")]
    statuses = Counter(record["result"]["status"] for record in records)
    auto_records = [record for record in records if record["result"]["status"] == "auto_route"]
    auto_correct = [
        record
        for record in auto_records
        if safe_correct(
            cases_by_id[record["task_id"]],
            record["result"]["status"],
            record["result"].get("suggested_label"),
        )
        is True
    ]
    auto_wrong = [
        record
        for record in auto_records
        if safe_correct(
            cases_by_id[record["task_id"]],
            record["result"]["status"],
            record["result"].get("suggested_label"),
        )
        is False
    ]
    invalid_records = [
        record
        for record in records
        if cases_by_id[record["task_id"]]["subset"] == "invalid_input"
    ]
    hard_records = [
        record
        for record in records
        if cases_by_id[record["task_id"]]["subset"] == "hard_error"
    ]
    tool_failure_records = [
        record for record in records if record["result"].get("tool_errors")
    ]
    recovered = []
    for record in tool_failure_records:
        failed_tools = {item["tool"] for item in record["result"]["tool_errors"]}
        later_success = {
            trace["tool"]
            for trace in record["result"]["tool_trace"]
            if trace["result"].get("ok") and trace["tool"] in failed_tools
        }
        if not record["result"].get("fallback") and later_success == failed_tools:
            recovered.append(record["task_id"])
    durations = [record["result"]["duration_seconds"] for record in records]
    steps = [record["result"]["steps"] for record in records]
    tool_calls = [record["result"]["tool_calls"] for record in records]
    prompt_tokens = sum(record["result"]["usage"]["prompt_tokens"] for record in records)
    completion_tokens = sum(
        record["result"]["usage"]["completion_tokens"] for record in records
    )
    return {
        "mode": mode,
        "task_count": len(records),
        "completed_count": len(completed),
        "completion_rate": len(completed) / len(records) if records else 0.0,
        "status_counts": dict(statuses),
        "auto_route_count": len(auto_records),
        "auto_route_rate": len(auto_records) / len(records) if records else 0.0,
        "auto_route_correct_count": len(auto_correct),
        "auto_route_accuracy": len(auto_correct) / len(auto_records) if auto_records else None,
        "auto_route_error_count": len(auto_wrong),
        "human_review_count": statuses.get("human_review", 0),
        "rejected_count": statuses.get("rejected", 0),
        "invalid_case_count": len(invalid_records),
        "invalid_rejection_count": sum(
            record["result"]["status"] == "rejected" for record in invalid_records
        ),
        "hard_case_count": len(hard_records),
        "hard_auto_error_count": sum(
            record["result"]["status"] == "auto_route"
            and record["result"].get("suggested_label")
            != cases_by_id[record["task_id"]]["true_label"]
            for record in hard_records
        ),
        "hard_safe_count": sum(
            record["result"]["status"] == "human_review" for record in hard_records
        ),
        "skill_load_count": sum(
            record["result"].get("skill_loaded") is not None for record in records
        ),
        "tool_failure_task_count": len(tool_failure_records),
        "tool_failure_recovery_count": len(recovered),
        "tool_failure_recovered_task_ids": recovered,
        "mean_llm_steps": statistics.fmean(steps) if steps else 0.0,
        "mean_tool_calls": statistics.fmean(tool_calls) if tool_calls else 0.0,
        "mean_latency_seconds": statistics.fmean(durations) if durations else 0.0,
        "median_latency_seconds": statistics.median(durations) if durations else 0.0,
        "max_latency_seconds": max(durations, default=0.0),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def run_mode(
    mode: str,
    cases: list[dict[str, Any]],
    agent: NewsTriageLLMAgent,
    failure_plan: dict[str, dict[str, int]],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / f"{mode}_tasks.jsonl"
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as stream:
        for number, case in enumerate(cases, start=1):
            print(
                f"[{mode}] {number:02d}/{len(cases)} {case['task_id']} "
                f"({case['subset']}, {len(case['text'])} chars)",
                flush=True,
            )
            result = agent.run(
                case["text"],
                task_id=case["task_id"],
                require_skill=mode != "no_skill",
                inject_failures=failure_plan.get(case["task_id"]),
            )
            record = {
                "task_id": case["task_id"],
                "subset": case["subset"],
                "test_index": case["test_index"],
                "true_label": case["true_label"],
                "text_length": len(case["text"]),
                "text_sha256": case["text_sha256"],
                "input_excerpt": " ".join(case["text"].split())[:320],
                "fixed_pipeline": case["fixed_pipeline"],
                "result": result,
            }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    elapsed = time.perf_counter() - started
    cases_by_id = {case["task_id"]: case for case in cases}
    metrics = summarize_run(mode, records, cases_by_id)
    metrics["wall_clock_seconds"] = round(elapsed, 4)
    metrics["records_path"] = str(output_path.relative_to(ROOT))
    (RESULTS / f"{mode}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def load_records(mode: str) -> list[dict[str, Any]]:
    path = RESULTS / f"{mode}_tasks.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_consistency(
    cases: list[dict[str, Any]],
    agent: NewsTriageLLMAgent,
) -> None:
    original = {record["task_id"]: record for record in load_records("full")}
    consistency_cases = [
        case
        for case in cases
        if case["subset"] == "representative"
        and case["task_id"] in original
    ][:12]
    run_mode("consistency", consistency_cases, agent, failure_plan={})
    repeated = {record["task_id"]: record for record in load_records("consistency")}
    fields = []
    for case in consistency_cases:
        before = original[case["task_id"]]["result"]
        after = repeated[case["task_id"]]["result"]
        fields.append(
            {
                "task_id": case["task_id"],
                "status_agree": before["status"] == after["status"],
                "label_agree": before.get("suggested_label")
                == after.get("suggested_label"),
                "before_status": before["status"],
                "after_status": after["status"],
                "before_label": before.get("suggested_label"),
                "after_label": after.get("suggested_label"),
            }
        )
    metrics = json.loads((RESULTS / "consistency_metrics.json").read_text(encoding="utf-8"))
    metrics["paired_task_count"] = len(fields)
    metrics["status_agreement_count"] = sum(item["status_agree"] for item in fields)
    metrics["status_agreement_rate"] = (
        metrics["status_agreement_count"] / len(fields) if fields else 0.0
    )
    metrics["label_agreement_count"] = sum(item["label_agree"] for item in fields)
    metrics["label_agreement_rate"] = (
        metrics["label_agreement_count"] / len(fields) if fields else 0.0
    )
    metrics["comparisons"] = fields
    (RESULTS / "consistency_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fixed_pipeline_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(case["fixed_pipeline"]["status"] for case in cases)
    auto = [case for case in cases if case["fixed_pipeline"]["status"] == "auto_route"]
    correct = sum(case["fixed_pipeline"]["correct"] is True for case in auto)
    return {
        "task_count": len(cases),
        "status_counts": dict(statuses),
        "auto_route_count": len(auto),
        "auto_route_accuracy": correct / len(auto) if auto else None,
        "auto_route_error_count": len(auto) - correct,
        "human_review_count": statuses.get("human_review", 0),
        "rejected_count": statuses.get("rejected", 0),
    }


def build_metrics(cases: list[dict[str, Any]]) -> None:
    cases_by_id = {case["task_id"]: case for case in cases}
    records_by_mode = {
        mode: load_records(mode) for mode in ("full", "no_skill")
    }
    for mode, records in records_by_mode.items():
        if records:
            metrics = summarize_run(mode, records, cases_by_id)
            (RESULTS / f"{mode}_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    full = json.loads((RESULTS / "full_metrics.json").read_text(encoding="utf-8"))
    no_skill_path = RESULTS / "no_skill_metrics.json"
    no_skill = (
        json.loads(no_skill_path.read_text(encoding="utf-8"))
        if no_skill_path.exists()
        else None
    )
    subset_metrics = {}
    for subset in ("representative", "hard_error", "invalid_input"):
        subset_cases = [case for case in cases if case["subset"] == subset]
        records = [
            record
            for record in records_by_mode["full"]
            if record["subset"] == subset
        ]
        subset_metrics[subset] = summarize_run(
            f"full:{subset}", records, {case["task_id"]: case for case in subset_cases}
        )
    summary = {
        "run_date": "2026-11-22",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sample_seed": SAMPLE_SEED,
        "case_count": len(cases),
        "subset_counts": dict(Counter(case["subset"] for case in cases)),
        "fixed_pipeline_all": fixed_pipeline_summary(cases),
        "fixed_pipeline_representative": fixed_pipeline_summary(
            [case for case in cases if case["subset"] == "representative"]
        ),
        "full": full,
        "no_skill": no_skill,
        "full_subsets": subset_metrics,
        "consistency": json.loads(
            (RESULTS / "consistency_metrics.json").read_text(encoding="utf-8")
        )
        if (RESULTS / "consistency_metrics.json").exists()
        else None,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (RESULTS / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "task_id",
                "subset",
                "test_index",
                "true_label",
                "text_length",
                "selection_note",
                "text_sha256",
            ],
        )
        writer.writeheader()
        for case in cases:
            row = {key: case.get(key) for key in writer.fieldnames}
            row["text_length"] = len(case["text"])
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=["full", "no_skill", "consistency", "all", "metrics"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=0, help="limit tasks for a smoke run")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_test_data()
    predictions = load_prediction_rows()
    artifact = load_artifact()
    cases = make_cases(data, predictions, fixed_agent=artifact["agent"])
    if args.limit:
        cases = cases[: args.limit]

    agent = NewsTriageLLMAgent(
        artifact=artifact,
        skill_registry=SkillRegistry(),
        max_steps=10,
    )
    manifest = {
        "full_agent": agent.manifest(require_skill=True),
        "no_skill_ablation": agent.manifest(require_skill=False),
        "sample_seed": SAMPLE_SEED,
        "cases": [
            {key: value for key, value in case.items() if key != "text"}
            for case in cases
        ],
    }
    (RESULTS / "pipeline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    representative_ids = [case["task_id"] for case in cases if case["subset"] == "representative"]
    failure_plan: dict[str, dict[str, int]] = {}
    if len(representative_ids) >= 2:
        failure_plan[representative_ids[0]] = {"classify_topic": 1}
        failure_plan[representative_ids[-1]] = {"retrieve_evidence": 1}
    hard_ids = [case["task_id"] for case in cases if case["subset"] == "hard_error"]
    if hard_ids:
        failure_plan[hard_ids[0]] = {"classify_topic": 1}

    if args.run in ("full", "all"):
        run_mode("full", cases, agent, failure_plan)
    if args.run in ("no_skill", "all"):
        run_mode("no_skill", cases, agent, failure_plan={})
    if args.run in ("consistency", "all"):
        if not (RESULTS / "full_tasks.jsonl").exists():
            raise SystemExit("run full evaluation before consistency")
        run_consistency(cases, agent)
    if args.run in ("all", "metrics"):
        build_metrics(cases)
        summary = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "mode": "metrics",
                    "case_count": summary["case_count"],
                    "full": summary["full"],
                    "no_skill": summary["no_skill"],
                    "consistency": summary["consistency"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
