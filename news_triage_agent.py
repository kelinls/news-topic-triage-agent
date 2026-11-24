#!/usr/bin/env python3
"""Command-line entry point for the news triage agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "news_triage_agent.joblib"
DEMO_PATH = ROOT / "agent_results" / "demo_cases.json"


def load_agent():
    artifact = joblib.load(ARTIFACT_PATH)
    return artifact["agent"]


def print_result(result: dict) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_demo() -> int:
    cases = json.loads(DEMO_PATH.read_text(encoding="utf-8"))["cases"]
    agent = load_agent()
    for case in cases:
        print("=" * 78)
        print(case["name"])
        print("-" * 78)
        print(case["text"][:700])
        print("-" * 78)
        print_result(agent.triage(case["text"]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="News triage agent")
    parser.add_argument("--text", help="article text to triage")
    parser.add_argument("--file", type=Path, help="UTF-8 file containing article text")
    parser.add_argument("--demo", action="store_true", help="run the three built-in demo cases")
    args = parser.parse_args()

    if args.demo:
        return run_demo()
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("provide --text, --file, --demo, or pipe article text on stdin")
        return 2

    result = load_agent().triage(text)
    print_result(result)
    return 0 if result["status"] != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
