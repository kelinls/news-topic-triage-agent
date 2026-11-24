#!/usr/bin/env python3
"""CLI for the LLM-controlled news triage Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_agent import NewsTriageLLMAgent


ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "agent_results" / "demo_cases.json"


def print_result(result: dict, show_trace: bool) -> None:
    summary = {
        "task_id": result["task_id"],
        "status": result["status"],
        "suggested_label": result["suggested_label"],
        "reason": result["reason"],
        "uncertainty": result["uncertainty"],
        "skill_loaded": (result.get("skill_loaded") or {}).get("name"),
        "steps": result["steps"],
        "tool_calls": result["tool_calls"],
        "duration_seconds": result["duration_seconds"],
        "fallback": result["fallback"],
        "tool_path": [item["tool"] for item in result["tool_trace"]],
    }
    if show_trace:
        summary["tool_trace"] = result["tool_trace"]
        summary["usage"] = result["usage"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-controlled news triage Agent")
    parser.add_argument("--text", help="article text to triage")
    parser.add_argument("--file", type=Path, help="UTF-8 file containing article text")
    parser.add_argument("--demo", action="store_true", help="run the bundled demo cases")
    parser.add_argument("--trace", action="store_true", help="include the complete tool trace")
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="ablation: hide the skill tools (not the primary runtime)",
    )
    args = parser.parse_args()

    agent = NewsTriageLLMAgent()
    require_skill = not args.no_skill

    if args.demo:
        cases = json.loads(DEMO_PATH.read_text(encoding="utf-8"))["cases"][:2]
        for index, case in enumerate(cases, start=1):
            print("=" * 78)
            print(case["name"])
            result = agent.run(
                case["text"], task_id=f"demo-{index}", require_skill=require_skill
            )
            print_result(result, args.trace)
        return 0
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("provide --text, --file, --demo, or pipe text on stdin")
        return 2

    result = agent.run(text, task_id="cli", require_skill=require_skill)
    print_result(result, args.trace)
    return 0 if result["status"] != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
