#!/usr/bin/env python3
"""Offline tests for the LLM controller and tool loop."""

from __future__ import annotations

import json

from news_agent.artifacts import load_artifact
from news_agent.runtime import NewsTriageLLMAgent
from news_agent.skill_registry import SkillRegistry


def tool_call(call_id: str, name: str, arguments: dict | None = None) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}),
        },
    }


def response(call: dict) -> dict:
    return {
        "model": "fake-controller",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
    }


class ScriptedClient:
    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls
        self.index = 0
        self.model = "fake-controller"

    def complete(self, messages, tools, temperature=0.0):
        del messages, tools, temperature
        call = self.calls[self.index]
        self.index += 1
        return response(call)


def make_agent(calls: list[dict]) -> NewsTriageLLMAgent:
    return NewsTriageLLMAgent(
        artifact=load_artifact(),
        client=ScriptedClient(calls),
        skill_registry=SkillRegistry(),
        max_steps=10,
    )


def test_skill_and_tools_produce_auto_route() -> None:
    calls = [
        tool_call("1", "list_skills"),
        tool_call("2", "load_skill", {"name": "news-topic-classification"}),
        tool_call("3", "validate_text"),
        tool_call("4", "classify_topic"),
        tool_call("5", "retrieve_evidence", {"top_k": 3}),
        tool_call(
            "6",
            "submit_result",
            {
                "outcome": "auto_route",
                "label": "comp.graphics",
                "confidence": 0.99,
                "rationale": "Model and retrieved evidence both support graphics.",
            },
        ),
    ]
    agent = make_agent(calls)
    result = agent.run(
        "Computer graphics rendering uses polygon geometry, texture mapping, and image files. "
        "A graphics card accelerates shading and raster output.",
        task_id="auto",
    )
    assert result["status"] == "auto_route"
    assert result["suggested_label"] == "comp.graphics"
    assert result["skill_loaded"]["name"] == "news-topic-classification"
    assert [item["tool"] for item in result["tool_trace"]][0] == "list_skills"


def test_invalid_input_is_rejected_not_labeled() -> None:
    calls = [
        tool_call("1", "list_skills"),
        tool_call("2", "load_skill", {"name": "news-topic-classification"}),
        tool_call("3", "validate_text"),
        tool_call(
            "4",
            "submit_result",
            {
                "outcome": "rejected",
                "label": "none",
                "confidence": 0.0,
                "rationale": "Validation rejected the input as too short.",
            },
        ),
    ]
    result = make_agent(calls).run("too short", task_id="invalid")
    assert result["status"] == "rejected"
    assert result["suggested_label"] is None
    assert result["validation"]["accepted"] is False


def test_transient_tool_failure_is_visible_and_retried() -> None:
    calls = [
        tool_call("1", "list_skills"),
        tool_call("2", "load_skill", {"name": "news-topic-classification"}),
        tool_call("3", "validate_text"),
        tool_call("4", "classify_topic"),
        tool_call("5", "classify_topic"),
        tool_call("6", "retrieve_evidence", {"top_k": 5}),
        tool_call(
            "7",
            "submit_result",
            {
                "outcome": "auto_route",
                "label": "comp.graphics",
                "confidence": 0.95,
                "rationale": "The retried classification and retrieval support graphics.",
            },
        ),
    ]
    result = make_agent(calls).run(
        "A graphics renderer converts raster images and displays texture-mapped polygons. "
        "The software uses a graphics file format and a shading pipeline.",
        task_id="failure",
        inject_failures={"classify_topic": 1},
    )
    classify_trace = [
        item for item in result["tool_trace"] if item["tool"] == "classify_topic"
    ]
    assert classify_trace[0]["result"]["ok"] is False
    assert classify_trace[1]["result"]["ok"] is True
    assert result["status"] == "auto_route"


if __name__ == "__main__":
    test_skill_and_tools_produce_auto_route()
    test_invalid_input_is_rejected_not_labeled()
    test_transient_tool_failure_is_visible_and_retried()
    print("3 tests passed")
