#!/usr/bin/env python3
"""Validated tools exposed to the LLM controller."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .classifier import cosine_similarities, neighbor_distribution, softmax
from .skill_registry import Skill, SkillRegistry


class ToolArgumentError(ValueError):
    """Raised when the LLM supplies invalid tool arguments."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    terminal: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolContext:
    text: str
    artifact: dict[str, Any]
    skill_registry: SkillRegistry
    require_skill: bool = True
    loaded_skill: Skill | None = None
    normalized_text: str | None = None
    validation: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None
    tool_errors: list[dict[str, Any]] = field(default_factory=list)


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if expected == "string":
        if not isinstance(value, str):
            raise ToolArgumentError(f"{path} must be a string")
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ToolArgumentError(f"{path} is shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ToolArgumentError(f"{path} is longer than {schema['maxLength']}")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolArgumentError(f"{path} must be a number")
        if "minimum" in schema and value < float(schema["minimum"]):
            raise ToolArgumentError(f"{path} is below {schema['minimum']}")
        if "maximum" in schema and value > float(schema["maximum"]):
            raise ToolArgumentError(f"{path} is above {schema['maximum']}")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolArgumentError(f"{path} must be an integer")
        if "minimum" in schema and value < int(schema["minimum"]):
            raise ToolArgumentError(f"{path} is below {schema['minimum']}")
        if "maximum" in schema and value > int(schema["maximum"]):
            raise ToolArgumentError(f"{path} is above {schema['maximum']}")
    elif expected == "object":
        if not isinstance(value, dict):
            raise ToolArgumentError(f"{path} must be an object")
    elif expected == "array":
        if not isinstance(value, list):
            raise ToolArgumentError(f"{path} must be an array")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolArgumentError(f"{path} must be one of {schema['enum']}")


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolArgumentError("tool arguments must be a JSON object")
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = required - arguments.keys()
    if missing:
        raise ToolArgumentError(f"missing required arguments: {sorted(missing)}")
    unknown = arguments.keys() - properties.keys()
    if unknown and schema.get("additionalProperties") is False:
        raise ToolArgumentError(f"unknown arguments: {sorted(unknown)}")
    for key, value in arguments.items():
        if key in properties:
            _validate_value(value, properties[key], key)


def _excerpt(text: str, limit: int = 260) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


class NewsTriageToolRegistry:
    """Task-specific tools with state guards and injected-failure support."""

    def __init__(
        self,
        context: ToolContext,
        inject_failures: dict[str, int] | None = None,
    ) -> None:
        self.context = context
        self._call_counts: dict[str, int] = {}
        self._remaining_failures = dict(inject_failures or {})
        self._tools = self._build_tools()

    def _build_tools(self) -> dict[str, ToolDefinition]:
        classes = list(self.context.artifact["agent"].target_names)
        skill_tools = [
            ToolDefinition(
                name="list_skills",
                description="List installed skills with source, license, and file hash metadata.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._list_skills,
            ),
            ToolDefinition(
                name="load_skill",
                description=(
                    "Discover and load the task skill. Load the skill before the "
                    "other task-specific tools. Returns the complete codebook and procedure."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 3, "maxLength": 64}
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                handler=self._load_skill,
            ),
        ] if self.context.require_skill else []
        tools = skill_tools + [
            ToolDefinition(
                name="validate_text",
                description=(
                    "Normalize the article and check empty, too-short, and too-long input. "
                    "Returns whether classification may proceed."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._validate_text,
            ),
            ToolDefinition(
                name="classify_topic",
                description=(
                    "Run the trained TF-IDF + LinearSVC model on the validated article. "
                    "Returns candidate labels, decision scores, confidence, margin, and features."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._classify_topic,
            ),
            ToolDefinition(
                name="retrieve_evidence",
                description=(
                    "Retrieve the most similar training articles and their label distribution. "
                    "Use this to test whether the model candidate is supported by evidence."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8,
                        }
                    },
                    "required": ["top_k"],
                    "additionalProperties": False,
                },
                handler=self._retrieve_evidence,
            ),
            ToolDefinition(
                name="request_human_review",
                description=(
                    "Terminate the task with a human-review outcome when evidence is weak, "
                    "conflicting, out of scope, or a required tool cannot recover."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "minLength": 8, "maxLength": 1200},
                        "uncertainty": {
                            "type": "string",
                            "enum": [
                                "low_confidence",
                                "evidence_conflict",
                                "invalid_input",
                                "tool_failure",
                                "out_of_scope",
                            ],
                        },
                        "suggested_label": {
                            "type": "string",
                            "enum": classes + ["none"],
                        },
                    },
                    "required": ["reason", "uncertainty", "suggested_label"],
                    "additionalProperties": False,
                },
                handler=self._request_human_review,
                terminal=True,
            ),
            ToolDefinition(
                name="submit_result",
                description=(
                    "Terminally submit an accepted topic or reject uncodeable input. "
                    "For ambiguous or conflicting cases use request_human_review instead."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "enum": ["auto_route", "rejected"],
                        },
                        "label": {
                            "type": "string",
                            "enum": classes + ["none"],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 8,
                            "maxLength": 600,
                        },
                    },
                    "required": ["outcome", "label", "confidence", "rationale"],
                    "additionalProperties": False,
                },
                handler=self._submit_result,
                terminal=True,
            ),
        ]
        return {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        self._call_counts[name] = self._call_counts.get(name, 0) + 1
        if self._remaining_failures.get(name, 0) > 0:
            self._remaining_failures[name] -= 1
            error = {
                "ok": False,
                "error": "transient_tool_failure",
                "detail": f"{name} failed once during execution; retry if appropriate",
                "retryable": True,
            }
            self.context.tool_errors.append({"tool": name, **error})
            return error
        try:
            arguments = json.loads(raw_arguments or "{}")
            validate_arguments(arguments, self._tools[name].parameters)
            result = self._tools[name].handler(arguments)
            return {"ok": True, **result}
        except (json.JSONDecodeError, ToolArgumentError, ValueError) as exc:
            return {"ok": False, "error": "invalid_tool_arguments", "detail": str(exc)}
        except Exception as exc:  # Tool failures must be visible to the controller.
            error = {
                "ok": False,
                "error": type(exc).__name__,
                "detail": str(exc),
                "retryable": False,
            }
            self.context.tool_errors.append({"tool": name, **error})
            return error

    def _require_skill(self) -> None:
        if self.context.require_skill and self.context.loaded_skill is None:
            raise ValueError("load the task skill before using this tool")

    def _list_skills(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        return {"skills": self.context.skill_registry.list()}

    def _load_skill(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill = self.context.skill_registry.load(arguments["name"])
        self.context.loaded_skill = skill
        return {
            "skill": skill.public_metadata(),
            "content": skill.content,
        }

    def _validate_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        self._require_skill()
        agent = self.context.artifact["agent"]
        text = (self.context.text or "").replace("\x00", " ").strip()
        reason = None
        if len(text) < agent.min_input_chars:
            reason = f"text is shorter than {agent.min_input_chars} characters"
        elif len(text) > agent.max_input_chars:
            reason = f"text exceeds {agent.max_input_chars} characters"
        accepted = reason is None
        result = {
            "accepted": accepted,
            "characters": len(text),
            "min_characters": agent.min_input_chars,
            "max_characters": agent.max_input_chars,
            "reason": reason,
        }
        self.context.normalized_text = text
        self.context.validation = result
        return result

    def _require_valid_text(self) -> str:
        self._require_skill()
        if self.context.validation is None:
            raise ValueError("call validate_text before this tool")
        if not self.context.validation["accepted"]:
            raise ValueError("text did not pass validation")
        assert self.context.normalized_text is not None
        return self.context.normalized_text

    def _classify_topic(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        text = self._require_valid_text()
        agent = self.context.artifact["agent"]
        matrix = agent.vectorizer.transform([text])
        scores = np.asarray(agent.classifier.decision_function(matrix))
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        probabilities = softmax(scores, agent.policy.temperature)[0]
        order = np.argsort(probabilities)[::-1]
        top_index = int(order[0])
        second_index = int(order[1])
        label = agent.target_names[top_index]
        alternatives = [
            {
                "label": agent.target_names[int(index)],
                "label_zh": agent.label_names_zh.get(agent.target_names[int(index)], ""),
                "confidence": round(float(probabilities[index]), 6),
                "decision_score": round(float(scores[0, index]), 6),
            }
            for index in order[:3]
        ]
        result = {
            "candidate_label": label,
            "candidate_label_zh": agent.label_names_zh.get(label, ""),
            "confidence": round(float(probabilities[top_index]), 6),
            "margin": round(float(probabilities[top_index] - probabilities[second_index]), 6),
            "alternatives": alternatives,
            "top_features": agent._top_features(matrix, top_index),
            "note": "confidence is softmax(model scores), not a calibrated probability",
        }
        self.context.classification = result
        return result

    def _retrieve_evidence(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = self._require_valid_text()
        if self.context.classification is None:
            raise ValueError("call classify_topic before retrieve_evidence")
        agent = self.context.artifact["agent"]
        matrix = agent.vectorizer.transform([text])
        similarities = cosine_similarities(matrix, agent.train_matrix)
        distribution, order = neighbor_distribution(
            similarities,
            agent.train_labels,
            len(agent.target_names),
            int(arguments["top_k"]),
        )
        neighbor_index = int(np.argmax(distribution))
        neighbors = []
        for rank, index in enumerate(order, start=1):
            label = agent.target_names[int(agent.train_labels[index])]
            neighbors.append(
                {
                    "rank": rank,
                    "label": label,
                    "label_zh": agent.label_names_zh.get(label, label),
                    "similarity": round(float(similarities[index]), 6),
                    "excerpt": _excerpt(agent.train_documents[index]),
                }
            )
        distribution_rows = [
            {
                "label": agent.target_names[index],
                "label_zh": agent.label_names_zh.get(agent.target_names[index], ""),
                "share": round(float(distribution[index]), 6),
            }
            for index in np.argsort(distribution)[::-1]
            if float(distribution[index]) > 0
        ]
        candidate = self.context.classification["candidate_label"]
        result = {
            "neighbor_label": agent.target_names[neighbor_index],
            "neighbor_label_zh": agent.label_names_zh.get(agent.target_names[neighbor_index], ""),
            "neighbor_agreement": round(float(distribution[neighbor_index]), 6),
            "candidate_matches_evidence": candidate == agent.target_names[neighbor_index],
            "label_distribution": distribution_rows,
            "neighbors": neighbors,
        }
        self.context.evidence = result
        return result

    def _request_human_review(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._require_skill()
        suggested = arguments["suggested_label"]
        if suggested == "none" and self.context.classification is not None:
            suggested = self.context.classification["candidate_label"]
        result = {
            "status": "human_review",
            "suggested_label": None if suggested == "none" else suggested,
            "reason": arguments["reason"],
            "uncertainty": arguments["uncertainty"],
        }
        self.context.final_result = result
        return result

    def _submit_result(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._require_skill()
        outcome = arguments["outcome"]
        label = arguments["label"]
        if outcome == "rejected":
            if self.context.validation is None or self.context.validation["accepted"]:
                raise ValueError("rejected outcome requires failed validate_text")
            if label != "none":
                raise ValueError("rejected outcome requires label 'none'")
        else:
            if self.context.classification is None:
                raise ValueError("auto_route requires classify_topic")
            if self.context.evidence is None:
                raise ValueError("auto_route requires retrieve_evidence")
            if label == "none":
                raise ValueError("auto_route requires a codebook label")
        result = {
            "status": outcome,
            "suggested_label": None if label == "none" else label,
            "confidence": arguments["confidence"],
            "reason": arguments["rationale"],
        }
        self.context.final_result = result
        return result
