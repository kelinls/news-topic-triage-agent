"""Run the model/tool loop for a single news triage task."""

from __future__ import annotations

import json
import time
from typing import Any

from .artifacts import load_artifact
from .deepseek import DeepSeekClient, LLMClientError
from .prompts import system_prompt, user_prompt
from .skill_registry import SkillRegistry
from .tools import NewsTriageToolRegistry, ToolContext


class NewsTriageLLMAgent:
    """The LLM chooses tools; Python owns schemas, state, limits, and tracing."""

    def __init__(
        self,
        artifact: dict[str, Any] | None = None,
        client: DeepSeekClient | None = None,
        skill_registry: SkillRegistry | None = None,
        max_steps: int = 10,
    ) -> None:
        self.artifact = artifact or load_artifact()
        self.client = client or DeepSeekClient.from_environment()
        self.skill_registry = skill_registry or SkillRegistry()
        self.max_steps = int(max_steps)

    def manifest(self, require_skill: bool = True) -> dict[str, Any]:
        context = ToolContext(
            text="",
            artifact=self.artifact,
            skill_registry=self.skill_registry,
            require_skill=require_skill,
        )
        registry = NewsTriageToolRegistry(context)
        return {
            "model": self.client.model,
            "endpoint": self.client.endpoint,
            "temperature": 0.0,
            "max_steps": self.max_steps,
            "system_prompt": system_prompt(require_skill),
            "tools": registry.schemas(),
            "skills": self.skill_registry.list(),
            "artifact_model_version": self.artifact.get("model_version"),
            "artifact_trained_at": self.artifact.get("trained_at"),
        }

    def run(
        self,
        text: str,
        task_id: str = "task",
        require_skill: bool = True,
        inject_failures: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        context = ToolContext(
            text=text,
            artifact=self.artifact,
            skill_registry=self.skill_registry,
            require_skill=require_skill,
        )
        registry = NewsTriageToolRegistry(context, inject_failures=inject_failures)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(require_skill)},
            {"role": "user", "content": user_prompt(text)},
        ]
        tool_trace: list[dict[str, Any]] = []
        llm_trace: list[dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        llm_error = None

        for step_index in range(1, self.max_steps + 1):
            call_started = time.perf_counter()
            try:
                response = self.client.complete(messages, registry.schemas(), temperature=0.0)
            except LLMClientError as exc:
                llm_error = str(exc)
                break
            call_seconds = time.perf_counter() - call_started
            choice = response["choices"][0]
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls") or []
            response_usage = response.get("usage") or {}
            for key in usage:
                usage[key] += int(response_usage.get(key, 0) or 0)

            llm_trace.append(
                {
                    "step": step_index,
                    "latency_seconds": round(call_seconds, 4),
                    "finish_reason": choice.get("finish_reason"),
                    "model": response.get("model", self.client.model),
                    "content": message.get("content") or "",
                    "tool_calls": [
                        call.get("function", {}).get("name", "") for call in tool_calls
                    ],
                    "usage": response_usage,
                }
            )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "No terminal tool was called. Continue the task by calling an "
                            "available tool; do not answer in prose."
                        ),
                    }
                )
                continue

            for call_index, call in enumerate(tool_calls, start=1):
                function = call.get("function") or {}
                name = function.get("name", "")
                raw_arguments = function.get("arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_arguments}
                result = registry.execute(name, raw_arguments)
                tool_trace.append(
                    {
                        "step": step_index,
                        "call_in_step": call_index,
                        "call_id": call.get("id", ""),
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if context.final_result is not None:
                    break
            if context.final_result is not None:
                break

        duration = time.perf_counter() - started
        fallback = False
        if context.final_result is None:
            fallback = True
            reason = (
                "LLM service did not return a terminal decision"
                if llm_error
                else f"maximum step limit ({self.max_steps}) reached without a terminal tool"
            )
            context.final_result = {
                "status": "human_review",
                "suggested_label": None,
                "reason": reason,
                "uncertainty": "tool_failure" if llm_error else "out_of_scope",
            }

        return {
            "task_id": task_id,
            "status": context.final_result["status"],
            "suggested_label": context.final_result.get("suggested_label"),
            "reason": context.final_result.get("reason", ""),
            "uncertainty": context.final_result.get("uncertainty"),
            "skill_loaded": context.loaded_skill.public_metadata()
            if context.loaded_skill
            else None,
            "validation": context.validation,
            "classification": context.classification,
            "evidence": context.evidence,
            "steps": len(llm_trace),
            "tool_calls": len(tool_trace),
            "llm_calls": llm_trace,
            "tool_trace": tool_trace,
            "tool_errors": context.tool_errors,
            "usage": usage,
            "duration_seconds": round(duration, 4),
            "fallback": fallback,
            "llm_error": llm_error,
        }
