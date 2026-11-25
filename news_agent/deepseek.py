"""DeepSeek Chat Completions client."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import certifi


DEFAULT_MODEL = "deepseek-flash"
DEFAULT_ENDPOINT = "https://api.deepseek.com/chat/completions"


class LLMClientError(RuntimeError):
    """Raised when the external LLM service cannot return a usable response."""


@dataclass
class DeepSeekClient:
    api_key: str
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    timeout_seconds: int = 90
    max_retries: int = 3

    @classmethod
    def from_environment(cls) -> "DeepSeekClient":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise LLMClientError("DEEPSEEK_API_KEY is not set")
        return cls(
            api_key=api_key,
            model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
            endpoint=os.environ.get("DEEPSEEK_ENDPOINT", DEFAULT_ENDPOINT),
        )

    def _request_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(
            request, timeout=self.timeout_seconds, context=context
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = self._request_once(payload)
                if not data.get("choices"):
                    raise LLMClientError("response contains no choices")
                return data
            except urllib.error.HTTPError as exc:
                last_error = exc
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_retries:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
            time.sleep(min(8.0, 1.0 * (2**attempt)))
        raise LLMClientError(f"LLM request failed after retries: {type(last_error).__name__}")
