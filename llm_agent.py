"""Compatibility imports for the original agent module."""

from news_agent.artifacts import ARTIFACT_PATH, load_artifact
from news_agent.deepseek import DEFAULT_ENDPOINT, DEFAULT_MODEL, DeepSeekClient, LLMClientError
from news_agent.prompts import system_prompt, user_prompt
from news_agent.runtime import NewsTriageLLMAgent
