"""Paths and loading for the locally trained model bundle."""

from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "news_triage_agent.joblib"


def load_artifact(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    return joblib.load(path)
