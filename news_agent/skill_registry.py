#!/usr/bin/env python3
"""Discover and load project skills with source and license metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


class SkillNotFoundError(ValueError):
    """Raised when a requested skill is not installed."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    source: str
    license: str
    path: str
    sha256: str
    content: str

    def public_metadata(self) -> dict[str, str]:
        data = asdict(self)
        data.pop("content")
        return data


def _parse_front_matter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML front matter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("SKILL.md front matter is not closed")
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata


class SkillRegistry:
    """Load local, versioned skill documents exposed through a tool."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path(__file__).resolve().parents[1] / "skills").resolve()
        self._skills = self._discover()

    def _discover(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        if not self.root.exists():
            return skills
        for path in sorted(self.root.glob("*/SKILL.md")):
            content = path.read_text(encoding="utf-8")
            metadata = _parse_front_matter(content)
            required = {"name", "description", "source", "license"}
            missing = required - metadata.keys()
            if missing:
                raise ValueError(f"{path} is missing metadata: {sorted(missing)}")
            name = metadata["name"]
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", name):
                raise ValueError(f"invalid skill name {name!r}")
            skills[name] = Skill(
                name=name,
                description=metadata["description"],
                source=metadata["source"],
                license=metadata["license"],
                path=str(path.relative_to(self.root.parent)),
                sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                content=content,
            )
        return skills

    def list(self) -> list[dict[str, str]]:
        return [skill.public_metadata() for skill in self._skills.values()]

    def load(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillNotFoundError(
                f"unknown skill {name!r}; available: {sorted(self._skills)}"
            ) from exc
