from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SkillMetadata:
    name: str
    description: str
    skill_dir: str
    resources: list[str]


@dataclass
class SkillBundle:
    metadata: SkillMetadata
    full_content: str


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return value.strip('"').strip("'")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text

    metadata: dict[str, Any] = {}
    body_start = None

    for idx in range(1, len(lines)):
        line = lines[idx].rstrip()
        if line.strip() == "---":
            body_start = idx + 1
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value)

    if body_start is None:
        return {}, text

    body = "\n".join(lines[body_start:]).strip()
    return metadata, body


class SkillRegistry:
    def __init__(self, skills_root: str = "skills") -> None:
        self.skills_root = Path(skills_root)
        self.skills_root.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[str, SkillMetadata] = {}

    def scan(self) -> None:
        self._metadata.clear()

        for skill_dir in self.skills_root.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            raw = skill_file.read_text(encoding="utf-8")
            frontmatter, _ = _parse_frontmatter(raw)

            name = str(frontmatter.get("name", skill_dir.name)).strip()
            description = str(frontmatter.get("description", "")).strip()
            resources = frontmatter.get("resources", [])
            if isinstance(resources, str):
                resources = [resources]

            self._metadata[name] = SkillMetadata(
                name=name,
                description=description,
                skill_dir=str(skill_dir),
                resources=resources if isinstance(resources, list) else [],
            )

    def list_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "skill_dir": item.skill_dir,
                "resources": item.resources,
            }
            for item in self._metadata.values()
        ]

    def load_skill(self, skill_name: str) -> SkillBundle:
        if skill_name not in self._metadata:
            raise ValueError(f"Unknown skill: {skill_name}")

        metadata = self._metadata[skill_name]
        skill_dir = Path(metadata.skill_dir)
        skill_file = skill_dir / "SKILL.md"

        raw = skill_file.read_text(encoding="utf-8")
        _, body = _parse_frontmatter(raw)

        extra_parts: list[str] = []
        for resource_name in metadata.resources:
            resource_path = skill_dir / resource_name
            if resource_path.exists() and resource_path.is_file():
                extra_parts.append(
                    f"\n\n## Resource: {resource_name}\n\n"
                    + resource_path.read_text(encoding="utf-8")
                )

        full_content = body + "".join(extra_parts)

        return SkillBundle(
            metadata=metadata,
            full_content=full_content.strip(),
        )