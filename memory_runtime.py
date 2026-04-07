from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class EpisodeMemory:
    run_id: str
    user_input: str
    summary: str
    tags: list[str]


class MemoryStore:
    def __init__(self, root: str = "memory") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.profile_path = self.root / "profile.json"
        self.episodes_path = self.root / "episodes.jsonl"

        if not self.profile_path.exists():
            self.profile_path.write_text("{}", encoding="utf-8")

        if not self.episodes_path.exists():
            self.episodes_path.write_text("", encoding="utf-8")

    def load_profile(self) -> dict[str, Any]:
        raw = self.profile_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)

    def save_profile(self, profile: dict[str, Any]) -> None:
        self.profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_episode(self, episode: EpisodeMemory) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")

    def load_recent_episodes(self, limit: int = 5) -> list[dict[str, Any]]:
        lines = self.episodes_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return []
        selected = lines[-limit:]
        return [json.loads(line) for line in selected if line.strip()]