from __future__ import annotations

import json
from pathlib import Path

from state import AgentState


class ThreadStore:
    def __init__(self, root: str = "threads") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        return self.root / f"{thread_id}.json"

    def exists(self, thread_id: str) -> bool:
        return self._path(thread_id).exists()

    def load(self, thread_id: str) -> AgentState:
        path = self._path(thread_id)
        if not path.exists():
            raise ValueError(f"Thread not found: {thread_id}")
        raw = path.read_text(encoding="utf-8")
        return AgentState.model_validate_json(raw)

    def save(self, state: AgentState) -> str:
        path = self._path(state.thread_id)
        path.write_text(
            state.model_dump_json(indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)