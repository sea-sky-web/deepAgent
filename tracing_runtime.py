from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunTracer:
    def __init__(self, root: str = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_id: str, payload: dict[str, Any]) -> str:
        target = self.root / f"{run_id}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target)