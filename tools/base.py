from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def invoke(self, tool_input: dict[str, Any]) -> str:
        raise NotImplementedError