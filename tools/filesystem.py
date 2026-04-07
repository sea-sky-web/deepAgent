from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import BaseTool


class LocalWorkspaceBackend:
    def __init__(self, root: str = "workspace") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError("Path escapes workspace root")
        return candidate

    def ls(self, relative_path: str = ".") -> list[str]:
        target = self._resolve(relative_path)
        if not target.exists():
            return []
        if target.is_file():
            return [str(target.relative_to(self.root))]
        return sorted(str(p.relative_to(self.root)) for p in target.iterdir())

    def read_file(self, relative_path: str) -> str:
        target = self._resolve(relative_path)
        if not target.exists() or not target.is_file():
            raise ValueError(f"File not found: {relative_path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> str:
        target = self._resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target.relative_to(self.root))

    def append_file(self, relative_path: str, content: str) -> str:
        target = self._resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(content)
        return str(target.relative_to(self.root))


class LsWorkspaceTool(BaseTool):
    name = "ls_workspace"
    description = "列出 workspace 中某个目录下的文件"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要列出的目录，相对 workspace 根目录", "default": "."},
        },
        "additionalProperties": False,
    }

    def __init__(self, backend: LocalWorkspaceBackend) -> None:
        self.backend = backend

    def invoke(self, tool_input: dict[str, Any]) -> str:
        path = tool_input.get("path", ".")
        items = self.backend.ls(path)
        return "\n".join(items) if items else ""


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取 workspace 中某个文件的内容"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要读取的文件路径，相对 workspace 根目录"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, backend: LocalWorkspaceBackend) -> None:
        self.backend = backend

    def invoke(self, tool_input: dict[str, Any]) -> str:
        return self.backend.read_file(tool_input["path"])


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "向 workspace 写入文件，会覆盖原内容"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，相对 workspace 根目录"},
            "content": {"type": "string", "description": "文件内容"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, backend: LocalWorkspaceBackend) -> None:
        self.backend = backend

    def invoke(self, tool_input: dict[str, Any]) -> str:
        written = self.backend.write_file(
            relative_path=tool_input["path"],
            content=tool_input["content"],
        )
        return f"written: {written}"


class AppendFileTool(BaseTool):
    name = "append_file"
    description = "向 workspace 文件追加内容"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径，相对 workspace 根目录"},
            "content": {"type": "string", "description": "要追加的内容"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, backend: LocalWorkspaceBackend) -> None:
        self.backend = backend

    def invoke(self, tool_input: dict[str, Any]) -> str:
        appended = self.backend.append_file(
            relative_path=tool_input["path"],
            content=tool_input["content"],
        )
        return f"appended: {appended}"