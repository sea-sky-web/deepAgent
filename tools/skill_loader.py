from __future__ import annotations

from typing import Any

from skill_runtime import SkillRegistry
from tools.base import BaseTool


class LoadSkillTool(BaseTool):
    name = "load_skill"
    description = "按需加载某个 skill 的完整内容，用于 progressive disclosure"
    input_schema = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "要加载的 skill 名称"
            }
        },
        "required": ["skill_name"],
        "additionalProperties": False,
    }

    def __init__(self, skill_registry: SkillRegistry) -> None:
        self.skill_registry = skill_registry

    def invoke(self, tool_input: dict[str, Any]) -> str:
        skill_name = tool_input["skill_name"]
        bundle = self.skill_registry.load_skill(skill_name)
        return bundle.full_content