from __future__ import annotations

from typing import Any

from subagents import SubAgentManager
from tools.base import BaseTool


class TaskTool(BaseTool):
    name = "task"
    description = "把复杂子任务委派给专门子代理执行，主代理只接收最终结果"
    input_schema = {
        "type": "object",
        "properties": {
            "subagent_type": {
                "type": "string",
                "description": "子代理类型，例如 researcher / writer"
            },
            "task_input": {
                "type": "string",
                "description": "要交给子代理的具体任务"
            },
            "expected_output": {
                "type": "string",
                "description": "希望子代理返回的结果形式",
                "default": ""
            }
        },
        "required": ["subagent_type", "task_input"],
        "additionalProperties": False,
    }

    def __init__(self, manager: SubAgentManager) -> None:
        self.manager = manager

    def invoke(self, tool_input: dict[str, Any]) -> str:
        subagent_type = tool_input["subagent_type"]
        task_input = tool_input["task_input"]
        expected_output = tool_input.get("expected_output", "")

        return self.manager.run(
            subagent_type=subagent_type,
            task_input=task_input,
            expected_output=expected_output,
        )