from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm import LLMClient
from tools.registry import ToolRegistry


class SubAgentDecision(BaseModel):
    action: Literal["call_tool", "finalize", "fail"] = Field(...)
    reason: str = Field(...)
    tool_name: str | None = Field(default=None)
    tool_input: dict[str, Any] | None = Field(default=None)
    final_answer: str | None = Field(default=None)


@dataclass
class SubAgentSpec:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str]
    max_steps: int = 4


class SubAgentManager:
    def __init__(self, llm: LLMClient, tool_registry: ToolRegistry) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self._specs: dict[str, SubAgentSpec] = {}

    def register(self, spec: SubAgentSpec) -> None:
        self._specs[spec.name] = spec

    def list_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "allowed_tools": spec.allowed_tools,
                "max_steps": spec.max_steps,
            }
            for spec in self._specs.values()
        ]

    def run(self, subagent_type: str, task_input: str, expected_output: str | None = None) -> str:
        if subagent_type not in self._specs:
            raise ValueError(f"Unknown subagent_type: {subagent_type}")

        spec = self._specs[subagent_type]
        notes: list[str] = []
        last_tool_result: str | None = None

        allowed_tools = [self.tool_registry.get(name) for name in spec.allowed_tools]
        tool_desc = json.dumps(
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in allowed_tools
            ],
            ensure_ascii=False,
            indent=2,
        )

        for _ in range(spec.max_steps):
            system_prompt = f"""
你是一个子代理。
你的角色：{spec.name}
角色说明：{spec.description}

{spec.system_prompt}

你必须遵守：
1. 只能从允许的工具列表中选择工具
2. action 只能是 "call_tool" / "finalize" / "fail"
3. 如果调用工具，tool_input 必须符合工具的 input_schema
4. 当你已经有足够信息时，直接 finalize
5. 你的最终输出必须简洁、聚焦，只返回任务结果，不要暴露内部思维链
""".strip()

            user_prompt = f"""
子任务输入：
{task_input}

期望输出：
{expected_output or "无"}

已有 notes：
{json.dumps(notes, ensure_ascii=False, indent=2)}

最近一次工具结果：
{last_tool_result or "null"}

允许使用的工具：
{tool_desc}

请输出下一步动作。
""".strip()

            decision = self.llm.chat_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=SubAgentDecision,
            )

            if decision.action == "call_tool":
                if not decision.tool_name or decision.tool_input is None:
                    raise ValueError("Subagent returned call_tool but missing tool_name/tool_input")
                if decision.tool_name not in spec.allowed_tools:
                    raise ValueError(f"Tool {decision.tool_name} is not allowed for subagent {spec.name}")

                tool = self.tool_registry.get(decision.tool_name)
                output = tool.invoke(decision.tool_input)
                last_tool_result = output
                notes.append(f"[tool={decision.tool_name}] {output[:1000]}")
                continue

            if decision.action == "finalize":
                if decision.final_answer:
                    return decision.final_answer
                if notes:
                    return "\n".join(notes)
                return f"Subagent {spec.name} completed but returned empty result."

            if decision.action == "fail":
                raise ValueError(f"Subagent {spec.name} failed: {decision.reason}")

        raise ValueError(f"Subagent {spec.name} exceeded max_steps={spec.max_steps}")