from __future__ import annotations

from typing import Any

from llm import LLMClient
from schema import ReplanDecision
from tools.base import BaseTool


class ReplanTool(BaseTool):
    """
    动态规划工具，在执行过程中判断是否需要重新规划
    
    使用场景：
    - 执行 todo 后获得新信息，发现需要调整原计划
    - 例如搜索结果表明还需要调研其他方向
    - 自动调整 todo 列表（新增、删除、修改）
    """
    
    name = "replan"
    description = "根据已获得的信息重新评估任务计划，动态调整 todo 列表"
    input_schema = {
        "type": "object",
        "properties": {
            "current_todo_id": {
                "type": "string",
                "description": "刚完成的 todo ID"
            },
            "tool_result_summary": {
                "type": "string",
                "description": "最近一次工具执行的结果摘要"
            },
            "original_plan": {
                "type": "string",
                "description": "原始规划目标"
            },
            "current_todos": {
                "type": "string",
                "description": "当前待办列表（JSON 格式）"
            },
            "replan_count": {
                "type": "integer",
                "description": "已经 replan 的次数（防止无限循环）",
                "default": 0
            }
        },
        "required": ["current_todo_id", "tool_result_summary", "original_plan", "current_todos"],
        "additionalProperties": False,
    }

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def invoke(self, tool_input: dict[str, Any]) -> str:
        current_todo_id = tool_input["current_todo_id"]
        tool_result_summary = tool_input["tool_result_summary"]
        original_plan = tool_input["original_plan"]
        current_todos = tool_input["current_todos"]
        replan_count = tool_input.get("replan_count", 0)

        # 防止无限 replan
        if replan_count >= 2:
            return "已达到 replan 最大次数（2次），停止重新规划"

        system_prompt = """
你是一个规划评估器，需要判断是否需要动态调整任务计划。

当前机制：
1. 初始规划后开始执行 todo
2. 每完成一个 todo，获取新信息
3. 评估这些新信息是否触发规划调整

判断标准（何时需要 replan）：
- 新发现了需要额外调查的方向
- 原计划中某个 todo 已经不必要了
- 原计划的顺序需要调整
- 需要新增 todo 来完成用户的核心目标

输出格式要求：
1. should_replan: true/false
2. reason: 判断的理由
3. new_or_modified_todos: 如果需要 replan，提供具体的新增/修改 todo（格式同 TodoItem）
4. replan_count: 当前 replan 次数 + 1
""".strip()

        user_prompt = f"""
原始规划目标：
{original_plan}

当前待办列表：
{current_todos}

刚完成的 todo：
{current_todo_id}

该 todo 的执行结果摘要：
{tool_result_summary}

请判断：是否需要重新规划？

如果需要，请提供新增或修改的 todo（格式示例）：
{{
  "new_todos": [
    {{"id": "todo_4", "content": "新增的任务描述", "status": "pending"}},
  ],
  "removed_todos": ["todo_3"],  // 可选
  "explanation": "为什么做这些调整"
}}

请输出结构化判断结果。
""".strip()

        decision = self.llm.chat_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=ReplanDecision,
        )

        # 返回 JSON 字符串，便于主流程解析
        return decision.model_dump_json(ensure_ascii=False, indent=2)
