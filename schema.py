from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TodoItem(BaseModel):
    id: str = Field(..., description="todo 唯一标识")
    content: str = Field(..., description="todo 内容")
    status: str = Field(..., description='todo 状态，只能是 "pending" / "in_progress" / "completed"')


class Plan(BaseModel):
    goal: str = Field(..., description="当前任务的总体目标")
    todos: list[TodoItem] = Field(..., description="任务拆解后的 todo 列表")


class ActionDecision(BaseModel):
    action: str = Field(..., description='当前动作，只能是 "call_tool" / "finalize" / "fail"')
    reason: str = Field(..., description="为什么做这个动作")
    tool_name: str | None = Field(default=None, description="当 action=call_tool 时，需要调用的工具名")
    tool_input: dict[str, Any] | None = Field(
        default=None,
        description="当 action=call_tool 时，传给工具的结构化输入",
    )


class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="工具名称")
    tool_input: dict[str, Any] = Field(..., description="工具结构化输入")
    todo_id: str | None = Field(default=None, description="当前工具调用所属的 todo id")


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="工具名称")
    tool_input: dict[str, Any] = Field(..., description="工具结构化输入")
    tool_output: str = Field(..., description="工具输出")
    success: bool = Field(..., description="工具是否执行成功")
    todo_id: str | None = Field(default=None, description="该结果属于哪个 todo")


class FinalAnswer(BaseModel):
    answer: str = Field(..., description="最终输出给用户的答案")