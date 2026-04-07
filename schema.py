from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TodoItem(BaseModel):
    id: str = Field(...)
    content: str = Field(...)
    status: str = Field(...)


class Plan(BaseModel):
    goal: str = Field(...)
    todos: list[TodoItem] = Field(...)


class ActionDecision(BaseModel):
    action: str = Field(...)
    reason: str = Field(...)
    tool_name: str | None = Field(default=None)
    tool_input: dict[str, Any] | None = Field(default=None)


class ToolRequest(BaseModel):
    tool_name: str = Field(...)
    tool_input: dict[str, Any] = Field(...)
    todo_id: str | None = Field(default=None)


class ToolResult(BaseModel):
    tool_name: str = Field(...)
    tool_input: dict[str, Any] = Field(...)
    tool_output: str = Field(...)
    success: bool = Field(...)
    todo_id: str | None = Field(default=None)


class MemoryWrite(BaseModel):
    profile_updates: dict[str, Any] = Field(default_factory=dict)
    episode_summary: str = Field(...)
    tags: list[str] = Field(default_factory=list)


class StructuredFinalAnswer(BaseModel):
    answer: str = Field(..., description="最终回答")
    sources: list[str] = Field(default_factory=list, description="引用到的文件或信息来源")
    next_actions: list[str] = Field(default_factory=list, description="下一步建议")
    memory_write: MemoryWrite = Field(..., description="本次运行要写入 memory 的内容")


class IntentDecision(BaseModel):
    intent: Literal["new_task", "continue_task", "follow_up"] = Field(
        ...,
        description="当前用户输入的意图类型"
    )
    reason: str = Field(..., description="为什么这样判断")


class ReplanDecision(BaseModel):
    should_replan: bool = Field(
        ...,
        description="是否需要重新规划"
    )
    reason: str = Field(..., description="为什么需要/不需要重新规划")
    new_or_modified_todos: list[dict[str, Any]] = Field(
        default_factory=list,
        description="如果需要 replan，提供新增或修改的 todo 内容"
    )
    replan_count: int = Field(default=0, description="当前已 replan 的次数")