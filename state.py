from __future__ import annotations

from pydantic import BaseModel, Field

from schema import ActionDecision, Plan, TodoItem, ToolRequest, ToolResult


class AgentState(BaseModel):
    user_input: str = Field(..., description="用户原始输入")
    messages: list[dict] = Field(default_factory=list, description="对话消息历史")

    plan: Plan | None = Field(default=None, description="planner 生成的计划")
    todo_list: list[TodoItem] = Field(default_factory=list, description="运行中的 todo 列表")
    current_todo_id: str | None = Field(default=None, description="当前正在处理的 todo")

    current_action: ActionDecision | None = Field(default=None, description="executor 生成的当前动作")
    tool_request: ToolRequest | None = Field(default=None, description="待执行的工具请求")
    tool_result: ToolResult | None = Field(default=None, description="最近一次工具执行结果")

    research_notes: list[str] = Field(default_factory=list, description="压缩后的中间笔记")
    workspace_root: str = Field(default="workspace", description="本地 workspace 根目录")
    workspace_files: list[str] = Field(default_factory=list, description="已写入 workspace 的文件路径")

    step_history: list[str] = Field(default_factory=list, description="执行过程记录")
    final_answer: str | None = Field(default=None, description="最终答案")

    status: str = Field(default="initialized", description="当前运行状态")
    error: str | None = Field(default=None, description="错误信息")

    max_steps: int = Field(default=10, description="最大循环步数")
    current_step_count: int = Field(default=0, description="当前循环步数")