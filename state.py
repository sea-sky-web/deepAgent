from __future__ import annotations

from pydantic import BaseModel, Field

from schema import ActionDecision, Plan, TodoItem, ToolRequest, ToolResult, StructuredFinalAnswer


class AgentState(BaseModel):
    user_input: str = Field(...)
    messages: list[dict] = Field(default_factory=list)

    run_id: str = Field(...)

    plan: Plan | None = Field(default=None)
    todo_list: list[TodoItem] = Field(default_factory=list)
    current_todo_id: str | None = Field(default=None)

    current_action: ActionDecision | None = Field(default=None)
    tool_request: ToolRequest | None = Field(default=None)
    tool_result: ToolResult | None = Field(default=None)

    research_notes: list[str] = Field(default_factory=list)
    workspace_root: str = Field(default="workspace")
    workspace_files: list[str] = Field(default_factory=list)
    
    # 用于生成唯一的工具执行结果文件名
    tool_result_file_map: dict[str, str] = Field(default_factory=dict)  # tool_result_id -> file_path
    tool_result_counter: int = Field(default=0)  # 递增计数器用于生成唯一ID

    available_skills: list[dict] = Field(default_factory=list)
    loaded_skills: dict[str, str] = Field(default_factory=dict)

    memory_profile: dict = Field(default_factory=dict)
    recent_episodes: list[dict] = Field(default_factory=list)

    step_history: list[str] = Field(default_factory=list)

    final_answer: str | None = Field(default=None)
    structured_response: StructuredFinalAnswer | None = Field(default=None)

    status: str = Field(default="initialized")
    error: str | None = Field(default=None)

    max_steps: int = Field(default=10)
    current_step_count: int = Field(default=0)