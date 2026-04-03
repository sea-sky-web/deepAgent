from __future__ import annotations
from pydantic import BaseModel, Field
from schema import ActionDecision, Plan


class ChatMessage(BaseModel):
    """单轮对话消息，兼容 OpenAI chat 格式"""
    role: str = Field(..., description="消息角色，例如 user/assistant/system")
    content: str = Field(..., description="消息内容")


class AgentState(BaseModel):
    """Agent的状态类，包含当前的计划和决策"""
    user_input: str = Field(..., description="用户输入")
    messages: list[str | ChatMessage] = Field(
        ...,
        description="对话历史，可为纯文本字符串或包含 role/content 的消息字典",
    )
    plan: Plan | None = Field(None, description="planner 生成的计划")
    current_action: ActionDecision | None = Field(None, description="executor 生成的行动决策")
    step_history: list[str] = Field(default_factory=list, description="执行过的步骤历史")
    final_answer: str | None = Field(None, description="总结Agent生成的最终答案")
    status: str = Field(default="initialized", description="当前Agent运行状态")
    error: str | None = Field(None, description="错误信息")
