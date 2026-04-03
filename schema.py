from __future__ import annotations
from pydantic import BaseModel, Field
"""
约束Agent输出格式;
"""
class Plan(BaseModel):
    """规划Agent:负责制定整体的计划和策略，确定需要完成的任务和目标。"""
    goal: str = Field(..., description="整体目标")
    steps: list[str] = Field(..., description="实现目标的步骤列表")

class ActionDecision(BaseModel):
    """决策Agent:负责根据当前的情况和计划做出具体的行动决策，选择最合适的行动方案。"""
    action: str = Field(..., description="当前执行动作")
    reason: str = Field(..., description="执行动作的原因")

class FinalAnswer(BaseModel):
    """总结Agent:负责总结和归纳整个过程的结果，形成最终的答案或结论。"""
    answer: str = Field(..., description="最终的答案或结论")