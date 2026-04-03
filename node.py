from __future__ import annotations

from llm import LLMClient
from schema import ActionDecision, FinalAnswer, Plan
from state import AgentState


def planner_node(state: AgentState, llm: LLMClient) -> AgentState:
    system_prompt = (
        "你是一个任务规划器。"
        "你的职责是根据用户输入，生成一个非常简洁的任务目标和步骤列表。"
    )
    user_prompt = f"用户输入：{state.user_input}"

    plan = llm.chat_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=Plan,
    )

    state.plan = plan
    state.status = "planned"
    state.step_history.append("planner: generated plan")
    return state


def executor_node(state: AgentState, llm: LLMClient) -> AgentState:
    system_prompt = (
        "你是一个执行决策器。"
        "当前系统没有工具、没有子代理、没有记忆模块。"
        "所以你只能决定是否直接进入 finalizer。"
        "你必须返回 action: 'finalize'。"
    )
    user_prompt = f"""
用户输入：{state.user_input}

当前计划：
goal: {state.plan.goal if state.plan else ""}
steps: {state.plan.steps if state.plan else []}

请决定下一步动作。
""".strip()

    decision = llm.chat_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=ActionDecision,
    )

    state.current_action = decision
    state.status = "executing"
    state.step_history.append(f'executor: action={decision.action}')
    return state


def finalizer_node(state: AgentState, llm: LLMClient) -> AgentState:
    system_prompt = (
        "你是最终回答生成器。"
        "请基于用户输入、当前计划和执行动作，生成一个清晰、简洁的最终回答。"
    )

    user_prompt = f"""
用户输入：
{state.user_input}

计划：
{state.plan.model_dump_json(indent=2, ensure_ascii=False) if state.plan else "null"}

当前动作：
{state.current_action.model_dump_json(indent=2, ensure_ascii=False) if state.current_action else "null"}

请输出最终回答。
""".strip()

    result = llm.chat_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=FinalAnswer,
    )

    state.final_answer = result.answer
    state.status = "done"
    state.step_history.append("finalizer: generated final answer")
    return state