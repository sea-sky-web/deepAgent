from __future__ import annotations

from pprint import pprint

from config import Settings
from llm import LLMClient
from node import executor_node, finalizer_node, planner_node
from state import AgentState


def print_stage(title: str, state: AgentState) -> None:
    print(f"\n===== {title} =====")
    pprint(state.model_dump())


def main() -> None:
    settings = Settings.from_env()
    llm = LLMClient(settings)

    state = AgentState(
        user_input="帮我总结一下 RAG 的基本流程",
        messages=[
            "帮我总结一下 RAG 的基本流程"
        ],
    )

    print_stage("INITIAL STATE", state)

    try:
        state = planner_node(state, llm)
        print_stage("AFTER PLANNER", state)

        state = executor_node(state, llm)
        print_stage("AFTER EXECUTOR", state)

        if state.current_action and state.current_action.action.lower() == "finalize":
            state = finalizer_node(state, llm)
            print_stage("AFTER FINALIZER", state)
        else:
            state.status = "stopped"
            state.step_history.append("main: stopped because action is not finalize")
            print_stage("STOPPED", state)

    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        print_stage("FAILED", state)

    print("\n===== FINAL ANSWER =====")
    print(state.final_answer)


if __name__ == "__main__":
    main()