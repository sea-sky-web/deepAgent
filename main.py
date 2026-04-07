from __future__ import annotations

import uuid
from pprint import pprint

from config import Settings
from llm import LLMClient
from memory_runtime import MemoryStore
from tracing_runtime import RunTracer
from node import executor_node, finalizer_node, planner_node, tool_node, memory_update_node
from state import AgentState
from skill_runtime import SkillRegistry
from subagents import SubAgentManager, SubAgentSpec
from tools import (
    ToolRegistry,
    TavilySearchTool,
    LocalWorkspaceBackend,
    LsWorkspaceTool,
    ReadFileTool,
    WriteFileTool,
    AppendFileTool,
    TaskTool,
    LoadSkillTool,
)


def print_stage(title: str, state: AgentState) -> None:
    print(f"\n===== {title} =====")
    pprint(state.model_dump())


def main() -> None:
    settings = Settings.from_env()
    llm = LLMClient(settings)

    memory_store = MemoryStore(root="memory")
    tracer = RunTracer(root="runs")

    backend = LocalWorkspaceBackend(root="workspace")

    tool_registry = ToolRegistry()
    tool_registry.register(TavilySearchTool(api_key=settings.tavily_api_key))
    tool_registry.register(LsWorkspaceTool(backend))
    tool_registry.register(ReadFileTool(backend))
    tool_registry.register(WriteFileTool(backend))
    tool_registry.register(AppendFileTool(backend))

    subagent_manager = SubAgentManager(llm=llm, tool_registry=tool_registry)
    subagent_manager.register(
        SubAgentSpec(
            name="researcher",
            description="专门负责联网调研、信息收集和事实归纳",
            system_prompt="优先使用检索工具，提炼事实要点，避免空泛总结。",
            allowed_tools=["tavily_search", "read_file", "ls_workspace"],
            max_steps=4,
        )
    )
    subagent_manager.register(
        SubAgentSpec(
            name="writer",
            description="专门负责根据已有材料进行结构化写作与总结",
            system_prompt="优先读取已有文件和笔记，不要重复检索。",
            allowed_tools=["read_file", "ls_workspace"],
            max_steps=3,
        )
    )

    tool_registry.register(TaskTool(subagent_manager))

    skill_registry = SkillRegistry(skills_root="skills")
    skill_registry.scan()
    tool_registry.register(LoadSkillTool(skill_registry))

    run_id = uuid.uuid4().hex[:12]

    state = AgentState(
        run_id=run_id,
        user_input="帮我调研 RAG 的 chunking 和 rerank，并给出总结,注意要有实践的技术路线",
        messages=[
            {"role": "user", "content": "帮我调研 RAG 的 chunking 和 rerank，并给出总结"}
        ],
        max_steps=10,
        workspace_root="workspace",
        available_skills=skill_registry.list_metadata(),
        memory_profile=memory_store.load_profile(),
        recent_episodes=memory_store.load_recent_episodes(limit=5),
    )

    print_stage("INITIAL STATE", state)

    try:
        state = planner_node(state, llm)
        print_stage("AFTER PLANNER", state)

        while True:
            if state.current_step_count >= state.max_steps:
                state.status = "failed"
                state.error = f"Exceeded max steps: {state.max_steps}"
                state.step_history.append("main: exceeded max steps")
                print_stage("FAILED", state)
                break

            state.current_step_count += 1

            state = executor_node(state, llm, tool_registry)
            print_stage(f"AFTER EXECUTOR STEP {state.current_step_count}", state)

            if state.current_action is None:
                state.status = "failed"
                state.error = "executor returned no action"
                state.step_history.append("main: executor returned no action")
                print_stage("FAILED", state)
                break

            action = state.current_action.action.lower()

            if action == "call_tool":
                state = tool_node(state, tool_registry)
                print_stage(f"AFTER TOOL STEP {state.current_step_count}", state)
                continue

            if action == "finalize":
                state = finalizer_node(state, llm)
                print_stage("AFTER FINALIZER", state)

                state = memory_update_node(state, memory_store)
                print_stage("AFTER MEMORY UPDATE", state)
                break

            if action == "fail":
                state.status = "failed"
                state.error = state.current_action.reason
                state.step_history.append("main: executor chose fail")
                print_stage("FAILED", state)
                break

            state.status = "failed"
            state.error = f"Unknown action: {action}"
            state.step_history.append(f"main: unknown action={action}")
            print_stage("FAILED", state)
            break

    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        print_stage("FAILED", state)

    trace_path = tracer.save_run(run_id=state.run_id, payload=state.model_dump())

    print("\n===== FINAL ANSWER =====")
    print(state.final_answer)
    print(f"\nTrace saved to: {trace_path}")


if __name__ == "__main__":
    main()