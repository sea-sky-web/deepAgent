from __future__ import annotations

import uuid
from pprint import pprint

from config import Settings
from llm import LLMClient
from memory_runtime import MemoryStore
from tracing_runtime import RunTracer
from thread_store import ThreadStore
from node import (
    planner_node,
    executor_node,
    tool_node,
    finalizer_node,
    memory_update_node,
    intent_router_node,
)
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


def build_tooling(settings: Settings, llm: LLMClient, workspace_root: str, skill_registry: SkillRegistry):
    backend = LocalWorkspaceBackend(root=workspace_root)

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
    tool_registry.register(LoadSkillTool(skill_registry))
    return tool_registry


def create_new_thread_state(thread_id: str, settings: Settings, memory_store: MemoryStore, skill_registry: SkillRegistry) -> AgentState:
    return AgentState(
        thread_id=thread_id,
        run_id=uuid.uuid4().hex[:12],
        user_input="",
        messages=[],
        max_steps=10,
        workspace_root=f"workspace/{thread_id}",
        available_skills=skill_registry.list_metadata(),
        memory_profile=memory_store.load_profile(),
        recent_episodes=memory_store.load_recent_episodes(limit=5),
    )


def run_one_turn(state: AgentState, llm: LLMClient, tool_registry: ToolRegistry, memory_store: MemoryStore) -> AgentState:
    intent_decision = intent_router_node(state, llm)

    if intent_decision.intent == "new_task" or not state.todo_list:
        state.plan = None
        state.todo_list = []
        state.current_todo_id = None
        state.research_notes = []
        state.workspace_files = []
        state.loaded_skills = {}
        state.tool_result_counter = 0
        state.tool_result_file_map = {}
        state = planner_node(state, llm)

    while True:
        if state.current_step_count >= state.max_steps:
            state.status = "failed"
            state.error = f"Exceeded max steps: {state.max_steps}"
            state.step_history.append("main_chat: exceeded max steps")
            break

        state.current_step_count += 1
        state = executor_node(state, llm, tool_registry)

        if state.current_action is None:
            state.status = "failed"
            state.error = "executor returned no action"
            state.step_history.append("main_chat: executor returned no action")
            break

        action = state.current_action.action.lower()

        if action == "call_tool":
            state = tool_node(state, tool_registry)
            continue

        if action == "finalize":
            state = finalizer_node(state, llm)
            state = memory_update_node(state, memory_store)
            break

        if action == "fail":
            state.status = "failed"
            state.error = state.current_action.reason
            state.step_history.append("main_chat: executor chose fail")
            break

        state.status = "failed"
        state.error = f"Unknown action: {action}"
        state.step_history.append(f"main_chat: unknown action={action}")
        break

    return state


def main() -> None:
    settings = Settings.from_env()
    llm = LLMClient(settings)
    memory_store = MemoryStore(root="memory")
    tracer = RunTracer(root="runs")
    thread_store = ThreadStore(root="threads")

    skill_registry = SkillRegistry(skills_root="skills")
    skill_registry.scan()

    thread_id = input("请输入 thread_id（新建可直接输入一个新 id）: ").strip()
    if thread_store.exists(thread_id):
        state = thread_store.load(thread_id)
        print(f"已加载 thread: {thread_id}")
    else:
        state = create_new_thread_state(thread_id, settings, memory_store, skill_registry)
        print(f"已创建新 thread: {thread_id}")

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            thread_store.save(state)
            print("会话已保存，退出。")
            break

        state.reset_turn_fields()
        state.run_id = uuid.uuid4().hex[:12]
        state.memory_profile = memory_store.load_profile()
        state.recent_episodes = memory_store.load_recent_episodes(limit=5)
        state.available_skills = skill_registry.list_metadata()

        state.append_user_message(user_input)

        tool_registry = build_tooling(
            settings=settings,
            llm=llm,
            workspace_root=state.workspace_root,
            skill_registry=skill_registry,
        )

        state = run_one_turn(state, llm, tool_registry, memory_store)

        if state.final_answer:
            state.append_assistant_message(state.final_answer)
            print(f"\n助手: {state.final_answer}")
        else:
            print(f"\n助手执行失败: {state.error}")

        thread_path = thread_store.save(state)
        trace_path = tracer.save_run(run_id=state.run_id, payload=state.model_dump())
        print(f"\n[thread saved] {thread_path}")
        print(f"[trace saved]  {trace_path}")


if __name__ == "__main__":
    main()