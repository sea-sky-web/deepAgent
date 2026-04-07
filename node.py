from __future__ import annotations

import json

from llm import LLMClient
from schema import ActionDecision, StructuredFinalAnswer, Plan, ToolRequest, ToolResult, TodoItem
from state import AgentState
from tools.registry import ToolRegistry
from memory_runtime import MemoryStore, EpisodeMemory


def _find_next_pending_todo(state: AgentState) -> TodoItem | None:
    for todo in state.todo_list:
        if todo.status == "pending":
            return todo
    return None


def _find_current_todo(state: AgentState) -> TodoItem | None:
    if state.current_todo_id is None:
        return None
    for todo in state.todo_list:
        if todo.id == state.current_todo_id:
            return todo
    return None


def _set_todo_status(state: AgentState, todo_id: str, new_status: str) -> None:
    for todo in state.todo_list:
        if todo.id == todo_id:
            todo.status = new_status
            return
    raise ValueError(f"Todo not found: {todo_id}")


def _all_todos_completed(state: AgentState) -> bool:
    return len(state.todo_list) > 0 and all(todo.status == "completed" for todo in state.todo_list)


def _get_completed_todo_files(state: AgentState) -> dict[str, list[str]]:
    """获取每个已完成的todo对应的workspace notes文件列表"""
    result = {}
    for todo in state.todo_list:
        if todo.status == "completed":
            todo_files = [f for f in state.workspace_files if f.startswith(f"notes/{todo.id}_")]
            if todo_files:
                result[todo.id] = todo_files
    return result


def _compact_text(text: str, limit: int = 800) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def planner_node(state: AgentState, llm: LLMClient) -> AgentState:
    system_prompt = """
你是一个任务规划器。
你需要把用户任务拆成 todo 列表。
要求：
1. todos 数量控制在 2 到 4 个
2. 每个 todo 都要有唯一 id，例如 "todo_1"
3. 初始状态必须全部是 "pending"
4. todo 内容要简洁、可执行
""".strip()

    user_prompt = f"用户输入：{state.user_input}"

    plan = llm.chat_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=Plan,
    )

    state.plan = plan
    state.todo_list = plan.todos
    state.status = "planned"
    state.step_history.append("planner: generated plan with todos")
    return state


def executor_node(state: AgentState, llm: LLMClient, tool_registry: ToolRegistry) -> AgentState:
    if _all_todos_completed(state):
        state.current_action = ActionDecision(
            action="finalize",
            reason="all todos completed",
            tool_name=None,
            tool_input=None,
        )
        state.status = "executing"
        state.step_history.append("executor: all todos completed, finalize")
        return state

    current_todo = _find_current_todo(state)
    if current_todo is None:
        next_todo = _find_next_pending_todo(state)
        if next_todo is None:
            state.current_action = ActionDecision(
                action="fail",
                reason="no pending todo found",
                tool_name=None,
                tool_input=None,
            )
            state.status = "executing"
            state.step_history.append("executor: fail because no pending todo")
            return state

        state.current_todo_id = next_todo.id
        _set_todo_status(state, next_todo.id, "in_progress")
        current_todo = next_todo
        state.step_history.append(f"executor: start todo={current_todo.id}")

    tr = state.tool_result
    if tr is not None and tr.todo_id == current_todo.id and tr.success:
        if tr.tool_name == "load_skill":
            skill_name = tr.tool_input["skill_name"]
            state.loaded_skills[skill_name] = tr.tool_output
            state.tool_result = None
            state.tool_request = None
            state.status = "executing"
            state.step_history.append(f"executor: loaded skill={skill_name}")
            # Continue to redecide action instead of returning immediately
        
        if tr.tool_name in {"tavily_search", "task"}:
            compact = _compact_text(tr.tool_output)
            state.research_notes.append(f"[{current_todo.id}] {current_todo.content}\n{compact}")

            suffix = "task" if tr.tool_name == "task" else "search"
            
            # 生成唯一的文件名：使用计数器确保不重复
            state.tool_result_counter += 1
            result_id = f"{current_todo.id}_{state.tool_result_counter}"
            note_path = f"notes/{current_todo.id}_{suffix}_{state.tool_result_counter}.md"
            
            # 记录映射关系，便于后续读取
            state.tool_result_file_map[result_id] = note_path

            state.current_action = ActionDecision(
                action="call_tool",
                reason="persist tool result to workspace before completing todo",
                tool_name="write_file",
                tool_input={"path": note_path, "content": tr.tool_output},
            )
            state.tool_request = ToolRequest(
                tool_name="write_file",
                tool_input={"path": note_path, "content": tr.tool_output},
                todo_id=current_todo.id,
            )
            state.status = "executing"
            state.step_history.append(f"executor: offload {tr.tool_name} result to {note_path}")
            return state

        if tr.tool_name == "write_file":
            path = tr.tool_input["path"]
            if path not in state.workspace_files:
                state.workspace_files.append(path)

            _set_todo_status(state, current_todo.id, "completed")
            state.step_history.append(f"executor: completed todo={current_todo.id} after workspace write")
            state.current_todo_id = None
            state.tool_result = None
            current_todo = None

            if _all_todos_completed(state):
                state.current_action = ActionDecision(
                    action="finalize",
                    reason="all todos completed after workspace write",
                    tool_name=None,
                    tool_input=None,
                )
                state.status = "executing"
                state.step_history.append("executor: finalize after last todo")
                return state

    if current_todo is None:
        next_todo = _find_next_pending_todo(state)
        if next_todo is None:
            state.current_action = ActionDecision(
                action="finalize",
                reason="no pending todo remains",
                tool_name=None,
                tool_input=None,
            )
            state.status = "executing"
            return state

        state.current_todo_id = next_todo.id
        _set_todo_status(state, next_todo.id, "in_progress")
        current_todo = next_todo
        state.step_history.append(f"executor: move to next todo={current_todo.id}")

    tools_desc = json.dumps(tool_registry.list_tools(), ensure_ascii=False, indent=2)
    completed_todo_files = _get_completed_todo_files(state)

    system_prompt = """
你是一个严格的执行决策器。
当前系统采用 todo 驱动，并支持 skills 的按需加载。

规则：
1. 如果当前 todo 需要特定领域知识或专门操作规范，而系统中存在匹配 skill，且尚未加载，优先调用 load_skill
2. 如果当前 todo 是简单信息查找，可直接用 tavily_search
3. 如果当前 todo 是复杂调研、比较、归纳、分析，可优先使用 task
4. action 只能是 "call_tool" / "finalize" / "fail"
5. 如果 action=call_tool，tool_name 必须从提供的工具列表中选择
6. tool_input 必须符合 input_schema
7. 当生成 task 工具的输入时，如果需要读取之前的调研结果，使用下方列出的 workspace notes 文件路径
""".strip()

    user_prompt = f"""
用户输入：
{state.user_input}

当前 todo：
{current_todo.model_dump_json(indent=2, ensure_ascii=False) if current_todo else "null"}

todo 列表：
{json.dumps([todo.model_dump() for todo in state.todo_list], ensure_ascii=False, indent=2)}

已完成的 todo 及其生成的 notes 文件：
{json.dumps(completed_todo_files, ensure_ascii=False, indent=2)}

workspace 文件：
{json.dumps(state.workspace_files, ensure_ascii=False, indent=2)}

最近一次工具结果：
{state.tool_result.model_dump_json(indent=2, ensure_ascii=False) if state.tool_result else "null"}

可用工具：
{tools_desc}

可用 skills（仅元数据）：
{json.dumps(state.available_skills, ensure_ascii=False, indent=2)}

已加载 skills：
{json.dumps(list(state.loaded_skills.keys()), ensure_ascii=False, indent=2)}

请输出下一步动作。
""".strip()

    decision = llm.chat_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=ActionDecision,
    )

    state.current_action = decision
    state.status = "executing"

    if decision.action == "call_tool":
        if not decision.tool_name or decision.tool_input is None:
            raise ValueError("executor returned call_tool but tool_name/tool_input is missing")

        state.tool_request = ToolRequest(
            tool_name=decision.tool_name,
            tool_input=decision.tool_input,
            todo_id=current_todo.id if current_todo else None,
        )
        state.step_history.append(
            f"executor: action=call_tool, todo={current_todo.id if current_todo else None}, "
            f"tool={decision.tool_name}, input={decision.tool_input}"
        )
    elif decision.action == "finalize":
        state.step_history.append("executor: action=finalize")
    elif decision.action == "fail":
        state.step_history.append("executor: action=fail")
    else:
        raise ValueError(f"Unsupported action from executor: {decision.action}")

    return state


def tool_node(state: AgentState, tool_registry: ToolRegistry) -> AgentState:
    if state.tool_request is None:
        raise ValueError("tool_node called but state.tool_request is None")

    tool_name = state.tool_request.tool_name
    tool_input = state.tool_request.tool_input
    todo_id = state.tool_request.todo_id

    try:
        tool = tool_registry.get(tool_name)
        output = tool.invoke(tool_input)
        result = ToolResult(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=output,
            success=True,
            todo_id=todo_id,
        )
        state.step_history.append(f"tool_node: executed {tool_name} for todo={todo_id}")
    except Exception as exc:
        result = ToolResult(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=str(exc),
            success=False,
            todo_id=todo_id,
        )
        state.step_history.append(f"tool_node: failed {tool_name} for todo={todo_id}")

    state.tool_result = result
    state.tool_request = None
    state.status = "tool_executed"
    return state


def finalizer_node(state: AgentState, llm: LLMClient) -> AgentState:
    system_prompt = """
你是最终回答生成器。
请基于用户输入、todo 执行结果、research_notes、workspace 文件和已加载 skills，
输出结构化 JSON。
""".strip()

    user_prompt = f"""
用户输入：
{state.user_input}

todo 列表：
{json.dumps([todo.model_dump() for todo in state.todo_list], ensure_ascii=False, indent=2)}

research_notes：
{json.dumps(state.research_notes, ensure_ascii=False, indent=2)}

workspace 文件：
{json.dumps(state.workspace_files, ensure_ascii=False, indent=2)}

已加载 skills：
{json.dumps(list(state.loaded_skills.keys()), ensure_ascii=False, indent=2)}

skill 内容：
{json.dumps(state.loaded_skills, ensure_ascii=False, indent=2)}

已有 profile memory：
{json.dumps(state.memory_profile, ensure_ascii=False, indent=2)}

近期 episodes：
{json.dumps(state.recent_episodes, ensure_ascii=False, indent=2)}

请输出最终结构化结果。
""".strip()

    result = llm.chat_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=StructuredFinalAnswer,
    )

    state.structured_response = result
    state.final_answer = result.answer
    state.status = "done"
    state.step_history.append("finalizer: generated structured final answer")
    return state


def memory_update_node(state: AgentState, memory_store: MemoryStore) -> AgentState:
    if state.structured_response is None:
        raise ValueError("memory_update_node called but structured_response is None")

    memory_write = state.structured_response.memory_write

    profile = memory_store.load_profile()
    profile.update(memory_write.profile_updates)
    memory_store.save_profile(profile)

    episode = EpisodeMemory(
        run_id=state.run_id,
        user_input=state.user_input,
        summary=memory_write.episode_summary,
        tags=memory_write.tags,
    )
    memory_store.append_episode(episode)

    state.memory_profile = profile
    state.step_history.append("memory_update: persisted profile and episode memory")
    return state