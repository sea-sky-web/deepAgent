from __future__ import annotations

import json

from llm import LLMClient
from schema import ActionDecision, StructuredFinalAnswer, Plan, ToolRequest, ToolResult, TodoItem, IntentDecision
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
            
            # 【新增】动态规划检查：完成 todo 后，评估是否需要重新规划
            # 只在 search/task 的写入完成后才进行 replan 检查
            # （避免对每个 write_file 都 replan）
            if state.replan_count < 2:  # 防止无限循环
                # 查找对应的原始工具结果，用于 replan 决策
                search_notes = [n for n in state.research_notes if n.startswith(f"[{current_todo.id}]")]
                if search_notes:
                    result_summary = search_notes[-1][:500]  # 取最后一条笔记的前500字
                    
                    # 此处应该返回 replan 决策，但我们选择先完成 write_file
                    # 然后在下一个 executor 循环中统一处理
                    state.step_history.append(
                        f"executor: completed todo={current_todo.id}, will check if replan is needed"
                    )
            
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
当前系统采用 todo 驱动，并支持 skills 的按需加载和动态规划。

规则：
1. 如果当前 todo 需要特定领域知识或专门操作规范，而系统中存在匹配 skill，且尚未加载，优先调用 load_skill
2. 如果当前 todo 是简单信息查找，可直接用 tavily_search
3. 如果当前 todo 是复杂调研、比较、归纳、分析，可优先使用 task
4. 【新增】如果完成 search/task 后发现需要调整规划，可调用 replan 工具动态更新 todo 列表
5. action 只能是 "call_tool" / "finalize" / "fail"
6. 如果 action=call_tool，tool_name 必须从提供的工具列表中选择
7. tool_input 必须符合 input_schema
8. 当生成 task 工具的输入时，如果需要读取之前的调研结果，使用下方列出的 workspace notes 文件路径

【动态规划（Replan）说明】
- 触发条件：完成一个 search/task 工具后，如果发现需要新增、删除或修改 todo，可调用 replan
- 限制条件：每轮最多 replan 2 次，防止无限循环
- 回滚机制：如果 replan 后发现新的 todo 仍不能满足需求，可尝试再 replan 一次（共2次上限）
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

当前 replan 次数：{state.replan_count}

可用工具：
{tools_desc}

可用 skills（仅元数据）：
{json.dumps(state.available_skills, ensure_ascii=False, indent=2)}

已加载 skills：
{json.dumps(list(state.loaded_skills.keys()), ensure_ascii=False, indent=2)}

请输出下一步动作。如果需要动态调整规划，请调用 replan 工具。
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
        
        # 【新增】处理 replan 工具的结果：解析并更新 todo_list
        if tool_name == "replan":
            try:
                import json as json_module
                replan_result = json_module.loads(output)
                if replan_result.get("should_replan"):
                    # 新增 todo
                    for new_todo_spec in replan_result.get("new_or_modified_todos", []):
                        new_todo = TodoItem(
                            id=new_todo_spec.get("id", f"todo_{len(state.todo_list)+1}"),
                            content=new_todo_spec.get("content", ""),
                            status=new_todo_spec.get("status", "pending")
                        )
                        state.todo_list.append(new_todo)
                    
                    # 更新 replan 计数
                    state.replan_count = replan_result.get("replan_count", state.replan_count + 1)
                    state.step_history.append(
                        f"tool_node: replan applied, replan_count={state.replan_count}, "
                        f"added {len(replan_result.get('new_or_modified_todos', []))} new todos"
                    )
            except Exception as e:
                state.step_history.append(f"tool_node: failed to parse replan result: {str(e)}")
        
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

重要要求（务必遵守）：
1. answer：生成简洁、结构化的最终答案
2. sources：列出所有引用的文件或信息来源
3. next_actions：基于当前任务，给出 2-3 个后续建议
4. memory_write：
   - profile_updates：【必填】提取用户在本次交互中的新特征
     * 用户关注的领域（如 "RAG", "LLM 应用"）
     * 用户的专业背景（如 "AI 工程师", "产品经理"）
     * 用户的偏好风格（如 "喜欢理论 + 实践结合"）
     * 用户的关键问题（如 "追求 RAG 工程落地"）
   - episode_summary：本次任务的完成情况总结（1-2 句话）
   - tags：5-8 个标签，便于后续检索
""".strip()

    user_prompt = f"""
用户输入：
{state.user_input}

当前任务完成情况：
{json.dumps([todo.model_dump() for todo in state.todo_list], ensure_ascii=False, indent=2)}

本次调研的关键笔记：
{json.dumps(state.research_notes[-5:] if len(state.research_notes) > 5 else state.research_notes, ensure_ascii=False, indent=2)}

已生成的文件：
{json.dumps(state.workspace_files[-5:] if len(state.workspace_files) > 5 else state.workspace_files, ensure_ascii=False, indent=2)}

已有用户侧面信息（用于更新）：
{json.dumps(state.memory_profile, ensure_ascii=False, indent=2)}

最近的交互历史（帮助理解用户背景）：
{json.dumps(state.recent_episodes[-3:] if len(state.recent_episodes) > 3 else state.recent_episodes, ensure_ascii=False, indent=2)}

请输出最终结构化结果。特别注意要填充 profile_updates，不要留空。
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


def intent_router_node(state: AgentState, llm: LLMClient) -> IntentDecision:
    system_prompt = """
你是一个会话意图路由器。
你需要判断当前用户输入属于哪一种：

1. new_task
- 用户开启了一个新主题、新任务
- 不应复用当前 todo 规划

2. continue_task
- 用户希望继续当前未完成任务
- 可以直接复用当前 todo / workspace / notes

3. follow_up
- 用户基于刚才结果追问、展开、解释
- 应复用当前上下文，但不一定重新规划 todo

判断时请结合：
- 当前用户输入
- 当前 todo 列表
- 最近几轮 messages
""".strip()

    recent_messages = state.messages[-6:] if len(state.messages) > 6 else state.messages

    user_prompt = f"""
当前用户输入：
{state.user_input}

当前 todo 列表：
{json.dumps([todo.model_dump() for todo in state.todo_list], ensure_ascii=False, indent=2)}

最近消息：
{json.dumps(recent_messages, ensure_ascii=False, indent=2)}

请输出意图判断。
""".strip()

    decision = llm.chat_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=IntentDecision,
    )

    state.step_history.append(f"intent_router: intent={decision.intent}")
    return decision