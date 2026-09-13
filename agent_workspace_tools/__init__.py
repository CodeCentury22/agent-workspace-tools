import inspect
from typing import Dict, Any, List

from .utils import normalize_path, get_workspace_root, PathEscapeError
from .file_ops import (
    FILE_TOOLS_SCHEMA,
    FILE_TOOL_DISPATCHER,
    read_file,
    write_file,
    replace_in_file,
    insert_lines,
    append_to_file,
    regex_replace_in_file,
    apply_patch,
    get_file_digest,
)
from .async_runner import (
    SHELL_TOOLS_SCHEMA,
    ASYNC_TOOL_DISPATCHER,
    execute_async_subprocess,
    start_background_task,
    get_background_task_status,
    list_background_tasks,
    cancel_background_task,
)
from .git_utils import (
    get_git_status_changes,
    get_git_branch,
    get_recent_git_commits,
    get_git_diff,
    get_git_summary,
)

# Combined tool schema injected into the LLM system prompt
WORKSPACE_TOOLS_SCHEMA: List[Dict[str, Any]] = FILE_TOOLS_SCHEMA + SHELL_TOOLS_SCHEMA

# Combined tool dispatcher for tool execution loops
WORKSPACE_TOOL_DISPATCHER: Dict[str, Any] = {**FILE_TOOL_DISPATCHER, **ASYNC_TOOL_DISPATCHER}


async def dispatch_tool(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Master tool dispatcher. Intelligently handles both synchronous file tool calls
    and asynchronous shell/runner commands.
    """
    func = WORKSPACE_TOOL_DISPATCHER.get(tool_name)
    if func is None:
        return {
            "tool": tool_name,
            "error": f"Unknown tool '{tool_name}'. Available tools: {', '.join(sorted(WORKSPACE_TOOL_DISPATCHER))}",
            "status": "ERROR",
        }
    try:
        # Await async runner tools, otherwise execute sync file tools normally
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)
    except Exception as e:
        return {
            "tool": tool_name,
            "error": f"Execution failed for '{tool_name}': {str(e)}",
            "status": "ERROR",
        }


__all__ = [
    "WORKSPACE_TOOLS_SCHEMA",
    "WORKSPACE_TOOL_DISPATCHER",
    "dispatch_tool",
    "normalize_path",
    "get_workspace_root",
    "PathEscapeError",
    "execute_async_subprocess",
    "start_background_task",
    "get_background_task_status",
    "list_background_tasks",
    "cancel_background_task",
    "get_git_status_changes",
    "get_git_branch",
    "get_recent_git_commits",
    "get_git_diff",
    "get_git_summary",
]