import inspect
from typing import Dict, Any, List

from .utils import normalize_path, get_workspace_root, PathEscapeError
from .file_ops import (
    FILE_TOOLS_SCHEMA,
    FILE_TOOL_DISPATCHER,
    TOOL_ALIASES,
    resolve_tool_alias,
    sanitize_tool_kwargs,
    read_file,
    write_file,
    replace_in_file,
    modify_file,
    edit_file,
    update_file,
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

    Hallucination-tolerant: alias tool names (e.g. `modify_file`) are resolved
    to canonical tools, and alias argument keys are sanitized before dispatch
    so minor schema mismatches heal instead of raising TypeError.
    """
    canonical = resolve_tool_alias(tool_name)
    # Prefer the raw name first so `modify_file` hits its smart dispatcher
    # (replace vs write routing); fall back to the canonical tool otherwise.
    func = WORKSPACE_TOOL_DISPATCHER.get(tool_name)
    if func is None:
        func = WORKSPACE_TOOL_DISPATCHER.get(canonical)
    if func is None:
        return {
            "tool": tool_name,
            "error": f"Unknown tool '{tool_name}'. Available tools: {', '.join(sorted(WORKSPACE_TOOL_DISPATCHER))}",
            "status": "ERROR",
        }
    try:
        # Sanitize file-tool kwargs (alias keys -> canonical keys).
        # NOTE: skip pre-sanitization for `modify_file` itself — its dispatcher
        # inspects raw kwargs to decide replace vs write routing.
        if tool_name != "modify_file" and (
            canonical in FILE_TOOL_DISPATCHER or tool_name in FILE_TOOL_DISPATCHER
        ):
            kwargs = sanitize_tool_kwargs(canonical, kwargs)
        # Await async runner tools, otherwise execute sync file tools normally
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            try:
                return func(**kwargs)
            except TypeError as te:
                # Last-resort healing: drop unexpected kwargs that match known
                # alias keys, then retry once before surfacing the error.
                import inspect as _inspect

                try:
                    sig = _inspect.signature(func)
                    accepts_var_kw = any(
                        p.kind == _inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                except (ValueError, TypeError):
                    accepts_var_kw = True
                if not accepts_var_kw:
                    allowed = set(sig.parameters)
                    filtered = {k: v for k, v in kwargs.items() if k in allowed}
                    if filtered != kwargs:
                        return func(**filtered)
                raise te
    except TypeError as e:
        return {
            "tool": tool_name,
            "error": (
                f"Invalid arguments for '{tool_name}' (aliased to '{canonical}'): {str(e)}. "
                f"Required keys: file_path + search_text/replace_text for edits, "
                f"or file_path + code_body for writes."
            ),
            "status": "ERROR",
        }
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
    "TOOL_ALIASES",
    "resolve_tool_alias",
    "sanitize_tool_kwargs",
    "read_file",
    "write_file",
    "replace_in_file",
    "modify_file",
    "edit_file",
    "update_file",
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