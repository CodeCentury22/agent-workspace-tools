import os
import shlex
from typing import Any, Dict, List, Tuple, Set
from .async_runner import execute_async_subprocess


async def _git_cmd(workspace_dir: str, args: List[str], timeout: float = 10.0) -> Dict[str, Any]:
    """Runs `git -C <dir> <args>` through the sanctioned async runner (HITL bypassed, internal)."""
    safe_dir = shlex.quote(workspace_dir)
    safe_args = " ".join(shlex.quote(a) for a in args)
    return await execute_async_subprocess(
        f"git -C {safe_dir} {safe_args}",
        timeout=timeout,
        bypass_hitl=True
    )


async def get_git_status_changes(workspace_dir: str) -> Tuple[bool, Set[str], Set[str]]:
    """Asynchronously queries git status in the target workspace."""
    check_res = await execute_async_subprocess(
        "git rev-parse --is-inside-work-tree",
        timeout=5.0,
        bypass_hitl=True
    )
    if check_res["returncode"] != 0 or check_res["stdout"].strip() != "true":
        return False, set(), set()

    status_res = await execute_async_subprocess(
        "git status --porcelain",
        timeout=10.0,
        bypass_hitl=True
    )
    if status_res["returncode"] != 0:
        return False, set(), set()

    files_to_update: Set[str] = set()
    files_to_delete: Set[str] = set()

    for line in status_res["stdout"].splitlines():
        if not line.strip():
            continue

        status_code = line[:2]
        file_path = line[3:].strip()

        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        full_path = os.path.join(workspace_dir, file_path)

        if "D" in status_code:
            files_to_delete.add(file_path)
        else:
            if os.path.isfile(full_path):
                files_to_update.add(file_path)

    return True, files_to_update, files_to_delete


async def get_git_branch(workspace_dir: str) -> str:
    """Returns the current branch name, or '' when not on a branch / not a git repo."""
    res = await _git_cmd(workspace_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    if res.get("returncode") != 0:
        return ""
    return res.get("stdout", "").strip()


async def get_recent_git_commits(workspace_dir: str, count: int = 10) -> List[Dict[str, str]]:
    """Returns the most recent commit hashes and one-line messages."""
    res = await _git_cmd(workspace_dir, ["log", "--oneline", "--no-decorate", f"-{count}"])
    commits: List[Dict[str, str]] = []
    if res.get("returncode") != 0:
        return commits
    for line in res.get("stdout", "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
    return commits


async def get_git_diff(workspace_dir: str, staged: bool = False, stat_only: bool = True) -> str:
    """Returns a diff (--stat by default) or empty string when no changes / not a git repo."""
    args = ["diff"]
    if staged:
        args.append("--cached")
    if stat_only:
        args.append("--stat")
    res = await _git_cmd(workspace_dir, args, timeout=10.0)
    if res.get("returncode") != 0:
        return ""
    return res.get("stdout", "").strip()


async def get_git_summary(
    workspace_dir: str,
    commits: int = 10,
    include_full_diff: bool = False,
) -> Dict[str, Any]:
    """Composes a single git context snapshot suitable for agent prompt enrichment."""
    is_git, files_to_update, files_to_delete = await get_git_status_changes(workspace_dir)
    summary: Dict[str, Any] = {
        "is_git_repo": is_git,
        "branch": "",
        "recent_commits": [],
        "diff_stat": "",
        "uncommitted_modified": sorted(files_to_update),
        "uncommitted_deleted": sorted(files_to_delete),
    }
    if is_git:
        summary["branch"] = await get_git_branch(workspace_dir)
        summary["recent_commits"] = await get_recent_git_commits(workspace_dir, commits)
        summary["diff_stat"] = await get_git_diff(workspace_dir)
        if include_full_diff:
            summary["diff"] = await get_git_diff(workspace_dir, stat_only=False)
    return summary