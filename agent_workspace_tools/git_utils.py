import os
from typing import Tuple, Set
from .async_runner import execute_async_subprocess


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