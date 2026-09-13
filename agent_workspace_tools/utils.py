import os
from pathlib import Path
from typing import Optional, Set

# Known project folders used to re-anchor erratic relative paths
PROJECT_FOLDERS: Set[str] = {
    "src", "app", "libs", "components", "environments",
}


class PathEscapeError(ValueError):
    """Raised when a path would escape the configured workspace sandbox."""


def get_workspace_root() -> Path:
    """Returns the workspace root (AGENT_WORKSPACE_ROOT env var or cwd)."""
    configured = os.environ.get("AGENT_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd()


def _is_within(path: Path, root: Path) -> bool:
    """True if `path` is inside `root` (resolved paths expected)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _atomic_write(path_obj: Path, content: str) -> None:
    """Writes content atomically via a temp sibling file + os.replace."""
    tmp_path = path_obj.with_name(f".{path_obj.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path_obj)
    finally:
        tmp_path.unlink(missing_ok=True)


def normalize_path(path_str: str, enforce_sandbox: Optional[bool] = None) -> str:
    """Converts input path to an absolute, cross-platform POSIX path string."""
    if not isinstance(path_str, str) or not path_str.strip():
        path_str = "."

    posix_input = path_str.replace("\\", "/").strip()
    path_obj = Path(posix_input)

    if not path_obj.is_absolute():
        parts = path_obj.parts
        cleaned_parts = [p for p in parts if p not in (".", "..")]
        project_part = next((p for p in parts if p in PROJECT_FOLDERS), None)

        if ".." in parts or posix_input.startswith("../"):
            if project_part is not None:
                start = parts.index(project_part)
                path_obj = Path(*parts[start:])
            else:
                path_obj = Path(*cleaned_parts) if cleaned_parts else Path(".")

        path_obj = get_workspace_root() / path_obj

    resolved = path_obj.resolve()

    if enforce_sandbox is None:
        enforce_sandbox = bool(os.environ.get("AGENT_WORKSPACE_ROOT"))
    workspace_root = get_workspace_root()
    if enforce_sandbox and not _is_within(resolved, workspace_root):
        raise PathEscapeError(
            f"Path '{path_str}' resolves to '{resolved}' outside the workspace "
            f"root '{workspace_root}'. Agents may only operate inside the workspace."
        )

    return resolved.as_posix()