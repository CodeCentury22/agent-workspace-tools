import fnmatch
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent_core_utils import track_latency, audit_logger
from .utils import (
    normalize_path,
    get_workspace_root,
    _atomic_write,
    PathEscapeError,
)

DEFAULT_MAX_READ_BYTES: int = int(
    os.environ.get("AGENT_FILE_TOOLS_MAX_READ_BYTES", "1000000")
)

SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def read_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if not path_obj.exists():
            return {"file_path": posix_path, "error": "File does not exist.", "status": "ERROR"}
        if path_obj.is_dir():
            return {"file_path": posix_path, "error": "Path is a directory. Provide a file path or use list_files.", "status": "ERROR"}

        size_limit = max_bytes if max_bytes is not None else DEFAULT_MAX_READ_BYTES
        size_bytes = path_obj.stat().st_size
        if size_bytes > size_limit:
            return {
                "file_path": posix_path,
                "error": f"File is {size_bytes} bytes, exceeding safety limit.",
                "size_bytes": size_bytes,
                "status": "ERROR",
            }

        with open(posix_path, "r", encoding="utf-8") as f:
            content = f.read()

        result: Dict[str, Any] = {
            "file_path": posix_path,
            "content": content,
            "size_bytes": size_bytes,
            "status": "SUCCESS",
        }

        if start_line is not None or end_line is not None:
            lines = content.splitlines()
            total_lines = len(lines)
            start = start_line if start_line is not None else 1
            end = end_line if end_line is not None else total_lines
            if start < 1 or end < 1 or start > end:
                return {"file_path": posix_path, "error": "Invalid line range.", "total_lines": total_lines, "status": "ERROR"}
            selected = lines[start - 1 : end]
            result.update({
                "content": "\n".join(selected),
                "start_line": start,
                "end_line": min(end, total_lines),
                "line_count": len(selected),
                "total_lines": total_lines,
            })
            if end > total_lines:
                result["truncated"] = True
        return result
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def write_file(file_path: str, code_body: str, overwrite: bool = True) -> Dict[str, Any]:
    try:
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if path_obj.exists() and not overwrite:
            return {"file_path": posix_path, "error": "File exists and overwrite is set to False.", "status": "DENIED"}

        if path_obj.exists():
            existing_content = path_obj.read_text(encoding="utf-8")
            if existing_content == code_body:
                return {
                    "file_path": posix_path,
                    "status": "NO_CHANGE",
                    "message": "File content is identical to existing file. Proceed to fix code or call 'done'.",
                }

        path_obj.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path_obj, code_body)

        return {"file_path": posix_path, "bytes_written": len(code_body.encode("utf-8")), "status": "SUCCESS"}
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def replace_in_file(
    file_path: str,
    search_text: str,
    replace_text: str,
    replace_all: bool = True,
) -> Dict[str, Any]:
    try:
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if not path_obj.exists():
            return {"file_path": posix_path, "error": "File does not exist. Use write_file to create new files.", "status": "ERROR"}
        if not search_text:
            return {"file_path": posix_path, "error": "search_text must not be empty.", "status": "ERROR"}

        existing_content = path_obj.read_text(encoding="utf-8")
        occurrences = existing_content.count(search_text)

        if occurrences == 0:
            return {"file_path": posix_path, "error": "The specified search_text was not found in the file.", "status": "ERROR"}

        if replace_all:
            new_content = existing_content.replace(search_text, replace_text)
            occurrences_replaced = occurrences
        else:
            new_content = existing_content.replace(search_text, replace_text, 1)
            occurrences_replaced = 1

        if existing_content == new_content:
            return {"file_path": posix_path, "status": "NO_CHANGE", "message": "The replacement resulted in no changes."}

        _atomic_write(path_obj, new_content)

        return {
            "file_path": posix_path,
            "status": "SUCCESS",
            "message": "Text successfully replaced.",
            "occurrences_replaced": occurrences_replaced,
        }
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def delete_file(file_path: str) -> Dict[str, Any]:
    try:
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if not path_obj.exists():
            return {"file_path": posix_path, "error": "File does not exist.", "status": "ERROR"}
        if path_obj.is_dir():
            return {"file_path": posix_path, "error": "Path is a directory.", "status": "ERROR"}

        path_obj.unlink()
        return {"file_path": posix_path, "status": "SUCCESS"}
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def list_files(
    directory: str = ".",
    recursive: bool = False,
    include_dirs: bool = False,
    pattern: Optional[str] = None,
    max_results: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        dir_obj = Path(normalize_path(directory))
        posix_dir = dir_obj.as_posix()

        if not dir_obj.exists() or not dir_obj.is_dir():
            return {"directory": posix_dir, "error": "Directory does not exist or is not a directory.", "status": "ERROR"}

        files: List[str] = []
        directories: List[str] = []

        if recursive:
            for p in sorted(dir_obj.rglob("*"), key=lambda p: p.as_posix()):
                rel = p.relative_to(dir_obj).as_posix()
                if p.is_dir() and include_dirs:
                    directories.append(rel)
                elif p.is_file():
                    files.append(rel)
        else:
            for p in sorted(dir_obj.iterdir(), key=lambda p: p.name):
                if p.is_dir() and include_dirs:
                    directories.append(p.name)
                elif p.is_file():
                    files.append(p.name)

        if pattern:
            files = [f for f in files if fnmatch.fnmatch(Path(f).name, pattern)]
        if max_results is not None:
            files = files[:max_results]

        res: Dict[str, Any] = {"directory": posix_dir, "files": files, "count": len(files), "status": "SUCCESS"}
        if include_dirs:
            res["directories"] = directories
        return res
    except Exception as e:
        return {"directory": directory, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def search_in_files(
    directory: str,
    pattern: str,
    glob_pattern: str = "*",
    max_results: int = 50,
    case_sensitive: bool = True,
) -> Dict[str, Any]:
    try:
        dir_obj = Path(normalize_path(directory))
        if not pattern or not pattern.strip():
            return {"directory": str(directory), "error": "Pattern must not be empty.", "status": "ERROR"}

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        matches: List[Dict[str, Any]] = []

        for root, dirs, files in os.walk(dir_obj):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for fname in sorted(files):
                if not fnmatch.fnmatch(fname, glob_pattern):
                    continue
                full_path = Path(root) / fname
                rel_path = full_path.relative_to(dir_obj).as_posix()
                try:
                    text = full_path.read_text(encoding="utf-8")
                except Exception:
                    continue
                if "\x00" in text:
                    continue
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        matches.append({"path": rel_path, "line_number": line_no, "line": line.rstrip()})
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        return {"directory": dir_obj.as_posix(), "pattern": pattern, "matches": matches, "count": len(matches), "status": "SUCCESS"}
    except Exception as e:
        return {"directory": directory, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def get_file_info(file_path: str) -> Dict[str, Any]:
    try:
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)
        if not path_obj.exists():
            return {"file_path": posix_path, "error": "Path does not exist.", "status": "ERROR"}

        st = path_obj.stat()
        suffix = path_obj.suffix
        return {
            "file_path": posix_path,
            "name": path_obj.name,
            "is_file": path_obj.is_file(),
            "is_dir": path_obj.is_dir(),
            "size_bytes": st.st_size,
            "extension": suffix.lstrip(".") if suffix else None,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "status": "SUCCESS",
        }
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def create_directory(directory: str) -> Dict[str, Any]:
    try:
        dir_obj = Path(normalize_path(directory))
        existed = dir_obj.exists()
        dir_obj.mkdir(parents=True, exist_ok=True)
        return {"directory": dir_obj.as_posix(), "created": not existed, "status": "SUCCESS"}
    except Exception as e:
        return {"directory": directory, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def copy_file(source_path: str, destination_path: str, overwrite: bool = False) -> Dict[str, Any]:
    try:
        src = Path(normalize_path(source_path))
        dst = Path(normalize_path(destination_path))

        if not src.exists():
            return {"source_path": src.as_posix(), "error": "Source path does not exist.", "status": "ERROR"}
        if dst.exists() and not overwrite:
            return {"source_path": src.as_posix(), "destination_path": dst.as_posix(), "error": "Destination exists.", "status": "DENIED"}

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=overwrite)
        else:
            shutil.copy2(src, dst)
        return {"source_path": src.as_posix(), "destination_path": dst.as_posix(), "status": "SUCCESS"}
    except Exception as e:
        return {"source_path": source_path, "destination_path": destination_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def move_file(source_path: str, destination_path: str, overwrite: bool = False) -> Dict[str, Any]:
    try:
        src = Path(normalize_path(source_path))
        dst = Path(normalize_path(destination_path))

        if not src.exists():
            return {"source_path": src.as_posix(), "error": "Source path does not exist.", "status": "ERROR"}
        if dst.exists() and not overwrite:
            return {"source_path": src.as_posix(), "destination_path": dst.as_posix(), "error": "Destination exists.", "status": "DENIED"}

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return {"source_path": src.as_posix(), "destination_path": dst.as_posix(), "status": "SUCCESS"}
    except Exception as e:
        return {"source_path": source_path, "destination_path": destination_path, "error": str(e), "status": "ERROR"}


def _func_schema(name: str, desc: str, props: Dict[str, Any], req: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": False},
        },
    }


FILE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    _func_schema("read_file", "Reads text content from a specified file.", {"file_path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["file_path"]),
    _func_schema("write_file", "Writes content to a file, atomically. Overwrites ENTIRE file. Use replace_in_file for edits.", {"file_path": {"type": "string"}, "code_body": {"type": "string"}, "overwrite": {"type": "boolean", "default": True}}, ["file_path", "code_body"]),
    _func_schema("replace_in_file", "Replaces an exact text block in a file with new text. ALWAYS use this instead of write_file for edits.", {"file_path": {"type": "string"}, "search_text": {"type": "string"}, "replace_text": {"type": "string"}, "replace_all": {"type": "boolean", "default": True}}, ["file_path", "search_text", "replace_text"]),
    _func_schema("delete_file", "Deletes a target file if it exists.", {"file_path": {"type": "string"}}, ["file_path"]),
    _func_schema("list_files", "Lists files in a target directory.", {"directory": {"type": "string", "default": "."}, "recursive": {"type": "boolean", "default": False}, "include_dirs": {"type": "boolean", "default": False}, "pattern": {"type": "string"}}, []),
    _func_schema("search_in_files", "Searches text files under a directory for a regex pattern with line numbers.", {"directory": {"type": "string"}, "pattern": {"type": "string"}, "glob_pattern": {"type": "string", "default": "*"}, "max_results": {"type": "integer", "default": 50}}, ["directory", "pattern"]),
    _func_schema("get_file_info", "Returns metadata about a file or directory.", {"file_path": {"type": "string"}}, ["file_path"]),
    _func_schema("create_directory", "Creates a directory and parent directories.", {"directory": {"type": "string"}}, ["directory"]),
    _func_schema("copy_file", "Copies a file or directory.", {"source_path": {"type": "string"}, "destination_path": {"type": "string"}, "overwrite": {"type": "boolean", "default": False}}, ["source_path", "destination_path"]),
    _func_schema("move_file", "Moves or renames a file or directory.", {"source_path": {"type": "string"}, "destination_path": {"type": "string"}, "overwrite": {"type": "boolean", "default": False}}, ["source_path", "destination_path"]),
]

FILE_TOOL_DISPATCHER: Dict[str, Any] = {
    "read_file": read_file,
    "write_file": write_file,
    "replace_in_file": replace_in_file,
    "delete_file": delete_file,
    "list_files": list_files,
    "search_in_files": search_in_files,
    "get_file_info": get_file_info,
    "create_directory": create_directory,
    "copy_file": copy_file,
    "move_file": move_file,
}