import fnmatch
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ---------------------------------------------------------------------
# Hallucination-tolerant alias & argument sanitization
# ---------------------------------------------------------------------
# Models (Cline, local LLMs) frequently hallucinate `modify_file` or send
# generic keys (`content`, `body`, `code`, `old_string`, ...) instead of the
# exact `search_text` / `replace_text` / `code_body` keys. These maps heal
# such mismatches in 0ms CPU time instead of tripping the circuit breaker.

FILE_PATH_ALIASES = ("path", "filepath", "file", "relative_path", "filename", "target_file")

# search_text aliases for replace-style edits
SEARCH_TEXT_ALIASES = (
    "search",
    "search_string",
    "old_text",
    "old_string",
    "old_content",
    "find_text",
    "find",
    "original_text",
    "original",
)

# replace_text aliases for replace-style edits
REPLACE_TEXT_ALIASES = (
    "replace",
    "replacement",
    "replacement_text",
    "new_text",
    "new_string",
    "new_content",
    # generic payload keys — only mapped when the canonical key is absent
    # AND the call is a replace-style edit (search present or tool is
    # replace_in_file / modify_file / regex_replace_in_file).
    "content",
    "body",
    "code",
    "code_body",
    "text",
    "file_content",
    "data",
)

# code_body aliases for full-file writes
CODE_BODY_ALIASES = (
    "code",
    "content",
    "body",
    "text",
    "new_text",
    "file_content",
    "data",
    "replace_text",
    "new_string",
    "new_content",
)

# Canonical tool aliases: hallucinated name -> canonical tool.
# `modify_file` is resolved dynamically (replace vs write) by `modify_file()`;
# the map entry documents the default route for introspection/helpers.
TOOL_ALIASES: Dict[str, str] = {
    "modify_file": "replace_in_file",
    "edit_file": "replace_in_file",
    "update_file": "replace_in_file",
    "patch_file": "apply_patch",
}


def resolve_tool_alias(tool_name: str) -> str:
    """Maps a hallucinated/alias tool name to its canonical tool name."""
    return TOOL_ALIASES.get(tool_name, tool_name)


def _pop_first_present(data: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    """Pops and returns the first present alias value, or None."""
    for name in names:
        if name in data and data[name] is not None:
            return data.pop(name)
    return None


def sanitize_tool_kwargs(tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Heals minor schema mismatches by mapping alias keys to canonical keys.

    - Never overwrites an explicitly provided canonical key.
    - `file_path` aliases apply to every file tool.
    - `search_text` / `replace_text` aliases apply to replace-style tools.
    - `code_body` aliases apply to `write_file`.
    - `content` / `body` / `code` map to `replace_text` when a search key is
      present (replace intent), otherwise to `code_body` for write intent.
    """
    if not isinstance(kwargs, dict):
        return kwargs
    data = dict(kwargs)
    canonical = resolve_tool_alias(tool_name)

    # file_path aliases (all file tools)
    if "file_path" not in data:
        found = _pop_first_present(data, FILE_PATH_ALIASES)
        if found is not None:
            data["file_path"] = found

    if canonical in ("replace_in_file", "modify_file"):
        if "search_text" not in data:
            found = _pop_first_present(data, SEARCH_TEXT_ALIASES)
            if found is not None:
                data["search_text"] = found
        if "replace_text" not in data:
            found = _pop_first_present(data, REPLACE_TEXT_ALIASES)
            if found is not None:
                data["replace_text"] = found
    elif canonical == "write_file":
        if "code_body" not in data:
            found = _pop_first_present(data, CODE_BODY_ALIASES)
            if found is not None:
                data["code_body"] = found
    elif canonical == "regex_replace_in_file":
        if "pattern" not in data:
            found = _pop_first_present(
                data, SEARCH_TEXT_ALIASES + ("search_text", "search", "regex", "find_text")
            )
            if found is not None:
                data["pattern"] = found
        if "replace_text" not in data:
            found = _pop_first_present(data, REPLACE_TEXT_ALIASES)
            if found is not None:
                data["replace_text"] = found
    elif canonical == "append_to_file":
        if "content" not in data:
            found = _pop_first_present(data, CODE_BODY_ALIASES + ("replace_text",))
            if found is not None:
                data["content"] = found
    elif canonical == "insert_lines":
        if "new_text" not in data:
            found = _pop_first_present(data, CODE_BODY_ALIASES + ("replace_text", "content"))
            if found is not None:
                data["new_text"] = found

    return data


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
def write_file(
    file_path: str,
    code_body: Optional[str] = None,
    overwrite: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        # --- Hallucination-tolerant kwargs healing (never overrides explicit args) ---
        if code_body is None:
            healed = sanitize_tool_kwargs("write_file", {"file_path": file_path, **kwargs})
            code_body = healed.get("code_body")
            if "file_path" in healed:
                file_path = healed["file_path"]
        elif kwargs:
            healed = sanitize_tool_kwargs(
                "write_file", {"file_path": file_path, "code_body": code_body, **kwargs}
            )
            file_path = healed.get("file_path", file_path)
            code_body = healed.get("code_body", code_body)
            if isinstance(healed.get("overwrite"), bool):
                overwrite = healed["overwrite"]
        if code_body is None:
            return {
                "file_path": file_path,
                "error": "code_body: Field required. Provide full file content to write.",
                "status": "ERROR",
            }
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


def _normalize_line(line: str, level: int) -> str:
    """Progressively looser line normalization for fuzzy matching.

    level 1: strip leading/trailing whitespace (tolerates indentation drift).
    level 2: also collapse internal whitespace runs to a single space
              (tolerates extra/missing spaces, tabs-vs-spaces mid-line).
    """
    if level == 1:
        return line.strip()
    return re.sub(r"\s+", " ", line).strip()


def _fuzzy_find_matches(content: str, search_text: str) -> List[Tuple[int, int]]:
    """Locates whitespace/indentation-tolerant matches of `search_text` in `content`.

    Returns a list of (start, end) character offsets into `content` for the
    exact substrings that correspond to `search_text` once minor per-line
    whitespace differences are ignored. Tries progressively looser
    normalization levels and stops at the first level that yields a match.
    Returns [] if no tolerant match is found at any level. This exists so
    that models which reproduce a code block with slightly different
    indentation, trailing spaces, or tab/space mixes are still healed
    instead of tripping a hard "search_text not found" failure.
    """
    search_lines = search_text.splitlines()
    if not search_lines:
        return []

    content_lines = content.splitlines(keepends=True)
    bare_lines = [ln.rstrip("\n").rstrip("\r") for ln in content_lines]
    window = len(search_lines)
    if window == 0 or window > len(bare_lines):
        return []

    # Precompute character offsets for the start of each content line.
    offsets: List[int] = []
    cursor = 0
    for ln in content_lines:
        offsets.append(cursor)
        cursor += len(ln)

    for level in (1, 2):
        target = [_normalize_line(l, level) for l in search_lines]
        matches: List[Tuple[int, int]] = []
        for i in range(0, len(bare_lines) - window + 1):
            candidate = [_normalize_line(l, level) for l in bare_lines[i : i + window]]
            if candidate == target:
                start = offsets[i]
                end = offsets[i + window - 1] + len(content_lines[i + window - 1])
                matches.append((start, end))
        if matches:
            return matches
    return []


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def replace_in_file(
    file_path: str,
    search_text: Optional[str] = None,
    replace_text: Optional[str] = None,
    replace_all: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        # --- Hallucination-tolerant kwargs healing (never overrides explicit args) ---
        _explicit = {
            "file_path": file_path,
            **({"search_text": search_text} if search_text is not None else {}),
            **({"replace_text": replace_text} if replace_text is not None else {}),
            **kwargs,
        }
        healed = sanitize_tool_kwargs("replace_in_file", _explicit)
        file_path = healed.get("file_path", file_path)
        search_text = healed.get("search_text", search_text)
        replace_text = healed.get("replace_text", replace_text)
        if isinstance(healed.get("replace_all"), bool):
            replace_all = healed["replace_all"]
        if search_text is None:
            return {
                "file_path": file_path,
                "error": "search_text: Field required. Provide the exact text block to find.",
                "status": "ERROR",
            }
        if replace_text is None:
            return {
                "file_path": file_path,
                "error": "replace_text: Field required. Provide the replacement text.",
                "status": "ERROR",
            }
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if not path_obj.exists():
            return {"file_path": posix_path, "error": "File does not exist. Use write_file to create new files.", "status": "ERROR"}
        if not search_text:
            return {"file_path": posix_path, "error": "search_text must not be empty.", "status": "ERROR"}

        existing_content = path_obj.read_text(encoding="utf-8")
        occurrences = existing_content.count(search_text)
        fuzzy_matched = False

        if occurrences == 0:
            # --- Fuzzy fallback: tolerate whitespace/indentation drift ---
            # Models frequently reproduce a code block with slightly different
            # leading indentation, trailing spaces, or tab/space mixes. Before
            # failing outright, retry with a whitespace-normalized line match.
            fuzzy_spans = _fuzzy_find_matches(existing_content, search_text)
            if not fuzzy_spans:
                return {
                    "file_path": posix_path,
                    "error": (
                        "The specified search_text was not found in the file, even after "
                        "a whitespace/indentation-tolerant fuzzy match attempt. Re-read the "
                        "file with read_file and copy the EXACT text block (including "
                        "original indentation) for search_text, or use apply_patch instead."
                    ),
                    "status": "ERROR",
                }
            fuzzy_matched = True
            occurrences = len(fuzzy_spans)
            spans = fuzzy_spans if replace_all else fuzzy_spans[:1]
            occurrences_replaced = len(spans)
            new_content = existing_content
            # Splice from the last match backwards so earlier offsets stay valid.
            for start, end in reversed(spans):
                new_content = new_content[:start] + replace_text + new_content[end:]
        elif replace_all:
            new_content = existing_content.replace(search_text, replace_text)
            occurrences_replaced = occurrences
        else:
            new_content = existing_content.replace(search_text, replace_text, 1)
            occurrences_replaced = 1

        if existing_content == new_content:
            return {"file_path": posix_path, "status": "NO_CHANGE", "message": "The replacement resulted in no changes."}

        _atomic_write(path_obj, new_content)

        result: Dict[str, Any] = {
            "file_path": posix_path,
            "status": "SUCCESS",
            "message": "Text successfully replaced.",
            "occurrences_replaced": occurrences_replaced,
        }
        if fuzzy_matched:
            result["fuzzy_matched"] = True
            result["message"] = (
                "Text successfully replaced using whitespace/indentation-tolerant fuzzy "
                "matching (exact search_text was not found verbatim)."
            )
        return result
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def modify_file(file_path: str, **kwargs: Any) -> Dict[str, Any]:
    """Hallucination-tolerant alias dispatcher.

    Models frequently emit `modify_file` instead of `replace_in_file` /
    `write_file`. This dispatcher heals argument aliases and routes by
    payload structure:

    - search-like payload (``search_text`` or aliases such as ``old_string``,
      ``search``, ``find``) -> :func:`replace_in_file`
    - content-only payload (``replace_text`` / ``content`` / ``body`` /
      ``code`` / ``code_body`` without any search key) -> :func:`write_file`
      (full-file overwrite; creates the file when missing).
    """
    healed = sanitize_tool_kwargs("modify_file", {"file_path": file_path, **kwargs})
    target = healed.get("file_path", file_path)
    has_search = ("search_text" in healed and healed["search_text"] is not None)

    if has_search:
        result = replace_in_file(
            target,
            healed.get("search_text"),
            healed.get("replace_text"),
            healed.get("replace_all", True),
        )
        routed_to = "replace_in_file"
    else:
        replacement = healed.get("replace_text")
        if replacement is None:
            return {
                "file_path": target,
                "error": (
                    "search_text: Field required. Provide the exact text block to find, "
                    "or provide full replacement content (content/body/code) to overwrite the file."
                ),
                "status": "ERROR",
                "routed_via": "modify_file",
            }
        result = write_file(target, replacement, overwrite=True)
        routed_to = "write_file"

    if isinstance(result, dict):
        result = dict(result)
        result.setdefault("routed_via", f"modify_file->{routed_to}")
        result.setdefault("tool", "modify_file")
    return result


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
        info: Dict[str, Any] = {
            "file_path": posix_path,
            "name": path_obj.name,
            "is_file": path_obj.is_file(),
            "is_dir": path_obj.is_dir(),
            "size_bytes": st.st_size,
            "extension": suffix.lstrip(".") if suffix else None,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "status": "SUCCESS",
        }
        if path_obj.is_file() and st.st_size <= DEFAULT_MAX_READ_BYTES:
            digest = hashlib.sha256()
            with open(posix_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    digest.update(chunk)
            info["sha256"] = digest.hexdigest()
        return info
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


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def insert_lines(
    file_path: str,
    insert_line: int,
    new_text: Optional[str] = None,
    create_if_missing: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Inserts `new_text` at a 1-based line boundary without disturbing surrounding lines."""
    try:
        healed = sanitize_tool_kwargs(
            "insert_lines", {"file_path": file_path, "insert_line": insert_line, **({"new_text": new_text} if new_text is not None else {}), **kwargs}
        )
        file_path = healed.get("file_path", file_path)
        insert_line = healed.get("insert_line", insert_line)
        new_text = healed.get("new_text", new_text)
        if isinstance(healed.get("create_if_missing"), bool):
            create_if_missing = healed["create_if_missing"]
        if new_text is None:
            return {"file_path": file_path, "error": "new_text: Field required.", "status": "ERROR"}
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if not path_obj.exists():
            if not create_if_missing:
                return {
                    "file_path": posix_path,
                    "error": "File does not exist. Pass create_if_missing=True to create it.",
                    "status": "ERROR",
                }
            lines: List[str] = []
            had_trailing_newline = True  # newly created text files end with a newline
        else:
            original_content = path_obj.read_text(encoding="utf-8")
            lines = original_content.splitlines()
            had_trailing_newline = original_content.endswith("\n")

        if insert_line < 1 or insert_line > len(lines) + 1:
            return {
                "file_path": posix_path,
                "error": f"insert_line must be between 1 and {len(lines) + 1} (total_lines={len(lines)}).",
                "insert_line": insert_line,
                "status": "ERROR",
            }

        insertion_lines = new_text.splitlines()
        new_lines = lines[: insert_line - 1] + insertion_lines + lines[insert_line - 1 :]
        new_content = "\n".join(new_lines)
        if had_trailing_newline and new_content:
            new_content += "\n"

        _atomic_write(path_obj, new_content)
        return {
            "file_path": posix_path,
            "status": "SUCCESS",
            "insert_line": insert_line,
            "lines_inserted": len(insertion_lines),
            "total_lines": len(new_lines),
        }
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def append_to_file(
    file_path: str,
    content: Optional[str] = None,
    ensure_newline: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Appends `content` to the end of a file, creating the file when it does not exist."""
    try:
        healed = sanitize_tool_kwargs(
            "append_to_file", {"file_path": file_path, **({"content": content} if content is not None else {}), **kwargs}
        )
        file_path = healed.get("file_path", file_path)
        content = healed.get("content", content)
        if isinstance(healed.get("ensure_newline"), bool):
            ensure_newline = healed["ensure_newline"]
        if content is None:
            return {"file_path": file_path, "error": "content: Field required.", "status": "ERROR"}
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        original_content = path_obj.read_text(encoding="utf-8") if path_obj.exists() else ""
        existed = path_obj.exists()
        had_trailing_newline = original_content.endswith("\n")

        if content == "":
            if existed:
                return {"file_path": posix_path, "status": "NO_CHANGE", "message": "File content is identical to existing file."}
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path_obj, "")
            return {"file_path": posix_path, "status": "SUCCESS", "chars_appended": 0, "created": True, "total_bytes": 0}

        updated = original_content
        if ensure_newline and updated and not had_trailing_newline:
            updated += "\n"
        updated += content

        path_obj.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path_obj, updated)
        return {
            "file_path": posix_path,
            "status": "SUCCESS",
            "chars_appended": len(content),
            "created": not existed,
            "total_bytes": len(updated.encode("utf-8")),
        }
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}
@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def regex_replace_in_file(
    file_path: str,
    pattern: Optional[str] = None,
    replace_text: Optional[str] = None,
    count: int = 0,
    case_sensitive: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Regex find & replace, best for refactors where matches vary in whitespace/casing."""
    try:
        _explicit: Dict[str, Any] = {"file_path": file_path, **kwargs}
        if pattern is not None:
            _explicit["pattern"] = pattern
        if replace_text is not None:
            _explicit["replace_text"] = replace_text
        healed = sanitize_tool_kwargs("regex_replace_in_file", _explicit)
        file_path = healed.get("file_path", file_path)
        pattern = healed.get("pattern", pattern)
        replace_text = healed.get("replace_text", replace_text)
        if "count" in healed:
            count = healed["count"]
        if "case_sensitive" in healed:
            case_sensitive = healed["case_sensitive"]
        if pattern is None:
            return {"file_path": file_path, "error": "pattern: Field required.", "status": "ERROR"}
        if replace_text is None:
            return {"file_path": file_path, "error": "replace_text: Field required.", "status": "ERROR"}
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if not path_obj.exists():
            return {"file_path": posix_path, "error": "File does not exist.", "status": "ERROR"}
        if not pattern or not pattern.strip():
            return {"file_path": posix_path, "error": "Pattern must not be empty.", "status": "ERROR"}

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {"file_path": posix_path, "error": f"Invalid regex pattern: {e}", "status": "ERROR"}

        existing_content = path_obj.read_text(encoding="utf-8")
        new_content, occurrences = compiled.subn(replace_text, existing_content, count=count)

        if occurrences == 0:
            return {
                "file_path": posix_path,
                "error": f"Pattern '{pattern}' did not match the file.",
                "match_count": 0,
                "status": "NO_MATCH",
            }
        if new_content == existing_content:
            return {"file_path": posix_path, "status": "NO_CHANGE", "message": "Replacement resulted in no changes."}

        _atomic_write(path_obj, new_content)
        return {
            "file_path": posix_path,
            "status": "SUCCESS",
            "match_count": occurrences,
            "bytes_written": len(new_content.encode("utf-8")),
        }
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}


@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def get_file_digest(file_path: str, algorithm: str = "sha256") -> Dict[str, Any]:
    """Computes a cryptographic digest of a file for change verification."""
    try:
        posix_path = normalize_path(file_path)
        path_obj = Path(posix_path)

        if not path_obj.exists():
            return {"file_path": posix_path, "error": "File does not exist.", "status": "ERROR"}
        if path_obj.is_dir():
            return {"file_path": posix_path, "error": "Path is a directory.", "status": "ERROR"}
        if algorithm not in hashlib.algorithms_available:
            return {"file_path": posix_path, "error": f"Unsupported hash algorithm '{algorithm}'.", "status": "ERROR"}

        digest = hashlib.new(algorithm)
        with open(posix_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)

        return {
            "file_path": posix_path,
            "algorithm": algorithm,
            "digest": digest.hexdigest(),
            "size_bytes": path_obj.stat().st_size,
            "status": "SUCCESS",
        }
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "ERROR"}
# ---------------------------------------------------------------------
# Unified diff (patch) application for multi-file atomic edits
# ---------------------------------------------------------------------

@dataclass
class _PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[Tuple[str, str]]  # (kind, text) with kind in {"ctx", "del", "add"}


@dataclass
class _FilePatch:
    old_path: Optional[str]
    new_path: Optional[str]
    hunks: List[_PatchHunk]


_PATCH_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _strip_git_prefix(path_str: str) -> str:
    if path_str.startswith(("a/", "b/")):
        return path_str[2:]
    return path_str


def _parse_unified_patch(patch_text: str) -> List[_FilePatch]:
    """Parses a git-style unified diff into ordered per-file patches (raises ValueError on corruption)."""
    file_patches: List[_FilePatch] = []
    current: Optional[_FilePatch] = None
    hunk: Optional[_PatchHunk] = None

    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith(("diff --git ", "index ", "new file mode ", "deleted file mode ", "old mode ", "new mode ")):
            continue
        if line.startswith("--- "):
            old_path = line[4:].strip()
            current = _FilePatch(
                old_path=None if old_path == "/dev/null" else _strip_git_prefix(old_path),
                new_path=None,
                hunks=[],
            )
            file_patches.append(current)
            hunk = None
            continue
        if line.startswith("+++ "):
            if current is None:
                raise ValueError("Malformed unified diff: '+++' header appears before any '---' header.")
            new_path = line[4:].strip()
            current.new_path = None if new_path == "/dev/null" else _strip_git_prefix(new_path)
            continue
        header_match = _PATCH_HUNK_RE.match(line)
        if header_match:
            if current is None:
                raise ValueError("Malformed unified diff: hunk '@@' header appears before any file header.")
            old_count = int(header_match.group(2)) if header_match.group(2) else 1
            new_count = int(header_match.group(4)) if header_match.group(4) else 1
            hunk = _PatchHunk(
                old_start=int(header_match.group(1)),
                old_count=old_count,
                new_start=int(header_match.group(3)),
                new_count=new_count,
                lines=[],
            )
            current.hunks.append(hunk)
            continue
        if hunk is not None:
            if line.startswith(" "):
                hunk.lines.append(("ctx", line[1:]))
            elif line.startswith("-"):
                hunk.lines.append(("del", line[1:]))
            elif line.startswith("+"):
                hunk.lines.append(("add", line[1:]))
            elif line.startswith("\\"):
                # "\ No newline at end of file" marker - handled by trailing-newline preservation
                continue
            continue

    # Validate hunk body counts (catches corruption & stray lines inside hunks)
    for fp in file_patches:
        for h in fp.hunks:
            expected_old = sum(1 for k, _ in h.lines if k in ("ctx", "del"))
            expected_new = sum(1 for k, _ in h.lines if k in ("ctx", "add"))
            if expected_old != h.old_count or expected_new != h.new_count:
                raise ValueError(
                    f"Malformed unified diff hunk '@@ -{h.old_start},{h.old_count} +{h.new_start},{h.new_count} @@': "
                    f"expected {h.old_count} old line(s) and {h.new_count} new line(s), "
                    f"found {expected_old} and {expected_new}."
                )
    return file_patches
@track_latency
@audit_logger(log_file="file_tools_telemetry.jsonl")
def apply_patch(
    patch_text: str,
    directory: str = ".",
    allow_delete: bool = True,
) -> Dict[str, Any]:
    """Applies a git-style unified diff across one or more files atomically (validates before writing)."""
    try:
        if not patch_text or not patch_text.strip():
            return {"error": "patch_text must not be empty.", "status": "ERROR"}

        file_patches = _parse_unified_patch(patch_text)
        if not file_patches:
            return {
                "error": "No file changes found in the patch. Ensure '---'/'+++' file headers and '@@' hunks are present.",
                "status": "ERROR",
            }

        # --- Phase 1: dry-run validation against the live filesystem ---
        planned: List[Dict[str, Any]] = []
        for fp in file_patches:
            rel = fp.new_path if fp.new_path is not None else fp.old_path
            if rel is None:
                raise ValueError("Patch file entry has neither a source nor destination path.")

            posix_path = normalize_path(os.path.join(directory, rel))
            path_obj = Path(posix_path)

            if fp.old_path is not None and fp.new_path is None:
                if not allow_delete:
                    return {"file_path": posix_path, "error": "Patch attempts to delete a file but allow_delete=False.", "status": "DENIED"}
                if not path_obj.exists():
                    planned.append(
                        {"file_path": posix_path, "operation": "deleted", "additions": 0, "deletions": 0, "content": None}
                    )
                    continue
                if path_obj.is_dir():
                    raise ValueError(f"Patch attempts to delete directory '{posix_path}' instead of a file.")
                operation = "deleted"
            elif not path_obj.exists():
                operation = "created"
            else:
                operation = "updated"

            existing = path_obj.read_text(encoding="utf-8") if path_obj.exists() else ""
            had_trailing_newline = existing.endswith("\n")
            lines = existing.splitlines()
            shift = 0
            additions = deletions = 0

            for h in fp.hunks:
                additions += sum(1 for k, _ in h.lines if k == "add")
                deletions += sum(1 for k, _ in h.lines if k == "del")
                start = h.old_start - 1 + shift
                if h.old_count == 0 and start < 0:
                    # Insertion-before-line-1 style hunks (@@ -0,0 +1,N @@)
                    start = 0
                expected = [ln for k, ln in h.lines if k in ("ctx", "del")]
                actual = lines[start : start + len(expected)]
                if actual != expected:
                    raise ValueError(
                        f"Patch does not apply cleanly to '{rel}': hunk '@@ -{h.old_start},{h.old_count} "
                        f"+{h.new_start},{h.new_count} @@' context did not match the file at line {start + 1}."
                    )
                replacement = [ln for k, ln in h.lines if k in ("ctx", "add")]
                lines[start : start + len(expected)] = replacement
                shift += len(replacement) - len(expected)

            if fp.new_path is None:
                content = None
            else:
                content = "\n".join(lines)
                if content and (operation == "created" or had_trailing_newline):
                    content += "\n"

            planned.append(
                {
                    "file_path": posix_path,
                    "operation": operation,
                    "additions": additions,
                    "deletions": deletions,
                    "content": content,
                }
            )

        # --- Phase 2: apply every planned change (all hunks already validated) ---
        results = []
        total_adds = total_dels = 0
        for plan in planned:
            target = Path(plan["file_path"])
            if plan["operation"] == "deleted":
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(target, plan["content"])
            total_adds += plan["additions"]
            total_dels += plan["deletions"]
            results.append(
                {
                    "file_path": plan["file_path"],
                    "operation": plan["operation"],
                    "additions": plan["additions"],
                    "deletions": plan["deletions"],
                }
            )

        return {
            "status": "SUCCESS",
            "message": f"Applied patch across {len(results)} file(s).",
            "files": results,
            "patch_stats": {
                "files_updated": sum(1 for r in results if r["operation"] == "updated"),
                "files_created": sum(1 for r in results if r["operation"] == "created"),
                "files_deleted": sum(1 for r in results if r["operation"] == "deleted"),
                "additions": total_adds,
                "deletions": total_dels,
            },
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


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
    _func_schema(
        "replace_in_file",
        (
            "Performs a surgical find-and-replace of an EXACT, verbatim text block inside an existing file, "
            "leaving the rest of the file untouched. This is the PREFERRED tool for editing part of a file — "
            "always use it instead of write_file whenever you only need to change a few lines. The value of "
            "'search_text' must match the file's current on-disk content character-for-character, including "
            "indentation, blank lines, and line endings; the safest way to guarantee this is to call read_file "
            "immediately beforehand and copy the exact block you intend to replace rather than retyping it from "
            "memory. Example call: "
            '{"name": "replace_in_file", "arguments": {"file_path": "src/app/utils.py", '
            '"search_text": "def add(a, b):\\n    return a + b\\n", '
            '"replace_text": "def add(a, b):\\n    return a + b  # fixed rounding\\n", "replace_all": true}}. '
            "Edge cases: if 'search_text' is not found verbatim, the tool automatically retries with a "
            "whitespace/indentation-tolerant fuzzy match before giving up (the response includes "
            "'fuzzy_matched: true' when this fallback was used) — if that also fails you get status 'ERROR' "
            "and should re-read the file and copy the block again rather than guessing; if 'search_text' and "
            "'replace_text' are identical the call returns status 'NO_CHANGE'; set 'replace_all' to false to "
            "touch only the first occurrence when the same text appears multiple times. There is no "
            "'modify_file' tool — call replace_in_file directly by name for edits."
        ),
        {
            "file_path": {
                "type": "string",
                "description": "Path to the existing file to edit, relative to the workspace root or absolute. The file must already exist; use write_file to create new files.",
            },
            "search_text": {
                "type": "string",
                "description": (
                    "The EXACT, verbatim block of text to find in the file, including original indentation and "
                    "line breaks. Prefer copying this directly from a prior read_file result rather than "
                    "retyping it, since even a single mismatched space can cause a 'not found' error (though a "
                    "whitespace-tolerant fuzzy fallback will attempt to recover from minor indentation drift)."
                ),
            },
            "replace_text": {
                "type": "string",
                "description": "The exact text that should replace every matched occurrence of 'search_text'. Provide an empty string to delete the matched block entirely.",
            },
            "replace_all": {
                "type": "boolean",
                "default": True,
                "description": "When true (default), every occurrence of 'search_text' is replaced. Set to false to replace only the first occurrence.",
            },
        },
        ["file_path", "search_text", "replace_text"],
    ),
    _func_schema(
        "modify_file",
        (
            "DEPRECATED ALIAS — do not call this tool by name; it exists only so that hallucinated calls to a "
            "generic 'modify_file' still succeed instead of tripping the circuit breaker. Internally it inspects "
            "the arguments you provide and transparently routes to replace_in_file when a search-style key "
            "('search_text' or aliases like 'old_string'/'search'/'find') is present, or to write_file (full "
            "overwrite) when only replacement content ('content'/'body'/'code'/'code_body') is supplied without "
            "any search key. Prefer calling replace_in_file (for edits) or write_file (for full overwrites) "
            "directly by their real names, since that gives you precise control and clearer error messages "
            "instead of relying on this best-effort routing guess."
        ),
        {
            "file_path": {"type": "string", "description": "Path to the file to modify."},
            "search_text": {"type": "string", "description": "Exact text to find; if provided, routes to replace_in_file."},
            "replace_text": {"type": "string", "description": "Replacement text, or full file content when no search_text is given (routes to write_file)."},
            "replace_all": {"type": "boolean", "default": True, "description": "Replace every occurrence when routed to replace_in_file."},
        },
        ["file_path"],
    ),
    _func_schema(
        "write_file",
        (
            "Writes FULL content to a file, atomically (temp-file + rename, so readers never see a "
            "partial write), completely OVERWRITING any existing file at that path or creating it if "
            "missing (parent directories are created automatically). Use this ONLY when you intend to "
            "replace the entire file contents — e.g. creating a brand-new file, or rewriting a file so "
            "small that reproducing it in full is cheaper/safer than a surgical edit. For editing part "
            "of an existing file, prefer replace_in_file or apply_patch instead, since write_file requires "
            "you to supply every single line of the file (including lines you don't want to change) or "
            "those lines WILL be lost. Example call: "
            '{"name": "write_file", "arguments": {"file_path": "src/app/utils.py", '
            '"code_body": "def add(a, b):\\n    return a + b\\n", "overwrite": true}}. '
            "Edge cases: if the file already exists and 'overwrite' is false, the call is rejected with "
            "status 'DENIED' instead of silently clobbering data; if 'code_body' is byte-for-byte identical "
            "to the current file contents, the call is a no-op and returns status 'NO_CHANGE' instead of "
            "SUCCESS. There is no 'modify_file' or 'update_file' tool — those names are aliases that route "
            "to this tool (or to replace_in_file) automatically, but you should call write_file directly by name."
        ),
        {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write, relative to the workspace root (e.g. 'src/app/utils.py') or absolute. Parent directories are created automatically if they do not exist.",
            },
            "code_body": {
                "type": "string",
                "description": (
                    "The COMPLETE new contents of the file, exactly as it should appear on disk after the "
                    "write, including every import, blank line, and trailing newline you want kept. This is "
                    "NOT a diff or a partial snippet — anything you omit will be permanently removed from the "
                    "file. Example: \"import os\\n\\ndef main():\\n    print('hi')\\n\"."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "default": True,
                "description": "If false and the file already exists, the write is refused (status 'DENIED') instead of replacing its contents. Defaults to true.",
            },
        },
        ["file_path", "code_body"],
    ),
    _func_schema("insert_lines", "Inserts new text at an exact 1-based line boundary without rewriting surrounding content. Use to add imports, functions, or config entries precisely.", {"file_path": {"type": "string"}, "insert_line": {"type": "integer"}, "new_text": {"type": "string"}, "create_if_missing": {"type": "boolean", "default": False}}, ["file_path", "insert_line", "new_text"]),
    _func_schema("append_to_file", "Appends content to the end of a file, creating the file when it does not exist.", {"file_path": {"type": "string"}, "content": {"type": "string"}, "ensure_newline": {"type": "boolean", "default": True}}, ["file_path", "content"]),
    _func_schema("regex_replace_in_file", "Performs a regex find-and-replace in a file. Prefer over replace_in_file for refactors where matches vary in whitespace or casing.", {"file_path": {"type": "string"}, "pattern": {"type": "string"}, "replace_text": {"type": "string"}, "count": {"type": "integer", "default": 0}, "case_sensitive": {"type": "boolean", "default": True}}, ["file_path", "pattern", "replace_text"]),
    _func_schema(
        "apply_patch",
        (
            "Applies one or more git-style unified diffs across one or many files in a single ATOMIC "
            "operation: every hunk in every file is validated against the current on-disk content first, "
            "and only if ALL hunks apply cleanly are ANY files actually written — if even one hunk's context "
            "does not match, nothing is changed and you get status 'ERROR'. This is the PREFERRED tool for "
            "multi-file refactors, or when you already have a diff (e.g. from git or from your own reasoning) "
            "since it is far less failure-prone than reproducing exact search_text blocks by hand. Each file "
            "entry MUST start with a '--- a/<path>' and '+++ b/<path>' header pair (use '/dev/null' as the "
            "source to create a new file, or as the destination to delete a file), followed by one or more "
            "'@@ -old_start,old_count +new_start,new_count @@' hunk headers whose body lines begin with a "
            "literal ' ' (context, unchanged), '-' (line removed), or '+' (line added) — do NOT omit the "
            "leading space/-/+ marker column, since that is how the parser distinguishes context from content. "
            "Example call: "
            '{"name": "apply_patch", "arguments": {"patch_text": '
            '"--- a/src/greet.py\\n+++ b/src/greet.py\\n@@ -1,3 +1,3 @@\\n def greet(name):\\n-    return name\\n+    return f\\"Hi, {name}\\"\\n", '
            '"directory": ".", "allow_delete": true}}. '
            "Edge cases: creating a new file uses '--- /dev/null' / '+++ b/<path>' with an '@@ -0,0 +1,N @@' "
            "hunk containing only '+' lines; deleting a file uses '--- a/<path>' / '+++ /dev/null' with all "
            "'-' lines and requires 'allow_delete' to be true or the call returns status 'DENIED'; a hunk whose "
            "old-line/new-line counts in the '@@' header do not match the actual number of context/removed/added "
            "lines is rejected as a 'Malformed unified diff' ERROR before anything is applied."
        ),
        {
            "patch_text": {
                "type": "string",
                "description": "One or more concatenated unified-diff file entries (git 'diff --git' preambles are optional and ignored; only the '---'/'+++' headers and '@@' hunks are required). See the tool description for the exact format and examples.",
            },
            "directory": {
                "type": "string",
                "default": ".",
                "description": "Base directory that the paths inside the patch headers are resolved relative to. Defaults to the current workspace root.",
            },
            "allow_delete": {
                "type": "boolean",
                "default": True,
                "description": "If false, any hunk that deletes a file (target '/dev/null') causes the whole call to be refused with status 'DENIED' instead of applied.",
            },
        },
        ["patch_text"],
    ),
    _func_schema("get_file_digest", "Returns a cryptographic hash (default sha256) of a file's contents for change verification.", {"file_path": {"type": "string"}, "algorithm": {"type": "string", "default": "sha256"}}, ["file_path"]),
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
    "modify_file": modify_file,
    "edit_file": replace_in_file,
    "update_file": replace_in_file,
    "insert_lines": insert_lines,
    "append_to_file": append_to_file,
    "regex_replace_in_file": regex_replace_in_file,
    "apply_patch": apply_patch,
    "get_file_digest": get_file_digest,
    "delete_file": delete_file,
    "list_files": list_files,
    "search_in_files": search_in_files,
    "get_file_info": get_file_info,
    "create_directory": create_directory,
    "copy_file": copy_file,
    "move_file": move_file,
}


# Backwards-compatible alias entry points (direct function references so
# `WORKSPACE_TOOL_DISPATCHER["edit_file"] is replace_in_file` holds).
edit_file = replace_in_file
update_file = replace_in_file