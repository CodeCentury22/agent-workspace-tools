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
    new_text: str,
    create_if_missing: bool = False,
) -> Dict[str, Any]:
    """Inserts `new_text` at a 1-based line boundary without disturbing surrounding lines."""
    try:
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
    content: str,
    ensure_newline: bool = True,
) -> Dict[str, Any]:
    """Appends `content` to the end of a file, creating the file when it does not exist."""
    try:
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
    pattern: str,
    replace_text: str,
    count: int = 0,
    case_sensitive: bool = True,
) -> Dict[str, Any]:
    """Regex find & replace, best for refactors where matches vary in whitespace/casing."""
    try:
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
    _func_schema("write_file", "Writes content to a file, atomically. Overwrites ENTIRE file. Use replace_in_file for edits.", {"file_path": {"type": "string"}, "code_body": {"type": "string"}, "overwrite": {"type": "boolean", "default": True}}, ["file_path", "code_body"]),
    _func_schema("replace_in_file", "Replaces an exact text block in a file with new text. ALWAYS use this instead of write_file for edits.", {"file_path": {"type": "string"}, "search_text": {"type": "string"}, "replace_text": {"type": "string"}, "replace_all": {"type": "boolean", "default": True}}, ["file_path", "search_text", "replace_text"]),
    _func_schema("insert_lines", "Inserts new text at an exact 1-based line boundary without rewriting surrounding content. Use to add imports, functions, or config entries precisely.", {"file_path": {"type": "string"}, "insert_line": {"type": "integer"}, "new_text": {"type": "string"}, "create_if_missing": {"type": "boolean", "default": False}}, ["file_path", "insert_line", "new_text"]),
    _func_schema("append_to_file", "Appends content to the end of a file, creating the file when it does not exist.", {"file_path": {"type": "string"}, "content": {"type": "string"}, "ensure_newline": {"type": "boolean", "default": True}}, ["file_path", "content"]),
    _func_schema("regex_replace_in_file", "Performs a regex find-and-replace in a file. Prefer over replace_in_file for refactors where matches vary in whitespace or casing.", {"file_path": {"type": "string"}, "pattern": {"type": "string"}, "replace_text": {"type": "string"}, "count": {"type": "integer", "default": 0}, "case_sensitive": {"type": "boolean", "default": True}}, ["file_path", "pattern", "replace_text"]),
    _func_schema("apply_patch", "Applies a git-style unified diff ('---'/'+++' headers with '@@' hunks) across one or more files atomically. Preferred for multi-file changes.", {"patch_text": {"type": "string"}, "directory": {"type": "string", "default": "."}, "allow_delete": {"type": "boolean", "default": True}}, ["patch_text"]),
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