import pytest
from pathlib import Path

from agent_workspace_tools.file_ops import (
    FILE_TOOLS_SCHEMA,
    FILE_TOOL_DISPATCHER,
    read_file,
    write_file,
    replace_in_file,
    delete_file,
    list_files,
    search_in_files,
    get_file_info,
    create_directory,
    copy_file,
    move_file,
)
from agent_workspace_tools.utils import normalize_path, PathEscapeError

EXPECTED_TOOL_NAMES = {
    "read_file",
    "write_file",
    "replace_in_file",
    "delete_file",
    "list_files",
    "search_in_files",
    "get_file_info",
    "create_directory",
    "copy_file",
    "move_file",
}

# ---------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------

def test_file_tools_schema_structure():
    tool_names = {t["function"]["name"] for t in FILE_TOOLS_SCHEMA}
    assert tool_names == EXPECTED_TOOL_NAMES

def test_tool_dispatcher_mapping():
    assert set(FILE_TOOL_DISPATCHER) == EXPECTED_TOOL_NAMES
    for name in FILE_TOOL_DISPATCHER:
        assert callable(FILE_TOOL_DISPATCHER[name])

# ---------------------------------------------------------------------
# Path normalization & sandbox
# ---------------------------------------------------------------------

def test_normalize_path_sanitization():
    messy_path = "../../../james/Desktop/workspace/src/app/features/pizzeria.ts"
    normalized = normalize_path(messy_path)
    assert "src/app/features/pizzeria.ts" in normalized
    assert "james" not in normalized

def test_normalize_path_backslashes():
    normalized = normalize_path("workspace\\example.py")
    assert "workspace/example.py" in normalized
    assert "\\" not in normalized

def test_normalize_path_empty_and_dot():
    assert normalize_path("") == Path.cwd().as_posix()
    assert normalize_path(".") == Path.cwd().as_posix()
    assert normalize_path("..") == Path.cwd().as_posix()

def test_normalize_path_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    inside = tmp_path / "ok.txt"
    inside.write_text("x")
    assert normalize_path("ok.txt") == inside.resolve().as_posix()
    assert normalize_path(str(inside)) == inside.resolve().as_posix()

    with pytest.raises(PathEscapeError):
        normalize_path("/etc/passwd")

def test_normalize_path_sandbox_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    assert normalize_path("/etc/passwd", enforce_sandbox=False) == Path("/etc/passwd").resolve().as_posix()

# ---------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------

def test_write_and_read_file(tmp_path):
    test_file = str(tmp_path / "sample.py")
    content = "print('Hello Agent')"

    write_res = write_file(test_file, content)
    assert write_res["status"] == "SUCCESS"
    assert write_res["bytes_written"] > 0
    assert write_res["file_path"] == Path(test_file).resolve().as_posix()

    read_res = read_file(test_file)
    assert read_res["status"] == "SUCCESS"
    assert read_res["content"] == content

def test_write_file_no_change(tmp_path):
    test_file = str(tmp_path / "identical.txt")
    content = "Same content"

    write_file(test_file, content)
    res = write_file(test_file, content)
    assert res["status"] == "NO_CHANGE"
    assert "identical" in res["message"]

def test_write_file_no_overwrite(tmp_path):
    test_file = str(tmp_path / "existing.txt")
    write_file(test_file, "Initial content")

    res = write_file(test_file, "New Content", overwrite=False)
    assert res["status"] == "DENIED"
    assert "overwrite" in res["error"]

def test_write_is_atomic_no_temp_leftovers(tmp_path):
    test_file = str(tmp_path / "atomic.txt")
    write_file(test_file, "data")
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".atomic.txt")]
    assert leftovers == []

def test_read_file_line_range(tmp_path):
    test_file = str(tmp_path / "many.txt")
    write_file(test_file, "\n".join(f"line {i}" for i in range(1, 11)))

    res = read_file(test_file, start_line=3, end_line=5)
    assert res["status"] == "SUCCESS"
    assert res["content"] == "line 3\nline 4\nline 5"
    assert res["total_lines"] == 10
    assert res["line_count"] == 3

def test_read_file_range_beyond_eof(tmp_path):
    test_file = str(tmp_path / "short.txt")
    write_file(test_file, "a\nb\n")
    res = read_file(test_file, start_line=1, end_line=99)
    assert res["status"] == "SUCCESS"
    assert res["total_lines"] == 2
    assert res["truncated"] is True

def test_read_file_over_size_limit(tmp_path):
    test_file = str(tmp_path / "big.txt")
    write_file(test_file, "x" * 100)
    res = read_file(test_file, max_bytes=50)
    assert res["status"] == "ERROR"
    assert "safety limit" in res["error"]

def test_read_missing_file(tmp_path):
    res = read_file(str(tmp_path / "nope.txt"))
    assert res["status"] == "ERROR"

def test_read_directory_returns_error(tmp_path):
    (tmp_path / "somedir").mkdir()
    res = read_file(str(tmp_path / "somedir"))
    assert res["status"] == "ERROR"

# ---------------------------------------------------------------------
# replace / delete / list / search
# ---------------------------------------------------------------------

def test_replace_in_file(tmp_path):
    test_file = str(tmp_path / "replace_test.txt")
    write_file(test_file, "def hello():\n    print('old code')\n")

    res = replace_in_file(test_file, "print('old code')", "print('new code')")
    assert res["status"] == "SUCCESS"
    assert "new code" in read_file(test_file)["content"]

    res_not_found = replace_in_file(test_file, "print('missing code')", "print('new code')")
    assert res_not_found["status"] == "ERROR"
    assert "not found" in res_not_found["error"]

    res_no_change = replace_in_file(test_file, "print('new code')", "print('new code')")
    assert res_no_change["status"] == "NO_CHANGE"

def test_replace_in_file_single_occurrence(tmp_path):
    test_file = str(tmp_path / "multi.txt")
    write_file(test_file, "spam spam spam")

    res = replace_in_file(test_file, "spam", "eggs", replace_all=False)
    assert res["status"] == "SUCCESS"
    assert res["occurrences_replaced"] == 1
    assert read_file(test_file)["content"] == "eggs spam spam"

    res_all = replace_in_file(test_file, "spam", "ham")
    assert res_all["occurrences_replaced"] == 2
    assert read_file(test_file)["content"] == "eggs ham ham"

def test_replace_in_file_missing_file(tmp_path):
    res = replace_in_file(str(tmp_path / "absent.txt"), "a", "b")
    assert res["status"] == "ERROR"

def test_delete_file(tmp_path):
    test_file = str(tmp_path / "temp.txt")
    write_file(test_file, "To be deleted")

    del_res = delete_file(test_file)
    assert del_res["status"] == "SUCCESS"
    assert not Path(test_file).exists()

    del_res2 = delete_file(test_file)
    assert del_res2["status"] == "ERROR"

def test_list_files(tmp_path):
    write_file(str(tmp_path / "a.txt"), "A")
    write_file(str(tmp_path / "b.txt"), "B")

    res = list_files(str(tmp_path))
    assert res["status"] == "SUCCESS"
    assert res["files"] == ["a.txt", "b.txt"]
    assert res["count"] == 2

def test_list_files_recursive_with_dirs_and_pattern(tmp_path):
    write_file(str(tmp_path / "mod.py"), "x")
    write_file(str(tmp_path / "test_me.py"), "x")
    (tmp_path / "pkg").mkdir()
    write_file(str(tmp_path / "pkg" / "core.py"), "x")

    res = list_files(str(tmp_path), recursive=True, include_dirs=True, pattern="*.py")
    assert res["status"] == "SUCCESS"
    assert res["files"] == ["mod.py", "pkg/core.py", "test_me.py"]
    assert res["directories"] == ["pkg"]

def test_list_files_missing_directory(tmp_path):
    res = list_files(str(tmp_path / "nope"))
    assert res["status"] == "ERROR"

def test_search_in_files(tmp_path):
    write_file(str(tmp_path / "a.py"), "def greet():\n    return 'hi'\n")
    write_file(str(tmp_path / "b.txt"), "nothing here\n")
    (tmp_path / "deep").mkdir()
    write_file(str(tmp_path / "deep" / "c.py"), "def greet():\n    pass\n")

    res = search_in_files(str(tmp_path), pattern="def greet", glob_pattern="*.py")
    assert res["status"] == "SUCCESS"
    assert res["count"] == 2
    assert {m["path"] for m in res["matches"]} == {"a.py", "deep/c.py"}
    assert {m["line_number"] for m in res["matches"]} == {1}

def test_search_in_files_case_insensitive_and_empty(tmp_path):
    write_file(str(tmp_path / "x.py"), "TODO fix me\n")
    res = search_in_files(str(tmp_path), pattern="todo", case_sensitive=False)
    assert res["count"] == 1

    res_empty = search_in_files(str(tmp_path), pattern="   ")
    assert res_empty["status"] == "ERROR"

# ---------------------------------------------------------------------
# fs utilities
# ---------------------------------------------------------------------

def test_get_file_info(tmp_path):
    target = tmp_path / "info.txt"
    write_file(str(target), "hello")
    res = get_file_info(str(target))
    assert res["status"] == "SUCCESS"
    assert res["is_file"] is True
    assert res["is_dir"] is False
    assert res["size_bytes"] == 5
    assert res["extension"] == "txt"

def test_get_file_info_missing(tmp_path):
    res = get_file_info(str(tmp_path / "nope"))
    assert res["status"] == "ERROR"

def test_create_directory(tmp_path):
    target = tmp_path / "new" / "nested" / "dir"
    res = create_directory(str(target))
    assert res["status"] == "SUCCESS"
    assert res["created"] is True
    assert target.is_dir()

def test_copy_file(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "sub" / "dst.txt"
    write_file(str(src), "copy me")

    res = copy_file(str(src), str(dst))
    assert res["status"] == "SUCCESS"
    assert dst.read_text() == "copy me"

def test_move_file(tmp_path):
    src = tmp_path / "move.txt"
    dst = tmp_path / "moved.txt"
    write_file(str(src), "moving")

    res = move_file(str(src), str(dst))
    assert res["status"] == "SUCCESS"
    assert not src.exists()
    assert dst.read_text() == "moving"