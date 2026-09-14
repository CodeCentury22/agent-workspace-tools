import pytest
from pathlib import Path

from agent_workspace_tools.file_ops import (
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
    "insert_lines",
    "append_to_file",
    "regex_replace_in_file",
    "apply_patch",
    "get_file_digest",
    "delete_file",
    "list_files",
    "search_in_files",
    "get_file_info",
    "create_directory",
    "copy_file",
    "move_file",
}

# Hallucinated/alias tool names that must route transparently (dispatcher-only).
EXPECTED_ALIAS_TOOL_NAMES = {
    "modify_file",
    "edit_file",
    "update_file",
}

# ---------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------

def test_file_tools_schema_structure():
    tool_names = {t["function"]["name"] for t in FILE_TOOLS_SCHEMA}
    # Canonical tools must all be advertised; modify_file alias is also advertised.
    assert EXPECTED_TOOL_NAMES <= tool_names
    assert "modify_file" in tool_names

def test_tool_dispatcher_mapping():
    assert EXPECTED_TOOL_NAMES <= set(FILE_TOOL_DISPATCHER)
    assert EXPECTED_ALIAS_TOOL_NAMES <= set(FILE_TOOL_DISPATCHER)
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

# ---------------------------------------------------------------------
# insert_lines / append_to_file
# ---------------------------------------------------------------------

def test_insert_lines_mid_file(tmp_path):
    target = tmp_path / "code.py"
    write_file(str(target), "line1\nline2\nline3\n")

    res = insert_lines(str(target), insert_line=2, new_text="inserted_a\ninserted_b")
    assert res["status"] == "SUCCESS"
    assert res["lines_inserted"] == 2
    assert res["total_lines"] == 5

    content = target.read_text()
    assert content.startswith("line1\ninserted_a\ninserted_b\nline2\nline3")
    assert content.endswith("\n")  # trailing newline preserved


def test_insert_lines_at_boundaries_and_create(tmp_path):
    target = tmp_path / "top.txt"
    write_file(str(target), "a\nb\n")

    res = insert_lines(str(target), insert_line=1, new_text="TOP")
    assert res["status"] == "SUCCESS"
    assert target.read_text().startswith("TOP\na\nb")

    res = insert_lines(str(target), insert_line=4, new_text="BOTTOM")
    assert res["status"] == "SUCCESS"
    assert target.read_text().endswith("BOTTOM\n")

    missing = tmp_path / "new.txt"
    res = insert_lines(str(missing), insert_line=1, new_text="created")
    assert res["status"] == "ERROR"
    res = insert_lines(str(missing), insert_line=1, new_text="created", create_if_missing=True)
    assert res["status"] == "SUCCESS"
    assert missing.read_text() == "created\n"


def test_insert_lines_out_of_bounds(tmp_path):
    target = tmp_path / "oob.txt"
    write_file(str(target), "only\n")
    res = insert_lines(str(target), insert_line=5, new_text="x")
    assert res["status"] == "ERROR"
    assert "insert_line" in res["error"]


def test_append_to_file_existing_and_new(tmp_path):
    target = tmp_path / "log.txt"
    write_file(str(target), "first")
    res = append_to_file(str(target), "second")
    assert res["status"] == "SUCCESS"
    assert target.read_text() == "first\nsecond"  # ensure_newline inserted between

    brand_new = tmp_path / "fresh.txt"
    res = append_to_file(str(brand_new), "hello")
    assert res["status"] == "SUCCESS"
    assert res["created"] is True
    assert brand_new.read_text() == "hello"


def test_append_to_file_no_change(tmp_path):
    target = tmp_path / "nochange.txt"
    write_file(str(target), "static")
    res = append_to_file(str(target), "")
    assert res["status"] == "NO_CHANGE"


# ---------------------------------------------------------------------
# regex_replace_in_file
# ---------------------------------------------------------------------

def test_regex_replace_in_file_all_count_and_case(tmp_path):
    target = tmp_path / "refactor.py"
    write_file(str(target), "call(arg);\ncall(  arg  );\nCALL(arg);\n")

    res = regex_replace_in_file(str(target), pattern=r"call\(\s*arg\s*\)", replace_text="invoke(arg)")
    assert res["status"] == "SUCCESS"
    assert res["match_count"] == 2  # case-sensitive -> leaves CALL untouched

    res = regex_replace_in_file(str(target), pattern=r"call\(\s*arg\s*\)", replace_text="invoke(arg)", case_sensitive=False)
    assert res["status"] == "SUCCESS"
    assert res["match_count"] == 1
    assert "CALL" not in target.read_text()

    write_file(str(target), "x,y\nx,y\nx,y\n")
    res = regex_replace_in_file(str(target), pattern=r",", replace_text=";", count=1)
    assert res["match_count"] == 1
    assert target.read_text() == "x;y\nx,y\nx,y\n"


def test_regex_replace_in_file_no_match_and_invalid(tmp_path):
    target = tmp_path / "nomatch.txt"
    write_file(str(target), "hello world")

    res = regex_replace_in_file(str(target), pattern="nope", replace_text="x")
    assert res["status"] == "NO_MATCH"
    assert res["match_count"] == 0

    res = regex_replace_in_file(str(target), pattern="([unclosed", replace_text="x")
    assert res["status"] == "ERROR"
    assert "Invalid regex" in res["error"]


# ---------------------------------------------------------------------
# get_file_digest
# ---------------------------------------------------------------------

def test_get_file_digest_matches_get_file_info(tmp_path):
    target = tmp_path / "digest.txt"
    write_file(str(target), "hello agent")

    digest_res = get_file_digest(str(target))
    assert digest_res["status"] == "SUCCESS"
    assert digest_res["algorithm"] == "sha256"
    assert len(digest_res["digest"]) == 64

    info_res = get_file_info(str(target))
    assert info_res["sha256"] == digest_res["digest"]


def test_get_file_digest_errors(tmp_path):
    res = get_file_digest(str(tmp_path / "missing.txt"))
    assert res["status"] == "ERROR"

    target = tmp_path / "badalg.txt"
    write_file(str(target), "x")
    res = get_file_digest(str(target), algorithm="not_a_real_hash_xyz")
    assert res["status"] == "ERROR"
# ---------------------------------------------------------------------
# apply_patch (unified diff application)
# ---------------------------------------------------------------------

MID_PATCH = (
    "--- a/src/greet.py\n"
    "+++ b/src/greet.py\n"
    "@@ -1,5 +1,6 @@\n"
    " def greet(name):\n"
    "     return f\"Hello, {name}\"\n"
    " \n"
    "+\n"
    " def bye():\n"
    "     return \"Goodbye\"\n"
)


def test_apply_patch_updates_multiple_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "greet.py").write_text("def greet(name):\n    return f\"Hello, {name}\"\n\ndef bye():\n    return \"Goodbye\"\n")

    res = apply_patch(MID_PATCH, directory=str(tmp_path))
    assert res["status"] == "SUCCESS"
    assert res["patch_stats"]["files_updated"] == 1
    assert res["patch_stats"]["additions"] == 1
    updated = (tmp_path / "src" / "greet.py").read_text()
    assert "def greet(name):\n    return f\"Hello, {name}\"\n\n\ndef bye():" in updated
    assert updated.endswith("\n")


def test_apply_patch_creates_new_file_and_deletes(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nomega\n")

    multi_file_patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,3 @@
 alpha
+beta
 omega

diff --git a/new.txt b/new.txt
new file mode 100644
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+first line
+second line
"""
    res = apply_patch(multi_file_patch, directory=str(tmp_path))
    assert res["status"] == "SUCCESS"
    assert res["patch_stats"]["files_updated"] == 1
    assert res["patch_stats"]["files_created"] == 1
    assert (tmp_path / "a.txt").read_text() == "alpha\nbeta\nomega\n"
    assert (tmp_path / "new.txt").read_text() == "first line\nsecond line\n"

    delete_patch = """--- a/a.txt
+++ /dev/null
@@ -1,3 +0,0 @@
-alpha
-beta
-omega
"""
    res = apply_patch(delete_patch, directory=str(tmp_path), allow_delete=True)
    assert res["status"] == "SUCCESS"
    assert res["patch_stats"]["files_deleted"] == 1
    assert not (tmp_path / "a.txt").exists()

    res = apply_patch(delete_patch, directory=str(tmp_path), allow_delete=False)
    assert res["status"] == "DENIED"


def test_apply_patch_atomic_failure_leaves_files_untouched(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nomega\n")

    # Second hunk context does not match b.txt (which says "wrong").
    bad_patch = """--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,3 @@
 alpha
+beta
 omega

--- a/b.txt
+++ b/b.txt
@@ -1,1 +1,1 @@
-wrong
+fixed
"""
    res = apply_patch(bad_patch, directory=str(tmp_path))
    assert res["status"] == "ERROR"
    assert "does not apply cleanly" in res["error"]
    # Nothing was written: a.txt must retain original content.
    assert (tmp_path / "a.txt").read_text() == "alpha\nomega\n"


def test_apply_patch_rejects_empty_and_malformed(tmp_path):
    res = apply_patch("")
    assert res["status"] == "ERROR"

    res = apply_patch("some freeform text without headers")
    assert res["status"] == "ERROR"

    bad_hunk = """--- a/x.txt
+++ b/x.txt
@@ -1,3 +1,3 @@
 a
-b
 c
"""
    write_file(str(tmp_path / "x.txt"), "a\nb\nc\n")
    res = apply_patch(bad_hunk, directory=str(tmp_path))
    assert res["status"] == "ERROR"
    assert "Malformed unified diff" in res["error"]