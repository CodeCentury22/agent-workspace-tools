import pytest
from pathlib import Path

from agent_workspace_tools import dispatch_tool, WORKSPACE_TOOL_DISPATCHER, WORKSPACE_TOOLS_SCHEMA
from agent_workspace_tools.file_ops import (
    FILE_TOOL_DISPATCHER,
    TOOL_ALIASES,
    modify_file,
    replace_in_file,
    resolve_tool_alias,
    sanitize_tool_kwargs,
    write_file,
)


def test_modify_file_registered_in_dispatcher():
    assert "modify_file" in FILE_TOOL_DISPATCHER
    assert "modify_file" in WORKSPACE_TOOL_DISPATCHER
    assert callable(FILE_TOOL_DISPATCHER["modify_file"])
    names = {t["function"]["name"] for t in WORKSPACE_TOOLS_SCHEMA}
    assert "modify_file" in names


def test_resolve_tool_alias():
    assert resolve_tool_alias("modify_file") == "replace_in_file"
    assert resolve_tool_alias("edit_file") == "replace_in_file"
    assert resolve_tool_alias("replace_in_file") == "replace_in_file"
    assert TOOL_ALIASES["modify_file"] == "replace_in_file"


def test_modify_file_routes_to_replace(tmp_path):
    target = tmp_path / "edit.txt"
    target.write_text("hello old world")
    res = modify_file(str(target), search_text="old", replace_text="new")
    assert res["status"] == "SUCCESS"
    assert "modify_file->replace_in_file" in res["routed_via"]
    assert target.read_text() == "hello new world"


def test_modify_file_heals_alias_args(tmp_path):
    target = tmp_path / "alias.txt"
    target.write_text("foo OLD bar")
    # Cline-style hallucinated keys: old_string + content/body
    res = modify_file(str(target), old_string="OLD", content="NEW")
    assert res["status"] == "SUCCESS"
    assert target.read_text() == "foo NEW bar"


def test_modify_file_content_only_overwrites(tmp_path):
    target = tmp_path / "overwrite.txt"
    target.write_text("stale")
    res = modify_file(str(target), content="fresh full content")
    assert res["status"] == "SUCCESS"
    assert "modify_file->write_file" in res["routed_via"]
    assert target.read_text() == "fresh full content"


def test_replace_in_file_heals_alias_kwargs(tmp_path):
    target = tmp_path / "heal.txt"
    target.write_text("alpha beta")
    res = replace_in_file(str(target), old_text="beta", new_text="gamma")
    assert res["status"] == "SUCCESS"
    assert target.read_text() == "alpha gamma"


def test_write_file_heals_content_alias(tmp_path):
    target = str(tmp_path / "w.txt")
    res = write_file(target, content="via alias")
    assert res["status"] == "SUCCESS"


def test_sanitize_never_overwrites_canonical():
    healed = sanitize_tool_kwargs(
        "replace_in_file",
        {"file_path": "a.txt", "search_text": "x", "replace_text": "keep", "content": "drop"},
    )
    assert healed["replace_text"] == "keep"


@pytest.mark.asyncio
async def test_dispatch_tool_modify_file_alias(tmp_path):
    target = tmp_path / "d.txt"
    target.write_text("one two")
    res = await dispatch_tool("modify_file", file_path=str(target), search_text="two", replace_text="three")
    assert res["status"] == "SUCCESS"
    assert Path(str(target)).read_text() == "one three"


@pytest.mark.asyncio
async def test_dispatch_tool_heals_alias_keys(tmp_path):
    target = tmp_path / "e.txt"
    target.write_text("aaa bbb")
    res = await dispatch_tool(
        "replace_in_file", file_path=str(target), old_string="bbb", body="ccc"
    )
    assert res["status"] == "SUCCESS"
    assert target.read_text() == "aaa ccc"
