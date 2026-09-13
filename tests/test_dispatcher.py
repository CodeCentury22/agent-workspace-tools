import pytest
from agent_workspace_tools import dispatch_tool, WORKSPACE_TOOLS_SCHEMA

def test_schema_merge():
    """Verify both file and async tools are present in the unified schema."""
    names = {t["function"]["name"] for t in WORKSPACE_TOOLS_SCHEMA}
    assert "read_file" in names
    assert "run_shell_command" in names
    assert "replace_in_file" in names
    assert "insert_lines" in names
    assert "regex_replace_in_file" in names
    assert "apply_patch" in names
    assert "get_file_digest" in names
    assert "start_background_task" in names
    assert "list_background_tasks" in names
    assert "cancel_background_task" in names


def test_schema_names_match_dispatcher_keys():
    """Every advertised tool must have a callable dispatcher entry."""
    from agent_workspace_tools import WORKSPACE_TOOL_DISPATCHER
    for tool in WORKSPACE_TOOLS_SCHEMA:
        name = tool["function"]["name"]
        assert name in WORKSPACE_TOOL_DISPATCHER
        assert callable(WORKSPACE_TOOL_DISPATCHER[name])

@pytest.mark.asyncio
async def test_unified_dispatcher(tmp_path):
    """Verify dispatcher routes both sync and async functions seamlessly."""
    
    # 1. Test Sync File Tool
    target = tmp_path / "hello.txt"
    write_res = await dispatch_tool("write_file", file_path=str(target), code_body="hello world")
    assert write_res["status"] == "SUCCESS"
    
    # 2. Test Async Shell Tool
    shell_res = await dispatch_tool("run_shell_command", command="echo 'runner test'")
    assert shell_res["status"] == "SUCCESS"
    assert "runner test" in shell_res["stdout"]

@pytest.mark.asyncio
async def test_dispatch_tool_unknown_tool():
    """Verify gracefully handles invalid tool execution."""
    result = await dispatch_tool("definitely_not_a_tool")
    assert result["status"] == "ERROR"
    assert "Unknown tool" in result["error"]