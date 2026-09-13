import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from agent_workspace_tools.async_runner import (
    is_high_risk,
    intercept_and_sanitize_command,
    execute_async_subprocess,
    start_background_task,
    get_background_task_status,
    clean_success_stderr,
    filter_errors_only,
    extract_error_files,
    enrich_build_error,
    summarize_success_output
)
from agent_workspace_tools.git_utils import get_git_status_changes

def test_is_high_risk_detection():
    assert is_high_risk("rm -rf /tmp/test") is True
    assert is_high_risk("sudo apt update") is True
    assert is_high_risk("ls -la") is False

@patch("agent_workspace_tools.async_runner.has_active_mcp_config", return_value=False)
def test_intercept_and_sanitize_command_daemons(mock_mcp_config):
    blocked, _, err = intercept_and_sanitize_command("ng mcp")
    assert blocked is True
    assert "SYSTEM INTERCEPT" in err

    blocked, _, err = intercept_and_sanitize_command("npx mcp-server-git")
    assert blocked is True
    assert "SYSTEM INTERCEPT" in err

    blocked, _, err = intercept_and_sanitize_command("ng serve")
    assert blocked is True
    assert "interactive daemon" in err

    blocked, _, _ = intercept_and_sanitize_command("vite")
    assert blocked is True

    blocked, _, _ = intercept_and_sanitize_command("flutter run")
    assert blocked is True

@patch("agent_workspace_tools.async_runner.has_active_mcp_config", return_value=True)
def test_intercept_and_sanitize_command_mcp_configured(mock_mcp_config):
    blocked, sanitized, err = intercept_and_sanitize_command("npx mcp-server-git")
    assert blocked is False
    assert sanitized == "npx mcp-server-git"
    assert err == ""

def test_intercept_and_sanitize_command_passthrough_clean_commands():
    blocked, sanitized, _ = intercept_and_sanitize_command("ng test --no-watch")
    assert blocked is False
    assert sanitized == "ng test --no-watch"

@pytest.mark.asyncio
async def test_execute_async_subprocess_blocked():
    result = await execute_async_subprocess("ng serve")
    assert result["status"] == "BLOCKED"
    assert result["returncode"] == 1
    assert "System Guardrail Error" in result["stderr"]

@pytest.mark.asyncio
@patch("agent_workspace_tools.async_runner.has_active_mcp_config", return_value=False)
async def test_execute_async_subprocess_mcp_blocked(mock_mcp_config):
    result = await execute_async_subprocess("ng mcp")
    assert result["status"] == "BLOCKED"
    assert result["returncode"] == 1
    assert "SYSTEM INTERCEPT" in result["stderr"]

@pytest.mark.asyncio
async def test_execute_async_subprocess_success():
    result = await execute_async_subprocess("echo 'Async Runner Test'")
    assert result["status"] == "SUCCESS"
    assert result["stdout"] == "Async Runner Test"

@pytest.mark.asyncio
async def test_execute_async_subprocess_timeout():
    result = await execute_async_subprocess("sleep 2", timeout=0.2)
    assert result["status"] == "TIMEOUT"
    assert result["returncode"] == -9

@pytest.mark.asyncio
@patch("agent_workspace_tools.async_runner.request_human_approval", return_value=False)
async def test_execute_async_subprocess_hitl_denied(mock_hitl):
    result = await execute_async_subprocess("rm -rf /dummy/path")
    assert result["status"] == "DENIED"
    assert result["returncode"] == -1
    mock_hitl.assert_called_once()

# =====================================================================
# Tests for Background Subprocess Execution
# =====================================================================

@pytest.mark.asyncio
async def test_start_and_get_background_task():
    start_res = await start_background_task("echo 'Background Worker Finished'")
    assert start_res["status"] == "STARTED"
    task_id = start_res["task_id"]

    await asyncio.sleep(0.1)

    status_res = await get_background_task_status(task_id)
    assert status_res["status"] == "SUCCESS"
    assert status_res["returncode"] == 0
    assert "Background Worker Finished" in status_res["stdout"]

# =====================================================================
# Tests for git_utils.py
# =====================================================================

@pytest.mark.asyncio
@patch("agent_workspace_tools.git_utils.execute_async_subprocess", new_callable=AsyncMock)
async def test_get_git_status_changes_non_git_repo(mock_exec):
    mock_exec.return_value = {
        "returncode": 128,
        "stdout": "",
        "stderr": "fatal: not a git repository",
        "status": "ERROR"
    }

    is_git, update_files, delete_files = await get_git_status_changes("/fake/dir")

    assert is_git is False
    assert update_files == set()
    assert delete_files == set()

@pytest.mark.asyncio
@patch("agent_workspace_tools.git_utils.os.path.isfile", return_value=True)
@patch("agent_workspace_tools.git_utils.execute_async_subprocess", new_callable=AsyncMock)
async def test_get_git_status_changes_parses_updates_and_deletes(mock_exec, mock_isfile):
    mock_exec.side_effect = [
        {"returncode": 0, "stdout": "true", "stderr": "", "status": "SUCCESS"},
        {
            "returncode": 0,
            "stdout": " M src/app.ts\n?? src/new_component.ts\n D src/old_component.ts\n M \"src/file with spaces.ts\"",
            "stderr": "",
            "status": "SUCCESS"
        }
    ]

    is_git, update_files, delete_files = await get_git_status_changes("/fake/dir")

    assert is_git is True
    assert "src/app.ts" in update_files
    assert "src/new_component.ts" in update_files
    assert "src/file with spaces.ts" in update_files
    assert "src/old_component.ts" in delete_files

# =====================================================================
# Tests for Error Filtering & Enrichment
# =====================================================================

def test_filter_errors_only():
    verbose_stderr = (
        "▲ [WARNING] Exceeds maximum budget\n"
        "▲ [WARNING] NG02956: Not implemented\n"
        "✘ [ERROR] NG8008: Required input 'formField' from component Input must be specified.\n"
        "Some extra stack trace info here."
    )
    summarized = filter_errors_only(verbose_stderr)
    assert "✘ [ERROR] NG8008" in summarized
    assert "WARNING" not in summarized

def test_extract_error_files_implicit_component():
    stderr_with_component = "✘ [ERROR] NG8008: Required input 'formField' from component Input must be specified. \n  Error occurs in the template of component LoginPage."
    files = extract_error_files(stderr_with_component)
    assert "login-page.html / login-page.ts" in files

def test_enrich_build_error_with_files():
    stderr_with_path = "✘ [ERROR] NG8002: Can't bind to 'formControlName'. \n  src/app/auth/login.html:42:5"
    enriched = enrich_build_error("ng build", stderr_with_path)
    assert "🛑 [BUILD/TEST ERROR INTERCEPT]" in enriched
    assert "`src/app/auth/login.html`" in enriched

def test_summarize_success_output():
    verbose_stdout = (
        "Initial chunk files | Names\n"
        "Application bundle generation complete. [2.788 seconds]\n"
        "Output location: /dist"
    )
    summarized = summarize_success_output(verbose_stdout)
    assert "Application bundle generation complete. [SUCCESS]" in summarized

def test_clean_success_stderr():
    verbose_success_stderr = (
        "▲ [WARNING] Exceeds maximum budget\n"
        "Some benign info message\n"
    )
    cleaned = clean_success_stderr(verbose_success_stderr)
    assert "Exceeds maximum budget" not in cleaned
    assert "Some benign info message" in cleaned