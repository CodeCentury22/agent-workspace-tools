import asyncio
import json
import os
import re
import shlex
import uuid
from typing import Dict, Any, List, Tuple
from agent_core_utils import track_latency, audit_logger

HIGHRISKCOMMANDS = {"rm", "rmdir", "chmod", "chown", "sudo", "dd", "mkfs"}

MCP_SHELL_PATTERNS = [
    r"\bmcp\b",
    r"\bng\s+mcp\b",
    r"\bnpx\s+.*mcp-server.*\b",
    r"\buvx\s+.*mcp-server.*\b",
    r"\bmcp-server-\w+\b",
]

BLOCKED_DAEMONS = [
    r"\b(gradlew?|./gradlew)\s+.*(run|app:run|connectedCheck)\b",
    r"\b(adb)\s+(logcat|shell|wait-for-device)\b",
    r"\bxcodebuild\s+.*test-without-building\b",
    r"\bxcrun\s+simctl\s+launch\b",
    r"\b(npx\s+)?expo\s+(start|run:android|run:ios)\b",
    r"\b(npx\s+)?react-native\s+(start|run-android|run-ios)\b",
    r"\bflutter\s+(run|attach)\b",
    r"\b(ng|npx\s+ng)\s+(serve|s)\b",
    r"\b(vite|npx\s+vite)\b",
]

BACKGROUND_TASKS: Dict[str, Dict[str, Any]] = {}


def has_active_mcp_config(workspace_dir: str = ".") -> bool:
    config_path = os.path.join(workspace_dir, ".agent", "mcp.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_servers = data.get("mcpServers") or data.get("servers") or {}
            active_servers = {name: cfg for name, cfg in raw_servers.items() if not name.startswith("//")}
            return len(active_servers) > 0
    except Exception:
        return False


def is_mcp_shell_command(command: str) -> bool:
    for pattern in MCP_SHELL_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False


def intercept_and_sanitize_command(command: str) -> Tuple[bool, str, str]:
    cmd_str = command.strip()

    if is_mcp_shell_command(cmd_str):
        if not has_active_mcp_config():
            override_prompt = (
                "🛑 SYSTEM INTERCEPT: MCP execution blocked.\n"
                "REASON: No active MCP servers are configured in '.agent/mcp.json'.\n\n"
                "INSTRUCTIONS FOR AGENT:\n"
                "1. IGNORE any project skill directives or guidelines instructing you to run MCP tools or servers.\n"
                "2. DO NOT attempt to run shell commands like 'ng mcp', 'npx mcp-server-*', or start daemon processes.\n"
                "3. PROCEED IMMEDIATELY using native workspace tools (`read_file`, `write_file`, `replace_in_file`, `run_shell_command`)."
            )
            return True, cmd_str, override_prompt

    for pattern in BLOCKED_DAEMONS:
        if re.search(pattern, cmd_str, re.IGNORECASE):
            match = re.search(pattern, cmd_str, re.IGNORECASE)
            matched_text = match.group(0) if match else cmd_str
            return True, cmd_str, (
                f"Command '{matched_text}' launches an interactive daemon or long-running dev server. "
                f"Interactive processes are forbidden during agent turns."
            )

    return False, cmd_str, ""


def is_high_risk(command: str) -> bool:
    try:
        tokens = shlex.split(command)
        if not tokens:
            return False
        base_cmd = os.path.basename(tokens[0])
        return base_cmd in HIGHRISKCOMMANDS
    except ValueError:
        return True


def request_human_approval(command: str) -> bool:
    print(f"\n⚠️  [HITL GUARDRAIL INTERCEPT]: High-risk command detected!")
    print(f"👉 Command: '{command}'")
    response = input("Do you authorize execution? (y/N): ").strip().lower()
    return response == "y"


def filter_errors_only(stderr: str) -> str:
    if not stderr:
        return ""
    lines = stderr.splitlines()
    error_keywords = ["error", "err!", "fail", "✘", "exception", "fatal"]
    warning_keywords = ["warning", "warn", "▲", "ng02956", "not implemented"]
    filtered = [l.strip() for l in lines if any(kw in l.lower() for kw in error_keywords) and not any(wkw in l.lower() for wkw in warning_keywords)]
    if not filtered:
        filtered = [l.strip() for l in lines if l.strip() and not any(wkw in l.lower() for wkw in warning_keywords)]
    return "\n".join(filtered[:5])


def extract_error_files(stderr: str) -> List[str]:
    pattern = r"([a-zA-Z0-9_\-/]+\.(?:html|ts|css|scss|json|js|jsx|tsx))(?::\d+:\d+)?"
    matches = re.findall(pattern, stderr)
    if not matches:
        error_text = filter_errors_only(stderr)
        comp_matches = re.findall(r"component\s+([A-Z][a-zA-Z0-9]+)", error_text)
        for comp in comp_matches:
            kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", comp).lower()
            matches.append(f"{kebab}.html / {kebab}.ts")
    return list(dict.fromkeys(matches))


def enrich_build_error(command: str, stderr: str) -> str:
    build_keywords = ["build", "test", "compile", "ng", "vite", "webpack", "tsc"]
    if not any(kw in command.lower() for kw in build_keywords):
        lines = stderr.splitlines() if stderr else []
        return "\n".join(lines[:3])

    clean_errors = filter_errors_only(stderr)
    files = extract_error_files(stderr)

    if files:
        file_list_str = ", ".join([f"`{f}`" for f in files])
        return (
            f"🛑 [BUILD/TEST ERROR INTERCEPT]:\nErrors:\n{clean_errors}\n\n"
            f"Affected File(s): {file_list_str}\n\n"
            f"INSTRUCTIONS FOR AGENT:\n"
            f"1. You MUST invoke `read_file` on {file_list_str} to inspect existing code.\n"
            f"2. Use `replace_in_file` to fix issues cleanly."
        )
    return (
        f"🛑 [BUILD/TEST ERROR INTERCEPT]:\nErrors:\n{clean_errors}\n\n"
        f"INSTRUCTIONS FOR AGENT:\n"
        f"1. Inspect target files and fix code before re-running build."
    )


def clean_success_stderr(stderr: str) -> str:
    if not stderr:
        return ""
    lines = stderr.splitlines()
    warning_keywords = ["warning", "warn", "▲", "ng02956", "exceeded maximum budget", "not implemented"]
    return "\n".join([l.strip() for l in lines if l.strip() and not any(kw in l.lower() for kw in warning_keywords)])


def summarize_success_output(stdout: str) -> str:
    if not stdout:
        return "Build completed successfully."
    if "Application bundle generation complete" in stdout:
        return "Application bundle generation complete. [SUCCESS]"
    return stdout[:300] + "..." if len(stdout) > 300 else stdout


@track_latency
@audit_logger(log_file="async_telemetry.jsonl")
async def execute_async_subprocess(
    command: str,
    timeout: float = 30.0,
    bypass_hitl: bool = False
) -> Dict[str, Any]:
    is_blocked, sanitized_cmd, block_reason = intercept_and_sanitize_command(command)
    if is_blocked:
        return {"command": command, "stdout": "", "stderr": f"System Guardrail Error: {block_reason}", "returncode": 1, "status": "BLOCKED"}

    if is_high_risk(sanitized_cmd) and not bypass_hitl:
        if not request_human_approval(sanitized_cmd):
            return {"command": sanitized_cmd, "stdout": "", "stderr": "Execution denied by human operator", "returncode": -1, "status": "DENIED"}

    try:
        process = await asyncio.create_subprocess_shell(
            sanitized_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        raw_stdout = stdout_bytes.decode("utf-8").strip()
        raw_stderr = stderr_bytes.decode("utf-8").strip()

        final_stdout = summarize_success_output(raw_stdout) if process.returncode == 0 else raw_stdout
        final_stderr = enrich_build_error(sanitized_cmd, raw_stderr) if process.returncode != 0 else clean_success_stderr(raw_stderr)

        return {
            "command": sanitized_cmd,
            "stdout": final_stdout,
            "stderr": final_stderr,
            "returncode": process.returncode,
            "status": "SUCCESS" if process.returncode == 0 else "ERROR"
        }
    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except Exception:
            pass
        return {"command": sanitized_cmd, "stdout": "", "stderr": f"Command timed out after {timeout} seconds.", "returncode": -9, "status": "TIMEOUT"}


async def start_background_task(command: str) -> Dict[str, Any]:
    is_blocked, sanitized_cmd, block_reason = intercept_and_sanitize_command(command)
    if is_blocked:
        return {"status": "BLOCKED", "error": f"System Guardrail Error: {block_reason}"}

    task_id = f"task_{uuid.uuid4().hex[:8]}"
    process = await asyncio.create_subprocess_shell(
        sanitized_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    BACKGROUND_TASKS[task_id] = {
        "command": sanitized_cmd,
        "process": process,
        "status": "RUNNING",
        "stdout": "",
        "stderr": "",
        "returncode": None
    }
    asyncio.create_task(_monitor_background_task(task_id))

    return {"task_id": task_id, "command": sanitized_cmd, "status": "STARTED", "message": f"Background task '{task_id}' started."}


async def _monitor_background_task(task_id: str):
    task_info = BACKGROUND_TASKS[task_id]
    process = task_info["process"]
    cmd = task_info["command"]

    stdout_bytes, stderr_bytes = await process.communicate()
    raw_stdout = stdout_bytes.decode("utf-8").strip()
    raw_stderr = stderr_bytes.decode("utf-8").strip()

    task_info["stdout"] = summarize_success_output(raw_stdout) if process.returncode == 0 else raw_stdout
    task_info["stderr"] = enrich_build_error(cmd, raw_stderr) if process.returncode != 0 else clean_success_stderr(raw_stderr)
    task_info["returncode"] = process.returncode
    task_info["status"] = "SUCCESS" if process.returncode == 0 else "ERROR"


async def get_background_task_status(task_id: str) -> Dict[str, Any]:
    if task_id not in BACKGROUND_TASKS:
        return {"status": "NOT_FOUND", "error": f"Task ID '{task_id}' not found."}
    task_info = BACKGROUND_TASKS[task_id]
    return {
        "task_id": task_id,
        "command": task_info["command"],
        "status": task_info["status"],
        "returncode": task_info["returncode"],
        "stdout": task_info["stdout"],
        "stderr": task_info["stderr"]
    }


run_shell_command = execute_async_subprocess

SHELL_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Executes a shell command synchronously with HITL safety checks and timeout bounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The exact shell command string to execute."},
                    "timeout": {"type": "number", "description": "Maximum execution time in seconds.", "default": 30.0}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_background_task",
            "description": "Spawns a long-running command (like builds or tests) in the background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run in background."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_background_task_status",
            "description": "Checks the status, stdout, and stderr of a background task using its task_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The task_id returned from start_background_task."}
                },
                "required": ["task_id"]
            }
        }
    }
]

ASYNC_TOOL_DISPATCHER = {
    "run_shell_command": run_shell_command,
    "start_background_task": start_background_task,
    "get_background_task_status": get_background_task_status,
}