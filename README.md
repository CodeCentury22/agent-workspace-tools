# agent-workspace-tools

Unified file system, shell execution, async background task, and git context tools for autonomous AI coding agents.

This micro-library is the single sanctioned implementation layer for **all** workspace mutation,
subprocess execution, and background job polling performed by `agent-cli`. Orchestration code must
never shell out or edit files directly — it routes through `WORKSPACE_TOOLS_SCHEMA` and
`WORKSPACE_TOOL_DISPATCHER` exposed here.

## Features

- **Sandboxed file operations** — POSIX-normalized, path-escape-safe read/write/edit/delete/list/search
  with atomic writes (`os.replace` temp-sibling pattern).
- **Surgical code editing** — `insert_lines` (1-based line insertion), `append_to_file`,
  `regex_replace_in_file` (for refactors where matches vary), and `apply_patch`
  (git-style unified diff applied atomically across multiple files with dry-run validation).
- **Change verification** — `get_file_digest` and `sha256` in `get_file_info`.
- **Guarded shell runner** — async subprocess execution with HITL approval for high-risk commands
  (`rm`, `sudo`, `dd`, `mkfs`, …), interactive daemon/MCP-server interception, timeout bounds,
  and build-error enrichment that tells the agent which files to inspect next.
- **Background task lifecycle** — spawn long-running builds/tests, poll status, list all tasks,
  and cancel stuck runs. Bounded registry (`AGENT_MAX_BACKGROUND_TASKS`, default 50).
- **Git context** — branch, recent commits, diff stats, and uncommitted change sets for prompt
  enrichment and incremental vector-memory synchronization.

## Installation

```bash
uv add "agent-workspace-tools @ git+https://github.com/CodeCentury22/agent-workspace-tools.git@v0.2.1"
```

Or via `pyproject.toml`:

```toml
[tool.uv.sources]
agent-workspace-tools = { git = "https://github.com/CodeCentury22/agent-workspace-tools.git", tag = "v0.2.1" }
```

## Tool Inventory

The package exposes `WORKSPACE_TOOLS_SCHEMA` (OpenAI-style function schemas injected into the LLM
system prompt) and `WORKSPACE_TOOL_DISPATCHER`, plus `dispatch_tool(tool_name, **kwargs)` as the
single master entry point that transparently routes sync file tools and async shell tools.

### File tools (sync)

| Tool                    | Purpose                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `read_file`             | Read a file (optional inclusive line range, byte cap).                  |
| `write_file`            | Atomically overwrite/create a file (returns `NO_CHANGE` when identical).|
| `replace_in_file`       | Exact-text block replacement.                                           |
| `insert_lines`          | Insert text at a 1-based line boundary (optional `create_if_missing`).  |
| `append_to_file`        | Append text to an existing or new file.                                 |
| `regex_replace_in_file` | Regex find & replace with `count` and `case_sensitive` controls.        |
| `apply_patch`           | Apply a git-style unified diff atomically (multi-file).                 |
| `get_file_digest`       | `sha256` (or any `hashlib` algorithm) of a file.                        |
| `delete_file`           | Delete a file.                                                          |
| `list_files`            | List files/dirs (recursive, glob `pattern`, `max_results`).             |
| `search_in_files`       | Regex search with line numbers (respects `SKIP_DIRS`).                  |
| `get_file_info`         | Metadata incl. size, extension, mtime, and `sha256`.                    |
| `create_directory`      | `mkdir -p`.                                                             |
| `copy_file` / `move_file` | Copy/move files or trees.                                             |
### Shell tools (async)

| Tool                       | Purpose                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| `run_shell_command`        | Execute a command with HITL checks, `cwd` support, timeout bounds. |
| `start_background_task`    | Spawn a long-running command (build/test) in the background.       |
| `get_background_task_status` | Poll stdout/stderr/returncode of a background task.              |
| `list_background_tasks`    | Enumerate all tracked tasks.                                       |
| `cancel_background_task`   | Terminate a running task (SIGTERM → SIGKILL).                      |

### Python APIs (not LLM-facing)

- `normalize_path`, `get_workspace_root`, `PathEscapeError` — sandbox helpers.
- `get_git_status_changes(workspace_dir)` — `(is_git, updated_set, deleted_set)` for vector sync.
- `get_git_summary(workspace_dir, commits=10)` — branch/commits/diff-stat/change sets snapshot.
- `get_git_branch`, `get_recent_git_commits`, `get_git_diff` — building blocks for the above.

## Safety Model

1. **Sandbox**: when `AGENT_WORKSPACE_ROOT` is set, every normalized path is resolved and verified
   to stay inside the workspace; escapes raise `PathEscapeError`. Erratic relative paths are
   re-anchored onto known project folders (`src`, `app`, `libs`, …).
2. **HITL**: high-risk commands go through `request_human_approval()`; rejection returns `DENIED`
   and skips execution.
3. **Daemon/MCP interception**: interactive dev servers (`ng serve`, `vite`, `flutter run`,
   `gradlew run`, adb logcat, expo/react-native, …) and unconfigured MCP server starts are
   blocked with explicit agent instructions.
4. **Output hygiene**: successful build output is summarized; failed builds are enriched with
   affected file lists and remediation instructions so the agent's next action is a targeted
   `read_file` + `replace_in_file` instead of blind retries.

## Development

```bash
uv sync
uv run pytest            # 74 tests
```

Release flow (bump version, tag, publish):

```bash
python release.py 0.2.0 -m "release: bump agent-workspace-tools to v0.2.0"
```

Then cascade dependents (`agent-cli`, `agent-llm-client`) by updating their
`[tool.uv.sources]` tag references and running `uv lock && uv sync`.

## License

MIT License © CodeCentury22
