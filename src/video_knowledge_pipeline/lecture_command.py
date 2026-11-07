from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .lecture_pipeline import _local_cli_command
from .models import now_iso
from .powershell import run_powershell_command as _run_command
from .storage import append_jsonl, ensure_project_dirs, read_json, read_jsonl


def run_planned_lecture_command(
    plan_json: str | Path,
    command_name: str,
    *,
    execute: bool = False,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """Run or preview one command from a prepared lecture pipeline plan."""
    plan_path = Path(plan_json)
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("lecture pipeline plan must be a JSON object")
    command_map = _command_map(plan, plan_path=plan_path)
    if command_name not in command_map:
        raise ValueError(f"unknown planned command: {command_name}")

    project = Path(str(plan.get("project", "")))
    command = command_map[command_name]
    result: dict[str, Any] = {
        "plan_path": str(plan_path),
        "project": str(project),
        "command_name": command_name,
        "command": command,
        "execute": execute,
        "timeout_seconds": int(timeout_seconds or 0),
        "started_at": now_iso(),
    }
    if not execute:
        result.update({"status": "preview", "returncode": None, "stdout": "", "stderr": ""})
        return _write_command_log(project, result)

    try:
        completed = _run_command(command, cwd=project if project.exists() else plan_path.parent, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "status": "timeout",
                "returncode": None,
                "stdout": _timeout_stream(exc.output),
                "stderr": _timeout_stderr(exc, timeout_seconds=timeout_seconds),
                "finished_at": now_iso(),
            }
        )
        return _write_command_log(project, result)
    result.update(
        {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "finished_at": now_iso(),
        }
    )
    return _write_command_log(project, result)


def lecture_command_log(root: str | Path) -> dict[str, Any]:
    """Return and render the persisted planned-command execution log."""
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "lecture-command-runs.jsonl"
    markdown_path = paths["notes"] / "lecture-command-runs.md"
    rows = read_jsonl(log_path)
    markdown_path.write_text(_render_command_log_markdown(root, rows), encoding="utf-8")
    return {
        "project": str(root),
        "log_path": str(log_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "commands": rows,
    }


def _command_map(plan: dict[str, Any], *, plan_path: Path) -> dict[str, str]:
    commands = plan.get("commands") if isinstance(plan.get("commands"), dict) else {}
    pipeline_commands = plan.get("run_pipeline_commands") if isinstance(plan.get("run_pipeline_commands"), dict) else {}
    project = str(plan.get("project", ""))
    plan_path_text = str(plan.get("plan_path") or plan_path)
    result = {str(key): str(value) for key, value in commands.items() if value}
    result.update({f"pipeline.{key}": str(value) for key, value in pipeline_commands.items() if value})
    if plan_path_text:
        result["status"] = _local_cli_command(["status-lecture-pipeline", plan_path_text])
        result["run_ready"] = _local_cli_command(["run-ready-lecture-pipeline", plan_path_text])
        if project:
            result["health"] = _local_cli_command(["lecture-health", project, "--plan-json", plan_path_text])
    return result



def _write_command_log(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "lecture-command-runs.jsonl"
    record = {
        "created_at": now_iso(),
        "command_name": result.get("command_name", ""),
        "execute": bool(result.get("execute")),
        "status": result.get("status", ""),
        "returncode": result.get("returncode"),
        "command": result.get("command", ""),
        "stdout_tail": _tail(result.get("stdout", "")),
        "stderr_tail": _tail(result.get("stderr", "")),
    }
    append_jsonl(log_path, [record])
    log = lecture_command_log(root)
    result["command_log"] = {
        "record": record,
        "log_path": log["log_path"],
        "markdown_path": log["markdown_path"],
        "count": log["count"],
    }
    return result


def _render_command_log_markdown(root: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Lecture Planned Command Runs",
        "",
        f"- Project: `{root}`",
        f"- Count: {len(rows)}",
        "",
        "| Time | Command | Execute | Status | Return |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        execute = "yes" if row.get("execute") else "no"
        lines.append(
            f"| {row.get('created_at', '')} | `{row.get('command_name', '')}` | {execute} | {row.get('status', '')} | {row.get('returncode', '')} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _tail(value: Any, *, limit: int = 4000) -> str:
    text = str(value or "")
    return text[-limit:]


def _timeout_stream(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _timeout_stderr(exc: subprocess.TimeoutExpired, *, timeout_seconds: int) -> str:
    stderr = _timeout_stream(exc.stderr)
    message = f"timeout after {int(timeout_seconds or 0)}s: {exc}"
    return (stderr + "\n" + message).strip() if stderr else message
