from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .lecture_pipeline import status_lecture_pipeline_plan
from .models import now_iso
from .powershell import run_powershell_command as _run_command
from .storage import append_jsonl, ensure_project_dirs, read_json, read_jsonl


ALLOWED_EXTRACTORS = {"vidclaude", "peepshow", "vidwise"}


def run_extractor_plan(
    plan_json: str | Path,
    extractor: str,
    *,
    execute: bool = False,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """Preview or run one planned visual extractor command from a lecture plan."""
    if extractor not in ALLOWED_EXTRACTORS:
        raise ValueError(f"unsupported extractor: {extractor}")
    plan_path = Path(plan_json).expanduser().resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("lecture pipeline plan must be a JSON object")
    commands = plan.get("commands") if isinstance(plan.get("commands"), dict) else {}
    command = str(commands.get(extractor) or "").strip()
    if not command:
        raise ValueError(f"plan has no command for extractor: {extractor}")
    project = Path(str(plan.get("project") or "")).expanduser()
    before = status_lecture_pipeline_plan(plan_path)
    result: dict[str, Any] = {
        "plan_path": str(plan_path),
        "project": str(project),
        "extractor": extractor,
        "command": command,
        "execute": execute,
        "timeout_seconds": int(timeout_seconds or 0),
        "started_at": now_iso(),
        "before_ready": bool((before.get("ready") or {}).get(extractor)),
        "status": "preview",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not execute:
        return _write_extractor_log(project, result, before)

    try:
        completed = _run_command(command, cwd=project if project.exists() else plan_path.parent, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        after = status_lecture_pipeline_plan(plan_path)
        result.update(
            {
                "status": "timeout",
                "returncode": None,
                "stdout": _timeout_stream(exc.output),
                "stderr": _timeout_stderr(exc, timeout_seconds=timeout_seconds),
                "finished_at": now_iso(),
                "after_ready": bool((after.get("ready") or {}).get(extractor)),
                "pipeline_status": after,
            }
        )
        return _write_extractor_log(project, result, after)
    after = status_lecture_pipeline_plan(plan_path)
    result.update(
        {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "finished_at": now_iso(),
            "after_ready": bool((after.get("ready") or {}).get(extractor)),
            "pipeline_status": after,
        }
    )
    return _write_extractor_log(project, result, after)


def extractor_run_log(root: str | Path) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "extractor-command-runs.jsonl"
    markdown_path = paths["notes"] / "extractor-command-runs.md"
    rows = read_jsonl(log_path)
    markdown_path.write_text(_render_extractor_log_markdown(root, rows), encoding="utf-8")
    return {
        "project": str(root),
        "log_path": str(log_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "commands": rows,
        "last": rows[-1] if rows else {},
    }



def _write_extractor_log(root: Path, result: dict[str, Any], pipeline_status: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "extractor-command-runs.jsonl"
    extractor = str(result.get("extractor") or "")
    record = {
        "created_at": now_iso(),
        "extractor": extractor,
        "execute": bool(result.get("execute")),
        "status": result.get("status", ""),
        "returncode": result.get("returncode"),
        "before_ready": bool(result.get("before_ready")),
        "after_ready": bool((pipeline_status.get("ready") or {}).get(extractor)),
        "command": result.get("command", ""),
        "stdout_tail": _tail(result.get("stdout", "")),
        "stderr_tail": _tail(result.get("stderr", "")),
    }
    append_jsonl(log_path, [record])
    log = extractor_run_log(root)
    result["extractor_log"] = {
        "record": record,
        "log_path": log["log_path"],
        "markdown_path": log["markdown_path"],
        "count": log["count"],
    }
    if "pipeline_status" not in result:
        result["pipeline_status"] = pipeline_status
    return result


def _render_extractor_log_markdown(root: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Visual Extractor Command Runs",
        "",
        f"- Project: `{root}`",
        f"- Count: {len(rows)}",
        "",
        "| Time | Extractor | Execute | Status | Return | Ready before | Ready after |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        execute = "yes" if row.get("execute") else "no"
        before = "yes" if row.get("before_ready") else "no"
        after = "yes" if row.get("after_ready") else "no"
        lines.append(
            f"| {row.get('created_at', '')} | `{row.get('extractor', '')}` | {execute} | {row.get('status', '')} | {row.get('returncode', '')} | {before} | {after} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _tail(value: Any, *, limit: int = 4000) -> str:
    return str(value or "")[-limit:]


def _timeout_stream(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _timeout_stderr(exc: subprocess.TimeoutExpired, *, timeout_seconds: int) -> str:
    stderr = _timeout_stream(exc.stderr)
    message = f"timeout after {int(timeout_seconds or 0)}s: {exc}"
    return (stderr + "\n" + message).strip() if stderr else message
