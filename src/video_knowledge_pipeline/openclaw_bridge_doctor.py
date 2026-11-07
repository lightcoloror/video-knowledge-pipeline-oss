from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import now_iso
from .openclaw_bridge_status import openclaw_bridge_status

TASK_NAME = "VideoKnowledgeOpenClawHttp"


def openclaw_bridge_doctor(*, timeout_seconds: float = 2.0, project_root: str | Path = "") -> dict[str, Any]:
    """Read-only diagnostic report for the OpenClaw host bridge."""

    root = Path(project_root).expanduser().resolve() if project_root else Path(__file__).resolve().parents[2]
    status = openclaw_bridge_status(timeout_seconds=timeout_seconds, check_health=True, check_task=True)
    startup = _startup_status(root)
    logs = _recent_logs(root)
    findings = _findings(status=status, startup=startup, logs=logs)
    ok = bool(status.get("ok"))
    return {
        "schema": "video_knowledge_pipeline.openclaw_bridge_doctor.v1",
        "created_at": now_iso(),
        "ok": ok,
        "status": "ok" if ok else "needs_operator_action",
        "project_root": str(root),
        "bridge_status": status,
        "startup_fallback": startup,
        "recent_logs": logs,
        "findings": findings,
        "visible_powershell_commands": _visible_powershell_commands(root),
        "operator_boundary": {
            "no_video_processing": True,
            "no_cloud_calls": True,
            "no_task_registration_performed": True,
        },
        "next_actions": _next_actions(ok=ok, findings=findings),
    }


def _startup_status(root: Path) -> dict[str, Any]:
    startup_file = _startup_file()
    target = root / "scripts" / "start-openclaw-http-background.ps1"
    return {
        "checked": True,
        "startup_file": str(startup_file) if startup_file else "",
        "exists": bool(startup_file and startup_file.exists()),
        "target_script": str(target),
        "target_exists": target.exists(),
    }


def _startup_file() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "VideoKnowledgeOpenClawHttp.cmd"


def _recent_logs(root: Path) -> dict[str, Any]:
    log_dir = root / ".local" / "logs"
    stdout = log_dir / "openclaw-http.stdout.log"
    stderr = log_dir / "openclaw-http.stderr.log"
    return {
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def _tail(path: Path, *, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _findings(*, status: dict[str, Any], startup: dict[str, Any], logs: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not status.get("running"):
        findings.append({"key": "bridge_not_running", "severity": "block", "message": "8931 bridge is not listening."})
    task = status.get("scheduled_task") if isinstance(status.get("scheduled_task"), dict) else {}
    task_text = " ".join(str(task.get(key) or "") for key in ("status", "message", "stderr_excerpt", "error"))
    if "Access is denied" in task_text or "拒绝" in task_text:
        findings.append({"key": "access_denied_register_task", "severity": "warn", "message": "Task Scheduler registration/query needs visible PowerShell or more permissions."})
    elif task.get("checked") and not task.get("exists"):
        findings.append({"key": "scheduled_task_not_registered", "severity": "warn", "message": f"{TASK_NAME} is not registered."})
    if not startup.get("exists"):
        findings.append({"key": "startup_fallback_not_installed", "severity": "info", "message": "Startup folder fallback is not installed."})
    if "listening on http://127.0.0.1:8931" in str(logs.get("stdout_tail") or "") and not status.get("running"):
        findings.append({"key": "started_then_exited", "severity": "warn", "message": "Bridge log shows a recent start, but the port is no longer listening."})
    stderr = str(logs.get("stderr_tail") or "")
    if "No module named" in stderr or "ModuleNotFoundError" in stderr:
        findings.append({"key": "missing_python_or_module", "severity": "block", "message": "Bridge Python module import failed."})
    if "Address already in use" in stderr or "WinError 10048" in stderr:
        findings.append({"key": "port_in_use", "severity": "block", "message": "Configured bridge port is already in use."})
    return findings


def _visible_powershell_commands(root: Path) -> list[str]:
    return [
        f"cd {root}",
        r".\scripts\openclaw-http-task.ps1 register",
        r".\scripts\openclaw-http-task.ps1 start",
        r".\scripts\openclaw-http-startup-folder.ps1 install",
        r".\scripts\video-knowledge.ps1 openclaw-bridge-status",
        r".\scripts\video-knowledge.ps1 openclaw-bridge-doctor",
        r"Invoke-RestMethod http://127.0.0.1:8931/health",
        r"Invoke-RestMethod http://127.0.0.1:8931/contract",
    ]


def _next_actions(*, ok: bool, findings: list[dict[str, str]]) -> list[str]:
    if ok:
        return ["openclaw_can_call_vkp_http_bridge"]
    keys = {finding.get("key") for finding in findings}
    if "missing_python_or_module" in keys:
        return ["inspect_openclaw_http_logs", "fix_python_environment"]
    if "port_in_use" in keys:
        return ["inspect_port_owner", "restart_or_reconfigure_bridge"]
    if "access_denied_register_task" in keys:
        return ["run_visible_powershell_task_registration_or_install_startup_fallback"]
    actions: list[str] = []
    if "scheduled_task_not_registered" in keys:
        actions.extend(["run_visible_powershell_openclaw_http_task_register", "run_visible_powershell_openclaw_http_task_start"])
    if "startup_fallback_not_installed" in keys:
        actions.append("install_startup_folder_fallback_if_task_registration_fails")
    if "started_then_exited" in keys:
        actions.extend(["inspect_openclaw_http_logs", "start_bridge_from_visible_powershell"])
    actions.append("rerun_openclaw_bridge_doctor")
    return _dedupe(actions)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
