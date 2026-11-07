from __future__ import annotations

import json
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .bridge_state_contract import evaluate_bridge_state
from .config import config_status, service_config, service_url
from .models import now_iso
from .path_defaults import project_root

TASK_NAME = "VideoKnowledgeOpenClawHttp"


def openclaw_bridge_status(*, timeout_seconds: float = 2.0, check_health: bool = True, check_task: bool = True) -> dict[str, Any]:
    """Return a side-effect-free status report for the OpenClaw HTTP bridge."""

    config = config_status()
    try:
        service = service_config("openclaw_http")
        host = str(service.get("host") or "127.0.0.1")
        port = int(service.get("port") or 8931)
        path = str(service.get("path") or "/call")
        local_call_url = service_url("openclaw_http")
    except Exception as exc:  # noqa: BLE001 - this is an operator status command.
        return {
            "schema": "video_knowledge_pipeline.openclaw_bridge_status.v1",
            "created_at": now_iso(),
            "ok": False,
            "configured": False,
            "running": False,
            "listening": False,
            "healthy": False,
            "bridge_state": "config_invalid",
            "bridge_call_execution_allowed": False,
            "pipeline_offline_ready": False,
            "configured_is_online": False,
            "status": "config_error",
            "runtime_disposition": "stopped_by_design",
            "auto_start_attempted": False,
            "config_status": config,
            "error": str(exc),
            "next_actions": ["fix_video_knowledge_pipeline_config"],
        }

    docker_call_url = ""
    service_urls = config.get("service_urls") if isinstance(config.get("service_urls"), dict) else {}
    if service_urls:
        docker_call_url = str(service_urls.get("openclaw_http_docker") or "")
    socket_probe = _socket_probe(host, port, timeout_seconds)
    running = bool(socket_probe.get("listening"))
    health: dict[str, Any] = {}
    health_error: dict[str, Any] = {}
    if running and check_health:
        health_url = _health_url(local_call_url, path)
        health, health_error = _read_health(health_url, timeout_seconds)
    task = _scheduled_task_status(TASK_NAME) if check_task else {"checked": False}
    health_probe = {
        "checked": bool(running and check_health),
        "ok": bool(health.get("ok")) if health else False,
        "state": (
            "healthy"
            if health and health.get("ok")
            else "unhealthy"
            if running and check_health
            else "not_checked"
        ),
    }
    pipeline_capability = _pipeline_offline_capability_status()
    state_contract = evaluate_bridge_state(
        configured=bool(config.get("ok")),
        socket_probe=socket_probe,
        health_probe=health_probe,
        scheduled_task=task,
        pipeline_capability=pipeline_capability,
    )

    health_ok = bool(health.get("ok")) if health else False
    ok = bool(config.get("ok")) and running and (health_ok if check_health else True)
    status = "ok" if ok else "not_running"
    if running and check_health and not health_ok:
        status = "health_failed"
    elif not bool(config.get("ok")):
        status = "config_invalid"

    result = {
        "schema": "video_knowledge_pipeline.openclaw_bridge_status.v1",
        "created_at": now_iso(),
        "ok": ok,
        "configured": bool(config.get("ok")),
        "running": running,
        "listening": state_contract["listening"],
        "healthy": state_contract["healthy"],
        "status": status,
        "bridge_state": state_contract["bridge_state"],
        "runtime_disposition": state_contract["runtime_disposition"],
        "stale_runtime_record": state_contract["stale_runtime_record"],
        "bridge_call_execution_allowed": state_contract["bridge_call_execution_allowed"],
        "pipeline_offline_ready": state_contract["pipeline_offline_ready"],
        "configured_is_online": state_contract["configured_is_online"],
        "availability": state_contract["availability"],
        "review_evidence_policy": state_contract["review_evidence_policy"],
        "socket_probe": socket_probe,
        "health_probe": health_probe,
        "pipeline_capability": pipeline_capability,
        "auto_start_attempted": False,
        "host": host,
        "port": port,
        "path": path,
        "host_call_url": local_call_url,
        "docker_call_url": docker_call_url,
        "health_url": _health_url(local_call_url, path),
        "start_command": str(project_root() / "scripts" / "start-openclaw-http.cmd"),
        "scheduled_task": task,
        "operator_boundary": {
            "kind": "host_bridge_required",
            "summary": "Start the host-side VKP bridge before OpenClaw Docker calls video knowledge tools.",
            "no_video_processing": True,
            "no_cloud_calls": True,
        },
        "next_actions": _next_actions(
            config_ok=bool(config.get("ok")),
            running=running,
            health_ok=health_ok,
            check_health=check_health,
            task_exists=bool(task.get("exists")),
        ),
        "config_status": {
            "ok": bool(config.get("ok")),
            "config_path": str(config.get("config_path") or ""),
            "service_urls": service_urls,
            "validation": config.get("validation") if isinstance(config.get("validation"), dict) else {},
        },
    }
    if health:
        result["health"] = health
    if health_error:
        result["health_error"] = health_error
    return result


def _socket_connects(host: str, port: int, timeout_seconds: float) -> bool:
    return bool(_socket_probe(host, port, timeout_seconds).get("listening"))


def _socket_probe(host: str, port: int, timeout_seconds: float) -> dict[str, Any]:
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.1, float(timeout_seconds))):
            return {"checked": True, "listening": True, "state": "listening"}
    except (socket.timeout, TimeoutError) as exc:
        return {"checked": True, "listening": False, "state": "timeout", "error": str(exc)}
    except ConnectionRefusedError as exc:
        return {"checked": True, "listening": False, "state": "refused", "error": str(exc)}
    except OSError as exc:
        return {"checked": True, "listening": False, "state": "error", "error": str(exc)}


def _pipeline_offline_capability_status() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cli_script = root / "scripts" / "video-knowledge.ps1"
    route_module = root / "src" / "video_knowledge_pipeline" / "offline_quality_router.py"
    cli_exists = cli_script.exists()
    route_exists = route_module.exists()
    return {
        "ready": cli_exists and route_exists,
        "execution_mode": "local_cli",
        "bridge_required": False,
        "cli_script": str(cli_script),
        "cli_script_exists": cli_exists,
        "offline_quality_route_module": str(route_module),
        "offline_quality_route_discoverable": route_exists,
    }


def _health_url(call_url: str, path: str) -> str:
    if not call_url:
        return ""
    suffix = path if path.startswith("/") else f"/{path}"
    if suffix and call_url.endswith(suffix):
        return call_url[: -len(suffix)] + "/health"
    return call_url.rstrip("/") + "/health"


def _read_health(url: str, timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if not url:
        return {}, {"code": "missing_health_url", "message": "health URL could not be derived"}
    try:
        with urllib.request.urlopen(url, timeout=max(0.1, float(timeout_seconds))) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return (payload if isinstance(payload, dict) else {}), {}
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {}, {"code": "health_request_failed", "message": str(exc)}


def _scheduled_task_status(task_name: str) -> dict[str, Any]:
    if not _is_windows():
        return {"checked": False, "exists": False, "status": "unsupported_platform", "task_name": task_name}
    try:
        completed = subprocess.run(
            ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST", "/V"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"checked": True, "exists": False, "status": "query_failed", "task_name": task_name, "error": str(exc)}
    if completed.returncode != 0:
        return {
            "checked": True,
            "exists": False,
            "status": "not_registered",
            "task_name": task_name,
            "stderr_excerpt": _excerpt(completed.stderr),
        }
    parsed = _parse_schtasks_list(completed.stdout)
    return {
        "checked": True,
        "exists": True,
        "status": "registered",
        "task_name": task_name,
        "task_to_run": parsed.get("Task To Run") or parsed.get("任务运行") or "",
        "last_run_time": parsed.get("Last Run Time") or parsed.get("上次运行时间") or "",
        "last_result": parsed.get("Last Result") or parsed.get("上次结果") or "",
        "scheduled_task_state": parsed.get("Status") or parsed.get("状态") or "",
    }


def _parse_schtasks_list(stdout: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        rows[key.strip()] = value.strip()
    return rows


def _is_windows() -> bool:
    return "\\" in str(__file__) or __import__("os").name == "nt"


def _excerpt(value: str, limit: int = 600) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _next_actions(*, config_ok: bool, running: bool, health_ok: bool, check_health: bool, task_exists: bool) -> list[str]:
    if not config_ok:
        return ["fix_video_knowledge_pipeline_config"]
    if not running:
        if task_exists:
            return ["start_openclaw_http_scheduled_task", "rerun_openclaw_bridge_status"]
        return ["register_or_start_openclaw_http_bridge", "rerun_openclaw_bridge_status"]
    if check_health and not health_ok:
        return ["inspect_openclaw_http_health", "restart_openclaw_http_bridge"]
    return ["openclaw_can_call_vkp_http_bridge"]
