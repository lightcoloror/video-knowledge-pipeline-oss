from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .media_capability_registry import PROTOCOL
from .media_task_protocol import normalise_media_task_result, sanitise_provider_content


SCHEMA = "video_knowledge_pipeline.mediakit_cli_adapter.v1"
_TASK_ID_RE = re.compile(r"(?:task[_ -]?id)\s*[:=]\s*([A-Za-z0-9_-]+)", re.IGNORECASE)


def mediakit_cli_status(command: str = "") -> dict[str, Any]:
    configured = str(command or os.environ.get("VKP_MEDIAKIT_CLI") or "").strip()
    resolved = configured if configured and Path(configured).is_file() else shutil.which(configured or "mediakit-cli")
    return {
        "schema": SCHEMA,
        "available": bool(resolved),
        "command": str(resolved or ""),
        "install_command": "npx @volcengine/mediakit-cli install -y",
        "credential_env": "MEDIAKIT_API_KEY",
        "secrets_exposed": False,
    }


def execute_mediakit_cli_task(
    plan: dict[str, Any],
    *,
    api_key: str,
    command: str = "",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Execute a consent-bound MediaKit plan through Volcengine's official CLI."""
    if str(plan.get("schema_version") or "") != PROTOCOL:
        return _failure(plan, "invalid_media_plan")
    if str(plan.get("execution_location") or "") != "remote":
        return _failure(plan, "invalid_execution_location")
    if not str(api_key or "").strip():
        return _failure(plan, "mediakit_credential_missing")
    status = mediakit_cli_status(command)
    executable = str(status.get("command") or "")
    if not executable:
        return _failure(plan, "mediakit_cli_unavailable", details={"install_command": status["install_command"]})
    try:
        arguments = _cli_arguments(plan, executable)
    except ValueError as exc:
        return _failure(plan, "invalid_media_plan", details={"message": str(exc)})
    timeout = max(1, int(_plan_timeout(plan)))
    environment = dict(os.environ)
    environment["MEDIAKIT_API_KEY"] = str(api_key)
    try:
        submit = run(arguments, shell=False, capture_output=True, text=True, timeout=timeout, env=environment, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _failure(plan, "mediakit_cli_execution_failed", details={"exception": type(exc).__name__})
    first_payload = _decode_cli_output(submit.stdout)
    if submit.returncode != 0:
        return _failure(plan, "mediakit_cli_submission_failed", details={"returncode": int(submit.returncode), "output": _bounded_output(submit.stdout, submit.stderr)})
    task_id = _find_task_id(first_payload) or _find_task_id(submit.stdout)
    final_payload = first_payload
    requests = 1
    if task_id:
        poll_args = [executable, "shared", "query-task", "--task-id", task_id, "--poll-complete"]
        try:
            poll = run(poll_args, shell=False, capture_output=True, text=True, timeout=timeout, env=environment, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _failure(plan, "mediakit_cli_poll_failed", details={"exception": type(exc).__name__, "task_id": task_id})
        requests += 1
        if poll.returncode != 0:
            return _failure(plan, "mediakit_cli_poll_failed", details={"returncode": int(poll.returncode), "task_id": task_id, "output": _bounded_output(poll.stdout, poll.stderr)})
        final_payload = _decode_cli_output(poll.stdout)
    provider_payload = {
        "task_id": task_id,
        "status": _provider_status(final_payload),
        "result": final_payload,
    }
    result = normalise_media_task_result(
        plan,
        provider_payload,
        transport="official_mediakit_cli",
        request_count=requests,
    )
    result["candidate_only"] = True
    result["network_audit"] = {
        **dict(result.get("network_audit") or {}),
        "remote_requests_made": True,
        "source_artifact_uploaded": True,
        "provider_managed_local_upload": True,
        "fallback_attempted": False,
    }
    return result


def _cli_arguments(plan: dict[str, Any], executable: str) -> list[str]:
    task = str(plan.get("provider_task") or "").strip()
    artifacts = [row for row in plan.get("artifacts") or [] if isinstance(row, dict)]
    if not task or not artifacts:
        raise ValueError("provider task and at least one artifact are required")
    arguments = [executable, "--cloud", "video", task]
    paths = [str(row.get("path") or "") for row in artifacts]
    if any(not value for value in paths):
        raise ValueError("media plan artifact path is missing")
    if len(paths) == 1:
        data_type = str(artifacts[0].get("data_type") or "video")
        arguments.extend(["--audio-url" if data_type == "audio" else "--video-url", paths[0]])
    else:
        arguments.extend(["--video-urls", json.dumps(paths, ensure_ascii=False)])
    for key, value in sorted(dict(plan.get("parameters") or {}).items()):
        option = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            arguments.extend([option, "true" if value else "false"])
        elif isinstance(value, (str, int, float)):
            arguments.extend([option, str(value)])
        else:
            arguments.extend([option, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
    return arguments


def _decode_cli_output(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return sanitise_provider_content(value)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"output": text[:8000]}
    return sanitise_provider_content(value) if isinstance(value, dict) else {"result": sanitise_provider_content(value)}


def _find_task_id(value: Any) -> str:
    if isinstance(value, str):
        match = _TASK_ID_RE.search(value)
        return match.group(1) if match else ""
    if isinstance(value, dict):
        for key in ("task_id", "taskId", "id"):
            candidate = str(value.get(key) or "").strip()
            if candidate:
                return candidate
        for child in value.values():
            found = _find_task_id(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_task_id(child)
            if found:
                return found
    return ""


def _provider_status(value: dict[str, Any]) -> str:
    for key in ("status", "state", "task_status"):
        candidate = str(value.get(key) or "").strip()
        if candidate:
            return candidate
    return "succeeded"


def _plan_timeout(plan: dict[str, Any]) -> int:
    options = plan.get("provider_options") if isinstance(plan.get("provider_options"), dict) else {}
    return int(options.get("timeout_seconds") or 900)


def _bounded_output(stdout: str, stderr: str) -> str:
    text = (str(stdout or "") + "\n" + str(stderr or "")).strip()
    return text[:8000]


def _failure(plan: dict[str, Any], code: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.media_task_result.v1",
        "schema_version": PROTOCOL,
        "ok": False,
        "terminal": True,
        "task": str(plan.get("task") or ""),
        "status": "failed",
        "provider": "volcengine_mediakit",
        "content": {},
        "candidate_only": True,
        "error": {"code": code, "details": sanitise_provider_content(details or {})},
        "network_audit": {"transport": "official_mediakit_cli", "remote_requests_made": False, "fallback_attempted": False},
    }
