from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .asr_chunk_batch_workflow import (
    SCHEMA as WORKFLOW_SCHEMA,
    SUBMISSION_TOOL,
    build_asr_chunk_batch_workflow,
)
from .storage import read_json
from .trusted_broker_http_client import (
    call_loopback_broker_tool,
    require_loopback_mcp_url,
)
from .file_hash import sha256_file as _file_sha256


SCHEMA = "video_knowledge_pipeline.asr_chunk_batch_submission.v1"
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]
Revalidator = Callable[[dict[str, Any], Path], dict[str, Any]]


def submit_asr_chunk_batch_workflow(
    workflow_path: str | Path,
    *,
    broker_url: str = "http://127.0.0.1:8766/mcp",
    execute: bool = False,
    transport: Transport | None = None,
    revalidator: Revalidator | None = None,
) -> dict[str, Any]:
    """Revalidate and optionally submit one saved ASR chunk workflow to loopback MCP."""

    path = Path(workflow_path).expanduser().resolve()
    clean_broker_url = require_loopback_mcp_url(broker_url)
    saved = _read_workflow(path)
    refreshed = (revalidator or _revalidate_saved_workflow)(saved, path)
    expected_identity = str(saved.get("workflow_sha256") or "")
    actual_identity = str(refreshed.get("workflow_sha256") or "")
    if not expected_identity or expected_identity != actual_identity:
        raise ValueError("ASR chunk workflow is stale or its current inputs changed")

    submission = refreshed.get("submission")
    submission = submission if isinstance(submission, dict) else {}
    if submission.get("tool") != SUBMISSION_TOOL:
        raise ValueError("ASR chunk workflow selected an unsupported Broker tool")
    arguments = submission.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("ASR chunk workflow submission arguments are missing")

    result = {
        "schema": SCHEMA,
        "ok": True,
        "status": "ready",
        "workflow_path": str(path),
        "workflow_sha256": _file_sha256(path),
        "workflow_identity": actual_identity,
        "chunk_count": int(refreshed.get("chunk_count") or 0),
        "broker_url": clean_broker_url,
        "broker_tool": SUBMISSION_TOOL,
        "execute": bool(execute),
        "submission_performed": False,
        "broker_control_requests_made": 0,
        "direct_provider_requests_made": False,
        "provider_execution_delegated_to_broker": False,
        "automatic_retry": False,
        "automatic_fallback": False,
    }
    if not execute:
        return result

    broker_result = (
        transport(clean_broker_url, dict(arguments))
        if transport is not None
        else call_loopback_broker_tool(
            clean_broker_url, SUBMISSION_TOOL, dict(arguments)
        )
    )
    if not isinstance(broker_result, dict):
        raise ValueError("Broker workflow response must be a JSON object")
    broker_status = str(broker_result.get("status") or "broker_response")
    explicit_ok = broker_result.get("ok")
    broker_ok = (
        bool(explicit_ok)
        if isinstance(explicit_ok, bool)
        else broker_status in {"accepted", "existing_result"}
        and bool(str(broker_result.get("job_id") or "").strip())
    )
    result.update(
        {
            "ok": broker_ok,
            "status": broker_status,
            "submission_performed": True,
            "broker_control_requests_made": 1,
            "provider_execution_delegated_to_broker": True,
            "broker_result": broker_result,
        }
    )
    return result


def _read_workflow(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ASR chunk workflow not found: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("ASR chunk workflow must be a JSON object")
    if payload.get("schema") != WORKFLOW_SCHEMA:
        raise ValueError("unsupported ASR chunk workflow schema")
    if payload.get("status") != "ready" or not payload.get("ok"):
        raise ValueError("ASR chunk workflow is not ready")
    return payload


def _revalidate_saved_workflow(saved: dict[str, Any], _: Path) -> dict[str, Any]:
    nodes = saved.get("nodes") if isinstance(saved.get("nodes"), list) else []
    consent_paths = [
        str(row.get("consent_path") or "")
        for row in nodes
        if isinstance(row, dict)
    ]
    if len(consent_paths) != int(saved.get("chunk_count") or -1):
        raise ValueError("ASR chunk workflow node count is inconsistent")
    audit = (
        saved.get("activity_audit")
        if isinstance(saved.get("activity_audit"), dict)
        else {}
    )
    submission = (
        saved.get("submission")
        if isinstance(saved.get("submission"), dict)
        else {}
    )
    arguments = (
        submission.get("arguments")
        if isinstance(submission.get("arguments"), dict)
        else {}
    )
    return build_asr_chunk_batch_workflow(
        str(saved.get("chunk_manifest") or ""),
        consent_paths,
        bundle_dir=str(saved.get("bundle_dir") or "") or None,
        activity_audit_path=str(audit.get("path") or "") or None,
        max_parallel_global=int(arguments.get("max_parallel_global") or 4),
        max_parallel_per_destination=int(
            arguments.get("max_parallel_per_destination") or 2
        ),
        write=False,
    )
