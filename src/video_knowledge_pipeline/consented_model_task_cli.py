from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .trusted_model_connector import execute_consented_model_task
from .trusted_model_connector_policy import TrustedModelConnectorPolicy


SCHEMA = "video_knowledge_pipeline.consented_model_task_cli.v1"
EXIT_SUCCESS = 0
EXIT_EXECUTION_FAILED = 1
EXIT_BLOCKED = 2
EXIT_INVALID = 3


def run_consented_model_task_cli(
    consent_path: str | Path,
    *,
    route_revision: str,
    write: bool | None,
    executor: Callable[..., dict[str, Any]] = execute_consented_model_task,
    policy: TrustedModelConnectorPolicy | None = None,
) -> tuple[dict[str, Any], int]:
    """Execute through the trusted connector and return JSON plus a stable exit code."""
    consent_value = str(consent_path or "").strip()
    revision = str(route_revision or "").strip()
    if not consent_value:
        return _invalid("consent_path is required"), EXIT_INVALID
    if not revision:
        return _invalid("route_revision is required and must be exact"), EXIT_INVALID
    if write is None:
        return _invalid("choose exactly one of --write or --no-write"), EXIT_INVALID

    path = Path(consent_value).expanduser().resolve()
    if not path.is_file():
        return _invalid(f"consent_path is not a file: {path}"), EXIT_INVALID
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return _invalid(f"consent_path must contain valid JSON: {path}"), EXIT_INVALID
    if not isinstance(payload, dict):
        return _invalid("consent payload must be an object"), EXIT_INVALID

    try:
        active_policy = policy or TrustedModelConnectorPolicy.from_environment(
            default_root=Path(__file__).resolve().parents[2]
        )
        active_policy.require_consent_scope(
            path,
            require_execution_contract=True,
        )
    except (OSError, ValueError) as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "policy_blocked",
            "consent_path": str(path),
            "route_revision": revision,
            "error": str(exc),
            "remote_requests_made": False,
        }, EXIT_BLOCKED

    try:
        result = executor(
            path,
            expected_route_revision=revision,
            write=bool(write),
        )
    except Exception as exc:  # Keep the machine contract stable for unexpected failures.
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "internal_error",
            "consent_path": str(path),
            "route_revision": revision,
            "error_type": type(exc).__name__,
            "remote_requests_made": False,
        }, EXIT_EXECUTION_FAILED
    return result, consented_model_task_exit_code(result)


def consented_model_task_exit_code(result: dict[str, Any]) -> int:
    if result.get("ok") is True:
        return EXIT_SUCCESS
    status = str(result.get("status") or "").strip().lower()
    if status in {
        "consent_busy",
        "consent_required",
        "cost_limit_exceeded_after_response",
        "policy_blocked",
        "route_required",
    }:
        return EXIT_BLOCKED
    if status == "invalid_input":
        return EXIT_INVALID
    return EXIT_EXECUTION_FAILED


def _invalid(message: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": "invalid_input",
        "error": message,
        "remote_requests_made": False,
    }
