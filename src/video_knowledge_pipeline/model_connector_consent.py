from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file as _sha256
from .media_capability_registry import MEDIA_CAPABILITIES, media_capability
from .model_api_settings import resolve_model_api_route
from .model_output_contracts import normalise_output_contract
from .model_task_gateway import MODEL_TASKS, model_task_api_call
from .provider_config_safety import secretless_provider_config
from .storage import bundle_write_lock, read_json, write_json
from .time_utils import parse_utc_datetime_or_none as _parse_datetime, utc_now_iso_seconds

SCHEMA_V1 = "video_knowledge_pipeline.model_connector_consent.v1"
SCHEMA_V2 = "video_knowledge_pipeline.model_connector_consent.v2"
SCHEMA = SCHEMA_V2
SUPPORTED_SCHEMAS = frozenset({SCHEMA_V1, SCHEMA_V2})
STATUS_SCHEMA = "video_knowledge_pipeline.model_connector_consent_status.v1"
UPLOAD_MANIFEST_SCHEMA = "video_knowledge_pipeline.model_upload_manifest.v1"
DEFAULT_FILENAME = "model-connector-consent.json"
DEFAULT_LOCK_TIMEOUT_SECONDS = 10.0
DEFAULT_STALE_LOCK_SECONDS = 900.0
_PROVIDER_OPTION_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SECRET_PROVIDER_OPTION_KEY_RE = re.compile(
    r"(?:api_?key|(?:^|_)token(?:$|_)|secret|password|authorization)",
    re.IGNORECASE,
)
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")


def create_model_connector_consent(
    root_dir: str | Path,
    *,
    task: str,
    artifact_paths: list[str | Path],
    provider_config: dict[str, Any] | None = None,
    route_snapshot: dict[str, Any] | None = None,
    instructions: str = "",
    asr_prompt: str = "",
    output_contract: dict[str, Any] | None = None,
    purpose: str = "approved online model task",
    expires_hours: float = 24.0,
    max_calls: int = 1,
    max_estimated_cost_usd: float | None = None,
    max_cost_per_call_usd: float | None = None,
    max_retries_per_call: int | None = None,
    confirm_data_export: bool = False,
    output_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_key = _task_key(task)
    task_spec = _connector_task_spec(task_key)
    model_type = str(task_spec.get("model_type") or "")
    locked_asr_prompt = _normalise_asr_prompt(asr_prompt) if model_type == "asr" else ""
    if asr_prompt and model_type != "asr":
        raise ValueError("asr_prompt is accepted only for ASR tasks")
    if expires_hours <= 0:
        raise ValueError("expires_hours must be positive")
    if max_calls <= 0:
        raise ValueError("max_calls must be positive")
    total_cost_limit = _positive_money(
        max_estimated_cost_usd, label="max_estimated_cost_usd"
    )
    per_call_cost_limit = (
        _positive_money(max_cost_per_call_usd, label="max_cost_per_call_usd")
        if max_cost_per_call_usd is not None
        else _money(total_cost_limit / int(max_calls))
    )
    if per_call_cost_limit > total_cost_limit:
        raise ValueError("max_cost_per_call_usd cannot exceed max_estimated_cost_usd")
    retry_limit = _optional_retry_limit(max_retries_per_call)
    artifacts = [
        _artifact_record(Path(value).expanduser().resolve()) for value in artifact_paths
    ]
    if not artifacts:
        raise ValueError("at least one artifact is required")
    _validate_artifact_modalities(task_key, artifacts)
    semantic_pack = _transcript_semantic_pack_from_artifacts(artifacts)
    if semantic_pack is not None:
        from .transcript_semantic_correction import (
            transcript_semantic_correction_model_instructions,
            transcript_semantic_correction_output_contract,
        )

        locked_output_contract = normalise_output_contract(
            transcript_semantic_correction_output_contract(output_contract)
        )
        locked_instructions = transcript_semantic_correction_model_instructions(
            instructions
        )
    else:
        locked_output_contract = normalise_output_contract(output_contract)
        locked_instructions = str(instructions or "").strip()
    if route_snapshot is None:
        if task_key in MEDIA_CAPABILITIES:
            raise ValueError(
                "media capability consent requires a fixed route_snapshot; provider_config is not accepted"
            )
        preview = model_task_api_call(
            task_key,
            provider_config=secretless_provider_config(provider_config),
            execute=False,
            write=False,
        )
        provider = _provider_identity(preview)
        route = _direct_route_snapshot(task_key, provider)
    else:
        if provider_config:
            raise ValueError(
                "provider_config and route_snapshot are mutually exclusive"
            )
        route = _normalise_route_snapshot(route_snapshot)
        provider = _legacy_provider_identity(route["deployments"][0])
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else root / DEFAULT_FILENAME
    )
    confirmed = bool(confirm_data_export)
    upload_manifest = _upload_manifest(artifacts)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "consent_id": str(uuid4()),
        "status": "active" if confirmed else "confirmation_required",
        "user_confirmed_data_export": confirmed,
        "task": task_key,
        "model_type": task_spec["model_type"],
        "provider": provider,
        "authorized_deployments": route["deployments"],
        "authorized_destinations": _route_destinations(route),
        "route": {
            "route_id": route["route_id"],
            "route_revision": route["route_revision"],
            "virtual_model": route["virtual_model"],
            "execution_location": route["execution_location"],
            "route_snapshot_sha256": route["route_snapshot_sha256"],
            **(
                {"destinations": route["destinations"]}
                if route.get("destinations")
                else {}
            ),
        },
        "instructions": locked_instructions,
        "instructions_sha256": _text_sha256(locked_instructions),
        "instruction_transport": "audit_only"
        if model_type == "asr"
        else "model_prompt",
        "asr_prompt": locked_asr_prompt,
        "asr_prompt_sha256": _text_sha256(locked_asr_prompt),
        "asr_prompt_transport": "provider_audio_prompt"
        if model_type == "asr"
        else "not_applicable",
        "output_contract": locked_output_contract,
        "output_contract_sha256": canonical_json_sha256(locked_output_contract),
        "artifacts": artifacts,
        "upload_manifest": upload_manifest,
        "operator_confirmation": {
            "confirmed": confirmed,
            "confirmed_at": created_at.isoformat() if confirmed else "",
            "confirmation_method": "visible_operator_shell" if confirmed else "",
            "exact_manifest_sha256": upload_manifest["manifest_sha256"],
        },
        "scope": {
            "purpose": str(purpose or "approved online model task"),
            "max_calls": int(max_calls),
            "max_estimated_cost_usd": total_cost_limit,
            "max_cost_per_call_usd": per_call_cost_limit,
            "total_bytes": sum(int(row["bytes"]) for row in artifacts),
            "allowed_data_types": sorted({str(row["data_type"]) for row in artifacts}),
            "prohibited_data_types": [
                "credentials",
                "api_keys",
                "cookies",
                "browser_sessions",
            ],
        },
        "usage": {
            "calls_attempted": 0,
            "calls_completed": 0,
            "cost_committed_usd": 0.0,
            "cost_reported_usd": 0.0,
            "cost_unreported_calls": 0,
            "cost_limit_exceeded": False,
            "last_attempt_at": "",
        },
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(hours=float(expires_hours))).isoformat(),
        "operator_boundary": {
            "project_gate_only": True,
            "does_not_override_agent_platform_policy": True,
            "consent_creation_not_exposed_by_mcp": True,
            "api_key_persisted": False,
            "only_hashed_artifacts_may_be_read": True,
            "remote_execution_default": "deny",
            "automatic_publish_allowed": False,
            "unlisted_file_upload_allowed": False,
            "silent_local_cloud_fallback_allowed": False,
        },
        "consent_path": str(path),
    }
    if retry_limit is not None:
        payload["scope"]["max_retries_per_call"] = retry_limit
    payload["consent_sha256"] = _payload_sha256(payload)
    if write:
        write_json(path, payload)
        path.with_suffix(".md").write_text(
            render_model_connector_consent_markdown(payload), encoding="utf-8"
        )
    return payload


def resolve_model_connector_consent_route(
    task: str,
    *,
    route_id: str,
    route_revision: str,
    settings_path: str | Path | None = None,
    policy: Any | None = None,
) -> dict[str, Any]:
    """Resolve and validate the exact configured remote route before consent creation."""
    task_key = _task_key(task)
    if not str(route_id or "").strip() or not str(route_revision or "").strip():
        raise ValueError(
            "route_id and route_revision are both required for route-based consent"
        )
    model_type = str(MODEL_TASKS[task_key]["model_type"])
    route = resolve_model_api_route(
        model_type,
        execution_location="remote",
        settings_path=settings_path,
    )
    if str(route.get("route_id") or "") != str(route_id):
        raise ValueError("configured route_id differs from requested route_id")
    if str(route.get("route_revision") or "") != str(route_revision):
        raise ValueError(
            "configured route revision differs from requested route revision"
        )
    if str(route.get("execution_location") or "") != "remote":
        raise ValueError("route-based export consent requires a remote route")
    deployments = (
        route.get("deployments") if isinstance(route.get("deployments"), list) else []
    )
    if not deployments:
        raise ValueError("configured remote route has no deployments")
    if policy is not None:
        for deployment in deployments:
            policy.require_destination_identity(deployment)
    return route


def validate_model_connector_consent(
    consent_path: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    route_snapshot: dict[str, Any] | None = None,
    expected_route_revision: str = "",
    expected_task: str = "",
    expected_calls: int = 1,
) -> dict[str, Any]:
    path = Path(consent_path).expanduser().resolve()
    blockers: list[dict[str, str]] = []
    if not path.is_file():
        return _status(
            path,
            "missing",
            [{"key": "consent_missing", "message": f"Consent file not found: {path}"}],
        )
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _status(
            path, "invalid", [{"key": "consent_unreadable", "message": str(exc)}]
        )
    schema = str(payload.get("schema") or "") if isinstance(payload, dict) else ""
    if not isinstance(payload, dict) or schema not in SUPPORTED_SCHEMAS:
        return _status(
            path,
            "invalid",
            [
                {
                    "key": "consent_schema_invalid",
                    "message": "Unsupported consent schema",
                }
            ],
        )
    if payload.get("status") != "active" or not payload.get(
        "user_confirmed_data_export"
    ):
        blockers.append(
            {
                "key": "consent_not_active",
                "message": "Data export was not explicitly confirmed or consent is inactive",
            }
        )
    if str(payload.get("consent_sha256") or "") != _payload_sha256(payload):
        blockers.append(
            {
                "key": "consent_integrity_failed",
                "message": "Consent changed after creation",
            }
        )
    task = _task_key(str(payload.get("task") or ""))
    if expected_task and task != _task_key(expected_task):
        blockers.append(
            {
                "key": "consent_task_mismatch",
                "message": "Requested task differs from consent",
            }
        )

    authorised_provider = (
        payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    )
    if provider_config is not None:
        if task in MEDIA_CAPABILITIES:
            blockers.append(
                {
                    "key": "media_provider_config_forbidden",
                    "message": "Media capability validation accepts only the saved route snapshot",
                }
            )
        else:
            try:
                preview = model_task_api_call(
                    task,
                    provider_config=secretless_provider_config(provider_config),
                    execute=False,
                    write=False,
                )
                current_provider = _provider_identity(preview)
            except (TypeError, ValueError) as exc:
                blockers.append({"key": "provider_config_invalid", "message": str(exc)})
                current_provider = {}
            for key in ("provider", "model", "base_url", "interface"):
                if str(current_provider.get(key) or "") != str(
                    authorised_provider.get(key) or ""
                ):
                    blockers.append(
                        {
                            "key": f"consent_provider_{key}_mismatch",
                            "message": f"Provider {key} differs from consent",
                        }
                    )

    authorised_deployments = (
        payload.get("authorized_deployments")
        if schema == SCHEMA_V2
        and isinstance(payload.get("authorized_deployments"), list)
        else [authorised_provider]
    )
    route = (
        payload.get("route")
        if schema == SCHEMA_V2 and isinstance(payload.get("route"), dict)
        else {}
    )
    if schema == SCHEMA_V2:
        try:
            stored_deployments = [
                _normalise_deployment(row) for row in authorised_deployments
            ]
        except ValueError as exc:
            blockers.append(
                {"key": "consent_authorized_deployments_invalid", "message": str(exc)}
            )
            stored_deployments = []
        stored_route = {
            "route_id": str(route.get("route_id") or ""),
            "route_revision": str(route.get("route_revision") or ""),
            "virtual_model": str(route.get("virtual_model") or ""),
            "execution_location": str(route.get("execution_location") or ""),
            "deployments": stored_deployments,
        }
        if isinstance(route.get("destinations"), list):
            stored_route["destinations"] = _normalise_route_destinations(
                route.get("destinations")
            )
        stored_snapshot_hash = _route_snapshot_sha256(stored_route)
        if not stored_deployments:
            blockers.append(
                {
                    "key": "consent_authorized_deployments_empty",
                    "message": "Consent has no authorized deployments",
                }
            )
        if str(route.get("route_snapshot_sha256") or "") != stored_snapshot_hash:
            blockers.append(
                {
                    "key": "consent_route_snapshot_integrity_failed",
                    "message": "Consent route snapshot changed after creation",
                }
            )
        expected_destinations = _route_destinations(stored_route)
        raw_authorised_destinations = payload.get("authorized_destinations")
        if isinstance(raw_authorised_destinations, list):
            try:
                authorised_destinations = _normalise_route_destinations(
                    raw_authorised_destinations
                )
            except ValueError as exc:
                blockers.append(
                    {
                        "key": "consent_authorized_destinations_invalid",
                        "message": str(exc),
                    }
                )
                authorised_destinations = []
            if authorised_destinations != expected_destinations:
                blockers.append(
                    {
                        "key": "consent_authorized_destinations_mismatch",
                        "message": "Authorized destination set differs from the locked route",
                    }
                )
        else:
            authorised_destinations = expected_destinations
        if expected_route_revision and str(route.get("route_revision") or "") != str(
            expected_route_revision
        ):
            blockers.append(
                {
                    "key": "consent_route_revision_mismatch",
                    "message": "Requested route revision differs from consent",
                }
            )
        if route_snapshot is not None:
            try:
                current_route = _normalise_route_snapshot(route_snapshot)
            except ValueError as exc:
                blockers.append({"key": "route_snapshot_invalid", "message": str(exc)})
                current_route = {}
            if current_route:
                if current_route["deployments"] != stored_deployments:
                    blockers.append(
                        {
                            "key": "consent_authorized_deployments_mismatch",
                            "message": "Current route deployments differ from consent",
                        }
                    )
                if str(current_route["route_revision"]) != str(
                    route.get("route_revision") or ""
                ):
                    blockers.append(
                        {
                            "key": "consent_route_revision_mismatch",
                            "message": "Current route revision differs from consent",
                        }
                    )
                if str(current_route["route_snapshot_sha256"]) != str(
                    route.get("route_snapshot_sha256") or ""
                ):
                    blockers.append(
                        {
                            "key": "consent_route_snapshot_mismatch",
                            "message": "Current route snapshot differs from consent",
                        }
                    )
                if _route_destinations(current_route) != expected_destinations:
                    blockers.append(
                        {
                            "key": "consent_route_destinations_mismatch",
                            "message": "Current route destinations differ from consent",
                        }
                    )
    else:
        authorised_destinations = _route_destinations(
            {"deployments": authorised_deployments}
        )

    if schema == SCHEMA_V2 and provider_config is not None and route_snapshot is None:
        if not str(route.get("route_id") or "").startswith("direct-"):
            blockers.append(
                {
                    "key": "consent_route_snapshot_required",
                    "message": "Route-based v2 consent must be validated against the current route snapshot",
                }
            )
    if schema == SCHEMA_V1 and route_snapshot is not None:
        try:
            current_route = _normalise_route_snapshot(route_snapshot)
            current_deployments = current_route["deployments"]
        except ValueError as exc:
            blockers.append({"key": "route_snapshot_invalid", "message": str(exc)})
            current_deployments = []
        current_provider = (
            _legacy_provider_identity(current_deployments[0])
            if len(current_deployments) == 1
            else {}
        )
        for key in ("provider", "model", "base_url", "interface"):
            if str(current_provider.get(key) or "") != str(
                authorised_provider.get(key) or ""
            ):
                blockers.append(
                    {
                        "key": f"consent_provider_{key}_mismatch",
                        "message": f"Current singleton route provider {key} differs from v1 consent",
                    }
                )
    artifacts = (
        payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    )
    for row in artifacts:
        if not isinstance(row, dict):
            blockers.append(
                {
                    "key": "artifact_record_invalid",
                    "message": "Artifact record is not an object",
                }
            )
            continue
        artifact_path = Path(str(row.get("path") or "")).expanduser().resolve()
        if not artifact_path.is_file():
            blockers.append(
                {
                    "key": "artifact_missing",
                    "message": f"Artifact missing: {artifact_path}",
                }
            )
            continue
        if int(row.get("bytes") or -1) != artifact_path.stat().st_size or str(
            row.get("sha256") or ""
        ) != _sha256(artifact_path):
            blockers.append(
                {
                    "key": "artifact_changed",
                    "message": f"Artifact changed after consent: {artifact_path}",
                }
            )
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    requested = max(0, int(expected_calls))
    remaining = int(scope.get("max_calls") or 0) - int(
        usage.get("calls_attempted") or 0
    )
    if requested > remaining:
        blockers.append(
            {
                "key": "consent_call_limit_exceeded",
                "message": "Requested calls exceed remaining consent allowance",
            }
        )
    cost_status = (
        _validate_v2_export_contract(
            payload,
            artifacts=artifacts,
            expected_calls=requested,
            blockers=blockers,
        )
        if schema == SCHEMA_V2
        else _legacy_cost_status()
    )
    expires_at = _parse_datetime(payload.get("expires_at"))
    if expires_at is None or expires_at <= datetime.now(timezone.utc):
        blockers.append({"key": "consent_expired", "message": "Consent expired"})
    result = _status(path, "active" if not blockers else "blocked", blockers)
    result.update(
        {
            "consent_id": str(payload.get("consent_id") or ""),
            "consent_schema": schema,
            "task": task,
            "model_type": str(payload.get("model_type") or ""),
            "provider": authorised_provider,
            "authorized_deployments": authorised_deployments,
            "authorized_destinations": authorised_destinations,
            "route": route,
            "artifacts": artifacts,
            "upload_manifest": payload.get("upload_manifest")
            if schema == SCHEMA_V2
            else {},
            "operator_confirmation": payload.get("operator_confirmation")
            if schema == SCHEMA_V2
            else {},
            "instructions": str(payload.get("instructions") or ""),
            "instruction_transport": str(
                payload.get("instruction_transport") or "legacy_unspecified"
            ),
            "asr_prompt": str(payload.get("asr_prompt") or ""),
            "asr_prompt_sha256": str(payload.get("asr_prompt_sha256") or ""),
            "asr_prompt_transport": str(
                payload.get("asr_prompt_transport") or "legacy_unspecified"
            ),
            "output_contract": (
                payload.get("output_contract")
                if isinstance(payload.get("output_contract"), dict)
                else {}
            ),
            "output_contract_sha256": str(payload.get("output_contract_sha256") or ""),
            "scope": scope,
            "usage": usage,
            "remaining_calls": max(0, remaining),
            "remaining_estimated_cost_usd": cost_status["remaining_estimated_cost_usd"],
            "requested_cost_reservation_usd": cost_status[
                "requested_cost_reservation_usd"
            ],
            "expires_at": str(payload.get("expires_at") or ""),
            "platform_policy_may_still_block": True,
        }
    )
    return result


def model_connector_consent_lock_path(consent_path: str | Path) -> Path:
    path = Path(consent_path).expanduser().resolve()
    return path.parent / f".{path.name}.usage.lock"


def reserve_model_connector_attempt(
    consent_path: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    route_snapshot: dict[str, Any] | None = None,
    expected_route_revision: str = "",
    expected_task: str = "",
    expected_calls: int = 1,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    stale_after_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
) -> dict[str, Any]:
    path = Path(consent_path).expanduser().resolve()
    requested = max(1, int(expected_calls))
    lock_path = model_connector_consent_lock_path(path)
    try:
        with bundle_write_lock(
            path.parent,
            operation="model_connector_consent_reservation",
            timeout_seconds=lock_timeout_seconds,
            lock_name=lock_path.name,
            stale_after_seconds=stale_after_seconds,
        ):
            status = validate_model_connector_consent(
                path,
                provider_config=provider_config,
                route_snapshot=route_snapshot,
                expected_route_revision=expected_route_revision,
                expected_task=expected_task,
                expected_calls=requested,
            )
            if not status.get("valid"):
                status["reserved"] = False
                return status
            if status.get("consent_schema") != SCHEMA_V2:
                status.update(
                    {
                        "status": "blocked",
                        "valid": False,
                        "reserved": False,
                    }
                )
                status.setdefault("blockers", []).append(
                    {
                        "key": "consent_v2_required",
                        "message": "Remote execution requires consent v2; v1 remains status-only compatibility",
                    }
                )
                return status
            requested_cost = _money(status.get("requested_cost_reservation_usd") or 0)
            payload = read_json(path)
            _update_usage(
                payload,
                attempted_delta=requested,
                cost_committed_delta=requested_cost,
            )
            _write_consent_payload(path, payload)
            usage = (
                payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            )
            scope = (
                payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
            )
            status.update(
                {
                    "reserved": True,
                    "reservation_count": requested,
                    "reserved_cost_usd": requested_cost,
                    "upload_manifest_sha256": str(
                        (payload.get("upload_manifest") or {}).get("manifest_sha256")
                        or ""
                    ),
                    "usage": usage,
                    "remaining_calls": max(
                        0,
                        int(scope.get("max_calls") or 0)
                        - int(usage.get("calls_attempted") or 0),
                    ),
                    "remaining_estimated_cost_usd": _money(
                        float(scope.get("max_estimated_cost_usd") or 0)
                        - float(usage.get("cost_committed_usd") or 0)
                    ),
                }
            )
            return status
    except RuntimeError as exc:
        if "bundle_write_lock_busy" not in str(exc):
            raise
        result = _status(
            path,
            "consent_busy",
            [
                {
                    "key": "consent_reservation_busy",
                    "message": "Another process is updating this consent; retry later",
                }
            ],
        )
        result["reserved"] = False
        result["lock_path"] = str(lock_path)
        result["next_action"] = "retry_consent_reservation"
        return result


def record_model_connector_attempt(
    consent_path: str | Path,
    *,
    completed: bool = False,
    completed_calls: int | None = None,
    reserved_cost_usd: float = 0.0,
    reported_cost_usd: float | None = None,
    cost_unreported_calls: int = 0,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    stale_after_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
) -> dict[str, Any]:
    path = Path(consent_path).expanduser().resolve()
    lock_path = model_connector_consent_lock_path(path)
    with bundle_write_lock(
        path.parent,
        operation="model_connector_consent_usage",
        timeout_seconds=lock_timeout_seconds,
        lock_name=lock_path.name,
        stale_after_seconds=stale_after_seconds,
    ):
        payload = read_json(path)
        if completed_calls is None:
            attempted_delta = 0 if completed else 1
            completed_delta = 1 if completed else 0
        else:
            attempted_delta = 0
            completed_delta = max(0, int(completed_calls))
        _update_usage(
            payload,
            attempted_delta=attempted_delta,
            completed_delta=completed_delta,
        )
        _reconcile_cost_usage(
            payload,
            reserved_cost_usd=reserved_cost_usd,
            reported_cost_usd=reported_cost_usd,
            cost_unreported_calls=cost_unreported_calls,
        )
        _write_consent_payload(path, payload)
        return payload


def _update_usage(
    payload: dict[str, Any],
    *,
    attempted_delta: int = 0,
    completed_delta: int = 0,
    cost_committed_delta: float = 0.0,
) -> None:
    if not isinstance(payload, dict) or payload.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("unsupported consent schema")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    attempted = int(usage.get("calls_attempted") or 0) + max(0, int(attempted_delta))
    completed = min(
        attempted, int(usage.get("calls_completed") or 0) + max(0, int(completed_delta))
    )
    usage["calls_attempted"] = attempted
    usage["calls_completed"] = completed
    usage["cost_committed_usd"] = _money(
        float(usage.get("cost_committed_usd") or 0)
        + max(0.0, float(cost_committed_delta))
    )
    usage["last_attempt_at"] = (
        utc_now_iso_seconds()
    )
    payload["usage"] = usage
    payload["consent_sha256"] = _payload_sha256(payload)


def _reconcile_cost_usage(
    payload: dict[str, Any],
    *,
    reserved_cost_usd: float,
    reported_cost_usd: float | None,
    cost_unreported_calls: int,
) -> None:
    if payload.get("schema") != SCHEMA_V2:
        return
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    reserved = _money(max(0.0, float(reserved_cost_usd)))
    committed = _money(max(0.0, float(usage.get("cost_committed_usd") or 0)))
    if reported_cost_usd is None:
        usage["cost_unreported_calls"] = int(
            usage.get("cost_unreported_calls") or 0
        ) + max(0, int(cost_unreported_calls))
    else:
        reported = _money(max(0.0, float(reported_cost_usd)))
        committed = _money(max(0.0, committed - reserved) + reported)
        usage["cost_reported_usd"] = _money(
            float(usage.get("cost_reported_usd") or 0) + reported
        )
    usage["cost_committed_usd"] = committed
    limit = _money(float(scope.get("max_estimated_cost_usd") or 0))
    usage["cost_limit_exceeded"] = bool(committed > limit)
    payload["usage"] = usage
    payload["consent_sha256"] = _payload_sha256(payload)


def _write_consent_payload(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
    path.with_suffix(".md").write_text(
        render_model_connector_consent_markdown(payload), encoding="utf-8"
    )


def revoke_model_connector_consent(
    consent_path: str | Path, *, write: bool = True
) -> dict[str, Any]:
    path = Path(consent_path).expanduser().resolve()
    if not write:
        return _revoke_model_connector_consent_unlocked(path, write=False)
    lock_path = model_connector_consent_lock_path(path)
    try:
        with bundle_write_lock(
            path.parent,
            operation="model_connector_consent_revoke",
            timeout_seconds=DEFAULT_LOCK_TIMEOUT_SECONDS,
            lock_name=lock_path.name,
            stale_after_seconds=DEFAULT_STALE_LOCK_SECONDS,
        ):
            return _revoke_model_connector_consent_unlocked(path, write=True)
    except RuntimeError as exc:
        if "bundle_write_lock_busy" not in str(exc):
            raise
        result = _status(
            path,
            "consent_busy",
            [
                {
                    "key": "consent_revoke_busy",
                    "message": "Another process is updating this consent; retry later",
                }
            ],
        )
        result["next_action"] = "retry_consent_revoke"
        return result


def _revoke_model_connector_consent_unlocked(
    path: Path, *, write: bool
) -> dict[str, Any]:
    if not path.is_file():
        return _status(
            path,
            "missing",
            [{"key": "consent_missing", "message": f"Consent file not found: {path}"}],
        )
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema") not in SUPPORTED_SCHEMAS:
        return _status(
            path,
            "invalid",
            [
                {
                    "key": "consent_schema_invalid",
                    "message": "Unsupported consent schema",
                }
            ],
        )
    payload["status"] = "revoked"
    payload["revoked_at"] = (
        utc_now_iso_seconds()
    )
    payload["consent_sha256"] = _payload_sha256(payload)
    if write:
        _write_consent_payload(path, payload)
    return {
        "schema": str(payload.get("schema") or ""),
        "status": "revoked",
        "consent_path": str(path),
        "consent_id": payload.get("consent_id"),
    }


def render_model_connector_consent_markdown(payload: dict[str, Any]) -> str:
    provider = (
        payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    )
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    manifest = (
        payload.get("upload_manifest")
        if isinstance(payload.get("upload_manifest"), dict)
        else {}
    )
    confirmation = (
        payload.get("operator_confirmation")
        if isinstance(payload.get("operator_confirmation"), dict)
        else {}
    )
    lines = [
        "# Online Model Connector Consent",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Consent ID: `{payload.get('consent_id', '')}`",
        f"- Task: `{payload.get('task', '')}` / `{payload.get('model_type', '')}`",
        f"- Provider: `{provider.get('provider', '')}` / `{provider.get('model', '')}`",
        f"- Endpoint: `{provider.get('base_url', '')}`",
        f"- Purpose: `{scope.get('purpose', '')}`",
        f"- Calls: `{usage.get('calls_attempted', 0)}` attempted / `{scope.get('max_calls', 0)}` authorised",
        f"- Cost: $`{usage.get('cost_committed_usd', 0)}` committed / $`{scope.get('max_estimated_cost_usd', 0)}` authorised",
        f"- Per-call cost ceiling: $`{scope.get('max_cost_per_call_usd', 0)}`",
        f"- Per-call retry ceiling: `{scope.get('max_retries_per_call', 'route_default')}`",
        f"- Operator confirmed: `{confirmation.get('confirmed', False)}` at `{confirmation.get('confirmed_at', '')}`",
        f"- Upload manifest SHA-256: `{manifest.get('manifest_sha256', '')}`",
        f"- Expires: `{payload.get('expires_at', '')}`",
        "",
        "## Exact authorised upload manifest",
        "",
    ]
    for row in payload.get("artifacts") or []:
        lines.append(
            f"- `{row.get('data_type', '')}` `{row.get('path', '')}` ({row.get('bytes', 0)} bytes, SHA-256 `{row.get('sha256', '')}`)"
        )
    lines.extend(
        [
            "",
            "VKP denies remote execution by default. Only the files listed above may be uploaded under the locked provider route, consent v2, operator confirmation, call limit, and cost limit.",
            "This consent does not override the agent host's policy. Automatic publishing, unlisted uploads, silent local/cloud fallback, and persisted secrets remain prohibited.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact not found: {path}")
    return {
        "path": str(path),
        "data_type": _data_type(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _transcript_semantic_pack_from_artifacts(
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for artifact in artifacts:
        path = Path(str(artifact.get("path") or ""))
        if path.suffix.lower() != ".json":
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema")
            == "video_knowledge_pipeline.transcript_semantic_correction_pack.v1"
        ):
            return payload
    return None


def _money(value: Any) -> float:
    return round(float(value), 8)


def _positive_money(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a positive finite USD amount")
    try:
        amount = _money(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite USD amount") from exc
    if amount <= 0 or not math.isfinite(amount):
        raise ValueError(f"{label} must be a positive finite USD amount")
    return amount


def _optional_retry_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_retries_per_call must be an integer")
    if value < 0 or value > 10:
        raise ValueError("max_retries_per_call must be between 0 and 10")
    return value


def _upload_manifest(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    files = [dict(row) for row in artifacts]
    return {
        "schema": UPLOAD_MANIFEST_SCHEMA,
        "file_count": len(files),
        "total_bytes": sum(int(row.get("bytes") or 0) for row in files),
        "files": files,
        "manifest_sha256": _upload_manifest_sha256(files),
    }


def _upload_manifest_sha256(files: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "path": str(row.get("path") or ""),
            "data_type": str(row.get("data_type") or ""),
            "bytes": int(row.get("bytes") or 0),
            "sha256": str(row.get("sha256") or ""),
        }
        for row in files
    ]
    return canonical_json_sha256(canonical)


def _legacy_cost_status() -> dict[str, Any]:
    return {
        "remaining_estimated_cost_usd": None,
        "requested_cost_reservation_usd": None,
        "max_estimated_cost_usd": None,
        "max_cost_per_call_usd": None,
    }


def _validate_v2_export_contract(
    payload: dict[str, Any],
    *,
    artifacts: list[dict[str, Any]],
    expected_calls: int,
    blockers: list[dict[str, str]],
) -> dict[str, Any]:
    manifest = payload.get("upload_manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != UPLOAD_MANIFEST_SCHEMA
    ):
        blockers.append(
            {
                "key": "consent_upload_manifest_missing",
                "message": "Consent v2 requires an explicit upload manifest",
            }
        )
        manifest = {}
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    manifest_hash = _upload_manifest_sha256(files) if files else ""
    paths = [str(row.get("path") or "") for row in files if isinstance(row, dict)]
    if files != artifacts:
        blockers.append(
            {
                "key": "consent_upload_manifest_mismatch",
                "message": "Upload manifest files differ from authorized artifacts",
            }
        )
    if len(paths) != len(files) or len(paths) != len(set(paths)):
        blockers.append(
            {
                "key": "consent_upload_manifest_duplicate",
                "message": "Upload manifest must list each file exactly once",
            }
        )
    if (
        int(manifest.get("file_count") or -1) != len(files)
        or int(manifest.get("total_bytes") or -1)
        != sum(int(row.get("bytes") or 0) for row in files if isinstance(row, dict))
        or str(manifest.get("manifest_sha256") or "") != manifest_hash
    ):
        blockers.append(
            {
                "key": "consent_upload_manifest_integrity_failed",
                "message": "Upload manifest count, bytes, or SHA-256 changed",
            }
        )

    confirmation = (
        payload.get("operator_confirmation")
        if isinstance(payload.get("operator_confirmation"), dict)
        else {}
    )
    if (
        not confirmation.get("confirmed")
        or str(confirmation.get("confirmation_method") or "")
        not in {"visible_operator_shell", "parent_business_authorization"}
        or _parse_datetime(confirmation.get("confirmed_at")) is None
        or str(confirmation.get("exact_manifest_sha256") or "") != manifest_hash
    ):
        blockers.append(
            {
                "key": "consent_operator_confirmation_invalid",
                "message": "Operator confirmation must bind to the exact upload manifest",
            }
        )
    elif (
        str(confirmation.get("confirmation_method") or "")
        == "parent_business_authorization"
    ):
        from .model_business_authorization import (
            validate_parent_authorized_child_consent,
        )

        parent_status = validate_parent_authorized_child_consent(payload)
        if not parent_status.get("valid"):
            blockers.extend(parent_status.get("blockers") or [])

    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    try:
        total_limit = _positive_money(
            scope.get("max_estimated_cost_usd"), label="scope.max_estimated_cost_usd"
        )
        per_call_limit = _positive_money(
            scope.get("max_cost_per_call_usd"), label="scope.max_cost_per_call_usd"
        )
        if per_call_limit > total_limit:
            raise ValueError(
                "scope.max_cost_per_call_usd cannot exceed total cost limit"
            )
    except ValueError as exc:
        blockers.append({"key": "consent_cost_limit_invalid", "message": str(exc)})
        total_limit = 0.0
        per_call_limit = 0.0
    try:
        retry_limit = _optional_retry_limit(scope.get("max_retries_per_call"))
    except ValueError as exc:
        blockers.append(
            {
                "key": "consent_retry_limit_invalid",
                "message": str(exc),
            }
        )
        retry_limit = None
    try:
        committed = _money(usage.get("cost_committed_usd") or 0)
    except (TypeError, ValueError):
        committed = -1.0
    if committed < 0 or not math.isfinite(committed):
        blockers.append(
            {
                "key": "consent_cost_usage_invalid",
                "message": "Consent cost usage is invalid",
            }
        )
        committed = max(0.0, committed if math.isfinite(committed) else 0.0)
    requested_cost = _money(per_call_limit * max(0, int(expected_calls)))
    remaining_cost = _money(max(0.0, total_limit - committed))
    if requested_cost > remaining_cost:
        blockers.append(
            {
                "key": "consent_cost_limit_exceeded",
                "message": "Requested calls exceed remaining consent cost allowance",
            }
        )
    if committed > total_limit or bool(usage.get("cost_limit_exceeded")):
        blockers.append(
            {
                "key": "consent_cost_limit_already_exceeded",
                "message": "Consent cost limit was already exceeded",
            }
        )
    return {
        "remaining_estimated_cost_usd": remaining_cost,
        "requested_cost_reservation_usd": requested_cost,
        "max_estimated_cost_usd": total_limit,
        "max_cost_per_call_usd": per_call_limit,
        "max_retries_per_call": retry_limit,
    }


def _data_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".jsonl", ".srt", ".vtt", ".csv", ".tsv"}:
        return "text"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        return "image"
    if suffix in {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}:
        return "audio"
    if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".ts", ".m4v"}:
        return "video"
    raise ValueError(f"unsupported artifact type: {path.suffix or path.name}")


def _validate_artifact_modalities(task: str, artifacts: list[dict[str, Any]]) -> None:
    if task in MEDIA_CAPABILITIES:
        capability = media_capability(task)
        types = [str(row["data_type"]) for row in artifacts]
        allowed = set(capability["artifact_types"])
        if any(value not in allowed for value in types):
            raise ValueError(
                f"{task} artifacts must use one of: {', '.join(sorted(allowed))}"
            )
        minimum = int(capability["min_artifacts"])
        maximum = int(capability["max_artifacts"])
        if not minimum <= len(artifacts) <= maximum:
            raise ValueError(
                f"{task} requires between {minimum} and {maximum} artifacts"
            )
        return
    modality = str(_connector_task_spec(task)["modality"])
    types = [str(row["data_type"]) for row in artifacts]
    if modality == "text" and any(value != "text" for value in types):
        raise ValueError("text tasks accept text artifacts only")
    if modality in {"image", "multi_image"} and any(
        value != "image" for value in types
    ):
        raise ValueError("vision tasks accept image artifacts only")
    if modality == "audio" and (types != ["audio"]):
        raise ValueError("ASR tasks require exactly one audio artifact")


def _provider_identity(preview: dict[str, Any]) -> dict[str, str]:
    plan = (
        preview.get("request_plan")
        if isinstance(preview.get("request_plan"), dict)
        else {}
    )
    provider = plan.get("provider") if isinstance(plan.get("provider"), dict) else {}
    return {
        "provider": str(provider.get("provider") or ""),
        "model": str(provider.get("model") or ""),
        "base_url": _normalise_url(str(provider.get("base_url") or "")),
        "interface": str(provider.get("interface") or plan.get("interface") or ""),
    }


def _direct_route_snapshot(task: str, provider: dict[str, Any]) -> dict[str, Any]:
    deployment = _normalise_deployment(provider)
    seed = {
        "route_id": f"direct-{task}",
        "execution_location": "remote",
        "deployments": [deployment],
    }
    revision = canonical_json_sha256(seed)
    route = {
        **seed,
        "route_revision": revision,
        "virtual_model": f"vkp-remote-direct-{task}-{revision[:12]}",
    }
    route["route_snapshot_sha256"] = _route_snapshot_sha256(route)
    return route


def _normalise_route_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("route_snapshot must be an object")
    route_id = str(value.get("route_id") or "").strip()
    if not route_id:
        raise ValueError("route_snapshot.route_id is required")
    execution_location = str(value.get("execution_location") or "").strip().lower()
    if execution_location != "remote":
        raise ValueError("data export consent route must use execution_location=remote")
    deployments_raw = value.get("deployments")
    if not isinstance(deployments_raw, list) or not deployments_raw:
        raise ValueError("route_snapshot.deployments must be a non-empty list")
    deployments = [_normalise_deployment(row) for row in deployments_raw]
    destinations = _normalise_route_destinations(value.get("destinations"))
    revision = str(value.get("route_revision") or "").strip()
    if not revision:
        revision_seed = {
            "route_id": route_id,
            "execution_location": execution_location,
            "deployments": deployments,
            **({"destinations": destinations} if destinations else {}),
        }
        revision = canonical_json_sha256(revision_seed)
    virtual_model = str(value.get("virtual_model") or "").strip()
    if not virtual_model:
        virtual_model = f"vkp-remote-route-{revision[:12]}"
    route = {
        "route_id": route_id,
        "route_revision": revision,
        "virtual_model": virtual_model,
        "execution_location": execution_location,
        "deployments": deployments,
        **({"destinations": destinations} if destinations else {}),
    }
    route["route_snapshot_sha256"] = _route_snapshot_sha256(route)
    return route


def _normalise_deployment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("route deployment must be an object")
    provider = str(value.get("provider") or "").strip()
    model = str(value.get("model") or "").strip()
    base_url = _normalise_url(str(value.get("base_url") or "").strip())
    interface = str(value.get("interface") or "openai_compatible").strip()
    if not provider or not model or not base_url or not interface:
        raise ValueError(
            "route deployment requires provider, model, base_url, and interface"
        )
    result: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "interface": interface,
    }
    auth_fields = {
        "auth_mode",
        "api_key_optional",
        "provider_options",
        "required_provider_options",
        "environment_bindings",
    }
    if auth_fields.intersection(value):
        auth_mode = str(value.get("auth_mode") or "api_key_dpapi").strip()
        if auth_mode not in {"api_key_dpapi", "external_environment"}:
            raise ValueError("route deployment auth_mode is invalid")
        result["auth_mode"] = auth_mode
        result["api_key_optional"] = bool(value.get("api_key_optional"))
        provider_options = _normalise_consent_provider_options(
            value.get("provider_options")
        )
        result["provider_options"] = provider_options
        result["required_provider_options"] = _normalise_required_provider_options(
            value.get("required_provider_options"), provider_options=provider_options
        )
        result["environment_bindings"] = _normalise_environment_bindings(
            value.get("environment_bindings")
        )
    if str(value.get("litellm_provider") or "").strip():
        result["litellm_provider"] = str(value["litellm_provider"]).strip()
    if str(value.get("id") or "").strip():
        result["id"] = str(value["id"]).strip()
    if str(value.get("adapter_backend") or "").strip():
        result["adapter_backend"] = str(value["adapter_backend"]).strip()
    if value.get("timeout_seconds") not in (None, ""):
        result["timeout_seconds"] = int(value["timeout_seconds"])
    return result


def _normalise_consent_provider_options(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("route deployment provider_options must be an object")
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not _PROVIDER_OPTION_KEY_RE.fullmatch(key):
            raise ValueError("route deployment provider option key is invalid")
        if _SECRET_PROVIDER_OPTION_KEY_RE.search(key):
            raise ValueError(
                "route deployment provider options must not contain secrets"
            )
        if not isinstance(raw_value, (str, int, bool)) or isinstance(raw_value, float):
            raise ValueError("route deployment provider option value is invalid")
        if isinstance(raw_value, str):
            clean = raw_value.strip()
            if len(clean) > 500 or any(character in clean for character in "\r\n\x00"):
                raise ValueError("route deployment provider option value is invalid")
            if clean.startswith("os.environ/"):
                raise ValueError(
                    "route deployment environment references are catalog-controlled"
                )
            result[key] = clean
        else:
            result[key] = raw_value
    return {key: result[key] for key in sorted(result)}


def _normalise_required_provider_options(
    value: Any,
    *,
    provider_options: dict[str, Any],
) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("route deployment required_provider_options must be a list")
    result: list[str] = []
    for raw_key in value:
        key = str(raw_key or "").strip()
        if not _PROVIDER_OPTION_KEY_RE.fullmatch(key):
            raise ValueError("route deployment required provider option is invalid")
        if _SECRET_PROVIDER_OPTION_KEY_RE.search(key):
            raise ValueError(
                "route deployment required provider options must not contain secrets"
            )
        if key in result:
            raise ValueError(
                "route deployment required provider options must be unique"
            )
        if provider_options.get(key) in (None, ""):
            raise ValueError(
                f"route deployment required provider option is missing: {key}"
            )
        result.append(key)
    return sorted(result)


def _normalise_environment_bindings(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("route deployment environment_bindings must be a list")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("route deployment environment binding must be an object")
        param = str(raw.get("param") or "").strip()
        env = str(raw.get("env") or "").strip()
        pair = (param, env)
        if not _PROVIDER_OPTION_KEY_RE.fullmatch(param) or not _ENV_NAME_RE.fullmatch(
            env
        ):
            raise ValueError("route deployment environment binding is invalid")
        if pair in seen:
            raise ValueError("route deployment environment bindings must be unique")
        seen.add(pair)
        result.append(
            {"param": param, "env": env, "required": bool(raw.get("required"))}
        )
    return sorted(result, key=lambda row: (str(row["param"]), str(row["env"])))


def _legacy_provider_identity(deployment: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(deployment.get(key) or "")
        for key in ("provider", "model", "base_url", "interface")
    }


def _route_snapshot_sha256(route: dict[str, Any]) -> str:
    canonical = {
        "route_id": str(route.get("route_id") or ""),
        "route_revision": str(route.get("route_revision") or ""),
        "virtual_model": str(route.get("virtual_model") or ""),
        "execution_location": str(route.get("execution_location") or ""),
        "deployments": [
            _normalise_deployment(row) for row in route.get("deployments") or []
        ],
    }
    destinations = _normalise_route_destinations(route.get("destinations"))
    if destinations:
        canonical["destinations"] = destinations
    return canonical_json_sha256(canonical)


def _normalise_route_destinations(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("route destinations must be a list")
    destinations: list[str] = []
    for raw in value:
        parsed = urllib.parse.urlsplit(str(raw or "").strip())
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ValueError("route destination must be an explicit HTTPS origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "route destination must not contain credentials, query, or fragment"
            )
        if parsed.path not in {"", "/"}:
            raise ValueError("route destination must be an origin without a path")
        host = parsed.hostname.lower()
        rendered_host = f"[{host}]" if ":" in host else host
        port = f":{parsed.port}" if parsed.port and parsed.port != 443 else ""
        origin = f"https://{rendered_host}{port}"
        if origin not in destinations:
            destinations.append(origin)
    return sorted(destinations)


def _route_destinations(route: dict[str, Any]) -> list[str]:
    explicit = _normalise_route_destinations(route.get("destinations"))
    if explicit:
        return explicit
    destinations: list[str] = []
    for row in route.get("deployments") or []:
        if not isinstance(row, dict):
            continue
        parsed = urllib.parse.urlsplit(str(row.get("base_url") or ""))
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            continue
        host = parsed.hostname.lower()
        rendered_host = f"[{host}]" if ":" in host else host
        port = f":{parsed.port}" if parsed.port and parsed.port != 443 else ""
        origin = f"https://{rendered_host}{port}"
        if origin not in destinations:
            destinations.append(origin)
    return sorted(destinations)


def _connector_task_spec(task: str) -> dict[str, Any]:
    if task in MEDIA_CAPABILITIES:
        capability = media_capability(task)
        return {
            "model_type": "media_service",
            "modality": "audio_or_video" if task == "video_asr" else "video",
            "migration_status": "candidate_only",
            "capability": capability,
        }
    return MODEL_TASKS[task]


def _task_key(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if key in MEDIA_CAPABILITIES:
        return key
    if key not in MODEL_TASKS or MODEL_TASKS[key]["migration_status"] == "deferred":
        raise ValueError(f"unsupported connector task: {value}")
    return key


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_asr_prompt(value: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > 4000:
        raise ValueError("asr_prompt must not exceed 4000 characters")
    return text


def _payload_sha256(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("consent_sha256", None)
    return canonical_json_sha256(value)


def _normalise_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )


def _status(path: Path, status: str, blockers: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "status": status,
        "valid": status == "active",
        "consent_path": str(path),
        "blockers": blockers,
        "next_action": "execute_consent"
        if status == "active"
        else "create_or_refresh_consent_in_visible_operator_shell",
        "operator_boundary": {
            "project_gate_only": True,
            "does_not_override_agent_platform_policy": True,
            "mcp_cannot_create_or_confirm_consent": True,
        },
    }


def _load_provider_config(value: str) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value).expanduser()
    data = (
        json.loads(path.read_text(encoding="utf-8-sig"))
        if path.is_file()
        else json.loads(value)
    )
    if not isinstance(data, dict):
        raise ValueError("provider config must be a JSON object")
    return data


def _load_output_contract(value: str) -> dict[str, Any] | None:
    if not str(value or "").strip():
        return None
    path = Path(value).expanduser()
    data = (
        json.loads(path.read_text(encoding="utf-8-sig"))
        if path.is_file()
        else json.loads(value)
    )
    if not isinstance(data, dict):
        raise ValueError("output contract must be a JSON object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and inspect VKP online-model export consent"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("root_dir")
    create.add_argument("--task", required=True)
    create.add_argument("--artifact", action="append", required=True)
    create.add_argument("--provider-config", default="")
    create.add_argument("--route-id", default="")
    create.add_argument("--route-revision", default="")
    create.add_argument("--settings-path", default="")
    create.add_argument("--instructions", default="")
    create.add_argument(
        "--asr-prompt",
        default="",
        help="ASR lexical context/hotwords only; task/audit instructions belong in --instructions",
    )
    create.add_argument("--output-contract", default="")
    create.add_argument("--purpose", default="approved online model task")
    create.add_argument("--expires-hours", type=float, default=24.0)
    create.add_argument("--max-calls", type=int, default=1)
    create.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        required=True,
    )
    create.add_argument("--max-cost-per-call-usd", type=float, default=None)
    create.add_argument("--max-retries-per-call", type=int, default=None)
    create.add_argument("--confirm-data-export", action="store_true")
    create.add_argument("--output-path", default="")
    status = sub.add_parser("status")
    status.add_argument("consent_path")
    status.add_argument("--provider-config", default="")
    status.add_argument("--route-id", default="")
    status.add_argument("--route-revision", default="")
    status.add_argument("--settings-path", default="")
    status.add_argument("--expected-task", default="")
    status.add_argument("--expected-calls", type=int, default=1)
    revoke = sub.add_parser("revoke")
    revoke.add_argument("consent_path")
    args = parser.parse_args()
    if args.command == "create":
        provider_config = _load_provider_config(args.provider_config)
        route_snapshot = None
        if args.route_id or args.route_revision:
            if provider_config:
                parser.error(
                    "--provider-config cannot be combined with route-based consent"
                )
            from .trusted_model_connector_policy import TrustedModelConnectorPolicy

            project_root = Path(__file__).resolve().parents[2]
            policy = TrustedModelConnectorPolicy.from_environment(
                default_root=project_root
            )
            policy.require_path(args.root_dir, label="root_dir", must_exist=False)
            for artifact in args.artifact:
                policy.require_path(artifact, label="artifact")
            route_snapshot = resolve_model_connector_consent_route(
                args.task,
                route_id=args.route_id,
                route_revision=args.route_revision,
                settings_path=args.settings_path or None,
                policy=policy,
            )
        result = create_model_connector_consent(
            args.root_dir,
            task=args.task,
            artifact_paths=args.artifact,
            provider_config=None if route_snapshot else provider_config,
            route_snapshot=route_snapshot,
            instructions=args.instructions,
            asr_prompt=args.asr_prompt,
            output_contract=_load_output_contract(args.output_contract),
            purpose=args.purpose,
            expires_hours=args.expires_hours,
            max_calls=args.max_calls,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            max_cost_per_call_usd=args.max_cost_per_call_usd,
            max_retries_per_call=args.max_retries_per_call,
            confirm_data_export=args.confirm_data_export,
            output_path=args.output_path or None,
        )
    elif args.command == "status":
        provider_config = _load_provider_config(args.provider_config)
        route_snapshot = None
        if args.route_id or args.route_revision:
            if provider_config:
                parser.error(
                    "--provider-config cannot be combined with route-based validation"
                )
            consent = read_json(Path(args.consent_path).expanduser().resolve())
            task = args.expected_task or str(consent.get("task") or "")
            from .trusted_model_connector_policy import TrustedModelConnectorPolicy

            project_root = Path(__file__).resolve().parents[2]
            policy = TrustedModelConnectorPolicy.from_environment(
                default_root=project_root
            )
            policy.require_consent_scope(args.consent_path, expected_task=task)
            route_snapshot = resolve_model_connector_consent_route(
                task,
                route_id=args.route_id,
                route_revision=args.route_revision,
                settings_path=args.settings_path or None,
                policy=policy,
            )
        result = validate_model_connector_consent(
            args.consent_path,
            provider_config=None if route_snapshot else (provider_config or None),
            route_snapshot=route_snapshot,
            expected_route_revision=args.route_revision,
            expected_task=args.expected_task,
            expected_calls=args.expected_calls,
        )
    else:
        result = revoke_model_connector_consent(args.consent_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") not in {"missing", "invalid", "blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
