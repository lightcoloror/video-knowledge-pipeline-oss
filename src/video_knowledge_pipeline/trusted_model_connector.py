from __future__ import annotations

import json
import math
import os
import urllib.parse
from pathlib import Path
from typing import Any
from uuid import uuid4

from .file_hash import sha256_file
from .asr_response_quality import assess_asr_response
from .media_capability_registry import MEDIA_CAPABILITIES
from .media_capability_registry import media_capability_registry_status
from .media_connector_consent import media_connector_consent_status, reserve_media_connector_attempt
from .media_route_settings import build_media_route_snapshot, media_route_settings_status
from .media_task_protocol import build_media_task_plan
from .mediakit_cli_adapter import execute_mediakit_cli_task, mediakit_cli_status
from .model_connector_consent import (
    _optional_retry_limit,
    record_model_connector_attempt,
    reserve_model_connector_attempt,
    validate_model_connector_consent,
)
from .model_api_settings import (
    resolve_model_api_provider_config,
    resolve_model_api_route,
)
from .model_gateway import model_gateway_runtime_readiness
from .model_provider_catalog import (
    provider_catalog_status,
    providers_for_capability,
)
from .model_output_contracts import validate_model_output
from .model_task_gateway import MODEL_TASKS, model_task_api_call
from .model_runtime_client import (
    SCHEMA as MODEL_RUNTIME_RESULT_SCHEMA,
    _authorise_consented_remote_runtime,
)
from .models import now_iso
from .multimodal_frame_analyzer import run_multimodal_frame_analysis
from .provider_config_safety import secretless_provider_config
from .storage import read_json, write_json
from .temporal_visual_analyzer import run_temporal_visual_analysis


SCHEMA = "video_knowledge_pipeline.trusted_model_connector.v1"


def _catalog_capability(model_type: str) -> str:
    value = str(model_type or "").strip().lower()
    if value in {"asr", "ocr"}:
        return value
    if value in {
        "document_visual",
        "semantic_frame",
        "temporal_sequence",
        "video_segment",
    }:
        return "vision"
    return "text"


def trusted_model_connector_capabilities() -> dict[str, Any]:
    rows = []
    for task, spec in MODEL_TASKS.items():
        if spec["migration_status"] == "deferred":
            continue
        catalog_capability = _catalog_capability(str(spec["model_type"]))
        rows.append(
            {
                "task": task,
                "model_type": spec["model_type"],
                "modality": spec["modality"],
                "providers": list(spec["providers"]),
                "execution": spec["execution"],
                "catalog_capability": catalog_capability,
                "provider_profile_ids": providers_for_capability(catalog_capability),
            }
        )
    return {
        "schema": SCHEMA,
        "status": "ready",
        "tasks": rows,
        "modalities": sorted({str(row["modality"]) for row in rows}),
        "provider_catalog": provider_catalog_status(),
        "media_capability_catalog": media_capability_registry_status(),
        "media_route_status": media_route_settings_status(),
        "consent_contract": {
            "exact_artifact_hashes": True,
            "explicit_per_file_upload_manifest": True,
            "operator_confirmation_bound_to_manifest": True,
            "atomic_call_and_cost_reservation": True,
            "task_and_provider_locked": True,
            "task_route_revision_and_deployments_locked": True,
            "all_candidate_destinations_allowlisted": True,
            "v1_single_deployment_compatibility": True,
            "v1_remote_execution_allowed": False,
            "call_cost_limit_and_expiry": True,
            "cross_process_call_reservation": True,
            "temporal_group_call_reservation": True,
            "one_business_confirmation": True,
            "hash_linked_derived_artifact_admission": True,
            "aggregate_parent_call_and_cost_limits": True,
            "stale_lock_recovery": True,
            "mcp_can_create_consent": False,
            "mcp_can_create_operator_confirmation": False,
            "mcp_can_create_parent_bound_child_consent": True,
            "secrets_from_environment_only": True,
        },
        "operator_boundary": _operator_boundary(),
    }


def trusted_model_connector_status(
    consent_path: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    expected_task: str = "",
    expected_route_revision: str = "",
    expected_calls: int = 1,
) -> dict[str, Any]:
    consent_file = Path(consent_path).expanduser().resolve()
    try:
        preview = read_json(consent_file)
        if isinstance(preview, dict) and str(preview.get("task") or "") in MEDIA_CAPABILITIES:
            return media_connector_consent_status(
                consent_file,
                expected_route_revision=expected_route_revision,
                expected_calls=expected_calls,
            )
        config, route = _resolve_execution_route(
            consent_file,
            provider_config=provider_config,
            expected_route_revision=expected_route_revision,
        )
    except (OSError, ValueError) as exc:
        return {
            "schema": SCHEMA,
            "status": "route_required",
            "valid": False,
            "consent_path": str(consent_file),
            "blockers": [{"key": "consent_route_unavailable", "message": str(exc)}],
        }
    return validate_model_connector_consent(
        consent_file,
        provider_config=config if provider_config is not None else None,
        route_snapshot=route,
        expected_route_revision=expected_route_revision,
        expected_task=expected_task,
        expected_calls=expected_calls,
    )


def execute_consented_model_task(
    consent_path: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    expected_route_revision: str = "",
    write: bool = True,
) -> dict[str, Any]:
    consent_file = Path(consent_path).expanduser().resolve()
    try:
        consent_preview = read_json(consent_file)
    except (OSError, ValueError):
        consent_preview = {}
    preview_task = (
        str(consent_preview.get("task") or "")
        if isinstance(consent_preview, dict)
        else ""
    )
    if preview_task in MEDIA_CAPABILITIES:
        return _execute_consented_media_task(
            consent_file,
            expected_route_revision=expected_route_revision,
            write=write,
        )
    try:
        config, route = _resolve_execution_route(
            consent_file,
            provider_config=provider_config,
            expected_route_revision=expected_route_revision,
        )
        consent_preview = read_json(consent_file)
        preview_artifacts = (
            consent_preview.get("artifacts")
            if isinstance(consent_preview.get("artifacts"), list)
            else []
        )
        retry_limit = _consent_max_retries(consent_preview)
        if route is not None:
            retry_policy = (
                route.get("retry_policy")
                if isinstance(route.get("retry_policy"), dict)
                else {}
            )
            route_retry_limit = _optional_retry_limit(
                retry_policy.get("max_retries")
            )
            route_retry_limit = 0 if route_retry_limit is None else route_retry_limit
            if retry_limit is not None and retry_limit > route_retry_limit:
                return {
                    "schema": SCHEMA,
                    "ok": False,
                    "status": "retry_policy_conflict",
                    "consent_path": str(consent_file),
                    "error": (
                        "consent retry ceiling exceeds the route-locked retry policy"
                    ),
                    "remote_requests_made": False,
                    "consent_reserved": False,
                    "operator_boundary": _operator_boundary(),
                }
        preview_task = str(consent_preview.get("task") or "")
        if preview_task == "online_ocr":
            requested_calls = len(preview_artifacts)
        elif preview_task == "temporal_visual_analysis":
            requested_calls = len(_temporal_artifact_groups(preview_artifacts))
        else:
            requested_calls = 1
        requested_calls = max(1, requested_calls)
        provider_attempt_cap = requested_calls * (1 + (retry_limit or 0))
    except (OSError, ValueError) as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "route_required",
            "consent_path": str(consent_file),
            "error": str(exc),
            "operator_boundary": _operator_boundary(),
        }
    if _route_uses_proxy(route):
        gateway_preflight = model_gateway_runtime_readiness()
        if not gateway_preflight.get("ready"):
            return {
                "schema": SCHEMA,
                "ok": False,
                "status": "gateway_unavailable",
                "consent_path": str(consent_file),
                "error": "VKP LiteLLM Proxy is not healthy; consent was not reserved.",
                "gateway_preflight": gateway_preflight,
                "remote_requests_made": False,
                "consent_reserved": False,
                "operator_boundary": _operator_boundary(),
            }
    reservation = reserve_model_connector_attempt(
        consent_file,
        provider_config=config if provider_config is not None else None,
        route_snapshot=route,
        expected_route_revision=expected_route_revision,
        expected_calls=provider_attempt_cap,
    )
    if not reservation.get("reserved"):
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "consent_busy"
            if reservation.get("status") == "consent_busy"
            else "consent_required",
            "consent": reservation,
            "operator_boundary": _operator_boundary(),
        }
    payload = reservation
    task = str(payload["task"])
    artifacts = (
        payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    )
    execution_id = f"model_connector_{uuid4().hex[:12]}"
    output_dir = consent_file.parent / "model-connector-runs" / execution_id
    consent_id = str(payload.get("consent_id") or "")
    runtime_config = {**config, "consent_id": consent_id}
    runtime_route = route if isinstance(route, dict) else payload.get("route")
    runtime_revision = (
        str(runtime_route.get("route_revision") or "")
        if isinstance(runtime_route, dict)
        else ""
    )
    with _authorise_consented_remote_runtime(
        consent_id=consent_id,
        route_revision=runtime_revision,
        max_calls=provider_attempt_cap,
    ):
        if task == "online_ocr":
            model_result, completed_calls = _execute_individual_ocr_calls(
                artifacts,
                provider_config=runtime_config,
                prompt=str(payload.get("instructions") or ""),
                max_retries=retry_limit,
            )
        elif task == "temporal_visual_analysis":
            model_result, completed_calls = _execute_temporal_group_calls(
                artifacts,
                provider_config=runtime_config,
                prompt=str(payload.get("instructions") or ""),
                route=runtime_route if isinstance(runtime_route, dict) else {},
                consent_id=consent_id,
                max_retries=retry_limit,
            )
        else:
            kwargs = _model_inputs(task, artifacts)
            try:
                model_result = model_task_api_call(
                    task,
                    provider_config=runtime_config,
                    prompt=_provider_prompt(task, payload),
                    execute=True,
                    max_retries=retry_limit,
                    write=False,
                    **kwargs,
                )
            except Exception as exc:  # Persist a recovery report for provider failures.
                model_result = {
                    "ok": False,
                    "status": "provider_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            model_result = _attach_asr_quality(task, payload, model_result)
            completed_calls = 1 if model_result.get("ok") else 0
    reported_cost = _model_result_estimated_cost(model_result)
    attempt = record_model_connector_attempt(
        consent_file,
        completed_calls=completed_calls,
        reserved_cost_usd=float(reservation.get("reserved_cost_usd") or 0),
        reported_cost_usd=reported_cost,
        cost_unreported_calls=0 if reported_cost is not None else provider_attempt_cap,
    )
    attempt_usage = (
        attempt.get("usage")
        if isinstance(attempt, dict) and isinstance(attempt.get("usage"), dict)
        else {}
    )
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    cost_limit_exceeded = bool(attempt_usage.get("cost_limit_exceeded"))
    model_transport_ok = bool(model_result.get("ok"))
    output_validation = validate_model_output(
        _model_result_content(model_result),
        payload.get("output_contract")
        if isinstance(payload.get("output_contract"), dict)
        else None,
        transport_ok=model_transport_ok,
    )
    task_output_validation = _task_specific_output_validation(
        task,
        _model_result_content(model_result),
        artifacts,
    )
    asr_output_validation = _asr_task_output_validation(task, model_result)
    if asr_output_validation:
        task_output_validation = asr_output_validation
    combined_contract_ok = bool(output_validation["contract_ok"]) and bool(
        task_output_validation.get("contract_ok", True)
    )
    combined_quality_gate_passed = bool(
        output_validation["quality_gate_passed"]
    ) and bool(task_output_validation.get("quality_gate_passed", True))
    if task_output_validation:
        output_validation = {
            **output_validation,
            "task_specific_validation": task_output_validation,
            "contract_ok": combined_contract_ok,
            "quality_gate_passed": combined_quality_gate_passed,
            "status": (
                "contract_failed"
                if not combined_contract_ok
                else (
                    "quality_gate_failed"
                    if not combined_quality_gate_passed
                    else output_validation["status"]
                )
            ),
        }
    execution_ok = model_transport_ok and not cost_limit_exceeded
    execution_status = (
        "cost_limit_exceeded_after_response"
        if cost_limit_exceeded
        else str(
            model_result.get("status")
            or ("completed" if model_result.get("ok") else "failed")
        )
    )
    result = {
        "schema": SCHEMA,
        "execution_id": execution_id,
        "ok": execution_ok,
        "status": execution_status,
        "task": task,
        "model_type": str(payload.get("model_type") or ""),
        "consent_id": str(payload.get("consent_id") or ""),
        "consent_path": str(consent_file),
        "route": route or payload.get("route") or {},
        "artifact_paths": [
            str(row.get("path") or "") for row in artifacts if isinstance(row, dict)
        ],
        "upload_manifest": payload.get("upload_manifest") or {},
        "model_result": model_result,
        "transport_ok": model_transport_ok,
        "contract_ok": combined_contract_ok,
        "quality_gate_passed": combined_quality_gate_passed,
        "production_qualified": (execution_ok and combined_quality_gate_passed),
        "output_validation": output_validation,
        "usage": attempt_usage,
        "cost_control": {
            "max_estimated_cost_usd": scope.get("max_estimated_cost_usd"),
            "max_cost_per_call_usd": scope.get("max_cost_per_call_usd"),
            "reserved_cost_usd": reservation.get("reserved_cost_usd"),
            "provider_attempt_cap": provider_attempt_cap,
            "reported_cost_usd": reported_cost,
            "reported_cost_known": reported_cost is not None,
            "cost_committed_usd": attempt_usage.get("cost_committed_usd"),
            "remaining_estimated_cost_usd": _remaining_cost(scope, attempt_usage),
            "cost_limit_exceeded": cost_limit_exceeded,
        },
        "operator_boundary": _operator_boundary(),
        "updated_at": now_iso(),
    }
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "connector-execution.json"
        write_json(report_path, result)
        result["artifacts"] = {"execution_report": str(report_path)}
    return result


def _execute_consented_media_task(
    consent_file: Path,
    *,
    expected_route_revision: str,
    write: bool,
) -> dict[str, Any]:
    """Execute one MediaKit task after the ordinary v2 consent reservation."""
    try:
        preview = read_json(consent_file)
        if not isinstance(preview, dict):
            raise ValueError("media consent payload must be an object")
        task = str(preview.get("task") or "")
        route_status = build_media_route_snapshot(task)
        route = preview.get("route") if isinstance(preview.get("route"), dict) else route_status["route"]
        cli = mediakit_cli_status()
        if not cli.get("available"):
            return {
                "schema": SCHEMA,
                "ok": False,
                "status": "mediakit_cli_unavailable",
                "consent_path": str(consent_file),
                "error": "The official MediaKit CLI is not installed or not on PATH; consent was not reserved.",
                "next_action": str(cli.get("install_command") or "npx @volcengine/mediakit-cli install -y"),
                "remote_requests_made": False,
                "consent_reserved": False,
                "operator_boundary": _operator_boundary(),
            }
        reservation = reserve_media_connector_attempt(
            consent_file,
            expected_route_revision=expected_route_revision,
            expected_calls=1,
            route_snapshot=route,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "route_required",
            "consent_path": str(consent_file),
            "error": str(exc),
            "operator_boundary": _operator_boundary(),
        }
    if not reservation.get("reserved"):
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "consent_busy" if reservation.get("status") == "consent_busy" else "consent_required",
            "consent": reservation,
            "operator_boundary": _operator_boundary(),
        }
    payload = reservation
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    execution_id = f"media_connector_{uuid4().hex[:12]}"
    output_dir = consent_file.parent / "model-connector-runs" / execution_id
    try:
        plan = build_media_task_plan(
            str(payload.get("task") or ""),
            execution_location="remote",
            route_id=str(route.get("route_id") or ""),
            route_revision=str(route.get("route_revision") or ""),
            artifact_paths=[str(row.get("path") or "") for row in artifacts if isinstance(row, dict)],
            artifact_hashes=[str(row.get("sha256") or "") for row in artifacts if isinstance(row, dict)],
            consent_id=str(payload.get("consent_id") or ""),
            allowed_roots=None,
        )
        settings = route_status.get("settings") if isinstance(route_status.get("settings"), dict) else {}
        plan["provider_options"] = {"timeout_seconds": int(settings.get("timeout_seconds") or 900)}
        media_result = execute_mediakit_cli_task(
            plan,
            api_key=str(os.environ.get("MEDIAKIT_API_KEY") or ""),
            command=str(cli.get("command") or ""),
        )
    except Exception as exc:
        media_result = {"ok": False, "status": "provider_exception", "error": {"code": type(exc).__name__}, "content": {}}
    transport_ok = bool(media_result.get("ok")) and str(media_result.get("status") or "") == "succeeded"
    completed_calls = 1 if transport_ok else 0
    attempt = record_model_connector_attempt(
        consent_file,
        completed_calls=completed_calls,
        reserved_cost_usd=float(reservation.get("reserved_cost_usd") or 0),
        reported_cost_usd=None,
        cost_unreported_calls=1,
    )
    usage = attempt.get("usage") if isinstance(attempt, dict) and isinstance(attempt.get("usage"), dict) else {}
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    cost_limit_exceeded = bool(usage.get("cost_limit_exceeded"))
    result = {
        "schema": SCHEMA,
        "execution_id": execution_id,
        "ok": transport_ok and not cost_limit_exceeded,
        "status": "cost_limit_exceeded_after_response" if cost_limit_exceeded else str(media_result.get("status") or "failed"),
        "task": str(payload.get("task") or ""),
        "model_type": "mediakit",
        "consent_id": str(payload.get("consent_id") or ""),
        "consent_path": str(consent_file),
        "route": route,
        "artifact_paths": [str(row.get("path") or "") for row in artifacts if isinstance(row, dict)],
        "upload_manifest": payload.get("upload_manifest") or {},
        "model_result": media_result,
        "transport_ok": transport_ok,
        "contract_ok": transport_ok,
        "quality_gate_passed": False,
        "production_qualified": False,
        "output_validation": {
            "status": "candidate_evidence_pending_validation",
            "contract_ok": transport_ok,
            "quality_gate_passed": False,
        },
        "usage": usage,
        "cost_control": {
            "max_estimated_cost_usd": scope.get("max_estimated_cost_usd"),
            "max_cost_per_call_usd": scope.get("max_cost_per_call_usd"),
            "reserved_cost_usd": reservation.get("reserved_cost_usd"),
            "reported_cost_usd": None,
            "reported_cost_known": False,
            "cost_committed_usd": usage.get("cost_committed_usd"),
            "remaining_estimated_cost_usd": _remaining_cost(scope, usage),
            "cost_limit_exceeded": cost_limit_exceeded,
        },
        "operator_boundary": _operator_boundary(),
        "updated_at": now_iso(),
    }
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "connector-execution.json"
        write_json(report_path, result)
        result["artifacts"] = {"execution_report": str(report_path)}
    return result

EXECUTION_RECEIPT_SCHEMA = "video_knowledge_pipeline.model_execution_receipt.v1"


def compact_model_execution_receipt(result: dict[str, Any]) -> dict[str, Any]:
    """Return a transport-small receipt while keeping full model output local."""
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    report_path = Path(str(artifacts.get("execution_report") or "")).expanduser()
    route = result.get("route") if isinstance(result.get("route"), dict) else {}
    model_result = result.get("model_result") if isinstance(result.get("model_result"), dict) else {}
    deployment = model_result.get("deployment") if isinstance(model_result.get("deployment"), dict) else {}
    return {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "ok": bool(result.get("ok")),
        "status": str(result.get("status") or ""),
        "execution_id": str(result.get("execution_id") or ""),
        "task": str(result.get("task") or ""),
        "consent_id": str(result.get("consent_id") or ""),
        "route_identity": {
            "route_id": str(route.get("route_id") or ""),
            "route_revision": str(route.get("route_revision") or ""),
            "virtual_model": str(route.get("virtual_model") or ""),
        },
        "deployment": {
            "id": str(deployment.get("id") or deployment.get("profile_id") or ""),
            "provider": str(deployment.get("provider") or model_result.get("provider") or ""),
            "model": str(deployment.get("model") or ""),
        },
        "transport_ok": bool(result.get("transport_ok")),
        "contract_ok": bool(result.get("contract_ok")),
        "quality_gate_passed": bool(result.get("quality_gate_passed")),
        "production_qualified": bool(result.get("production_qualified")),
        "usage": result.get("usage") if isinstance(result.get("usage"), dict) else {},
        "cost_control": result.get("cost_control") if isinstance(result.get("cost_control"), dict) else {},
        "network_accounting": _network_accounting_summary(model_result),
        "execution_report": str(report_path) if str(report_path) not in {"", "."} else "",
        "execution_report_bytes": report_path.stat().st_size if report_path.is_file() else 0,
        "full_result_json_bytes": len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")),
        "content_returned_over_mcp": False,
        "content_available_in_local_execution_report": report_path.is_file(),
        "updated_at": str(result.get("updated_at") or now_iso()),
    }


def _network_accounting_summary(value: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            accounting = node.get("network_accounting")
            if isinstance(accounting, dict):
                rows.append(accounting)
            for key, child in node.items():
                if key != "network_accounting":
                    collect(child)
        elif isinstance(node, list):
            for child in node:
                collect(child)

    collect(value)
    return {
        "scope": "vkp_to_loopback_gateway_payload",
        "call_count": len(rows),
        "gateway_request_bytes": sum(int(row.get("gateway_request_bytes") or 0) for row in rows),
        "gateway_response_bytes": sum(int(row.get("gateway_response_bytes") or 0) for row in rows),
        "source_artifact_bytes": sum(int(row.get("source_artifact_bytes") or 0) for row in rows),
        "provider_wire_bytes_exact": False,
    }

def _execute_individual_ocr_calls(
    artifacts: list[dict[str, Any]],
    *,
    provider_config: dict[str, Any],
    prompt: str,
    max_retries: int | None = None,
) -> tuple[dict[str, Any], int]:
    calls: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    completed = 0
    for position, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        try:
            call = model_task_api_call(
                "online_ocr",
                provider_config=provider_config,
                prompt=prompt,
                image_paths=[str(artifact.get("path") or "")],
                execute=True,
                max_retries=max_retries,
                write=False,
            )
        except Exception as exc:
            call = {
                "ok": False,
                "status": "provider_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        calls.append(call)
        if not call.get("ok"):
            continue
        completed += 1
        for page in _ocr_pages(call.get("content")):
            if not isinstance(page, dict):
                continue
            row = dict(page)
            row.setdefault("index", position)
            row["image_path"] = str(artifact.get("path") or "")
            row["source_artifact_sha256"] = str(artifact.get("sha256") or "")
            row["evidence_status"] = "candidate"
            pages.append(row)
    total = len(artifacts)
    ok = completed == total and total > 0
    status = (
        "completed" if ok else ("partial_ocr_failure" if completed else "ocr_failed")
    )
    return (
        {
            "ok": ok,
            "status": status,
            "content": {"pages": pages},
            "calls": calls,
            "call_count": total,
            "completed_call_count": completed,
            "failed_call_count": max(0, total - completed),
        },
        completed,
    )


def _temporal_artifact_groups(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = Path(str(artifact.get("path") or "")).expanduser().resolve()
        key = str(path.parent)
        group = grouped.setdefault(
            key,
            {
                "group_id": path.parent.name,
                "group_path": str(path.parent),
                "artifacts": [],
            },
        )
        group["artifacts"].append(artifact)
    rows = list(grouped.values())
    for row in rows:
        row["artifacts"] = sorted(
            row["artifacts"],
            key=lambda value: str(value.get("path") or "").lower(),
        )
    return sorted(rows, key=_temporal_group_sort_key)


def _temporal_group_sort_key(group: dict[str, Any]) -> tuple[int, int | str]:
    group_id = str(group.get("group_id") or "")
    return (0, int(group_id)) if group_id.isdigit() else (1, group_id.lower())


def _execute_temporal_group_calls(
    artifacts: list[dict[str, Any]],
    *,
    provider_config: dict[str, Any],
    prompt: str,
    route: dict[str, Any],
    consent_id: str,
    max_retries: int | None = None,
) -> tuple[dict[str, Any], int]:
    groups = _temporal_artifact_groups(artifacts)
    completed = 0
    contents: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    usage: dict[str, float] = {}
    latency_ms = 0.0
    numeric_cost = 0.0
    cost_known = True
    selected_deployment: Any = {}
    selected_provider: Any = ""
    for group in groups:
        group_artifacts = group["artifacts"]
        image_paths = [str(row.get("path") or "") for row in group_artifacts]
        try:
            call = model_task_api_call(
                "temporal_visual_analysis",
                provider_config=provider_config,
                prompt=prompt,
                image_paths=image_paths,
                execute=True,
                max_retries=max_retries,
                write=False,
            )
        except Exception as exc:
            call = {
                "ok": False,
                "status": "provider_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        runtime = _model_runtime_result(call)
        call_ok = bool(runtime.get("ok", call.get("ok")))
        completed += int(call_ok)
        call_content = runtime.get("content", call.get("content"))
        response = (
            runtime.get("response")
            if isinstance(runtime.get("response"), dict)
            else (
                call.get("response") if isinstance(call.get("response"), dict) else {}
            )
        )
        if call_content is None:
            call_content = response.get("content")
        group_evidence = {
            "group_id": group["group_id"],
            "group_path": group["group_path"],
            "frame_count": len(group_artifacts),
            "frames": [
                {
                    "path": str(row.get("path") or ""),
                    "sha256": str(row.get("sha256") or ""),
                    "bytes": int(row.get("bytes") or 0),
                }
                for row in group_artifacts
            ],
        }
        evidence.append(group_evidence)
        contents.append(
            {
                "group_id": group["group_id"],
                "status": str(
                    runtime.get("status")
                    or call.get("status")
                    or ("completed" if call_ok else "failed")
                ),
                "content": call_content,
                "evidence_status": "candidate",
            }
        )
        deployment = runtime.get("deployment") or call.get("deployment") or {}
        provider = runtime.get("provider") or call.get("provider") or ""
        if not selected_deployment and deployment:
            selected_deployment = deployment
        if not selected_provider and provider:
            selected_provider = provider
        call_latency = runtime.get("latency_ms", call.get("latency_ms"))
        if isinstance(call_latency, (int, float)):
            latency_ms += float(call_latency)
        for key, value in (
            (runtime.get("usage") or {}).items()
            if isinstance(runtime.get("usage"), dict)
            else []
        ):
            if isinstance(value, (int, float)):
                usage[str(key)] = usage.get(str(key), 0.0) + float(value)
        cost = runtime.get("estimated_cost", call.get("estimated_cost"))
        if isinstance(cost, (int, float)):
            numeric_cost += float(cost)
        else:
            cost_known = False
        calls.append(
            {
                "group_id": group["group_id"],
                "ok": call_ok,
                "status": str(runtime.get("status") or call.get("status") or ""),
                "latency_ms": call_latency,
                "deployment": deployment,
                "provider": provider,
                "usage": runtime.get("usage") or {},
                "estimated_cost": cost,
                "error": str(runtime.get("error") or call.get("error") or ""),
            }
        )
    total = len(groups)
    ok = total > 0 and completed == total
    status = (
        "completed"
        if ok
        else ("partial_temporal_failure" if completed else "temporal_failed")
    )
    route_deployments = (
        route.get("deployments") if isinstance(route.get("deployments"), list) else []
    )
    if not selected_deployment and route_deployments:
        selected_deployment = route_deployments[0]
    if not selected_provider and isinstance(selected_deployment, dict):
        selected_provider = str(selected_deployment.get("provider") or "")
    return (
        {
            "schema": MODEL_RUNTIME_RESULT_SCHEMA,
            "ok": ok,
            "status": status,
            "task": "temporal_sequence",
            "execution_location": str(route.get("execution_location") or ""),
            "route_id": str(route.get("route_id") or ""),
            "route_revision": str(route.get("route_revision") or ""),
            "deployment": selected_deployment,
            "provider": selected_provider,
            "latency_ms": latency_ms,
            "usage": usage,
            "estimated_cost": numeric_cost if cost_known else None,
            "content": {"groups": contents},
            "evidence": evidence,
            "consent_id": str(consent_id or ""),
            "call_count": total,
            "completed_call_count": completed,
            "failed_call_count": max(0, total - completed),
            "calls": calls,
            "failure_recovery": "not_needed" if ok else "retry_failed_temporal_groups",
            "remote_requests_made": str(route.get("execution_location") or "")
            == "remote"
            and total > 0,
        },
        completed,
    )


def _model_runtime_result(call: dict[str, Any]) -> dict[str, Any]:
    """Return the authoritative runtime layer from a gateway wrapper."""

    if call.get("schema") == MODEL_RUNTIME_RESULT_SCHEMA:
        return call
    nested = call.get("runtime_result")
    return nested if isinstance(nested, dict) else call


def _ocr_pages(content: Any) -> list[Any]:
    value = content
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"pages": [{"markdown": value}]} if value.strip() else {"pages": []}
    if isinstance(value, dict):
        pages = value.get("pages") or value.get("items") or []
        return pages if isinstance(pages, list) else []
    return value if isinstance(value, list) else []


def resolve_legacy_bundle_vision_route(
    mode: str,
    *,
    expected_route_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a singleton explicit legacy vision route without accepting caller URLs."""
    mode_key = str(mode or "").strip().lower()
    task = (
        "multimodal_frame_analysis"
        if mode_key == "semantic"
        else "temporal_visual_analysis"
    )
    if mode_key not in {"semantic", "temporal"}:
        raise ValueError("mode must be semantic or temporal")
    model_type = str(MODEL_TASKS[task]["model_type"])
    route = resolve_model_api_route(model_type, execution_location="remote")
    if not str(expected_route_revision or "").strip():
        raise ValueError(
            "route_revision is required for legacy bundle vision execution"
        )
    if str(route.get("route_revision") or "") != str(expected_route_revision):
        raise ValueError(
            "configured route revision differs from requested route revision"
        )
    deployments = (
        route.get("deployments") if isinstance(route.get("deployments"), list) else []
    )
    if len(deployments) != 1:
        raise ValueError(
            "legacy bundle vision requires a singleton remote route; use consent v2 for pools"
        )
    deployment = deployments[0]
    if str(deployment.get("adapter_backend") or "").strip().lower() != "legacy":
        raise ValueError(
            "legacy bundle vision requires adapter_backend=legacy; use consent v2 for Proxy routes"
        )
    config = resolve_model_api_provider_config(model_type, execution_location="remote")
    if not config:
        raise ValueError(f"remote provider config is unavailable for {task}")
    return _secretless_resolved_provider_config(config), route


def execute_consented_bundle_vision(
    bundle_dir: str | Path,
    *,
    mode: str,
    indexes: list[int],
    export_consent: str | Path,
    provider_config: dict[str, Any] | None = None,
    frame_count: int = 8,
    image_max_edge: int = 512,
    image_jpeg_quality: int = 55,
    retries: int = 2,
    retry_delay_seconds: float = 3.0,
) -> dict[str, Any]:
    config = secretless_provider_config(provider_config)
    selected = sorted({int(value) for value in indexes if int(value) > 0})
    if not selected:
        raise ValueError("at least one positive timeline index is required")
    confirmed = ",".join(str(value) for value in selected)
    common = {
        "bundle_dir": bundle_dir,
        "execute": True,
        "provider_config": config,
        "limit": len(selected),
        "indexes": selected,
        "confirm_vision_calls": len(selected),
        "confirm_vision_indexes": confirmed,
        "image_probe_max_edge": int(image_max_edge),
        "image_probe_jpeg_quality": int(image_jpeg_quality),
        "vision_retries": int(retries),
        "vision_retry_delay_seconds": float(retry_delay_seconds),
        "execution_actor": "trusted_model_connector",
        "export_consent": export_consent,
    }
    mode_key = str(mode or "").strip().lower()
    if mode_key == "semantic":
        return run_multimodal_frame_analysis(**common)
    if mode_key == "temporal":
        return run_temporal_visual_analysis(frame_count=int(frame_count), **common)
    raise ValueError("mode must be semantic or temporal")


def execute_local_model_task(
    task: str,
    artifact_paths: list[str | Path],
    *,
    route_id: str = "",
    instructions: str = "",
    write: bool = True,
) -> dict[str, Any]:
    task_key = str(task or "").strip().lower().replace("-", "_")
    if (
        task_key not in MODEL_TASKS
        or MODEL_TASKS[task_key]["migration_status"] == "deferred"
    ):
        raise ValueError(f"unsupported connector task: {task}")
    model_type = str(MODEL_TASKS[task_key]["model_type"])
    try:
        route = resolve_model_api_route(model_type, execution_location="local")
        if route_id and str(route["route_id"]) != str(route_id):
            raise ValueError("requested local route_id differs from configured route")
        if any(
            not _is_loopback_url(str(row.get("base_url") or ""))
            for row in route["deployments"]
        ):
            raise ValueError("local-only route contains a non-loopback deployment")
        config = resolve_model_api_provider_config(
            model_type, execution_location="local"
        )
        if not config:
            raise ValueError("local model provider config is unavailable")
    except ValueError as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "local_gateway_unavailable",
            "task": task_key,
            "error": str(exc),
            "remote_requests_made": False,
            "operator_boundary": {
                **_operator_boundary(),
                "user_consent_required": False,
                "data_export": False,
            },
        }
    paths = [Path(value).expanduser().resolve() for value in artifact_paths]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("all local model artifacts must exist")
    artifacts = [_artifact_audit_record(path) for path in paths]
    kwargs = _model_inputs(task_key, artifacts)
    execution_id = f"local_model_connector_{uuid4().hex[:12]}"
    if task_key == "temporal_visual_analysis":
        model_result, _completed_calls = _execute_temporal_group_calls(
            artifacts,
            provider_config=config,
            prompt=str(instructions or ""),
            route=route,
            consent_id="",
        )
    else:
        try:
            model_result = model_task_api_call(
                task_key,
                provider_config=config,
                prompt=str(instructions or ""),
                execute=True,
                write=False,
                **kwargs,
            )
        except Exception as exc:
            model_result = {
                "ok": False,
                "status": "provider_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
    result = {
        "schema": SCHEMA,
        "execution_id": execution_id,
        "ok": bool(model_result.get("ok")),
        "status": str(
            model_result.get("status")
            or ("completed" if model_result.get("ok") else "failed")
        ),
        "task": task_key,
        "model_type": model_type,
        "route": route,
        "artifacts": artifacts,
        "model_result": model_result,
        "remote_requests_made": False,
        "operator_boundary": {
            **_operator_boundary(),
            "user_consent_required": False,
            "data_export": False,
        },
        "updated_at": now_iso(),
    }
    if write:
        output_dir = paths[0].parent / "model-connector-local-runs" / execution_id
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "connector-execution.json"
        write_json(report_path, result)
        result["execution_report"] = str(report_path)
    return result


def _resolve_execution_route(
    consent_file: Path,
    *,
    provider_config: dict[str, Any] | None,
    expected_route_revision: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if provider_config is not None:
        return secretless_provider_config(provider_config), None
    payload = read_json(consent_file)
    if not isinstance(payload, dict):
        raise ValueError("consent payload must be an object")
    task = str(payload.get("task") or "")
    if task not in MODEL_TASKS:
        raise ValueError("consent task is unsupported")
    model_type = str(payload.get("model_type") or MODEL_TASKS[task]["model_type"])
    route = resolve_model_api_route(model_type, execution_location="remote")
    if expected_route_revision and str(route["route_revision"]) != str(
        expected_route_revision
    ):
        raise ValueError(
            "configured route revision differs from requested route revision"
        )
    config = resolve_model_api_provider_config(model_type, execution_location="remote")
    if not config:
        raise ValueError("remote model provider config is unavailable")
    return _secretless_resolved_provider_config(config), route


def _secretless_resolved_provider_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only locally resolved runtime credentials before policy validation."""

    redacted = dict(config)
    redacted.pop("api_key", None)
    redacted.pop("api_key_source", None)
    return secretless_provider_config(redacted)


def _artifact_audit_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _is_loopback_url(value: str) -> bool:
    parsed = urllib.parse.urlsplit(str(value or ""))
    return parsed.scheme in {"http", "https"} and str(
        parsed.hostname or ""
    ).lower() in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _route_uses_proxy(route: dict[str, Any] | None) -> bool:
    if not isinstance(route, dict):
        return False
    return any(
        str(row.get("adapter_backend") or "").strip().lower() == "proxy"
        for row in route.get("deployments") or []
        if isinstance(row, dict)
    )


def _consent_max_retries(payload: dict[str, Any]) -> int | None:
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    value = scope.get("max_retries_per_call")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("consent max_retries_per_call must be an integer")
    if value < 0 or value > 10:
        raise ValueError("consent max_retries_per_call must be between 0 and 10")
    return value


def _model_inputs(task: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    modality = str(MODEL_TASKS[task]["modality"])
    paths = [
        Path(str(row.get("path") or "")).expanduser().resolve()
        for row in artifacts
        if isinstance(row, dict)
    ]
    if task == "smart_summary_global_reduce":
        return _smart_summary_global_reduce_inputs(paths)
    if modality == "text":
        sections = []
        for path in paths:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            sections.append(f"\n\n## Source: {path.name}\n\n{text}")
        return {"input_text": "".join(sections).strip()}
    if modality in {"image", "multi_image"}:
        return {"image_paths": [str(path) for path in paths]}
    if modality == "audio":
        return {"audio_path": str(paths[0])}
    if modality == "video":
        if len(paths) != 1:
            raise ValueError("video connector tasks require exactly one artifact")
        return {"video_path": str(paths[0])}
    raise ValueError(f"unsupported connector modality: {modality}")


def _smart_summary_global_reduce_inputs(paths: list[Path]) -> dict[str, Any]:
    """Load the already prepared, exact-hash-bound Reduce request.

    Intent: preserve the mature-summary request contract through the Broker.
    Decision: forward its validated messages and generation controls directly.
    Reason: flattening this artifact into generic input_text drops JSON mode and
    token limits, which can yield reasoning-only or otherwise unusable output.
    Evidence: smart_summary_global_reduce_request.v1 is produced locally and is
    bound by both the parent business authorization and the child consent hash.
    Effective scope: only smart_summary_global_reduce; generic text tasks retain
    their existing artifact concatenation behavior.
    """
    if len(paths) != 1:
        raise ValueError(
            "smart summary global reduce requires exactly one request artifact"
        )
    payload = json.loads(paths[0].read_text(encoding="utf-8-sig"))
    if payload.get("schema") != (
        "video_knowledge_pipeline.smart_summary_global_reduce_request.v1"
    ):
        raise ValueError("invalid smart summary global reduce request schema")
    if payload.get("task") != "smart_summary_global_reduce":
        raise ValueError("invalid smart summary global reduce request task")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("smart summary global reduce messages must be a non-empty list")
    normalized_messages: list[dict[str, str]] = []
    for row in messages:
        if not isinstance(row, dict):
            raise ValueError("smart summary global reduce message must be an object")
        role = str(row.get("role") or "").strip()
        content = row.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("smart summary global reduce message role is invalid")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("smart summary global reduce message content is required")
        normalized_messages.append({"role": role, "content": content})

    parameters = payload.get("generation_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("smart summary global reduce generation_parameters are required")
    max_tokens = parameters.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("smart summary global reduce max_tokens must be an integer")
    if max_tokens < 1 or max_tokens > 20_000:
        raise ValueError("smart summary global reduce max_tokens is out of range")
    temperature = parameters.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("smart summary global reduce temperature must be numeric")
    if float(temperature) < 0.0 or float(temperature) > 2.0:
        raise ValueError("smart summary global reduce temperature is out of range")
    if parameters.get("response_format") != "json_object":
        raise ValueError(
            "smart summary global reduce response_format must be json_object"
        )
    return {
        "messages": normalized_messages,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def _model_result_content(model_result: dict[str, Any]) -> Any:
    runtime = _model_runtime_result(model_result)
    return runtime.get("content", model_result.get("content"))


def _provider_prompt(task: str, payload: dict[str, Any]) -> str:
    spec = MODEL_TASKS.get(task) or {}
    if str(spec.get("model_type") or "") == "asr":
        # Older v2 files did not distinguish task instructions from provider
        # lexical context. Fail closed: never forward their instructions.
        return str(payload.get("asr_prompt") or "")
    return str(payload.get("instructions") or "")


def _attach_asr_quality(
    task: str,
    payload: dict[str, Any],
    model_result: dict[str, Any],
) -> dict[str, Any]:
    spec = MODEL_TASKS.get(task) or {}
    if str(spec.get("model_type") or "") != "asr" or not model_result.get("ok"):
        return model_result
    runtime = _model_runtime_result(model_result)
    raw = runtime.get("raw_output")
    if not isinstance(raw, dict):
        raw = model_result.get("raw_response")
    quality = assess_asr_response(
        raw,
        task_instructions=str(payload.get("instructions") or ""),
        asr_prompt=str(payload.get("asr_prompt") or ""),
    )
    enriched = dict(model_result)
    enriched["asr_quality"] = quality
    if isinstance(runtime, dict) and runtime is not enriched:
        nested = dict(runtime)
        nested["asr_quality"] = quality
        enriched["runtime_result"] = nested
    if quality["status"] in {"review_required", "degraded", "failed"}:
        enriched["status"] = quality["status"]
    return enriched


def _asr_task_output_validation(
    task: str,
    model_result: dict[str, Any],
) -> dict[str, Any]:
    spec = MODEL_TASKS.get(task) or {}
    if str(spec.get("model_type") or "") != "asr":
        return {}
    quality = model_result.get("asr_quality")
    if not isinstance(quality, dict):
        return {
            "contract_ok": True,
            "quality_gate_passed": False,
            "status": "asr_quality_unavailable",
        }
    return {
        "contract_ok": True,
        "quality_gate_passed": bool(quality.get("quality_gate_passed")),
        "status": str(quality.get("status") or "asr_quality_unavailable"),
        "asr_quality": quality,
    }


def _task_specific_output_validation(
    task: str,
    content: Any,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    if task not in {"transcript_correction_pack", "transcript_semantic_correction"}:
        return {}
    for artifact in artifacts:
        path = Path(str(artifact.get("path") or ""))
        if path.suffix.lower() != ".json":
            continue
        try:
            pack = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            not isinstance(pack, dict)
            or pack.get("schema")
            != "video_knowledge_pipeline.transcript_semantic_correction_pack.v1"
        ):
            continue
        from .transcript_semantic_correction import (
            validate_transcript_semantic_model_output,
        )

        return validate_transcript_semantic_model_output(content, pack)
    return {}


def _model_result_estimated_cost(model_result: dict[str, Any]) -> float | None:
    value = model_result.get("estimated_cost")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        amount = float(value)
        return round(amount, 8) if amount >= 0 and math.isfinite(amount) else None
    calls = (
        model_result.get("calls") if isinstance(model_result.get("calls"), list) else []
    )
    if not calls:
        return None
    costs: list[float] = []
    for call in calls:
        cost = call.get("estimated_cost") if isinstance(call, dict) else None
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            return None
        amount = float(cost)
        if amount < 0 or not math.isfinite(amount):
            return None
        costs.append(amount)
    return round(sum(costs), 8)


def _remaining_cost(scope: dict[str, Any], usage: dict[str, Any]) -> float:
    limit = float(scope.get("max_estimated_cost_usd") or 0)
    committed = float(usage.get("cost_committed_usd") or 0)
    return round(max(0.0, limit - committed), 8)


def _operator_boundary() -> dict[str, Any]:
    return {
        "formal_user_installed_connector": True,
        "user_consent_required": True,
        "consent_created_outside_mcp": True,
        "secrets_from_environment_only": True,
        "does_not_override_agent_platform_policy": True,
        "no_indirect_policy_bypass": True,
        "no_implicit_bulk_export": True,
        "remote_execution_default": "deny",
        "automatic_publish_allowed": False,
        "unlisted_file_upload_allowed": False,
        "silent_local_cloud_fallback_allowed": False,
    }
