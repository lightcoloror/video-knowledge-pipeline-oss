from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file as _sha256
from .model_connector_consent import (
    _data_type,
    _normalise_route_snapshot,
    _route_destinations,
    _task_key,
    _optional_retry_limit,
    create_model_connector_consent,
    render_model_connector_consent_markdown,
    resolve_model_connector_consent_route,
)
from .model_connector_consent import (
    _payload_sha256 as _consent_payload_sha256,
)
from .storage import bundle_write_lock, read_json, write_json
from .time_utils import parse_utc_datetime_or_none as _parse_datetime, utc_now_iso_seconds

SCHEMA = "video_knowledge_pipeline.model_business_authorization.v1"
STATUS_SCHEMA = "video_knowledge_pipeline.model_business_authorization_status.v1"
CHILD_BINDING_SCHEMA = (
    "video_knowledge_pipeline.model_business_authorization_child_binding.v1"
)
DEFAULT_FILENAME = "model-business-authorization.json"
_SUMMARY_INPUT_PACK_SCHEMA = "video_knowledge_pipeline.smart_summary_input_pack.v1"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def create_model_business_authorization(
    root_dir: str | Path,
    *,
    bundle_dir: str | Path,
    bundle_dirs: list[str | Path] | None = None,
    source_paths: list[str | Path],
    stages: list[dict[str, Any]],
    purpose: str,
    max_calls: int,
    max_estimated_cost_usd: float,
    expires_hours: float = 24.0,
    confirm_data_export: bool = False,
    output_path: str | Path | None = None,
    policy: Any | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Create one operator confirmation envelope for a bounded video workflow.

    Future artifacts are not pre-approved by path patterns. They must be inside the
    bound Bundle and must form a hash-linked chain from an exact source anchor or a
    previously admitted artifact before a child consent v2 can be created.
    """

    root = Path(root_dir).expanduser().resolve()
    bundle = Path(bundle_dir).expanduser().resolve()
    bundle_records = _normalise_bundle_records(bundle, bundle_dirs, policy=policy)
    sources = [_artifact_record(Path(value).expanduser().resolve()) for value in source_paths]
    if not sources:
        raise ValueError("at least one exact source artifact is required")
    if policy is not None:
        for source in sources:
            policy.require_path(source["path"], label="business authorization source")
    normalised_stages = [_normalise_stage(row, policy=policy) for row in stages]
    if not normalised_stages:
        raise ValueError("at least one authorized stage is required")
    stage_ids = [str(row["id"]) for row in normalised_stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("business authorization stage ids must be unique")
    total_calls = _positive_int(max_calls, label="max_calls")
    total_cost = _positive_money(
        max_estimated_cost_usd, label="max_estimated_cost_usd"
    )
    if expires_hours <= 0 or not math.isfinite(float(expires_hours)):
        raise ValueError("expires_hours must be positive and finite")
    if sum(int(row["max_calls"]) for row in normalised_stages) < total_calls:
        raise ValueError("max_calls cannot exceed the sum of stage call limits")
    if sum(float(row["max_estimated_cost_usd"]) for row in normalised_stages) < total_cost:
        raise ValueError(
            "max_estimated_cost_usd cannot exceed the sum of stage cost limits"
        )
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else root / DEFAULT_FILENAME
    )
    confirmed = bool(confirm_data_export)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "authorization_id": str(uuid4()),
        "status": "active" if confirmed else "confirmation_required",
        "user_confirmed_data_export": confirmed,
        "purpose": str(purpose or "approved video model workflow").strip(),
        "bundle": bundle_records[0],
        "bundles": bundle_records,
        "sources": sources,
        "stages": normalised_stages,
        "scope": {
            "max_calls": total_calls,
            "max_estimated_cost_usd": total_cost,
            "max_artifacts": sum(int(row["max_artifacts"]) for row in normalised_stages),
            "max_total_bytes": sum(int(row["max_total_bytes"]) for row in normalised_stages),
            "automatic_retry_allowed": any(int(row.get("max_retries_per_call") or 0) > 0 for row in normalised_stages),
            "automatic_fallback_allowed": False,
            "automatic_publish_allowed": False,
        },
        "usage": {
            "child_consents_created": 0,
            "calls_authorized": 0,
            "cost_authorized_usd": 0.0,
            "artifacts_admitted": 0,
            "bytes_admitted": 0,
        },
        "admissions": [],
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(hours=float(expires_hours))).isoformat(),
        "operator_confirmation": {
            "confirmed": confirmed,
            "confirmed_at": created_at.isoformat() if confirmed else "",
            "confirmation_method": "visible_operator_shell" if confirmed else "",
            "confirmation_scope": "exact_sources_and_bounded_derived_artifact_rules",
        },
        "operator_boundary": {
            "one_business_confirmation": True,
            "derived_artifacts_require_hash_linked_lineage": True,
            "child_consent_v2_still_required": True,
            "new_destination_model_route_task_or_budget_requires_reauthorization": True,
            "unlisted_file_upload_allowed": False,
            "silent_local_cloud_fallback_allowed": False,
            "automatic_publish_allowed": False,
            "does_not_override_agent_platform_policy": True,
        },
        "authorization_path": str(path),
    }
    payload["authorization_sha256"] = _authorization_sha256(payload)
    payload["document_sha256"] = _document_sha256(payload)
    if write:
        write_json(path, payload)
        path.with_suffix(".md").write_text(
            render_model_business_authorization_markdown(payload), encoding="utf-8"
        )
    return payload


def validate_model_business_authorization(
    authorization_path: str | Path, *, policy: Any | None = None
) -> dict[str, Any]:
    path = Path(authorization_path).expanduser().resolve()
    blockers: list[dict[str, str]] = []
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _status(path, "invalid", [{"key": "authorization_unreadable", "message": str(exc)}])
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return _status(path, "invalid", [{"key": "authorization_schema_invalid", "message": "Unsupported business authorization schema"}])
    if str(payload.get("authorization_sha256") or "") != _authorization_sha256(payload):
        blockers.append({"key": "authorization_policy_integrity_failed", "message": "Immutable business authorization fields changed"})
    if str(payload.get("document_sha256") or "") != _document_sha256(payload):
        blockers.append({"key": "authorization_document_integrity_failed", "message": "Business authorization document changed outside the atomic writer"})
    if payload.get("status") != "active" or not payload.get("user_confirmed_data_export"):
        blockers.append({"key": "authorization_not_active", "message": "Business data export was not confirmed or was revoked"})
    expires_at = _parse_datetime(payload.get("expires_at"))
    if expires_at is None or expires_at <= datetime.now(timezone.utc):
        blockers.append({"key": "authorization_expired", "message": "Business authorization expired"})
    bundles = _bound_bundle_records(payload)
    for bundle_record in bundles:
        bundle = Path(str(bundle_record.get("path") or "")).expanduser().resolve()
        if not (bundle / "manifest.json").is_file() or not (bundle / "timeline.json").is_file():
            blockers.append({"key": "authorization_bundle_missing", "message": f"Bound Bundle is unavailable: {bundle}"})
    if policy is not None:
        try:
            policy.require_path(path, label="business authorization")
            for bundle_record in bundles:
                bundle = Path(str(bundle_record.get("path") or "")).expanduser().resolve()
                policy.require_path(bundle / "manifest.json", label="bundle manifest")
                policy.require_path(bundle / "timeline.json", label="bundle timeline")
            for stage in payload.get("stages") or []:
                for deployment in stage.get("route_snapshot", {}).get("deployments", []):
                    policy.require_destination_identity(deployment)
        except (OSError, ValueError) as exc:
            blockers.append({"key": "authorization_policy_blocked", "message": str(exc)})
    if not bundles:
        blockers.append({"key": "authorization_bundle_missing", "message": "No bound Bundle is recorded"})
    bundle = Path(str(bundles[0].get("path") or "")).expanduser().resolve() if bundles else Path()
    for source in payload.get("sources") or []:
        changed = _artifact_change(source)
        if changed:
            blockers.append({"key": "authorization_source_changed", "message": changed})
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    if int(usage.get("calls_authorized") or 0) > int(scope.get("max_calls") or 0):
        blockers.append({"key": "authorization_call_limit_exceeded", "message": "Authorized child calls exceed the business limit"})
    if float(usage.get("cost_authorized_usd") or 0) > float(scope.get("max_estimated_cost_usd") or 0) + 1e-8:
        blockers.append({"key": "authorization_cost_limit_exceeded", "message": "Authorized child cost exceeds the business limit"})
    result = _status(path, "active" if not blockers else "blocked", blockers)
    result.update(
        {
            "authorization_id": str(payload.get("authorization_id") or ""),
            "authorization_sha256": str(payload.get("authorization_sha256") or ""),
            "bundle_dir": str(bundle),
            "bundle_dirs": [str(row.get("path") or "") for row in bundles],
            "stages": [str(row.get("id") or "") for row in payload.get("stages") or [] if isinstance(row, dict)],
            "usage": usage,
            "scope": scope,
            "remaining_calls": max(0, int(scope.get("max_calls") or 0) - int(usage.get("calls_authorized") or 0)),
            "remaining_estimated_cost_usd": round(max(0.0, float(scope.get("max_estimated_cost_usd") or 0) - float(usage.get("cost_authorized_usd") or 0)), 8),
        }
    )
    return result


def find_reusable_model_business_authorization(
    candidate_paths: list[str | Path],
    *,
    bundle_dir: str | Path,
    source_paths: list[str | Path],
    task: str,
    route_id: str,
    route_revision: str,
    producer: str,
    required_calls: int = 1,
    policy: Any | None = None,
) -> dict[str, Any]:
    """Find exactly one active parent that already covers an exact workflow scope.

    This is intentionally a matcher, never an implicit parent grant: a changed
    source, destination, route revision, producer, or exhausted cap remains a
    reauthorization blocker rather than silently broadening export scope.
    """
    bundle = Path(bundle_dir).expanduser().resolve()
    sources = [_artifact_record(Path(value).expanduser().resolve()) for value in source_paths]
    if not sources:
        raise ValueError("at least one exact source artifact is required for reuse matching")
    matches: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for value in candidate_paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            continue
        status = validate_model_business_authorization(path, policy=policy)
        if not status.get("valid"):
            observations.append({"authorization_path": str(path), "reason": "inactive_or_invalid"})
            continue
        parent = read_json(path)
        bound = {str(row.get("path") or "") for row in _bound_bundle_records(parent)}
        if str(bundle) not in bound:
            observations.append({"authorization_path": str(path), "reason": "bundle_not_bound"})
            continue
        known_sources = {(str(row.get("path") or ""), str(row.get("sha256") or ""), int(row.get("bytes") or 0), str(row.get("data_type") or "")) for row in parent.get("sources") or [] if isinstance(row, dict)}
        if any((str(row["path"]), str(row["sha256"]), int(row["bytes"]), str(row["data_type"])) not in known_sources for row in sources):
            observations.append({"authorization_path": str(path), "reason": "exact_source_scope_changed"})
            continue
        stages = [row for row in parent.get("stages") or [] if isinstance(row, dict) and str(row.get("task") or "") == _task_key(task)]
        stage_matches = []
        for stage in stages:
            route = stage.get("route_snapshot") if isinstance(stage.get("route_snapshot"), dict) else {}
            if str(route.get("route_id") or "") != str(route_id) or str(route.get("route_revision") or "") != str(route_revision):
                continue
            if str(producer) not in {str(item) for item in stage.get("allowed_producers") or []}:
                continue
            used = [row for row in parent.get("admissions") or [] if isinstance(row, dict) and str(row.get("stage_id") or "") == str(stage.get("id") or "")]
            remaining_stage_calls = int(stage.get("max_calls") or 0) - sum(int(row.get("max_calls") or 0) for row in used)
            if remaining_stage_calls < int(required_calls):
                continue
            stage_matches.append(stage)
        if len(stage_matches) != 1:
            observations.append({"authorization_path": str(path), "reason": "route_producer_or_stage_scope_changed"})
            continue
        if int(status.get("remaining_calls") or 0) < int(required_calls):
            observations.append({"authorization_path": str(path), "reason": "call_cap_exhausted"})
            continue
        matches.append({"authorization_path": str(path), "authorization_id": str(parent.get("authorization_id") or ""), "stage_id": str(stage_matches[0].get("id") or ""), "remaining_calls": int(status.get("remaining_calls") or 0)})
    if len(matches) == 1:
        return {"ok": True, "status": "reusable", "user_confirmation_reused": True, "new_user_confirmation_required": False, "match": matches[0], "observations": observations}
    if len(matches) > 1:
        return {"ok": False, "status": "ambiguous", "new_user_confirmation_required": False, "matches": matches, "observations": observations}
    return {"ok": False, "status": "scope_expansion_required", "new_user_confirmation_required": True, "matches": [], "observations": observations}


def create_model_business_authorization_from_plan(
    plan_path: str | Path,
    *,
    confirm_data_export: bool = False,
    output_path: str | Path | None = None,
    policy: Any | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Resolve a secretless plan into one immutable business authorization."""

    path = Path(plan_path).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("business authorization plan must be a JSON object")
    if (
        payload.get("schema")
        != "video_knowledge_pipeline.model_business_authorization_plan.v1"
    ):
        raise ValueError("unsupported business authorization plan schema")
    root_dir = Path(
        str(payload.get("root_dir") or path.parent)
    ).expanduser().resolve()
    stages: list[dict[str, Any]] = []
    for raw in payload.get("stages") or []:
        if not isinstance(raw, dict):
            raise ValueError(
                "each business authorization plan stage must be an object"
            )
        stage = dict(raw)
        route = stage.get("route_snapshot")
        if not isinstance(route, dict):
            route = resolve_model_connector_consent_route(
                str(stage.get("task") or ""),
                route_id=str(stage.get("route_id") or ""),
                route_revision=str(stage.get("route_revision") or ""),
                settings_path=(
                    str(
                        stage.get("settings_path")
                        or payload.get("settings_path")
                        or ""
                    )
                    or None
                ),
                policy=policy,
            )
        stage["route_snapshot"] = route
        stage.pop("route_id", None)
        stage.pop("route_revision", None)
        stage.pop("settings_path", None)
        stages.append(stage)
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    result = create_model_business_authorization(
        root_dir,
        bundle_dir=str(payload.get("bundle_dir") or ""),
        bundle_dirs=[str(value) for value in payload.get("bundle_dirs") or []],
        source_paths=[str(value) for value in payload.get("source_paths") or []],
        stages=stages,
        purpose=str(
            payload.get("purpose") or "approved video model workflow"
        ),
        max_calls=int(scope.get("max_calls") or 0),
        max_estimated_cost_usd=float(
            scope.get("max_estimated_cost_usd") or 0
        ),
        expires_hours=float(payload.get("expires_hours") or 24.0),
        confirm_data_export=confirm_data_export,
        output_path=output_path,
        policy=policy,
        write=write,
    )
    result["source_plan_path"] = str(path)
    result["source_plan_sha256"] = _sha256(path)
    return result

def create_business_child_consent(
    authorization_path: str | Path,
    *,
    stage_id: str,
    artifact_paths: list[str | Path],
    producer: str,
    input_paths: list[str | Path],
    max_calls: int = 1,
    output_path: str | Path | None = None,
    policy: Any | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Admit exact derived artifacts and mint consent v2 without a new prompt."""

    parent_path = Path(authorization_path).expanduser().resolve()
    lock_name = f".{parent_path.name}.business.lock"
    with bundle_write_lock(
        parent_path.parent,
        operation="model_business_child_consent",
        timeout_seconds=10.0,
        lock_name=lock_name,
        stale_after_seconds=900.0,
    ):
        status = validate_model_business_authorization(parent_path, policy=policy)
        if not status.get("valid"):
            raise ValueError(_blocker_message(status))
        parent = read_json(parent_path)
        stage = _stage(parent, stage_id)
        clean_producer = str(producer or "").strip()
        if clean_producer not in stage["allowed_producers"]:
            raise ValueError("producer is not authorized for this business stage")
        artifacts = [_artifact_record(Path(value).expanduser().resolve()) for value in artifact_paths]
        if not artifacts:
            raise ValueError("at least one derived artifact is required")
        bound_bundles = [Path(str(row.get("path") or "")).expanduser().resolve() for row in _bound_bundle_records(parent)]
        artifact_bundles: set[Path] = set()
        for artifact in artifacts:
            artifact_path = Path(artifact["path"])
            matches = [bundle for bundle in bound_bundles if artifact_path == bundle or artifact_path.is_relative_to(bundle)]
            if len(matches) != 1:
                raise ValueError("derived artifacts must remain inside the bound Bundle")
            artifact_bundles.add(matches[0])
        if len(artifact_bundles) != 1:
            raise ValueError("one child consent cannot mix derived artifacts from multiple bound Bundles")
        artifact_bundle = next(iter(artifact_bundles))
        inputs = [_artifact_record(Path(value).expanduser().resolve()) for value in input_paths]
        if not inputs:
            raise ValueError("at least one exact lineage input is required")
        known_hashes = {
            str(row.get("sha256") or "") for row in parent.get("sources") or []
        }
        known_hashes.update(
            str(row.get("sha256") or "")
            for admission in parent.get("admissions") or []
            for row in admission.get("artifacts") or []
            if isinstance(admission, dict) and isinstance(row, dict)
        )
        if not any(str(row["sha256"]) in known_hashes for row in inputs):
            raise ValueError("lineage inputs are not linked to an exact source or prior admission")
        for row in inputs:
            if str(row["sha256"]) not in known_hashes:
                raise ValueError("every lineage input must be an exact source or prior admission")
        calls = _positive_int(max_calls, label="max_calls")
        child_cost = round(calls * float(stage["max_cost_per_call_usd"]), 8)
        identity = _canonical_sha256(
            {
                "authorization_id": parent["authorization_id"],
                "authorization_sha256": parent["authorization_sha256"],
                "stage_id": stage["id"],
                "producer": clean_producer,
                "artifacts": artifacts,
                "inputs": inputs,
                "calls": calls,
            }
        )
        for admission in parent.get("admissions") or []:
            if isinstance(admission, dict) and admission.get("identity_sha256") == identity:
                existing = Path(str(admission.get("child_consent_path") or ""))
                if existing.is_file():
                    child = read_json(existing)
                    return {
                        "ok": True,
                        "status": "existing_child_consent",
                        "authorization_path": str(parent_path),
                        "admission_id": str(admission.get("admission_id") or ""),
                        "consent_path": str(existing),
                        "consent_id": str(child.get("consent_id") or "") if isinstance(child, dict) else "",
                        "route_revision": str(stage["route_snapshot"]["route_revision"]),
                        "user_confirmation_reused": True,
                    }
        _require_capacity(
            parent,
            stage=stage,
            artifacts=artifacts,
            calls=calls,
            cost=child_cost,
        )
        admission_id = f"admission_{identity[:16]}"
        child_path = (
            Path(output_path).expanduser().resolve()
            if output_path
            else parent_path.parent / "business-child-consents" / f"{stage['id']}-{identity[:16]}.json"
        )
        if not (child_path == parent_path.parent or child_path.is_relative_to(parent_path.parent)):
            raise ValueError("child consent path must remain beside or below the business authorization")
        binding = {
            "schema": CHILD_BINDING_SCHEMA,
            "authorization_path": str(parent_path),
            "authorization_id": str(parent["authorization_id"]),
            "authorization_sha256": str(parent["authorization_sha256"]),
            "stage_id": str(stage["id"]),
            "admission_id": admission_id,
            "identity_sha256": identity,
            "bundle_path": str(artifact_bundle),
        }
        child = create_model_connector_consent(
            parent_path.parent,
            task=str(stage["task"]),
            artifact_paths=[row["path"] for row in artifacts],
            route_snapshot=stage["route_snapshot"],
            instructions=str(stage.get("instructions") or ""),
            asr_prompt=str(stage.get("asr_prompt") or ""),
            output_contract=stage.get("output_contract") if isinstance(stage.get("output_contract"), dict) else None,
            purpose=f"{parent['purpose']} / {stage['id']}",
            expires_hours=_remaining_hours(parent),
            max_calls=calls,
            max_estimated_cost_usd=child_cost,
            max_cost_per_call_usd=float(stage["max_cost_per_call_usd"]),
            max_retries_per_call=int(stage.get("max_retries_per_call") or 0),
            confirm_data_export=True,
            output_path=child_path,
            write=False,
        )
        child["operator_confirmation"] = {
            "confirmed": True,
            "confirmed_at": str(parent["operator_confirmation"]["confirmed_at"]),
            "confirmation_method": "parent_business_authorization",
            "exact_manifest_sha256": str(child["upload_manifest"]["manifest_sha256"]),
            "parent_authorization": binding,
        }
        child["operator_boundary"]["parent_business_authorization_required"] = True
        child["consent_sha256"] = _consent_payload_sha256(child)
        admission = {
            "schema": "video_knowledge_pipeline.model_business_artifact_admission.v1",
            "admission_id": admission_id,
            "identity_sha256": identity,
            "stage_id": str(stage["id"]),
            "task": str(stage["task"]),
            "producer": clean_producer,
            "bundle_path": str(artifact_bundle),
            "artifacts": artifacts,
            "lineage_inputs": inputs,
            "max_calls": calls,
            "max_estimated_cost_usd": child_cost,
            "child_consent_path": str(child_path),
            "child_consent_id": str(child["consent_id"]),
            "child_upload_manifest_sha256": str(child["upload_manifest"]["manifest_sha256"]),
            "admitted_at": utc_now_iso_seconds(),
        }
        if write:
            write_json(child_path, child)
            child_path.with_suffix(".md").write_text(
                render_model_connector_consent_markdown(child), encoding="utf-8"
            )
            parent.setdefault("admissions", []).append(admission)
            usage = parent.setdefault("usage", {})
            usage["child_consents_created"] = int(usage.get("child_consents_created") or 0) + 1
            usage["calls_authorized"] = int(usage.get("calls_authorized") or 0) + calls
            usage["cost_authorized_usd"] = round(float(usage.get("cost_authorized_usd") or 0) + child_cost, 8)
            usage["artifacts_admitted"] = int(usage.get("artifacts_admitted") or 0) + len(artifacts)
            usage["bytes_admitted"] = int(usage.get("bytes_admitted") or 0) + sum(int(row["bytes"]) for row in artifacts)
            parent["document_sha256"] = _document_sha256(parent)
            write_json(parent_path, parent)
            parent_path.with_suffix(".md").write_text(
                render_model_business_authorization_markdown(parent), encoding="utf-8"
            )
        return {
            "ok": True,
            "status": "child_consent_created" if write else "child_consent_preview",
            "authorization_path": str(parent_path),
            "authorization_id": str(parent["authorization_id"]),
            "admission_id": admission_id,
            "consent_path": str(child_path),
            "consent_id": str(child["consent_id"]),
            "route_revision": str(stage["route_snapshot"]["route_revision"]),
            "upload_manifest_sha256": str(child["upload_manifest"]["manifest_sha256"]),
            "user_confirmation_reused": True,
            "new_user_confirmation_required": False,
            "child_consent": child,
        }


def preflight_business_child_consents(
    authorization_path: str | Path,
    requests: list[dict[str, Any]],
    *,
    policy: Any | None = None,
) -> dict[str, Any]:
    """Validate a bounded child-consent batch without writing any artifact."""

    parent_path = Path(authorization_path).expanduser().resolve()
    status = validate_model_business_authorization(parent_path, policy=policy)
    if not status.get("valid"):
        raise ValueError(_blocker_message(status))
    if not requests:
        raise ValueError("at least one business child consent request is required")
    shadow = read_json(parent_path)
    known_identities = {
        str(row.get("identity_sha256") or "")
        for row in shadow.get("admissions") or []
        if isinstance(row, dict)
    }
    preview_rows: list[dict[str, Any]] = []
    new_identities: set[str] = set()

    for index, raw in enumerate(requests, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"business child request {index} must be an object")
        stage_id = str(raw.get("stage_id") or "").strip()
        artifact_paths = list(raw.get("artifact_paths") or [])
        producer = str(raw.get("producer") or "").strip()
        input_paths = list(raw.get("input_paths") or [])
        calls = _positive_int(raw.get("max_calls", 1), label="max_calls")
        preview = create_business_child_consent(
            parent_path,
            stage_id=stage_id,
            artifact_paths=artifact_paths,
            producer=producer,
            input_paths=input_paths,
            max_calls=calls,
            output_path=raw.get("output_path") or None,
            policy=policy,
            write=False,
        )
        row = {
            "position": index,
            "stage_id": stage_id,
            "status": str(preview.get("status") or ""),
            "consent_path": str(preview.get("consent_path") or ""),
            "consent_id": str(preview.get("consent_id") or ""),
            "route_revision": str(preview.get("route_revision") or ""),
            "new_admission_required": preview.get("status") != "existing_child_consent",
        }
        if preview.get("status") == "existing_child_consent":
            preview_rows.append(row)
            continue

        child = preview.get("child_consent")
        child = child if isinstance(child, dict) else {}
        confirmation = child.get("operator_confirmation")
        confirmation = confirmation if isinstance(confirmation, dict) else {}
        binding = confirmation.get("parent_authorization")
        binding = binding if isinstance(binding, dict) else {}
        identity = str(binding.get("identity_sha256") or "")
        if not identity:
            raise ValueError("business child preview did not return a bound identity")
        if identity in known_identities or identity in new_identities:
            raise ValueError("business child batch contains duplicate requests")
        stage = _stage(shadow, stage_id)
        artifacts = [
            dict(value)
            for value in child.get("artifacts") or []
            if isinstance(value, dict)
        ]
        child_cost = round(calls * float(stage["max_cost_per_call_usd"]), 8)
        _require_capacity(
            shadow,
            stage=stage,
            artifacts=artifacts,
            calls=calls,
            cost=child_cost,
        )
        shadow.setdefault("admissions", []).append(
            {
                "identity_sha256": identity,
                "stage_id": stage_id,
                "max_calls": calls,
                "max_estimated_cost_usd": child_cost,
                "artifacts": artifacts,
            }
        )
        usage = shadow.setdefault("usage", {})
        usage["calls_authorized"] = int(usage.get("calls_authorized") or 0) + calls
        usage["cost_authorized_usd"] = round(
            float(usage.get("cost_authorized_usd") or 0) + child_cost, 8
        )
        usage["artifacts_admitted"] = int(
            usage.get("artifacts_admitted") or 0
        ) + len(artifacts)
        usage["bytes_admitted"] = int(usage.get("bytes_admitted") or 0) + sum(
            int(value.get("bytes") or 0) for value in artifacts
        )
        new_identities.add(identity)
        row["identity_sha256"] = identity
        preview_rows.append(row)

    scope = shadow.get("scope") if isinstance(shadow.get("scope"), dict) else {}
    usage = shadow.get("usage") if isinstance(shadow.get("usage"), dict) else {}
    return {
        "schema": "video_knowledge_pipeline.model_business_child_batch_preflight.v1",
        "status": "ready",
        "ok": True,
        "authorization_path": str(parent_path),
        "authorization_id": str(shadow.get("authorization_id") or ""),
        "request_count": len(requests),
        "new_admission_count": len(new_identities),
        "existing_consent_count": sum(
            row["status"] == "existing_child_consent" for row in preview_rows
        ),
        "children": preview_rows,
        "projected_usage": usage,
        "projected_remaining_calls": max(
            0, int(scope.get("max_calls") or 0) - int(usage.get("calls_authorized") or 0)
        ),
        "write_performed": False,
        "provider_call_performed": False,
    }

def validate_parent_authorized_child_consent(
    child_payload: dict[str, Any], *, policy: Any | None = None
) -> dict[str, Any]:
    confirmation = child_payload.get("operator_confirmation") if isinstance(child_payload.get("operator_confirmation"), dict) else {}
    if str(confirmation.get("confirmation_method") or "") != "parent_business_authorization":
        return {"valid": False, "blockers": [{"key": "parent_binding_missing", "message": "Child consent is not bound to a parent business authorization"}]}
    binding = confirmation.get("parent_authorization") if isinstance(confirmation.get("parent_authorization"), dict) else {}
    path = Path(str(binding.get("authorization_path") or "")).expanduser().resolve()
    status = validate_model_business_authorization(path, policy=policy)
    blockers = list(status.get("blockers") or [])
    if blockers:
        return {"valid": False, "blockers": blockers}
    parent = read_json(path)
    if str(binding.get("authorization_id") or "") != str(parent.get("authorization_id") or "") or str(binding.get("authorization_sha256") or "") != str(parent.get("authorization_sha256") or ""):
        blockers.append({"key": "parent_binding_mismatch", "message": "Child consent parent identity does not match"})
    admission = next((row for row in parent.get("admissions") or [] if isinstance(row, dict) and row.get("admission_id") == binding.get("admission_id")), None)
    if admission is None:
        blockers.append({"key": "parent_admission_missing", "message": "Exact child artifact admission is missing"})
    else:
        stage = _stage(parent, str(binding.get("stage_id") or ""))
        checks = {
            "task": (str(child_payload.get("task") or ""), str(stage.get("task") or "")),
            "route_revision": (str((child_payload.get("route") or {}).get("route_revision") or ""), str(stage["route_snapshot"].get("route_revision") or "")),
            "upload_manifest": (str((child_payload.get("upload_manifest") or {}).get("manifest_sha256") or ""), str(admission.get("child_upload_manifest_sha256") or "")),
            "consent_id": (str(child_payload.get("consent_id") or ""), str(admission.get("child_consent_id") or "")),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                blockers.append({"key": f"parent_child_{label}_mismatch", "message": f"Child consent {label} differs from the admitted value"})
        if str(binding.get("identity_sha256") or "") != str(admission.get("identity_sha256") or ""):
            blockers.append({"key": "parent_admission_identity_mismatch", "message": "Child admission identity differs from the parent journal"})
    return {"valid": not blockers, "blockers": blockers, "authorization_path": str(path), "authorization_id": str(parent.get("authorization_id") or "") if isinstance(parent, dict) else ""}


def render_model_business_authorization_markdown(payload: dict[str, Any]) -> str:
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    lines = [
        "# VKP Model Business Authorization",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Authorization ID: `{payload.get('authorization_id', '')}`",
        f"- Purpose: {payload.get('purpose', '')}",
        f"- Bound Bundle: `{(payload.get('bundle') or {}).get('path', '')}`",
        f"- Calls: `{usage.get('calls_authorized', 0)}` authorized / `{scope.get('max_calls', 0)}` maximum",
        f"- Cost: $`{usage.get('cost_authorized_usd', 0)}` authorized / $`{scope.get('max_estimated_cost_usd', 0)}` maximum",
        f"- Derived artifacts: `{usage.get('artifacts_admitted', 0)}` admitted",
        f"- Expires: `{payload.get('expires_at', '')}`",
        "",
        "## Exact source anchors",
        "",
    ]
    for row in payload.get("sources") or []:
        lines.append(f"- `{row.get('path', '')}` ({row.get('bytes', 0)} bytes, SHA-256 `{row.get('sha256', '')}`)")
    lines.extend(["", "## Authorized stages", ""])
    for stage in payload.get("stages") or []:
        destinations = ", ".join(stage.get("authorized_destinations") or [])
        lines.append(f"- `{stage.get('id', '')}`: `{stage.get('task', '')}` via `{stage.get('route_snapshot', {}).get('route_id', '')}` revision `{stage.get('route_snapshot', {}).get('route_revision', '')}`; destinations `{destinations}`; calls `{stage.get('max_calls', 0)}`; cost $`{stage.get('max_estimated_cost_usd', 0)}`")
    lines.extend([
        "",
        "## Boundary",
        "",
        "This is one business-level confirmation, not a wildcard upload grant. Every derived file must stay inside the bound Bundle, be hash-linked to an exact source or prior admission, match an authorized producer/task/route, and receive an exact child consent v2 before Broker execution.",
        "A new destination, model, route revision, task, producer, source, retry policy, call cap, cost cap, or unlinked file requires a new business authorization.",
        "",
    ])
    return "\n".join(lines)


def _normalise_stage(value: Any, *, policy: Any | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("each business authorization stage must be an object")
    stage_id = str(value.get("id") or "").strip()
    if not _IDENTIFIER_RE.fullmatch(stage_id):
        raise ValueError("stage id must be a path-free identifier")
    task = _task_key(str(value.get("task") or ""))
    route = _normalise_route_snapshot(value.get("route_snapshot"))
    if policy is not None:
        for deployment in route["deployments"]:
            policy.require_destination_identity(deployment)
    producers = sorted({str(row or "").strip() for row in value.get("allowed_producers") or [] if str(row or "").strip()})
    if not producers or any(not _IDENTIFIER_RE.fullmatch(row) for row in producers):
        raise ValueError("allowed_producers must contain path-free identifiers")
    max_calls = _positive_int(value.get("max_calls"), label=f"{stage_id}.max_calls")
    max_cost = _positive_money(value.get("max_estimated_cost_usd"), label=f"{stage_id}.max_estimated_cost_usd")
    per_call = _positive_money(value.get("max_cost_per_call_usd"), label=f"{stage_id}.max_cost_per_call_usd")
    if per_call > max_cost:
        raise ValueError("stage max_cost_per_call_usd cannot exceed stage total cost")
    retries = _optional_retry_limit(value.get("max_retries_per_call"))
    retries = 0 if retries is None else retries
    return {
        "id": stage_id,
        "task": task,
        "route_snapshot": route,
        "authorized_destinations": _route_destinations(route),
        "allowed_producers": producers,
        "max_calls": max_calls,
        "max_estimated_cost_usd": max_cost,
        "max_cost_per_call_usd": per_call,
        "max_retries_per_call": retries,
        "max_artifacts": _positive_int(value.get("max_artifacts"), label=f"{stage_id}.max_artifacts"),
        "max_total_bytes": _positive_int(value.get("max_total_bytes"), label=f"{stage_id}.max_total_bytes"),
        "max_artifacts_per_child": _positive_int(value.get("max_artifacts_per_child", value.get("max_artifacts")), label=f"{stage_id}.max_artifacts_per_child"),
        "max_bytes_per_child": _positive_int(value.get("max_bytes_per_child", value.get("max_total_bytes")), label=f"{stage_id}.max_bytes_per_child"),
        "instructions": str(value.get("instructions") or "").strip(),
        "asr_prompt": str(value.get("asr_prompt") or "").strip(),
        "output_contract": value.get("output_contract") if isinstance(value.get("output_contract"), dict) else {},
    }


def _require_capacity(parent: dict[str, Any], *, stage: dict[str, Any], artifacts: list[dict[str, Any]], calls: int, cost: float) -> None:
    scope = parent["scope"]
    usage = parent["usage"]
    if int(usage.get("calls_authorized") or 0) + calls > int(scope["max_calls"]):
        raise ValueError("business authorization call limit exceeded")
    if float(usage.get("cost_authorized_usd") or 0) + cost > float(scope["max_estimated_cost_usd"]) + 1e-8:
        raise ValueError("business authorization cost limit exceeded")
    rows = [row for row in parent.get("admissions") or [] if isinstance(row, dict) and row.get("stage_id") == stage["id"]]
    stage_calls = sum(int(row.get("max_calls") or 0) for row in rows)
    stage_cost = sum(float(row.get("max_estimated_cost_usd") or 0) for row in rows)
    stage_artifacts = sum(len(row.get("artifacts") or []) for row in rows)
    stage_bytes = sum(sum(int(item.get("bytes") or 0) for item in row.get("artifacts") or [] if isinstance(item, dict)) for row in rows)
    artifact_bytes = sum(int(row["bytes"]) for row in artifacts)
    if stage_calls + calls > int(stage["max_calls"]):
        raise ValueError("business stage call limit exceeded")
    if stage_cost + cost > float(stage["max_estimated_cost_usd"]) + 1e-8:
        raise ValueError("business stage cost limit exceeded")
    if len(artifacts) > int(stage["max_artifacts_per_child"]) or artifact_bytes > int(stage["max_bytes_per_child"]):
        raise ValueError("child artifact count or bytes exceeds the stage per-child limit")
    if stage_artifacts + len(artifacts) > int(stage["max_artifacts"]) or stage_bytes + artifact_bytes > int(stage["max_total_bytes"]):
        raise ValueError("business stage artifact count or byte limit exceeded")


def _stage(parent: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for row in parent.get("stages") or []:
        if isinstance(row, dict) and str(row.get("id") or "") == str(stage_id or ""):
            return row
    raise ValueError(f"business authorization stage does not exist: {stage_id}")


def _normalise_bundle_records(bundle: Path, extra_bundles: list[str | Path] | None, *, policy: Any | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for value in [bundle, *(extra_bundles or [])]:
        candidate = Path(value).expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not (candidate / "manifest.json").is_file() or not (candidate / "timeline.json").is_file():
            raise ValueError("each bundle_dir must contain manifest.json and timeline.json")
        if policy is not None:
            policy.require_path(candidate / "manifest.json", label="bundle manifest")
            policy.require_path(candidate / "timeline.json", label="bundle timeline")
        rows.append({
            "path": str(candidate),
            "manifest_sha256_at_confirmation": _sha256(candidate / "manifest.json"),
            "timeline_sha256_at_confirmation": _sha256(candidate / "timeline.json"),
        })
    return rows


def _bound_bundle_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("bundles")
    rows = [dict(row) for row in raw if isinstance(row, dict) and str(row.get("path") or "").strip()] if isinstance(raw, list) else []
    if rows:
        return rows
    legacy = payload.get("bundle")
    return [dict(legacy)] if isinstance(legacy, dict) and str(legacy.get("path") or "").strip() else []


def _artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact not found: {path}")
    record = {
        "path": str(path),
        "data_type": _data_type(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    lineage = _summary_input_pack_lineage(path)
    if lineage is not None:
        record.update(
            {
                "validation_mode": "derived_summary_input_lineage.v1",
                "lineage": lineage,
            }
        )
    return record


def _artifact_change(row: Any) -> str:
    if not isinstance(row, dict):
        return "Source record is invalid"
    path = Path(str(row.get("path") or "")).expanduser().resolve()
    if not path.is_file():
        return f"Source artifact is missing: {path}"
    if row.get("validation_mode") == "derived_summary_input_lineage.v1":
        return _summary_input_pack_change(row, path)
    if int(row.get("bytes") or -1) != path.stat().st_size or str(row.get("sha256") or "") != _sha256(path):
        return f"Source artifact changed after business authorization: {path}"
    return ""


def _summary_input_pack_lineage(path: Path) -> dict[str, str] | None:
    """Return the stable, operator-bound lineage for a generated summary pack.

    The pack is deliberately rebuilt during summary preflight.  Its timestamp,
    quality diagnostics and workflow fields are therefore not parent-authorization
    inputs.  The canonical transcript and companion courseware stay separate exact
    sources; this record only permits rebuilds which retain their declared lineage.
    """
    if path.name != "smart-summary-input-pack.json":
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _SUMMARY_INPUT_PACK_SCHEMA:
        return None
    transcript_path = str(payload.get("transcript_source") or "").strip()
    transcript_sha256 = str(payload.get("transcript_source_sha256") or "").strip()
    if not transcript_path or not transcript_sha256:
        return None
    companion = payload.get("companion_courseware")
    companion_path = (
        str(companion.get("bundle_copy_path") or "").strip()
        if isinstance(companion, dict)
        else ""
    )
    return {
        "transcript_source_path": transcript_path,
        "transcript_source_sha256": transcript_sha256,
        "companion_courseware_path": companion_path,
    }


def _summary_input_pack_change(row: dict[str, Any], path: Path) -> str:
    expected = row.get("lineage")
    if not isinstance(expected, dict):
        return f"Derived summary input lineage is invalid: {path}"
    current = _summary_input_pack_lineage(path)
    if current is None:
        return f"Derived summary input lineage is invalid: {path}"
    for key in (
        "transcript_source_path",
        "transcript_source_sha256",
        "companion_courseware_path",
    ):
        if str(current.get(key) or "") != str(expected.get(key) or ""):
            return f"Derived summary input lineage changed after business authorization: {path}"
    return ""


def _authorization_sha256(payload: dict[str, Any]) -> str:
    immutable = {key: payload.get(key) for key in ("schema", "authorization_id", "purpose", "bundle", "sources", "stages", "scope", "created_at", "expires_at", "operator_confirmation", "operator_boundary", "authorization_path")}
    if "bundles" in payload:
        immutable["bundles"] = payload.get("bundles")

    return _canonical_sha256(immutable)


def _document_sha256(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("document_sha256", None)
    return _canonical_sha256(value)


def _canonical_sha256(value: Any) -> str:
    return canonical_json_sha256(value)




def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _positive_money(value: Any, *, label: str) -> float:
    try:
        result = round(float(value), 8)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite amount") from exc
    if result <= 0 or not math.isfinite(result):
        raise ValueError(f"{label} must be a positive finite amount")
    return result


def _remaining_hours(parent: dict[str, Any]) -> float:
    expires = _parse_datetime(parent.get("expires_at"))
    if expires is None:
        raise ValueError("business authorization expiry is invalid")
    return max(1.0 / 3600.0, (expires - datetime.now(timezone.utc)).total_seconds() / 3600.0)


def _blocker_message(status: dict[str, Any]) -> str:
    return "; ".join(str(row.get("message") or row.get("key") or "blocked") for row in status.get("blockers") or []) or "business authorization is blocked"


def _status(path: Path, status: str, blockers: list[dict[str, str]]) -> dict[str, Any]:
    return {"schema": STATUS_SCHEMA, "status": status, "valid": status == "active", "authorization_path": str(path), "blockers": blockers, "next_action": "create_child_consent_without_new_confirmation" if status == "active" else "create_or_refresh_one_business_authorization"}
