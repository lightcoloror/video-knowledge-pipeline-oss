from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .canonical_json import canonical_json_sha256

from .asr_vad_activity_audit import SCHEMA as ACTIVITY_AUDIT_SCHEMA
from .asr_vad_chunking import SCHEMA as CHUNK_MANIFEST_SCHEMA
from .model_connector_consent import (
    SCHEMA_V2 as CONSENT_SCHEMA_V2,
    validate_model_connector_consent,
)
from .model_business_authorization import (
    create_business_child_consent,
    preflight_business_child_consents,
    validate_model_business_authorization,
)
from .models import now_iso
from .storage import read_json, write_json
from .file_hash import sha256_file as _file_sha256


SCHEMA = "video_knowledge_pipeline.asr_chunk_batch_workflow.v1"
BUSINESS_PREPARATION_SCHEMA = (
    "video_knowledge_pipeline.asr_chunk_business_workflow_preparation.v1"
)
SUBMISSION_TOOL = "submit_consented_model_workflow_tool"


def build_asr_chunk_business_workflow(
    chunk_manifest: str | Path,
    authorization_path: str | Path,
    *,
    stage_id: str,
    producer: str,
    lineage_input_paths: Sequence[str | Path],
    output_path: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    activity_audit_path: str | Path | None = None,
    max_parallel_global: int = 4,
    max_parallel_per_destination: int = 2,
    policy: Any | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Reuse one confirmed business authorization for every exact ASR chunk.

    This is orchestration only: child consent construction remains owned by
    model_business_authorization and workflow compilation remains owned by
    build_asr_chunk_batch_workflow. No provider request is made here.
    """

    manifest_path = Path(chunk_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ASR chunk manifest not found: {manifest_path}")
    manifest = _object(read_json(manifest_path), "ASR chunk manifest")
    _validate_manifest_state(manifest)
    chunks = _completed_chunks(manifest)
    if len(chunks) > 64:
        raise ValueError("ASR chunk workflow exceeds the existing 64-node batch limit")
    _validate_parallelism(max_parallel_global, max_parallel_per_destination)

    parent_path = Path(authorization_path).expanduser().resolve()
    parent_status = validate_model_business_authorization(parent_path, policy=policy)
    if not parent_status.get("valid"):
        blocker_keys = [
            str(row.get("key") or "blocked")
            for row in parent_status.get("blockers") or []
            if isinstance(row, dict)
        ]
        raise ValueError(
            "business authorization is not active: "
            + (",".join(blocker_keys) or "unknown")
        )
    authorized_bundle = Path(str(parent_status.get("bundle_dir") or "")).resolve()
    requested_bundle = _bundle_dir(bundle_dir) if bundle_dir else authorized_bundle
    if requested_bundle != authorized_bundle:
        raise ValueError("bundle_dir does not match the business authorization")

    lineage_paths = [Path(value).expanduser().resolve() for value in lineage_input_paths]
    if not lineage_paths:
        raise ValueError("at least one exact lineage input is required")
    chunk_artifacts = [_validate_chunk_artifact(chunk) for chunk in chunks]
    child_requests = [
        {
            "stage_id": stage_id,
            "artifact_paths": [artifact_path],
            "producer": producer,
            "input_paths": lineage_paths,
            "max_calls": 1,
        }
        for artifact_path in chunk_artifacts
    ]
    batch_preflight = preflight_business_child_consents(
        parent_path, child_requests, policy=policy
    )
    child_results: list[dict[str, Any]] = []
    if write:
        prepared_children: Sequence[dict[str, Any]] = [
            create_business_child_consent(
                parent_path,
                stage_id=stage_id,
                artifact_paths=[artifact_path],
                producer=producer,
                input_paths=lineage_paths,
                max_calls=1,
                policy=policy,
                write=True,
            )
            for artifact_path in chunk_artifacts
        ]
    else:
        prepared_children = [
            dict(value)
            for value in batch_preflight.get("children") or []
            if isinstance(value, dict)
        ]
    for chunk, artifact_path, child in zip(
        chunks, chunk_artifacts, prepared_children, strict=True
    ):
        child_results.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "chunk_position": int(chunk["position"]),
                "artifact_path": str(artifact_path),
                "artifact_sha256": str(chunk["output_sha256"]),
                "status": str(child.get("status") or ""),
                "consent_path": str(child.get("consent_path") or ""),
                "consent_id": str(child.get("consent_id") or ""),
                "route_revision": str(child.get("route_revision") or ""),
            }
        )

    preparation = {
        "schema": BUSINESS_PREPARATION_SCHEMA,
        "status": "ready" if write else "preview",
        "ok": True,
        "write": bool(write),
        "authorization_path": str(parent_path),
        "authorization_id": str(parent_status.get("authorization_id") or ""),
        "stage_id": str(stage_id),
        "producer": str(producer),
        "chunk_manifest": str(manifest_path),
        "chunk_manifest_sha256": _file_sha256(manifest_path),
        "chunk_count": len(chunks),
        "child_consents": child_results,
        "new_user_confirmation_required": False,
        "provider_call_performed": False,
        "batch_preflight": batch_preflight,
        "workflow": {},
    }
    if not write:
        return preparation

    workflow = build_asr_chunk_batch_workflow(
        manifest_path,
        [row["consent_path"] for row in child_results],
        output_path=output_path,
        bundle_dir=authorized_bundle,
        activity_audit_path=activity_audit_path,
        max_parallel_global=max_parallel_global,
        max_parallel_per_destination=max_parallel_per_destination,
        write=True,
    )
    preparation["workflow"] = workflow
    preparation["workflow_path"] = str(workflow["output_path"])
    preparation["workflow_sha256"] = str(workflow["workflow_sha256"])
    return preparation


def build_asr_chunk_batch_workflow(
    chunk_manifest: str | Path,
    consent_paths: Sequence[str | Path],
    *,
    output_path: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    activity_audit_path: str | Path | None = None,
    max_parallel_global: int = 4,
    max_parallel_per_destination: int = 2,
    write: bool = True,
) -> dict[str, Any]:
    """Compile exact ASR chunk consents into the existing Broker workflow contract."""

    manifest_path = Path(chunk_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ASR chunk manifest not found: {manifest_path}")
    manifest = _object(read_json(manifest_path), "ASR chunk manifest")
    _validate_manifest_state(manifest)
    activity_audit = _validated_activity_audit(
        activity_audit_path, manifest=manifest
    )
    chunks = _completed_chunks(manifest)
    if len(chunks) > 64:
        raise ValueError("ASR chunk workflow exceeds the existing 64-node batch limit")
    paths = [Path(value).expanduser().resolve() for value in consent_paths]
    if len(paths) != len(chunks):
        raise ValueError("consent_paths must contain exactly one consent per ASR chunk")
    if len({str(path).casefold() for path in paths}) != len(paths):
        raise ValueError("consent_paths must not contain duplicates")
    _validate_parallelism(max_parallel_global, max_parallel_per_destination)

    bundle = _bundle_dir(bundle_dir)
    nodes: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for chunk, consent_path in zip(chunks, paths, strict=True):
        artifact_path = _validate_chunk_artifact(chunk)
        consent_status = validate_model_connector_consent(
            consent_path,
            expected_task="cloud_asr",
            expected_calls=1,
        )
        if not consent_status.get("valid"):
            blocker_keys = [
                str(row.get("key") or "blocked")
                for row in consent_status.get("blockers") or []
                if isinstance(row, dict)
            ]
            raise ValueError(
                f"ASR chunk consent is not executable: {consent_path}; "
                f"blockers={','.join(blocker_keys) or 'unknown'}"
            )
        if consent_status.get("consent_schema") != CONSENT_SCHEMA_V2:
            raise ValueError("ASR chunk workflow requires consent v2")
        _require_zero_retry(consent_status, consent_path)
        artifact = _single_consent_artifact(consent_status, consent_path)
        if str(Path(str(artifact.get("path") or "")).expanduser().resolve()).casefold() != str(
            artifact_path
        ).casefold():
            raise ValueError(
                f"consent artifact does not match ASR chunk order: {consent_path}"
            )
        if int(artifact.get("bytes") or -1) != int(chunk["output_bytes"]):
            raise ValueError(f"consent byte count does not match ASR chunk: {consent_path}")
        if str(artifact.get("sha256") or "") != str(chunk["output_sha256"]):
            raise ValueError(f"consent SHA-256 does not match ASR chunk: {consent_path}")
        destinations = [
            str(value or "").strip()
            for value in consent_status.get("authorized_destinations") or []
            if str(value or "").strip()
        ]
        if len(set(destinations)) != 1:
            raise ValueError("each ASR chunk consent must lock exactly one destination")
        route = (
            consent_status.get("route")
            if isinstance(consent_status.get("route"), dict)
            else {}
        )
        route_revision = str(route.get("route_revision") or "").strip()
        if not route_revision:
            raise ValueError("ASR chunk consent route_revision is missing")
        node_id = str(chunk["chunk_id"])
        nodes.append(
            {
                "id": node_id,
                "consent_path": str(consent_path),
                "depends_on": [],
            }
        )
        evidence.append(
            {
                "node_id": node_id,
                "chunk_id": node_id,
                "chunk_position": int(chunk["position"]),
                "artifact_path": str(artifact_path),
                "artifact_bytes": int(chunk["output_bytes"]),
                "artifact_sha256": str(chunk["output_sha256"]),
                "core_start": float(chunk["core_start"]),
                "core_end": float(chunk["core_end"]),
                "artifact_start": float(chunk["artifact_start"]),
                "artifact_end": float(chunk["artifact_end"]),
                "consent_path": str(consent_path),
                "consent_sha256": _file_sha256(consent_path),
                "consent_id": str(consent_status.get("consent_id") or ""),
                "route_revision": route_revision,
                "destination": destinations[0],
                "remaining_calls": int(consent_status.get("remaining_calls") or 0),
            }
        )

    submission_nodes = [
        {
            "id": row["id"],
            "consent_path": row["consent_path"],
            "depends_on": [],
        }
        for row in nodes
    ]
    submission_arguments = {
        "bundle_dir": str(bundle or ""),
        "nodes": submission_nodes,
        "write": True,
        "max_parallel_global": int(max_parallel_global),
        "max_parallel_per_destination": int(max_parallel_per_destination),
    }
    identity = {
        "chunk_manifest_sha256": _file_sha256(manifest_path),
        "bundle_dir": str(bundle or ""),
        "nodes": [
            {
                "id": row["node_id"],
                "consent_sha256": row["consent_sha256"],
                "artifact_sha256": row["artifact_sha256"],
                "route_revision": row["route_revision"],
                "destination": row["destination"],
            }
            for row in evidence
        ],
        "max_parallel_global": int(max_parallel_global),
        "max_parallel_per_destination": int(max_parallel_per_destination),
    }
    if activity_audit:
        identity["activity_audit_sha256"] = activity_audit["sha256"]
    workflow_sha256 = _payload_sha256(identity)
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else manifest_path.with_name("asr-chunk-batch-workflow.json")
    )
    result = {
        "schema": SCHEMA,
        "status": "ready",
        "ok": True,
        "write": bool(write),
        "chunk_manifest": str(manifest_path),
        "chunk_manifest_sha256": _file_sha256(manifest_path),
        "bundle_dir": str(bundle or ""),
        "chunk_count": len(chunks),
        "consent_count": len(paths),
        "workflow_sha256": workflow_sha256,
        "activity_audit": activity_audit,
        "activity_audit_sha256": str(activity_audit.get("sha256") or ""),
        "nodes": evidence,
        "submission": {
            "performed": False,
            "transport": "trusted_capability_broker_mcp",
            "tool": SUBMISSION_TOOL,
            "arguments": submission_arguments,
        },
        "operator_boundary": {
            "provider_call_performed": False,
            "consent_created_or_modified": False,
            "batch_submitted": False,
            "existing_batch_scheduler_required": True,
            "exact_route_revision_revalidated_by_broker": True,
            "automatic_retry": False,
            "automatic_fallback": False,
            "canonical_transcript_modified": False,
            "vad_activity_candidates_resolved": bool(activity_audit),
        },
        "output_path": str(target),
        "created_at": now_iso(),
    }
    if write:
        write_json(target, result)
    return result


def _validate_manifest_state(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != CHUNK_MANIFEST_SCHEMA:
        raise ValueError("unsupported ASR chunk manifest schema")
    if manifest.get("status") != "completed" or not manifest.get("ok"):
        raise ValueError("ASR chunk manifest must be fully completed before submission")
    chunks = manifest.get("chunks") if isinstance(manifest.get("chunks"), list) else []
    if int(manifest.get("chunk_count") or -1) != len(chunks):
        raise ValueError("ASR chunk manifest count is inconsistent")
    if int(manifest.get("completed_chunk_count") or -1) != len(chunks):
        raise ValueError("ASR chunk manifest is missing completed chunks")
    if int(manifest.get("failed_chunk_count") or 0) != 0:
        raise ValueError("ASR chunk manifest contains failed chunks")


def _completed_chunks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = [
        dict(row) for row in manifest.get("chunks") or [] if isinstance(row, dict)
    ]
    chunks.sort(key=lambda row: int(row.get("position") or 0))
    if not chunks or any(row.get("status") != "completed" for row in chunks):
        raise ValueError("every ASR chunk must have completed local extraction")
    expected_positions = list(range(1, len(chunks) + 1))
    if [int(row.get("position") or 0) for row in chunks] != expected_positions:
        raise ValueError("ASR chunk positions must be contiguous and one-based")
    if len({str(row.get("chunk_id") or "") for row in chunks}) != len(chunks):
        raise ValueError("ASR chunk ids must be unique")
    return chunks


def _validate_chunk_artifact(chunk: dict[str, Any]) -> Path:
    path = Path(str(chunk.get("output_path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ASR chunk artifact not found: {path}")
    if path.stat().st_size != int(chunk.get("output_bytes") or -1):
        raise ValueError(f"ASR chunk artifact byte count changed: {path}")
    if _file_sha256(path) != str(chunk.get("output_sha256") or ""):
        raise ValueError(f"ASR chunk artifact SHA-256 changed: {path}")
    return path


def _single_consent_artifact(
    consent_status: dict[str, Any], consent_path: Path
) -> dict[str, Any]:
    artifacts = [
        dict(row)
        for row in consent_status.get("artifacts") or []
        if isinstance(row, dict)
    ]
    if len(artifacts) != 1:
        raise ValueError(f"each ASR chunk consent must authorize one file: {consent_path}")
    return artifacts[0]


def _require_zero_retry(consent_status: dict[str, Any], consent_path: Path) -> None:
    scope = (
        consent_status.get("scope")
        if isinstance(consent_status.get("scope"), dict)
        else {}
    )
    value = scope.get("max_retries_per_call")
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(
            f"ASR chunk consent must explicitly lock max_retries_per_call=0: {consent_path}"
        )


def _validated_activity_audit(
    value: str | Path | None,
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ASR VAD activity audit not found: {path}")
    payload = _object(read_json(path), "ASR VAD activity audit")
    if payload.get("schema") != ACTIVITY_AUDIT_SCHEMA:
        raise ValueError("unsupported ASR VAD activity audit schema")
    if payload.get("status") != "passed" or not payload.get("vad_coverage_verified"):
        raise ValueError("ASR VAD activity audit has unresolved candidate gaps")
    source = payload.get("source_media")
    source = source if isinstance(source, dict) else {}
    manifest_source_sha = str(manifest.get("source_sha256") or "")
    if not manifest_source_sha or str(source.get("sha256") or "") != manifest_source_sha:
        raise ValueError("ASR VAD activity audit source media does not match chunk manifest")
    manifest_vad_sha = str(manifest.get("vad_sha256") or "")
    if not manifest_vad_sha or str(payload.get("vad_sha256") or "") != manifest_vad_sha:
        raise ValueError("ASR VAD activity audit VAD JSON does not match chunk manifest")
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "schema": ACTIVITY_AUDIT_SCHEMA,
        "status": "passed",
        "vad_coverage_verified": True,
        "candidate_gap_count": 0,
        "source_media_sha256": manifest_source_sha,
        "vad_sha256": manifest_vad_sha,
    }


def _validate_parallelism(global_limit: int, destination_limit: int) -> None:
    if not 1 <= int(global_limit) <= 64:
        raise ValueError("max_parallel_global must be between 1 and 64")
    if not 1 <= int(destination_limit) <= int(global_limit):
        raise ValueError(
            "max_parallel_per_destination must be between 1 and max_parallel_global"
        )


def _bundle_dir(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser().resolve()
    if not (path / "manifest.json").is_file() or not (path / "timeline.json").is_file():
        raise ValueError("bundle_dir must contain manifest.json and timeline.json")
    return path


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value



def _payload_sha256(value: Any) -> str:
    return canonical_json_sha256(value)
