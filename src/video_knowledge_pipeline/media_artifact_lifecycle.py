from __future__ import annotations

import importlib.util
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .file_hash import sha256_file
from .path_defaults import workspace_root
from .storage import read_json


SCOPE_SCHEMA_VERSION = "video_knowledge_pipeline.media_artifact_scope.v1"
DOCTOR_SCHEMA_VERSION = "video_knowledge_pipeline.media_artifact_doctor.v1"
SCAN_SCHEMA_VERSION = "video_knowledge_pipeline.media_artifact_scan_receipt.v1"
PLAN_SCHEMA_VERSION = "video_knowledge_pipeline.media_artifact_plan_receipt.v1"
RESTORE_SCHEMA_VERSION = "video_knowledge_pipeline.media_artifact_restore_receipt.v1"
PROFILE_ID = "media"
CLEANUP_MODE = "plan_only"
MINIMUM_FREE_BYTES = 536870912
WARNING_FREE_BYTES = 5368709120
OUTPUT_HEADROOM_MULTIPLIER = 1.25
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_ITEMS = 1000
MAX_MAX_DEPTH = 8
MAX_MAX_ITEMS = 10000
SYNTHETIC_MARKER = "VKP_SYNTHETIC_ARTIFACT_FIXTURE_V1"

ALLOWED_CANDIDATE_KINDS = {
    "rebuildable_chunk",
    "extracted_frame",
    "proxy_video",
    "failed_render",
    "duplicate_transcode",
    "superseded_temp_output",
}
PROTECTED_CLASSES = {
    "original_media",
    "reviewed_subtitles",
    "timeline",
    "locked_version",
    "final_deliverable",
    "qa_hash_approval_receipt",
    "sensitive_media",
    "active_run_checkpoint",
}
SAFE_CANDIDATE_SENSITIVITY = {"synthetic", "public", "internal"}
CORE_CONTRACT_FILES = (
    "artifact_lifecycle_core.py",
    "profiles/artifact-lifecycle-profiles.v1.json",
    "schemas/artifact-lifecycle-manifest.v1.schema.json",
    "schemas/artifact-lifecycle-operation-plan.v1.schema.json",
    "schemas/artifact-lifecycle-operation-receipt.v1.schema.json",
)
SELF_MEDIA_CONTRACT_FILES = (
    "scripts/artifact_lifecycle.py",
    "schemas/artifact-lifecycle-manifest.v2.schema.json",
)
EXPECTED_CORE_HASHES = {
    "artifact_lifecycle_core.py": "197236dc537a37c21a5431a38cbf19397359cc46c361c70493e3f5081bc736ab",
    "profiles/artifact-lifecycle-profiles.v1.json": "42e7ab281752a9ce3981eda0991447b143187ce06de5470f7a9be8b5f973c548",
    "schemas/artifact-lifecycle-manifest.v1.schema.json": "fa2e8646755321bd6095e9695929e10860aee23567b89df9b2319c44a9d6b971",
    "schemas/artifact-lifecycle-operation-plan.v1.schema.json": "eb0a8636e18b5b1da37317ee95893e3aa1d67932cf0ddf5c84b9c77f0b15a14b",
    "schemas/artifact-lifecycle-operation-receipt.v1.schema.json": "691481a1980047ec0fc45f007e949996c7c202dbab6cbfa8416d5ae1e40820a9",
}
EXPECTED_SELF_MEDIA_HASHES = {
    "scripts/artifact_lifecycle.py": "b6549f63d26aea6ef9dbca58172df597f347957343a21af42cc433c712aba98c",
    "schemas/artifact-lifecycle-manifest.v2.schema.json": "fbeea388f396c404455c14282bf2a65086d6a5f15c4149f1c6d2ae9f223339e2",
}
_CORE_CACHE: dict[str, ModuleType] = {}


class MediaArtifactLifecycleError(RuntimeError):
    def __init__(self, error_code: str, detail: str = "") -> None:
        self.error_code = str(error_code or "media_artifact_lifecycle_blocked")
        self.detail = str(detail or "")
        super().__init__(f"{self.error_code}: {self.detail}".rstrip(": "))


def media_artifact_doctor(
    bundle_dir: str | Path | None = None,
    *,
    scope_path: str | Path | None = None,
    core_root: str | Path | None = None,
    self_media_root: str | Path | None = None,
    large_task: bool = False,
    estimated_output_bytes: int | None = None,
    explicit_non_large: bool = False,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    blocking_reason_codes: list[str] = []
    warnings: list[str] = []
    scope_summary: dict[str, Any] = {"status": "not_supplied"}
    disk: dict[str, Any] | None = None
    core_snapshot: dict[str, Any] = {"status": "unavailable"}
    self_media_snapshot: dict[str, Any] = {"status": "unavailable"}
    profile: dict[str, Any] = {}
    operation_class = "not_requested"
    if large_task:
        if explicit_non_large and estimated_output_bytes is not None:
            blocking_reason_codes.append("disk_preflight_mode_conflict")
            operation_class = "invalid"
        elif explicit_non_large:
            operation_class = "non_large_declared"
        elif estimated_output_bytes is None:
            blocking_reason_codes.append("estimated_output_bytes_required")
            operation_class = "large"
        elif not isinstance(estimated_output_bytes, int) or isinstance(estimated_output_bytes, bool) or estimated_output_bytes <= 0:
            blocking_reason_codes.append("estimated_output_bytes_must_be_positive_integer")
            operation_class = "large"
        else:
            operation_class = "large"
        reason_codes.extend(blocking_reason_codes)
    try:
        core_dir = _resolve_core_root(core_root)
        self_media_dir = _resolve_self_media_root(self_media_root)
        core_snapshot = _contract_snapshot(core_dir, CORE_CONTRACT_FILES)
        self_media_snapshot = _contract_snapshot(self_media_dir, SELF_MEDIA_CONTRACT_FILES)
        _assert_contract_pins(core_snapshot, EXPECTED_CORE_HASHES, "CORE_CONTRACT_PIN_DRIFT")
        _assert_contract_pins(self_media_snapshot, EXPECTED_SELF_MEDIA_HASHES, "SELF_MEDIA_CONTRACT_PIN_DRIFT")
        core = _load_core(core_dir)
        profile = _media_profile(core_dir)
        if profile.get("cleanup_mode") != CLEANUP_MODE or profile.get("automatic_purge") is not False:
            raise MediaArtifactLifecycleError("UNSAFE_MEDIA_PROFILE", str(profile))
        expected_disk_policy = {
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "warning_free_bytes": WARNING_FREE_BYTES,
            "output_headroom_multiplier": OUTPUT_HEADROOM_MULTIPLIER,
        }
        observed_disk_policy = {key: profile.get(key) for key in expected_disk_policy}
        if observed_disk_policy != expected_disk_policy or profile.get("large_output_preflight") is not True:
            raise MediaArtifactLifecycleError("MEDIA_PROFILE_DISK_POLICY_DRIFT", str(observed_disk_policy))
        if "minimum_free_ratio" in profile:
            raise MediaArtifactLifecycleError("LEGACY_MEDIA_PROFILE_RATIO_PRESENT", str(profile.get("minimum_free_ratio")))
        if bundle_dir is not None:
            bundle = _resolve_bundle(bundle_dir, core)
            scope_file = _resolve_scope_path(bundle, scope_path, core)
            scope = _load_scope(scope_file)
            _validate_scope(scope, bundle, core)
            compatibility = _compatibility_status(scope, bundle, core)
            scope_summary = {
                "status": "ready",
                "bundle": str(bundle),
                "scope_path": str(scope_file),
                "scope_sha256": sha256_file(scope_file),
                "artifact_roots": [str(value) for value in scope["artifact_roots"]],
                "authoritative_paths": [str(item["path"]) for item in scope["authoritative_paths"]],
                "declared_artifact_count": len(scope["artifacts"]),
                "disk_policy": dict(scope["disk_policy"]),
                "compatibility": compatibility,
            }
            if large_task and operation_class == "large" and not blocking_reason_codes:
                disk_probe_root = core.resolve_root(workspace_root())
                disk = dict(
                    core.free_space_status(
                        disk_probe_root,
                        None,
                        minimum_free_bytes=MINIMUM_FREE_BYTES,
                        warning_free_bytes=WARNING_FREE_BYTES,
                        estimated_output_bytes=estimated_output_bytes,
                        output_headroom_multiplier=OUTPUT_HEADROOM_MULTIPLIER,
                    )
                )
                disk.setdefault("probe_path", str(disk_probe_root))
                if disk.get("status") == "blocked_low_disk":
                    reason_codes.append("blocked_low_disk")
                    blocking_reason_codes.append("blocked_low_disk")
                elif disk.get("status") == "low_disk_allowed":
                    reason_codes.append("low_disk_allowed")
                    warnings.append("low_disk_allowed")
        elif large_task:
            reason_codes.append("bundle_required_for_large_task_preflight")
            blocking_reason_codes.append("bundle_required_for_large_task_preflight")
    except MediaArtifactLifecycleError as exc:
        reason_codes.append(exc.error_code)
        blocking_reason_codes.append(exc.error_code)
        scope_summary = {"status": "blocked", "detail": exc.detail}
        profile = {}
    except Exception as exc:  # noqa: BLE001 - convert upstream errors to a stable doctor result.
        error_code = _upstream_error_code(exc)
        reason_codes.append(error_code)
        blocking_reason_codes.append(error_code)
        scope_summary = {"status": "blocked", "detail": str(exc)}
        profile = {}
    status = "blocked" if blocking_reason_codes else "ready"
    return {
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "status": status,
        "ready": status == "ready",
        "lifecycle_profile": PROFILE_ID,
        "cleanup_mode": CLEANUP_MODE,
        "automatic_purge": False,
        "core_contract": core_snapshot,
        "self_media_v2_contract": self_media_snapshot,
        "media_profile": profile,
        "scope": scope_summary,
        "large_task_preflight": {
            "requested": bool(large_task),
            "operation_class": operation_class,
            "estimated_output_bytes": estimated_output_bytes,
            "policy": {
                "minimum_free_bytes": MINIMUM_FREE_BYTES,
                "warning_free_bytes": WARNING_FREE_BYTES,
                "output_headroom_multiplier": OUTPUT_HEADROOM_MULTIPLIER,
            },
            "disk": disk,
            "warning_only": bool(disk and disk.get("warning_only")),
            "starts_task": False,
        },
        "interfaces": {
            "doctor": "media-artifact-doctor",
            "preflight": "media-artifact-preflight",
            "scan": "media-artifact-scan",
            "explain": "media-artifact-explain",
            "plan_gc": "media-artifact-plan-gc",
            "verify": "media-artifact-verify-plan",
            "restore": "media-artifact-restore --execute-synthetic",
        },
        "reason_codes": sorted(set(reason_codes)),
        "blocking_reason_codes": sorted(set(blocking_reason_codes)),
        "warnings": sorted(set(warnings)),
        "boundary": {
            "explicit_roots_only": True,
            "bounded_scan": True,
            "plan_only": True,
            "quarantine_interface_exposed": False,
            "purge_interface_exposed": False,
            "real_media_mutation_allowed": False,
            "network_provider_upload_publish": False,
        },
        "filesystem_mutation_performed": False,
    }


def scan_media_artifacts(
    bundle_dir: str | Path,
    *,
    scope_path: str | Path,
    output_path: str | Path,
    checkpoint_dir: str | Path,
    core_root: str | Path | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> dict[str, Any]:
    _validate_limits(max_depth, max_items)
    core_dir = _resolve_core_root(core_root)
    core = _load_core(core_dir)
    bundle = _resolve_bundle(bundle_dir, core)
    scope_file = _resolve_scope_path(bundle, scope_path, core)
    scope = _load_scope(scope_file)
    _validate_scope(scope, bundle, core)
    output = Path(output_path).expanduser().absolute()
    checkpoints = Path(checkpoint_dir).expanduser().absolute()
    roots = [_resolve_relative(bundle, value, core, must_exist=True) for value in scope["artifact_roots"]]
    _assert_metadata_destination_safe(output, roots, scope, bundle, core)
    _assert_metadata_destination_safe(checkpoints, roots, scope, bundle, core, directory=True)

    total_items = 0
    total_files = 0
    gate_evidence = []
    for root in roots:
        gate = _bounded_root_gate(root, bundle, core, max_depth=max_depth, remaining_items=max_items - total_items)
        total_items += gate["item_count"]
        total_files += gate["file_count"]
        gate_evidence.append(gate)
    if total_items > max_items:
        raise MediaArtifactLifecycleError("SCAN_ITEM_LIMIT_EXCEEDED", str(max_items))

    declarations = _declaration_index(scope)
    compatibility = _compatibility_view(scope, bundle, core)
    scanned: list[dict[str, Any]] = []
    scan_errors: list[dict[str, str]] = []
    checked_times: list[str] = []
    checkpoints.mkdir(parents=True, exist_ok=True)
    for index, root in enumerate(roots, start=1):
        root_relative = root.relative_to(bundle).as_posix()
        checkpoint = checkpoints / f"root-{index:02d}-{_short_hash(root_relative)}.sqlite"
        try:
            partial = core.scan_root(
                root,
                profile_id=PROFILE_ID,
                producer_tool_id=str(scope["producer_tool_id"]),
                owner_lane=str(scope["owner_lane"]),
                closure_run_id=str(scope["closure_run_id"]),
                max_files=max_items,
                checkpoint=checkpoint,
            )
        except Exception as exc:  # noqa: BLE001
            raise _wrap_upstream(exc) from exc
        checked_times.append(str((partial.get("scan") or {}).get("checked_at") or "1970-01-01T00:00:00Z"))
        scan_errors.extend(dict(item) for item in (partial.get("scan") or {}).get("errors") or [])
        for item in partial.get("artifacts") or []:
            relative = (PurePosixPath(root_relative) / PurePosixPath(str(item["path"]))).as_posix()
            scanned.append(_overlay_scanned_artifact(item, relative, declarations, compatibility, scope, scan_errors))

    for declared in scope["authoritative_paths"]:
        scanned.append(_scan_exact_declared(bundle, declared, core))
        path = _resolve_relative(bundle, declared["path"], core, must_exist=True)
        checked_times.append(_mtime_iso(path))

    found_paths = {str(item["path"]) for item in scanned}
    for path in sorted(set(declarations) - found_paths):
        scan_errors.append({"path": path, "error": "DECLARED_ARTIFACT_MISSING"})
    _apply_hardlink_coverage(scanned)
    _validate_relationships(scanned)
    artifacts = sorted(scanned, key=lambda item: (str(item["path"]).casefold(), str(item["artifact_id"])))
    manifest = {
        "schema_version": core.SCHEMA_VERSION,
        "manifest_id": f"vkp-media-{core.canonical_sha256([item['artifact_id'] for item in artifacts])[:20]}",
        "closure_run_id": scope["closure_run_id"],
        "producer_tool_id": scope["producer_tool_id"],
        "owner_lane": scope["owner_lane"],
        "root_identity": core.directory_identity(bundle),
        "profile_id": PROFILE_ID,
        "cleanup_mode": CLEANUP_MODE,
        "automatic_purge": False,
        "scan": {
            "checked_at": max(checked_times, default="1970-01-01T00:00:00Z"),
            "explicit_roots": [str(value) for value in scope["artifact_roots"]],
            "max_depth": max_depth,
            "max_items": max_items,
            "observed_items": total_items,
            "observed_files_in_roots": total_files,
            "gate_evidence": gate_evidence,
            "errors": sorted(scan_errors, key=lambda item: (item.get("path", ""), item.get("error", ""))),
            "streaming_or_incremental": True,
            "whole_workspace_scan": False,
        },
        "adapter": {
            "schema_version": SCAN_SCHEMA_VERSION,
            "scope_path": str(scope_file),
            "scope_sha256": sha256_file(scope_file),
            "core_contract": _contract_snapshot(core_dir, CORE_CONTRACT_FILES),
            "compatibility": _compatibility_status(scope, bundle, core),
            "protection_policy": list(scope["protection_policy"]),
            "active_run": dict(scope["active_run"]),
        },
        "artifacts": artifacts,
    }
    manifest = core.attach_hash(manifest, "manifest_sha256")
    errors = core.validate_manifest(manifest)
    if errors:
        raise MediaArtifactLifecycleError("CORE_MANIFEST_INVALID", ";".join(errors))
    core.atomic_write_json(output, manifest)
    return {
        "schema_version": SCAN_SCHEMA_VERSION,
        "status": "blocked" if scan_errors else "passed",
        "manifest_path": str(output),
        "manifest_sha256": manifest["manifest_sha256"],
        "artifact_count": len(artifacts),
        "candidate_count": sum("candidate_kind" in item for item in artifacts),
        "protected_count": sum("protection_class" in item for item in artifacts),
        "scan_errors": manifest["scan"]["errors"],
        "filesystem_mutation_performed": False,
        "metadata_files_written": [str(output), *[str(path) for path in sorted(checkpoints.glob("*.sqlite"))]],
    }


def explain_media_artifact(
    manifest_path: str | Path,
    artifact_id: str,
    *,
    core_root: str | Path | None = None,
) -> dict[str, Any]:
    core = _load_core(_resolve_core_root(core_root))
    manifest = _load_manifest(manifest_path, core)
    item = next((row for row in manifest["artifacts"] if row.get("artifact_id") == artifact_id), None)
    if item is None:
        raise MediaArtifactLifecycleError("ARTIFACT_NOT_FOUND", artifact_id)
    blockers = _adapter_item_blockers(item)
    return {
        "status": "passed",
        "artifact": item,
        "candidate_kind_allowed": item.get("candidate_kind") in ALLOWED_CANDIDATE_KINDS,
        "blocking_gates": sorted(blockers),
        "authorizes_operation": False,
        "filesystem_mutation_performed": False,
    }


def plan_media_artifact_gc(
    manifest_path: str | Path,
    *,
    bundle_dir: str | Path,
    quarantine_root: str | Path,
    output_path: str | Path,
    core_root: str | Path | None = None,
) -> dict[str, Any]:
    core_dir = _resolve_core_root(core_root)
    core = _load_core(core_dir)
    manifest = _load_manifest(manifest_path, core)
    bundle = _resolve_bundle(bundle_dir, core)
    quarantine = core.resolve_root(quarantine_root)
    _verify_adapter_contract(manifest, bundle, core_dir, core)
    output = Path(output_path).expanduser().absolute()
    _assert_manifest_metadata_destination_safe(output, manifest, bundle, core)
    active = dict((manifest.get("adapter") or {}).get("active_run") or {})
    owner_state = str(active.get("owner_state") or "unknown")
    dispatch_state = str(active.get("dispatch_state") or "unknown")
    process_state = "idle_confirmed" if active.get("state") == "idle_confirmed" and not active.get("process_ids") else "active"
    try:
        plan = core.plan_operation(
            manifest,
            bundle,
            quarantine,
            "quarantine",
            owner_state=owner_state,
            process_state=process_state,
            dispatch_state=dispatch_state,
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap_upstream(exc) from exc
    by_id = {str(item["artifact_id"]): item for item in manifest["artifacts"]}
    for entry in plan["entries"]:
        item = by_id[str(entry["artifact_id"])]
        blockers = set(str(value) for value in entry.get("blocking_gates") or [])
        blockers.update(_adapter_item_blockers(item))
        entry["blocking_gates"] = sorted(blockers)
        entry["eligible"] = not blockers
    plan["gate_evidence"].update(
        {
            "adapter_scope_sha256": (manifest.get("adapter") or {}).get("scope_sha256"),
            "lifecycle_profile": PROFILE_ID,
            "cleanup_mode": CLEANUP_MODE,
            "active_run_state": active.get("state", "unknown"),
        }
    )
    plan["boundary"] = "VKP plan-only media GC metadata; no quarantine, purge, delete, move, overwrite, upload, provider or publish authority."
    plan = core.attach_hash(plan, "plan_sha256")
    validation = core.validate_plan(plan)
    if validation:
        raise MediaArtifactLifecycleError("CORE_PLAN_INVALID", ";".join(validation))
    core.atomic_write_json(output, plan)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "planned",
        "plan_path": str(output),
        "plan_sha256": plan["plan_sha256"],
        "eligible_count": sum(bool(item["eligible"]) for item in plan["entries"]),
        "blocked_count": sum(not bool(item["eligible"]) for item in plan["entries"]),
        "authorization_state": "not_authorized",
        "cleanup_mode": CLEANUP_MODE,
        "filesystem_mutation_performed": False,
    }


def verify_media_artifact_plan(
    plan_path: str | Path,
    manifest_path: str | Path,
    *,
    bundle_dir: str | Path,
    quarantine_root: str | Path,
    core_root: str | Path | None = None,
) -> dict[str, Any]:
    core_dir = _resolve_core_root(core_root)
    core = _load_core(core_dir)
    manifest = _load_manifest(manifest_path, core)
    plan = _read_object(Path(plan_path), "PLAN_READ_FAILED")
    bundle = _resolve_bundle(bundle_dir, core)
    quarantine = core.resolve_root(quarantine_root)
    errors: list[str] = []
    try:
        _verify_adapter_contract(manifest, bundle, core_dir, core)
    except MediaArtifactLifecycleError as exc:
        errors.append(exc.error_code)
    verification = core.verify_plan(plan, manifest, bundle, quarantine)
    errors.extend(str(value) for value in verification.get("errors") or [])
    for item in manifest["artifacts"]:
        try:
            path = core.artifact_path(bundle, item)
            if not path.is_file():
                errors.append(f"artifact_missing:{item['artifact_id']}")
                continue
            stat_result = path.stat()
            identity = item.get("file_identity") or {}
            if int(identity.get("volume_id") or -1) != int(stat_result.st_dev) or int(identity.get("file_id") or -1) != int(stat_result.st_ino):
                errors.append(f"artifact_identity_drift:{item['artifact_id']}")
            if int(item.get("bytes") or -1) != int(stat_result.st_size):
                errors.append(f"artifact_size_drift:{item['artifact_id']}")
            if sha256_file(path) != str((item.get("digest") or {}).get("value") or ""):
                errors.append(f"artifact_digest_drift:{item['artifact_id']}")
            if int(identity.get("hardlink_count") or 1) != int(stat_result.st_nlink):
                errors.append(f"hardlink_count_drift:{item['artifact_id']}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"artifact_verification_failed:{item.get('artifact_id')}:{_upstream_error_code(exc)}")
    result = {
        "status": "passed" if not errors else "blocked",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "plan_sha256": plan.get("plan_sha256"),
        "cleanup_mode": CLEANUP_MODE,
        "authorization_state": "not_authorized",
        "filesystem_mutation_performed": False,
    }
    return result


def restore_synthetic_media_artifact(
    plan_path: str | Path,
    receipt_path: str | Path,
    *,
    bundle_dir: str | Path,
    quarantine_root: str | Path,
    output_path: str | Path,
    execute_synthetic: bool = False,
    core_root: str | Path | None = None,
) -> dict[str, Any]:
    core = _load_core(_resolve_core_root(core_root))
    bundle = _resolve_bundle(bundle_dir, core)
    quarantine = core.resolve_root(quarantine_root)
    marker_path = bundle / ".vkp-synthetic-artifact-fixture.json"
    marker = _read_object(marker_path, "SYNTHETIC_MARKER_REQUIRED")
    if marker.get("contract") != SYNTHETIC_MARKER or marker.get("fixture_only") is not True or marker.get("allow_restore_test") is not True:
        raise MediaArtifactLifecycleError("SYNTHETIC_RESTORE_NOT_AUTHORIZED", str(marker_path))
    plan = _read_object(Path(plan_path), "PLAN_READ_FAILED")
    receipt = _read_object(Path(receipt_path), "RECEIPT_READ_FAILED")
    errors = []
    if not core.verify_hash(plan, "plan_sha256"):
        errors.append("plan_hash_drift")
    if not core.verify_hash(receipt, "receipt_sha256"):
        errors.append("receipt_hash_drift")
    if receipt.get("operation") != "quarantine" or receipt.get("status") != "completed":
        errors.append("quarantine_receipt_invalid")
    if receipt.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("receipt_plan_hash_mismatch")
    if receipt.get("closure_run_id") != marker.get("closure_run_id") or plan.get("closure_run_id") != marker.get("closure_run_id"):
        errors.append("synthetic_closure_identity_mismatch")
    for item in receipt.get("items") or []:
        source = Path(str(item.get("to") or ""))
        target = Path(str(item.get("from") or ""))
        if not core.is_under(source, quarantine) or not core.is_under(target, bundle):
            errors.append(f"restore_path_outside_scope:{item.get('artifact_id')}")
        if source.suffix != ".fixture" or target.suffix != ".fixture":
            errors.append(f"non_synthetic_restore_item:{item.get('artifact_id')}")
    if errors:
        raise MediaArtifactLifecycleError("SYNTHETIC_RESTORE_BLOCKED", ";".join(sorted(set(errors))))
    if not execute_synthetic:
        return {
            "schema_version": RESTORE_SCHEMA_VERSION,
            "status": "ready_to_restore_synthetic",
            "execute_synthetic_required": True,
            "filesystem_mutation_performed": False,
        }
    output = Path(output_path).expanduser().absolute()
    _assert_plan_metadata_destination_safe(output, plan, bundle, core)
    try:
        restored = core.restore_from_receipt(receipt, plan, bundle, quarantine, output)
    except Exception as exc:  # noqa: BLE001
        raise _wrap_upstream(exc) from exc
    return {
        "schema_version": RESTORE_SCHEMA_VERSION,
        "status": "passed",
        "closure_run_id": restored.get("closure_run_id"),
        "receipt_path": str(output),
        "restored_count": len(restored.get("items") or []),
        "filesystem_mutation_performed": True,
        "mutation_scope": "synthetic_fixture_only",
    }


def _resolve_core_root(value: str | Path | None) -> Path:
    raw = value or os.environ.get("ARTIFACT_LIFECYCLE_CORE_ROOT") or (workspace_root() / "artifact-lifecycle-core")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise MediaArtifactLifecycleError("ARTIFACT_LIFECYCLE_CORE_NOT_FOUND", str(root))
    return root


def _resolve_self_media_root(value: str | Path | None) -> Path:
    raw = value or os.environ.get("SELF_MEDIA_CREATION_SYSTEM_ROOT") or (workspace_root() / "self-media-creation-system")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise MediaArtifactLifecycleError("SELF_MEDIA_CONTRACT_NOT_FOUND", str(root))
    return root


def _contract_snapshot(root: Path, relative_paths: tuple[str, ...]) -> dict[str, Any]:
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise MediaArtifactLifecycleError("UPSTREAM_CONTRACT_FILE_MISSING", str(path))
        files.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    identity_seed = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": str(root),
        "files": files,
        "identity_sha256": hashlib.sha256(identity_seed).hexdigest(),
        "read_only": True,
    }


def _load_core(root: Path) -> ModuleType:
    snapshot = _contract_snapshot(root, CORE_CONTRACT_FILES)
    _assert_contract_pins(snapshot, EXPECTED_CORE_HASHES, "CORE_CONTRACT_PIN_DRIFT")
    key = str(snapshot["identity_sha256"])
    if key in _CORE_CACHE:
        return _CORE_CACHE[key]
    source = root / "artifact_lifecycle_core.py"
    spec = importlib.util.spec_from_file_location(f"vkp_artifact_lifecycle_core_{key[:12]}", source)
    if spec is None or spec.loader is None:
        raise MediaArtifactLifecycleError("CORE_IMPORT_FAILED", str(source))
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise MediaArtifactLifecycleError("CORE_IMPORT_FAILED", str(exc)) from exc
    finally:
        sys.dont_write_bytecode = previous
    _CORE_CACHE[key] = module
    return module


def _media_profile(core_root: Path) -> dict[str, Any]:
    value = _read_object(core_root / "profiles" / "artifact-lifecycle-profiles.v1.json", "PROFILE_READ_FAILED")
    for item in value.get("profiles") or []:
        if isinstance(item, dict) and item.get("profile_id") == PROFILE_ID:
            return dict(item)
    raise MediaArtifactLifecycleError("MEDIA_PROFILE_NOT_FOUND", str(core_root))


def _assert_contract_pins(snapshot: dict[str, Any], expected: dict[str, str], error_code: str) -> None:
    observed = {str(item["path"]): str(item["sha256"]) for item in snapshot.get("files") or []}
    if observed != expected:
        changed = sorted(path for path in set(observed) | set(expected) if observed.get(path) != expected.get(path))
        raise MediaArtifactLifecycleError(error_code, ",".join(changed))


def _load_scope(path: Path) -> dict[str, Any]:
    value = _read_object(path, "SCOPE_READ_FAILED")
    schema_path = Path(__file__).resolve().parent / "schemas" / "media-artifact-scope.v1.schema.json"
    schema = _read_object(schema_path, "SCOPE_SCHEMA_READ_FAILED")
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(value) for value in first.path) or "$"
        raise MediaArtifactLifecycleError("SCOPE_SCHEMA_INVALID", f"{location}: {first.message}")
    return value


def _validate_scope(scope: dict[str, Any], bundle: Path, core: ModuleType) -> None:
    if set(scope.get("protection_policy") or []) != PROTECTED_CLASSES:
        raise MediaArtifactLifecycleError("PROTECTION_POLICY_INCOMPLETE", ",".join(sorted(set(PROTECTED_CLASSES) - set(scope.get("protection_policy") or []))))
    declarations = [*scope["authoritative_paths"], *scope["artifacts"]]
    ids: set[str] = set()
    paths: set[str] = set()
    for item in declarations:
        if item["artifact_id"] in ids:
            raise MediaArtifactLifecycleError("ARTIFACT_ID_DUPLICATED", str(item["artifact_id"]))
        if item["path"] in paths:
            raise MediaArtifactLifecycleError("ARTIFACT_PATH_DUPLICATED", str(item["path"]))
        ids.add(str(item["artifact_id"]))
        paths.add(str(item["path"]))
        if item["closure_run_id"] != scope["closure_run_id"] or item["producer_tool_id"] != scope["producer_tool_id"] or item["owner_lane"] != scope["owner_lane"]:
            raise MediaArtifactLifecycleError("ARTIFACT_OWNER_IDENTITY_MISMATCH", str(item["artifact_id"]))
        _resolve_relative(bundle, item["path"], core, must_exist=True)
    roots = [_resolve_relative(bundle, value, core, must_exist=True) for value in scope["artifact_roots"]]
    for index, root in enumerate(roots):
        if not root.is_dir():
            raise MediaArtifactLifecycleError("ARTIFACT_ROOT_NOT_DIRECTORY", str(root))
        for other in roots[index + 1 :]:
            if _path_is_under(root, other) or _path_is_under(other, root):
                raise MediaArtifactLifecycleError("ARTIFACT_ROOTS_OVERLAP", f"{root} / {other}")
    for item in scope["authoritative_paths"]:
        path = _resolve_relative(bundle, item["path"], core, must_exist=True)
        if any(_path_is_under(path, root) for root in roots):
            raise MediaArtifactLifecycleError("AUTHORITATIVE_PATH_INSIDE_ARTIFACT_ROOT", str(path))
        if not item.get("pin") or item.get("artifact_role") not in core.PROTECTED_ROLES:
            raise MediaArtifactLifecycleError("AUTHORITATIVE_PATH_NOT_PROTECTED", str(item["artifact_id"]))
    for item in scope["artifacts"]:
        path = _resolve_relative(bundle, item["path"], core, must_exist=True)
        if sum(_path_is_under(path, root) for root in roots) != 1:
            raise MediaArtifactLifecycleError("ARTIFACT_OUTSIDE_EXPLICIT_ROOT", str(path))
        if item.get("candidate_kind") not in ALLOWED_CANDIDATE_KINDS:
            raise MediaArtifactLifecycleError("CANDIDATE_KIND_NOT_ALLOWED", str(item.get("candidate_kind")))
    known = ids
    for item in declarations:
        unknown = sorted(set(item.get("depends_on") or []) - known)
        if unknown:
            raise MediaArtifactLifecycleError("UNKNOWN_DEPENDENCY", f"{item['artifact_id']}:{','.join(unknown)}")
    checkpoint_ids = set(scope["active_run"].get("checkpoint_artifact_ids") or [])
    if not checkpoint_ids.issubset(ids):
        raise MediaArtifactLifecycleError("ACTIVE_CHECKPOINT_UNKNOWN", ",".join(sorted(checkpoint_ids - ids)))
    for checkpoint_id in checkpoint_ids:
        item = next(row for row in scope["authoritative_paths"] if row["artifact_id"] == checkpoint_id)
        if item.get("protection_class") != "active_run_checkpoint" or not item.get("pin"):
            raise MediaArtifactLifecycleError("ACTIVE_CHECKPOINT_NOT_PROTECTED", checkpoint_id)


def _resolve_bundle(value: str | Path, core: ModuleType) -> Path:
    try:
        return core.resolve_root(value)
    except Exception as exc:  # noqa: BLE001
        raise _wrap_upstream(exc) from exc


def _resolve_scope_path(bundle: Path, value: str | Path | None, core: ModuleType) -> Path:
    raw = value or ".lifecycle/media-artifact-scope.v1.json"
    path = Path(raw).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not _path_is_under(resolved, bundle):
            raise MediaArtifactLifecycleError("SCOPE_OUTSIDE_BUNDLE", str(resolved))
        core.assert_no_reparse(resolved, bundle)
        if not resolved.is_file():
            raise MediaArtifactLifecycleError("SCOPE_NOT_FILE", str(resolved))
        return resolved
    return _resolve_relative(bundle, str(raw), core, must_exist=True)


def _resolve_relative(bundle: Path, raw: str | Path, core: ModuleType, *, must_exist: bool) -> Path:
    text = str(raw).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise MediaArtifactLifecycleError("PATH_OUTSIDE_BUNDLE", text)
    path = bundle.joinpath(*pure.parts)
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise MediaArtifactLifecycleError("PATH_ACCESS_FAILED", f"{path}: {exc}") from exc
    if not _path_is_under(resolved, bundle):
        raise MediaArtifactLifecycleError("PATH_OUTSIDE_BUNDLE", text)
    if must_exist:
        try:
            core.assert_no_reparse(resolved, bundle)
        except Exception as exc:  # noqa: BLE001
            raise _wrap_upstream(exc) from exc
    return resolved


def _bounded_root_gate(root: Path, bundle: Path, core: ModuleType, *, max_depth: int, remaining_items: int) -> dict[str, Any]:
    if remaining_items < 0:
        raise MediaArtifactLifecycleError("SCAN_ITEM_LIMIT_EXCEEDED", "0")
    stack: list[tuple[Path, int]] = [(root, 0)]
    item_count = 0
    file_count = 0
    while stack:
        directory, depth = stack.pop()
        try:
            core.assert_no_reparse(directory, bundle)
            entries = os.scandir(directory)
        except Exception as exc:  # noqa: BLE001
            raise _wrap_upstream(exc) from exc
        with entries:
            for entry in entries:
                item_count += 1
                if item_count > remaining_items:
                    raise MediaArtifactLifecycleError("SCAN_ITEM_LIMIT_EXCEEDED", str(remaining_items))
                path = Path(entry.path)
                try:
                    if entry.is_symlink() or core._is_reparse(path):
                        raise MediaArtifactLifecycleError("REPARSE_POINT_BLOCKED", str(path))
                    if entry.is_dir(follow_symlinks=False):
                        if depth >= max_depth:
                            raise MediaArtifactLifecycleError("SCAN_DEPTH_LIMIT_EXCEEDED", str(path))
                        stack.append((path, depth + 1))
                    elif entry.is_file(follow_symlinks=False):
                        file_count += 1
                except MediaArtifactLifecycleError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise _wrap_upstream(exc) from exc
    return {"root": str(root), "max_depth": max_depth, "item_count": item_count, "file_count": file_count}


def _declaration_index(scope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["path"]): dict(item) for item in [*scope["authoritative_paths"], *scope["artifacts"]]}


def _compatibility_view(scope: dict[str, Any], bundle: Path, core: ModuleType) -> dict[str, dict[str, Any]]:
    raw = scope.get("compatibility_manifest")
    if not raw:
        return {}
    path = _resolve_relative(bundle, str(raw), core, must_exist=True)
    try:
        manifest = core.load_manifest(path)
    except Exception as exc:  # noqa: BLE001
        raise _wrap_upstream(exc) from exc
    return {str(item.get("path")): dict(item) for item in manifest.get("artifacts") or [] if isinstance(item, dict)}


def _compatibility_status(scope: dict[str, Any], bundle: Path, core: ModuleType) -> dict[str, Any]:
    view = _compatibility_view(scope, bundle, core)
    return {
        "source": "self-media-artifact-lifecycle-manifest.v2" if scope.get("compatibility_manifest") else "none",
        "manifest_path": str(scope.get("compatibility_manifest") or ""),
        "artifact_count": len(view),
        "loaded_through_core_compatibility_view": bool(scope.get("compatibility_manifest")),
        "rewritten": False,
    }


def _overlay_scanned_artifact(
    scanned: dict[str, Any],
    relative: str,
    declarations: dict[str, dict[str, Any]],
    compatibility: dict[str, dict[str, Any]],
    scope: dict[str, Any],
    scan_errors: list[dict[str, str]],
) -> dict[str, Any]:
    declared = declarations.get(relative)
    if declared is None:
        compat = compatibility.get(relative)
        item = dict(scanned)
        item.update(dict(compat or {}))
        item.update(
            {
                "artifact_id": str((compat or {}).get("artifact_id") or f"undeclared-{_short_hash(relative)}"),
                "closure_run_id": scope["closure_run_id"],
                "producer_tool_id": scope["producer_tool_id"],
                "owner_lane": scope["owner_lane"],
                "path": relative,
                "artifact_role": "evidence_receipt_manifest",
                "digest": dict(scanned["digest"]),
                "bytes": int(scanned["bytes"]),
                "lifecycle_state": "locked",
                "storage_state": "present",
                "depends_on": list((compat or {}).get("depends_on") or []),
                "referenced_by": list((compat or {}).get("referenced_by") or []),
                "sensitivity": "unknown",
                "rebuild_proof": {"status": "unknown", "command": None, "input_hashes": [], "dependency_lock": None, "verification_ref": None},
                "minimum_keep_until": None,
                "pin": True,
                "compatibility_source": "self-media-v2" if compat else "none",
            }
        )
        scan_errors.append({"path": relative, "error": "DECLARATION_MISSING"})
        return item
    item = dict(scanned)
    item.update(declared)
    item["path"] = relative
    item["storage_state"] = "present"
    item["digest"] = dict(scanned["digest"])
    item["bytes"] = int(scanned["bytes"])
    if declared["digest"]["value"] != scanned["digest"]["value"]:
        scan_errors.append({"path": relative, "error": "DECLARED_DIGEST_DRIFT"})
    if int(declared["bytes"]) != int(scanned["bytes"]):
        scan_errors.append({"path": relative, "error": "DECLARED_SIZE_DRIFT"})
    compat = compatibility.get(relative)
    if compat:
        compat_digest = str((compat.get("digest") or {}).get("value") or "")
        if compat_digest not in {"", "0" * 64, scanned["digest"]["value"]}:
            scan_errors.append({"path": relative, "error": "SELF_MEDIA_V2_DIGEST_DRIFT"})
        item["compatibility_source"] = "self-media-v2"
    return item


def _scan_exact_declared(bundle: Path, declared: dict[str, Any], core: ModuleType) -> dict[str, Any]:
    path = _resolve_relative(bundle, declared["path"], core, must_exist=True)
    if not path.is_file():
        raise MediaArtifactLifecycleError("AUTHORITATIVE_PATH_NOT_FILE", str(path))
    info = path.stat()
    digest = sha256_file(path)
    if digest != declared["digest"]["value"] or int(info.st_size) != int(declared["bytes"]):
        raise MediaArtifactLifecycleError("AUTHORITATIVE_IDENTITY_DRIFT", str(declared["artifact_id"]))
    item = dict(declared)
    item.update(
        {
            "digest": {"algorithm": "sha256", "value": digest, "status": "verified"},
            "bytes": int(info.st_size),
            "storage_state": "present",
            "file_identity": {
                "volume_id": int(info.st_dev),
                "file_id": int(info.st_ino),
                "hardlink_count": int(info.st_nlink),
                "observed_links": 0,
            },
        }
    )
    return item


def _apply_hardlink_coverage(items: list[dict[str, Any]]) -> None:
    observed: dict[tuple[int, int], int] = {}
    for item in items:
        identity = item.get("file_identity") or {}
        key = (int(identity.get("volume_id") or 0), int(identity.get("file_id") or 0))
        observed[key] = observed.get(key, 0) + 1
    for item in items:
        identity = dict(item.get("file_identity") or {})
        key = (int(identity.get("volume_id") or 0), int(identity.get("file_id") or 0))
        identity["observed_links"] = observed.get(key, 0)
        item["file_identity"] = identity


def _validate_relationships(items: list[dict[str, Any]]) -> None:
    ids = {str(item["artifact_id"]) for item in items}
    if len(ids) != len(items):
        raise MediaArtifactLifecycleError("ARTIFACT_ID_DUPLICATED_AFTER_SCAN")
    for item in items:
        unknown = sorted(set(item.get("depends_on") or []) - ids)
        if unknown:
            raise MediaArtifactLifecycleError("UNKNOWN_DEPENDENCY_AFTER_SCAN", f"{item['artifact_id']}:{','.join(unknown)}")


def _adapter_item_blockers(item: dict[str, Any]) -> set[str]:
    blockers: set[str] = set()
    if item.get("protection_class"):
        blockers.add(f"permanent_protection:{item.get('protection_class')}")
    if item.get("candidate_kind") not in ALLOWED_CANDIDATE_KINDS:
        blockers.add("candidate_kind_not_governed")
    if item.get("sensitivity") not in SAFE_CANDIDATE_SENSITIVITY:
        blockers.add("sensitivity_not_cleared")
    proof = item.get("rebuild_proof") or {}
    if item.get("candidate_kind") and not (
        proof.get("status") == "verified"
        and proof.get("command")
        and proof.get("input_hashes")
        and proof.get("dependency_lock")
        and proof.get("verification_ref")
    ):
        blockers.add("rebuild_proof_incomplete")
    if item.get("referenced_by"):
        blockers.add("references_not_cleared")
    return blockers


def _verify_adapter_contract(manifest: dict[str, Any], bundle: Path, core_root: Path, core: ModuleType) -> None:
    adapter = manifest.get("adapter") if isinstance(manifest.get("adapter"), dict) else {}
    scope_path = Path(str(adapter.get("scope_path") or ""))
    if not scope_path.is_file() or sha256_file(scope_path) != adapter.get("scope_sha256"):
        raise MediaArtifactLifecycleError("SCOPE_HASH_DRIFT", str(scope_path))
    current = _contract_snapshot(core_root, CORE_CONTRACT_FILES)
    recorded = adapter.get("core_contract") if isinstance(adapter.get("core_contract"), dict) else {}
    if current.get("identity_sha256") != recorded.get("identity_sha256"):
        raise MediaArtifactLifecycleError("CORE_CONTRACT_DRIFT", str(core_root))
    if manifest.get("root_identity") != core.directory_identity(bundle):
        raise MediaArtifactLifecycleError("ROOT_IDENTITY_DRIFT", str(bundle))


def _load_manifest(path: str | Path, core: ModuleType) -> dict[str, Any]:
    try:
        return core.load_manifest(Path(path).expanduser().resolve())
    except Exception as exc:  # noqa: BLE001
        raise _wrap_upstream(exc) from exc


def _assert_metadata_destination_safe(
    destination: Path,
    roots: list[Path],
    scope: dict[str, Any],
    bundle: Path,
    core: ModuleType,
    *,
    directory: bool = False,
) -> None:
    _assert_destination_no_reparse(destination, core)
    if any(_path_is_under(destination, root) for root in roots):
        raise MediaArtifactLifecycleError("METADATA_DESTINATION_INSIDE_ARTIFACT_ROOT", str(destination))
    for declared in scope["authoritative_paths"]:
        authoritative = _resolve_relative(bundle, declared["path"], core, must_exist=True)
        if destination == authoritative:
            raise MediaArtifactLifecycleError("METADATA_WOULD_OVERWRITE_AUTHORITATIVE", str(destination))
    candidate = destination if directory else destination.parent
    candidate.mkdir(parents=True, exist_ok=True)
    _assert_destination_no_reparse(destination, core)


def _assert_manifest_metadata_destination_safe(
    destination: Path,
    manifest: dict[str, Any],
    bundle: Path,
    core: ModuleType,
) -> None:
    _assert_destination_no_reparse(destination, core)
    explicit_roots = [
        _resolve_relative(bundle, value, core, must_exist=True)
        for value in ((manifest.get("scan") or {}).get("explicit_roots") or [])
    ]
    if any(_path_is_under(destination, root) for root in explicit_roots):
        raise MediaArtifactLifecycleError("METADATA_DESTINATION_INSIDE_ARTIFACT_ROOT", str(destination))
    for item in manifest.get("artifacts") or []:
        if item.get("protection_class") and destination == core.artifact_path(bundle, item):
            raise MediaArtifactLifecycleError("METADATA_WOULD_OVERWRITE_AUTHORITATIVE", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_destination_no_reparse(destination, core)


def _assert_plan_metadata_destination_safe(
    destination: Path,
    plan: dict[str, Any],
    bundle: Path,
    core: ModuleType,
) -> None:
    _assert_destination_no_reparse(destination, core)
    for item in plan.get("entries") or []:
        source = Path(str(item.get("source_path") or item.get("from") or ""))
        if source and destination == source.resolve(strict=False):
            raise MediaArtifactLifecycleError("METADATA_WOULD_OVERWRITE_ARTIFACT", str(destination))
    if _path_is_under(destination, bundle / "artifacts"):
        raise MediaArtifactLifecycleError("METADATA_DESTINATION_INSIDE_ARTIFACT_ROOT", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_destination_no_reparse(destination, core)


def _assert_destination_no_reparse(destination: Path, core: ModuleType) -> None:
    current = destination if destination.exists() else destination.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.exists():
        raise MediaArtifactLifecycleError("METADATA_DESTINATION_PARENT_MISSING", str(destination))
    lexical = destination.absolute()
    existing: list[Path] = []
    cursor = lexical if lexical.exists() else lexical.parent
    while True:
        if cursor.exists():
            existing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    try:
        for path in existing:
            if core._is_reparse(path):
                raise MediaArtifactLifecycleError("REPARSE_POINT_BLOCKED", str(path))
    except MediaArtifactLifecycleError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap_upstream(exc) from exc


def _validate_limits(max_depth: int, max_items: int) -> None:
    if not 1 <= int(max_depth) <= MAX_MAX_DEPTH:
        raise MediaArtifactLifecycleError("SCAN_DEPTH_LIMIT_INVALID", str(max_depth))
    if not 1 <= int(max_items) <= MAX_MAX_ITEMS:
        raise MediaArtifactLifecycleError("SCAN_ITEM_LIMIT_INVALID", str(max_items))


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except Exception as exc:  # noqa: BLE001
        raise MediaArtifactLifecycleError(code, f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MediaArtifactLifecycleError(code, f"{path}: object required")
    return value


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _mtime_iso(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _upstream_error_code(exc: Exception) -> str:
    return str(getattr(exc, "code", None) or getattr(exc, "error_code", None) or type(exc).__name__)


def _wrap_upstream(exc: Exception) -> MediaArtifactLifecycleError:
    return MediaArtifactLifecycleError(_upstream_error_code(exc), str(getattr(exc, "detail", None) or exc))
