from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .artifact_freshness import (
    SNAPSHOT_SCHEMA,
    build_dependency_snapshot,
    canonical_json_sha256,
    validate_dependency_snapshot,
)
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json


ATTESTATION_SCHEMA = "video_knowledge_pipeline.review_attestation.v1"
VALIDATION_SCHEMA = "video_knowledge_pipeline.review_attestation_validation.v1"
INDEX_SCHEMA = "video_knowledge_pipeline.review_attestation_index.v1"


def create_review_attestation(
    bundle_dir: str | Path,
    *,
    target: str,
    artifact_paths: Iterable[dict[str, Any] | str | Path],
    approved_by: str,
    comment: str = "",
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    _require_bundle(root)
    clean_target = _safe_id(target)
    operator = str(approved_by or "").strip()
    if not operator:
        raise ValueError("approved_by is required")
    snapshot = build_dependency_snapshot(
        root,
        subject=f"review:{clean_target}",
        inputs=artifact_paths,
        producer_schema=ATTESTATION_SCHEMA,
    )
    identity = {
        "schema": ATTESTATION_SCHEMA,
        "target": clean_target,
        "approved_by": operator,
        "approved_at": now_iso(),
        "comment": str(comment or ""),
        "dependency_snapshot": snapshot,
        "security_boundary": {
            "content_integrity_only": True,
            "cryptographic_operator_identity_claimed": False,
            "bundle_truth_replaced": False,
        },
    }
    record_sha256 = canonical_json_sha256(identity)
    attestation = {
        **identity,
        "attestation_id": f"att-{record_sha256[:20]}",
        "record_sha256": record_sha256,
    }
    attestation_path = root / "review-attestations" / f"{attestation['attestation_id']}.json"
    attestation["path"] = str(attestation_path)
    result = {
        "schema": "video_knowledge_pipeline.review_attestation_create.v1",
        "status": "created" if write else "preview",
        "attestation": attestation,
        "path": str(attestation_path),
        "write": bool(write),
    }
    if write:
        with bundle_write_lock(root, operation="review_attestation", timeout_seconds=2.0):
            if attestation_path.exists():
                existing = read_json(attestation_path)
                if existing != attestation:
                    raise ValueError(f"immutable attestation collision: {attestation_path}")
            else:
                write_json(attestation_path, attestation)
            index_path = root / "review-attestations" / "index.json"
            index = _read_index(index_path)
            index["current"][clean_target] = {
                "attestation_id": attestation["attestation_id"],
                "path": attestation_path.relative_to(root).as_posix(),
                "record_sha256": record_sha256,
                "approved_at": attestation["approved_at"],
            }
            index["updated_at"] = now_iso()
            write_json(index_path, index)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            if isinstance(manifest, dict):
                manifest["review_attestation_index"] = "review-attestations/index.json"
                write_json(manifest_path, manifest)
            result["index_path"] = str(index_path)
    return result


def validate_review_attestation(
    bundle_dir: str | Path,
    *,
    target: str = "",
    attestation_path: str | Path | None = None,
    expected_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    _require_bundle(root)
    path = _resolve_attestation_path(root, target=target, attestation_path=attestation_path)
    if path is None or not path.is_file():
        return _validation("missing", path=str(path or ""), issues=[{"key": "attestation_missing"}])
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != ATTESTATION_SCHEMA:
        return _validation("invalid", path=str(path), issues=[{"key": "invalid_attestation_schema"}])
    snapshot = value.get("dependency_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SNAPSHOT_SCHEMA:
        return _validation("invalid", path=str(path), issues=[{"key": "invalid_dependency_snapshot"}])
    identity = {
        "schema": ATTESTATION_SCHEMA,
        "target": str(value.get("target") or ""),
        "approved_by": str(value.get("approved_by") or ""),
        "approved_at": str(value.get("approved_at") or ""),
        "comment": str(value.get("comment") or ""),
        "dependency_snapshot": snapshot,
        "security_boundary": value.get("security_boundary") if isinstance(value.get("security_boundary"), dict) else {},
    }
    expected_record_hash = canonical_json_sha256(identity)
    issues: list[dict[str, Any]] = []
    if str(value.get("record_sha256") or "") != expected_record_hash:
        issues.append({"key": "record_hash_mismatch", "expected": expected_record_hash, "actual": value.get("record_sha256", "")})
    snapshot_validation = validate_dependency_snapshot(root, snapshot)
    issues.extend(snapshot_validation.get("issues") or [])
    dependency_set_matches = True
    if expected_snapshot is not None:
        actual_inputs_hash = canonical_json_sha256(snapshot.get("inputs") if isinstance(snapshot.get("inputs"), list) else [])
        expected_inputs_hash = canonical_json_sha256(
            expected_snapshot.get("inputs") if isinstance(expected_snapshot.get("inputs"), list) else []
        )
        dependency_set_matches = actual_inputs_hash == expected_inputs_hash
        if not dependency_set_matches:
            issues.append(
                {"key": "attestation_dependency_set_mismatch", "expected": expected_inputs_hash, "actual": actual_inputs_hash}
            )
    if any(issue.get("key") == "record_hash_mismatch" for issue in issues):
        status = "invalid"
    elif not dependency_set_matches:
        status = "stale"
    else:
        status = str(snapshot_validation.get("status") or "invalid")
        if status == "fresh":
            status = "valid"
    return _validation(
        status,
        path=str(path),
        target=str(value.get("target") or ""),
        attestation_id=str(value.get("attestation_id") or ""),
        record_sha256=str(value.get("record_sha256") or ""),
        snapshot_validation=snapshot_validation,
        issues=issues,
    )


def _resolve_attestation_path(
    root: Path,
    *,
    target: str,
    attestation_path: str | Path | None,
) -> Path | None:
    if attestation_path:
        path = Path(attestation_path).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        if resolved != root and not resolved.is_relative_to(root):
            raise ValueError(f"attestation path is outside bundle: {resolved}")
        return resolved
    clean_target = _safe_id(target)
    index = _read_index(root / "review-attestations" / "index.json")
    row = index.get("current", {}).get(clean_target)
    if not isinstance(row, dict) or not str(row.get("path") or ""):
        return None
    return (root / str(row["path"])).resolve()


def _read_index(path: Path) -> dict[str, Any]:
    if path.is_file():
        value = read_json(path)
        if isinstance(value, dict) and value.get("schema") == INDEX_SCHEMA and isinstance(value.get("current"), dict):
            return value
    return {"schema": INDEX_SCHEMA, "current": {}, "updated_at": now_iso()}


def _require_bundle(root: Path) -> None:
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"manifest.json not found: {root / 'manifest.json'}")


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    if not cleaned:
        raise ValueError("target is required")
    return cleaned


def _validation(status: str, **values: Any) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "status": status,
        "passed": status == "valid",
        "checked_at": now_iso(),
        **values,
    }
