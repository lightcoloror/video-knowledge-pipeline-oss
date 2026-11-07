from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .artifact_validation import artifact_evidence
from .canonical_json import canonical_json_sha256
from .models import now_iso


SNAPSHOT_SCHEMA = "video_knowledge_pipeline.artifact_dependency_snapshot.v1"
VALIDATION_SCHEMA = "video_knowledge_pipeline.artifact_dependency_validation.v1"



def build_dependency_snapshot(
    bundle_dir: str | Path,
    *,
    subject: str,
    inputs: Iterable[dict[str, Any] | str | Path],
    source_run_id: str = "",
    producer_schema: str = "",
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    rows = [_input_reference(root, item) for item in inputs]
    rows.sort(key=lambda row: (str(row.get("role") or ""), str(row.get("path") or "")))
    identity = {
        "schema": SNAPSHOT_SCHEMA,
        "subject": str(subject or "").strip(),
        "inputs": rows,
        "source_run_id": str(source_run_id or ""),
        "producer_schema": str(producer_schema or ""),
    }
    if not identity["subject"]:
        raise ValueError("dependency snapshot subject is required")
    return {
        **identity,
        "snapshot_sha256": canonical_json_sha256(identity),
        "created_at": now_iso(),
    }


def validate_dependency_snapshot(
    bundle_dir: str | Path,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SNAPSHOT_SCHEMA:
        return _validation("invalid", issues=[{"key": "invalid_schema"}])
    supplied_hash = str(snapshot.get("snapshot_sha256") or "")
    identity = {
        "schema": SNAPSHOT_SCHEMA,
        "subject": str(snapshot.get("subject") or ""),
        "inputs": snapshot.get("inputs") if isinstance(snapshot.get("inputs"), list) else [],
        "source_run_id": str(snapshot.get("source_run_id") or ""),
        "producer_schema": str(snapshot.get("producer_schema") or ""),
    }
    expected_hash = canonical_json_sha256(identity)
    if supplied_hash != expected_hash:
        issues.append(
            {
                "key": "snapshot_hash_mismatch",
                "expected": expected_hash,
                "actual": supplied_hash,
            }
        )
    current_inputs: list[dict[str, Any]] = []
    missing = False
    stale = False
    for recorded in identity["inputs"]:
        if not isinstance(recorded, dict):
            issues.append({"key": "invalid_input_reference"})
            continue
        try:
            path = _resolve_bundle_path(root, str(recorded.get("path") or ""))
        except ValueError as exc:
            issues.append({"key": "invalid_input_path", "path": recorded.get("path"), "detail": str(exc)})
            continue
        if not path.is_file():
            missing = True
            issues.append({"key": "input_missing", "path": str(recorded.get("path") or ""), "role": recorded.get("role", "")})
            continue
        current = _reference_for_path(root, path, role=str(recorded.get("role") or "artifact"))
        current_inputs.append(current)
        changed = [
            key
            for key in ("bytes", "sha256", "canonical_json_sha256")
            if str(recorded.get(key, "")) != str(current.get(key, ""))
        ]
        if changed:
            stale = True
            issues.append(
                {
                    "key": "input_changed",
                    "path": str(recorded.get("path") or ""),
                    "role": recorded.get("role", ""),
                    "changed_fields": changed,
                    "expected": {key: recorded.get(key, "") for key in changed},
                    "actual": {key: current.get(key, "") for key in changed},
                }
            )
    if any(issue.get("key") in {"invalid_schema", "snapshot_hash_mismatch", "invalid_input_reference", "invalid_input_path"} for issue in issues):
        status = "invalid"
    elif missing:
        status = "missing"
    elif stale:
        status = "stale"
    else:
        status = "fresh"
    return _validation(
        status,
        snapshot_sha256=supplied_hash,
        current_inputs=current_inputs,
        issues=issues,
    )


def _input_reference(root: Path, item: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(item, dict):
        raw_path = item.get("path") or item.get("artifact_path")
        role = str(item.get("role") or item.get("key") or "artifact")
    else:
        raw_path = item
        role = "artifact"
    path = _resolve_bundle_path(root, str(raw_path or ""))
    if not path.is_file():
        raise FileNotFoundError(f"dependency artifact does not exist: {path}")
    return _reference_for_path(root, path, role=role)


def _reference_for_path(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    evidence = artifact_evidence(path)
    row: dict[str, Any] = {
        "role": str(role or "artifact"),
        "path": path.relative_to(root).as_posix(),
        "bytes": int(evidence["bytes"]),
        "sha256": str(evidence["sha256"]),
    }
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            value = None
        if value is not None:
            row["canonical_json_sha256"] = canonical_json_sha256(value)
    return row


def _resolve_bundle_path(root: Path, value: str) -> Path:
    if not value.strip():
        raise ValueError("dependency artifact path is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(f"dependency artifact is outside bundle: {resolved}")
    return resolved


def _validation(
    status: str,
    *,
    snapshot_sha256: str = "",
    current_inputs: list[dict[str, Any]] | None = None,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "status": status,
        "passed": status == "fresh",
        "snapshot_sha256": snapshot_sha256,
        "current_inputs": current_inputs or [],
        "issues": issues or [],
        "checked_at": now_iso(),
    }
