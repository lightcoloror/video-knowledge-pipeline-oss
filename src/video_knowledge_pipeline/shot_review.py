from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .technical_shot_detection import (
    BOUNDARY_SCHEMA,
    SCHEMA as TECHNICAL_SHOT_SCHEMA,
    load_verified_technical_shots,
)
from .transcript import format_timestamp


NOTES_SCHEMA = "video_knowledge_pipeline.shot_review_notes.v1"
APPLIED_SCHEMA = "video_knowledge_pipeline.shot_review_applied.v1"
REVIEWED_BOUNDARIES_PATH = "exports/technical-shot-boundaries.reviewed.json"
APPLIED_PATH = "exports/shot-review-applied.json"
TEMPLATE_PATH = "exports/shot-review-notes.template.json"
ALLOWED_FIELDS = {"shot_type", "camera_movement", "composition", "lighting"}


def build_shot_review_template(
    bundle_dir: str | Path,
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Snapshot current derived shot artifacts for a browser-local draft."""

    root = Path(bundle_dir).expanduser().resolve()
    shots, provenance = load_verified_technical_shots(root)
    if not shots:
        return {
            "schema": NOTES_SCHEMA,
            "status": "blocked_missing_technical_shots",
            "ok": False,
            "bundle_dir": str(root),
            "shots": [],
            "field_corrections": [],
            "source_artifacts": [],
            "source_revision": "",
            "operator_boundary": _operator_boundary(),
        }
    sources = [_source_from_provenance(root, provenance)]
    facts_path = root / "exports" / "shot-facts.json"
    facts = _read_object(facts_path)
    if facts_path.is_file():
        sources.append(_source(root, facts_path, kind="shot_facts"))
    fusion_path = root / "exports" / "technical-shot-boundary-fusion.json"
    fusion = _read_object(fusion_path)
    if fusion_path.is_file():
        sources.append(_source(root, fusion_path, kind="boundary_fusion"))
    facts_by_id = {
        str(row.get("shot_id") or ""): row
        for row in facts.get("shots") or []
        if isinstance(row, dict)
    }
    template = {
        "schema": NOTES_SCHEMA,
        "status": "draft",
        "ok": True,
        "bundle_dir": str(root),
        "review_status": "draft",
        "review_id": "",
        "reviewed_at": "",
        "source_artifacts": sources,
        "source_revision": _source_revision(sources),
        "media": _media_binding(root),
        "shots": [
            {
                "shot_id": str(row.get("shot_id") or f"technical-shot-{index:04d}"),
                "index": index,
                "start": float(row.get("start") or 0.0),
                "end": float(row.get("end") or 0.0),
                "source_shot_ids": [str(row.get("shot_id") or "")],
                "fields": {
                    key: (facts_by_id.get(str(row.get("shot_id") or ""), {}).get("fields") or {}).get(key)
                    for key in sorted(ALLOWED_FIELDS)
                },
            }
            for index, row in enumerate(shots, start=1)
        ],
        "field_corrections": [],
        "fusion_candidates": list(fusion.get("candidates") or []),
        "operator_boundary": _operator_boundary(),
    }
    if write:
        write_json(root / TEMPLATE_PATH, template)
    return template


def apply_shot_review_notes(
    bundle_dir: str | Path,
    review_notes: str | Path | dict[str, Any],
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Install one explicit human review as a derived, freshness-bound view."""

    root = Path(bundle_dir).expanduser().resolve()
    notes = _load_notes(review_notes)
    if notes.get("schema") != NOTES_SCHEMA:
        raise ValueError(f"review notes schema must be {NOTES_SCHEMA}")
    if str(notes.get("review_status") or "") != "human_confirmed":
        raise ValueError("review_status must be human_confirmed before formal apply")
    source_artifacts = notes.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise ValueError("source_artifacts must bind the reviewed inputs")
    _validate_sources(root, source_artifacts)
    if str(notes.get("source_revision") or "") != _source_revision(source_artifacts):
        raise ValueError("source_revision does not match source_artifacts")
    shots = _validate_shots(notes.get("shots"))
    review_id = str(notes.get("review_id") or "").strip()
    if not review_id:
        raise ValueError("review_id is required")
    corrections = _validate_field_corrections(notes.get("field_corrections"), shots)
    reviewed_at = str(notes.get("reviewed_at") or now_iso())
    media = notes.get("media") if isinstance(notes.get("media"), dict) else {}
    current_media = _media_binding(root)
    if media.get("sha256") and current_media.get("sha256") != media.get("sha256"):
        raise ValueError("reviewed media SHA-256 no longer matches the bundle")
    reviewed_shots = [
        {
            "shot_id": f"technical-shot-{index:04d}",
            "index": index,
            "start": round(row["start"], 6),
            "end": round(row["end"], 6),
            "duration": round(row["end"] - row["start"], 6),
            "start_time": format_timestamp(row["start"]),
            "end_time": format_timestamp(row["end"]),
            "boundary_kind": "technical_shot",
            "source_shot_ids": list(row["source_shot_ids"]),
            "human_confirmed": True,
        }
        for index, row in enumerate(shots, start=1)
    ]
    boundaries = [
        {
            "schema": BOUNDARY_SCHEMA,
            "boundary_id": f"technical-boundary-{index:04d}",
            "seconds": row["start"],
            "time": row["start_time"],
            "backend": "human_review",
            "candidate_only": False,
            "human_confirmed": True,
        }
        for index, row in enumerate(reviewed_shots[1:], start=1)
    ]
    reviewed = {
        "schema": TECHNICAL_SHOT_SCHEMA,
        "status": "completed",
        "ok": True,
        "bundle_dir": str(root),
        "media": current_media,
        "boundary_kind": "technical_shot",
        "backend": "human_review",
        "strict": True,
        "shot_count": len(reviewed_shots),
        "boundary_count": len(boundaries),
        "shots": reviewed_shots,
        "boundaries": boundaries,
        "source_artifacts": source_artifacts,
        "source_revision": notes["source_revision"],
        "review_id": review_id,
        "reviewed_at": reviewed_at,
        "human_confirmed": True,
        "candidate_only": False,
        "timeline_mutated": False,
        "operator_boundary": _operator_boundary(),
        "updated_at": now_iso(),
    }
    applied = {
        "schema": APPLIED_SCHEMA,
        "status": "completed",
        "ok": True,
        "bundle_dir": str(root),
        "review_id": review_id,
        "reviewed_at": reviewed_at,
        "source_artifacts": source_artifacts,
        "source_revision": notes["source_revision"],
        "reviewed_boundaries": REVIEWED_BOUNDARIES_PATH,
        "field_corrections": corrections,
        "next_action": "rerun shot-language-analysis so corrected boundaries and fields become one fresh shot_facts.v1 projection",
        "timeline_mutated": False,
        "media_mutated": False,
        "operator_boundary": _operator_boundary(),
        "updated_at": now_iso(),
    }
    if write:
        manifest_path = root / "manifest.json"
        with bundle_write_lock(root, operation="shot_review_apply", timeout_seconds=1.0):
            write_json(root / REVIEWED_BOUNDARIES_PATH, reviewed)
            write_json(root / APPLIED_PATH, applied)
            manifest = _read_object(manifest_path)
            manifest["technical_shot_boundaries_reviewed_json"] = REVIEWED_BOUNDARIES_PATH
            manifest["shot_review_applied_json"] = APPLIED_PATH
            manifest["shot_review_id"] = review_id
            write_json(manifest_path, manifest)
    return applied


def shot_review_status(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    applied_path = root / APPLIED_PATH
    reviewed_path = root / REVIEWED_BOUNDARIES_PATH
    if not applied_path.is_file() or not reviewed_path.is_file():
        return {"status": "not_applied", "ok": False}
    applied = _read_object(applied_path)
    try:
        _validate_sources(root, applied.get("source_artifacts") or [])
    except ValueError as exc:
        return {"status": "stale", "ok": False, "reason": str(exc)}
    return {
        "status": "active",
        "ok": True,
        "review_id": applied.get("review_id"),
        "reviewed_boundaries": str(reviewed_path),
        "field_correction_count": len(applied.get("field_corrections") or []),
    }


def _load_notes(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    path = Path(value).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("shot review notes must be a JSON object")
    return payload


def _validate_sources(root: Path, rows: list[Any]) -> None:
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source_artifacts entries must be objects")
        raw = str(row.get("path") or "").strip()
        expected = str(row.get("sha256") or "").strip().lower()
        path = (root / raw).resolve() if raw and not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"review source is outside the bundle: {path}") from exc
        if not path.is_file():
            raise ValueError(f"review source is missing: {path}")
        if not expected or sha256_file(path).lower() != expected:
            raise ValueError(f"review source changed: {path}")


def _validate_shots(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("shots must contain at least one reviewed shot")
    result: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, row in enumerate(value, start=1):
        if not isinstance(row, dict):
            raise ValueError("reviewed shots must be objects")
        start, end = float(row.get("start")), float(row.get("end"))
        if start < 0 or end <= start:
            raise ValueError(f"reviewed shot {index} has an invalid range")
        if start < previous_end - 0.001:
            raise ValueError("reviewed shots must be ordered and non-overlapping")
        source_ids = [str(item) for item in row.get("source_shot_ids") or [] if str(item)]
        result.append({"start": start, "end": end, "source_shot_ids": source_ids})
        previous_end = end
    return result


def _validate_field_corrections(value: Any, shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    valid_ids = {f"technical-shot-{index:04d}" for index in range(1, len(shots) + 1)}
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("field_corrections entries must be objects")
        shot_id = str(row.get("shot_id") or "")
        field = str(row.get("field") or "")
        if shot_id not in valid_ids:
            raise ValueError(f"field correction references unknown shot: {shot_id}")
        if field not in ALLOWED_FIELDS:
            raise ValueError(f"unsupported corrected shot field: {field}")
        value_out = row.get("value")
        if value_out in (None, "", [], {}):
            raise ValueError("field correction value must not be empty")
        key = (shot_id, field)
        if key in seen:
            raise ValueError(f"duplicate field correction: {shot_id}/{field}")
        seen.add(key)
        result.append({"shot_id": shot_id, "field": field, "value": value_out})
    return result


def _source_from_provenance(root: Path, value: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(value.get("path") or "")).resolve()
    if not path.is_file():
        raise ValueError("technical-shot provenance has no readable source artifact")
    return _source(root, path, kind="technical_shot_boundaries")


def _source(root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"shot-review source must be inside the bundle: {resolved}") from exc
    return {"kind": kind, "path": relative, "sha256": sha256_file(resolved)}


def _source_revision(rows: list[Any]) -> str:
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _media_binding(root: Path) -> dict[str, Any]:
    manifest = _read_object(root / "manifest.json")
    raw = str(manifest.get("media_path") or manifest.get("source_path") or "").strip()
    if not raw:
        return {"status": "unavailable", "path": "", "sha256": ""}
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        return {"status": "missing", "path": str(path), "sha256": ""}
    return {"status": "bound", "path": str(path), "sha256": sha256_file(path)}


def _operator_boundary() -> dict[str, Any]:
    return {
        "local_review_only": True,
        "draft_auto_save_is_not_formal_apply": True,
        "formal_apply_requires_human_confirmed": True,
        "original_evidence_mutated": False,
        "timeline_mutated": False,
        "media_mutated": False,
        "no_network_call": True,
    }


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}
