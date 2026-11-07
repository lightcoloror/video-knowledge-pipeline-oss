from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, TypeVar

from .file_hash import sha256_file
from .storage import read_json


CONTENT_SCOPE_SCHEMA_V1 = "video_knowledge_pipeline.content_scope.v1"
CONTENT_SCOPE_SCHEMA = "video_knowledge_pipeline.content_scope.v2"
SUPPORTED_CONTENT_SCOPE_SCHEMAS = {CONTENT_SCOPE_SCHEMA_V1, CONTENT_SCOPE_SCHEMA}
CONTENT_SCOPE_FILENAME = "content-scope.json"
SUPPORTED_PROJECTIONS = {"continuous_full_body", "full_transcript", "smart_summary"}

_T = TypeVar("_T")


def load_content_scope(
    bundle_dir: str | Path,
    *,
    transcript_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load an operator-confirmed projection scope without changing source evidence.

    The scope controls reader projections only. Raw/intermediate transcripts are
    never filtered, and an explicit but malformed or stale scope fails closed so
    an export cannot silently re-introduce excluded warm-up material.
    """

    root = Path(bundle_dir).expanduser().resolve()
    path = root / CONTENT_SCOPE_FILENAME
    if not path.is_file():
        return {
            "schema": CONTENT_SCOPE_SCHEMA,
            "status": "not_configured",
            "active": False,
            "path": str(path),
            "excluded_intervals": [],
        }

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("content-scope.json must be a JSON object")
    schema = str(payload.get("schema") or "")
    if schema not in SUPPORTED_CONTENT_SCOPE_SCHEMAS:
        raise ValueError(
            "content-scope schema must be one of "
            f"{sorted(SUPPORTED_CONTENT_SCOPE_SCHEMAS)}"
        )
    if str(payload.get("status") or "") != "confirmed":
        raise ValueError("content-scope status must be confirmed before reader projections use it")

    basis_source = _bundle_file(root, payload.get("transcript_source"), field="transcript_source")
    expected_sha256 = str(payload.get("transcript_source_sha256") or "").strip().lower()
    if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
        raise ValueError("content-scope transcript_source_sha256 must be a lowercase SHA-256 digest")
    if not basis_source.is_file():
        raise ValueError(f"content-scope transcript_source is missing: {basis_source}")
    actual_sha256 = sha256_file(basis_source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "content-scope transcript_source_sha256 does not match the bound transcript; "
            "reconfirm the time boundary before exporting"
        )

    intervals = _validated_intervals(payload.get("excluded_intervals"), schema=schema)
    selected_path = Path(transcript_path).expanduser().resolve() if transcript_path else None
    return {
        **payload,
        "schema": schema,
        "status": "confirmed",
        "active": bool(intervals),
        "path": str(path),
        "content_scope_sha256": sha256_file(path),
        "transcript_source": str(basis_source),
        "transcript_source_sha256": actual_sha256,
        "selected_transcript_path": str(selected_path) if selected_path else "",
        "excluded_intervals": intervals,
    }


def apply_content_scope_to_cues(
    cues: Iterable[_T],
    scope: dict[str, Any],
    *,
    projection: str,
) -> tuple[list[_T], dict[str, Any]]:
    return _apply_content_scope(cues, scope, projection=projection, kind="transcript_cue")


def apply_content_scope_to_timeline(
    rows: Iterable[_T],
    scope: dict[str, Any],
    *,
    projection: str,
) -> tuple[list[_T], dict[str, Any]]:
    return _apply_content_scope(rows, scope, projection=projection, kind="timeline_item")


def content_scope_projection_summary(
    scope: dict[str, Any],
    *,
    projection: str,
    cue_report: dict[str, Any],
    timeline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intervals = _projection_intervals(scope, projection)
    timeline = timeline_report or {}
    projection_audit = [
        event
        for event in (cue_report.get("events") or [])
        if isinstance(event, dict)
        and (
            str(event.get("disposition") or "") != "retained_semantic"
            or str(event.get("action") or "") != "retain"
        )
    ]
    return {
        "schema": str(scope.get("schema") or CONTENT_SCOPE_SCHEMA),
        "status": str(scope.get("status") or "not_configured"),
        "active": bool(
            intervals
            or int(cue_report.get("projection_omitted_count") or 0)
            or int((cue_report.get("disposition_counts") or {}).get("review_gap") or 0)
        ),
        "projection": projection,
        "path": str(scope.get("path") or ""),
        "content_scope_sha256": str(scope.get("content_scope_sha256") or ""),
        "transcript_source": str(scope.get("transcript_source") or ""),
        "transcript_source_sha256": str(scope.get("transcript_source_sha256") or ""),
        "excluded_intervals": intervals,
        "excluded_transcript_cue_count": int(cue_report.get("excluded_count") or 0),
        "retained_transcript_cue_count": int(cue_report.get("retained_count") or 0),
        "partial_transcript_overlap_count": int(cue_report.get("partial_overlap_count") or 0),
        "excluded_timeline_item_count": int(timeline.get("excluded_count") or 0),
        "retained_timeline_item_count": int(timeline.get("retained_count") or 0),
        "partial_timeline_overlap_count": int(timeline.get("partial_overlap_count") or 0),
        "disposition_counts": dict(cue_report.get("disposition_counts") or {}),
        "projection_event_count": int(cue_report.get("event_count") or 0),
        "projection_omitted_cue_count": int(cue_report.get("projection_omitted_count") or 0),
        "replacement_count": int(cue_report.get("replacement_count") or 0),
        "projection_audit": projection_audit[:500],
        "projection_audit_truncated": len(projection_audit) > 500,
        "retention_boundary": {
            "intermediate_evidence_preserved": True,
            "raw_transcript_preserved": True,
            "full_transcript_preserved": not (
                projection == "full_transcript" and bool(intervals)
            ),
            "source_transcript_modified": False,
        },
    }


def _apply_content_scope(
    values: Iterable[_T],
    scope: dict[str, Any],
    *,
    projection: str,
    kind: str,
) -> tuple[list[_T], dict[str, Any]]:
    intervals = _projection_intervals(scope, projection)
    items = list(values)
    if not intervals:
        return items, {
            "projection": projection,
            "kind": kind,
            "excluded_count": 0,
            "retained_count": len(items),
            "partial_overlap_count": 0,
            "excluded_ids": [],
            "partial_overlap_ids": [],
            "excluded_items": [],
            "partial_overlap_items": [],
        }

    kept: list[_T] = []
    excluded_ids: list[str] = []
    partial_ids: list[str] = []
    excluded_items: list[dict[str, Any]] = []
    partial_items: list[dict[str, Any]] = []
    for position, value in enumerate(items, start=1):
        start, end = _bounds(value)
        identity = _identity(value, fallback=f"{kind}-{position:06d}")
        containing = [row for row in intervals if start >= row["start"] and end <= row["end"]]
        if containing:
            excluded_ids.append(identity)
            interval = containing[0]
            excluded_items.append(
                {
                    "id": identity,
                    "start": start,
                    "end": end,
                    "interval_id": interval.get("interval_id") or "",
                    "category": interval.get("category") or interval.get("reason") or "",
                    "reason": interval.get("reason") or "",
                    "disposition": "excluded_confirmed",
                }
            )
            continue
        overlapping = [row for row in intervals if _overlaps(start, end, row["start"], row["end"])]
        if overlapping:
            partial_ids.append(identity)
            interval = overlapping[0]
            partial_items.append(
                {
                    "id": identity,
                    "start": start,
                    "end": end,
                    "interval_id": interval.get("interval_id") or "",
                    "category": interval.get("category") or interval.get("reason") or "",
                    "reason": "partial_scope_overlap_requires_review",
                    "disposition": "review_gap",
                }
            )
        kept.append(value)
    return kept, {
        "projection": projection,
        "kind": kind,
        "excluded_count": len(excluded_ids),
        "retained_count": len(kept),
        "partial_overlap_count": len(partial_ids),
        "excluded_ids": excluded_ids[:500],
        "partial_overlap_ids": partial_ids[:500],
        "excluded_items": excluded_items[:500],
        "partial_overlap_items": partial_items[:500],
    }


def content_scope_replacements(
    scope: dict[str, Any],
    *,
    projection: str,
) -> list[dict[str, Any]]:
    """Return operator-confirmed editorial replacements for one projection."""

    rows: list[dict[str, Any]] = []
    for interval in _projection_intervals(scope, projection):
        replacements = interval.get("replacement_by_projection")
        if not isinstance(replacements, dict):
            continue
        text = str(replacements.get(projection) or "").strip()
        if not text:
            continue
        rows.append(
            {
                "interval_id": str(interval.get("interval_id") or ""),
                "start": float(interval["start"]),
                "end": float(interval["end"]),
                "text": text,
                "category": str(interval.get("category") or interval.get("reason") or ""),
                "reason": str(interval.get("reason") or ""),
                "decision": "operator_confirmed",
            }
        )
    return rows


def _validated_intervals(value: Any, *, schema: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("content-scope excluded_intervals must be a non-empty array")
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"content-scope interval {position} must be an object")
        start = _finite_seconds(raw.get("start"), field=f"excluded_intervals[{position}].start")
        end = _finite_seconds(raw.get("end"), field=f"excluded_intervals[{position}].end")
        if start < 0 or end <= start:
            raise ValueError(f"content-scope interval {position} must satisfy 0 <= start < end")
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"content-scope interval {position} requires reason")
        if str(raw.get("decision") or "") != "operator_confirmed":
            raise ValueError(f"content-scope interval {position} must be operator_confirmed")
        category = str(raw.get("category") or raw.get("reason") or "").strip()
        if schema == CONTENT_SCOPE_SCHEMA and not str(raw.get("category") or "").strip():
            raise ValueError(f"content-scope interval {position} requires category in v2")
        disposition = str(raw.get("disposition") or "excluded_confirmed").strip()
        if disposition != "excluded_confirmed":
            raise ValueError(
                f"content-scope interval {position} disposition must be excluded_confirmed"
            )
        exclude_from = raw.get("exclude_from")
        if not isinstance(exclude_from, list) or not exclude_from:
            raise ValueError(f"content-scope interval {position} requires exclude_from")
        projections = [str(item) for item in exclude_from if str(item)]
        unsupported = sorted(set(projections) - SUPPORTED_PROJECTIONS)
        if unsupported:
            raise ValueError(f"content-scope interval {position} has unsupported projections: {unsupported}")
        replacements = raw.get("replacement_by_projection") or {}
        if not isinstance(replacements, dict):
            raise ValueError(
                f"content-scope interval {position} replacement_by_projection must be an object"
            )
        unsupported_replacements = sorted(set(str(key) for key in replacements) - SUPPORTED_PROJECTIONS)
        if unsupported_replacements:
            raise ValueError(
                f"content-scope interval {position} has unsupported replacement projections: "
                f"{unsupported_replacements}"
            )
        normalized_replacements: dict[str, str] = {}
        for key, replacement in replacements.items():
            replacement_text = str(replacement or "").strip()
            if len(replacement_text) > 1000:
                raise ValueError(
                    f"content-scope interval {position} replacement for {key} exceeds 1000 characters"
                )
            if replacement_text:
                normalized_replacements[str(key)] = replacement_text
        rows.append(
            {
                **raw,
                "interval_id": str(raw.get("interval_id") or f"interval-{position:04d}"),
                "start": start,
                "end": end,
                "reason": reason,
                "category": category,
                "disposition": disposition,
                "decision": "operator_confirmed",
                "exclude_from": projections,
                "retain_in": [str(item) for item in (raw.get("retain_in") or []) if str(item)],
                "replacement_by_projection": normalized_replacements,
            }
        )
    return sorted(rows, key=lambda row: (row["start"], row["end"], row["reason"]))


def _projection_intervals(scope: dict[str, Any], projection: str) -> list[dict[str, Any]]:
    if projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(f"unsupported content-scope projection: {projection}")
    rows = scope.get("excluded_intervals") if isinstance(scope.get("excluded_intervals"), list) else []
    return [row for row in rows if isinstance(row, dict) and projection in (row.get("exclude_from") or [])]


def _bundle_file(root: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"content-scope {field} is required")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"content-scope {field} must stay inside the Bundle") from exc
    return resolved


def _bounds(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        start = _finite_seconds(value.get("start"), field="item.start", default=0.0)
        end = _finite_seconds(value.get("end"), field="item.end", default=start)
    else:
        start = _finite_seconds(getattr(value, "start", 0.0), field="item.start", default=0.0)
        end = _finite_seconds(getattr(value, "end", start), field="item.end", default=start)
    return start, max(start, end)


def _identity(value: Any, *, fallback: str) -> str:
    if isinstance(value, dict):
        for key in ("segment_id", "id", "index"):
            if value.get(key) is not None and str(value.get(key)).strip():
                return str(value.get(key))
    else:
        for key in ("segment_id", "id", "index"):
            candidate = getattr(value, key, None)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
    return fallback


def _finite_seconds(value: Any, *, field: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"content-scope {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"content-scope {field} must be finite")
    return result


def _overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    if end == start:
        return other_start <= start < other_end
    return start < other_end and end > other_start
