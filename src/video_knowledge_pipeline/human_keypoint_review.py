from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .models import now_iso
from .smart_summary_keypoint_eval import GOLDSET_SCHEMA, normalize_claim_scope, reference_binding_status
from .storage import read_json, write_json
from .transcript import format_timestamp


def build_human_keypoint_goldset(
    bundle_dir: str | Path,
    review_payload: dict[str, Any],
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Bind explicit review selections to the Smart Summary human gold set.

    Intent: let the existing loopback review workspace create the independent
    human key-point evidence required by the Smart Summary quality gate.
    Decision: reuse ``lecture_review_notes.v1`` rows and merge only rows with an
    explicit ``human_key_point_confirmed`` flag into ``human-key-points.json``.
    Reason: candidate facts or generated summaries must never promote
    themselves into human gold; an explicit operator selection is the boundary.
    Evidence: VKP's existing review service already provides localStorage,
    bundle-revision checks, CSRF protection, a bundle write lock and atomic JSON
    replacement; the retrieval gold-set path likewise distinguishes pending
    rows from ``human_confirmed`` rows.
    Effective scope: local evaluation metadata only. This does not change the
    transcript, Timeline, summary prose, provider route or upload permissions.
    """

    root = Path(bundle_dir).expanduser().resolve()
    timeline_path = root / "timeline.json"
    manifest_path = root / "manifest.json"
    if not timeline_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"not a VKP webui bundle: {root}")
    timeline_value = read_json(timeline_path)
    manifest_value = read_json(manifest_path)
    if not isinstance(timeline_value, list) or not isinstance(manifest_value, dict):
        raise ValueError("bundle manifest/timeline schema is invalid")
    timeline = {
        _integer(row.get("index")): row
        for row in timeline_value
        if isinstance(row, dict) and _integer(row.get("index"))
    }
    reviews = _review_rows(review_payload)
    confirmed_reviews = [
        review
        for review in reviews
        if bool(review.get("human_key_point_confirmed"))
    ]
    output_path = root / "exports" / "human-key-points.json"
    if not confirmed_reviews:
        return {
            "schema": "video_knowledge_pipeline.human_key_point_writeback.v1",
            "status": "not_updated",
            "ok": True,
            "output_path": str(output_path),
            "incoming_confirmed_count": 0,
            "source_timeline_indexes": [],
            "write": bool(write),
        }
    existing = _existing_goldset(output_path)
    binding_status = reference_binding_status(existing, bundle_root=root)['status']
    rebind_prior_hash = ''
    if binding_status not in {'fresh', 'legacy_unbound'}:
        prior_indexes = {_integer(row.get('source_timeline_index'))
                         for row in existing.get('key_points', []) if isinstance(row, dict)}
        reviewed_indexes = {_integer(row.get('timeline_index') or row.get('index')) for row in confirmed_reviews}
        rebind_prior_hash = str(review_payload.get('human_key_point_rebind_prior_sha256') or '')
        if (
            binding_status != 'stale' or not prior_indexes or 0 in prior_indexes
            or not prior_indexes <= reviewed_indexes
            or rebind_prior_hash != sha256_file(output_path)
        ):
            raise ValueError('existing human reference bindings drifted; exact prior hash and explicit review of every prior point required')
    existing_points = [
        dict(row)
        for row in existing.get("key_points") or []
        if isinstance(row, dict)
    ]
    by_source_index: dict[int, dict[str, Any]] = {}
    unbound_existing: list[dict[str, Any]] = []
    for row in existing_points:
        source_index = _integer(row.get("source_timeline_index"))
        if source_index:
            by_source_index[source_index] = row
        else:
            unbound_existing.append(row)

    updates: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for row_number, review in enumerate(confirmed_reviews, start=1):
        source_index = _integer(review.get("timeline_index") or review.get("index"))
        if not source_index or source_index not in timeline:
            raise ValueError(
                f"human key point row {row_number} has no bound timeline item"
            )
        if source_index in seen_indexes:
            raise ValueError(
                f"duplicate human key point for timeline index {source_index}"
            )
        seen_indexes.add(source_index)
        text = str(review.get("human_key_point_text") or "").strip()
        if not text:
            raise ValueError(
                f"human key point row {row_number} requires human_key_point_text"
            )
        timeline_row = timeline[source_index]
        aliases = _string_list(review.get("human_key_point_aliases"), split_text=True)
        evidence_ids = _string_list(review.get("human_key_point_evidence_ids"))
        if not evidence_ids:
            evidence_ids = _string_list(timeline_row.get("source_segment_ids"))
        evidence_ids = _dedupe([f"timeline:{source_index}", *evidence_ids])
        time_range = str(review.get("time_range") or "").strip() or _time_range(
            timeline_row
        )
        prior = by_source_index.get(source_index, {})
        stable_id = str(prior.get("id") or "").strip() or _stable_id(
            source_index, text
        )
        point = {
            "id": stable_id,
            "text": text,
            "aliases": aliases,
            "time_range": time_range,
            "evidence_ids": evidence_ids,
            "source_kind": "human_confirmed",
            "review_status": "human_confirmed",
            "source_timeline_index": source_index,
            "reviewed_at": str(review.get("reviewed_at") or now_iso()),
        }
        if 'human_key_point_claim_scope' in review or 'claim_scope' in prior:
            point['claim_scope'] = normalize_claim_scope(
                review.get('human_key_point_claim_scope', prior.get('claim_scope', {})))
        by_source_index[source_index] = point
        updates.append(point)

    merged = [
        *unbound_existing,
        *[by_source_index[index] for index in sorted(by_source_index)],
    ]
    payload = {
        "schema": GOLDSET_SCHEMA,
        "bundle_dir": str(root),
        "review_source": "lecture_review_notes.v1",
        "key_points": merged,
        "human_confirmed_count": len(merged),
        "source_bindings": _source_bindings(
            root,
            manifest=manifest_value,
            timeline_path=timeline_path,
        ),
        "updated_at": now_iso(),
    }
    if rebind_prior_hash:
        payload['reference_rebind'] = {
            'prior_goldset_sha256': rebind_prior_hash,
            'prior_snapshot': f'exports/human-key-points-history/{rebind_prior_hash}.json',
            'all_prior_points_reviewed': True,
        }
    status = "ready_to_write" if updates else "not_updated"
    result = {
        "schema": "video_knowledge_pipeline.human_key_point_writeback.v1",
        "status": status,
        "ok": True,
        "output_path": str(output_path),
        "incoming_confirmed_count": len(updates),
        "human_confirmed_count": len(merged),
        "source_timeline_indexes": [
            row["source_timeline_index"] for row in updates
        ],
        "write": bool(write),
        "payload": payload,
    }
    if write and updates:
        if rebind_prior_hash:
            from .courseware_material import safe_path
            history_path = safe_path(root, payload['reference_rebind']['prior_snapshot'], exists=False)
            if history_path.exists():
                if read_json(history_path) != existing:
                    raise ValueError('human reference rebind history drift')
            else:
                write_json(history_path, existing)
        write_json(output_path, payload)
        manifest_value["human_key_points_json"] = (
            "exports/human-key-points.json"
        )
        manifest_value["human_key_points_summary"] = {
            "status": "human_confirmed",
            "count": len(merged),
            "updated_at": payload["updated_at"],
        }
        write_json(manifest_path, manifest_value)
        result["status"] = "written"
    return result


def _existing_goldset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != GOLDSET_SCHEMA:
        raise ValueError(
            f"existing human key points must use schema {GOLDSET_SCHEMA}"
        )
    return value


def _source_bindings(
    root: Path,
    *,
    manifest: dict[str, Any],
    timeline_path: Path,
) -> list[dict[str, str]]:
    candidates = [timeline_path]
    for key in (
        "source_arbitrated_transcript_json",
        "corrected_transcript_json",
        "normalized_transcript_json",
    ):
        value = str(manifest.get(key) or "").strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            candidates.append(path.resolve())
            break
    rows: list[dict[str, str]] = []
    for path in candidates:
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("review payload must be a JSON object")
    for key in ("reviews", "items", "notes"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _string_list(value: Any, *, split_text: bool = False) -> list[str]:
    if isinstance(value, str):
        values = value.split("|") if split_text else [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return _dedupe([str(item).strip() for item in values if str(item).strip()])


def _time_range(row: dict[str, Any]) -> str:
    start = _number(row.get("start"))
    end = max(start, _number(row.get("end")))
    return f"{format_timestamp(start)} - {format_timestamp(end)}"


def _stable_id(source_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{source_index}:{text}".encode("utf-8")
    ).hexdigest()[:12]
    return f"human-kp-{source_index:04d}-{digest}"


def _integer(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
