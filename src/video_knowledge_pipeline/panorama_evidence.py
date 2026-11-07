from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .panorama_session import CANONICAL_DIRECTIONS


PANORAMA_COVERAGE_SCHEMA = "video_knowledge_pipeline.panorama_coverage.v1"


def aggregate_panorama_evidence(
    item: Mapping[str, Any],
    *,
    verified_entity_pairs: set[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Aggregate spherical evidence without promoting partial views to global facts."""

    value = item.get("panorama_visual_understanding")
    sessions = [dict(row) for row in value.get("sessions") or [] if isinstance(row, Mapping)] if isinstance(value, Mapping) else []
    observed: set[str] = set()
    confirmed = inferred = unavailable = 0
    review_gaps: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    complete_sessions = 0
    for session in sessions:
        rows = [dict(row) for row in session.get("viewport_observations") or [] if isinstance(row, Mapping)]
        directions = {
            str(row.get("direction") or "")
            for row in rows
            if row.get("observed") is True and str(row.get("direction") or "") in CANONICAL_DIRECTIONS
        }
        observed.update(directions)
        observations.extend(rows)
        if directions == set(CANONICAL_DIRECTIONS):
            complete_sessions += 1
        level = str(session.get("evidence_level") or "not_seen").strip().lower()
        if level == "confirmed":
            confirmed += 1
        elif level == "inferred":
            inferred += 1
        elif level == "unavailable":
            unavailable += 1
    missing = [direction for direction in CANONICAL_DIRECTIONS if direction not in observed]
    if missing:
        review_gaps.append({"key": "panorama_faces_missing", "missing_directions": missing})
    missing_poles = [direction for direction in ("up", "down") if direction in missing]
    if missing_poles:
        review_gaps.append({"key": "panorama_poles_missing", "missing_directions": missing_poles})

    temporal = _temporal_coverage(item)
    if temporal["status"] == "incomplete":
        review_gaps.append(
            {
                "key": "panorama_temporal_directions_missing",
                "missing_directions": list(temporal["missing_directions"]),
            }
        )
    geometric_complete = bool(sessions) and not missing and complete_sessions == len(sessions)
    semantic_level = (
        "confirmed"
        if confirmed and not inferred and not unavailable and confirmed == len(sessions)
        else "inferred"
        if confirmed or inferred
        else "unavailable"
        if unavailable
        else "not_seen"
    )
    return {
        "schema": "video_knowledge_pipeline.panorama_evidence_state.v1",
        "seen": bool(sessions),
        "geometric": {
            "status": "complete" if geometric_complete else "incomplete" if sessions else "not_seen",
            "session_count": len(sessions),
            "complete_session_count": complete_sessions,
            "covered_directions": [direction for direction in CANONICAL_DIRECTIONS if direction in observed],
            "missing_directions": missing,
            "poles_covered": not missing_poles,
        },
        "semantic": {
            "status": semantic_level,
            "confirmed_sessions": confirmed,
            "inferred_sessions": inferred,
            "unavailable_sessions": unavailable,
            "not_seen_sessions": 0 if sessions else 1,
        },
        "temporal": temporal,
        "cross_view_entities": _cross_view_entities(observations, verified_entity_pairs or set()),
        "review_gaps": review_gaps,
        "global_claim_limit": "facts are bounded to observed viewports and timestamps; segment-global absence is not established",
    }


def panorama_claim_scope(
    evidence: Mapping[str, Any],
    *,
    claim_type: str,
    requested_scope: str,
) -> dict[str, Any]:
    geometric = evidence.get("geometric") if isinstance(evidence.get("geometric"), Mapping) else {}
    semantic = evidence.get("semantic") if isinstance(evidence.get("semantic"), Mapping) else {}
    complete = geometric.get("status") == "complete" and semantic.get("status") == "confirmed"
    if str(claim_type) == "negative":
        allowed = bool(complete and requested_scope == "observed_timestamp")
        reason = (
            "negative claim is bounded to a fully observed six-view timestamp"
            if allowed
            else "partial views or segment-global scope cannot establish absence"
        )
        return {"allowed": allowed, "scope": requested_scope, "reason": reason}
    return {
        "allowed": bool(evidence.get("seen") and requested_scope in {"observed_viewport", "observed_timestamp"}),
        "scope": requested_scope,
        "reason": "positive facts still require direct frame references",
    }


def panorama_bundle_coverage(
    manifest: Mapping[str, Any],
    timeline: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in timeline if isinstance(row, Mapping)]
    profile = manifest.get("panorama_profile") if isinstance(manifest.get("panorama_profile"), Mapping) else {}
    plan = manifest.get("panorama_analysis_plan") if isinstance(manifest.get("panorama_analysis_plan"), Mapping) else {}
    evidence_indexes = [
        _index(row, position)
        for position, row in enumerate(rows, start=1)
        if isinstance(row.get("panorama_visual_understanding"), Mapping)
    ]
    planned = _positive_indexes(plan.get("selected_indexes"))
    required = bool(profile.get("ready") or planned or evidence_indexes)
    expected_indexes = planned or (list(range(1, len(rows) + 1)) if bool(profile.get("ready")) else evidence_indexes)
    temporal_expected = _positive_indexes(plan.get("temporal_required_indexes"))
    geometric_complete: list[int] = []
    semantic_covered: list[int] = []
    temporal_complete: list[int] = []
    review_gaps: list[dict[str, Any]] = []
    by_index = {_index(row, position): row for position, row in enumerate(rows, start=1)}
    for index in expected_indexes:
        evidence = aggregate_panorama_evidence(by_index.get(index, {}))
        if evidence["geometric"]["status"] == "complete":
            geometric_complete.append(index)
        if evidence["semantic"]["status"] in {"confirmed", "inferred"}:
            semantic_covered.append(index)
        if index in temporal_expected and evidence["temporal"]["status"] == "complete":
            temporal_complete.append(index)
        for gap in evidence["review_gaps"]:
            review_gaps.append({"index": index, **dict(gap)})
    missing = [index for index in expected_indexes if index not in geometric_complete or index not in semantic_covered]
    temporal_missing = [index for index in temporal_expected if index not in temporal_complete]
    return {
        "schema": PANORAMA_COVERAGE_SCHEMA,
        "required": required,
        "expected_segment_indexes": expected_indexes,
        "expected_segments": len(expected_indexes),
        "geometric_complete_segments": len(geometric_complete),
        "semantic_covered_segments": len(semantic_covered),
        "temporal_expected_segments": len(temporal_expected),
        "temporal_complete_segments": len(temporal_complete),
        "missing_segment_indexes": missing,
        "temporal_missing_indexes": temporal_missing,
        "review_gaps": review_gaps,
        "status": "not_required" if not required else "complete" if not missing and not temporal_missing else "review_gap",
    }


def _temporal_coverage(item: Mapping[str, Any]) -> dict[str, Any]:
    value = item.get("panorama_temporal_understanding")
    if not isinstance(value, Mapping):
        return {"status": "not_required", "required_directions": [], "covered_directions": [], "missing_directions": []}
    required = [str(row) for row in value.get("required_directions") or [] if str(row) in CANONICAL_DIRECTIONS]
    covered = [
        str(row.get("direction") or "")
        for row in value.get("tracks") or []
        if isinstance(row, Mapping)
        and str(row.get("evidence_level") or "") in {"confirmed", "inferred"}
        and int(row.get("observed_frame_count") or 0) >= 5
    ]
    missing = [direction for direction in required if direction not in covered]
    return {
        "status": "complete" if required and not missing else "incomplete" if required else "not_required",
        "required_directions": required,
        "covered_directions": [direction for direction in required if direction in covered],
        "missing_directions": missing,
    }


def _cross_view_entities(
    observations: list[dict[str, Any]], verified: set[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    for row in observations:
        viewport_id = str(row.get("viewport_id") or "")
        for entity in row.get("entity_candidates") or []:
            label = str(entity or "").strip()
            if label and viewport_id and viewport_id not in occurrences[label]:
                occurrences[label].append(viewport_id)
    results: list[dict[str, Any]] = []
    for entity, viewports in sorted(occurrences.items()):
        if len(viewports) < 2:
            continue
        for left_pos, left in enumerate(viewports[:-1]):
            for right in viewports[left_pos + 1 :]:
                forward = (left, right, entity)
                reverse = (right, left, entity)
                results.append(
                    {
                        "entity_candidate": entity,
                        "viewport_ids": [left, right],
                        "status": "verified_same_entity" if forward in verified or reverse in verified else "possible_duplicate",
                    }
                )
    return results


def _positive_indexes(value: Any) -> list[int]:
    values = value if isinstance(value, list) else []
    return sorted({int(row) for row in values if str(row).isdigit() and int(row) > 0})


def _index(item: Mapping[str, Any], position: int) -> int:
    try:
        value = int(item.get("index") or position)
    except (TypeError, ValueError):
        value = position
    return value if value > 0 else position
