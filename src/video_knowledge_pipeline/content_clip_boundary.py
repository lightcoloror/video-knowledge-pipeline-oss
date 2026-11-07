from __future__ import annotations

from typing import Any

from .transcript import format_timestamp


SHOT_BOUNDARY_STRATEGIES = {
    "action_start_peak_end",
    "whole_technical_shot",
    "cause_peak_result",
    "cause_change_result",
}


def build_content_clip_boundary(
    *,
    semantic_start: float,
    semantic_end: float,
    transcript_segments: list[dict[str, Any]],
    technical_shots: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    boundary_strategy: str,
    duration: dict[str, Any],
    media_end: float,
) -> dict[str, Any]:
    """Project existing transcript/shot/screen evidence into editable cut ranges.

    Intent: use content-specific evidence boundaries without running a second
    media pipeline. Decision: consume verified technical shots and existing
    transcript/Timeline ranges; missing evidence remains explicit. Reason:
    sentence, action, screen, and B-roll clips do not share one safe boundary.
    Evidence: VKP technical_shot_boundaries.v1 and transcript segment lineage;
    moys-asr-workflow editor-utils timing behavior at pinned commit
    949bc84058cdae1d9c021c50203e6d2742f9392c (64/64 local tests).
    Effective scope: derived recommended and safe-extension ranges only.
    """

    semantic_start = max(0.0, float(semantic_start))
    semantic_end = max(semantic_start, float(semantic_end))
    reason: list[str] = []
    evidence_status = "confirmed"
    source_shots = _overlapping(technical_shots, semantic_start, semantic_end)
    source_segments = _overlapping(transcript_segments, semantic_start, semantic_end)
    source_timeline = _overlapping(timeline_rows, semantic_start, semantic_end)
    word_timestamps = [
        word
        for row in source_segments
        for word in row.get("words") or []
        if isinstance(word, dict) and (word.get("start") is not None or word.get("end") is not None)
    ]

    if boundary_strategy in SHOT_BOUNDARY_STRATEGIES:
        if source_shots:
            recommended_start, recommended_end = _outer_range(source_shots)
            reason.append("snapped_to_verified_technical_shot")
        else:
            recommended_start, recommended_end = semantic_start, semantic_end
            evidence_status = "unavailable"
            reason.append("technical_shot_evidence_missing")
    elif boundary_strategy == "stable_screen_content":
        if source_timeline:
            recommended_start, recommended_end = _outer_range(source_timeline)
            reason.append("bounded_by_existing_screen_evidence_rows")
        elif source_shots:
            recommended_start, recommended_end = _outer_range(source_shots)
            evidence_status = "inferred"
            reason.append("screen_stability_unavailable_used_technical_shot")
        else:
            recommended_start, recommended_end = semantic_start, semantic_end
            evidence_status = "unavailable"
            reason.append("screen_boundary_evidence_missing")
    elif boundary_strategy == "instruction_to_stable_result":
        ranges = [*source_segments, *source_shots]
        if ranges:
            recommended_start, recommended_end = _outer_range(ranges)
            evidence_status = "inferred"
            reason.append("combined_instruction_and_existing_shot_ranges")
        else:
            recommended_start, recommended_end = semantic_start, semantic_end
            evidence_status = "unavailable"
            reason.append("tutorial_completion_evidence_missing")
    elif boundary_strategy == "audio_event_bounds":
        recommended_start, recommended_end = semantic_start, semantic_end
        evidence_status = "inferred"
        reason.append("audio_event_range_requires_clip_only_confirmation")
    else:
        if source_segments:
            recommended_start, recommended_end = _outer_range(source_segments)
            reason.append("expanded_to_complete_transcript_segments")
        else:
            recommended_start, recommended_end = semantic_start, semantic_end
            evidence_status = "unavailable"
            reason.append("transcript_boundary_evidence_missing")

    recommended_start, recommended_end, duration_reasons = _fit_duration(
        recommended_start,
        recommended_end,
        duration,
        media_end=max(media_end, semantic_end),
    )
    reason.extend(duration_reasons)
    safe_start, safe_end = _safe_extension(
        recommended_start,
        recommended_end,
        transcript_segments=transcript_segments,
        technical_shots=technical_shots,
        maximum=float(duration.get("maximum_seconds") or 0.0),
        media_end=max(media_end, semantic_end),
    )
    return {
        "status": evidence_status,
        "semantic_match_range": _time_range(semantic_start, semantic_end),
        "recommended_cut_range": _time_range(recommended_start, recommended_end),
        "safe_extension_range": _time_range(safe_start, safe_end),
        "source_shot_ids": [str(row.get("shot_id") or row.get("id") or "") for row in source_shots if str(row.get("shot_id") or row.get("id") or "")],
        "source_segment_ids": [str(row.get("segment_id") or "") for row in source_segments if str(row.get("segment_id") or "")],
        "timeline_indexes": [int(row["index"]) for row in source_timeline if str(row.get("index") or "").isdigit()],
        "boundary_reason": reason,
        "word_timestamp_used": bool(word_timestamps),
        "human_boundary_review_required": evidence_status != "confirmed" or boundary_strategy not in {"complete_sentence", "whole_technical_shot"},
    }


def _overlapping(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [row for row in rows if _end(row) > start and _start(row) < end]


def _start(row: dict[str, Any]) -> float:
    return float(row.get("start") or row.get("start_seconds") or 0.0)


def _end(row: dict[str, Any]) -> float:
    start = _start(row)
    return max(start, float(row.get("end") or row.get("end_seconds") or start))


def _outer_range(rows: list[dict[str, Any]]) -> tuple[float, float]:
    return min(_start(row) for row in rows), max(_end(row) for row in rows)


def _fit_duration(start: float, end: float, duration: dict[str, Any], *, media_end: float) -> tuple[float, float, list[str]]:
    minimum = max(0.0, float(duration.get("minimum_seconds") or 0.0))
    maximum = max(0.0, float(duration.get("maximum_seconds") or 0.0))
    reasons: list[str] = []
    current = end - start
    if minimum and current < minimum:
        extra = minimum - current
        start = max(0.0, start - extra / 2.0)
        end = min(media_end, end + extra / 2.0)
        if end - start < minimum:
            start = max(0.0, end - minimum)
            end = min(media_end, start + minimum)
        reasons.append("extended_to_minimum_duration")
    if maximum and end - start > maximum:
        center = (start + end) / 2.0
        start = max(0.0, center - maximum / 2.0)
        end = min(media_end, start + maximum)
        start = max(0.0, end - maximum)
        reasons.append("clamped_to_maximum_duration_requires_review")
    return start, max(start, end), reasons


def _safe_extension(
    start: float,
    end: float,
    *,
    transcript_segments: list[dict[str, Any]],
    technical_shots: list[dict[str, Any]],
    maximum: float,
    media_end: float,
) -> tuple[float, float]:
    nearby = [row for row in [*transcript_segments, *technical_shots] if _end(row) >= start - 2.0 and _start(row) <= end + 2.0]
    if nearby:
        safe_start, safe_end = _outer_range(nearby)
    else:
        safe_start, safe_end = max(0.0, start - 2.0), min(media_end, end + 2.0)
    if maximum and safe_end - safe_start > maximum:
        safe_start = max(0.0, start - max(0.0, maximum - (end - start)) / 2.0)
        safe_end = min(media_end, safe_start + maximum)
    return safe_start, max(safe_start, safe_end)


def _time_range(start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(float(start), 6),
        "end": round(float(end), 6),
        "start_time": format_timestamp(float(start)),
        "end_time": format_timestamp(float(end)),
    }
