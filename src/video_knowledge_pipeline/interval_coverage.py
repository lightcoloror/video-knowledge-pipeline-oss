from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge positive time intervals using the existing ASR coverage semantics."""

    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def merge_nonnegative_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Clamp interval bounds to zero before using the shared merge contract."""

    return merge_intervals(
        (max(0.0, float(start)), max(0.0, float(end)))
        for start, end in intervals
    )


def closed_intervals_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    """Return whether two closed time intervals overlap or touch."""

    return max(left_start, right_start) <= min(left_end, right_end)


def interval_intersection_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    """Return the positive intersection duration of two time intervals."""

    return max(
        0.0,
        min(float(left_end), float(right_end))
        - max(float(left_start), float(right_start)),
    )


def interval_intersection_over_union(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    """Return temporal intersection-over-union for two intervals."""

    intersection = interval_intersection_seconds(
        left_start, left_end, right_start, right_end
    )
    union = max(float(left_end), float(right_end)) - min(
        float(left_start), float(right_start)
    )
    return intersection / union if union > 0 else 0.0


def interval_intersection_over_shorter(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    """Return the intersection as a fraction of the shorter valid interval."""

    intersection = interval_intersection_seconds(
        left_start, left_end, right_start, right_end
    )
    shorter = min(
        max(0.0, float(left_end) - float(left_start)),
        max(0.0, float(right_end) - float(right_start)),
    )
    return intersection / shorter if shorter > 0 else 0.0

def interval_coverage(
    target_intervals: Iterable[tuple[float, float]],
    covered_intervals: Iterable[tuple[float, float]],
    *,
    minimum_gap_seconds: float,
    minimum_coverage_ratio: float | None = None,
    maximum_uncovered_seconds: float | None = None,
) -> dict[str, Any]:
    """Measure interval coverage without hiding cumulative short gaps.

    Intent: keep the existing per-gap noise threshold while allowing strict
    callers to reject many short gaps whose cumulative coverage is unsafe.
    Decision: preserve ``gaps`` and add complete uncovered evidence plus
    optional ratio/cumulative budgets.
    Reason: a minimum duration alone can report ``passed`` at zero coverage.
    Evidence: an ASR probe reproduced this with three 1.9-second speech gaps.
    Effective scope: historical callers keep their old status semantics unless
    they opt into the new budgets.
    """

    minimum_gap = max(0.0, float(minimum_gap_seconds))
    ratio_floor = (
        None
        if minimum_coverage_ratio is None
        else min(1.0, max(0.0, float(minimum_coverage_ratio)))
    )
    uncovered_budget = (
        None
        if maximum_uncovered_seconds is None
        else max(0.0, float(maximum_uncovered_seconds))
    )
    targets = merge_intervals(target_intervals)
    covered = merge_intervals(covered_intervals)
    target_seconds = sum(end - start for start, end in targets)
    covered_seconds = 0.0
    uncovered: list[tuple[float, float]] = []
    for target_start, target_end in targets:
        cursor = target_start
        for covered_start, covered_end in covered:
            overlap_start = max(target_start, covered_start)
            overlap_end = min(target_end, covered_end)
            if overlap_end <= overlap_start:
                continue
            if overlap_start > cursor:
                uncovered.append((cursor, overlap_start))
            covered_seconds += overlap_end - overlap_start
            cursor = max(cursor, overlap_end)
        if cursor < target_end:
            uncovered.append((cursor, target_end))
    all_gaps = [
        {
            "start": round(start, 6),
            "end": round(end, 6),
            "duration_seconds": round(end - start, 6),
        }
        for start, end in uncovered
    ]
    gaps = [
        row for row in all_gaps if float(row["duration_seconds"]) >= minimum_gap
    ]
    subthreshold_gaps = [
        row for row in all_gaps if float(row["duration_seconds"]) < minimum_gap
    ]
    bounded_covered_seconds = min(covered_seconds, target_seconds)
    uncovered_seconds = max(0.0, target_seconds - bounded_covered_seconds)
    coverage_ratio = (
        bounded_covered_seconds / target_seconds if target_seconds else 1.0
    )
    status_reasons: list[str] = []
    if gaps:
        status_reasons.append("material_gap")
    if ratio_floor is not None and coverage_ratio < ratio_floor:
        status_reasons.append("coverage_ratio_below_minimum")
    if uncovered_budget is not None and uncovered_seconds > uncovered_budget:
        status_reasons.append("cumulative_uncovered_exceeds_budget")
    return {
        "status": "degraded" if status_reasons else "passed",
        "target_interval_count": len(targets),
        "covered_interval_count": len(covered),
        "target_seconds": round(target_seconds, 6),
        "covered_seconds": round(bounded_covered_seconds, 6),
        "uncovered_seconds": round(uncovered_seconds, 6),
        "coverage_ratio": round(coverage_ratio, 6),
        "minimum_gap_seconds": minimum_gap,
        "minimum_coverage_ratio": ratio_floor,
        "maximum_uncovered_seconds": uncovered_budget,
        "status_reasons": status_reasons,
        "all_gaps": all_gaps,
        "subthreshold_gap_count": len(subthreshold_gaps),
        "subthreshold_uncovered_seconds": round(
            sum(float(row["duration_seconds"]) for row in subthreshold_gaps),
            6,
        ),
        "subthreshold_gaps": subthreshold_gaps,
        "gaps": gaps,
    }
