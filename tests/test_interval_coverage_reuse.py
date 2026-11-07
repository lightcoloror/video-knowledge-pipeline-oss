from __future__ import annotations

from pathlib import Path

import video_knowledge_pipeline.asr_chunk_batch_merge as asr_chunk_batch_merge
import video_knowledge_pipeline.asr_consensus as asr_consensus
import video_knowledge_pipeline.lecture_package as lecture_package
import video_knowledge_pipeline.smart_summary_chapters as smart_summary_chapters
import video_knowledge_pipeline.transcript_semantic_summary_impact as summary_impact
from video_knowledge_pipeline.interval_coverage import (
    closed_intervals_overlap,
    interval_intersection_over_shorter,
    interval_intersection_over_union,
    interval_intersection_seconds,
    merge_nonnegative_intervals,
)


def test_closed_interval_overlap_preserves_touching_endpoint_contract() -> None:
    assert closed_intervals_overlap(0.0, 5.0, 4.0, 6.0) is True
    assert closed_intervals_overlap(0.0, 5.0, 5.0, 6.0) is True
    assert closed_intervals_overlap(1.0, 1.0, 1.0, 2.0) is True
    assert closed_intervals_overlap(0.0, 5.0, 5.001, 6.0) is False
    assert closed_intervals_overlap(5.0, 1.0, 2.0, 3.0) is False


def test_summary_callers_keep_private_compatibility_aliases() -> None:
    assert smart_summary_chapters._overlaps is closed_intervals_overlap
    assert summary_impact._overlaps is closed_intervals_overlap


def test_closed_interval_overlap_has_one_source_owner() -> None:
    source_root = Path(__file__).parents[1] / "src" / "video_knowledge_pipeline"
    implementation = "return max(left_start, right_start) <= min(left_end, right_end)"
    owners = {
        path.name
        for path in source_root.glob("*.py")
        if implementation in path.read_text(encoding="utf-8-sig")
    }

    assert owners == {"interval_coverage.py"}

def test_nonnegative_interval_merge_preserves_lecture_contract() -> None:
    intervals = [
        (-5.0, -1.0),
        (-2.0, 2.0),
        (1.0, 3.0),
        (3.0, 4.0),
        (6.0, 5.0),
    ]

    assert merge_nonnegative_intervals(intervals) == [(0.0, 4.0)]


def test_lecture_merge_entrypoint_is_shared_nonnegative_owner() -> None:
    assert lecture_package._merge_intervals is merge_nonnegative_intervals

def test_interval_overlap_metrics_keep_distinct_denominators() -> None:
    assert interval_intersection_seconds(0, 10, 5, 20) == 5
    assert interval_intersection_over_union(0, 10, 5, 20) == 0.25
    assert interval_intersection_over_shorter(0, 10, 5, 20) == 0.5


def test_interval_overlap_metrics_reject_zero_or_reversed_duration() -> None:
    assert interval_intersection_seconds(5, 5, 5, 5) == 0
    assert interval_intersection_over_union(5, 5, 5, 5) == 0
    assert interval_intersection_over_shorter(8, 3, 0, 10) == 0


def test_asr_overlap_callers_keep_their_established_denominators() -> None:
    assert asr_consensus._overlap_ratio is interval_intersection_over_union
    assert asr_chunk_batch_merge._time_overlap_ratio(
        {"start": 0, "end": 10},
        {"start": 5, "end": 20},
    ) == interval_intersection_over_shorter(0, 10, 5, 20)
