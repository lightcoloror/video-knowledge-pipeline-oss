from __future__ import annotations

from video_knowledge_pipeline.numeric_normalization import (
    number_evidence_map,
    numeric_mentions_equivalent,
)
from video_knowledge_pipeline.smart_summary_codex import _number_display_mentions


def test_verified_spoken_event_range_matches_written_summary_range() -> None:
    source = number_evidence_map("每个月组织一两场小型群活动")
    summary = number_evidence_map("每月组织1–2场小型活动")

    assert source.keys() == summary.keys()
    assert set(source) == {"range:场:1-2:场"}
    assert numeric_mentions_equivalent(
        "每个月组织一两场小型群活动",
        "每月组织1–2场小型活动",
    )


def test_compact_chinese_year_keeps_existing_digit_sequence_semantics() -> None:
    assert numeric_mentions_equivalent("发生在二四年", "发生在24年")


def test_unrelated_digit_groups_are_not_silently_joined_or_ranged() -> None:
    examples = {
        "关键词666 888": {"number:666:", "number:888:"},
        "会议日期7 24": {"number:7:", "number:24:"},
    }

    for text, expected in examples.items():
        assert set(number_evidence_map(text)) == expected


def test_range_display_keeps_both_endpoints() -> None:
    assert _number_display_mentions({"1-2场"}) == ["1-2场"]
