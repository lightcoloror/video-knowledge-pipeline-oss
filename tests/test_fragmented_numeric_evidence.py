from __future__ import annotations

from video_knowledge_pipeline.numeric_normalization import (
    number_evidence_map,
    numeric_mentions_equivalent,
)


def test_fragmented_year_and_amount_match_canonical_claims() -> None:
    assert numeric_mentions_equivalent("发生在202 3年的年底", "发生在2023年") is True
    assert numeric_mentions_equivalent("许佳业绩330 0", "许佳业绩3300") is True
    assert numeric_mentions_equivalent("课件写着3,300", "课件写着3300") is True


def test_short_date_like_digit_groups_are_not_silently_joined() -> None:
    evidence = number_evidence_map("会议日期是7 24")

    assert "number:724:" not in evidence
