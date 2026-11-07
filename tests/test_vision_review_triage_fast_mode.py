from __future__ import annotations

from video_knowledge_pipeline.vision_review_triage import _spread_limit


def test_spread_limit_keeps_full_timeline_coverage() -> None:
    rows = [
        {
            "index": index,
            "start": float(index * 10),
            "score": 5,
        }
        for index in range(1, 13)
    ]

    selected = _spread_limit(rows, 6)

    assert [row["index"] for row in selected] == [1, 3, 5, 8, 10, 12]


def test_spread_limit_one_keeps_highest_risk_item() -> None:
    rows = [
        {"index": 1, "score": 3},
        {"index": 6, "score": 9},
        {"index": 12, "score": 5},
    ]

    selected = _spread_limit(rows, 1)

    assert [row["index"] for row in selected] == [6]
    assert _spread_limit(rows, None) == rows
