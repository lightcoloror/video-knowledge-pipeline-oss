from __future__ import annotations

import pytest

from video_knowledge_pipeline.models import TranscriptCue
from video_knowledge_pipeline.punctuation_model_stage import _content_only
from video_knowledge_pipeline.transcript_agent_readable import _apply_import_rows
from video_knowledge_pipeline.transcript_readable_llm import (
    _agent_substitute_polish_text,
    _apply_rows,
)


def _cue(text: str) -> TranscriptCue:
    return TranscriptCue(start=0.0, end=3.0, text=text)


def test_llm_readable_accepts_surface_only_punctuation_and_whitespace() -> None:
    original = "如果收益是10%，中意人寿不在例外名单内"
    candidate = "如果收益是 10%，\n中意人寿不在例外名单内！"

    accepted, rejected = _apply_rows(
        [_cue(original)], {"segments": [{"index": 0, "text": candidate}]}
    )

    assert rejected == []
    assert accepted[0]["text"] == candidate
    assert _content_only(original) == _content_only(candidate)


@pytest.mark.parametrize(
    ("original", "candidate", "expected_issue"),
    [
        ("保费是1000元", "保费是10000元", "number_anchor_changed"),
        ("收益率是10%", "收益率是20%", "percentage_anchor_changed"),
        ("日期是2026年8月31日", "日期是2025年8月31日", "date_anchor_changed"),
        ("中意人寿王老师发言", "中医人寿王老师发言", "named_entity_candidate_changed"),
        ("这个方案不可以退款", "这个方案可以退款", "negation_anchor_changed"),
        ("如果审核通过才能生效", "审核通过就能生效", "condition_anchor_changed"),
        ("既往症除外，但可申请复核", "既往症可申请复核", "exception_anchor_changed"),
        ("使用Qwen3-VL处理", "使用Qwen3处理", "latin_token_anchor_changed"),
    ],
)
def test_llm_readable_rejects_semantic_anchor_changes(
    original: str, candidate: str, expected_issue: str
) -> None:
    accepted, rejected = _apply_rows(
        [_cue(original)], {"segments": [{"index": 0, "text": candidate}]}
    )

    assert accepted == []
    assert rejected[0]["reason"] == "readable_semantic_guard_failed"
    assert expected_issue in {issue["kind"] for issue in rejected[0]["issues"]}
    assert "content_character_sequence_changed" in {
        issue["kind"] for issue in rejected[0]["issues"]
    }


def test_agent_readable_merge_validates_joined_source_content_and_order() -> None:
    cues = [
        TranscriptCue(start=0.0, end=2.0, text="第一段有100元"),
        TranscriptCue(start=2.0, end=4.0, text="第二段不退款"),
    ]
    safe = {
        "segments": [
            {
                "index": 0,
                "source_indexes": [0, 1],
                "text": "第一段有100元；第二段不退款。",
            }
        ]
    }

    accepted, rejected = _apply_import_rows(cues, safe)

    assert rejected == []
    assert accepted[0]["source_indexes"] == [0, 1]

    unsafe = {
        "segments": [
            {
                "index": 0,
                "source_indexes": [0, 1],
                "text": "第二段不退款；第一段有100元。",
            }
        ]
    }
    accepted, rejected = _apply_import_rows(cues, unsafe)

    assert accepted == []
    assert rejected[0]["reason"] == "readable_semantic_guard_failed"
    assert "content_character_sequence_changed" in {
        issue["kind"] for issue in rejected[0]["issues"]
    }


def test_local_readable_substitute_no_longer_performs_semantic_replacements() -> None:
    original = "买虫二则一同意心更了解的价明晚八点o我找一下"

    polished = _agent_substitute_polish_text(original)

    assert "买虫" in polished
    assert "二则一" in polished
    assert "同意心" in polished
    assert "更了解的价" in polished
    assert "明晚八点o" in polished
    assert _content_only(original) == _content_only(polished)
