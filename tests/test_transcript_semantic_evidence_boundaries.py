from __future__ import annotations

from video_knowledge_pipeline.models import TranscriptCue
from video_knowledge_pipeline.transcript_semantic_correction import (
    _has_strong_action_evidence,
    _has_strong_number_evidence,
    _result_template,
    _timeline_match_for_cue,
)


def _candidate(source_type: str, evidence_level: str, text: str) -> dict:
    return {
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "source_type": source_type,
                "evidence_level": evidence_level,
                "text": text,
            }
        ]
    }


def test_unconfirmed_visual_text_and_tags_are_not_strong_number_or_action_evidence() -> None:
    for source_type in ("ocr", "structured_visual", "tagger"):
        for evidence_level in ("unverified", "inferred", "candidate_weak"):
            candidate = _candidate(
                source_type,
                evidence_level,
                "页面显示100元，并提示点击确认",
            )
            assert not _has_strong_number_evidence(
                candidate,
                ["evidence-1"],
                corrected_text="100元",
            )
            assert not _has_strong_action_evidence(
                candidate,
                ["evidence-1"],
                corrected_text="点击确认",
            )


def test_confirmed_direct_visual_evidence_remains_eligible_but_tags_do_not_prove_action() -> None:
    confirmed_ocr = _candidate("ocr", "confirmed", "页面显示100元")
    confirmed_structured = _candidate(
        "structured_visual",
        "confirmed",
        "按钮状态显示点击确认",
    )
    confirmed_tags = _candidate("tagger", "confirmed", "点击 确认")

    assert _has_strong_number_evidence(
        confirmed_ocr,
        ["evidence-1"],
        corrected_text="100元",
    )
    assert _has_strong_action_evidence(
        confirmed_structured,
        ["evidence-1"],
        corrected_text="点击确认",
    )
    assert not _has_strong_action_evidence(
        confirmed_tags,
        ["evidence-1"],
        corrected_text="点击确认",
    )


def test_positive_duration_ranges_require_positive_overlap() -> None:
    cue = TranscriptCue(start=0.0, end=1.0, text="相邻边界")
    item, alignment = _timeline_match_for_cue(
        [{"index": 2, "start": 1.0, "end": 2.0}],
        cue,
        cue.start,
        cue.end,
    )

    assert item is None
    assert alignment["status"] == "not_matched"


def test_zero_duration_points_may_use_the_explicit_rounding_window() -> None:
    cue = TranscriptCue(start=3.0, end=3.0, text="时间点")
    item, alignment = _timeline_match_for_cue(
        [{"index": 3, "start": 3.125, "end": 3.125}],
        cue,
        cue.start,
        cue.end,
    )

    assert item is not None
    assert alignment["basis"] == "bounded_time_window"
    assert alignment["distance_seconds"] == 0.125


def test_result_template_lists_all_high_risk_semantic_types() -> None:
    template = _result_template(
        {
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "original_text": "原文",
                    "evidence_ids": ["evidence-1"],
                }
            ]
        }
    )
    correction_type = template["decisions"][0]["correction_type"]

    for value in (
        "negation",
        "condition",
        "exception",
        "person_name",
        "organization",
        "date_time",
        "money",
        "percentage",
    ):
        assert value in correction_type
