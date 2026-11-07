import pytest

from video_knowledge_pipeline.asr_local_agreement import (
    measure_boundary_lcs_dedup,
    measure_local_agreement,
    measure_timestamped_local_agreement,
)


def test_chinese_local_agreement_uses_character_prefix() -> None:
    result = measure_local_agreement(
        "保险方案需要先确认需求",
        "保险方案可以再做调整",
        language="zh-CN",
    )

    assert result["token_mode"] == "character"
    assert result["common_prefix"] == "保险方案"
    assert result["common_prefix_unit_count"] == 4
    assert result["exact_match"] is False
    assert result["automatic_merge_allowed"] is False


def test_word_local_agreement_matches_complete_words_case_insensitively() -> None:
    result = measure_local_agreement(
        "The customer needs a plan",
        "the CUSTOMER needs more context",
        language="en",
    )

    assert result["token_mode"] == "word"
    assert result["common_prefix"] == "the customer needs"
    assert result["common_prefix_unit_count"] == 3
    assert result["agreement_over_shorter"] == 0.6


def test_equal_text_confirms_the_complete_prefix_without_off_by_one_loss() -> None:
    result = measure_local_agreement("同一个边界", "同一个边界", language="zh")

    assert result["common_prefix"] == "同一个边界"
    assert result["common_prefix_unit_count"] == 5
    assert result["agreement_over_shorter"] == 1.0
    assert result["exact_match"] is True


def test_empty_text_never_becomes_an_exact_or_automatic_agreement() -> None:
    result = measure_local_agreement("", "", language="zh")

    assert result["common_prefix"] == ""
    assert result["agreement_over_shorter"] == 0.0
    assert result["exact_match"] is False
    assert result["candidate_only"] is True


def test_timestamped_agreement_crops_to_overlap_and_confirms_word_prefix() -> None:
    result = measure_timestamped_local_agreement(
        [
            {"start": 7.0, "end": 7.8, "word": "忽略"},
            {"start": 8.0, "end": 8.4, "word": "保险"},
            {"start": 8.4, "end": 8.9, "word": "方案"},
            {"start": 8.9, "end": 9.4, "word": "需要"},
        ],
        [
            {"start": 8.05, "end": 8.45, "word": "保险"},
            {"start": 8.45, "end": 8.95, "word": "方案"},
            {"start": 8.95, "end": 9.45, "word": "可以"},
        ],
        overlap_start=8.0,
        overlap_end=9.5,
    )

    assert result["common_prefix_words"] == ["保险", "方案"]
    assert result["common_prefix_word_count"] == 2
    assert result["agreement_over_shorter"] == pytest.approx(2 / 3)
    assert result["usable_for_review_ranking"] is True
    assert result["automatic_merge_allowed"] is False
    assert result["upstream"]["project"] == "ufal/whisper_streaming"


def test_timestamped_agreement_is_explicitly_unavailable_without_words() -> None:
    result = measure_timestamped_local_agreement(
        [],
        [],
        overlap_start=8.0,
        overlap_end=9.0,
    )

    assert result["status"] == "unavailable"
    assert result["usable_for_review_ranking"] is False
    assert result["unavailable_reason"] == "word_timestamps_missing_in_overlap"

def test_boundary_lcs_dedup_removes_only_confident_repeated_right_prefix() -> None:
    result = measure_boundary_lcs_dedup(
        "前文内容。方案做好了之后，先跟客户约一个时间，比如腾讯会议。",
        "好了之后，先跟客户约一个时间，比如腾讯会议。接着讲解产品。",
        language="zh",
    )

    assert result["status"] == "matched"
    assert result["automatic_merge_allowed"] is True
    assert result["right_prefix_character_count"] > 0
    assert result["matched_unit_count"] >= 8
    assert result["confidence"] == 1.0
    assert result["upstream"]["project"] == "CrispASR / NVIDIA NeMo"


def test_boundary_lcs_dedup_fails_closed_for_unrelated_boundary_text() -> None:
    result = measure_boundary_lcs_dedup(
        "上一块结束于保险方案设计原则。",
        "下一块从客户服务案例开始。",
        language="zh",
    )

    assert result["status"] == "unmatched"
    assert result["automatic_merge_allowed"] is False
    assert result["right_prefix_character_count"] == 0
    assert result["requires_human_review"] is True
