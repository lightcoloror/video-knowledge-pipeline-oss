from __future__ import annotations

from video_knowledge_pipeline.numeric_normalization import numeric_mentions_equivalent, number_evidence_map, strip_number_mentions
from video_knowledge_pipeline.transcript_semantic_correction import _visual_conflict_text
from video_knowledge_pipeline.quality_benchmark import transcript_quality_metrics


def test_equivalent_spoken_and_arabic_amounts_share_canonical_value() -> None:
    assert numeric_mentions_equivalent("保费两百万元", "保费200万元") is True
    assert set(number_evidence_map("保费两百万元")) == {"currency:2000000:元"}


def test_equivalent_spoken_year_digits_share_canonical_value() -> None:
    assert numeric_mentions_equivalent("发生在二四年", "发生在24年") is True


def test_real_amount_difference_is_not_equivalent() -> None:
    assert numeric_mentions_equivalent("保费200万元", "保费300万元") is False


def test_visual_conflict_ignores_equivalent_numeric_forms_but_keeps_real_delta() -> None:
    assert _visual_conflict_text("客户保费是两百万元。", "课件：保费 200万元") == ""
    assert _visual_conflict_text("客户保费是两百万元。", "课件：保费 300万元") == "300万元"

def test_equivalent_approximate_amounts_and_spoken_percentages_share_canonical_values() -> None:
    assert numeric_mentions_equivalent("关联保费有200多万", "关联保费有两百多万") is True
    assert numeric_mentions_equivalent("标题要花30%的心思", "标题要花百分之三十的心思") is True


def test_quality_metrics_do_not_penalize_equivalent_number_rendering() -> None:
    metrics = transcript_quality_metrics(
        "关联保费有200多万，标题要花30%的心思。",
        "关联保费有两百多万，标题要花百分之三十的心思。",
    )

    assert metrics["number_error_count"] == 0
    assert metrics["entity_accuracy"] == 1.0


def test_equivalent_bare_arabic_and_chinese_numbers_share_canonical_values() -> None:
    pairs = [("100", "一百"), ("300", "三百"), ("8000", "八千")]

    for arabic, chinese in pairs:
        assert numeric_mentions_equivalent(arabic, chinese) is True
        assert strip_number_mentions(arabic) == ""
        assert strip_number_mentions(chinese) == ""


def test_bare_chinese_number_detection_does_not_treat_common_words_as_numbers() -> None:
    for text in ("统一", "一方面", "一会儿", "万一", "千万不要"):
        assert number_evidence_map(text) == {}
        assert strip_number_mentions(text) == text


def test_spoken_counted_time_unit_matches_written_duration() -> None:
    assert numeric_mentions_equivalent("不到6个小时", "约6小时")
    assert numeric_mentions_equivalent("六个小时", "6小时")
    assert set(number_evidence_map("不到6个小时")) == {"duration:360:分钟"}
