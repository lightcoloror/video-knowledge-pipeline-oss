from __future__ import annotations

from video_knowledge_pipeline.smart_summary_global_reduce import (
    _review_gap_promotions,
    _shape_quality,
)


def _fact_pack() -> dict[str, object]:
    return {
        "sections": [
            {
                "section_id": "chapter-0001",
                "facts": [
                    {
                        "text": "3300 元只有模型候选，没有可靠来源证据",
                        "fact_status": "review_gap_not_fact",
                    }
                ],
                "evidence_refs": [
                    {
                        "evidence_id": "gap-0001",
                        "snippet": "明年保证收益百分之百",
                        "fact_status": "review_gap_not_fact",
                    }
                ],
            }
        ]
    }


def _summary(*, confirmed_text: str = "", pending_text: str = "") -> str:
    headings = (
        "基本信息",
        "一句话概览",
        "核心主题 / 课程主线",
        "分段总结",
        "关键观点 / 方法论",
        "可执行动作清单",
        "高频话术 / 可复用表达",
        "待复核点 / 低置信内容",
    )
    sections: list[str] = []
    for heading in headings:
        text = pending_text if heading.startswith("待复核点") else ("普通内容" * 30)
        if heading.startswith("关键观点") and confirmed_text:
            text += confirmed_text
        sections.append(f"## {heading}\n{text}\n00:00:00.000 00:02:00.000")
    return "生成方式：`codex_llm_rewrite_final`。\n" + "\n".join(sections)


def test_review_gap_content_is_rejected_outside_pending_section() -> None:
    phrase = "3300 元只有模型候选，没有可靠来源证据"
    content = _summary(confirmed_text=phrase)

    assert _review_gap_promotions(content, _fact_pack()) == [phrase]
    quality = _shape_quality(
        content,
        expected_ids={"chapter-0001"},
        rows=[{"time_range": "00:00:00.000 - 00:02:00.000"}],
        fact_pack=_fact_pack(),
    )
    gate = next(row for row in quality["checks"] if row["key"] == "review_gap_not_promoted")
    assert gate["passed"] is False
    assert quality["passed"] is False


def test_review_gap_content_is_allowed_inside_pending_section() -> None:
    content = _summary(
        pending_text=(
            "3300 元只有模型候选，没有可靠来源证据；"
            "明年保证收益百分之百。"
        )
    )

    assert _review_gap_promotions(content, _fact_pack()) == []
    quality = _shape_quality(
        content,
        expected_ids={"chapter-0001"},
        rows=[{"time_range": "00:00:00.000 - 00:02:00.000"}],
        fact_pack=_fact_pack(),
    )
    gate = next(row for row in quality["checks"] if row["key"] == "review_gap_not_promoted")
    assert gate["passed"] is True
