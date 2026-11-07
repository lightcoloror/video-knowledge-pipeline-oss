from __future__ import annotations

from video_knowledge_pipeline.smart_summary_section_apply import (
    _extract_bullets,
    _normalize_section_body,
    _section_markdown,
)

def test_section_apply_removes_generic_model_wrapper_before_rendering() -> None:
    source = (
        "### \u7ae0\u8282\u6807\u9898\\n\\n"
        "\u597d\u7684\uff0c\u8fd9\u662f\u6839\u636e\u60a8\u63d0\u4f9b\u7684\u8f93\u5165\u548c\u6307\u4ee4\uff0c\u4e3a\u201c\u7ae0\u8282 0004\u201d\uff0800:19:56 - 00:29:44\uff09\u91cd\u5199\u7684 `smart-summary.md` \u7ae0\u8282\u3002\\n\\n"
        "### \u5ba2\u6237\u793e\u7fa4\u8fd0\u8425\\n\\n"
        "- \u6301\u7eed\u63d0\u4f9b\u5b9e\u7528\u4fe1\u606f\u6765\u5efa\u7acb\u4fe1\u4efb\u3002"
    )
    source = source.replace("\\n", "\n")

    result = _normalize_section_body(source)
    rendered = "\n".join(
        _section_markdown(
            {"title": "schema: internal", "start": 0, "end": 10, "final_markdown": source}
        )
    )

    assert "schema:" not in rendered
    assert "\u5ba2\u6237\u793e\u7fa4\u8fd0\u8425" in rendered

    assert "\u597d\u7684" not in result
    assert "smart-summary.md" not in result
    assert result.startswith("- ")
    assert "\u6301\u7eed\u63d0\u4f9b\u5b9e\u7528\u4fe1\u606f" in result


def test_section_apply_caps_secondary_navigation_lists_without_truncating_sections() -> None:
    sections = [
        {
            "start_time": f"00:0{index}:00",
            "final_markdown": "客户需要先确认问题，再记录证据并安排下一步行动。",
        }
        for index in range(9)
    ]

    actions = _extract_bullets(sections, fallback="- none", mode="actions")
    expressions = _extract_bullets(sections, fallback="- none", mode="expressions")

    assert len(actions) == 7
    assert len(expressions) == 7
    assert all("客户需要" in row for row in actions)
