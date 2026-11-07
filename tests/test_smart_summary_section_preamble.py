from __future__ import annotations

from video_knowledge_pipeline.smart_summary_section_apply import _normalize_section_body


def test_section_body_strips_known_llm_wrapper_preamble() -> None:
    body = _normalize_section_body(
        "好的，这是根据您提供的输入和指令，为“章节 0004”（00:19:56 - 00:29:44）重写的 `smart-summary.md` 章节。\n\n"
        "### 客户社群运营\n\n- 保留的正文。"
    )

    assert "章节 0004" not in body
    assert body == "- 保留的正文。"

def test_section_body_strips_outer_heading_before_known_llm_wrapper() -> None:
    body = _normalize_section_body(
        "### `00:19:56.897 - 00:29:44.353` generated title\n\n"
        "好的，这是根据您提供的输入和指令，为“章节 0004”（00:19:56 - 00:29:44）重写的 `smart-summary.md` 章节。\n\n"
        "### Generated provider heading\n\n- 保留的正文。"
    )

    assert "章节 0004" not in body
    assert "Generated provider heading" not in body
    assert body == "- 保留的正文。"
