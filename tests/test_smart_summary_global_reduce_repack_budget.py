from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_global_reduce import (
    DEFAULT_REDUCE_FACTS_PER_SECTION,
    DEFAULT_REDUCE_QUOTE_REFS_PER_SECTION,
    DEFAULT_REDUCE_REVIEW_REFS_PER_SECTION,
    DEFAULT_REDUCE_SECTION_MARKDOWN_CHARS,
    _chapter_fact_pack,
    _reduce_prompt_plan,
)


def test_eleven_chapter_fact_pack_reuses_section_evidence_groups_within_budget(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    workflow_sections: list[dict[str, object]] = []
    for section_index in range(1, 12):
        section_id = f"chapter-{section_index:04d}"
        time_range = f"00:{section_index - 1:02d}:00.000 - 00:{section_index:02d}:00.000"
        rows.append(
            {
                "section_id": section_id,
                "title": f"第 {section_index} 章",
                "time_range": time_range,
                "final_markdown": f"章节 {section_index} 的独有内容。" + ("内容" * 1200),
            }
        )
        citations = [
            {
                "citation_id": f"{section_id}-moment-with-a-stable-and-deliberately-long-identifier-{citation_index:02d}",
                "chunk_kind": "moment",
                "time_range": time_range,
                "snippet": f"第 {section_index} 章证据 {citation_index}",
                "source": "video_moment_index",
                "fact_status": "candidate_evidence",
            }
            for citation_index in range(1, 9)
        ]
        evidence = {
            "summary_sentences": [f"摘要事实 {index}" for index in range(1, 7)],
            "key_points": [f"关键事实 {index}" for index in range(1, 8)],
            "actions": [],
            "reusable_expressions": [],
            "visual_notes": [],
            "citations": citations,
            "semantic_correction_items": [],
        }
        workflow_sections.append(
            {
                "section_id": section_id,
                "title": f"第 {section_index} 章",
                "time_range": time_range,
                "evidence": evidence,
                "citations": citations,
                "semantic_correction_items": [],
            }
        )

    pack = _chapter_fact_pack(
        tmp_path / "bundle",
        rows,
        {"sections": workflow_sections},
    )
    plan = _reduce_prompt_plan(
        tmp_path / "bundle",
        rows,
        {"mainline": "完整课程主线"},
        fact_pack=pack,
        max_input_chars=60000,
    )

    assert len(plan["prompt"]) <= 60000
    payload = json.loads(plan["prompt"].split("输入 JSON：\n", 1)[1])
    assert max(len(row["markdown"]) for row in payload["chapters"]) <= (
        DEFAULT_REDUCE_SECTION_MARKDOWN_CHARS
    )
    assert all("本章中段已按上下文预算压缩" in row["markdown"] for row in payload["chapters"])
    assert max(len(row["facts"]) for row in payload["chapters"]) <= DEFAULT_REDUCE_FACTS_PER_SECTION
    assert max(len(row["evidence_refs"]) for row in payload["chapters"]) <= (
        DEFAULT_REDUCE_QUOTE_REFS_PER_SECTION + DEFAULT_REDUCE_REVIEW_REFS_PER_SECTION + 1
    )
    assert all(
        "member_evidence_ids" not in ref
        for row in payload["chapters"] for ref in row["evidence_refs"]
    )
    assert plan["all_sections_included"] is True
    assert plan["clipped_section_ids"] == []
    assert all(f"chapter-{index:04d}" in plan["prompt"] for index in range(1, 12))
    assert all(f"chapter-{index:04d}:eligible-evidence-set" in plan["prompt"] for index in range(1, 12))
    for section in pack["sections"]:
        group = next(
            ref
            for ref in section["evidence_refs"]
            if ref["evidence_id"].endswith(":eligible-evidence-set")
        )
        assert len(group["member_evidence_ids"]) == 8
        assert all(fact["evidence_ids"] == [group["evidence_id"]] for fact in section["facts"])
