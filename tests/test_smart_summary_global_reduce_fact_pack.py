from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_global_reduce import (
    CHAPTER_FACT_PACK_SCHEMA,
    _chapter_fact_pack,
    _reduce_prompt_plan,
    run_smart_summary_global_reduce,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _workflow_section(*, citations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "section_id": "chapter-0001",
        "title": "需求确认",
        "time_range": "00:00:00.000 - 00:02:00.000",
        "evidence": {
            "summary_sentences": ["说话人希望成人配置 30 万保额。"],
            "key_points": ["预算不足时考虑降低保额。"],
            "actions": [],
            "reusable_expressions": [],
            "visual_notes": [],
            "citations": citations,
            "semantic_correction_items": [],
        },
        "citations": citations,
        "semantic_correction_items": [
            {
                "candidate_id": "semcorr-0001",
                "time_range": "00:01:00.000 - 00:01:05.000",
                "corrected_text": "明亚保险",
                "correction_status": "human_confirmed",
                "evidence_ids": ["moment-0001"],
            },
            {
                "candidate_id": "semcorr-0002",
                "time_range": "00:01:10.000 - 00:01:15.000",
                "candidate_text": "待确认术语",
                "correction_status": "candidate",
                "needs_human_review": True,
            },
        ],
    }


def test_chapter_fact_pack_projects_existing_evidence_without_promoting_review_gaps(tmp_path: Path) -> None:
    rows = [
        {
            "section_id": "chapter-0001",
            "title": "需求确认",
            "time_range": "00:00:00.000 - 00:02:00.000",
            "final_markdown": "本章讨论成人保额与预算。",
        }
    ]
    citations = [
        {
            "citation_id": "moment-0001",
            "chunk_kind": "moment",
            "time_range": "00:00:10.000 - 00:00:30.000",
            "snippet": "两个大人先做三十万保额",
            "source": "video_moment_index",
            "fact_status": "candidate_evidence",
        },
        {
            "citation_id": "visual-0001",
            "chunk_kind": "visual_evidence",
            "time_range": "00:00:20.000 - 00:00:25.000",
            "snippet": "PPT 显示三十万",
            "visual_snippet": "PPT 显示三十万",
            "source": "video_rag_chunks",
            "fact_status": "candidate_evidence",
        },
        {
            "citation_id": "gap-0001",
            "chunk_kind": "review_gap",
            "time_range": "00:01:30.000 - 00:01:40.000",
            "snippet": "数字 3300 尚无可靠来源",
            "source": "video_rag_chunks",
            "fact_status": "review_gap_not_fact",
        },
    ]
    workflow = {"sections": [_workflow_section(citations=citations)]}

    pack = _chapter_fact_pack(tmp_path / "bundle", rows, workflow)
    section = pack["sections"][0]

    assert pack["schema"] == CHAPTER_FACT_PACK_SCHEMA
    assert len(pack["revision"]) == 64
    assert section["evidence_status"] == "evidence_bound"
    assert {"asr", "visual", "human_confirmed"} <= set(section["source_kinds"])
    assert "gap-0001" in section["review_only_evidence_ids"]
    assert "semcorr-0002" in section["review_only_evidence_ids"]
    assert "gap-0001" not in section["eligible_evidence_ids"]
    assert "semcorr-0002" not in section["eligible_evidence_ids"]
    assert all(fact["fact_status"] == "candidate_evidence" for fact in section["facts"])
    assert all("gap-0001" not in fact["evidence_ids"] for fact in section["facts"])

    plan = _reduce_prompt_plan(
        tmp_path / "bundle",
        rows,
        {},
        fact_pack=pack,
        max_input_chars=10000,
    )
    assert "review_gap_not_fact" in plan["prompt"]
    assert "不得提升为确定事实" in plan["prompt"]
    assert "不做外部事实裁判" in plan["prompt"]
    assert "gap-0001" in plan["prompt"]


def test_review_only_numeric_claim_stays_review_only_and_fact_pack_is_persisted(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    review_only_section = _workflow_section(
        citations=[
            {
                "citation_id": "gap-number-3300",
                "chunk_kind": "review_gap",
                "time_range": "00:00:30.000 - 00:00:40.000",
                "snippet": "3300 元只有模型候选，没有来源证据",
                "source": "video_rag_chunks",
                "fact_status": "review_gap_not_fact",
            }
        ]
    )
    review_only_section["semantic_correction_items"] = []
    _write_json(root / "manifest.json", {"title": "测试课程"})
    _write_json(
        exports / "smart-summary-section-workflow.json",
        {"sections": [review_only_section]},
    )
    _write_json(
        exports / "smart-summary-section-llm-revisions.json",
        {
            "rows": [
                {
                    "section_id": "chapter-0001",
                    "title": "需求确认",
                    "time_range": "00:00:00.000 - 00:02:00.000",
                    "final_markdown": "本章包含一个尚无来源证据的 3300 元候选数字。",
                }
            ]
        },
    )
    _write_json(exports / "course-map.json", {"mainline": "需求确认"})

    result = run_smart_summary_global_reduce(root, execute=False, write=True)
    pack = json.loads((exports / "smart-summary-chapter-fact-pack.json").read_text(encoding="utf-8"))
    section = pack["sections"][0]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert result["status"] == "planned"
    assert result["reduce_stage"]["evidence_lineage_complete"] is True
    assert result["map_stage"]["review_only_sections"] == 1
    assert section["evidence_status"] == "review_only"
    assert section["eligible_evidence_ids"] == []
    assert all(fact["fact_status"] == "review_gap_not_fact" for fact in section["facts"])
    assert manifest["smart_summary_chapter_fact_pack_json"] == "exports/smart-summary-chapter-fact-pack.json"
    assert manifest["smart_summary_chapter_fact_pack_summary"]["revision"] == pack["revision"]
