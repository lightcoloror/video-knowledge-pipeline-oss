from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_section_workflow import build_smart_summary_section_workflow


def test_smart_summary_section_workflow_injects_moment_citations(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    try:
        _write_json(root / "manifest.json", {"title": "Citation Smoke"})
        _write_json(
            root / "timeline.json",
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 12,
                    "transcript": "第一节讲客户特点、信任建立和成交原则。",
                    "visual_text": "PPT: 客户特点 / 信任 / 成交原则",
                    "frame_path": "frames/frame-0001.jpg",
                },
                {
                    "index": 2,
                    "start": 30,
                    "end": 42,
                    "transcript": "第二节讲问题链和陌客营销流程。",
                    "visual_understanding": {"summary": "屏幕展示问题链流程图"},
                    "frame_path": "frames/frame-0002.jpg",
                },
            ],
        )
        _write_json(
            exports / "smart-summary-chapters.json",
            {
                "title": "Citation Smoke",
                "chapters": [
                    {
                        "index": 1,
                        "title": "客户特点",
                        "start": 0,
                        "end": 15,
                        "start_time": "00:00:00.000",
                        "end_time": "00:00:15.000",
                        "summary_sentences": ["第一节说明客户特点和信任建立。"],
                        "key_points": [{"time": "00:00:00.000", "text": "先建立信任"}],
                        "actions": [{"time": "00:00:05.000", "text": "记录成交原则"}],
                    },
                    {
                        "index": 2,
                        "title": "问题链",
                        "start": 30,
                        "end": 45,
                        "start_time": "00:00:30.000",
                        "end_time": "00:00:45.000",
                        "summary_sentences": ["第二节说明问题链。"],
                        "key_points": [{"time": "00:00:30.000", "text": "问题链降低信息不对称"}],
                        "actions": [{"time": "00:00:35.000", "text": "按流程提问"}],
                    },
                ],
            },
        )

        (exports / "video-rag-chunks.jsonl").write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False)
                for row in [
                    {
                        "id": "citation-smoke:visual:0001",
                        "text": "Visual evidence: PPT 写着客户特点、信任建立、成交原则。",
                        "metadata": {
                            "chunk_kind": "visual_evidence",
                            "start": 0,
                            "end": 12,
                            "start_time": "00:00:00.000",
                            "end_time": "00:00:12.000",
                            "timeline_indexes": [1],
                            "tags": ["visual_evidence"],
                            "keywords": ["客户特点", "信任"],
                            "has_visual_evidence": True,
                            "has_temporal_evidence": False,
                            "evidence_paths": ["frames/frame-0001.jpg"],
                        },
                    },
                    {
                        "id": "citation-smoke:review-gap:0001",
                        "text": "Review gap: 客户特点这一节有 OCR 小字待复核。",
                        "metadata": {
                            "chunk_kind": "review_gap",
                            "start": 0,
                            "end": 12,
                            "start_time": "00:00:00.000",
                            "end_time": "00:00:12.000",
                            "timeline_indexes": [1],
                            "tags": ["review_gap"],
                            "keywords": ["OCR", "复核"],
                            "has_visual_evidence": True,
                            "has_temporal_evidence": False,
                            "evidence_paths": ["frames/frame-0001-crop.jpg"],
                        },
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = build_smart_summary_section_workflow(root)
        todo = json.loads((exports / "smart-summary-section-todo.json").read_text(encoding="utf-8"))
        moment_index = json.loads((exports / "video-moment-index.json").read_text(encoding="utf-8"))
        run_json = json.loads((root / "runs" / "smart-summary-section-workflow" / "run.json").read_text(encoding="utf-8"))

        assert result["citation_source"]["available"] is True
        assert result["citation_source"]["video_rag_chunk_count"] == 2
        assert result["citation_source"]["video_rag_chunks_by_kind"]["visual_evidence"] == 1
        assert moment_index["summary"]["chunks"] >= 1
        assert result["sections"][0]["citations"][0]["citation_id"] == "moment-0001"
        citation_sources = {row["source"] for row in result["sections"][0]["citations"]}
        citation_kinds = {row.get("chunk_kind") for row in result["sections"][0]["citations"]}
        assert "video_rag_chunks" in citation_sources
        assert "visual_evidence" in citation_kinds
        assert any(row.get("fact_status") == "review_gap_not_fact" for row in result["sections"][0]["citations"])
        assert 1 in result["sections"][0]["citations"][0]["timeline_indexes"]
        assert "证据引用" in result["sections"][0]["rewrite_prompt"]
        assert todo["rows"][0]["citations"][0]["source"] == "video_moment_index"
        assert any(row["source"] == "video_rag_chunks" for row in todo["rows"][0]["citations"])
        assert "citations:1" in (exports / "smart-summary-section-workflow.md").read_text(encoding="utf-8")
        assert run_json["status"] == "needs_input"
        assert "smart-summary-section-editor" in run_json["retry_command"]
        assert run_json["failed_items"][0]["reason"] == "section_revision_pending"
        assert run_json["failed_items"][0]["suggested_next_tool"] == "smart_summary_section_editor"
        assert "smart-summary-section-apply" in run_json["failed_items"][0]["suggested_apply_command"]
        assert run_json["failed_items"][0]["citation_count"] >= 1
    finally:
        # The managed pytest runtime owns and removes tmp_path.
        pass


def test_smart_summary_section_workflow_builds_video_rag_chunks_when_missing(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    try:
        _write_json(root / "manifest.json", {"title": "Auto RAG Smoke"})
        _write_json(
            root / "timeline.json",
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 18,
                    "transcript": "这一节讲成交原则和信任动作。",
                    "visual_text": "PPT: 成交原则 / 信任动作",
                    "quality_issues": ["missing_visual_text"],
                    "frame_path": "frames/frame-0001.jpg",
                }
            ],
        )
        _write_json(
            exports / "smart-summary-chapters.json",
            {
                "title": "Auto RAG Smoke",
                "chapters": [
                    {
                        "index": 1,
                        "title": "成交原则",
                        "start": 0,
                        "end": 20,
                        "start_time": "00:00:00.000",
                        "end_time": "00:00:20.000",
                        "summary_sentences": ["本节说明成交原则。"],
                        "key_points": [{"text": "信任是成交前提"}],
                        "actions": [{"text": "记录信任动作"}],
                    }
                ],
            },
        )

        result = build_smart_summary_section_workflow(root)

        assert (exports / "video-rag-chunks.jsonl").exists()
        assert result["citation_source"]["video_rag_chunk_count"] >= 1
        assert any(row["source"] == "video_rag_chunks" for row in result["sections"][0]["citations"])
    finally:
        # The managed pytest runtime owns and removes tmp_path.
        pass

def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_smart_summary_section_workflow_injects_semantic_correction_candidates(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    try:
        _write_json(root / "manifest.json", {"title": "Semantic Section Smoke", "normalized_transcript_json": "normalized-transcript.json"})
        _write_json(
            root / "timeline.json",
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 12,
                    "transcript": "这里这个很重要大家看一下",
                    "visual_text": "客户信任建立流程：确认需求，给出解决方案",
                },
                {"index": 2, "start": 60, "end": 70, "transcript": "第二节讲跟进复盘。"},
            ],
        )
        _write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 12, "text": "这里这个很重要大家看一下"}, {"start": 60, "end": 70, "text": "第二节讲跟进复盘。"}]})
        _write_json(
            exports / "smart-summary-chapters.json",
            {
                "title": "Semantic Section Smoke",
                "chapters": [
                    {
                        "index": 1,
                        "title": "客户信任",
                        "start": 0,
                        "end": 20,
                        "start_time": "00:00:00.000",
                        "end_time": "00:00:20.000",
                        "summary_sentences": ["这一节提示看屏幕上的客户信任流程。"],
                        "key_points": [{"text": "客户信任是成交前提"}],
                        "actions": [{"text": "确认需求"}],
                    },
                    {
                        "index": 2,
                        "title": "跟进复盘",
                        "start": 60,
                        "end": 80,
                        "start_time": "00:01:00.000",
                        "end_time": "00:01:20.000",
                        "summary_sentences": ["第二节说明跟进复盘。"],
                        "key_points": [{"text": "持续记录"}],
                        "actions": [{"text": "沉淀清单"}],
                    },
                ],
            },
        )
        _write_json(
            exports / "smart-summary-input-pack.json",
            {
                "schema": "video_knowledge_pipeline.smart_summary_input_pack.v1",
                "transcript_semantic_correction": {
                    "exists": True,
                    "final_status": "needs_codex_or_llm_review",
                    "pack_path": str(root / "transcript-semantic-correction-pack.json"),
                    "candidate_count": 1,
                    "semantic_attention_count": 1,
                },
            },
        )
        _write_json(
            root / "transcript-semantic-correction-pack.json",
            {
                "status": "pack_ready",
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "semcorr-0001",
                        "correction_type": "concept",
                        "risk_level": "medium",
                        "start": 2,
                        "end": 8,
                        "time_range": "00:00:02.000 - 00:00:08.000",
                        "original_text": "这里这个很重要大家看一下",
                        "candidate_text": "客户信任建立流程",
                        "reason": "deictic_or_low_information_transcript_with_support_concept",
                    }
                ],
            },
        )
        _write_json(root / "transcript-semantic-correction-status.json", {"semantic_attention_items": [{"candidate_id": "semcorr-0001", "priority_score": 130}]})

        result = build_smart_summary_section_workflow(root)
        todo = json.loads((exports / "smart-summary-section-todo.json").read_text(encoding="utf-8"))
        markdown = (exports / "smart-summary-section-workflow.md").read_text(encoding="utf-8")

        first = result["sections"][0]
        second = result["sections"][1]
        assert result["transcript_semantic_correction"]["final_status"] == "needs_codex_or_llm_review"
        assert result["semantic_attention_candidate_count"] == 1
        assert "transcript_semantic_correction_pending" in first["reasons"]
        assert first["semantic_correction_items"][0]["candidate_id"] == "semcorr-0001"
        assert "客户信任建立流程" in first["rewrite_prompt"]
        assert "ASR/字幕语义纠错状态：needs_codex_or_llm_review" in first["rewrite_prompt"]
        assert not second["semantic_correction_items"]
        assert todo["rows"][0]["semantic_correction_items"][0]["candidate_id"] == "semcorr-0001"
        assert "ASR/subtitle semantic correction" in markdown
        assert "semantic_correction_items:1" in markdown
    finally:
        # The managed pytest runtime owns and removes tmp_path.
        pass
