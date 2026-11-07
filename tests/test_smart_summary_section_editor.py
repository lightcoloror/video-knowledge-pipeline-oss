from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from video_knowledge_pipeline.smart_summary_section_editor import build_smart_summary_section_editor


def test_smart_summary_section_editor_writes_static_review_ui() -> None:
    root = Path("outputs") / ("pytest-smart-summary-section-editor-" + uuid.uuid4().hex) / "bundle"
    if root.exists():
        shutil.rmtree(root)
    exports = root / "exports"
    exports.mkdir(parents=True)
    try:
        _write_json(root / "manifest.json", {"title": "Editor Smoke", "media_path": str(root / "video.mp4"), "normalized_transcript_json": "normalized-transcript.json"})
        _write_json(
            root / "normalized-transcript.json",
            {
                "segments": [
                    {"start": 0, "end": 5, "text": "第一节介绍客户特点和成交原则。"},
                    {"start": 8, "end": 12, "text": "这里需要形成可以执行的动作。"},
                    {"start": 31, "end": 34, "text": "第二节进入陌客营销的问题链。"},
                ]
            },
        )
        _write_json(
            root / "source-arbitrated-transcript.json",
            {"segments": [{"start": 0, "end": 5, "text": "第一节介绍客户特点和成交原则。"}]},
        )
        _write_json(
            exports / "smart-summary-input-pack.json",
            {
                "transcript_source": str(root / "source-arbitrated-transcript.json"),
                "transcript_source_decision": {
                    "selected_label": "source_arbitrated_transcript",
                    "selected_path": str(root / "source-arbitrated-transcript.json"),
                    "uses_corrected_transcript": True,
                    "priority": "corrected_transcript_preferred",
                    "priority_reason": "source-arbitrated transcript is preferred after term/tool-name arbitration.",
                    "raw_asr_path": str(root / "normalized-transcript.json"),
                },
            },
        )
        _write_json(
            exports / "smart-summary-section-workflow.json",
            {
                "schema": "video_knowledge_pipeline.smart_summary_section_workflow.v1",
                "title": "Editor Smoke",
                "sections": [
                    {
                        "section_id": "chapter-0001",
                        "chapter_index": 1,
                        "title": "客户特点",
                        "start": 0,
                        "end": 15,
                        "start_time": "00:00:00.000",
                        "end_time": "00:00:15.000",
                        "status": "needs_rewrite",
                        "reasons": ["global_summary_quality_failed"],
                        "rewrite_prompt": "重写第一节。",
                        "evidence": {"summary_sentences": ["客户特点候选摘要"], "key_points": ["成交原则"], "semantic_correction_items": [{"candidate_id": "semcorr-0001", "correction_type": "concept", "risk_level": "medium", "time_range": "00:00:02.000 - 00:00:08.000", "original_text": "这里这个很重要大家看一下", "candidate_text": "客户信任建立流程", "reason": "deictic_or_low_information_transcript_with_support_concept", "semantic_attention": True}], "citations": [{"citation_id": "moment-0001", "chunk_kind": "moment", "time_range": "00:00:00.000 - 00:00:15.000", "timeline_indexes": [1], "snippet": "第一节 citation", "evidence_paths": ["frames/frame-0001.jpg"], "source": "video_moment_index", "fact_status": "candidate_evidence"}, {"citation_id": "rag-review-gap-0001", "chunk_kind": "review_gap", "time_range": "00:00:00.000 - 00:00:15.000", "timeline_indexes": [1], "snippet": "OCR 小字待复核", "evidence_paths": ["frames/frame-0001-crop.jpg"], "source": "video_rag_chunks", "fact_status": "review_gap_not_fact"}]},
                        "semantic_correction_items": [{"candidate_id": "semcorr-0001", "correction_type": "concept", "risk_level": "medium", "time_range": "00:00:02.000 - 00:00:08.000", "original_text": "这里这个很重要大家看一下", "candidate_text": "客户信任建立流程", "reason": "deictic_or_low_information_transcript_with_support_concept", "semantic_attention": True}],
                        "citations": [{"citation_id": "moment-0001", "chunk_kind": "moment", "time_range": "00:00:00.000 - 00:00:15.000", "timeline_indexes": [1], "snippet": "第一节 citation", "evidence_paths": ["frames/frame-0001.jpg"], "source": "video_moment_index", "fact_status": "candidate_evidence"}, {"citation_id": "rag-review-gap-0001", "chunk_kind": "review_gap", "time_range": "00:00:00.000 - 00:00:15.000", "timeline_indexes": [1], "snippet": "OCR 小字待复核", "evidence_paths": ["frames/frame-0001-crop.jpg"], "source": "video_rag_chunks", "fact_status": "review_gap_not_fact"}],
                    },
                    {
                        "section_id": "chapter-0002",
                        "chapter_index": 2,
                        "title": "问题链",
                        "start": 30,
                        "end": 45,
                        "start_time": "00:00:30.000",
                        "end_time": "00:00:45.000",
                        "status": "ready",
                        "reasons": [],
                        "rewrite_prompt": "重写第二节。",
                        "evidence": {"actions": ["列出问题链"]},
                    },
                ],
            },
        )
        _write_json(
            exports / "smart-summary-section-todo.json",
            {
                "schema": "video_knowledge_pipeline.smart_summary_section_todo.v1",
                "rows": [{"section_id": "chapter-0001", "draft_markdown": "## 客户特点\n\n待完善。"}],
            },
        )

        result = build_smart_summary_section_editor(root)

        html_path = root / "smart-summary-section-editor.html"
        html = html_path.read_text(encoding="utf-8")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        editor_json = json.loads((root / "smart-summary-section-editor.json").read_text(encoding="utf-8"))
        semantic_review_template = json.loads((exports / "smart-summary-section-semantic-review-notes.template.json").read_text(encoding="utf-8"))
        run_json = json.loads((root / "runs" / "smart-summary-section-editor" / "run.json").read_text(encoding="utf-8"))

        assert result["section_count"] == 2
        assert result["sections"][0]["transcript_excerpt"][0]["text"] == "第一节介绍客户特点和成交原则。"
        assert "smartSummaryEditorData" in html
        assert "下载修订 JSON" in html
        assert "下载语义复核 JSON" in html
        assert "semanticReviewNotesPayload" in html
        assert "transcript-semantic-correction-review-notes.json" in html
        assert "copySemanticReviewImportCommand" in html
        assert "import-transcript-semantic-review-notes" in html
        assert "smart-summary-section-apply" in html
        assert "chapter-0001" in html
        assert "moment-0001" in html
        assert "review_gap_not_fact" in html
        assert "video_rag_chunks" in html
        assert "ASR/字幕语义纠错候选" in html
        assert "semcorr-0001" in html
        assert "客户信任建立流程" in html
        assert result["sections"][0]["semantic_correction_items"][0]["candidate_id"] == "semcorr-0001"
        assert result["sections"][0]["citations"][0]["source"] == "video_moment_index"
        assert result["sections"][0]["citations"][1]["source"] == "video_rag_chunks"
        assert manifest["smart_summary_section_editor_html"] == "smart-summary-section-editor.html"
        assert manifest["smart_summary_section_semantic_review_notes_template"] == "exports/smart-summary-section-semantic-review-notes.template.json"
        assert manifest["mcp_smart_summary_section_editor_args"] == "mcp-smart-summary-section-editor.args.json"
        assert editor_json["operator_boundary"]["no_direct_writeback"] is True
        assert editor_json["artifacts"]["semantic_review_notes_template"].endswith("smart-summary-section-semantic-review-notes.template.json")
        assert semantic_review_template["schema"] == "video_knowledge_pipeline.transcript_semantic_review_notes.v1"
        assert semantic_review_template["review_mode"] == "template_needs_human_decision"
        assert semantic_review_template["reviews"][0]["candidate_id"] == "semcorr-0001"
        assert semantic_review_template["reviews"][0]["status"] == "needs_more_evidence"
        assert semantic_review_template["reviews"][0]["corrected_text"] == "客户信任建立流程"
        assert semantic_review_template["reviews"][0]["human_confirmed"] is False
        assert run_json["status"] == "completed"
    finally:
        if root.exists():
            shutil.rmtree(root.parent, ignore_errors=True)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
