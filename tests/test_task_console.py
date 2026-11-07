from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import audit_bundle_mcp_args, run_mcp_call
from video_knowledge_pipeline.run_artifact_registry import register_bundle_run
from video_knowledge_pipeline.task_console import _render_task_console_html, _run_queue_group, export_subqueue_action_plan, export_task_console
from video_knowledge_pipeline.term_correction_status import term_correction_status
from video_knowledge_pipeline.transcript_semantic_correction import build_transcript_semantic_correction_pack, transcript_semantic_correction_status, validate_transcript_semantic_correction
from video_knowledge_pipeline.webui_bridge import export_webui_bundle


def _write_minimal_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "UI smoke lesson",
                "review_html": "review.html",
                "media_path": "lesson.mp4",
                "knowledge_coverage_markdown": "knowledge-coverage.md",
                "knowledge_note_markdown": "exports/knowledge-note.md",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 4,
                    "transcript": "老师说看这个界面。",
                    "visual_route": "semantic_frame",
                    "visual_text": "工具名称",
                    "tagger_tags": ["工具名", "操作演示"],
                    "needs_human_review": True,
                },
                {
                    "index": 2,
                    "start": 4,
                    "end": 8,
                    "transcript": "接下来演示步骤。",
                    "visual_route": "temporal_sequence",
                    "structured_visual": {"text": "步骤一"},
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review.html").write_text("<html>review</html>", encoding="utf-8")
    (bundle / "lesson.mp4").write_bytes(b"fake video")


def test_processing_queue_groups_external_reuse_run_types() -> None:
    cases = {
        "funasr_python_runner": "asr_transcript",
        "transcript_source_arbitration": "asr_transcript",
        "asr_env_status": "asr_transcript",
        "platform_subtitle_import": "asr_transcript",
        "transcript_correction_pack": "asr_transcript",
        "term_correction_impact_report": "asr_transcript",
        "prepare_transcript_edit_session": "asr_transcript",
        "apply_transcript_edits": "asr_transcript",
        "visual_structure_ebook": "document_ocr",
        "high_res_tile_plan": "document_ocr",
        "tile_result_import_build": "document_ocr",
        "tile_result_merge": "document_ocr",
        "vision_review_queue": "vision",
        "multimodal_frame_analysis": "vision",
        "temporal_visual_analysis": "vision",
        "local_vlm_serving_smoke": "vision",
        "timeline_alignment_audit": "timeline_rag",
        "video_moment_index": "timeline_rag",
        "video_rag_search": "timeline_rag",
        "long_video_memory_pack": "timeline_rag",
        "frame_recapture_plan": "timeline_rag",
        "supplemental_frame_sampling": "timeline_rag",
        "smart_summary_input_pack": "summary_export",
        "smart_summary_chapter_pack": "summary_export",
        "smart_summary_section_workflow": "summary_export",
        "smart_summary_section_editor": "summary_export",
        "smart_summary_section_apply": "summary_export",
        "smart_summary_codex": "summary_export",
        "knowledge_note_export": "summary_export",
        "bilinote_mind_map_prompt_pack": "summary_export",
        "external_capability_pack": "summary_export",
        "review_closure_status": "review",
        "multimodal_sample_review": "review",
        "scene_candidate_review": "review",
        "human_sample_eval": "review",
        "prepare_review_session": "review",
        "human_review_import": "review",
    }
    for run_type, expected in cases.items():
        assert _run_queue_group({"run_type": run_type, "run_id": run_type, "title": run_type}) == expected

def test_term_correction_status_prioritizes_failed_codex_validation(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "term-correction-closure.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_correction_closure.v1",
                "status": "needs_term_review",
                "semantic_review_status": "codex_validation_failed",
                "term_validation_status": "no_accepted_decisions",
                "accepted_validation_decisions": 0,
                "rejected_validation_decisions": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-codex-validation.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_validation.v1",
                "status": "no_accepted_decisions",
                "ok": False,
                "accepted_decision_count": 0,
                "rejected_decision_count": 2,
                "rejected_decisions": [
                    {
                        "candidate_id": "term-1",
                        "canonical": "UnsafeTerm",
                        "confidence": 0.97,
                        "rejection_reasons": ["missing_semantic_rationale", "missing_evidence_indexes"],
                    },
                    {
                        "candidate_id": "term-2",
                        "canonical": "UnknownTerm",
                        "confidence": 0.96,
                        "rejection_reasons": ["missing_semantic_rationale", "unknown_candidate_id"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    status = term_correction_status(bundle)

    assert status["status"] == "needs_codex_term_validation"
    assert status["next_action_key"] == "term_arbitration_codex_validate"
    assert status["term_validation_status"] == "no_accepted_decisions"
    assert status["accepted_validation_decisions"] == 0
    assert status["rejected_validation_decisions"] == 2
    assert status["validation_rejection_reasons"][0] == {"reason": "missing_semantic_rationale", "count": 2}
    assert {row["reason"] for row in status["validation_rejection_reasons"]} >= {"missing_evidence_indexes", "unknown_candidate_id"}
    assert status["validation_rejected_decisions"][0]["canonical"] == "UnsafeTerm"
    assert "missing_evidence_indexes" in status["validation_rejected_decisions"][0]["rejection_reasons"]
    assert status["artifacts"]["term_validation_markdown"].endswith("term-arbitration-codex-validation.md")
    assert status["artifacts"]["term_prompt_markdown"].endswith("term-arbitration-codex-prompt.md")
    assert status["artifacts"]["term_result_codex_markdown"].endswith("term-arbitration-codex-result.codex.md")
    assert status["codex_substitute"]["mode"] == "codex_substitute_for_online_text_llm"
    assert status["codex_substitute"]["online_llm_api_required"] is False
    assert status["codex_substitute"]["validation_required"] is True
    assert status["codex_substitute"]["next_action_key"] == "term_arbitration_codex_validate"
    assert "validate-term-arbitration-codex-result" in status["codex_substitute"]["commands"]["validate_result"]
    assert "term-arbitration-codex-result.codex.md" in status["codex_substitute"]["commands"]["import_and_close"]
    html = _render_task_console_html({
        "title": "Term Status Console",
        "bundle_dir": str(bundle),
        "status": {"counts": {}, "term_correction": status},
        "artifacts": [],
        "commands": [],
    })
    assert "Codex 术语预检需要补证据" in html
    assert "missing_semantic_rationale" in html
    assert "UnsafeTerm" in html


def test_task_console_shows_transcript_semantic_correction_details(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Semantic UI", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 4, "text": "今天讲 browser base"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 0, "start": 0, "end": 4, "transcript": "今天讲 browser base", "visual_text": "Browserbase"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "exports").mkdir(exist_ok=True)
    (bundle / "exports" / "smart-summary-chapters.json").write_text(
        json.dumps({"chapters": [{"index": 1, "title": "浏览器自动化工具", "start": 0, "end": 10, "start_time": "00:00:00.000", "end_time": "00:00:10.000"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    pack = build_transcript_semantic_correction_pack(bundle, write=True)
    candidate = pack["candidates"][0]
    result_path = bundle / "bad-semantic-result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
                "decisions": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "action": "replace",
                        "correction_type": candidate["correction_type"],
                        "original_text": candidate["original_text"],
                        "corrected_text": "Browserbase",
                        "confidence": 0.5,
                        "semantic_rationale": "证据不足，应该进入人工复核。",
                        "evidence_ids": candidate["evidence_ids"],
                        "safe_to_apply": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    validate_transcript_semantic_correction(bundle, input_json=result_path, write=True)
    status = transcript_semantic_correction_status(bundle, write=True)
    html = _render_task_console_html(
        {
            "title": "Semantic Correction Console",
            "bundle_dir": str(bundle),
            "status": {"counts": {}, "semantic_correction": status},
            "artifacts": [],
            "commands": [],
        }
    )

    assert "通用语义纠错闭环进度摘要" in html
    assert "自动候选" in html
    assert "需人工候选" in html
    assert "候选类型" in html
    assert "证据来源" in html
    assert "来源投票 / 字幕可靠性摘要" in html
    assert "semantic-source-vote" in html
    assert "预检拒绝原因" in html
    assert "待人工复核样例" in html
    assert "按章节/风险分组" in html
    assert "浏览器自动化工具" in html
    assert "confidence_below_threshold" in html
    assert "Browserbase" in html
    assert "候选分组预览" in html
    assert "proper_noun" in html
    assert "语义纠错人工编辑表单" in html
    assert "copySemanticReviewNotes" in html
    assert "downloadSemanticReviewNotes" in html
    assert "transcript-semantic-correction-review-notes.json" in html
    assert "LLM/Codex 草稿" in html
    assert "LLM 下一步" in html
    status["review_closure_summary"] = {
        "review_result_imported": True,
        "imported_review_decision_count": 2,
        "accepted_imported_review_decision_count": 1,
        "rejected_imported_review_decision_count": 1,
        "skipped_review_note_count": 1,
        "closed_review_decision_count": 1,
        "open_review_required_count": 1,
        "validation_status": "review_required",
        "next_action_key": "run_transcript_semantic_correction_closure",
        "next_action_command": ".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-closure bundle --input-json result.review.json",
        "actions": {"replace": 1, "needs_human_review": 1},
        "skipped": [{"row_number": 2, "candidate_id": "missing-candidate", "reason": "missing_or_unknown_candidate_or_no_review_decision"}],
    }
    status.update(
        {
            "closure_status": "completed",
            "closure_ok": True,
            "closure_applied_correction_count": 1,
            "closure_changed_segment_count": 1,
            "corrected_transcript_exists": True,
            "corrected_transcript_path": str(bundle / "source-arbitrated-transcript.json"),
            "summary_impact_status": "passed",
            "summary_impact_ok": True,
            "summary_absorption_rate": 1.0,
            "summary_residual_original_total": 0,
            "commands": {"summary_impact": ".\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report bundle"},
        }
    )
    html = _render_task_console_html(
        {
            "title": "Semantic Correction Console",
            "bundle_dir": str(bundle),
            "status": {"counts": {}, "semantic_correction": status},
            "artifacts": [],
            "commands": [],
        }
    )
    assert "语义纠错复核导入结果" in html
    assert "预检接受" in html
    assert "预检拒绝" in html
    assert "导入跳过" in html
    assert "已关闭" in html
    assert "仍待处理" in html
    assert "run_transcript_semantic_correction_closure" in html
    assert "missing-candidate" in html
    assert "语义纠错导出闭环" in html
    assert "纠正版 transcript" in html
    assert "Summary impact" in html
    assert "吸收率" in html


def test_task_console_shows_transcript_semantic_repair_run_progress(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-repair-run"
    bundle.mkdir(parents=True)
    html = _render_task_console_html(
        {
            "title": "Semantic Repair Run Console",
            "bundle_dir": str(bundle),
            "status": {
                "counts": {},
                "semantic_repair_queue": {
                    "status": "machine_actions_available",
                    "bundle_count": 1,
                    "summary": {"action_required_count": 1, "machine_action_available_count": 1, "human_review_required_count": 0},
                    "json_path": str(bundle / "exports" / "transcript-semantic-repair-queue.json"),
                    "markdown_path": str(bundle / "exports" / "transcript-semantic-repair-queue.md"),
                    "operator_boundary": {"preview_only": True},
                    "items": [
                        {
                            "bundle_dir": str(bundle),
                            "action_key": "run_impact",
                            "action_status": "needs_execution",
                            "action_kind": "local_report",
                            "semantic_status": "needs_impact_report",
                            "llm_draft_status": "prompt_ready",
                            "human_review_required": False,
                            "retry_command": "retry semantic impact",
                            "reason": "impact report missing",
                            "progress": {"percent": 62},
                        }
                    ],
                },
                "semantic_repair_run": {
                    "status": "completed_with_errors",
                    "json_path": str(bundle / "exports" / "transcript-semantic-repair-run.json"),
                    "markdown_path": str(bundle / "exports" / "transcript-semantic-repair-run.md"),
                    "summary": {"action_count": 2, "executed_count": 1, "planned_count": 0, "failed_count": 1, "operator_required_count": 1},
                    "executions": [
                        {"action_key": "run_impact", "run_status": "failed", "executed": False, "error": "FileNotFoundError: impact source missing"},
                        {"action_key": "review_candidates", "run_status": "skipped_operator_required", "executed": False, "reason": "Human review required"},
                    ],
                },
            },
            "artifacts": [],
            "commands": [],
            "bridge": {},
        }
    )

    assert "通用语义纠错修复/重试队列" in html
    assert "语义纠错最新执行结果" in html
    assert "completed_with_errors" in html
    assert "FileNotFoundError: impact source missing" in html
    assert "skipped_operator_required" in html
    assert "transcript-semantic-repair-run.json" in html



def test_task_console_shows_semantic_batch_review_pack_panel(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-batch-review-panel"
    bundle.mkdir(parents=True)
    exports = bundle / "exports"
    exports.mkdir(parents=True)
    html = _render_task_console_html(
        {
            "title": "Semantic Batch Review Console",
            "bundle_dir": str(bundle),
            "status": {
                "counts": {},
                "semantic_batch_review": {
                    "status": "codex_draft_ready",
                    "review_item_count": 12,
                    "todo_review_count": 12,
                    "draft_review_count": 12,
                    "draft_by_review_status": {"accept_correction": 2, "needs_more_evidence": 10},
                    "imported_decision_count": 5,
                    "imported_accepted_decision_count": 2,
                    "imported_review_required_count": 3,
                    "imported_closure_ready_bundle_count": 1,
                    "imported_open_review_bundle_count": 1,
                    "import_by_validation_status": {"accepted_with_rejections": 1},
                    "import_post_next_action_counts": {"run_closure": 1},
                    "import_next_actions": ["run closure after import"],
                    "skipped_count": 0,
                    "editable_reviews": [
                        {
                            "review_id": "b001-semcorr-0001",
                            "bundle_dir": str(bundle),
                            "bundle_title": "Semantic Batch Review Console",
                            "candidate_id": "semcorr-0001",
                            "correction_type": "proper_noun",
                            "risk_level": "medium",
                            "time_range": {"start": 1, "end": 3},
                            "original_text": "playright",
                            "suggested_text": "Playwright",
                            "context_text": "playright client",
                            "evidence": [{"evidence_id": "ev-1", "source_type": "page_metadata", "text": "Playwright client", "path": str(exports / "metadata.json")}],
                            "evidence_ids": ["ev-1"],
                            "review_status": "accept_correction",
                            "corrected_text": "Playwright",
                            "confidence": 0.95,
                            "review_note": "metadata supports Playwright",
                        }
                    ],
                    "editable_review_count": 1,
                    "skipped_count": 0,
                    "paths": {
                        "review_pack_json": str(exports / "transcript-semantic-batch-review-pack.json"),
                        "todo_json": str(exports / "transcript-semantic-batch-review-notes.todo.json"),
                        "codex_draft_json": str(exports / "transcript-semantic-batch-review-notes.codex-draft.json"),
                    },
                    "commands": {
                        "build_review_pack": "build batch review pack",
                        "codex_review_draft": "build codex draft",
                        "import_codex_draft": "import codex draft",
                    },
                },
            },
            "artifacts": [],
            "commands": [],
            "bridge": {},
        }
    )

    assert "通用语义纠错批量复核包" in html
    assert "codex_draft_ready" in html
    assert "accept_correction" in html
    assert "needs_more_evidence" in html
    assert "Still review open" in html
    assert "accepted_with_rejections" in html
    assert "run closure after import" in html
    assert "批量复核编辑器" in html
    assert "semantic-batch-review-row" in html
    assert "collectSemanticBatchReviewNotes" in html
    assert "downloadSemanticBatchReviewNotes" in html
    assert "filterSemanticBatchReviews" in html
    assert "setSemanticBatchVisibleStatus" in html
    assert "semanticBatchStatusFilter" in html
    assert "semanticBatchRiskFilter" in html
    assert "semanticBatchVisibleCount" in html
    assert "data-risk-level" in html
    assert "data-review-status" in html
    assert "data-start-seconds=\"1.000\"" in html
    assert "seekToSemanticBatchReview" in html
    assert "播放此处" in html
    assert "metadata.json" in html
    assert "b001-semcorr-0001" in html
    assert "ev-1" in html
    assert "transcript-semantic-batch-review-pack.json" in html
    assert "build batch review pack" in html
    assert "import codex draft" in html

def test_task_console_shows_semantic_attention_items_for_concept_candidates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-attention"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Semantic Attention", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 5, "text": "这里这个很重要大家看一下"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "start": 0,
                    "end": 5,
                    "visual_text": "客户信任建立流程：确认需求，给出解决方案",
                    "tagger_tags": ["步骤", "概念"],
                    "tagger_annotations": [{"text": "重点概念：客户信任建立流程"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "exports").mkdir(exist_ok=True)
    build_transcript_semantic_correction_pack(bundle, write=True)
    status = transcript_semantic_correction_status(bundle, write=True)

    html = _render_task_console_html(
        {
            "title": "Semantic Attention Console",
            "bundle_dir": str(bundle),
            "status": {"counts": {}, "semantic_correction": status},
            "artifacts": [],
            "commands": [],
        }
    )

    assert "语义重点复核队列" in html
    assert "concept" in html
    assert "客户信任建立流程" in html
    assert "deictic_or_low_information_transcript_with_support_concept" in html


def test_task_console_semantic_editor_exposes_split_merge_fields() -> None:
    html = _render_task_console_html(
        {
            "title": "Semantic Structure Editor",
            "bundle_dir": "D:/tmp/bundle",
            "status": {
                "counts": {},
                "semantic_correction": {
                    "status": "needs_human_review_or_new_result",
                    "review_required_items": [
                        {
                            "candidate_id": "semcorr-0001",
                            "correction_type": "segment_boundary",
                            "time_range": "00:00:10.000 - 00:00:42.000",
                            "start": 10,
                            "end": 42,
                            "segment_index": 3,
                            "original_text": "第一步先分析客户特点然后建立信任第二步再确认需求",
                            "suggested_text": "第一步，先分析客户特点，然后建立信任。第二步，再确认需求。",
                            "reject_reasons": ["split_segments_require_human_confirmation"],
                        }
                    ],
                },
            },
            "artifacts": [],
            "commands": [],
        }
    )

    assert "结构化断句/合并" in html
    assert 'data-field="segments"' in html
    assert 'data-field="merge_segment_indexes"' in html
    assert 'placeholder="3,4"' in html
    assert "parseJsonArrayField" in html
    assert "review.segments = segments" in html
    assert "review.merge_segment_indexes = mergeIndexes" in html

def test_export_task_console_writes_human_ui_and_agent_json(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_minimal_bundle(bundle)
    register_bundle_run(
        bundle,
        run_type="funasr_python_runner",
        run_id="funasr-run",
        status="needs_execution",
        title="FunASR local run",
        summary="Local SenseVoice/FunASR transcript is pending.",
        failed_items=[{"index": "audio", "reason": "asr_pending"}],
        retry_command="retry local asr",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="asr_env_status",
        run_id="asr-env-status",
        status="needs_retry",
        title="ASR env status",
        summary="Local ASR model cache needs checking.",
        failed_items=[{"index": "sensevoice", "reason": "model_not_ready"}],
        retry_command="retry asr env",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="platform_subtitle_import",
        run_id="platform-subtitle-import",
        status="needs_input",
        title="Platform subtitle import",
        summary="Platform subtitle sidecar needs import.",
        failed_items=[{"index": "subtitle", "reason": "subtitle_missing"}],
        retry_command="retry subtitle import",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="transcript_source_arbitration",
        run_id="transcript-source-arbitration",
        status="needs_review",
        title="Transcript source arbitration",
        summary="ASR and platform subtitle disagree on key terms.",
        failed_items=[{"index": "term-1", "reason": "source_conflict"}],
        retry_command="retry transcript source arbitration",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="transcript_correction_pack",
        run_id="transcript-correction-pack",
        status="needs_review",
        title="Transcript correction pack",
        summary="Terminology correction pack needs review.",
        failed_items=[{"index": "term-2", "reason": "terminology_conflict"}],
        retry_command="retry transcript correction",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="term_arbitration_codex",
        run_id="term-arbitration-codex",
        status="needs_input",
        title="Codex terminology arbitration",
        summary="Codex semantic term arbitration needs reviewed JSON import.",
        failed_items=[{"index": "codex", "reason": "codex_review_required"}],
        retry_command="retry codex term arbitration",
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="term_correction_impact_report",
        run_id="term-correction-impact-report",
        status="needs_retry",
        title="Term correction impact report",
        summary="Final exports still contain terminology aliases.",
        failed_items=[{"index": "final-export", "reason": "final_export_alias_remaining"}],
        retry_command="retry term impact report",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="prepare_transcript_edit_session",
        run_id="prepare-transcript-edit-session",
        status="needs_input",
        title="Prepare transcript edit session",
        summary="Manual transcript edit session needs edited notes.",
        failed_items=[{"index": "edit", "reason": "missing_transcript_edits"}],
        retry_command="retry transcript editor",
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="ebook_batch",
        run_id="ebook-batch-1",
        status="needs_retry",
        title="Ebook batch",
        summary="2 screenshots need retry.",
        artifacts=[{"key": "report", "path": "ebook-batch.md"}],
        failed_items=[{"index": 1, "reason": "ocr_text_empty", "detail": "empty whole-frame OCR", "suggested_next_tool": "high_res_tile_plan", "suggested_retry_command": "retry high-res tile for ebook", "ebook_retry_command": "retry ebook index 1", "multimodal_triage_command": "retry vision triage index 1", "review_command": "retry review pack", "evidence_paths": ["frames/0001.jpg", "visual-structure/timeline-0001"]}],
        retry_command="retry ebook batch",
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="tile_result_import_build",
        run_id="tile-result-import-build",
        status="needs_input",
        title="Tile result import build",
        summary="1 tile still needs OCR/VLM/human output.",
        artifacts=[{"key": "report", "path": "tile-result-import.md"}],
        failed_items=[{"index": 2, "reason": "tile_result_pending", "tile_id": "0002-01", "detail": "No result file found for tile 0002-01.", "tile_result_import_command": "retry tile result import item", "tile_result_merge_command": "retry tile result merge after import"}],
        retry_command="retry tile import build",
        next_actions=["Generate OCR/VLM/human result files for pending tiles, then rerun tile-result-import-build."],
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="vision_review_queue",
        run_id="vision-review-queue",
        status="needs_execution",
        title="Vision review queue",
        summary="10 hard frames need review queue execution.",
        failed_items=[{"index": 1, "reason": "semantic_frame_without_analysis"}],
        retry_command="retry vision review queue",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="multimodal_frame_analysis",
        run_id="semantic-frame-analysis",
        status="needs_execution",
        title="Semantic frame analysis",
        summary="Semantic frame candidates need multimodal review.",
        failed_items=[{"index": 1, "reason": "semantic_frame_without_analysis"}],
        retry_command="retry semantic vision",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="temporal_visual_analysis",
        run_id="temporal-visual-analysis",
        status="needs_execution",
        title="Temporal visual analysis",
        summary="Temporal frame groups need multimodal review.",
        failed_items=[{"index": 2, "reason": "temporal_sequence_without_analysis"}],
        retry_command="retry temporal vision",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="vision_provider_smoke",
        run_id="vision-provider-smoke",
        status="needs_retry",
        title="Vision provider smoke",
        summary="Configured provider is not ready.",
        failed_items=[{"index": "provider", "reason": "missing_api_key"}],
        retry_command="retry provider smoke",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="local_vlm_serving_smoke",
        run_id="local-vlm-serving-smoke",
        status="needs_execution",
        title="Local VLM serving smoke",
        summary="Local VLM smoke is plan-only until explicitly executed.",
        failed_items=[{"index": "local-vlm", "reason": "needs_execution"}],
        retry_command="retry local vlm smoke",
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="video_moment_index",
        run_id="video-moment-index-run",
        status="needs_retry",
        title="Video moment index",
        summary="Moment index needs refresh after timeline changes.",
        failed_items=[{"index": "moment", "reason": "moment_index_stale"}],
        retry_command="retry video moment index",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="video_rag_search",
        run_id="video-rag-search-run",
        status="needs_execution",
        title="VideoRAG search",
        summary="VideoRAG search pack needs execution.",
        failed_items=[{"index": "rag", "reason": "rag_index_missing"}],
        retry_command="retry video rag",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="long_video_memory_pack",
        run_id="long-video-memory-pack",
        status="needs_retry",
        title="Long video memory pack",
        summary="Long video memory pack needs rebuild.",
        failed_items=[{"index": "memory", "reason": "memory_pack_stale"}],
        retry_command="retry long video memory",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="frame_recapture_plan",
        run_id="frame-recapture-plan",
        status="needs_execution",
        title="Frame recapture plan",
        summary="Supplemental frames need recapture.",
        failed_items=[{"index": 2, "reason": "recapture_needed"}],
        retry_command="retry frame recapture",
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="smart_summary_input_pack",
        run_id="smart-summary-input-pack",
        status="needs_retry",
        title="Smart summary input pack",
        summary="Summary input pack needs refreshed evidence.",
        failed_items=[{"index": "input", "reason": "input_pack_stale"}],
        retry_command="retry summary input pack",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="smart_summary_section_workflow",
        run_id="smart-summary-section-workflow",
        status="needs_input",
        title="Smart summary section workflow",
        summary="Section workflow has TODO revisions.",
        failed_items=[{"index": "section-1", "reason": "section_revision_pending"}],
        retry_command="retry section workflow",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="smart_summary_section_apply",
        run_id="smart-summary-section-apply",
        status="needs_input",
        title="Smart summary section apply",
        summary="Section apply needs revision JSON.",
        failed_items=[{"index": "section-apply", "reason": "missing_revision_json"}],
        retry_command="retry section apply",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="knowledge_note_export",
        run_id="knowledge-note-export",
        status="needs_retry",
        title="Knowledge note export",
        summary="Knowledge export is stale.",
        failed_items=[{"index": "export", "reason": "export_stale"}],
        retry_command="retry knowledge export",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="content_candidate_pack",
        run_id="content-candidate-pack",
        status="needs_review",
        title="Content candidate pack",
        summary="Content candidate pack needs review before handoff.",
        failed_items=[{"index": "candidate-1", "reason": "candidate_review_required"}],
        retry_command="retry content candidates",
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="prepare_review_session",
        run_id="review-pack",
        status="needs_review",
        title="Review pack",
        summary="Review pack has open screen-text items.",
        failed_items=[{"index": 1, "reason": "missing_visual_text"}],
        retry_command="retry review pack",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="transcript_arbitration_review",
        run_id="transcript-arbitration-review",
        status="needs_review",
        title="Transcript arbitration review",
        summary="Transcript arbitration has low confidence conflicts.",
        failed_items=[{"index": 1, "reason": "low_confidence_conflict"}],
        retry_command="retry transcript arbitration review",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="multimodal_sample_review",
        run_id="multimodal-sample-review",
        status="needs_review",
        title="Multimodal sample review",
        summary="Human sample eval is pending.",
        failed_items=[{"index": "sample-1", "reason": "sample_eval_pending"}],
        retry_command="retry sample review",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="review_closure_status",
        run_id="review-closure-status",
        status="needs_retry",
        title="Review closure status",
        summary="Review closure still has open targets.",
        failed_items=[{"index": "closure", "reason": "review_targets_open"}],
        retry_command="retry review closure",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="human_review_import",
        run_id="human-review-import",
        status="needs_input",
        title="Human review import",
        summary="Human review import needs notes JSON.",
        failed_items=[{"index": "notes", "reason": "missing_review_notes"}],
        retry_command="retry human review import",
        write=True,
    )
    result = export_task_console(bundle, write=True, refresh=False)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    console = json.loads((bundle / "task-console.json").read_text(encoding="utf-8"))
    html = (bundle / "task-console.html").read_text(encoding="utf-8")
    settings_html = (bundle / "model-settings.html").read_text(encoding="utf-8")
    settings_json = json.loads((bundle / "model-settings.json").read_text(encoding="utf-8"))
    command_keys = {item["key"] for item in console["commands"]}
    commands_by_key = {item["key"]: item for item in console["commands"]}

    assert result["schema"] == "video_knowledge_pipeline.task_console.v1"
    assert result["model_settings_html_path"].endswith("model-settings.html")
    assert result["model_settings_json_path"].endswith("model-settings.json")
    assert manifest["task_console"] == "task-console.html"
    assert manifest["task_console_json"] == "task-console.json"
    assert manifest["model_settings"] == "model-settings.html"
    assert manifest["model_settings_json"] == "model-settings.json"
    assert settings_json["asr_runtime"]["provider"] == "funasr_sensevoice"
    assert any(row["provider"] == "speaches_openai_compatible" for row in settings_json["asr_service_adapters"])
    assert "ASR Runtime 设置" in settings_html
    assert "Speaches OpenAI-compatible ASR" in settings_html
    assert "Whisper-WebUI" in settings_html
    assert "Buzz" in settings_html
    assert "set-asr-runtime-profile" in settings_html
    assert manifest["mcp_export_task_console_args"] == "mcp-export-task-console.args.json"
    assert manifest["video_moment_index"] == "exports/video-moment-index.json"
    assert manifest["video_moment_index_markdown"] == "exports/video-moment-index.md"
    assert manifest["mcp_video_moment_index_args"] == "mcp-video-moment-index.args.json"
    assert manifest["mcp_video_rag_search_args"] == "mcp-video-rag-search.args.json"
    assert manifest["mcp_video_rag_service_plan_args"] == "mcp-video-rag-service-plan.args.json"
    assert manifest["mcp_prepare_transcript_edit_session_args"] == "mcp-prepare-transcript-edit-session.args.json"
    assert manifest["readable_transcript_json"] == "readable-transcript.json"
    assert manifest["llm_readable_transcript_json"] == "llm-readable-transcript.json"
    assert manifest["mcp_postprocess_asr_transcript_args"] == "mcp-postprocess-asr-transcript.args.json"
    assert manifest["mcp_readable_transcript_llm_polish_args"] == "mcp-readable-transcript-llm-polish.args.json"
    assert (bundle / "mcp-readable-transcript-llm-polish.args.json").exists()
    assert manifest["agent_readable_transcript_json"] == "agent-readable-transcript.json"
    assert manifest["agent_readable_transcript_markdown"] == "agent-readable-transcript.md"
    assert manifest["agent_readable_transcript_rewrite_json"] == "agent-readable-transcript-rewrite.json"
    assert manifest["agent_readable_transcript_rewrite_markdown"] == "agent-readable-transcript-rewrite.md"
    assert manifest["agent_readable_transcript_task_json"] == "agent-readable-transcript-task.json"
    assert manifest["agent_readable_transcript_task_markdown"] == "agent-readable-transcript-task.md"
    assert manifest["transcript_quality_gate_json"] == "transcript-quality-gate.json"
    assert manifest["transcript_quality_gate_markdown"] == "transcript-quality-gate.md"
    assert manifest["mcp_agent_readable_transcript_rewrite_args"] == "mcp-agent-readable-transcript-rewrite.args.json"
    assert manifest["mcp_transcript_quality_gate_args"] == "mcp-transcript-quality-gate.args.json"
    assert manifest["mcp_plan_local_asr_service_args"] == "mcp-plan-local-asr-service.args.json"
    assert manifest["mcp_run_local_asr_service_plan_args"] == "mcp-run-local-asr-service-plan.args.json"
    assert (bundle / "mcp-plan-local-asr-service.args.json").exists()
    assert (bundle / "mcp-run-local-asr-service-plan.args.json").exists()
    assert (bundle / "mcp-agent-readable-transcript-rewrite.args.json").exists()
    assert (bundle / "mcp-transcript-quality-gate.args.json").exists()
    agent_rewrite_args = json.loads((bundle / "mcp-agent-readable-transcript-rewrite.args.json").read_text(encoding="utf-8"))
    assert agent_rewrite_args["bundle_dir"] == str(bundle)
    assert agent_rewrite_args["agent_name"] == "local_agent"
    assert agent_rewrite_args["promote"] is True
    transcript_gate_args = json.loads((bundle / "mcp-transcript-quality-gate.args.json").read_text(encoding="utf-8"))
    assert transcript_gate_args["bundle_dir"] == str(bundle)
    assert transcript_gate_args["min_punctuation_per_1000"] == 50.0
    assert manifest["term_correction_impact_report_markdown"] == "term-correction-impact-report.md"
    assert manifest["term_arbitration_codex_markdown"] == "term-arbitration-codex.md"
    assert manifest["term_arbitration_codex_prompt_markdown"] == "term-arbitration-codex-prompt.md"
    assert manifest["term_arbitration_codex_result_codex_markdown"] == "term-arbitration-codex-result.codex.md"
    assert manifest["term_arbitration_codex_validation_markdown"] == "term-arbitration-codex-validation.md"
    assert manifest["term_arbitration_codex_validation_json"] == "term-arbitration-codex-validation.json"
    assert manifest["mcp_term_arbitration_codex_args"] == "mcp-term-arbitration-codex.args.json"
    assert manifest["mcp_term_correction_impact_report_args"] == "mcp-term-correction-impact-report.args.json"
    assert manifest["mcp_term_correction_closure_args"] == "mcp-term-correction-closure.args.json"
    assert manifest["mcp_term_correction_closure_codex_args"] == "mcp-term-correction-closure-codex.args.json"
    assert manifest["transcript_semantic_correction_pack_json"] == "transcript-semantic-correction-pack.json"
    assert manifest["transcript_semantic_correction_prompt_markdown"] == "transcript-semantic-correction-prompt.md"
    assert manifest["transcript_semantic_correction_llm_prompt_markdown"] == "transcript-semantic-correction-llm-prompt.md"
    assert manifest["transcript_semantic_candidate_discovery_pack_json"] == "transcript-semantic-candidate-discovery-pack.json"
    assert manifest["transcript_semantic_candidate_discovery_prompt_markdown"] == "transcript-semantic-candidate-discovery-prompt.md"
    assert manifest["transcript_semantic_candidate_discovery_template_json"] == "transcript-semantic-candidate-discovery-template.json"
    assert manifest["transcript_semantic_candidate_discovery_llm_prompt_markdown"] == "transcript-semantic-candidate-discovery-llm-prompt.md"
    assert manifest["transcript_semantic_candidate_suggestions_llm_markdown"] == "transcript-semantic-candidate-suggestions.llm.md"
    assert manifest["transcript_semantic_candidate_suggestions_import_json"] == "transcript-semantic-candidate-suggestions-import.json"
    assert manifest["transcript_semantic_correction_result_codex_markdown"] == "transcript-semantic-correction-result.codex.md"
    assert manifest["transcript_semantic_correction_result_llm_markdown"] == "transcript-semantic-correction-result.llm.md"
    assert manifest["transcript_semantic_correction_validation_markdown"] == "transcript-semantic-correction-validation.md"
    assert manifest["transcript_semantic_correction_closure_markdown"] == "transcript-semantic-correction-closure.md"
    assert manifest["transcript_semantic_correction_impact_report_markdown"] == "transcript-semantic-correction-impact-report.md"
    assert manifest["transcript_semantic_correction_readable_impact_markdown"] == "transcript-semantic-readable-impact-report.md"
    assert manifest["transcript_semantic_summary_impact_markdown"] == "transcript-semantic-summary-impact-report.md"
    assert manifest["transcript_semantic_correction_status_markdown"] == "transcript-semantic-correction-status.md"
    assert manifest["mcp_transcript_semantic_correction_pack_args"] == "mcp-transcript-semantic-correction-pack.args.json"
    assert manifest["mcp_transcript_semantic_correction_codex_draft_args"] == "mcp-transcript-semantic-correction-codex-draft.args.json"
    assert manifest["mcp_transcript_semantic_correction_llm_draft_args"] == "mcp-transcript-semantic-correction-llm-draft.args.json"
    assert manifest["mcp_transcript_semantic_candidate_discovery_pack_args"] == "mcp-transcript-semantic-candidate-discovery-pack.args.json"
    assert manifest["mcp_transcript_semantic_candidate_discovery_llm_draft_args"] == "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json"
    assert manifest["mcp_import_transcript_semantic_candidate_suggestions_args"] == "mcp-import-transcript-semantic-candidate-suggestions.args.json"
    assert manifest["mcp_validate_transcript_semantic_correction_args"] == "mcp-validate-transcript-semantic-correction.args.json"
    assert manifest["mcp_transcript_semantic_correction_closure_args"] == "mcp-transcript-semantic-correction-closure.args.json"
    assert manifest["mcp_transcript_semantic_correction_impact_report_args"] == "mcp-transcript-semantic-correction-impact-report.args.json"
    assert manifest["mcp_transcript_semantic_readable_impact_report_args"] == "mcp-transcript-semantic-readable-impact-report.args.json"
    assert manifest["mcp_transcript_semantic_summary_impact_report_args"] == "mcp-transcript-semantic-summary-impact-report.args.json"
    assert manifest["mcp_transcript_semantic_correction_status_args"] == "mcp-transcript-semantic-correction-status.args.json"
    assert manifest["mcp_import_transcript_semantic_review_notes_args"] == "mcp-import-transcript-semantic-review-notes.args.json"
    assert manifest["smart_summary_section_semantic_review_notes_template"] == "exports/smart-summary-section-semantic-review-notes.template.json"
    assert manifest["transcript_semantic_repair_queue_json"] == "exports/transcript-semantic-repair-queue.json"
    assert manifest["transcript_semantic_repair_queue_markdown"] == "exports/transcript-semantic-repair-queue.md"
    assert manifest["mcp_transcript_semantic_repair_queue_args"] == "mcp-transcript-semantic-repair-queue.args.json"
    assert manifest["transcript_semantic_repair_run_json"] == "exports/transcript-semantic-repair-run.json"
    assert manifest["transcript_semantic_repair_run_markdown"] == "exports/transcript-semantic-repair-run.md"
    assert manifest["mcp_transcript_semantic_repair_run_args"] == "mcp-transcript-semantic-repair-run.args.json"
    assert manifest["transcript_semantic_batch_review_pack_json"] == "exports/transcript-semantic-batch-review-pack.json"
    assert manifest["transcript_semantic_batch_review_pack_markdown"] == "exports/transcript-semantic-batch-review-pack.md"
    assert manifest["transcript_semantic_batch_review_notes_todo_json"] == "exports/transcript-semantic-batch-review-notes.todo.json"
    assert manifest["transcript_semantic_batch_codex_review_draft_json"] == "exports/transcript-semantic-batch-review-notes.codex-draft.json"
    assert manifest["transcript_semantic_batch_review_import_markdown"] == "exports/transcript-semantic-batch-review-import.md"
    assert manifest["mcp_transcript_semantic_batch_review_pack_args"] == "mcp-transcript-semantic-batch-review-pack.args.json"
    assert manifest["mcp_transcript_semantic_batch_codex_review_draft_args"] == "mcp-transcript-semantic-batch-codex-review-draft.args.json"
    assert manifest["mcp_transcript_semantic_batch_import_review_notes_args"] == "mcp-transcript-semantic-batch-import-review-notes.args.json"
    closure_codex_args = json.loads((bundle / "mcp-term-correction-closure-codex.args.json").read_text(encoding="utf-8"))
    assert closure_codex_args["input_json"].endswith("term-arbitration-codex-result.codex.md")
    assert closure_codex_args["accept_draft"] is False
    closure_args = json.loads((bundle / "mcp-term-correction-closure.args.json").read_text(encoding="utf-8"))
    assert closure_args["input_json"] == ""
    semantic_closure_args = json.loads((bundle / "mcp-transcript-semantic-correction-closure.args.json").read_text(encoding="utf-8"))
    assert semantic_closure_args["input_json"].endswith("transcript-semantic-correction-result.codex.md")
    assert semantic_closure_args["auto_apply"] is False
    assert semantic_closure_args["refresh_exports"] is True
    semantic_review_import_args = json.loads((bundle / "mcp-import-transcript-semantic-review-notes.args.json").read_text(encoding="utf-8"))
    assert semantic_review_import_args["bundle_dir"] == str(bundle)
    assert semantic_review_import_args["review_json"].endswith("transcript-semantic-correction-review-notes.json")
    assert semantic_review_import_args["min_confidence"] == 0.88
    semantic_repair_queue_args = json.loads((bundle / "mcp-transcript-semantic-repair-queue.args.json").read_text(encoding="utf-8"))
    assert semantic_repair_queue_args["batch_input"] == str(bundle)
    assert semantic_repair_queue_args["target_bundle_count"] == 1
    assert semantic_repair_queue_args["limit"] == 1
    semantic_summary_impact_args = json.loads((bundle / "mcp-transcript-semantic-summary-impact-report.args.json").read_text(encoding="utf-8"))
    assert semantic_summary_impact_args["bundle_dir"] == str(bundle)
    assert semantic_summary_impact_args["summary_path"] == ""
    semantic_repair_run_args = json.loads((bundle / "mcp-transcript-semantic-repair-run.args.json").read_text(encoding="utf-8"))
    assert semantic_repair_run_args["batch_input"] == str(bundle)
    assert semantic_repair_run_args["execute_safe_actions"] is False
    assert semantic_repair_run_args["allow_closure"] is False
    assert semantic_repair_run_args["allow_llm"] is False
    assert semantic_repair_run_args["provider_config"] == {}
    assert semantic_repair_run_args["llm_limit"] == 80
    semantic_batch_pack_args = json.loads((bundle / "mcp-transcript-semantic-batch-review-pack.args.json").read_text(encoding="utf-8"))
    assert semantic_batch_pack_args["batch_input"] == str(bundle)
    assert semantic_batch_pack_args["target_bundle_count"] == 1
    semantic_batch_codex_args = json.loads((bundle / "mcp-transcript-semantic-batch-codex-review-draft.args.json").read_text(encoding="utf-8"))
    assert semantic_batch_codex_args["review_pack_json"].endswith("transcript-semantic-batch-review-pack.json")
    semantic_batch_import_args = json.loads((bundle / "mcp-transcript-semantic-batch-import-review-notes.args.json").read_text(encoding="utf-8"))
    assert semantic_batch_import_args["review_json"].endswith("transcript-semantic-batch-review-notes.todo.json")
    bridge = console["bridge"]
    assert bridge["schema"] == "video_knowledge_pipeline.task_console.bridge.v1"
    assert bridge["tool"] == "transcript_semantic_repair_run"
    assert bridge["call_url"].endswith("/call")
    bridge_args = bridge["semantic_repair_run_arguments"]
    assert bridge_args["batch_input"] == str(bundle)
    assert bridge_args["execute_safe_actions"] is False
    assert bridge_args["allow_closure"] is False
    assert bridge_args["allow_llm"] is False
    assert bridge_args["provider_config"] == {}
    assert bridge["operator_boundary"]["cloud_calls"] == "disabled_by_default"
    assert "semanticRepairBridgeUrl" in html
    assert "runSemanticRepairViaBridge(false)" in html
    assert "runSemanticRepairViaBridge(true)" in html
    assert "transcript_semantic_repair_run" in html
    semantic_llm_draft_args = json.loads((bundle / "mcp-transcript-semantic-correction-llm-draft.args.json").read_text(encoding="utf-8"))
    assert semantic_llm_draft_args["execute"] is False
    semantic_candidate_discovery_args = json.loads((bundle / "mcp-transcript-semantic-candidate-discovery-pack.args.json").read_text(encoding="utf-8"))
    assert semantic_candidate_discovery_args["bundle_dir"] == str(bundle)
    assert semantic_candidate_discovery_args["input_json"].endswith("transcript-semantic-correction-pack.json")
    assert semantic_candidate_discovery_args["limit"] == 40
    semantic_candidate_discovery_llm_args = json.loads((bundle / "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json").read_text(encoding="utf-8"))
    assert semantic_candidate_discovery_llm_args["execute"] is False
    assert semantic_candidate_discovery_llm_args["provider_config"] == {}
    assert semantic_candidate_discovery_llm_args["input_json"].endswith("transcript-semantic-candidate-discovery-pack.json")
    semantic_candidate_import_args = json.loads((bundle / "mcp-import-transcript-semantic-candidate-suggestions.args.json").read_text(encoding="utf-8"))
    assert semantic_candidate_import_args["input_json"].endswith("transcript-semantic-candidate-suggestions.codex.md")
    semantic_validation_args = json.loads((bundle / "mcp-validate-transcript-semantic-correction.args.json").read_text(encoding="utf-8"))
    assert semantic_validation_args["input_json"].endswith("transcript-semantic-correction-result.codex.md")
    assert manifest["mcp_timeline_alignment_audit_args"] == "mcp-timeline-alignment-audit.args.json"
    assert manifest["timeline_alignment_audit_report"] == "timeline-alignment-audit.md"
    assert manifest["mcp_build_smart_summary_input_pack_args"] == "mcp-build-smart-summary-input-pack.args.json"
    assert manifest["mcp_build_smart_summary_chapters_args"] == "mcp-build-smart-summary-chapters.args.json"
    assert console["status"]["counts"]["timeline_items"] == 2
    assert console["status"]["counts"]["items_with_tagger_annotations"] == 1
    assert console["run_registry"]["run_count"] >= 28
    run_ids = {row["run_id"] for row in console["run_registry"]["runs"]}
    assert {"funasr-run", "asr-env-status", "platform-subtitle-import", "term-correction-impact-report", "prepare-transcript-edit-session", "ebook-batch-1", "tile-result-import-build", "video-moment-index-run", "video-rag-search-run", "long-video-memory-pack", "frame-recapture-plan", "semantic-frame-analysis", "temporal-visual-analysis", "smart-summary-input-pack", "review-pack", "human-review-import", "timeline-alignment-audit"} <= run_ids
    assert "term-arbitration-codex" in run_ids
    action_plan = console["subqueue_action_plan"]
    assert action_plan["schema"] == "video_knowledge_pipeline.subqueue_action_plan.v1"
    assert action_plan["action_required_count"] >= 1
    action_rows = {row["key"]: row for row in action_plan["rows"]}
    assert "asr_transcript:local_asr" in action_rows
    assert "timeline_rag:video_rag" in action_rows
    assert action_rows["asr_transcript:local_asr"]["command_bundle"] == "retry local asr"
    assert action_rows["asr_transcript:local_asr"]["action_status"] == "needs_execution"
    assert action_rows["asr_transcript:local_asr"]["action_kind"] == "explicit_execution_required"
    assert action_rows["asr_transcript:local_asr"]["machine_action_available"] is True
    assert action_rows["asr_transcript:subtitle_import"]["action_kind"] == "operator_input_required"
    assert action_rows["asr_transcript:term_arbitration"]["primary_command"] == "retry codex term arbitration"
    assert action_rows["asr_transcript:term_arbitration"]["action_kind"] == "operator_input_required"
    assert action_rows["asr_transcript:term_impact"]["primary_command"] == "retry term impact report"
    assert action_rows["asr_transcript:term_impact"]["action_kind"] == "retry_available"
    assert action_rows["asr_transcript:subtitle_import"]["operator_review_required"] is True
    assert action_rows["review:manual_import"]["action_kind"] == "operator_input_required"
    assert action_rows["timeline_rag:video_rag"]["retry_commands"][0] == "retry video rag"
    assert action_rows["timeline_rag:video_rag"]["action_kind"] == "explicit_execution_required"
    assert action_rows["timeline_rag:video_rag"]["blocked_reason"] == "rag_index_missing"
    assert action_plan["operator_boundary"]["no_process_started"] is True
    plan_result = export_subqueue_action_plan(bundle, write=True, refresh=False)
    assert plan_result["schema"] == "video_knowledge_pipeline.subqueue_action_plan.v1"
    assert plan_result["row_count"] == action_plan["row_count"]
    assert plan_result["operator_boundary"]["no_process_started"] is True
    assert plan_result["subqueue_action_plan_json_path"].endswith("subqueue-action-plan.json")
    assert (bundle / "subqueue-action-plan.json").exists()
    assert (bundle / "mcp-subqueue-action-plan.args.json").exists()
    manifest_after_plan = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_after_plan["subqueue_action_plan_json"] == "subqueue-action-plan.json"
    assert manifest_after_plan["mcp_subqueue_action_plan_args"] == "mcp-subqueue-action-plan.args.json"
    queue = console["processing_queue"]
    assert queue["schema"] == "video_knowledge_pipeline.task_processing_queue.v1"
    assert queue["action_required_count"] >= 1
    queue_by_key = {row["key"]: row for row in queue["groups"]}
    assert queue_by_key["asr_transcript"]["status"] == "action_required"
    asr_subqueues = {row["key"]: row for row in queue_by_key["asr_transcript"]["subqueues"]}
    assert {"local_asr", "asr_env", "subtitle_import", "source_arbitration", "transcript_correction", "term_impact", "transcript_editor", "other_asr_transcript"} <= set(asr_subqueues)
    assert "term_arbitration" in asr_subqueues
    assert asr_subqueues["local_asr"]["retry_commands"][0] == "retry local asr"
    assert asr_subqueues["asr_env"]["failed_items_preview"][0]["reason"] == "model_not_ready"
    assert asr_subqueues["subtitle_import"]["retry_commands"][0] == "retry subtitle import"
    assert asr_subqueues["source_arbitration"]["retry_commands"][0] == "retry transcript source arbitration"
    assert asr_subqueues["transcript_correction"]["retry_commands"][0] == "retry transcript correction"
    assert asr_subqueues["term_arbitration"]["retry_commands"][0] == "retry codex term arbitration"
    assert asr_subqueues["term_impact"]["retry_commands"][0] == "retry term impact report"
    assert asr_subqueues["transcript_editor"]["retry_commands"][0] == "retry transcript editor"
    assert queue_by_key["document_ocr"]["status"] == "action_required"
    document_failed_reasons = {row["reason"] for row in queue_by_key["document_ocr"]["failed_items_preview"]}
    assert "ocr_text_empty" in document_failed_reasons
    assert "tile_result_pending" in document_failed_reasons
    subqueues = {row["key"]: row for row in queue_by_key["document_ocr"]["subqueues"]}
    assert {"ebook", "screen_text_crop", "high_res_tile", "other_document_ocr"} <= set(subqueues)
    assert subqueues["ebook"]["status"] == "action_required"
    assert subqueues["ebook"]["failed_items_preview"][0]["suggested_next_tool"] == "high_res_tile_plan"
    assert subqueues["ebook"]["failed_items_preview"][0]["evidence_path_count"] == 2
    assert subqueues["ebook"]["retry_commands"][0] == "retry high-res tile for ebook"
    action_rows = {row["key"]: row for row in console["subqueue_action_plan"]["rows"]}
    assert action_rows["document_ocr:ebook"]["primary_command"] == "retry high-res tile for ebook"
    assert "next:high_res_tile_plan" in html
    assert subqueues["high_res_tile"]["status"] == "action_required"
    assert subqueues["high_res_tile"]["failed_items_preview"][0]["reason"] == "tile_result_pending"
    assert subqueues["high_res_tile"]["retry_commands"][0] == "retry tile result import item"
    assert subqueues["high_res_tile"]["failed_items_preview"][0]["tile_result_import_command"] == "retry tile result import item"
    assert queue_by_key["vision"]["status"] == "action_required"
    vision_subqueues = {row["key"]: row for row in queue_by_key["vision"]["subqueues"]}
    assert {"review_triage", "semantic_frame", "temporal_sequence", "provider_smoke", "local_vlm", "other_vision"} <= set(vision_subqueues)
    assert vision_subqueues["review_triage"]["status"] == "action_required"
    assert vision_subqueues["semantic_frame"]["retry_commands"][0] == "retry semantic vision"
    assert vision_subqueues["temporal_sequence"]["retry_commands"][0] == "retry temporal vision"
    assert vision_subqueues["provider_smoke"]["failed_items_preview"][0]["reason"] == "missing_api_key"
    assert vision_subqueues["local_vlm"]["retry_commands"][0] == "retry local vlm smoke"
    assert queue_by_key["timeline_rag"]["status"] == "action_required"
    timeline_subqueues = {row["key"]: row for row in queue_by_key["timeline_rag"]["subqueues"]}
    assert {"timeline_alignment", "moment_index", "video_rag", "long_video_memory", "recapture", "other_timeline_rag"} <= set(timeline_subqueues)
    assert timeline_subqueues["timeline_alignment"]["status"] == "action_required"
    assert timeline_subqueues["moment_index"]["retry_commands"][0] == "retry video moment index"
    assert timeline_subqueues["video_rag"]["retry_commands"][0] == "retry video rag"
    assert timeline_subqueues["long_video_memory"]["retry_commands"][0] == "retry long video memory"
    assert timeline_subqueues["recapture"]["retry_commands"][0] == "retry frame recapture"
    assert queue_by_key["summary_export"]["status"] == "action_required"
    summary_subqueues = {row["key"]: row for row in queue_by_key["summary_export"]["subqueues"]}
    assert {"summary_input", "section_workflow", "section_apply", "knowledge_export", "content_candidate", "other_summary_export"} <= set(summary_subqueues)
    assert summary_subqueues["summary_input"]["retry_commands"][0] == "retry summary input pack"
    assert summary_subqueues["section_workflow"]["failed_items_preview"][0]["reason"] == "section_revision_pending"
    action_rows = {row["key"]: row for row in console["subqueue_action_plan"]["rows"]}
    assert action_rows["summary_export:section_workflow"]["action_kind"] == "operator_input_required"
    assert action_rows["summary_export:section_workflow"]["operator_review_required"] is True
    assert summary_subqueues["section_apply"]["retry_commands"][0] == "retry section apply"
    assert summary_subqueues["knowledge_export"]["retry_commands"][0] == "retry knowledge export"
    assert summary_subqueues["content_candidate"]["retry_commands"][0] == "retry content candidates"
    assert queue_by_key["review"]["status"] == "action_required"
    review_subqueues = {row["key"]: row for row in queue_by_key["review"]["subqueues"]}
    assert {"review_pack", "transcript_arbitration", "sample_eval", "closure_status", "manual_import", "other_review"} <= set(review_subqueues)
    assert review_subqueues["review_pack"]["retry_commands"][0] == "retry review pack"
    assert review_subqueues["transcript_arbitration"]["failed_items_preview"][0]["reason"] == "low_confidence_conflict"
    assert review_subqueues["sample_eval"]["retry_commands"][0] == "retry sample review"
    assert review_subqueues["closure_status"]["retry_commands"][0] == "retry review closure"
    assert review_subqueues["manual_import"]["retry_commands"][0] == "retry human review import"
    assert console["status"]["timeline_alignment"]["status"] == "needs_input"
    assert console["status"]["timeline_alignment_issue_count"] == 0
    assert console["status"]["term_correction"]["status"] == "needs_term_arbitration"
    assert console["status"]["term_correction"]["next_action_key"] == "term_arbitration_codex"
    assert console["status"]["semantic_correction"]["status"] in {"missing_pack", "no_candidates"}
    assert console["status"]["semantic_correction"]["next_action_key"] in {"build_pack", "none"}
    assert commands_by_key["term_arbitration_codex"]["recommended"] is True
    assert "--input-json" in commands_by_key["term_correction_closure_codex_import"]["command"]
    assert commands_by_key["term_arbitration_codex"]["reason"] == "term_correction_status=needs_term_arbitration"
    assert {
        "triage",
        "timeline_alignment",
        "video_moment_index",
        "video_moment_search",
        "long_video_memory_pack",
        "video_rag_pack",
        "video_rag_search",
        "video_rag_service",
        "transcript_editor",
        "term_correction_impact",
        "term_arbitration_codex",
        "term_arbitration_codex_import",
        "term_arbitration_codex_validate",
        "term_correction_status",
        "term_correction_closure_codex_import",
        "transcript_semantic_correction_pack",
        "transcript_semantic_correction_codex_draft",
        "transcript_semantic_correction_llm_draft",
        "agent_readable_transcript_rewrite",
        "transcript_quality_gate",
        "transcript_semantic_candidate_discovery",
        "transcript_semantic_candidate_discovery_llm",
        "import_transcript_semantic_candidate_suggestions",
        "validate_transcript_semantic_correction",
        "transcript_semantic_correction_closure",
        "transcript_semantic_correction_impact",
        "transcript_semantic_readable_impact",
        "transcript_semantic_correction_status",
        "transcript_semantic_review_notes_import",
        "transcript_semantic_repair_queue",
        "transcript_semantic_repair_run",
        "transcript_semantic_batch_review_pack",
        "transcript_semantic_batch_codex_review_draft",
        "transcript_semantic_batch_import_review_notes",
        "transcript_semantic_batch_import_codex_draft",
        "apply_transcript_edits",
        "external_capability_pack",
        "visual_structure",
        "vision_preflight",
        "local_vlm_smoke",
        "semantic_vision",
        "volcengine_semantic_batch",
        "temporal_vision",
        "volcengine_temporal_batch",
        "export",
    } <= command_keys
    assert "import-transcript-semantic-review-notes" in commands_by_key["transcript_semantic_review_notes_import"]["command"]
    assert "transcript-semantic-batch-review-pack" in commands_by_key["transcript_semantic_batch_review_pack"]["command"]
    assert "transcript-semantic-batch-codex-review-draft" in commands_by_key["transcript_semantic_batch_codex_review_draft"]["command"]
    assert "transcript-semantic-batch-import-review-notes" in commands_by_key["transcript_semantic_batch_import_review_notes"]["command"]
    assert "transcript-semantic-candidate-discovery-pack" in commands_by_key["transcript_semantic_candidate_discovery"]["command"]
    assert "transcript-semantic-candidate-discovery-llm-draft" in commands_by_key["transcript_semantic_candidate_discovery_llm"]["command"]
    assert "import-transcript-semantic-candidate-suggestions" in commands_by_key["import_transcript_semantic_candidate_suggestions"]["command"]
    assert "transcript-semantic-correction-closure" in commands_by_key["transcript_semantic_correction_closure"]["command"]
    assert "--refresh-exports" in commands_by_key["transcript_semantic_correction_closure"]["command"]
    assert "章节语义复核 notes 模板" in html
    assert "导入章节语义复核 notes" in html
    assert "VKP 任务控制台" in html
    assert "术语闭环" in html
    assert "Codex预检" in html
    assert "预检接受/拒绝" in html
    assert "needs_term_arbitration" in html
    assert "任务历史" in html
    assert "Ebook batch" in html
    assert "retry local asr" in html
    assert "retry asr env" in html
    assert "retry subtitle import" in html
    assert "retry transcript source arbitration" in html
    assert "retry transcript correction" in html
    assert "retry term impact report" in html
    assert "retry codex term arbitration" in html
    assert "Codex 术语/工具名语义仲裁" in html
    assert "term-arbitration-codex-prompt.md" in html
    assert "term-arbitration-codex-pack.json" in html
    assert "term-arbitration-codex-result.codex.md" in html
    assert "Codex 术语回复草稿" in html
    assert "Only decisions with semantic rationale" in html
    assert "validate-term-arbitration-codex-result" in html
    assert "Codex 术语回复预检" in html
    assert "term-arbitration-codex-validation.md" in html
    assert "术语纠错状态" in html
    assert "mcp-term-correction-status.args.json" in html
    assert "导入 Codex 术语回复 MCP Args" in html
    assert "mcp-term-correction-closure-codex.args.json" in html
    assert "术语影响检查" in html
    assert "term-correction-impact-report.md" in html
    assert "通用 ASR/字幕语义纠错" in html
    assert "transcript-semantic-correction-pack" in html
    assert "语义错词候选发现 Prompt" in html
    assert "LLM 候选发现计划" in html
    assert "导入候选发现 suggestions" in html
    assert "transcript-semantic-candidate-discovery-pack" in html
    assert "transcript-semantic-candidate-discovery-llm-draft" in html
    assert "import-transcript-semantic-candidate-suggestions" in html
    assert "validate-transcript-semantic-correction" in html
    assert "transcript-semantic-correction-result.codex.md" in html
    assert "retry transcript editor" in html
    assert "retry high-res tile for ebook" in html
    assert "retry tile result import item" in html
    assert "0002-01" in html
    assert "subqueues" in html
    assert "子队列行动面板" in html
    assert "filterSubqueue" in html
    assert "subqueue-action-bundle" in html
    assert "explicit_execution_required" in html
    assert "operator_input_required" in html
    assert "可复制命令执行" in html
    assert "data-subqueue-full-key" in html
    assert "asr_transcript:local_asr" in html
    assert "timeline_rag:video_rag" in html
    assert "queue-subretry" in html
    assert "retry semantic vision" in html
    assert "retry temporal vision" in html
    assert "retry provider smoke" in html
    assert "retry local vlm smoke" in html
    assert "retry video moment index" in html
    assert "retry video rag" in html
    assert "retry long video memory" in html
    assert "retry frame recapture" in html
    assert "retry summary input pack" in html
    assert "retry section workflow" in html
    assert "retry section apply" in html
    assert "retry knowledge export" in html
    assert "retry content candidates" in html
    assert "retry review pack" in html
    assert "retry transcript arbitration review" in html
    assert "retry sample review" in html
    assert "retry review closure" in html
    assert "retry human review import" in html
    assert "review.html" in html
    assert "model-settings.html" in html
    assert "consoleVideo" in html
    assert "seekToMoment" in html
    assert "citationPanel" in html
    assert "片段搜索" in html
    assert "momentSearchInput" in html
    assert "video-moment-index.md" in html
    assert "video-rag-pack" in html
    assert "video-rag-search" in html
    assert "video-rag-service-plan" in html
    assert "prepare-transcript-edit-session" in html
    assert "local-vlm-serving-smoke" in html
    assert "external-capability-pack" in html
    assert (bundle / "exports" / "video-moment-index.json").exists()
    assert (bundle / "exports" / "video-moment-index.md").exists()
    assert "vision-review-triage" in html
    assert "时间轴对齐审计" in html
    assert "timeline-alignment-audit" in html
    assert (bundle / "timeline-alignment-audit.json").exists()
    assert (bundle / "timeline-alignment-audit.md").exists()
    assert settings_json["schema"] == "video_knowledge_pipeline.model_api_settings.v1"
    assert "vision-provider-matrix" in settings_html
    assert "vision-provider-smoke" in settings_html
    assert "vision-execution-preflight" in settings_html
    assert "run-volcengine-vision-batch.ps1" in settings_html
    assert "&lt;paste-key&gt;" in settings_html
    assert "run-multimodal-frame-analysis" in html
    assert "agent-readable-transcript-rewrite" in html
    assert "transcript-quality-gate" in html
    assert "run-volcengine-vision-batch.ps1" in html
    visual_structure = next(item for item in console["commands"] if item["key"] == "visual_structure")
    assert "--execute-ebook-pipeline" in visual_structure["command"]
    assert "--include-routes \\\"document_visual,mixed\\\"" in visual_structure["command"]
    assert visual_structure["safety"] == "local_ebook_pipeline"
    assert "cloud_call_requires_confirmation" in html

    audit = audit_bundle_mcp_args(bundle)
    assert audit["status"] == "ok"
    assert audit["blocked_count"] == 0
    assert any(row["key"] == "mcp_export_task_console_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_subqueue_action_plan_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_video_moment_index_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_video_rag_search_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_video_rag_service_plan_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_prepare_transcript_edit_session_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_plan_local_asr_service_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_run_local_asr_service_plan_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_postprocess_asr_transcript_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_readable_transcript_llm_polish_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_agent_readable_transcript_rewrite_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_quality_gate_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_timeline_alignment_audit_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_correction_pack_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_correction_codex_draft_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_correction_llm_draft_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_candidate_discovery_pack_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_candidate_discovery_llm_draft_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_import_transcript_semantic_candidate_suggestions_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_validate_transcript_semantic_correction_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_correction_closure_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_correction_impact_report_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_readable_impact_report_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_summary_impact_report_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_correction_status_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_import_transcript_semantic_review_notes_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_repair_queue_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_transcript_semantic_repair_run_args" for row in audit["rows"])

    timeline_result = run_mcp_call("timeline_alignment_audit", bundle / "mcp-timeline-alignment-audit.args.json")
    assert timeline_result["schema"] == "video_knowledge_pipeline.timeline_alignment_audit.v1"

    moment_result = run_mcp_call("video_moment_index", bundle / "mcp-video-moment-index.args.json")
    assert moment_result["schema"] == "video_knowledge_pipeline.video_moment_index.v1"

    mcp_result = run_mcp_call("export_task_console", bundle / "mcp-export-task-console.args.json")
    assert mcp_result["task_console_html_path"].endswith("task-console.html")

    subqueue_mcp_result = run_mcp_call("subqueue_action_plan", bundle / "mcp-subqueue-action-plan.args.json")
    assert subqueue_mcp_result["schema"] == "video_knowledge_pipeline.subqueue_action_plan.v1"


def test_webui_export_includes_task_console(tmp_path: Path) -> None:
    root = tmp_path / "run"
    packages = root / "lecture-packages"
    packages.mkdir(parents=True)
    (packages / "lecture-package.json").write_text(
        json.dumps(
            {
                "title": "export console lesson",
                "coverage": {},
                "sources": [],
                "timeline": [{"index": 1, "start": 0, "end": 1, "transcript": "hello", "visual_route": "semantic_frame"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"

    exported = export_webui_bundle(root, output_dir=bundle)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert exported["task_console_html_path"] == str(bundle / "task-console.html")
    assert exported["mcp_export_task_console_args_path"] == str(bundle / "mcp-export-task-console.args.json")
    assert (bundle / "task-console.html").exists()
    assert (bundle / "task-console.json").exists()
    assert manifest["task_console"] == "task-console.html"
    assert manifest["mcp_export_task_console_args"] == "mcp-export-task-console.args.json"




















def test_task_console_shows_transcript_semantic_candidate_discovery_suggestions(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    suggestions_path = bundle / "transcript-semantic-candidate-suggestions.codex.md"
    suggestions_path.write_text(
        "# suggestions\n\n```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1",
                "source": "codex_candidate_discovery_test",
                "suggestions": [
                    {
                        "source_segment_index": 3,
                        "original_text": "这个很重要",
                        "candidate_text": "客户信任建立流程",
                        "correction_type": "concept",
                        "confidence": 0.82,
                        "reason": "ASR 是低信息指代，OCR/上下文显示屏幕主题是客户信任建立流程。",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    import_path = bundle / "transcript-semantic-candidate-suggestions-import.json"
    import_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions_import.v1",
                "status": "imported",
                "suggestion_count": 2,
                "imported_candidate_count": 1,
                "skipped_count": 1,
                "imported_candidate_ids": ["semcorr-0007"],
                "skipped": [
                    {
                        "row_number": 2,
                        "reason": "duplicate_candidate",
                        "suggestion": {"original_text": "browser base", "candidate_text": "Browserbase"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    html = _render_task_console_html(
        {
            "title": "Candidate Discovery Console",
            "bundle_dir": str(bundle),
            "status": {
                "counts": {},
                "semantic_correction": {
                    "status": "no_candidates",
                    "candidate_discovery_status": "imported",
                    "candidate_discovery_next_action": "run_llm_draft_preview",
                    "candidate_discovery_segment_count": 12,
                    "candidate_discovery_suggestion_count": 2,
                    "candidate_discovery_imported_candidate_count": 1,
                    "candidate_discovery_skipped_count": 1,
                    "candidate_discovery_artifacts": {
                        "codex_suggestions_markdown": str(suggestions_path),
                        "import_json": str(import_path),
                    },
                    "commands": {
                        "import_candidate_suggestions": ".\\scripts\\video-knowledge.ps1 import-transcript-semantic-candidate-suggestions bundle --input-json transcript-semantic-candidate-suggestions.codex.md"
                    },
                },
            },
            "artifacts": [],
            "commands": [],
        }
    )

    assert "候选发现 suggestions 预览" in html
    assert "候选建议编辑器" in html
    assert "semantic-candidate-suggestion-row" in html
    assert "copySemanticCandidateSuggestions" in html
    assert "downloadSemanticCandidateSuggestions" in html
    assert "vkpReviewExportName('转录候选审核', JSON.parse(text))" in html
    assert "data-field=\"candidate_text\"" in html
    assert "客户信任建立流程" in html
    assert "codex_candidate_discovery_test" not in html
    assert "semcorr-0007" in html
    assert "duplicate_candidate" in html
    assert "import-transcript-semantic-candidate-suggestions" in html
