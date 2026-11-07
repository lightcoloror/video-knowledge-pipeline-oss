from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.run_artifact_registry import register_bundle_run
from video_knowledge_pipeline.storage import append_jsonl, read_json, write_json
from video_knowledge_pipeline.task_console import export_task_console
from video_knowledge_pipeline.video_workbench import export_video_workbench


def _write_bundle(root: Path) -> None:
    (root / "exports").mkdir(parents=True, exist_ok=True)
    write_json(
        root / "manifest.json",
        {
            "title": "Workbench fixture",
            "review_html": "review.html",
            "task_console": "task-console.html",
            "transcript_editor_html": "transcript-editor.html",
            "smart_summary_section_editor_html": "smart-summary-section-editor.html",
            "smart_summary_input_pack_markdown": "exports/smart-summary-input-pack.md",
            "smart_summary_chapters_markdown": "exports/smart-summary-chapters.md",
            "smart_summary_course_map_markdown": "exports/course-map.md",
            "knowledge_note_smart_summary_markdown": "exports/smart-summary.md",
            "knowledge_note_full_body_markdown": "exports/full-body.md",
            "knowledge_note_transcript_markdown": "exports/full-transcript.md",
            "transcript_source_arbitration_json": "transcript-source-arbitration.json",
            "transcript_source_arbitration_markdown": "transcript-source-arbitration.md",
            "term_correction_impact_report_json": "term-correction-impact-report.json",
            "term_correction_impact_report_markdown": "term-correction-impact-report.md",
            "term_correction_closure_markdown": "term-correction-closure.md",
            "term_arbitration_glossary_json": "term-arbitration-glossary.json",
            "video_rag_pack_markdown": "exports/video-rag-pack.md",
            "video_rag_search_markdown": "exports/video-rag-search.md",
            "video_rag_chunks_jsonl": "exports/video-rag-chunks.jsonl",
            "video_rag_search_backend": "sqlite",
            "video_rag_sqlite_index": "exports/video-rag-index.sqlite",
        },
    )
    write_json(
        root / "timeline.json",
        [
            {
                "index": 0,
                "start": 0.0,
                "end": 6.0,
                "transcript": "第一段讲工具选择。",
                "visual_text": "Browser Use / Playwright",
                "visual_route": "document_visual",
                "quality_issues": [],
                "frame_path": "frames/frame-0000.jpg",
            },
            {
                "index": 1,
                "start": 8.0,
                "end": 14.0,
                "corrected_transcript": "第二段讲封控风险。",
                "visual_understanding": {"summary": "画面展示浏览器自动化工具排名。"},
                "visual_route": "semantic_frame",
                "quality_issues": ["semantic_frame_without_analysis", "needs_high_res_tile_recovery", "tile_result_needs_review"],
                "tile_review_targets": [{"tile_id": "tile-0001", "confidence": 0.42, "reasons": ["low_confidence"]}],
            },
        ],
    )
    write_json(
        root / "timeline-alignment-audit.json",
        {
            "schema": "video_knowledge_pipeline.timeline_alignment_audit.v1",
            "summary": {"issue_count": 1},
            "items": [{"index": 1, "issues": ["review_start_mismatch"], "suggested_review_start": 8.0}],
        },
    )
    write_json(
        root / "exports" / "video-moment-index.json",
        {
            "schema": "video_knowledge_pipeline.video_moment_index.v1",
            "summary": {"chunks": 2, "duration_seconds": 14.0},
            "chunks": [
                {
                    "chunk_index": 1,
                    "start": 0.0,
                    "end": 6.0,
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:06.000",
                    "timeline_indexes": [0],
                    "keywords": ["Browser Use", "Playwright"],
                    "transcript_text": "第一段讲工具选择。",
                    "visual_text": "Browser Use / Playwright",
                    "has_visual_evidence": True,
                    "evidence_paths": ["frames/frame-0000.jpg"],
                },
                {
                    "chunk_index": 2,
                    "start": 8.0,
                    "end": 14.0,
                    "start_time": "00:00:08.000",
                    "end_time": "00:00:14.000",
                    "timeline_indexes": [1],
                    "keywords": ["封控", "风险"],
                    "transcript_text": "第二段讲封控风险。",
                    "has_temporal_evidence": True,
                    "evidence_paths": ["frames/frame-0001.jpg"],
                },
            ],
        },
    )
    append_jsonl(
        root / "exports" / "video-rag-chunks.jsonl",
        [
            {
                "id": "fixture:review_gap:0001",
                "text": "Review gap: tile_result_needs_review Browser Use 工具名需要复核。",
                "metadata": {
                    "chunk_kind": "review_gap",
                    "start": 8.0,
                    "end": 14.0,
                    "start_time": "00:00:08.000",
                    "end_time": "00:00:14.000",
                    "timeline_indexes": [1],
                    "tags": ["review_gap", "tile_result_needs_review"],
                    "keywords": ["Browser Use", "工具名", "复核"],
                    "has_visual_evidence": True,
                    "has_temporal_evidence": False,
                    "evidence_paths": ["frames/frame-0001.jpg"],
                },
            },
            {
                "id": "fixture:chapter_memory:L001",
                "text": "Chapter memory L001: Browser Use 与 Playwright 的工具选择主线。",
                "metadata": {
                    "chunk_kind": "chapter_memory",
                    "memory_level": "chapter",
                    "memory_id": "L001",
                    "child_memory_ids": ["M0001", "M0002"],
                    "child_moment_indexes": [1, 2],
                    "start": 0.0,
                    "end": 14.0,
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:14.000",
                    "timeline_indexes": [1, 2],
                    "tags": ["long_video_memory", "chapter_memory"],
                    "keywords": ["Browser Use", "Playwright", "工具选择"],
                    "has_visual_evidence": True,
                    "has_temporal_evidence": True,
                    "evidence_paths": ["frames/frame-0000.jpg", "frames/frame-0001.jpg"],
                },
            },
            {
                "id": "fixture:theme_memory:L001:theme",
                "text": "Theme memory L001: 浏览器自动化工具选择和封控风险。",
                "metadata": {
                    "chunk_kind": "theme_memory",
                    "memory_level": "theme",
                    "memory_id": "L001:theme",
                    "parent_memory_id": "L001",
                    "child_memory_ids": ["M0001", "M0002"],
                    "child_moment_indexes": [1, 2],
                    "start": 0.0,
                    "end": 14.0,
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:14.000",
                    "timeline_indexes": [1, 2],
                    "tags": ["long_video_memory", "theme_memory"],
                    "keywords": ["浏览器自动化", "封控风险"],
                    "has_visual_evidence": True,
                    "has_temporal_evidence": True,
                    "evidence_paths": ["frames/frame-0000.jpg", "frames/frame-0001.jpg"],
                },
            },            {
                "id": "fixture:content_asset:key_segments",
                "text": "Content asset: 浏览器自动化工具横评可作为内容素材候选。",
                "metadata": {
                    "chunk_kind": "content_asset",
                    "start": 0.0,
                    "end": 0.0,
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:00.000",
                    "timeline_indexes": [],
                    "tags": ["content_asset"],
                    "keywords": ["浏览器自动化", "内容素材"],
                    "has_visual_evidence": False,
                    "has_temporal_evidence": False,
                    "evidence_paths": ["exports/key-segments.md"],
                },
            },
        ],
    )
    write_json(
        root / "exports" / "video-rag-search.json",
        {
            "schema": "video_knowledge_pipeline.video_rag_search.v1",
            "retrieval_backend": "sqlite",
            "requested_retrieval_backend": "sqlite",
            "backend_status": "ok",
            "summary": {"chunks_loaded": 4, "hits": 2, "sqlite_index_exists": True},
            "operator_boundary": {"local_only": True, "no_vector_backend_started": True},
        },
    )
    (root / "exports" / "video-rag-index.sqlite").write_bytes(b"sqlite-fixture")
    write_json(
        root / "exports" / "content-candidate-pack.json",
        {
            "schema": "video_knowledge_pipeline.content_candidate_pack.v1",
            "candidate_count": 1,
            "citation_digest_candidate_count": 1,
            "review_required": True,
            "publication_allowed": False,
            "allowed_as_fact": False,
            "allowed_as_inspiration": True,
            "candidates": [
                {
                    "id": "candidate-001",
                    "timeline_index": 1,
                    "time_range": "00:00:08.000 - 00:00:14.000",
                    "candidate_types": ["method", "visual_explainer"],
                    "viewpoint": "第二段可以作为浏览器自动化工具选择方法的素材候选。",
                    "case_or_example": "画面展示浏览器自动化工具排名。",
                    "evidence_paths": ["frames/frame-0001.jpg"],
                    "citation_digest_status": "ready",
                        "summary_chapter_refs": [
                            {
                                "chapter_index": 1,
                                "chapter_title": "浏览器自动化工具选择方法",
                                "chapter_time_range": "00:00:00 - 00:00:12",
                            }
                        ],
                        "summary_chapter_ref_count": 1,
                    "evidence_citations": [
                        {
                            "source_type": "visual",
                            "time": "00:00:08.000 - 00:00:14.000",
                            "timeline_indexes": [1],
                            "text": "浏览器自动化工具选择方法的证据引用。",
                            "evidence_paths": ["frames/frame-0001.jpg"],
                        }
                    ],
                }
            ],
        },
    )
    (root / "exports" / "content-candidate-pack.md").write_text("# 内容素材候选包\n", encoding="utf-8")
    write_json(
        root / "human-sample-eval.json",
        {
            "schema": "video_knowledge_pipeline.human_sample_eval.v1",
            "status": "ready",
            "sample_count": 5,
            "labeled_rows": 3,
            "rates": {
                "content_candidate_usable_rate": 80.0,
                "content_candidate_evidence_sufficient_rate": 60.0,
                "human_sampled_multimodal_net_help_rate": 25.0,
            },
        },
    )
    (root / "human-sample-eval.md").write_text("# Human Sample Eval\n", encoding="utf-8")
    write_json(
        root / "vision-provider-smoke.json",
        {
            "schema": "lecture_vision_provider_smoke.v1",
            "status": "ok",
            "safe_to_execute": True,
            "provider": {"provider": "volcengine_coding_plan", "model": "ark-code-latest"},
            "error_class": "",
            "error_summary": "",
            "recovery_suggestion": "",
        },
    )
    (root / "vision-provider-smoke.md").write_text("# Vision Provider Smoke\n", encoding="utf-8")
    write_json(
        root / "vision-provider-matrix.json",
        {
            "schema": "lecture_vision_provider_matrix.v1",
            "status": "ok",
            "recommended_provider": "volcengine_coding_plan",
            "providers_requested": ["local_qwen_vl", "volcengine_coding_plan", "gemini"],
            "provider_ranking": [
                {"provider": "volcengine_coding_plan", "ready": True},
                {"provider": "gemini", "ready": False},
            ],
        },
    )
    (root / "vision-provider-matrix.md").write_text("# Vision Provider Matrix\n", encoding="utf-8")
    write_json(
        root / "local-vlm-serving-smoke.json",
        {
            "schema": "video_knowledge_pipeline.local_vlm_serving_smoke.v1",
            "ok": True,
            "execute": False,
            "provider": "local_qwen_vl",
            "profile": {"provider": "local_qwen_vl", "model": "Qwen/Qwen2.5-VL-3B-Instruct", "base_url": "http://127.0.0.1:8000/v1"},
            "input_spec": {"short_frame_group_image_count": 8},
            "capability_matrix": [
                {"capability": "openai_compatible_endpoint", "status": "planned"},
                {"capability": "multi_image_json", "status": "planned"},
            ],
        },
    )
    (root / "local-vlm-serving-smoke.md").write_text("# Local VLM Serving Smoke\n", encoding="utf-8")
    write_json(
        root / "transcript-source-arbitration.json",
        {
            "schema": "video_knowledge_pipeline.transcript_source_arbitration.v1",
            "summary": {
                "segments": 2,
                "changed_segments": 1,
                "review_segments": 1,
                "quality_status": "needs_review",
                "average_confidence": 0.71,
                "high_confidence_term_replacements": 1,
                "low_confidence_conflicts": 1,
            },
            "quality_summary": {
                "status": "needs_review",
                "total_segments": 2,
                "changed_segments": 1,
                "review_segments": 1,
                "average_confidence": 0.71,
                "high_confidence_term_replacements": 1,
                "low_confidence_conflicts": 1,
                "can_use_as_summary_input": False,
            },
            "review_rows": [
                {
                    "index": 1,
                    "start": 8.0,
                    "end": 14.0,
                    "original_text": "第二段讲风控风险。",
                    "corrected_text": "第二段讲封控风险。",
                    "chosen_source": "platform_subtitle",
                    "chosen_source_type": "platform_subtitle",
                    "confidence": 0.62,
                    "review_reason": "low_arbitration_confidence",
                    "alternatives": [
                        {"source_id": "asr", "source_type": "asr", "text": "第二段讲风控风险。", "score": 2.1},
                        {"source_id": "platform_subtitle", "source_type": "subtitle", "text": "第二段讲封控风险。", "score": 2.3},
                    ],
                }
            ],
        },
    )
    write_json(
        root / "term-correction-impact-report.json",
        {
            "schema": "video_knowledge_pipeline.term_correction_impact.v1",
            "status": "passed",
            "ok": True,
            "replacement_count": 2,
            "source_alias_total": 5,
            "output_alias_total": 0,
            "final_export_alias_total": 0,
            "reduction_rate": 1.0,
            "final_clean_rate": 1.0,
            "terms": [
                {
                    "alias": "playright",
                    "canonical": "Playwright",
                    "source_alias_count": 3,
                    "output_alias_count": 0,
                    "resolved_in_outputs": True,
                    "had_source_alias": True,
                },
                {
                    "alias": "brother mcp",
                    "canonical": "Browser MCP",
                    "source_alias_count": 2,
                    "output_alias_count": 0,
                    "resolved_in_outputs": True,
                    "had_source_alias": True,
                },
            ],
            "next_actions": [],
        },
    )
    (root / "term-correction-impact-report.md").write_text("# Term Correction Impact\n", encoding="utf-8")
    write_json(
        root / "term-arbitration-glossary.json",
        {
            "schema": "video_knowledge_pipeline.term_arbitration_glossary.v1",
            "terms": [
                {"canonical": "Playwright", "aliases": ["playright"], "confidence": 0.96, "review_required": False},
                {"canonical": "Browser MCP", "aliases": ["brother mcp"], "confidence": 0.94, "review_required": False},
            ],
        },
    )
    write_json(root / "source-arbitrated-transcript.json", {"segments": [{"start": 0.0, "end": 6.0, "text": "第一段讲 Playwright 工具选择。"}]})
    write_json(root / "exports" / "smart-summary-quality.json", {"schema": "video_knowledge_pipeline.smart_summary_quality.v1", "passed": True, "status": "passed"})
    write_json(
        root / "term-correction-closure.json",
        {
            "schema": "video_knowledge_pipeline.term_correction_closure.v1",
            "status": "completed",
            "steps": {"term_arbitration_codex": {"status": "imported"}, "term_correction_impact": {"status": "passed"}},
        },
    )
    (root / "term-correction-closure.md").write_text("# Term Correction Closure\n", encoding="utf-8")
    (root / "term-arbitration-codex.md").write_text("# Term Arbitration Codex\n", encoding="utf-8")
    (root / "term-arbitration-codex-prompt.md").write_text("# Codex Prompt\n", encoding="utf-8")
    (root / "term-arbitration-codex-validation.md").write_text("# Codex Validation\n", encoding="utf-8")
    write_json(root / "term-arbitration-codex-validation.json", {"schema": "video_knowledge_pipeline.term_arbitration_codex_validation.v1", "status": "ready_for_import", "ok": True, "accepted_decision_count": 2, "rejected_decision_count": 1})
    write_json(root / "term-arbitration-codex-result.template.json", {"schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1", "decisions": []})
    write_json(root / "term-arbitration-codex-result.json", {"schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1", "decisions": []})
    write_json(
        root / "review-closure-status.json",
        {
            "schema": "lecture_review_closure_status.v1",
            "summary": {"open": 3, "closed": 2},
            "open_by_reason": {"transcript_source_conflict": 1, "low_arbitration_confidence": 1, "pending_review": 2},
            "closed_by_reason": {"transcript_source_conflict": 2, "low_arbitration_confidence": 2},
            "closed_targets": [],
        },
    )
    for rel, text in {
        "review.html": "<html>review</html>",
        "targeted-review.html": "<html>targeted review</html>",
        "task-console.html": "<html>console</html>",
        "transcript-editor.html": "<html>transcript</html>",
        "smart-summary-section-editor.html": "<html>summary editor</html>",
        "review-closure-status.md": "# Review Closure Status\n",
        "review-pack.md": "# Review Pack\n",
        "transcript-source-arbitration.md": "# Transcript Source Arbitration\n",
        "timeline-alignment-audit.md": "# Timeline Alignment\n",
        "exports/video-moment-index.md": "# Video Moment Index\n",
        "exports/video-rag-pack.md": "# VideoRAG Pack\n",
        "exports/video-rag-search.md": "# VideoRAG Search\n",
        "exports/video-rag-service-plan.md": "# VideoRAG Service Plan\n",
        "exports/external-capability-pack.md": "# External Capability Pack\n",
        "exports/smart-summary.md": "# Smart summary\n",
        "exports/smart-summary-input-pack.md": "# Smart Summary Input Pack\n",
        "exports/smart-summary-chapters.md": "# Smart Summary Chapters\n",
        "exports/course-map.md": "# Course Map\n",
        "exports/full-body.md": "# Full Body\n",
        "exports/full-transcript.md": "# Transcript\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _fresh_bundle(tmp_path: Path) -> Path:
    return tmp_path / "bundle"


def test_export_video_workbench_writes_static_workspace(tmp_path: Path) -> None:
    bundle = _fresh_bundle(tmp_path)
    _write_bundle(bundle)
    register_bundle_run(
        bundle,
        run_type="smart_summary_chapter_pack",
        run_id="smart-summary-chapter-pack",
        status="completed",
        title="Smart summary chapter pack",
        summary="Chapter evidence pack generated.",
        artifacts=[{"key": "chapter_pack", "path": str(bundle / "exports" / "smart-summary-chapters.md")}],
        retry_command="rerun smart summary chapters",
        write=True,
    )

    register_bundle_run(
        bundle,
        run_type="video_moment_index",
        run_id="video-moment-index",
        status="completed",
        title="Video moment index",
        summary="Moment index generated.",
        artifacts=[{"key": "moment", "path": str(bundle / "exports" / "video-moment-index.md")}],
        retry_command="retry video moment index",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="long_video_memory_pack",
        run_id="long-video-memory-pack",
        status="completed",
        title="Long video memory pack",
        summary="Long video memory generated.",
        retry_command="retry long video memory",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="video_rag_pack",
        run_id="video-rag-pack",
        status="completed",
        title="VideoRAG pack",
        summary="VideoRAG chunks generated.",
        retry_command="retry video rag pack",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="video_rag_service",
        run_id="video-rag-service",
        status="needs_execution",
        title="VideoRAG local service plan",
        summary="Service plan requires explicit start.",
        retry_command="start video rag service",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="external_capability_pack",
        run_id="external-capability-pack",
        status="needs_input",
        title="External capability pack",
        summary="Content candidates need re-export.",
        failed_items=[{"index": "content_material_generation", "reason": "content_candidate_pack_needs_reexport"}],
        retry_command="retry external capability pack",
        write=True,
    )
    register_bundle_run(
        bundle,
        run_type="visual_structure_ebook",
        run_id="ebook-batch",
        status="needs_retry",
        title="Ebook OCR batch",
        summary="OCR returned empty text for a crop.",
        failed_items=[{"index": 1, "reason": "ocr_text_empty", "detail": "empty wrapper"}],
        retry_command="retry ebook batch",
        write=True,
    )

    result = export_video_workbench(bundle)

    assert result["schema"] == "video_knowledge_pipeline.video_workbench.v1"
    assert result["timeline_count"] == 2
    assert Path(result["paths"]["html"]).exists()
    assert Path(result["paths"]["json"]).exists()
    assert Path(result["paths"]["mcp_args"]).exists()
    assert Path(result["paths"]["ui_surface_registry_json"]).exists()
    assert Path(result["paths"]["ui_surface_registry_markdown"]).exists()
    assert result["ui_surface_registry"]["schema"] == "video_knowledge_pipeline.ui_surface_registry.v1"
    assert result["ui_surface_registry"]["surface_count"] == 14
    assert result["ui_surface_registry"]["operator_boundary"]["inventory_only"] is True
    surface_by_id = {row["surface_id"]: row for row in result["ui_surface_registry"]["surfaces"]}
    assert surface_by_id["video_workbench"]["status"] == "current"
    assert surface_by_id["targeted_review"]["status"] == "ready"
    assert surface_by_id["subtitle_editor"]["description"].startswith("复用 MOYS")
    assert result["ui_shell_assets"]["operator_boundary"]["no_runtime_cdn"] is True
    assert Path(result["ui_shell_assets"]["assets"]["tabler_css"]["path"]).exists()
    assert Path(result["ui_shell_assets"]["assets"]["media_chrome_js"]["path"]).exists()

    manifest = read_json(bundle / "manifest.json")
    assert manifest["video_workbench_html"] == "video-workbench.html"
    assert manifest["video_workbench_json"] == "video-workbench.json"
    assert manifest["mcp_video_workbench_args"] == "mcp-video-workbench.args.json"
    assert manifest["ui_surface_registry_json"] == "ui-surface-registry.json"
    assert manifest["ui_surface_registry_markdown"] == "ui-surface-registry.md"

    payload = read_json(bundle / "video-workbench.json")
    assert payload["timeline"][0]["transcript"] == "第一段讲工具选择。"
    assert payload["timeline"][1]["has_visual_understanding"]
    assert payload["review_closure"]["transcript_arbitration"]["open"] == 1
    assert payload["review_closure"]["transcript_arbitration"]["closed"] == 2
    assert payload["transcript_arbitration"]["status"] == "needs_review"
    assert payload["transcript_arbitration"]["quality_summary"]["status"] == "needs_review"
    assert payload["transcript_arbitration"]["quality_summary"]["high_confidence_term_replacements"] == 1
    assert payload["transcript_arbitration"]["quality_summary"]["can_use_as_summary_input"] is False
    assert payload["transcript_arbitration"]["review_rows"][0]["corrected_text"] == "第二段讲封控风险。"
    assert payload["term_correction"]["status"] == "completed"
    assert payload["term_correction"]["accepted_term_count"] == 2
    assert payload["term_correction"]["term_validation_status"] == "ready_for_import"
    assert payload["term_correction"]["accepted_validation_decisions"] == 2
    assert payload["term_correction"]["rejected_validation_decisions"] == 1
    assert payload["term_correction"]["codex_substitute"]["mode"] == "codex_substitute_for_online_text_llm"
    assert payload["term_correction"]["codex_substitute"]["online_llm_api_required"] is False
    assert "validate-term-arbitration-codex-result" in payload["term_correction"]["codex_substitute"]["commands"]["validate_result"]
    assert payload["term_correction"]["source_arbitrated_transcript_exists"] is True
    assert payload["term_correction"]["smart_summary_quality_passed"] is True
    assert payload["term_correction_impact"]["status"] == "passed"
    assert payload["term_correction_impact"]["final_export_alias_total"] == 0
    assert payload["term_correction_impact"]["terms"][0]["canonical"] == "Playwright"
    assert payload["transcript_arbitration"]["term_correction_impact"]["status"] == "passed"
    assert payload["semantic_correction"]["status"] in {"missing_pack", "no_candidates"}
    assert payload["semantic_correction"]["next_action_key"] in {"build_pack", "none"}
    assert payload["evidence_status"]["timeline_alignment"]["issue_count"] == 1
    assert payload["evidence_status"]["tile_review"]["target_count"] == 1
    assert payload["evidence_status"]["video_moment_index"]["chunk_count"] == 2
    assert payload["moment_index"]["chunks"][0]["keywords"] == ["Browser Use", "Playwright"]
    assert payload["moment_index"]["chunks"][0]["snippet"]
    assert payload["moment_index"]["chunks"][0]["evidence_paths"] == ["frames/frame-0000.jpg"]
    assert {row["chunk_kind"] for row in payload["video_rag_chunks"]} == {"review_gap", "content_asset", "chapter_memory", "theme_memory"}
    assert payload["video_rag_chunks"][0]["timeline_indexes"] == [1]
    assert payload["video_rag_status"]["retrieval_backend"] == "sqlite"
    assert payload["video_rag_status"]["sqlite_index_exists"] is True
    assert payload["video_rag_status"]["no_vector_backend_started"] is True
    assert payload["provider_status"]["status"] == "ready"
    assert payload["provider_status"]["vision_provider_smoke"]["provider"] == "volcengine_coding_plan"
    assert payload["provider_status"]["vision_provider_smoke"]["safe_to_execute"] is True
    assert payload["provider_status"]["vision_provider_matrix"]["recommended_provider"] == "volcengine_coding_plan"
    assert payload["provider_status"]["vision_provider_matrix"]["ready_count"] == 1
    assert payload["provider_status"]["local_vlm_serving_smoke"]["status"] == "plan_only"
    assert payload["provider_status"]["local_vlm_serving_smoke"]["capability_counts"] == {"planned": 2}
    assert payload["external_reuse_status"]["schema"] == "video_knowledge_pipeline.external_reuse_workbench_status.v1"
    assert payload["external_reuse_status"]["action_required_count"] >= 2
    external_by_key = {row["key"]: row for row in payload["external_reuse_status"]["capabilities"]}
    assert external_by_key["time_localization"]["status"] == "ready"
    assert external_by_key["long_video_memory"]["status"] == "ready"
    assert external_by_key["video_rag"]["status"] == "action_required"
    assert external_by_key["content_capability"]["status"] == "action_required"
    assert external_by_key["video_edit_handoff"]["status"] == "missing"
    assert "start video rag service" in external_by_key["video_rag"]["retry_commands"]
    assert "retry external capability pack" in external_by_key["content_capability"]["retry_commands"]
    assert payload["subqueue_action_plan"]["schema"] == "video_knowledge_pipeline.subqueue_action_plan.v1"
    action_rows = {row["key"]: row for row in payload["subqueue_action_plan"]["rows"]}
    assert action_rows["timeline_rag:video_rag"]["action_kind"] == "explicit_execution_required"
    assert action_rows["timeline_rag:video_rag"]["primary_command"] == "start video rag service"
    assert action_rows["summary_export:content_candidate"]["action_kind"] == "operator_input_required"
    assert action_rows["summary_export:content_candidate"]["operator_review_required"] is True
    assert payload["content_candidates"]["human_sample_eval"]["status"] == "ready"
    assert payload["content_candidates"]["human_sample_eval"]["candidate_usable_rate"] == 80.0
    assert payload["content_candidates"]["human_sample_eval"]["candidate_evidence_sufficient_rate"] == 60.0
    assert payload["content_candidates"]["filter_counts"]["usable"] == 1
    assert payload["content_candidates"]["filter_counts"]["evidence_low"] == 1
    assert payload["content_candidates"]["filter_counts"]["citation_ready"] == 1
    assert payload["content_candidates"]["filter_counts"]["chapter_linked"] == 1
    assert payload["content_candidates"]["filter_counts"]["chapter_missing"] == 0
    assert payload["content_candidates"]["filter_counts"]["moment_linked"] == 1
    assert payload["content_candidates"]["filter_counts"]["moment_missing"] == 0
    assert payload["content_candidates"]["citation_digest_candidate_count"] == 1
    assert payload["content_candidates"]["linked_moment_candidate_count"] == 1
    assert "usable" in payload["content_candidates"]["candidates"][0]["review_filters"]
    assert "chapter_linked" in payload["content_candidates"]["candidates"][0]["review_filters"]
    assert "chapter_missing" not in payload["content_candidates"]["candidates"][0]["review_filters"]
    assert "moment_linked" in payload["content_candidates"]["candidates"][0]["review_filters"]
    assert "moment_missing" not in payload["content_candidates"]["candidates"][0]["review_filters"]
    assert "evidence_low" in payload["content_candidates"]["candidates"][0]["review_filters"]
    assert "citation_ready" in payload["content_candidates"]["candidates"][0]["review_filters"]
    assert payload["content_candidates"]["candidates"][0]["citation_count"] == 1
    assert "浏览器自动化工具选择方法" in payload["content_candidates"]["candidates"][0]["citation_summary"]
    assert payload["content_candidates"]["candidates"][0]["summary_chapter_refs"][0]["chapter_index"] == 1
    assert payload["content_candidates"]["candidates"][0]["moment_link_count"] >= 2
    assert payload["content_candidates"]["candidates"][0]["moment_links"][0]["id"] == "moment:2"
    assert payload["content_candidates"]["candidates"][0]["moment_links"][1]["label"] == "review_gap"
    chapter = next(row for row in payload["video_rag_chunks"] if row["chunk_kind"] == "chapter_memory")
    assert chapter["memory_level"] == "chapter"
    assert chapter["memory_id"] == "L001"
    assert chapter["child_memory_ids"] == ["M0001", "M0002"]
    assert chapter["child_moment_indexes"] == [1, 2]
    artifact_keys = {row["key"] for row in payload["artifacts"]}
    assert "task_console" in artifact_keys
    assert "ui_surface_registry_json" in artifact_keys
    assert "ui_surface_registry_markdown" in artifact_keys
    assert "targeted_review_html" in artifact_keys
    assert "vision_review_queue_html" in artifact_keys
    assert "multimodal_sample_review_html" in artifact_keys
    assert "model_settings" in artifact_keys
    assert "knowledge_note_full_body_markdown" in artifact_keys
    assert "smart_summary_input_pack_markdown" in artifact_keys
    assert "smart_summary_chapters_markdown" in artifact_keys
    assert "smart_summary_course_map_markdown" in artifact_keys
    assert "video_edit_review_pack_markdown" in artifact_keys
    assert "video_edit_artifact_validation" in artifact_keys
    assert "video_edit_storyboard_candidates" in artifact_keys
    assert "long_video_fast_segment_plan_markdown" in artifact_keys
    assert "long_video_fast_segment_approved_json" in artifact_keys
    assert "long_video_fast_segment_render_receipt" in artifact_keys
    assert "term_correction_impact_report_markdown" in artifact_keys
    assert "term_arbitration_codex_result_codex_markdown" in artifact_keys
    assert "term_correction_closure_markdown" in artifact_keys
    assert "transcript_semantic_correction_pack_json" in artifact_keys
    assert "transcript_semantic_correction_prompt_markdown" in artifact_keys
    assert "transcript_semantic_correction_llm_prompt_markdown" in artifact_keys
    assert "transcript_semantic_correction_result_codex_markdown" in artifact_keys
    assert "transcript_semantic_correction_result_llm_markdown" in artifact_keys
    assert "transcript_semantic_correction_validation_markdown" in artifact_keys
    assert "transcript_semantic_correction_closure_markdown" in artifact_keys
    assert "transcript_semantic_correction_impact_report_markdown" in artifact_keys
    assert "transcript_semantic_correction_readable_impact_markdown" in artifact_keys
    assert "transcript_semantic_correction_status_markdown" in artifact_keys
    summary_queue = next(group for group in payload["processing_queue"]["groups"] if group["key"] == "summary_export")
    assert summary_queue["run_count"] >= 2
    assert summary_queue["action_required"] >= 1
    summary_subqueues = {row["key"]: row for row in summary_queue["subqueues"]}
    assert "rerun smart summary chapters" in summary_subqueues["summary_input"]["retry_commands"]
    assert "retry external capability pack" in summary_subqueues["content_candidate"]["retry_commands"]
    assert "timeline_alignment_issue" in payload["timeline"][1]["evidence_flags"]
    assert "needs_high_res_tile_recovery" in payload["timeline"][1]["evidence_flags"]
    assert "tile_result_needs_review" in payload["timeline"][1]["evidence_flags"]
    assert "transcript_source_conflict" in payload["timeline"][1]["evidence_flags"]
    assert payload["timeline"][1]["transcript_arbitration"]["review_reason"] == "low_arbitration_confidence"
    assert payload["timeline"][1]["timeline_alignment"]["issues"] == ["review_start_mismatch"]
    assert payload["timeline"][1]["tile_review_targets"][0]["tile_id"] == "tile-0001"

    html = (bundle / "video-workbench.html").read_text(encoding="utf-8")
    assert "视频知识工作台" in html
    assert "task-console.html" in html
    assert "targeted-review.html" in html
    assert "精确区间审核" in html
    assert "VKP // LOCAL // PRIMARY SURFACE" in html
    assert "assets/vkp-ui-shell/tabler-1.4.0.min.css" in html
    assert "assets/vkp-ui-shell/tabler-1.4.0.min.js" in html
    assert "assets/vkp-ui-shell/media-chrome-4.19.2.iife.min.js" in html
    assert "<media-controller" in html
    assert "<media-control-bar>" in html
    assert 'id="reviewWorkspace"' in html
    assert 'id="reviewSplitter"' in html
    assert 'role="separator"' in html
    assert "拖动调整视频与文字审核区域宽度" in html
    assert "VIDEO_PANE_LAYOUT_KEY" in html
    assert "bindReviewLayout()" in html
    assert "ResizeObserver" in html
    assert "max-height:48vh" not in html
    assert "画面 56% / 文字 44%" in html
    assert "UI_SURFACE_REGISTRY" in html
    assert "UI REGISTRY / V1" in html
    assert "ui-surface-registry.md" in html
    assert "READY ·" in html
    assert "总览" in html
    assert "评测" in html
    assert "第二段讲封控风险" in html
    assert "复核闭环" in html
    assert "字幕仲裁待复核" in html
    assert "字幕仲裁" in html
    assert "术语纠错闭环" in html
    assert "闭环状态" in html
    assert "通用语义纠错" in html
    assert "transcript_semantic_correction_pack_json" in html
    assert "transcript_semantic_correction_llm_prompt_markdown" in html
    assert "transcript_semantic_correction_readable_impact_markdown" in html
    assert "LLM/Codex 草稿" in html
    assert "build_pack" in html
    assert "已接受术语" in html
    assert "Codex预检" in html
    assert "预检接受/拒绝" in html
    assert "Codex 术语/工具名语义仲裁" in html
    assert "term-arbitration-codex-prompt.md" in html
    assert "term-arbitration-codex-result.codex.md" in html
    assert "Codex 术语回复草稿" in html
    assert "validate-term-arbitration-codex-result" in html
    assert "term-arbitration-codex-validation.md" in html
    assert "term-correction-closure.md" in html
    assert "术语纠错影响" in html
    assert "最终导出残留" in html
    assert "term-correction-impact-report.md" in html
    assert "Playwright" in html
    assert "仲裁质量" in html
    assert "高置信术语" in html
    assert "needs_review" in html
    assert "筛字幕冲突" in html
    assert "selectArbitration" in html
    assert "第二段讲封控风险" in html
    assert "review-closure-status.md" in html
    assert "Citation候选" in html
    assert "有Citation" in html
    assert "浏览器自动化工具选择方法" in html
    assert "证据状态" in html
    assert "时间错位" in html
    assert "Tile 待复核" in html
    assert "片段索引" in html
    assert "片段搜索" in html
    assert "momentSearchInput" in html
    assert "renderMomentSearch" in html
    assert "selectMoment" in html
    assert "selectSearchChunk" in html
    assert "VIDEO_RAG_CHUNKS" in html
    assert "review_gap" in html
    assert "content_asset" in html
    assert "chapter_memory" in html
    assert "theme_memory" in html
    assert "层级：" in html
    assert "child moment" in html
    assert "RAG chunks 4" in html
    assert "检索后端：sqlite / ok" in html
    assert "SQLite index ready" in html
    assert "no vector backend started" in html
    assert "视频 RAG 包" in html
    assert "video-rag-pack.md" in html
    assert "Browser Use" in html
    assert "setFilter('timeline_alignment_issue')" in html
    assert "setFilter('needs_high_res_tile_recovery')" in html
    assert "setFilter('tile_result_needs_review')" in html
    assert "setFilter('transcript_source_conflict')" in html
    assert "证据标记" in html
    assert "Tile 复核" in html
    assert "智能总结输入证据包" in html
    assert "smart-summary-input-pack.md" in html
    assert "智能总结章节证据包" in html
    assert "smart-summary-chapters.md" in html
    assert "课程地图" in html
    assert "course-map.md" in html
    assert "下一步调度" in html
    assert "explicit_execution_required" in html
    assert "operator_input_required" in html
    assert "复制首选命令" in html
    assert "start video rag service" in html
    assert "处理队列" in html
    assert "selectQueue" in html
    assert "QUEUE_GROUPS" in html
    assert "data-queue-key=\"document_ocr\"" in html
    assert "data-queue-key=\"summary_export\"" in html
    assert "rerun smart summary chapters" in html
    assert "队列：" in html
    assert "queue-detail-retry" in html
    assert "retry ebook batch" in html
    assert "ocr_text_empty" in html
    assert "Provider / 本地 VLM" in html
    assert "外部复用能力" in html
    assert "时间定位 / VTimeLLM" in html
    assert "长视频 memory / MovieChat" in html
    assert "VideoRAG 本地检索" in html
    assert "内容素材能力包" in html
    assert "start video rag service" in html
    assert "retry external capability pack" in html
    assert "video-rag-service-plan.md" in html
    assert "external-capability-pack.md" in html
    assert "Provider Smoke" in html
    assert "Provider Matrix" in html
    assert "本地 VLM Smoke" in html
    assert "volcengine_coding_plan" in html
    assert "plan_only" in html
    assert "read-only" in html
    assert "内容素材候选" in html
    assert "content-candidate-row" in html
    assert "filterContentCandidates" in html
    assert "chapter_linked" in html
    assert "chapter_missing" in html
    assert "moment_linked" in html
    assert "moment_missing" in html
    assert "已关联章节" in html
    assert "已关联片段" in html
    assert "片段互链" in html
    assert "关联片段" in html
    assert "moment #2" in html
    assert "只看未抽样" in html
    assert "抽样证据不足" in html
    assert "可继续加工" in html
    assert "候选可用率" in html
    assert "证据充分率" in html
    assert "80.0%" in html
    assert "60.0%" in html
    assert "human-sample-eval.md" in html
    assert "evidence_low" in html
    assert "usable" in html


def test_task_console_links_video_workbench(tmp_path: Path) -> None:
    bundle = _fresh_bundle(tmp_path)
    _write_bundle(bundle)

    result = export_task_console(bundle)

    manifest = read_json(bundle / "manifest.json")
    assert manifest["video_workbench_html"] == "video-workbench.html"
    assert manifest["mcp_video_workbench_args"] == "mcp-video-workbench.args.json"
    assert (bundle / "mcp-video-workbench.args.json").exists()

    artifact_keys = {row["key"] for row in result["artifacts"]}
    command_keys = {row["key"] for row in result["commands"]}
    assert "video_workbench_html" in artifact_keys
    assert "knowledge_note_full_body_markdown" in artifact_keys
    assert "video_workbench" in command_keys

    html = (bundle / "task-console.html").read_text(encoding="utf-8")
    assert "视频知识工作台" in html
    assert "export-video-workbench" in html
