from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.content_asset_status import content_asset_status, _semantic_asset_gate
from video_knowledge_pipeline.content_asset_batch import batch_content_asset_status, content_handoff_pack
from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.transcript_semantic_correction import build_transcript_semantic_correction_pack
from video_knowledge_pipeline.smart_summary_chapters import build_smart_summary_chapter_pack
from video_knowledge_pipeline.smart_summary_codex import generate_smart_summary_with_codex, smart_summary_quality_check
from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
from video_knowledge_pipeline.transcript_sidecar import ensure_review_transcript_sidecar
from video_knowledge_pipeline.transcript_postprocess import postprocess_asr_transcript


def test_review_transcript_sidecar_prefers_source_arbitrated_transcript(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-sidecar-source-arbitrated"
    bundle.mkdir()
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        "title": "Sidecar Source Arbitrated",
        "normalized_transcript_json": "normalized-transcript.json",
        "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
    }
    timeline = [{"index": 1, "start": 0, "end": 5, "transcript": "raw wrong tool"}]
    (bundle / "normalized-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 5, "text": "raw wrong tool"}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "source-arbitrated-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 5, "text": "corrected tool name"}]}, ensure_ascii=False), encoding="utf-8")

    sidecar = ensure_review_transcript_sidecar(bundle, manifest, timeline, title="Sidecar Source Arbitrated", write=True)

    assert sidecar["source"] == "source_arbitrated_transcript_json"
    assert sidecar["path"].endswith("source-arbitrated-transcript.json")
    assert manifest["review_transcript_sidecar"]["path"].endswith("source-arbitrated-transcript.json")


def _write_semantic_asset_gate_passed(bundle: Path) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    payloads = {
        "transcript-semantic-correction-pack.json": {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 1,
            "candidates": [{"candidate_id": "semcorr-test-0001", "segment_index": 0, "start": 0, "end": 2, "original_text": "old", "candidate_text": "new"}],
        },
        "transcript-semantic-correction-validation.json": {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_validation.v1",
            "status": "accepted",
            "accepted_decision_count": 1,
            "rejected_decision_count": 0,
            "review_required_count": 0,
            "accepted_decisions": [{"candidate_id": "semcorr-test-0001", "action": "keep_original", "original_text": "old", "corrected_text": "old", "confidence": 1.0}],
            "rejected_decisions": [],
            "review_rows": [],
        },
        "transcript-semantic-correction-closure.json": {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_closure.v1",
            "status": "completed",
            "ok": True,
            "applied_correction_count": 1,
            "changed_segment_count": 0,
        },
        "transcript-semantic-correction-impact-report.json": {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_impact_report.v1",
            "status": "passed",
            "ok": True,
            "final_residual_error_total": 0,
        },
        "transcript-semantic-readable-impact-report.json": {
            "schema": "video_knowledge_pipeline.transcript_semantic_readable_impact_report.v1",
            "status": "passed",
            "ok": True,
            "required_readable_residual_total": 0,
        },
        "transcript-semantic-summary-impact-report.json": {
            "schema": "video_knowledge_pipeline.transcript_semantic_summary_impact_report.v1",
            "status": "passed",
            "ok": True,
            "summary_absorption_rate": 1.0,
            "summary_residual_original_total": 0,
        },
    }
    for name, payload in payloads.items():
        (bundle / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def test_export_includes_source_channels_demo_notes_and_crop_audit(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    crop = bundle / "ocr-crops" / "timeline-0001" / "central_content.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"fake crop")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Export Test"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "acceptance-check.json").write_text(
        json.dumps({"status": "accepted_with_known_gaps", "next_action": {"key": "none"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 4,
                    "transcript": "老师讲软件界面。",
                    "visual_route": "semantic_frame",
                    "visual_text": "按钮：保存",
                    "visual_understanding": {"actions": ["展示保存按钮"], "interface_state": "设置页面", "evidence_frame_paths": [str(crop)]},
                    "screen_text_recovery": {"crop_paths": [str(crop)]},
                    "human_review": {"status": "accepted_known_gap", "comment": "小字已尽力恢复"},
                    "review_status": "accepted_known_gap",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _write_semantic_asset_gate_passed(bundle)

    exports_dir = bundle / "exports"
    exports_dir.mkdir()
    (exports_dir / "smart-summary-chapters.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.smart_summary_chapter_pack.v1",
                "chapters": [
                    {
                        "chapter_index": 1,
                        "citation_digest": [
                            {
                                "source_type": "visual",
                                "time": "00:00:00.000 - 00:00:04.000",
                                "timeline_indexes": [1],
                                "text": "保存按钮证据来自单帧视觉和 OCR。",
                                "evidence_paths": [str(crop)],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Export Test")

    note = Path(result["note_path"]).read_text(encoding="utf-8")
    transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    full_body = Path(result["full_body_path"]).read_text(encoding="utf-8")
    audit = Path(result["extraction_audit_path"]).read_text(encoding="utf-8")
    key_segments = Path(result["content_assets"]["key_segments_path"]).read_text(encoding="utf-8")
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    candidate_pack = json.loads((bundle / "exports" / "content-candidate-pack.json").read_text(encoding="utf-8"))
    assert candidate_pack["candidates"][0]["summary_chapter_refs"][0]["chapter_index"] == 1
    assert candidate_pack["summary_chapter_link_status"]["linked_candidate_count"] >= 1
    linked_chapters = json.loads((bundle / "exports" / "smart-summary-chapters.json").read_text(encoding="utf-8"))
    assert linked_chapters["content_candidate_pack_linked"] is True
    assert linked_chapters["chapters"][0]["linked_content_candidates"][0]["id"] == "candidate-001"
    linked_chapter_markdown = (bundle / "exports" / "smart-summary-chapters.md").read_text(encoding="utf-8")
    assert "Linked Content Candidates" in linked_chapter_markdown
    assert "  - 📑 智能总结" in note
    assert note.index("  - 📑 智能总结") < note.index("- 正文") < note.index("- 逐字稿")
    assert "- 逐字稿" in note
    assert "老师讲软件界面。" in full_body
    assert "00:00:" not in full_body
    assert result["content_assets"]["full_body_path"] == result["full_body_path"]
    assert "#### 信息来源" not in note
    assert "#### 演示了什么" in transcript
    assert "展示保存按钮" in transcript
    assert "OCR 裁剪证据" in audit
    assert str(crop) in audit
    assert "最终验收状态" in audit
    assert "accepted_with_known_gaps" in audit
    assert result["content_assets"]["review_required"] is True
    assert result["content_assets"]["publication_allowed"] is False
    assert Path(result["content_material_card_path"]).exists()
    assert Path(result["content_material_card_markdown_path"]).exists()
    material_card = json.loads(Path(result["content_material_card_path"]).read_text(encoding="utf-8"))
    assert material_card["review_required"] is True
    assert material_card["publication_allowed"] is False
    assert material_card["allowed_as_inspiration"] is True
    assert material_card["allowed_as_fact"] is False
    assert material_card["circle_of_friends_status"] == "needs_review_inspiration"
    assert material_card["term_correction"]["status"] == "needs_term_arbitration"
    assert material_card["term_correction"]["term_validation_status"] == "missing"
    assert "fact_check_before_claiming_truth" in material_card["human_confirmation_required"]
    assert summary["content_assets"]["content_material_card_path"].endswith("content-material-card.json")
    assert summary["content_assets"]["content_candidate_pack_path"].endswith("content-candidate-pack.json")
    assert summary["content_assets"]["content_candidate_pack_markdown_path"].endswith("content-candidate-pack.md")
    assert summary["content_assets"]["publication_allowed"] is False
    candidate_pack_path = Path(result["content_candidate_pack_path"])
    candidate_pack_markdown_path = Path(result["content_candidate_pack_markdown_path"])
    assert candidate_pack_path.exists()
    assert candidate_pack_markdown_path.exists()
    candidate_pack = json.loads(candidate_pack_path.read_text(encoding="utf-8"))
    candidate_pack_markdown = candidate_pack_markdown_path.read_text(encoding="utf-8")
    assert candidate_pack["review_required"] is True
    assert candidate_pack["publication_allowed"] is False
    assert candidate_pack["allowed_as_fact"] is False
    assert candidate_pack["allowed_as_inspiration"] is True
    assert candidate_pack["term_correction"]["status"] == "needs_term_arbitration"
    assert candidate_pack["term_correction"]["term_validation_status"] == "missing"
    assert candidate_pack["candidates"]
    assert candidate_pack["candidates"][0]["evidence_paths"]
    assert candidate_pack["candidates"][0]["term_correction_status"] == "needs_term_arbitration"
    assert candidate_pack["candidates"][0]["term_validation_status"] == "missing"
    assert candidate_pack["citation_digest_candidate_count"] == 1
    assert candidate_pack["candidates"][0]["citation_digest_status"] == "ready"
    assert candidate_pack["candidates"][0]["evidence_citations"]
    assert any("保存按钮证据" in str(row.get("text") or "") for row in candidate_pack["candidates"][0].get("evidence_citations", [])) or candidate_pack["candidates"][0]["evidence_citations"]
    assert "visual_explainer" in candidate_pack["candidates"][0]["candidate_types"]
    assert "内容素材候选包" in candidate_pack_markdown
    assert "证据引用 / Citation Digest" in candidate_pack_markdown
    assert "Citation Digest" in candidate_pack_markdown
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["knowledge_note_full_body_markdown"] == result["full_body_path"]
    assert manifest["knowledge_note_export"]["full_body_path"] == result["full_body_path"]
    assert manifest["content_candidate_pack_json"] == "exports/content-candidate-pack.json"
    assert manifest["content_candidate_pack_markdown"] == "exports/content-candidate-pack.md"
    assert "关键片段候选" in key_segments
    assert str(crop) in key_segments





def test_export_runs_transcript_evidence_pipeline_by_default(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-export-default-pipeline"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Default Pipeline", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 4, "text": "今天讲 play right m c p。"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 0, "start": 0, "end": 4, "transcript": "今天讲 play right m c p。", "visual_text": "Playwright MCP"}], ensure_ascii=False),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Default Pipeline")

    assert (bundle / "transcript-evidence-correction-pipeline.json").exists()
    assert (bundle / "transcript-semantic-correction-pack.json").exists()
    assert (bundle / "evidence-conflict-index.json").exists()
    assert not (bundle / "corrected-transcript.json").exists()
    assert result["transcript_evidence_correction_pipeline"]["status"] != "skipped"
    assert result["smart_summary_input_pack"]["transcript_source"].endswith("normalized-transcript.json")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["transcript_evidence_correction_pipeline_summary"]
    assert summary["status"] != "skipped"
    assert "semantic_candidate_count" in summary
    pipeline = json.loads((bundle / "transcript-evidence-correction-pipeline.json").read_text(encoding="utf-8"))
    step_names = {str(step.get("name") or "") for step in pipeline.get("steps", []) if isinstance(step, dict)}
    assert "semantic_correction_pack" in step_names
    assert "evidence_conflict_index" in step_names
    assert "transcript_quality_gate" in step_names


def test_export_can_skip_transcript_evidence_pipeline_for_legacy_debug(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-export-skip-pipeline"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Skip Pipeline", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 4, "text": "legacy raw transcript"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 0, "start": 0, "end": 4, "transcript": "legacy raw transcript"}], ensure_ascii=False),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Skip Pipeline", run_transcript_evidence_check=False)

    assert result["transcript_evidence_correction_pipeline"]["status"] == "skipped"
    assert not (bundle / "transcript-evidence-correction-pipeline.json").exists()


def test_smart_summary_uses_asr_sidecar_and_writes_codex_prompt(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smart-summary"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Smart Long Video", "media_path": "D:/media/long.mp4"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 5,
                    "transcript": "这是抽帧时间线里的开头片段。",
                    "visual_route": "semantic_frame",
                    "quality_issues": ["semantic_frame_without_analysis"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 5, "text": "开头讲客户信任和成交原则。"},
                    {"start": 1800, "end": 1810, "text": "三十分钟后继续讲陌客沟通的问题链和复盘动作。"},
                    {"start": 3590, "end": 3600, "text": "最后总结要记录客户问题并形成下一步跟进清单。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Smart Long Video")

    smart = Path(result["smart_summary_path"]).read_text(encoding="utf-8")
    prompt = Path(result["smart_summary_prompt_path"]).read_text(encoding="utf-8")
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    candidate_pack = json.loads((bundle / "exports" / "content-candidate-pack.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert "# Smart Long Video - 智能总结" in smart
    assert "needs_llm_summary" in smart
    assert "三十分钟后继续讲陌客沟通的问题链" in prompt
    assert "最后总结要记录客户问题" in prompt
    assert "Visual evidence status:" in prompt
    assert "Codex Smart Summary Prompt" in prompt
    assert "必须覆盖完整视频时长" in prompt
    assert summary["smart_summary_path"].endswith("smart-summary.md")
    assert summary["content_assets"]["smart_summary_path"].endswith("smart-summary.md")
    assert manifest["knowledge_note_smart_summary_markdown"].endswith("smart-summary.md")
    assert Path(result["content_assets"]["smart_summary_prompt_path"]).exists()


def test_full_transcript_uses_asr_sidecar_not_sampled_timeline(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-full-transcript"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Long Video"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 5,
                    "transcript": "这是抽帧时间线里的开头片段。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 5, "text": "这是完整 ASR 的开头。"},
                    {"start": 1200, "end": 1205, "text": "这是二十分钟后的内容，不能被抽帧上限截掉。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Long Video")

    transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    assert "Source: `normalized_asr`" not in transcript
    assert "Arbitration status: `raw_asr_fallback_not_arbitrated`" not in transcript
    assert "来源状态：`normalized_asr`" not in transcript
    assert "仲裁状态：`raw_asr_fallback_not_arbitrated`" not in transcript
    assert "这是二十分钟后的内容，不能被抽帧上限截掉。" in transcript
    assert "这是抽帧时间线里的开头片段。" not in transcript

def test_export_ignores_parse_failed_visual_understanding(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-parse-failed"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "parse failed"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 3,
                    "transcript": "这里看屏幕。",
                    "visual_route": "semantic_frame",
                    "visual_understanding": {
                        "schema": "lecture_visual_understanding.v1",
                        "parse_failed": True,
                        "validation_status": "incomplete",
                        "raw_model_output": "not valid json",
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="parse failed")

    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    candidate_pack = json.loads((bundle / "exports" / "content-candidate-pack.json").read_text(encoding="utf-8"))
    transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    assert summary["summary"]["items_with_visual_understanding"] == 0
    assert summary["summary"]["visual_understanding_missing"] == [1]
    assert "画面未可靠提取" in transcript
    assert "not valid json" not in transcript
def test_content_asset_status_reports_export_required_and_ready(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "status test"}), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "start": 0, "end": 2, "transcript": "这一段可作为选题灵感。"}], ensure_ascii=False),
        encoding="utf-8",
    )
    _write_semantic_asset_gate_passed(bundle)

    before = content_asset_status(bundle)
    assert before["status"] == "export_required"
    assert before["ok"] is False
    assert "run_export_knowledge_note" in before["next_actions"]

    build_smart_summary_chapter_pack(bundle, title="status test", target_chapters=1)
    export_knowledge_note(bundle, title="status test", run_transcript_evidence_check=False)
    after = content_asset_status(bundle)
    assert after["status"] == "ready_for_inspiration_review"
    assert after["ok"] is True
    assert after["publication_allowed"] is False
    assert after["review_required"] is True
    assert after["allowed_as_inspiration"] is True
    assert after["allowed_as_fact"] is False
    assert after["missing_fields"] == []
    assert after["content_candidate_pack_exists"] is True
    assert after["content_candidate_pack_safe"] is True
    assert after["content_candidate_count"] >= 1
    assert after["content_candidate_chapter_ref_count"] >= 1
    assert after["content_candidate_linked_chapter_count"] == 1
    assert after["content_candidate_chapter_refs_available"] is True
    assert after["content_candidate_linked_chapters"][0]["chapter_index"] == 1
    assert after["content_candidate_pack_path"].endswith("content-candidate-pack.json")
    assert after["content_candidate_pack_markdown_path"].endswith("content-candidate-pack.md")
    assert after["human_sample_eval_status"] == "not_available"
    assert after["term_correction_status"] == "needs_term_arbitration"
    assert after["term_validation_status"] == "missing"
    assert after["human_sample_eval_exists"] is False
    (bundle / "human-sample-eval.json").write_text(
        json.dumps(
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
                "interpretation": {"verdict": "multimodal_mixed_but_useful"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "human-sample-eval.md").write_text("# Human sample eval\n", encoding="utf-8")
    sampled = content_asset_status(bundle)
    assert sampled["status"] == "ready_for_inspiration_review"
    assert sampled["human_sample_eval_status"] == "ready"
    assert sampled["human_sample_eval_exists"] is True
    assert sampled["human_sample_eval_labeled_rows"] == 3
    assert sampled["human_sample_eval_content_candidate_usable_rate"] == 80.0
    assert sampled["human_sample_eval_content_candidate_evidence_sufficient_rate"] == 60.0
    assert sampled["human_sample_eval_multimodal_net_help_rate"] == 25.0


def test_batch_content_asset_status_and_handoff_pack_only_use_safe_ready_cards(tmp_path: Path) -> None:
    ready = tmp_path / "ready" / "webui-bundle"
    ready.mkdir(parents=True)
    (ready / "manifest.json").write_text(json.dumps({"title": "ready"}, ensure_ascii=False), encoding="utf-8")
    (ready / "timeline.json").write_text(json.dumps([{"index": 1, "transcript": "可作为灵感。"}], ensure_ascii=False), encoding="utf-8")
    _write_semantic_asset_gate_passed(ready)
    build_smart_summary_chapter_pack(ready, title="ready", target_chapters=1)
    export_knowledge_note(ready, title="ready", run_transcript_evidence_check=False)
    (ready / "human-sample-eval.json").write_text(
        json.dumps(
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
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (ready / "human-sample-eval.md").write_text("# Human sample eval\n", encoding="utf-8")
    (ready / "term-correction-closure.json").write_text(
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
    (ready / "term-arbitration-codex-validation.json").write_text(
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
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (ready / "transcript-semantic-correction-pack.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "semcorr-0001",
                        "segment_index": 0,
                        "start": 0,
                        "end": 4,
                        "time_range": "00:00:00.000 - 00:00:04.000",
                        "correction_type": "proper_noun",
                        "risk_level": "medium",
                        "original_text": "browser base",
                        "candidate_text": "Browserbase",
                        "suggested_text": "Browserbase",
                        "evidence_source_types": ["ocr"],
                        "evidence": [{"evidence_id": "timeline_0_visual_text", "source_type": "ocr", "text": "Browserbase"}],
                        "source_support_summary": {
                            "supports_candidate": ["ocr"],
                            "supports_original": ["asr_or_subtitle"],
                            "neutral": [],
                            "candidate_weight": 70,
                            "original_weight": 40,
                            "neutral_weight": 0,
                            "weight_margin": 30,
                            "dominant_side": "candidate",
                            "needs_review_by_source_vote": False,
                            "has_source_conflict": True,
                            "votes": [{
                                "source_type": "ocr",
                                "source_weight": 70,
                                "vote": "supports_candidate",
                                "text_excerpt": "Browserbase",
                            }],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (ready / "transcript-semantic-correction-validation.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_validation.v1",
                "status": "accepted",
                "accepted_decision_count": 1,
                "rejected_decision_count": 0,
                "review_required_count": 0,
                "accepted_decisions": [{"candidate_id": "semcorr-0001", "action": "keep_original", "correction_type": "proper_noun", "original_text": "browser base", "corrected_text": "browser base", "confidence": 1.0, "human_confirmed": True}],
                "rejected_decisions": [],
                "review_rows": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (ready / "transcript-semantic-correction-result.review.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
                "source": "human_review_notes",
                "decisions": [{"candidate_id": "semcorr-0001", "action": "keep_original", "correction_type": "proper_noun", "original_text": "browser base", "corrected_text": "browser base", "confidence": 1.0, "human_confirmed": True}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    missing = tmp_path / "missing" / "webui-bundle"
    missing.mkdir(parents=True)
    (missing / "manifest.json").write_text(json.dumps({"title": "missing"}, ensure_ascii=False), encoding="utf-8")
    (missing / "timeline.json").write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")

    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps({"items": [{"bundle_dir": str(ready)}, {"bundle_dir": str(missing)}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    status = batch_content_asset_status(manifest, output_dir=tmp_path / "out")
    assert status["count"] == 2
    assert status["status_counts"]["ready_for_inspiration_review"] == 1
    assert status["status_counts"]["export_required"] == 1
    assert status["items"][0]["content_candidate_pack_path"].endswith("content-candidate-pack.json")
    assert status["items"][0]["content_candidate_count"] >= 1
    assert status["items"][0]["content_candidate_linked_chapter_count"] == 1
    assert status["items"][0]["content_candidate_chapter_ref_count"] >= 1
    assert status["items"][0]["content_candidate_linked_chapters"][0]["chapter_index"] == 1
    assert status["items"][0]["human_sample_eval_status"] == "ready"
    assert status["items"][0]["human_sample_eval_content_candidate_usable_rate"] == 80.0
    assert status["items"][0]["human_sample_eval_content_candidate_evidence_sufficient_rate"] == 60.0
    assert status["items"][0]["term_correction_status"] == "needs_codex_term_validation"
    assert status["items"][0]["term_validation_status"] == "no_accepted_decisions"
    assert status["items"][0]["validation_rejection_reasons"][0] == {"reason": "missing_evidence_indexes", "count": 1}
    assert status["items"][0]["validation_rejected_decisions"][0]["canonical"] == "UnsafeTerm"
    assert status["items"][0]["term_next_action_key"] == "term_arbitration_codex_validate"
    assert status["items"][0]["semantic_correction_status"] == "impact_passed"
    assert status["items"][0]["semantic_correction_ui_summary"]["ui_state"] == "closed_and_export_checked"
    assert status["items"][0]["semantic_correction_ui_summary"]["auto_candidate_count"] == 1
    assert status["items"][0]["semantic_correction_ui_summary"]["accepted_decision_type_counts"]["proper_noun"] == 1
    assert status["items"][0]["semantic_correction_candidate_count"] == 1
    assert status["items"][0]["semantic_correction_review_count"] == 0
    assert status["items"][0]["semantic_correction_candidate_type_counts"]["proper_noun"] == 1
    assert status["items"][0]["semantic_correction_evidence_source_counts"]["ocr"] == 1
    assert status["items"][0]["semantic_correction_source_vote_summary"]["candidate_count_with_votes"] == 1
    assert status["items"][0]["semantic_correction_source_vote_summary"]["by_dominant_side"]["candidate"] == 1
    assert status["items"][0]["semantic_correction_review_closure_summary"]["closed_review_decision_count"] == 1
    assert status["semantic_correction_summary"]["candidate_count"] == 1
    assert status["semantic_correction_summary"]["accepted_count"] == 1
    assert status["semantic_correction_summary"]["closed_review_decision_count"] == 1
    assert status["semantic_correction_summary"]["auto_candidate_count"] == 1
    assert status["semantic_correction_summary"]["human_review_candidate_count"] == 0
    assert status["semantic_correction_summary"]["by_ui_state"]["closed_and_export_checked"] == 1
    assert status["semantic_correction_summary"]["by_accepted_type"]["proper_noun"] == 1
    assert status["semantic_correction_summary"]["by_status"]["impact_passed"] == 1
    assert sum(status["semantic_correction_summary"]["by_status"].values()) == 2
    assert status["semantic_correction_summary"]["by_candidate_type"]["proper_noun"] == 1
    assert status["semantic_correction_summary"]["by_risk_level"]["medium"] == 1
    assert status["semantic_correction_summary"]["by_evidence_source"]["ocr"] == 1
    assert status["semantic_correction_summary"]["source_vote_candidate_count"] == 1
    assert status["semantic_correction_summary"]["source_conflict_count"] == 1
    assert status["semantic_correction_summary"]["by_source_vote_dominant_side"]["candidate"] == 1
    assert status["semantic_correction_summary"]["by_candidate_support_source"]["ocr"] == 1
    assert status["semantic_correction_summary"]["by_review_action"]["keep_original"] == 1
    assert status["items"][0]["semantic_correction_chapter_risk_summary"][0]["chapter_index"] == 1
    assert status["semantic_correction_summary"]["chapter_risk_items"][0]["chapter_index"] == 1
    assert "validate-term-arbitration-codex-result" in status["items"][0]["term_optional_next_actions"][0]
    assert status["items"][0]["term_optional_next_action_artifacts"]["term_arbitration_codex_validate"].endswith("mcp-term-arbitration-codex-validate.args.json")
    assert status["items"][0]["term_optional_next_action_artifacts"]["term_correction_closure_codex"].endswith("mcp-term-correction-closure-codex.args.json")
    batch_markdown = Path(status["markdown_path"]).read_text(encoding="utf-8")
    assert "Candidates" in batch_markdown
    assert "Chapters" in batch_markdown
    assert "Term blockers" in batch_markdown
    assert "Term action" in batch_markdown
    assert "Semantic correction" in batch_markdown
    assert "Semantic review" in batch_markdown
    assert "Semantic Correction Summary" in batch_markdown
    assert "Auto candidates" in batch_markdown
    assert "UI state" in batch_markdown
    assert "Accepted type" in batch_markdown
    assert "Source conflicts" in batch_markdown
    assert "Source vote dominant side" in batch_markdown
    assert "Candidate support source" in batch_markdown
    assert "Candidate type" in batch_markdown
    assert "Evidence source" in batch_markdown
    assert "proper_noun" in batch_markdown
    assert "medium" in batch_markdown
    assert "Chapter Risk Items" in batch_markdown
    assert "impact_passed" in batch_markdown
    assert "closed=1" in batch_markdown
    assert "term_arbitration_codex_validate" in batch_markdown
    assert "missing_semantic_rationale" in batch_markdown
    assert Path(status["json_path"]).exists()
    assert Path(status["markdown_path"]).exists()

    handoff = content_handoff_pack(manifest, output_dir=tmp_path / "handoff")
    assert handoff["ready_count"] == 1
    assert handoff["items"][0]["publication_allowed"] is False
    assert handoff["items"][0]["allowed_as_fact"] is False
    assert handoff["items"][0]["content_candidate_pack_safe"] is True
    assert handoff["items"][0]["content_candidate_pack_path"].endswith("content-candidate-pack.json")
    assert handoff["items"][0]["content_candidate_linked_chapter_count"] == 1
    assert handoff["items"][0]["content_candidate_linked_chapters"][0]["candidate_count"] >= 1
    assert handoff["items"][0]["human_sample_eval_status"] == "ready"
    assert handoff["items"][0]["human_sample_eval_content_candidate_usable_rate"] == 80.0
    assert handoff["items"][0]["term_correction_status"] == "needs_codex_term_validation"
    assert handoff["items"][0]["term_validation_status"] == "no_accepted_decisions"
    assert handoff["items"][0]["validation_rejected_decisions"][0]["canonical"] == "UnsafeTerm"
    assert handoff["items"][0]["term_next_action_key"] == "term_arbitration_codex_validate"
    assert handoff["items"][0]["semantic_correction_review_closure_summary"]["closed_review_decision_count"] == 1
    assert handoff["semantic_correction_summary"]["closed_review_decision_count"] == 1
    assert handoff["semantic_correction_summary"]["by_ui_state"]["closed_and_export_checked"] == 1
    assert handoff["semantic_correction_summary"]["source_vote_candidate_count"] == 1
    assert handoff["semantic_correction_summary"]["by_source_vote_dominant_side"]["candidate"] == 1
    assert handoff["semantic_correction_summary"]["chapter_risk_items"][0]["chapter_index"] == 1
    assert "validate-term-arbitration-codex-result" in handoff["items"][0]["term_optional_next_actions"][0]
    assert handoff["items"][0]["term_optional_next_action_artifacts"]["term_correction_closure_codex"].endswith("mcp-term-correction-closure-codex.args.json")
    handoff_markdown = Path(handoff["markdown_path"]).read_text(encoding="utf-8")
    assert "Content candidate pack" in handoff_markdown
    assert "Smart Summary Chapter Links" in handoff_markdown
    assert "Codex term validation" in handoff_markdown
    assert "Term validation blockers" in handoff_markdown
    assert "Optional term action" in handoff_markdown
    assert "Optional term MCP args" in handoff_markdown
    assert "validate-term-arbitration-codex-result" in handoff_markdown
    assert "mcp-term-correction-closure-codex.args.json" in handoff_markdown
    assert "Semantic Correction Summary" in handoff_markdown
    assert "Chapter Risk Items" in handoff_markdown
    assert Path(handoff["json_path"]).exists()
    assert Path(handoff["markdown_path"]).exists()

# Moved from test_video_pipeline_smoke.py during Phase 10 split.

import json
from pathlib import Path

from video_knowledge_pipeline.acceptance_check import acceptance_check
from video_knowledge_pipeline.acceptance_run import run_acceptance_bundle, run_acceptance_run
from video_knowledge_pipeline.asr_adapter import normalize_asr_output
from video_knowledge_pipeline.asr_environment import asr_environment_status
from video_knowledge_pipeline.asr_execution import asr_smoke, run_asr_plan
from video_knowledge_pipeline.asr_runner import plan_asr_run
from video_knowledge_pipeline.batch_run import batch_video_knowledge_run
from video_knowledge_pipeline.bundle_next import bundle_advance, bundle_advance_log, bundle_advance_queue, bundle_next_action
from video_knowledge_pipeline.bundle_status import bundle_status_report, controlled_execution_check
from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser, main as cli_main, resolve_mcp_args_path, run_mcp_call
from video_knowledge_pipeline.config import config_status, resolve_vision_execution_profile, service_url, vision_execution_profile
from video_knowledge_pipeline.controlled_execution_smoke import controlled_execution_smoke
from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.transcript_semantic_correction import build_transcript_semantic_correction_pack
from video_knowledge_pipeline.lecture_package import render_lecture_review_html
from video_knowledge_pipeline.local_video_run import prepare_local_video_run
from video_knowledge_pipeline.local_vlm_server_adapter import local_vlm_adapter_plan
from video_knowledge_pipeline.ocr_backfill import run_ocr_backfill
from video_knowledge_pipeline.multimodal_frame_analyzer import (
    _normalise_visual_understanding,
    run_multimodal_frame_analysis,
    vision_analysis_apply_restore,
    vision_analysis_restore_plan,
    vision_analysis_run_log,
)
from video_knowledge_pipeline.peepshow_adapter import attach_peepshow_output_to_bundle
from video_knowledge_pipeline.review_session import apply_review_notes_to_bundle, prepare_review_session, validate_review_notes_for_bundle
from video_knowledge_pipeline.source_artifacts import build_source_artifact_index, summarize_manifest_source_artifacts
from video_knowledge_pipeline.storage import bundle_write_lock, write_json
from video_knowledge_pipeline.temporal_frame_groups import run_temporal_frame_groups
from video_knowledge_pipeline.temporal_visual_analyzer import _normalise_temporal_understanding, run_temporal_visual_analysis
from video_knowledge_pipeline.transcript_resegment import resegment_transcript
from video_knowledge_pipeline.vision_acceptance import vision_acceptance_plan
from video_knowledge_pipeline.video_frame_router import run_video_frame_router
from video_knowledge_pipeline.video_source import prepare_video_source
from video_knowledge_pipeline.vision_api import parse_model_json, resolve_provider_config, test_vision_provider as run_vision_provider_test
from video_knowledge_pipeline.vision_environment import vision_environment_status
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight
from video_knowledge_pipeline.vision_provider_smoke import rank_vision_providers, vision_provider_matrix, vision_provider_smoke
from video_knowledge_pipeline.webui_bridge import export_webui_bundle, refresh_bundle_review_html
import video_knowledge_pipeline.visual_structure as visual_structure
from video_knowledge_pipeline.visual_structure import run_visual_structure_plan



def test_source_artifact_summary_uses_local_video_fallback(tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"fake video")
    manifest = {
        "sources": [
            {
                "video_id": "local-1",
                "title": "local lesson",
                "path": str(video),
                "source_artifacts": {},
            }
        ]
    }

    summary = summarize_manifest_source_artifacts(manifest)

    assert summary["source_count"] == 1
    assert summary["sources_with_artifacts"] == 1
    assert summary["artifact_count"] == 1
    assert summary["tools"] == ["local_video"]
    assert summary["ok"] is True


def test_source_artifact_index_includes_local_video_and_bundle_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"fake video")
    package_json = tmp_path / "lecture-package.json"
    package_json.write_text("{}", encoding="utf-8")
    manifest_json = tmp_path / "manifest.json"
    manifest_json.write_text("{}", encoding="utf-8")
    timeline_json = tmp_path / "timeline.json"
    timeline_json.write_text("[]", encoding="utf-8")

    index = build_source_artifact_index(
        {
            "title": "traceable lesson",
            "source_package": str(package_json),
            "manifest_path": str(manifest_json),
            "timeline_path": str(timeline_json),
            "sources": [
                {
                    "video_id": "local-1",
                    "title": "local lesson",
                    "path": str(video),
                    "source_artifacts": {},
                }
            ],
        }
    )

    keys = {row["key"] for row in index["artifacts"]}
    assert "video" in keys
    assert "source_package" in keys
    assert "manifest" in keys
    assert "timeline" in keys
    assert index["available_count"] == 4
    assert "local_video" in index["tools"]
    assert "local_bundle" in index["tools"]


def test_export_knowledge_note_includes_transcript_visual_and_gaps(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake")
    (bundle / "manifest.json").write_text(
        json.dumps({"title": "测试课程", "knowledge_coverage": {"status": "blocked"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 4,
                    "transcript": "第一段讲口头知识。",
                    "visual_route": "semantic_frame",
                    "visual_understanding": {"objects": ["编辑器"], "actions": ["展示规则"], "confidence": 0.8},
                    "assets": [{"path": "assets/frame.jpg", "copied": "true"}],
                },
                {
                    "index": 2,
                    "start": 4,
                    "end": 8,
                    "transcript": "第二段讲连续操作。",
                    "visual_route": "temporal_sequence",
                    "temporal_visual_understanding": {"event_sequence": ["滚动页面"], "operation_steps": ["查看规则"], "confidence": 0.7},
                    "assets": [{"path": "assets/frame.jpg", "copied": "true"}],
                },
                {
                    "index": 3,
                    "start": 8,
                    "end": 12,
                    "transcript": "第三段有表格。",
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "review_status": "accepted",
                    "human_keep_image": True,
                    "human_review": {"status": "accepted", "comment": "表格需要保留截图核对。"},
                    "assets": [{"path": "assets/frame.jpg", "copied": "true"}],
                },
                {
                    "index": 4,
                    "start": 12,
                    "end": 16,
                    "transcript": "第四段需要看屏幕。",
                    "visual_route": "semantic_frame",
                    "quality_issues": ["semantic_frame_without_analysis"],
                    "assets": [{"path": "assets/frame.jpg", "copied": "true"}],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (bundle / "term-arbitration-glossary.json").write_text(
        json.dumps({"terms": [{"canonical_term": "Playwright", "aliases": ["play right"], "review_required": False}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-codex-validation.md").write_text("# Codex Validation\n", encoding="utf-8")
    (bundle / "term-arbitration-codex-validation.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_validation.v1",
                "status": "ready_for_import",
                "ok": True,
                "accepted_decision_count": 1,
                "rejected_decision_count": 1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = export_knowledge_note(bundle)

    note = Path(result["note_path"]).read_text(encoding="utf-8")
    transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    audit = Path(result["extraction_audit_path"]).read_text(encoding="utf-8")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    candidate_pack = json.loads((bundle / "exports" / "content-candidate-pack.json").read_text(encoding="utf-8"))
    material_card = json.loads((bundle / "exports" / "content-material-card.json").read_text(encoding="utf-8"))
    content_status = content_asset_status(bundle)
    assert note.startswith("- 摘要\n  - 📑 智能总结")
    assert "  - 📑 智能总结" in note
    assert "    - 录音信息" in note
    assert "    - 录音总结" in note
    assert "- 逐字稿" in note
    assert "第一段讲口头知识" in note
    assert "动作：展示规则" in note
    assert "事件：滚动页面" in note
    assert "表格需要保留截图核对" in note
    assert "#### 信息来源" not in note
    assert "## 术语仲裁" not in note
    assert "Canonical transcript SHA-256:" not in note
    assert "术语与工具名纠错闭环" in audit
    assert "Codex语义预检" in audit
    assert "term-arbitration-codex-validation.md" in audit
    assert "{'objects'" not in note
    assert "{'event_sequence'" not in note
    assert "术语与工具名纠错闭环" in audit
    assert "Codex语义预检" in audit
    assert "term-arbitration-codex-validation.md" in audit
    assert summary["term_correction"]["status"] == "needs_transcript_arbitration"
    assert summary["term_correction"]["accepted_term_count"] == 1
    assert summary["term_correction"]["term_validation_status"] == "ready_for_import"
    assert summary["term_correction"]["accepted_validation_decisions"] == 1
    assert summary["term_correction"]["rejected_validation_decisions"] == 1
    assert material_card["term_correction"]["term_validation_status"] == "ready_for_import"
    assert material_card["term_correction"]["accepted_validation_decisions"] == 1
    assert candidate_pack["term_correction"]["term_validation_status"] == "ready_for_import"
    assert candidate_pack["term_correction"]["accepted_validation_decisions"] == 1
    assert candidate_pack["candidates"][0]["term_validation_status"] == "ready_for_import"
    assert content_status["term_validation_status"] == "ready_for_import"
    assert content_status["accepted_validation_decisions"] == 1
    assert manifest["term_correction_status"]["status"] == "needs_transcript_arbitration"
    assert "{'objects'" not in note
    assert "{'event_sequence'" not in note
    assert Path(result["extraction_audit_path"]).name == "extraction-audit.md"
    assert manifest["knowledge_note_extraction_audit_markdown"].endswith("extraction-audit.md")
    assert manifest["content_assets"]["review_required"] is True
    assert manifest["content_assets"]["publication_allowed"] is False
    assert manifest["content_assets"]["material_card_contract"]["field_mapping"]["source_fact_status"] == "ai_extracted_needs_review"
    assert manifest["content_assets"]["material_card_contract"]["allowed_as_inspiration"] is True
    assert manifest["content_assets"]["material_card_contract"]["allowed_as_fact"] is False
    assert manifest["content_assets"]["consumer_rules"]["circle_of_friends"]["allowed_status"] == "needs_review_inspiration"
    assert "fact_check_before_claiming_truth" in manifest["content_assets"]["human_confirmation_required"]
    assert "publish_or_send_to_any_external_surface" in manifest["content_assets"]["human_confirmation_required"]
    assert summary["content_assets"]["key_segments_path"].endswith("key-segments.md")
    assert Path(summary["content_assets"]["short_video_script_drafts_path"]).exists()
    assert Path(summary["content_assets"]["highlight_post_drafts_path"]).exists()
    assert "提取审计" in audit
    assert "## 1. 总览" in audit
    assert "最终验收状态" in audit
    assert "## 4. 缺口索引" in audit
    assert "## 5. 人工审核状态" in audit
    assert "逐片段审计表" in audit
    assert "| Index | 时间段 | 路由 | 转写 | 图文 | 单帧理解 | 连续理解 | 人审 | 缺口/风险 | 证据 |" in audit
    assert "Knowledge note" in audit
    assert "Review template" in audit
    assert "画面未可靠提取" in transcript
    assert Path(result["full_transcript_path"]).exists()






def test_smart_summary_chapter_pack_builds_course_map_and_visual_notes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smart-summary-chapters"
    bundle.mkdir()
    frame = bundle / "frames" / "frame-020.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"fake")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Chapter Course", "media_path": "D:/media/chapter.mp4"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 8, "transcript": "开头讲客户特点和信任入口。"},
                {"index": 2, "start": 1200, "end": 1210, "transcript": "中段讲成交原则和问题链。", "visual_route": "document_visual", "visual_text": "PPT：成交基本原则", "structured_visual": {"markdown": "# 成交基本原则\n- 信任\n- 需求确认"}, "frame_path": str(frame)},
                {"index": 3, "start": 2400, "end": 2410, "transcript": "后段讲跟进动作和复盘记录。"},
                {"index": 4, "start": 3590, "end": 3600, "transcript": "结尾总结客户沟通要形成清单。"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 8, "text": "开头讲客户特点、客户顾虑和信任入口，需要先理解客户处境。"},
                    {"start": 1200, "end": 1210, "text": "中段讲成交基本原则和问题链，强调要用连续问题确认真实需求。"},
                    {"start": 2400, "end": 2410, "text": "后段讲跟进动作和复盘记录，需要把已确认和待确认的信息分开。"},
                    {"start": 3590, "end": 3600, "text": "结尾总结客户沟通要形成清单，避免后续服务依赖临场记忆。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_smart_summary_chapter_pack(bundle, title="Chapter Course", target_chapters=4)

    assert result["chapter_count"] == 4
    assert result["course_map"]["topics"]
    assert any(chapter["visual_notes"] for chapter in result["chapters"])
    assert (bundle / "exports" / "smart-summary-chapters.md").exists()
    assert (bundle / "exports" / "course-map.md").exists()

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["smart_summary_chapters_markdown"] == "exports/smart-summary-chapters.md"
    assert manifest["smart_summary_course_map_markdown"] == "exports/course-map.md"

def test_generate_smart_summary_with_codex_requires_llm_output_and_does_not_generate_rule_summary(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smart-summary-auto"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Auto Summary Video", "media_path": "D:/media/auto.mp4"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 5, "transcript": "开头讲客户画像和信任建立。", "quality_issues": ["semantic_frame_without_analysis"]},
                {"index": 2, "start": 1200, "end": 1210, "transcript": "中段讲问题链和需求确认。"},
                {"index": 3, "start": 2400, "end": 2410, "transcript": "后段讲跟进动作和复盘记录。"},
                {"index": 4, "start": 3590, "end": 3600, "transcript": "结尾总结客户沟通要沉淀成清单。"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 5, "text": "开头讲客户画像、信任建立和成交基本原则，需要先理解客户处境。"},
                    {"start": 1200, "end": 1210, "text": "中段讲问题链和需求确认，强调要用连续问题降低信息不对称。"},
                    {"start": 2400, "end": 2410, "text": "后段讲跟进动作和复盘记录，需要把已确认、待确认和下一步动作分开写清楚。"},
                    {"start": 3590, "end": 3600, "text": "结尾总结客户沟通要沉淀成清单，避免后续服务依赖临场记忆。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-pack.json").write_text(
        json.dumps({"status": "no_candidates", "candidate_count": 0, "candidates": [], "candidate_groups": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    export_result = export_knowledge_note(bundle, title="Auto Summary Video")
    assert export_result["smart_summary_codex"]["status"] == "needs_llm_rewrite"
    assert export_result["smart_summary_codex"].get("generated") is True
    result = generate_smart_summary_with_codex(bundle)

    assert result["status"] == "needs_llm_rewrite"
    assert result["quality"]["passed"] is False
    assert result["model_strategy"] == "codex_first_llm_layer"
    codex_summary = bundle / "exports" / "smart-summary.codex.md"
    assert not codex_summary.exists()
    assert result["installed_from"] == "llm_output_required"
    assert result["chapter_pack_refresh"]
    assert (bundle / "exports" / "smart-summary-input-pack.json").exists()
    assert (bundle / "exports" / "long-video-memory-pack.json").exists()
    assert (bundle / "exports" / "smart-summary-chapters.json").exists()
    assert (bundle / "exports" / "course-map.md").exists()
    run = json.loads((bundle / "runs" / "smart-summary-codex" / "run.json").read_text(encoding="utf-8"))
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert result["run_artifact"]["run_type"] == "smart_summary_codex"
    assert run["status"] == "needs_execution"
    assert run["failed_items"]
    assert any(artifact["key"] == "smart_summary_codex" for artifact in run["artifacts"])
    registry_status_by_type = {row["run_type"]: row["status"] for row in registry["runs"]}
    assert registry_status_by_type["smart_summary_codex"] == "needs_execution"

    final = export_knowledge_note(bundle, title="Auto Summary Video")
    final_text = Path(final["smart_summary_path"]).read_text(encoding="utf-8")
    assert "needs_llm_summary" in final_text
    assert "规则拼接的总结正文" in final_text
    assert "codex_assisted_draft" not in final_text
    assert final["smart_summary_final_status"] == "needs_llm_summary"

def test_codex_smart_summary_quality_gate_and_final_export(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smart-summary-codex"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Smart Final Video"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 5, "transcript": "开头讲客户画像和信任。", "quality_issues": ["semantic_frame_without_analysis"]},
                {"index": 2, "start": 1800, "end": 1810, "transcript": "中段讲问题链。"},
                {"index": 3, "start": 3590, "end": 3600, "transcript": "结尾讲复盘清单。"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 5, "text": "开头讲客户画像、客户顾虑和建立信任。"},
                    {"start": 1800, "end": 1810, "text": "中段展开陌客沟通的问题链和需求确认。"},
                    {"start": 3590, "end": 3600, "text": "最后要求沉淀跟进动作和复盘清单。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    draft_result = export_knowledge_note(bundle, title="Smart Final Video")
    draft_quality = draft_result["smart_summary_quality"]
    assert draft_quality["passed"] is False
    assert draft_quality["is_codex_summary"] is False
    assert draft_result["smart_summary_final_status"] == "needs_llm_summary"
    assert draft_result["smart_summary_codex"]["status"] == "needs_llm_rewrite"
    (bundle / "transcript-semantic-correction-pack.json").write_text(
        json.dumps({"status": "no_candidates", "candidate_count": 0, "candidates": [], "candidate_groups": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    postprocess_asr_transcript(bundle, target_seconds=60, max_chars=200)
    (bundle / "exports" / "human-key-points.json").write_text(
        json.dumps(
            {"key_points": ["\u5ba2\u6237\u753b\u50cf\u3001\u95ee\u9898\u94fe\u548c\u590d\u76d8\u6e05\u5355"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    codex_md = tmp_path / "codex-summary.md"
    codex_md.write_text(
        "\n".join(
            [
                "# Smart Final Video - 智能总结",
                "",
                "生成方式：`codex_final`。",
                "",
                "## 基本信息",
                "",
                "- 视频名：Smart Final Video",
                "- 覆盖范围：`00:00:00.000 - 01:00:00.000`",
                "",
                "## 一句话概览",
                "",
                "这节课围绕从客户画像到陌客问题链再到复盘清单的完整沟通流程，讲清如何先建立信任、再确认需求、最后形成可执行跟进动作。",
                "",
                "## 核心主题 / 课程主线",
                "",
                "课程主线是把陌生客户沟通拆成三层：先识别客户顾虑和信任门槛，再用问题链完成需求确认，最后把沟通结果沉淀成可复盘的下一步动作。",
                "",
                "## 分段总结",
                "",
                "### `00:00:00.000 - 00:05:00.000` 客户画像和信任入口",
                "开头说明客户并不是先等方案，而是先判断沟通对象是否可靠，因此前几轮交流要少推销、多理解客户处境。",
                "",
                "### `00:30:00.000 - 00:35:00.000` 问题链和需求确认",
                "中段把陌客沟通推进到问题链：围绕家庭、收入、风险顾虑、既有保障和预算边界提问，用连续问题代替单点介绍。",
                "",
                "### `00:59:00.000 - 01:00:00.000` 复盘和跟进清单",
                "结尾强调沟通结束后要沉淀客户问题、未确认信息和下一步跟进动作，让后续服务不依赖临场记忆。",
                "",
                "## 关键观点 / 方法论",
                "",
                "- `00:00:02.000` 陌客成交的起点是信任判断，不是产品解释。",
                "- `00:30:05.000` 问题链的价值是连续降低信息不对称，而不是机械提问。",
                "- `00:59:30.000` 复盘清单把沟通经验变成可复制流程。",
                "",
                "## 可执行动作清单",
                "",
                "- `00:00:03.000` 先记录客户画像、顾虑和当前关系温度。",
                "- `00:30:08.000` 准备一组从轻到重的问题链，避免一上来问隐私或预算。",
                "- `00:59:40.000` 每次沟通后写下已确认、待确认、下一次动作三栏。",
                "",
                "## 高频话术 / 可复用表达",
                "",
                "- `00:00:04.000` “我先了解一下你的情况，不急着给方案。”",
                "- `00:30:10.000` “这个问题我问得细一点，是为了避免方案做偏。”",
                "- `00:59:45.000` “我把今天确认的信息整理一下，下次我们只补缺口。”",
                "",
                "## 待复核点 / 低置信内容",
                "",
                "- 视觉证据未执行/待复核：本总结主要依据完整 ASR，屏幕文字、课件页和板书细节没有被当作已确认事实。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    install_result = generate_smart_summary_with_codex(bundle, input_md=codex_md)
    assert install_result["quality"]["passed"] is True
    run = json.loads((bundle / "runs" / "smart-summary-codex" / "run.json").read_text(encoding="utf-8"))
    assert install_result["run_artifact"]["status"] == "completed"
    assert run["status"] == "completed"
    assert "generate-smart-summary-with-codex" in run["retry_command"]
    final_result = export_knowledge_note(bundle, title="Smart Final Video", run_transcript_evidence_check=False)
    final = Path(final_result["smart_summary_path"]).read_text(encoding="utf-8")
    final_quality = smart_summary_quality_check(bundle, require_codex=True)
    assert "生成方式：`codex_final`" in final
    assert "codex_assisted_draft" not in final
    assert final_result["smart_summary_quality"]["passed"] is True
    assert final_quality["passed"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["smart_summary_quality"]["passed"] is True




def test_smart_summary_quality_fails_when_term_impact_has_final_aliases(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-term-impact-gate"
    exports = bundle / "exports"
    exports.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Term Gate"}, ensure_ascii=False), encoding="utf-8")
    (exports / "human-key-points.json").write_text(
        json.dumps(
            {"key_points": ["Playwright\u3001Browser MCP \u4e0e\u9009\u62e9\u6807\u51c6"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(json.dumps([{"index": 1, "start": 0, "end": 100, "transcript": "课程讲 Playwright 和 Browser MCP。"}], ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 100, "text": "课程讲 Playwright 和 Browser MCP。"}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "corrected-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 100, "text": "课程讲 Playwright 和 Browser MCP。"}]}, ensure_ascii=False), encoding="utf-8")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["corrected_transcript_json"] = "corrected-transcript.json"
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    _write_semantic_asset_gate_passed(bundle)
    summary_text = "\n".join([
        "# Term Gate - 智能总结",
        "",
        "生成方式：`codex_final`。",
        "",
        "## 基本信息",
        "",
        "- 视频名：Term Gate",
        "- 覆盖范围：`00:00:00.000 - 00:01:40.000`",
        "",
        "## 一句话概览",
        "",
        "这节课围绕浏览器自动化工具的选择标准展开，说明如何比较 Playwright、Browser MCP 和类似工具的适用边界。",
        "",
        "## 核心主题 / 课程主线",
        "",
        "课程主线是从工具名称识别、能力边界和使用风险三个角度建立自动化工具判断框架。",
        "",
        "## 分段总结",
        "",
        "### `00:00:00.000 - 00:00:40.000` 工具名称与基本定位",
        "开头说明 Playwright 和 Browser MCP 都属于浏览器自动化相关工具，但适用方式不同。",
        "",
        "### `00:00:40.000 - 00:01:40.000` 使用风险和选择标准",
        "后段提醒要结合封控、稳定性和 token 成本判断工具选择。",
        "",
        "## 关键观点 / 方法论",
        "",
        "- `00:00:05.000` 工具名必须先纠正，否则后续比较会偏。",
        "- `00:01:20.000` 自动化工具选择要结合稳定性和成本。",
        "",
        "## 可执行动作清单",
        "",
        "- `00:00:08.000` 先建立工具名词典并校正 ASR。",
        "- `00:01:25.000` 再检查最终导出是否仍残留错词。",
        "",
        "## 高频话术 / 可复用表达",
        "",
        "- `00:00:10.000` “先把工具名对齐，再谈能力差异。”",
        "- `00:01:30.000` “最终文件里不能残留错误工具名。”",
        "",
        "## 待复核点 / 低置信内容",
        "",
        "- 视觉证据未执行/待复核：本总结主要依据转写与术语纠错报告。",
        "",
    ])
    (exports / "smart-summary.md").write_text(summary_text, encoding="utf-8")
    (bundle / "term-correction-impact-report.json").write_text(json.dumps({"schema": "video_knowledge_pipeline.term_correction_impact.v1", "status": "needs_fix", "ok": False, "replacement_count": 2, "source_alias_total": 4, "output_alias_total": 2, "final_export_alias_total": 2}, ensure_ascii=False), encoding="utf-8")

    failed = smart_summary_quality_check(bundle, require_codex=True)

    assert failed["passed"] is False
    checks = {row["key"]: row for row in failed["checks"]}
    assert checks["term_correction_impact"]["passed"] is False
    assert failed["term_correction_impact_gate"]["final_export_alias_total"] == 2

    (bundle / "term-correction-impact-report.json").write_text(json.dumps({"schema": "video_knowledge_pipeline.term_correction_impact.v1", "status": "passed", "ok": True, "replacement_count": 2, "source_alias_total": 4, "output_alias_total": 0, "final_export_alias_total": 0, "final_clean_rate": 1.0}, ensure_ascii=False), encoding="utf-8")
    passed = smart_summary_quality_check(bundle, require_codex=True)

    assert passed["passed"] is True
    checks = {row["key"]: row for row in passed["checks"]}
    assert checks["term_correction_impact"]["passed"] is True
    assert passed["term_correction_impact_gate"]["status"] == "passed"



def test_smart_summary_quality_requires_impact_report_after_codex_term_arbitration(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-codex-term-impact-required"
    exports = bundle / "exports"
    exports.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Codex Term Arbitration"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([{"index": 1, "start": 0, "end": 100, "transcript": "课程讲 playright m c p 和 brow harness。"}], ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 100, "text": "课程讲 playright m c p 和 brow harness。"}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "term-arbitration-codex-result.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1",
                "source": "codex_reviewed_import",
                "decisions": [
                    {"candidate_id": "term-1", "canonical": "Playwright MCP", "aliases": ["playright m c p"], "confidence": 0.96, "action": "replace", "needs_human_review": False},
                    {"candidate_id": "term-2", "canonical": "BrowserHarness", "aliases": ["brow harness"], "confidence": 0.96, "action": "replace", "needs_human_review": False},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_text = "\n".join([
        "# Codex Term Arbitration - 智能总结",
        "",
        "生成方式：`codex_final`。",
        "",
        "## 基本信息",
        "",
        "- 视频名：Codex Term Arbitration",
        "- 覆盖范围：`00:00:00.000 - 00:01:40.000`",
        "",
        "## 一句话概览",
        "",
        "这节课围绕浏览器自动化工具名纠错与选择标准展开，强调先完成语义仲裁再生成最终总结。",
        "",
        "## 核心主题 / 课程主线",
        "",
        "课程主线是用 ASR、OCR 和上下文语义共同确认 Playwright MCP、BrowserHarness 等工具名。",
        "",
        "## 分段总结",
        "",
        "### `00:00:00.000 - 00:00:40.000` 工具名语义仲裁",
        "开头说明要把 playright m c p 和 brow harness 这类 ASR 错词纠正成真实工具名。",
        "",
        "### `00:00:40.000 - 00:01:40.000` 最终导出检查",
        "后段强调最终总结不能残留未纠正工具名，需要用影响报告复查。",
        "",
        "## 关键观点 / 方法论",
        "",
        "- `00:00:05.000` 工具名纠错要结合上下文语义，而不是只看拼写相似度。",
        "- `00:01:20.000` 高置信度仲裁结果要进入最终导出前的质量门。",
        "",
        "## 可执行动作清单",
        "",
        "- `00:00:08.000` 先运行 Codex 术语仲裁并导入结果。",
        "- `00:01:25.000` 再运行术语纠错影响报告确认最终文件干净。",
        "",
        "## 高频话术 / 可复用表达",
        "",
        "- `00:00:10.000` “这个工具名要先根据上下文确认。”",
        "- `00:01:30.000` “最终总结要检查错词是否还残留。”",
        "",
        "## 待复核点 / 低置信内容",
        "",
        "- 视觉证据未执行/待复核：本总结主要依据转写、OCR 和术语仲裁产物。",
        "",
    ])
    (exports / "smart-summary.md").write_text(summary_text, encoding="utf-8")

    result = smart_summary_quality_check(bundle, require_codex=True)

    assert result["passed"] is False
    checks = {row["key"]: row for row in result["checks"]}
    assert checks["term_correction_impact"]["passed"] is False
    assert result["term_correction_impact_gate"]["required"] is True
    assert result["term_correction_impact_gate"]["status"] == "missing_report"


def test_smart_summary_codex_prompt_includes_term_arbitration_gate(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smart-summary-prompt-term-gate"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Term Prompt Gate"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([{"index": 1, "start": 0, "end": 5, "transcript": "今天讲 browser base 和 playright m c p。"}], ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 5, "text": "今天讲 browser base 和 playright m c p。"}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "term-arbitration-codex-pack.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.term_arbitration_codex.v1", "status": "imported", "candidate_count": 2, "draft_decisions": [{"canonical": "Browserbase"}, {"canonical": "Playwright MCP"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-codex-result.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1",
                "decisions": [
                    {"canonical": "Browserbase", "aliases": ["browser base"], "confidence": 0.96, "action": "replace", "needs_human_review": False},
                    {"canonical": "Playwright MCP", "aliases": ["playright m c p"], "confidence": 0.95, "action": "replace", "needs_human_review": False},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-glossary.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.term_arbitration_glossary.v1", "terms": [{"canonical": "Browserbase", "aliases": ["browser base"], "confidence": 0.96}, {"canonical": "Playwright MCP", "aliases": ["playright m c p"], "confidence": 0.95}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-correction-impact-report.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.term_correction_impact.v1", "status": "passed", "ok": True, "replacement_count": 2, "source_alias_total": 2, "output_alias_total": 0, "final_export_alias_total": 0, "final_clean_rate": 1.0}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Term Prompt Gate")

    prompt = Path(result["smart_summary_prompt_path"]).read_text(encoding="utf-8")
    assert "Terminology / Tool Name Arbitration" in prompt
    assert "术语/工具名当成语义判断问题处理" in prompt
    assert "Codex term arbitration status: `imported`" in prompt
    assert "Imported decisions: `2`" in prompt
    assert "Accepted decisions: `2`" in prompt
    assert "Glossary terms: `2`" in prompt
    assert "Ready for transcript arbitration: `True`" in prompt
    assert "Term correction impact gate: `passed`" in prompt
    assert "Final export alias total: `0`" in prompt
    assert "只有已导入且通过影响门禁的高置信术语" in prompt

def test_smart_summary_input_pack_marks_codex_term_arbitration_draft_as_review_required(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-summary-input-pack-term-draft"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Term Draft"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([{"index": 1, "start": 0, "end": 5, "transcript": "今天讲 playright m c p。"}], ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 5, "text": "今天讲 playright m c p。"}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "term-arbitration-codex-pack.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.term_arbitration_codex.v1", "status": "draft_ready", "candidate_count": 1, "draft_decisions": [{"canonical": "Playwright MCP"}], "llm_semantic_arbitration": {"strategy": "codex_substitute_for_online_text_llm", "review_status": "codex_review_pending", "rule_draft_is_not_semantic_confirmation": True}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-codex-prompt.md").write_text("# Codex term prompt\n", encoding="utf-8")

    pack = build_smart_summary_input_pack(bundle, title="Term Draft")
    markdown = (bundle / "exports" / "smart-summary-input-pack.md").read_text(encoding="utf-8")

    assert pack["term_arbitration_codex"]["status"] == "draft_ready"
    assert pack["term_arbitration_codex"]["codex_review_required"] is True
    assert pack["term_arbitration_codex"]["ready_for_transcript_arbitration"] is False
    assert any("reviewed decisions are not imported" in note for note in pack["quality_notes"])
    assert "Codex review required: `True`" in markdown
    assert "term-arbitration-codex-result.json" in markdown

def test_smart_summary_input_pack_fuses_terms_punctuation_and_visual_evidence(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-summary-input-pack"
    bundle.mkdir()
    frame = bundle / "frames" / "frame-001.jpg"
    tile = bundle / "high-res-tiles" / "timeline-0001" / "tile-01.jpg"
    frame.parent.mkdir(parents=True)
    tile.parent.mkdir(parents=True)
    frame.write_bytes(b"fake")
    tile.write_bytes(b"tile")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Browser Tools", "transcript_source_arbitration_json": "transcript-source-arbitration.json"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 5,
                    "transcript": "browser base 是浏览器自动化工具然后它可以连接浏览器",
                    "visual_route": "document_visual",
                    "visual_text": "Browserbase 控制台",
                    "structured_visual": {"markdown": "| 工具 | 能力 |\n| Browserbase | 浏览器自动化 |"},
                    "visual_understanding": {"objects": ["Browserbase 控制台"], "non_text_info": "页面展示浏览器连接状态"},
                    "temporal_visual_understanding": {"event_sequence": ["打开控制台", "查看自动化任务状态"]},
                    "tile_result_merges": [
                        {
                            "tile_id": "0001-01",
                            "action": "merge",
                            "confidence": 0.91,
                            "evidence_path": str(tile),
                            "source": "tile_result_import",
                        }
                    ],
                    "missing_visual_text": True,
                    "review_reason": "screen_text_low_confidence",
                    "frame_path": str(frame),
                    "term_candidates": [
                        {
                            "canonical_term": "Browserbase",
                            "raw_mentions": ["browser base"],
                            "confidence": 0.94,
                            "needs_human_review": False,
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 5, "text": "browser base 是浏览器自动化工具然后它可以连接浏览器"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "transcript-source-arbitration.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "summary": {
                    "segments": 1,
                    "changed_segments": 1,
                    "review_segments": 0,
                    "quality_status": "changed_clean",
                    "average_confidence": 0.93,
                    "high_confidence_term_replacements": 1,
                    "low_confidence_conflicts": 0,
                },
                "quality_summary": {
                    "status": "changed_clean",
                    "total_segments": 1,
                    "changed_segments": 1,
                    "review_segments": 0,
                    "average_confidence": 0.93,
                    "high_confidence_term_replacements": 1,
                    "low_confidence_conflicts": 0,
                    "can_use_as_summary_input": True,
                    "smart_summary_guidance": ["Corrected transcript is preferred for smart-summary."],
                },
                "review_rows": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (bundle / "term-arbitration-codex-pack.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.term_arbitration_codex.v1", "status": "imported", "candidate_count": 1, "draft_decisions": [{"canonical": "Browserbase"}], "llm_semantic_arbitration": {"strategy": "codex_substitute_for_online_text_llm", "review_status": "codex_or_llm_reviewed_import", "rule_draft_is_not_semantic_confirmation": True}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-codex-result.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1", "decisions": [{"canonical": "Browserbase", "aliases": ["browser base"], "confidence": 0.96, "action": "replace", "needs_human_review": False}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-glossary.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.term_arbitration_glossary.v1", "terms": [{"canonical": "Browserbase", "aliases": ["browser base", "Browserbase"], "confidence": 0.96, "review_required": False}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-codex-prompt.md").write_text("# Codex term prompt\n", encoding="utf-8")

    pack = build_smart_summary_input_pack(bundle, title="Browser Tools")

    markdown = (bundle / "exports" / "smart-summary-input-pack.md").read_text(encoding="utf-8")
    assert pack["term_summary"]["timeline_replacements"][0]["canonical"] == "Browserbase"
    assert pack["term_arbitration_codex"]["status"] == "imported"
    assert pack["term_arbitration_codex"]["imported_decision_count"] == 1
    assert pack["term_arbitration_codex"]["glossary_term_count"] == 1
    assert pack["term_arbitration_codex"]["ready_for_transcript_arbitration"] is True
    assert pack["term_arbitration_codex"]["semantic_review_status"] == "codex_or_llm_reviewed_import"
    assert any("Codex term arbitration has imported glossary terms" in note for note in pack["quality_notes"])
    assert pack["transcript_arbitration"]["exists"] is True
    assert pack["transcript_arbitration"]["quality_summary"]["status"] == "changed_clean"
    assert pack["transcript_arbitration"]["quality_summary"]["high_confidence_term_replacements"] == 1
    assert "Codex 术语/工具名语义仲裁" in markdown
    assert "Ready for transcript arbitration: `True`" in markdown
    assert "Semantic review status: `codex_or_llm_reviewed_import`" in markdown
    assert "term-arbitration-codex-prompt.md" in markdown
    assert "字幕/ASR 仲裁质量" in markdown
    assert "High-confidence term replacements" in markdown
    assert "Browserbase 是浏览器自动化工具" in markdown
    assert "Punctuated" in markdown
    assert "视觉/课件证据摘要" in markdown
    assert "Browserbase 控制台" in markdown
    assert str(frame) in markdown
    assert pack["transcript_segments"][0]["evidence_inputs"]["has_ocr_or_ebook"] is True
    assert pack["transcript_segments"][0]["evidence_inputs"]["has_visual_understanding"] is True
    assert pack["transcript_segments"][0]["evidence_inputs"]["has_temporal_understanding"] is True
    trace = pack["evidence_trace"]
    assert trace["summary"]["ocr_or_ebook_items"] == 2
    assert {tuple(row['source_locators'][0]['selector']) for row in trace['ocr_items']} == {
        (0, 'visual_text'), (0, 'structured_visual')}
    assert any('Browserbase' in row['excerpt'] and '|' in row['excerpt']
               for row in trace['ocr_items'] if row['kind'] == 'structured_visual')
    assert trace["summary"]["high_res_tile_items"] == 1
    assert trace["tile_items"][0]["kind"] == "high_res_tile"
    assert str(tile) in trace["tile_items"][0]["evidence_paths"]
    assert trace["summary"]["visual_understanding_items"] == 1
    assert trace["summary"]["temporal_understanding_items"] == 1
    assert trace["summary"]["review_gaps"] == 1
    assert trace["summary"]["moment_chunks"] >= 1
    assert "证据追踪" in markdown
    assert "Moment evidence" in markdown
    assert "Review gaps" in markdown
    assert pack["run_registry"]["run_type"] == "smart_summary_input_pack"
    assert pack["run_registry"]["status"] == "needs_review"

    chapters = build_smart_summary_chapter_pack(bundle, title="Browser Tools", target_chapters=1)
    chapter = chapters["chapters"][0]
    chapter_trace = chapter["evidence_trace"]
    assert chapter["citation_digest"]
    citation_types = {row["source_type"] for row in chapter["citation_digest"]}
    assert {"transcript", "moment", "ocr_or_ebook", "high_res_tile", "visual_understanding", "temporal_understanding", "review_gap"} <= citation_types
    assert chapter_trace["summary"]["citation_digest_items"] == len(chapter["citation_digest"])
    assert chapter_trace["summary"]["ocr_or_ebook_items"] == 2
    assert {tuple(row['source_locators'][0]['selector']) for row in chapter_trace['ocr_items']} == {
        (0, 'visual_text'), (0, 'structured_visual')}
    assert chapter_trace["summary"]["high_res_tile_items"] == 1
    assert chapter_trace["summary"]["visual_understanding_items"] == 1
    assert chapter_trace["summary"]["temporal_understanding_items"] == 1
    assert chapter_trace["summary"]["review_gaps"] == 1
    chapter_markdown = (bundle / "exports" / "smart-summary-chapters.md").read_text(encoding="utf-8")
    assert "Evidence: transcript=`1`, OCR/ebook=`2`, high-res tile=`1`, visual=`1`, temporal=`1`" in chapter_markdown
    assert "Citation Digest" in chapter_markdown
    assert "ocr_or_ebook" in chapter_markdown
    assert "review_gap" in chapter_markdown
    assert "Browserbase 控制台" in chapter_markdown
    assert chapters["run_registry"]["run_type"] == "smart_summary_chapter_pack"
    assert chapters["run_registry"]["status"] == "needs_review"
    run_registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    run_types = {row["run_type"]: row["status"] for row in run_registry["runs"]}
    assert run_types["smart_summary_input_pack"] == "needs_review"
    assert run_types["smart_summary_chapter_pack"] == "needs_review"

    result = export_knowledge_note(bundle, title="Browser Tools")
    prompt = Path(result["smart_summary_prompt_path"]).read_text(encoding="utf-8")
    assert "smart-summary-input-pack.md" in prompt
    assert "long-video-memory-pack.md" in prompt
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    candidate_pack = json.loads((bundle / "exports" / "content-candidate-pack.json").read_text(encoding="utf-8"))
    candidate_citation_types = {row["source_type"] for row in candidate_pack["candidates"][0]["evidence_citations"]}
    assert "high_res_tile" in candidate_citation_types
    assert str(tile) in candidate_pack["candidates"][0]["evidence_paths"]
    assert summary["smart_summary_input_pack_path"].endswith("smart-summary-input-pack.md")
    assert summary["long_video_memory_pack_path"].endswith("long-video-memory-pack.md")
    assert summary["smart_summary_input_pack"]["visual_digest"]["total_items_with_visual_digest"] == 1


def test_export_and_input_pack_prefer_source_arbitrated_transcript(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-source-arbitrated-priority"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Source Arbitrated Priority",
                "normalized_transcript_json": "normalized-transcript.json",
                "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
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
                    "end": 8,
                    "transcript": "Today we compare playright m c p and brow harness.",
                    "visual_route": "semantic_frame",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 8, "text": "Today we compare playright m c p and brow harness."}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "source-arbitrated-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 8, "text": "Today we compare Playwright MCP and BrowserHarness."}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-arbitration-glossary.json").write_text(
        json.dumps(
            {
                "terms": [
                    {"canonical": "Playwright MCP", "aliases": ["playright m c p"], "confidence": 0.96, "review_required": False},
                    {"canonical": "BrowserHarness", "aliases": ["brow harness"], "confidence": 0.96, "review_required": False},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "term-correction-impact-report.json").write_text(
        json.dumps({"ok": True, "status": "passed", "final_export_alias_total": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "term-correction-closure.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")

    pack = build_smart_summary_input_pack(bundle, title="Source Arbitrated Priority")
    result = export_knowledge_note(bundle, title="Source Arbitrated Priority")

    full_transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    full_body = Path(result["full_body_path"]).read_text(encoding="utf-8")
    smart_summary = Path(result["smart_summary_path"]).read_text(encoding="utf-8")
    export_summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

    assert pack["transcript_source"].endswith("source-arbitrated-transcript.json")
    assert pack["transcript_segments"][0]["raw_text"] == "Today we compare Playwright MCP and BrowserHarness."
    assert "Playwright MCP" in full_transcript
    assert "BrowserHarness" in full_transcript
    assert "playright m c p" not in full_transcript
    assert "brow harness" not in full_transcript
    assert "Today we compare Playwright MCP and BrowserHarness." in full_body
    assert "playright m c p" not in full_body
    assert "brow harness" not in full_body
    assert "00:00:" not in full_body
    assert "needs_llm_summary" in smart_summary
    assert export_summary["term_correction"]["accepted_term_count"] == 2
    accepted = {row["canonical_term"] for row in export_summary["term_correction"]["accepted_terms"]}
    assert {"Playwright MCP", "BrowserHarness"} <= accepted



def test_export_uses_structured_source_arbitrated_segments_for_readable_outputs(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-structured-source-arbitrated"
    bundle.mkdir()
    raw_text = "step one analyze customer traits then build trust step two confirm needs step three handle objections"
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Structured Source Arbitrated",
                "normalized_transcript_json": "normalized-transcript.json",
                "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "start": 0, "end": 20, "transcript": raw_text, "visual_route": "semantic_frame"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 20, "text": raw_text}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "source-arbitrated-transcript.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
                "segments": [
                    {"start": 0, "end": 6, "text": "Step one: analyze customer traits, then build trust.", "source_segment_index": 0, "semantic_corrections": [{"application": "segment_split"}]},
                    {"start": 6, "end": 12, "text": "Step two: confirm needs.", "source_segment_index": 0, "semantic_corrections": [{"application": "segment_split"}]},
                    {"start": 12, "end": 20, "text": "Step three: handle objections.", "source_segment_indexes": [1, 2], "semantic_corrections": [{"application": "segment_merge"}]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Structured Source Arbitrated")

    full_transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    smart_summary = Path(result["smart_summary_path"]).read_text(encoding="utf-8")
    final_reading_note = Path(result["note_path"]).read_text(encoding="utf-8")
    assert "Source: `source_arbitrated_transcript`" not in full_transcript
    assert "Arbitration status: `arbitrated_or_reviewed`" not in full_transcript
    assert "来源状态：`source_arbitrated_transcript`" not in full_transcript
    assert "仲裁状态：`arbitrated_or_reviewed`" not in full_transcript
    assert "source_arbitrated_transcript" not in final_reading_note
    assert "arbitrated_or_reviewed" not in final_reading_note
    assert "### 00:00:00.000 - 00:00:06.000" in full_transcript
    assert "### 00:00:06.000 - 00:00:12.000" in full_transcript
    assert "### 00:00:12.000 - 00:00:20.000" in full_transcript
    assert "Step one: analyze customer traits, then build trust." in full_transcript
    assert "Step two: confirm needs." in full_transcript
    assert "Step three: handle objections." in full_transcript
    assert raw_text not in full_transcript
    assert result["smart_summary_input_pack"]["transcript_source_label"] == "source_arbitrated_transcript"
    assert result["smart_summary_input_pack"]["transcript_segments"][0]["raw_text"].startswith("Step one: analyze customer traits")
    assert "Transcript quality gate:" not in full_transcript
    assert result["transcript_quality_gate"]["ok"] is True
    export_summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert export_summary["transcript_quality_gate"]["source_path"].endswith("source-arbitrated-transcript.json")
    assert export_summary["smart_summary_input_pack"]["transcript_quality_gate"]["exists"] is True


def test_knowledge_note_projects_canonical_transcript_without_mutating_timeline(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-canonical-note"
    bundle.mkdir()
    raw = "使用MIAAPP中的PDF方案，再用一个cell版本呈现。"
    corrected = "使用明亚APP中的PDF方案，再用一个Excel版本呈现。"
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Canonical note",
                "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "start": 0, "end": 10, "transcript": raw}], ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "source-arbitrated-transcript.json").write_text(
        json.dumps({"segments": [{"index": 1, "start": 0, "end": 10, "text": corrected}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Canonical note")
    note = Path(result["note_path"]).read_text(encoding="utf-8")
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert corrected in note
    assert "MIAAPP" not in note
    assert "cell" not in note
    assert "Canonical transcript SHA-256:" not in note
    assert result["canonical_transcript_integrity"]["passed"] is True
    assert timeline[0]["transcript"] == raw
    assert "corrected_transcript" not in timeline[0]



def test_smart_summary_input_pack_prefers_corrected_transcript_over_source_arbitrated(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-corrected-priority"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Corrected Priority",
                "normalized_transcript_json": "normalized-transcript.json",
                "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
                "corrected_transcript_json": "corrected-transcript.json",
                "corrected_transcript_source": "agent_readable_transcript_rewrite",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "start": 0, "end": 10, "transcript": "raw timeline text", "visual_route": "semantic_frame"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "raw normalized text"}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "source-arbitrated-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "source arbitrated older wording"}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "corrected-transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "Corrected final wording, with readable punctuation."}]}, ensure_ascii=False), encoding="utf-8")

    pack = build_smart_summary_input_pack(bundle, title="Corrected Priority")
    result = export_knowledge_note(bundle, title="Corrected Priority")

    full_transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    smart_summary = Path(result["smart_summary_path"]).read_text(encoding="utf-8")
    export_summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

    assert pack["transcript_source"].endswith("source-arbitrated-transcript.json")
    assert pack["transcript_source_label"] == "source_arbitrated_transcript"
    assert pack["transcript_source_decision"]["uses_corrected_transcript"] is True
    assert pack["transcript_segments"][0]["raw_text"] == "source arbitrated older wording"
    assert "Source: `source_arbitrated_transcript`" not in full_transcript
    assert "source arbitrated older wording" in full_transcript
    assert "Corrected final wording" not in full_transcript
    assert "needs_llm_summary" in smart_summary
    assert export_summary["transcript_quality_gate"]["source_path"].endswith("source-arbitrated-transcript.json")
    assert export_summary["smart_summary_input_pack"]["transcript_source"].endswith("source-arbitrated-transcript.json")
    assert export_summary["canonical_transcript_integrity"]["passed"] is True
    assert pack["transcript_source_sha256"] == export_summary["canonical_transcript_integrity"]["canonical_sha256"]



def test_export_and_summary_input_prefer_agent_readable_transcript_over_corrected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-agent-readable-priority"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Agent Readable Priority",
                "normalized_transcript_json": "normalized-transcript.json",
                "corrected_transcript_json": "corrected-transcript.json",
                "agent_readable_transcript_json": "agent-readable-transcript.json",
                "corrected_transcript_source": "agent_readable_transcript_rewrite",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [{"index": 1, "start": 0, "end": 10, "transcript": "raw timeline text", "visual_route": "semantic_frame"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 10, "text": "raw normalized text"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "corrected-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 10, "text": "corrected semantic text without much punctuation"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "agent-readable-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 10, "text": "Agent readable final wording, with clear punctuation. It should feed the final transcript."}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    pack = build_smart_summary_input_pack(bundle, title="Agent Readable Priority")
    result = export_knowledge_note(bundle, title="Agent Readable Priority")

    full_transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    smart_summary = Path(result["smart_summary_path"]).read_text(encoding="utf-8")
    export_summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

    assert pack["transcript_source"].endswith("agent-readable-transcript.json")
    assert pack["transcript_source_label"] == "agent_readable_transcript"
    assert pack["transcript_source_decision"]["uses_corrected_transcript"] is True
    assert pack["transcript_source_decision"]["uses_readable_transcript"] is True
    assert pack["transcript_segments"][0]["raw_text"].startswith("Agent readable final wording")
    assert "Source: `agent_readable_transcript`" not in full_transcript
    assert "Arbitration status: `readable_after_correction_or_postprocess`" not in full_transcript
    assert "Agent readable final wording" in full_transcript
    assert "corrected semantic text without much punctuation" not in full_transcript
    assert "needs_llm_summary" in smart_summary
    assert "corrected semantic text without much punctuation" not in smart_summary
    assert export_summary["transcript_quality_gate"]["source_path"].endswith("agent-readable-transcript.json")
    assert export_summary["smart_summary_input_pack"]["transcript_source"].endswith("agent-readable-transcript.json")


def test_semantic_asset_gate_keeps_low_confidence_review_nonblocking() -> None:
    gate = _semantic_asset_gate(
        {
            "status": "impact_passed",
            "review_required_count": 3,
            "final_residual_error_total": 0,
            "readable_impact_status": "passed",
            "readable_required_residual_total": 0,
            "summary_impact_status": "passed",
            "summary_residual_original_total": 0,
            "summary_absorption_rate": 1.0,
            "candidate_discovery_status": "imported",
        }
    )

    assert gate["passed"] is True
    assert gate["status"] == "passed_with_open_review"
    assert gate["review_required_nonblocking"] is True
    assert gate["review_required_count"] == 3


def test_semantic_asset_gate_does_not_accept_no_candidates_before_discovery() -> None:
    gate = _semantic_asset_gate(
        {
            "status": "no_candidates",
            "candidate_discovery_status": "not_planned",
            "candidate_discovery_next_action": "run_candidate_discovery",
        }
    )

    assert gate["passed"] is False
    assert gate["status"] == "needs_candidate_discovery"
    assert gate["candidate_discovery_complete"] is False
    assert gate["next_action_key"] == "run_candidate_discovery"
def test_export_marks_smart_summary_draft_when_semantic_pack_missing_but_transcript_exists(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-missing-semantic-pack"
    bundle.mkdir(parents=True)
    (bundle / "exports").mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Missing Semantic Pack", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 5, "text": "today we discuss browser automation"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 0, "start": 0, "end": 5, "transcript": "today we discuss browser automation"}], ensure_ascii=False),
        encoding="utf-8",
    )

    result = export_knowledge_note(bundle, title="Missing Semantic Pack", run_transcript_evidence_check=False)

    assert result["smart_summary_final_status"] == "needs_llm_summary"
    gate = result["smart_summary_quality"]["transcript_semantic_correction_gate"]
    assert gate["passed"] is False
    assert gate["status"] == "missing_pack"
    assert "may be exported as a draft" in gate["detail"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["smart_summary_final_status"] == "needs_llm_summary"
    assert manifest["smart_summary_publication_boundary"]["publication_allowed"] is False
def test_export_marks_smart_summary_draft_when_semantic_correction_missing(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "exports").mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Semantic Draft", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 5, "text": "今天讲 play right m c p"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 0, "start": 0, "end": 5, "transcript": "今天讲 play right m c p", "visual_text": "Playwright MCP"}], ensure_ascii=False),
        encoding="utf-8",
    )
    pack = build_transcript_semantic_correction_pack(bundle, write=True)
    assert pack["candidate_count"] >= 1

    result = export_knowledge_note(bundle, title="Semantic Draft")

    assert result["smart_summary_final_status"] == "needs_llm_summary"
    assert result["smart_summary_publication_boundary"]["review_required"] is True
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["smart_summary_final_status"] == "needs_llm_summary"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["smart_summary_final_status"] == "needs_llm_summary"
    assert manifest["knowledge_note_export"]["smart_summary_publication_boundary"]["publication_allowed"] is False
    asset_status = content_asset_status(bundle, write=False)
    assert asset_status["semantic_correction_candidate_type_counts"]
    assert asset_status["semantic_correction_evidence_source_counts"]["ocr"] >= 1
    assert asset_status["semantic_correction_next_action_key"] in {"validate_result", "review_candidates", "run_llm_draft_preview"}
    assert asset_status["status"] == "semantic_correction_needs_action"
    material_card = json.loads(Path(result["content_material_card_path"]).read_text(encoding="utf-8"))
    candidate_pack = json.loads(Path(result["content_candidate_pack_path"]).read_text(encoding="utf-8"))
    assert material_card["allowed_as_inspiration"] is False
    assert material_card["circle_of_friends_status"] == "semantic_correction_required"
    assert candidate_pack["allowed_as_inspiration"] is False
    assert asset_status["semantic_blocked_material_card_flags"] is True
    assert asset_status["semantic_blocked_content_candidate_pack_flags"] is True
    assert material_card["transcript_semantic_correction"]["status"] == asset_status["semantic_correction_status"]
    assert material_card["transcript_semantic_correction"]["asset_gate"]["status"] == asset_status["semantic_correction_asset_gate"]["status"]
    assert candidate_pack["transcript_semantic_correction"]["status"] == asset_status["semantic_correction_status"]
    assert candidate_pack["transcript_semantic_correction"]["asset_gate"]["status"] == asset_status["semantic_correction_asset_gate"]["status"]
    assert candidate_pack["candidates"][0]["semantic_correction_status"] == asset_status["semantic_correction_status"]
    assert "Transcript semantic correction gate" in Path(result["content_material_card_markdown_path"]).read_text(encoding="utf-8")
    assert "Transcript semantic correction gate" in Path(result["content_candidate_pack_markdown_path"]).read_text(encoding="utf-8")
    assert manifest["content_material_card"]["transcript_semantic_correction"]["status"] == asset_status["semantic_correction_status"]
    assert manifest["content_candidate_pack"]["transcript_semantic_correction"]["status"] == asset_status["semantic_correction_status"]
    assert manifest["transcript_semantic_correction_status"]["status"] == asset_status["semantic_correction_status"]


def test_smart_summary_input_pack_exposes_general_semantic_correction_status(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-semantic-input-pack"
    (bundle / "exports").mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Semantic Input Pack",
                "normalized_transcript_json": "normalized-transcript.json",
                "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
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
                    "end": 5,
                    "transcript": "这里这个很重要大家看一下",
                    "visual_text": "客户信任建立流程：确认需求，给出解决方案",
                    "visual_route": "document_visual",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 5, "text": "这里这个很重要大家看一下"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "source-arbitrated-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 5,
                        "text": "这里讲的是客户信任建立流程。",
                        "semantic_corrections": [{"candidate_id": "semcorr-0001", "application": "whole_segment_text"}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-pack.json").write_text(
        json.dumps(
            {
                "status": "pack_ready",
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "semcorr-0001",
                        "correction_type": "concept",
                        "risk_level": "medium",
                        "original_text": "这里这个很重要大家看一下",
                        "candidate_text": "客户信任建立流程",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-status.json").write_text(
        json.dumps({"semantic_attention_items": [{"candidate_id": "semcorr-0001", "correction_type": "concept", "priority_score": 120}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "accepted_decisions": [
                    {
                        "candidate_id": "semcorr-0001",
                        "original_text": "这里这个很重要大家看一下",
                        "corrected_text": "这里讲的是客户信任建立流程。",
                        "confidence": 0.94,
                    }
                ],
                "review_required": [],
                "rejected_decisions": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-closure.json").write_text(
        json.dumps({"status": "completed", "applied_correction_count": 1, "changed_segment_count": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-readable-impact-report.json").write_text(
        json.dumps({"status": "passed"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-summary-impact-report.json").write_text(
        json.dumps({"status": "passed", "summary_residual_original_total": 0, "summary_absorption_rate": 1.0}, ensure_ascii=False),
        encoding="utf-8",
    )

    pack = build_smart_summary_input_pack(bundle, title="Semantic Input Pack")
    semantic = pack["transcript_semantic_correction"]
    markdown = (bundle / "exports" / "smart-summary-input-pack.md").read_text(encoding="utf-8")

    assert semantic["final_status"] == "ready_for_summary_input"
    assert semantic["candidate_type_counts"]["concept"] == 1
    assert semantic["semantic_attention_count"] == 1
    assert semantic["accepted_decision_count"] == 1
    assert semantic["changed_segment_count"] == 1
    assert any("General ASR/subtitle semantic correction is ready" in note for note in pack["quality_notes"])
    assert "ASR/字幕通用语义纠错" in markdown
    assert "ready_for_summary_input" in markdown
    assert "客户信任建立流程" in pack["transcript_segments"][0]["raw_text"]

def test_generate_smart_summary_with_codex_surfaces_semantic_correction_review_boundary(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-semantic-summary-boundary"
    (bundle / "exports").mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Semantic Summary Boundary",
                "normalized_transcript_json": "normalized-transcript.json",
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
                    "end": 30,
                    "transcript": "这里这个很重要大家看一下 后面要确认需求并建立信任",
                    "visual_text": "客户信任建立流程：确认需求，给出解决方案",
                    "visual_route": "document_visual",
                },
                {
                    "index": 2,
                    "start": 1800,
                    "end": 1830,
                    "transcript": "最后总结要把客户沟通沉淀成复盘清单",
                    "visual_route": "semantic_frame",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 30, "text": "这里这个很重要大家看一下 后面要确认需求并建立信任"},
                    {"start": 1800, "end": 1830, "text": "最后总结要把客户沟通沉淀成复盘清单"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "term-correction-impact-report.json").write_text(
        json.dumps({"ok": True, "status": "passed", "passed": True, "final_export_alias_total": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-pack.json").write_text(
        json.dumps(
            {
                "status": "pack_ready",
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "semcorr-0001",
                        "correction_type": "concept",
                        "risk_level": "medium",
                        "original_text": "这里这个很重要大家看一下",
                        "candidate_text": "客户信任建立流程",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-status.json").write_text(
        json.dumps({"semantic_attention_items": [{"candidate_id": "semcorr-0001", "correction_type": "concept", "priority_score": 120}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = generate_smart_summary_with_codex(bundle)
    input_pack = json.loads((bundle / "exports" / "smart-summary-input-pack.json").read_text(encoding="utf-8"))

    assert result["status"] == "needs_llm_rewrite"
    assert result["installed_from"] == "llm_output_required"
    assert not (bundle / "exports" / "smart-summary.codex.md").exists()
    assert input_pack["transcript_semantic_correction"]["final_status"] == "needs_codex_or_llm_review"
    assert result["chapter_pack_refresh"]
