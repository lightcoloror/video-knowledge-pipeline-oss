from __future__ import annotations

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



# Moved from test_video_pipeline_smoke.py during Phase 10 split.

def test_acceptance_bundle_run_reuses_existing_bundle(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "run" / "webui-bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")

    import video_knowledge_pipeline.acceptance_run as acceptance_run

    monkeypatch.setattr(acceptance_run, "prepare_local_video_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not prepare video")))
    monkeypatch.setattr(acceptance_run, "run_video_frame_router", lambda bundle_dir: {"report_path": str(bundle / "router.md")})
    monkeypatch.setattr(acceptance_run, "run_visual_structure_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "visual-structure-report.md")})
    monkeypatch.setattr(acceptance_run, "run_multimodal_frame_analysis", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "multimodal-frame-analysis-report.md")})
    monkeypatch.setattr(acceptance_run, "run_temporal_frame_groups", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "temporal-frame-groups-report.md")})
    monkeypatch.setattr(acceptance_run, "run_temporal_visual_analysis", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "temporal-visual-analysis-report.md")})
    monkeypatch.setattr(acceptance_run, "vision_acceptance_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "vision-acceptance-plan.md")})
    monkeypatch.setattr(acceptance_run, "audit_knowledge_coverage", lambda bundle_dir: {"coverage_markdown_path": str(bundle / "knowledge-coverage.md")})
    monkeypatch.setattr(acceptance_run, "export_knowledge_note", lambda bundle_dir, **kwargs: {"note_path": str(bundle / "exports" / "knowledge-note.md")})
    monkeypatch.setattr(
        acceptance_run,
        "bundle_status_report",
        lambda bundle_dir: {
            "status": "machine_action_available",
            "report_markdown_path": str(bundle / "bundle-status.md"),
            "next_action": {
                "status": "coverage_blocked",
                "key": "semantic_frame_understanding",
                "label": "补齐多模态单帧理解",
                "mcp_tool": "run_multimodal_frame_analysis",
                "mcp_args_path": str(bundle / "mcp-multimodal-frame-analysis.args.json"),
                "human_required": False,
            },
        },
    )

    result = run_acceptance_bundle(bundle, output_dir=tmp_path / "acceptance", title="已有 bundle")

    assert result["summary"]["workflow_status"] == "ok"
    assert result["summary"]["status"] == "machine_action_available"
    assert result["summary"]["next_action"]["key"] == "semantic_frame_understanding"
    assert result["bundle_dir"] == str(bundle.resolve())
    assert result["steps"][0]["key"] == "existing_bundle"
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "已有 bundle" in report
    assert str(bundle.resolve()) in report
    assert "Bundle Next Action" in report
    assert "run_multimodal_frame_analysis" in report


def test_acceptance_check_keeps_export_fresh_when_refresh_rewrites_derived_coverage(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-acceptance-export-fresh"
    exports = bundle / "exports"
    exports.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "fresh export"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "transcript": "已经导出的内容",
                    "visual_route": "semantic_frame",
                    "visual_understanding": {"objects": ["screen"]},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    export_knowledge_note(bundle, title="fresh export")

    report = acceptance_check(bundle, refresh=True)

    assert report["summary"]["export_freshness"] == "fresh"
    assert report["note_quality"]["export_freshness"] == "fresh"
    assert report["note_quality"]["dependency_snapshot_validation"]["passed"] is True
    assert Path(report["note_quality"]["full_body_path"]).exists()
    assert Path(report["note_quality"]["extraction_audit_path"]).exists()


def test_acceptance_check_blocks_export_when_timeline_changes_without_mtime_signal(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-acceptance-content-addressed"
    exports = bundle / "exports"
    exports.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "stale export"}), encoding="utf-8")
    timeline = bundle / "timeline.json"
    timeline.write_text(json.dumps([{"index": 1, "start": 0, "end": 1, "transcript": "A"}]), encoding="utf-8")
    export_knowledge_note(bundle, title="stale export")
    original_mtime = timeline.stat().st_mtime
    timeline.write_text(json.dumps([{"index": 1, "start": 0, "end": 1, "transcript": "B"}]), encoding="utf-8")
    import os
    os.utime(timeline, (original_mtime, original_mtime))
    report = acceptance_check(bundle, refresh=False)
    assert report["note_quality"]["export_freshness"] == "stale"
    assert any(row["key"] == "input_changed" for row in report["note_quality"]["dependency_snapshot_validation"]["issues"])


