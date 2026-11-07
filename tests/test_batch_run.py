from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.batch_run import batch_video_knowledge_run


def test_batch_run_writes_acceptance_dashboard_with_bundle_quality(tmp_path: Path) -> None:
    workspace = tmp_path / "batch"
    bundle = workspace / "lesson-001" / "webui-bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "knowledge_note_export": {"exported_at": "2026-06-13T00:00:00"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "acceptance-check.json").write_text(
        json.dumps(
            {
                "status": "accepted_with_known_gaps",
                "summary": {"semantic_missing": 0, "temporal_missing": 1, "export_freshness": "fresh"},
                "review_lifecycle": {"open_review_target_count": 2},
                "next_action": {"key": "done"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "knowledge-coverage.json").write_text(
        json.dumps(
            {
                "channels": [
                    {"key": "screen_text", "status": "weak", "covered_count": 3, "blocker_count": 4},
                ],
                "semantic_frame_without_analysis": 0,
                "temporal_sequence_without_analysis": 1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "batch-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_batch.v1",
                "workspace": str(workspace),
                "items": [
                    {
                        "id": "lesson-001",
                        "bundle_dir": str(bundle),
                        "title": "Lesson 001",
                        "expected_content_type": "software-demo",
                        "priority": "P1",
                        "notes": "needs screen text check",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = batch_video_knowledge_run(manifest, resume=True)

    summary = result["acceptance_summary"]
    assert summary["total"] == 1
    assert summary["screen_text_weak_or_blocked"] == 1
    row = summary["items"][0]
    assert row["acceptance_status"] == "accepted_with_known_gaps"
    assert row["expected_content_type"] == "software-demo"
    assert row["priority"] == "P1"
    assert row["screen_text_status"] == "weak"
    assert row["temporal_missing"] == 1
    assert row["review_pending_count"] == 2
    assert row["export_freshness"] == "fresh"
    assert (workspace / "batch-acceptance-summary.json").exists()
    report = (workspace / "batch-acceptance-summary.md").read_text(encoding="utf-8")
    assert "Batch Acceptance Summary" in report
    assert "needs screen text check" in report

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



def test_batch_run_cli_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "batch-manifest.json"
    workspace = tmp_path / "batch-workspace"
    manifest.write_text(
        json.dumps({"schema": "video_knowledge_batch.v1", "workspace": str(workspace), "items": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["batch-run", str(manifest), "--resume", "--no-write"])
    code = cli_main(["batch-run", str(manifest), "--resume", "--no-write"])

    assert args.command == "batch-run"
    assert args.batch_manifest == str(manifest)
    assert args.resume is True
    assert code == 0


def test_batch_run_skips_completed_bundles_by_default(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.batch_run as batch_run

    workspace = tmp_path / "batch"
    bundle = workspace / "lesson-001" / "webui-bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "acceptance-check.json").write_text(
        json.dumps({"status": "accepted_with_known_gaps", "next_action": {"key": "done"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = tmp_path / "batch-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_batch.v1",
                "workspace": str(workspace),
                "items": [{"id": "lesson-001", "media_path": str(tmp_path / "lesson.mp4"), "title": "Lesson 001"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_run, "run_acceptance_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should skip")))
    monkeypatch.setattr(batch_run, "run_acceptance_bundle", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should skip")))
    monkeypatch.setattr("video_knowledge_pipeline.workflow_completion.workflow_completion_status",
                        lambda *a, **k: {"status": "completed", "completed": True})

    result = batch_video_knowledge_run(manifest)

    assert result["summary"]["completed_or_skipped"] == 1
    assert result["items"][0]["status"] == "skipped_completed"
    assert result["items"][0]["action"] == "skip"
    assert (workspace / "batch-run.json").exists()
    assert (workspace / "batch-run.md").exists()


def test_batch_run_resume_existing_bundle_calls_preview_safe_acceptance_bundle(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.batch_run as batch_run

    workspace = tmp_path / "batch"
    bundle = workspace / "lesson-002" / "webui-bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    manifest = tmp_path / "batch-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_batch.v1",
                "workspace": str(workspace),
                "items": [{"id": "lesson-002", "bundle_dir": str(bundle), "title": "Lesson 002"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_next(bundle_dir, refresh=True):
        return {"status": "coverage_blocked", "next_action": {"key": "semantic_frame_understanding", "status": "coverage_blocked"}}

    def fake_acceptance_bundle(bundle_dir, **kwargs):
        calls.append({"bundle_dir": str(bundle_dir), **kwargs})
        return {
            "bundle_dir": str(bundle),
            "report_path": str(workspace / "lesson-002" / "acceptance-report.md"),
            "json_path": str(workspace / "lesson-002" / "acceptance-run.json"),
            "summary": {"status": "preview"},
        }

    monkeypatch.setattr(batch_run, "bundle_next_action", fake_next)
    monkeypatch.setattr(batch_run, "run_acceptance_bundle", fake_acceptance_bundle)
    monkeypatch.setattr(batch_run, "run_acceptance_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse bundle")))

    result = batch_video_knowledge_run(manifest, resume=True, execute_vision=False)

    assert result["items"][0]["status"] == "resumed"
    assert result["items"][0]["action"] == "acceptance_bundle_run"
    assert calls and calls[0]["execute_vision"] is False
    assert calls[0]["execute_ebook_pipeline"] is False


def test_batch_run_force_reexport_refreshes_reports_without_acceptance_run(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.batch_run as batch_run

    workspace = tmp_path / "batch"
    bundle = workspace / "lesson-003" / "webui-bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    manifest = tmp_path / "batch-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_batch.v1",
                "workspace": str(workspace),
                "items": [{"id": "lesson-003", "bundle_dir": str(bundle), "title": "Lesson 003"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_run, "run_acceptance_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run acceptance")))
    monkeypatch.setattr(batch_run, "run_acceptance_bundle", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run acceptance bundle")))
    monkeypatch.setattr(batch_run, "audit_knowledge_coverage", lambda bundle_dir, write=True: {"coverage_markdown_path": str(bundle / "knowledge-coverage.md")})
    monkeypatch.setattr(batch_run, "export_knowledge_note", lambda bundle_dir, write=True: {"note_path": str(bundle / "exports" / "knowledge-note.md")})
    monkeypatch.setattr(batch_run, "bundle_status_report", lambda bundle_dir, refresh=True: {"report_path": str(bundle / "bundle-status.md")})
    monkeypatch.setattr(batch_run, "acceptance_check", lambda bundle_dir, refresh=True, write=True: {"report_path": str(bundle / "acceptance-check.md")})
    monkeypatch.setattr(batch_run, "bundle_next_action", lambda bundle_dir, refresh=True: {"status": "ready", "next_action": {"key": "export_knowledge_note"}})

    result = batch_video_knowledge_run(manifest, force_reexport=True)

    assert result["items"][0]["status"] == "reexported"
    assert result["items"][0]["action"] == "force_reexport"
    assert result["items"][0]["reports"]["knowledge_note"].endswith("knowledge-note.md")


