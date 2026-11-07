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


def test_controlled_execution_smoke_plans_executes_and_restores_with_fixture(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-controlled-smoke"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 1,
            "visual_route": "semantic_frame",
            "frame_paths": [str(frame)],
            "quality_issues": ["semantic_frame_without_analysis", "missing_visual_understanding"],
        }
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    plan = controlled_execution_smoke(bundle)

    assert plan["execute"] is False
    assert plan["selected_action"]["kind"] == "semantic"
    assert plan["preflight"]["ready_to_execute"] is True
    assert plan["run_summary"]["status"] == "not_run"
    assert (bundle / "controlled-execution-smoke.md").exists()
    assert (bundle / "mcp-controlled-execution-smoke.args.json").exists()
    assert audit_bundle_mcp_args(bundle)["status"] == "ok"

    executed = controlled_execution_smoke(bundle, execute=True, restore_after=True, index=1)

    assert executed["execute"] is True
    assert executed["run_summary"]["status"] == "ok"
    assert executed["run_summary"]["updated_count"] == 1
    assert executed["restore_summary"]["status"] == "ok"
    assert executed["restore_summary"]["applied_count"] == 1
    assert executed["controlled_execution_check"]["ready_for_real_vision_execution"] is True
    restored = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert "visual_understanding" not in restored[0]
    assert "visual_understanding_updated_at" not in restored[0]
    assert "semantic_frame_without_analysis" in restored[0]["quality_issues"]


def test_prepare_local_video_run_can_build_initial_bundle(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.local_video_run as local_run

    monkeypatch.setattr(local_run, "add_video", lambda *args, **kwargs: {"video_id": "video_fake", "segment_count": 1})
    monkeypatch.setattr(local_run, "build_lecture_package", lambda *args, **kwargs: {"package_path": str(tmp_path / "package.json")})
    monkeypatch.setattr(
        local_run,
        "export_webui_bundle",
        lambda *args, **kwargs: {
            "bundle_dir": str(tmp_path / "run" / "webui-bundle"),
            "review_html_path": str(tmp_path / "run" / "webui-bundle" / "review.html"),
            "manifest_path": str(tmp_path / "run" / "webui-bundle" / "manifest.json"),
        },
    )

    result = prepare_local_video_run(media, tmp_path / "run", title="课程测试", build_initial_bundle=True)

    assert result["initial_bundle"]["status"] == "ok"
    assert result["initial_bundle"]["review_html"].endswith("review.html")
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "Initial Review Bundle" in markdown
    assert "run-video-frame-router" in markdown
