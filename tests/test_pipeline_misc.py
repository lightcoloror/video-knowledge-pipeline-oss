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

def test_direct_visual_tools_use_unified_batch_defaults(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 9876, "path": "/call"},
                    "mcp": {"transport": "stdio"},
                },
                "vision_execution": {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "multimodal_limit": 2,
                    "temporal_limit": 1,
                    "frame_count": 6,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"fake video")
    timeline = []
    for index in range(1, 5):
        timeline.append(
            {
                "index": index,
                "start": index,
                "end": index + 1,
                "visual_route": "mixed",
                "frame_paths": [str(frame)],
                "temporal_frame_paths": [str(frame), str(frame), str(frame), str(frame), str(frame), str(frame), str(frame)],
                "video_key": str(video),
            }
        )
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    multimodal_default = run_multimodal_frame_analysis(bundle)
    multimodal_all = run_multimodal_frame_analysis(bundle, limit=0)
    groups_default = run_temporal_frame_groups(bundle)
    temporal_default = run_temporal_visual_analysis(bundle)

    assert multimodal_default["summary"]["selected"] == 2
    assert multimodal_default["summary"]["limit"] == 2
    assert multimodal_all["summary"]["selected"] == 4
    assert multimodal_all["summary"]["limit"] == 0
    assert groups_default["summary"]["total"] == 1
    assert groups_default["summary"]["limit"] == 1
    assert groups_default["summary"]["frame_count"] == 6
    assert temporal_default["summary"]["total"] == 1
    assert temporal_default["summary"]["limit"] == 1
    assert temporal_default["summary"]["frame_count"] == 6


def test_visual_understanding_maps_model_frame_basenames_to_candidate_paths(tmp_path: Path) -> None:
    frame_path = tmp_path / "assets" / "0001-001_0000000000ms.jpg"
    payload = {
        "objects": ["screen"],
        "evidence_frame_paths": [frame_path.name],
    }

    result = _normalise_visual_understanding(payload, {"frame_paths": [str(frame_path)]})

    assert result["validation_status"] == "ok"
    assert result["evidence_frame_paths"] == [str(frame_path)]


def test_incomplete_visual_understanding_remains_coverage_gap() -> None:
    manifest = {
        "mcp_multimodal_frame_analysis_args": "frame.args.json",
        "mcp_temporal_visual_analysis_args": "temporal.args.json",
    }
    timeline = [
        {
            "index": 1,
            "visual_route": "semantic_frame",
            "visual_understanding": {
                "schema": "lecture_visual_understanding.v1",
                "validation_status": "incomplete",
                "validation_issues": ["missing_visual_content"],
                "evidence_frame_paths": ["frame.jpg"],
            },
            "assets": [{"path": "frame.jpg"}],
        },
        {
            "index": 2,
            "visual_route": "temporal_sequence",
            "temporal_visual_understanding": {
                "schema": "lecture_temporal_visual_understanding.v1",
                "parse_failed": True,
                "event_sequence": ["raw fallback"],
                "evidence_frame_paths": ["frame.jpg"],
            },
            "assets": [{"path": "frame.jpg"}],
        },
    ]

    coverage = build_knowledge_coverage(manifest, timeline)

    assert coverage["items_with_visual_understanding"] == 0
    assert coverage["items_with_temporal_understanding"] == 0
    assert coverage["semantic_frame_without_analysis"] == 1
    assert coverage["temporal_sequence_without_analysis"] == 1
    assert coverage["missing_visual_understanding"] == 2


