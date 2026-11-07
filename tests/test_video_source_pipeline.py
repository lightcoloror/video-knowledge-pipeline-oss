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

def test_extractor_and_peepshow_cli_contracts() -> None:
    run_args = build_parser().parse_args(["run-extractor-plan", "plan.json", "peepshow", "--execute", "--timeout-seconds", "30"])
    assert run_args.command == "run-extractor-plan"
    assert run_args.plan_json == "plan.json"
    assert run_args.extractor == "peepshow"
    assert run_args.execute is True
    assert run_args.timeout_seconds == 30

    attach_args = build_parser().parse_args(["attach-peepshow-output", "bundle", "peepshow-out", "--no-write"])
    assert attach_args.command == "attach-peepshow-output"
    assert attach_args.bundle_dir == "bundle"
    assert attach_args.output_dir == "peepshow-out"
    assert attach_args.no_write is True


def test_attach_peepshow_output_preserves_source_evidence_without_overwriting_timeline(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output = tmp_path / "peepshow-out"
    frames = output / "frames"
    frames.mkdir(parents=True)
    (frames / "frame-001.jpg").write_bytes(b"fake image")
    (output / "report.html").write_text("<html>report</html>", encoding="utf-8")
    (output / "ocr.json").write_text("{}", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "outputDir": str(output),
                "input": {"originalPath": str(tmp_path / "lesson.mp4"), "filename": "lesson.mp4"},
                "video": {"durationSeconds": 12, "width": 1920, "height": 1080, "fps": 30},
                "strategy": "scene",
                "frames": [
                    {
                        "timestampSeconds": 3.5,
                        "path": "frames/frame-001.jpg",
                        "ocr": {"text": "屏幕文字"},
                        "tags": ["slide", "demo"],
                    }
                ],
                "audio": {"transcript": {"segments": [{"start": 3.0, "end": 4.0, "text": "老师讲解这一页"}]}},
                "analysis": {
                    "summary": "Peepshow summary",
                    "perFrame": [{"description": "shows a slide", "tags": ["analysis-tag"]}],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original_timeline = [{"index": 1, "transcript": "原始时间线", "visual_route": "semantic_frame"}]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "peepshow attach"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(original_timeline, ensure_ascii=False), encoding="utf-8")

    result = attach_peepshow_output_to_bundle(bundle, output)

    assert result["frame_evidence_count"] == 1
    assert result["source_artifact_count"] >= 3
    assert json.loads((bundle / "timeline.json").read_text(encoding="utf-8")) == original_timeline
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["source_artifacts"]["tool"] == "peepshow"
    assert manifest["sources"][0]["peepshow_import_role"] == "optional_evidence_extractor"
    evidence = json.loads((bundle / "peepshow-evidence.json").read_text(encoding="utf-8"))
    frame_evidence = evidence["frames"][0]
    assert frame_evidence["ocr_text"] == "屏幕文字"
    assert frame_evidence["transcript_excerpt"] == "老师讲解这一页"
    assert frame_evidence["analysis"]["description"] == "shows a slide"
    assert set(frame_evidence["tags"]) == {"slide", "demo", "analysis-tag"}
    evidence_md = (bundle / "peepshow-evidence.md").read_text(encoding="utf-8")
    assert "Peepshow evidence is preserved as source material" in evidence_md
    source_index = json.loads((bundle / "source-artifacts.json").read_text(encoding="utf-8"))
    assert "peepshow" in source_index["tools"]
    assert (bundle / "source-peepshow-report.html").exists()


def test_prepare_video_source_accepts_local_file(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    result = prepare_video_source(str(media), tmp_path / "source")

    assert result["source_kind"] == "local_file"
    assert result["status"] == "ready"
    assert result["local_media_path"] == str(media.resolve())
    assert Path(result["provenance_markdown_path"]).exists()


def test_prepare_local_video_run_writes_human_report(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    result = prepare_local_video_run(media, tmp_path / "run", title="课程测试")

    assert result["schema"] == "video_knowledge_local_video_run.v1"
    assert result["title"] == "课程测试"
    assert Path(result["json_path"]).exists()
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "Video Knowledge Run" in markdown
    assert "ASR Command" in markdown
    assert result["asr_plan"]["preset"] == "sensevoice"


