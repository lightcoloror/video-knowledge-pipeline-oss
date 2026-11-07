from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import video_knowledge_pipeline.cli as cli_module

from video_knowledge_pipeline.acceptance_check import acceptance_check
from video_knowledge_pipeline.acceptance_run import run_acceptance_bundle, run_acceptance_run
from video_knowledge_pipeline.asr_adapter import normalize_asr_output
from video_knowledge_pipeline.asr_environment import asr_environment_status
from video_knowledge_pipeline.asr_execution import asr_smoke, run_asr_plan
from video_knowledge_pipeline.asr_runner import plan_asr_run
from video_knowledge_pipeline.batch_run import batch_video_knowledge_run
from video_knowledge_pipeline.bundle_next import bundle_advance, bundle_advance_log, bundle_advance_queue, bundle_next_action
from video_knowledge_pipeline.bundle_readiness import build_bundle_readiness
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

def test_pending_human_review_is_optional_not_blocking_readiness(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        "repair_status": {
            "items": [
                {"key": "time_gap_recapture", "status": "manual_required", "count": 1},
            ]
        },
    }
    timeline = [
        {
            "index": 1,
            "visual_route": "mixed",
            "material_types": ["table"],
            "quality_issues": ["needs_human_review"],
        },
        {
            "index": 2,
            "visual_route": "semantic_frame",
            "quality_issues": ["needs_human_review", "missing_visual_text"],
        },
    ]
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    readiness = build_bundle_readiness(manifest, timeline, bundle_dir=bundle)

    assert readiness["ready"] is True
    assert readiness["status"] == "ready"
    assert readiness["blockers"] == []
    optional = {item["key"]: item for item in readiness["optional_review_items"]}
    assert optional["pending_review"]["count"] == 2
    assert optional["pending_structured"]["count"] == 1
    assert optional["unreviewed_risk"]["count"] == 2
    assert optional["manual_repair"]["count"] == 1
    assert all(item["blocking"] is False for item in optional.values())

def test_bundle_next_action_and_advance_are_exposed_for_visual_gaps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LECTURE_VISION_API_KEY", raising=False)
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "source_package": ""}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 4,
                    "transcript": "老师展示界面状态。",
                    "visual_route": "semantic_frame",
                    "visual_text": "界面状态",
                    "frame_paths": [str(frame)],
                    "assets": [{"path": "assets/frame.jpg", "copied": "true"}],
                },
                {
                    "index": 2,
                    "start": 4,
                    "end": 8,
                    "transcript": "老师继续展示另一个界面。",
                    "visual_route": "semantic_frame",
                    "visual_text": "另一个界面",
                    "frame_paths": [str(frame)],
                    "assets": [{"path": "assets/frame.jpg", "copied": "true"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    next_action = bundle_next_action(bundle)

    action = next_action["next_action"]
    assert next_action["status"] == "coverage_blocked"
    assert action["mcp_tool"] == "run_multimodal_frame_analysis"
    assert action["human_required"] is False
    assert next_action["safe_smoke_action"]["mcp_tool"] == "controlled_execution_smoke"
    assert next_action["safe_smoke_action"]["mcp_args_path"].endswith("mcp-controlled-execution-smoke.args.json")
    status_report = bundle_status_report(bundle, refresh=False)
    assert status_report["safe_smoke_action"]["mcp_tool"] == "controlled_execution_smoke"
    assert "本地演练" in Path(status_report["report_markdown_path"]).read_text(encoding="utf-8")

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
                    "multimodal_limit": 1,
                    "temporal_limit": 1,
                    "frame_count": 8,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    advanced = bundle_advance(bundle, execute=False)

    assert advanced["status"] == "advanced"
    assert advanced["execute"] is False
    assert advanced["execution_profile"]["provider"]["provider"] == "gemini"
    assert advanced["execution_profile"]["multimodal_limit"] == 1
    assert advanced["before"]["next_action"]["mcp_tool"] == "run_multimodal_frame_analysis"
    assert advanced["action_result"]["item_count"] == 1
    assert advanced["action_result"]["summary"]["total"] == 2
    assert advanced["action_result"]["summary"]["selected"] == 1
    assert advanced["action_result"]["summary"]["provider"]["provider"] == "gemini"
    assert "items" not in advanced["action_result"]

    queue = bundle_advance_queue(bundle, max_steps=1, execute=False)
    queue_args = json.loads(Path(queue["mcp_args_path"]).read_text(encoding="utf-8"))
    assert queue["execution_profile"]["provider"]["provider"] == "gemini"
    assert queue["execution_profile"]["multimodal_limit"] == 1
    assert queue_args["provider_config"]["provider"] == "gemini"
    assert queue_args["multimodal_limit"] == 1
    assert queue_args["temporal_limit"] == 1
    assert queue_args["frame_count"] == 8

    blocked = bundle_advance(bundle, execute=True, provider_config={"provider": "gemini", "model": "gemini-2.5-flash"}, multimodal_limit=1)
    assert blocked["status"] == "blocked"
    assert blocked["blocked_reason"].startswith("vision execution preflight blocked")
    assert blocked["action_result"]["status"] == "vision_preflight_blocked"
    assert Path(blocked["action_result"]["preflight_path"]).exists()
    assert blocked["action_result"]["summary"]["ready_to_execute"] is False
    assert "missing_api_key" in blocked["action_result"]["summary"]["blocker_keys"]
    assert blocked["action_result"]["summary"]["provider"]["api_key_configured"] is False
    assert "items" not in blocked["action_result"]

    needs_confirmation = bundle_advance(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "model": "gemini-2.5-flash", "api_key": "secret"},
        multimodal_limit=1,
    )
    assert needs_confirmation["status"] == "blocked"
    assert needs_confirmation["action_result"]["status"] == "vision_confirmation_required"
    assert needs_confirmation["blocked_reason"].startswith("vision execution confirmation required")
    assert needs_confirmation["action_result"]["summary"]["expected_api_calls"] == 1
    assert needs_confirmation["action_result"]["summary"]["expected_indexes"] == "1"

    calls: dict[str, Any] = {}

    def fake_multimodal(bundle_dir, **kwargs):
        calls["kwargs"] = kwargs
        return {
            "bundle_dir": str(bundle),
            "summary": {"schema": "fake", "updated": 1},
            "report_path": str(bundle / "fake-vision.md"),
            "run_audit": {
                "record": {
                    "run_id": "semantic_frame-20260606-000000",
                    "kind": "semantic_frame",
                    "execute": True,
                    "status": "ok",
                    "updated_count": 1,
                    "timeline_diff_count": 1,
                    "bundle_dir": str(bundle),
                },
                "jsonl_path": str(bundle / "vision-analysis-runs.jsonl"),
                "markdown_path": str(bundle / "vision-analysis-runs.md"),
            },
            "vision_restore_hint": {
                "status": "ready",
                "run_id": "semantic_frame-20260606-000000",
                "kind": "semantic_frame",
                "updated_count": 1,
                "timeline_diff_count": 1,
                "audit_jsonl_path": str(bundle / "vision-analysis-runs.jsonl"),
                "audit_markdown_path": str(bundle / "vision-analysis-runs.md"),
                "restore_plan_command": f'python -m video_knowledge_pipeline.cli vision-analysis-restore-plan "{bundle}" --run-id semantic_frame-20260606-000000',
                "restore_apply_execute_command": f'python -m video_knowledge_pipeline.cli vision-analysis-apply-restore "{bundle}" --plan-json "{bundle / "vision-restore-plan.json"}" --execute --confirm-run-id semantic_frame-20260606-000000',
            },
        }

    monkeypatch.setattr("video_knowledge_pipeline.bundle_next.run_multimodal_frame_analysis", fake_multimodal)
    confirmed = bundle_advance(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "model": "gemini-2.5-flash", "api_key": "secret"},
        multimodal_limit=1,
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
    )
    assert confirmed["status"] == "advanced"
    assert calls["kwargs"]["execute"] is True
    assert confirmed["action_result"]["run_audit"]["run_id"] == "semantic_frame-20260606-000000"
    assert confirmed["action_result"]["vision_restore_hint"]["status"] == "ready"
    assert confirmed["action_result"]["vision_restore_hint"]["run_id"] == "semantic_frame-20260606-000000"
    assert "vision-analysis-restore-plan" in confirmed["action_result"]["vision_restore_hint"]["restore_plan_command"]
    assert "vision-analysis-apply-restore" in confirmed["action_result"]["vision_restore_hint"]["restore_apply_execute_command"]

    log = bundle_advance_log(bundle)
    assert log["count"] == 5
    assert log["advances"][-2]["action_artifacts"]["preflight_path"] == needs_confirmation["action_result"]["preflight_path"]
    assert log["last"]["action_artifacts"]["vision_run_id"] == "semantic_frame-20260606-000000"
    assert "vision-analysis-restore-plan" in log["last"]["action_artifacts"]["vision_restore_plan_command"]
    assert Path(log["markdown_path"]).exists()


def test_local_mcp_call_runs_confirmed_args_and_filters_extra_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LECTURE_VISION_API_KEY", raising=False)
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "visual_route": "semantic_frame", "frame_paths": [str(frame)]}], ensure_ascii=False),
        encoding="utf-8",
    )

    preflight = vision_execution_preflight(
        bundle,
        provider_config={"provider": "gemini", "api_key": "secret-value", "model": "gemini-2.5-flash"},
        semantic_limit=1,
        include_temporal=False,
    )
    args_path = bundle / "mcp-run-multimodal-frame-analysis-confirmed.args.json"
    args_payload = json.loads(args_path.read_text(encoding="utf-8"))
    assert preflight["ready_to_execute"] is True
    assert args_payload["provider_config"]["provider"] == "gemini"
    assert "api_key" not in args_payload["provider_config"]

    result = run_mcp_call("run_multimodal_frame_analysis", args_path)

    assert result["summary"]["status"] == "vision_provider_not_ready"
    assert result["summary"]["error"] == "missing_api_key"
    assert result["mcp_call"]["tool"] == "run_multimodal_frame_analysis"
    assert result["mcp_call"]["ignored_args"] == []

    log_args = bundle / "mcp-bundle-advance-log.args.json"
    log_args.write_text(json.dumps({"bundle_dir": str(bundle), "write": True}, ensure_ascii=False), encoding="utf-8")
    log = run_mcp_call("bundle_advance_log", log_args)
    assert log["mcp_call"]["ignored_args"] == ["write"]
    assert log["bundle_dir"] == str(bundle.resolve())


def test_local_mcp_args_resolver_finds_unique_project_relative_file(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "isolated-project"
    package_dir = project_root / "src" / "video_knowledge_pipeline"
    package_dir.mkdir(parents=True)
    fake_cli = package_dir / "cli.py"
    fake_cli.write_text("# synthetic module location\n", encoding="utf-8")
    marker = project_root / "tmp-unique-mcp-resolver.args.json"
    marker.write_text(json.dumps({"ok": True}, ensure_ascii=False), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(cli_module, "__file__", str(fake_cli))
    monkeypatch.chdir(outside)

    resolved = resolve_mcp_args_path(marker.name)

    assert resolved == marker


def test_bundle_mcp_args_audit_checks_supported_tools_and_args_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "mcp_multimodal_frame_analysis_args": "mcp-run-multimodal-frame-analysis.args.json",
                "mcp_status_report_args": "mcp-bundle-status-report.args.json",
                "mcp_refresh_args": "mcp-refresh-lecture-review.args.json",
                "mcp_unknown_args": "mcp-unknown.args.json",
                "mcp_missing_args": "mcp-missing.args.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "mcp-run-multimodal-frame-analysis.args.json").write_text(
        json.dumps({"bundle_dir": str(bundle), "execute": False}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "mcp-bundle-status-report.args.json").write_text(
        json.dumps({"bundle_dir": str(bundle), "refresh": False, "write": True}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "mcp-refresh-lecture-review.args.json").write_text(
        json.dumps({"project": str(bundle), "review_json": str(bundle / "review-notes.json")}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "mcp-unknown.args.json").write_text(json.dumps({"bundle_dir": str(bundle)}, ensure_ascii=False), encoding="utf-8")

    audit = audit_bundle_mcp_args(bundle)

    rows = {row["key"]: row for row in audit["rows"]}
    assert audit["status"] == "blocked"
    assert rows["mcp_multimodal_frame_analysis_args"]["ok"] is True
    assert rows["mcp_status_report_args"]["ok"] is True
    assert rows["mcp_status_report_args"]["ignored_args"] == []
    assert rows["mcp_refresh_args"]["ok"] is True
    assert rows["mcp_refresh_args"]["ignored_args"] == ["project"]
    assert rows["mcp_unknown_args"]["ok"] is False
    assert "unsupported_tool" in rows["mcp_unknown_args"]["issues"]
    assert rows["mcp_missing_args"]["ok"] is False
    assert "missing_args_file" in rows["mcp_missing_args"]["issues"]


