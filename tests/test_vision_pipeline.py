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
from video_knowledge_pipeline.storage import bundle_write_lock, read_json, write_json
from video_knowledge_pipeline.tagger_import import import_tagger_annotations
from video_knowledge_pipeline.temporal_frame_groups import run_temporal_frame_groups
from video_knowledge_pipeline.term_resolution import resolve_terms
from video_knowledge_pipeline.temporal_visual_analyzer import _normalise_temporal_understanding, run_temporal_visual_analysis
from video_knowledge_pipeline.transcript_resegment import resegment_transcript
from video_knowledge_pipeline.vision_acceptance import vision_acceptance_plan
from video_knowledge_pipeline.video_frame_router import run_video_frame_router
from video_knowledge_pipeline.video_source import prepare_video_source
from video_knowledge_pipeline.vision_api import parse_model_json, resolve_provider_config, test_vision_provider as run_vision_provider_test
from video_knowledge_pipeline.vision_environment import vision_environment_status
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight
from video_knowledge_pipeline.vision_provider_smoke import rank_vision_providers, vision_provider_matrix, vision_provider_smoke
from video_knowledge_pipeline.vision_review_triage import vision_review_triage
from video_knowledge_pipeline.webui_bridge import export_webui_bundle, refresh_bundle_review_html
import video_knowledge_pipeline.visual_structure as visual_structure
from video_knowledge_pipeline.visual_structure import run_visual_structure_plan



# Moved from test_video_pipeline_smoke.py during Phase 10 split.

def test_vision_execution_config_rejects_nested_secrets(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-vision-profile.json"
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
                    "provider_config": {"api_key": "must-not-live-here"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    status = config_status(config_path)

    issue_keys = {issue["key"] for issue in status["validation"]["issues"]}
    assert status["ok"] is False
    assert "secret_in_config" in issue_keys


def test_router_and_multimodal_limit(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"start": 0, "end": 1, "visual_text": "", "transcript": "老师展示界面", "frame_paths": [str(frame)], "material_types": ["interface"]},
                {"start": 1, "end": 2, "visual_text": "", "transcript": "老师指向按钮", "frame_paths": [str(frame)], "material_types": ["interface"]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    routed = run_video_frame_router(bundle)
    assert routed["summary"]["routes"]["semantic_frame"] == 2

    preview = run_multimodal_frame_analysis(bundle, limit=1, provider_config={"provider": "gemini", "model": "gemini-2.5-flash"})
    assert preview["summary"]["total"] == 2
    assert preview["summary"]["selected"] == 1
    assert preview["summary"]["provider"]["provider"] == "gemini"
    assert len(preview["items"]) == 1
    assert preview["run_registry"]["run_type"] == "multimodal_frame_analysis"
    assert preview["run_registry"]["status"] == "needs_execution"
    registry = read_json(bundle / "run-artifact-registry.json")
    assert any(row["run_type"] == "multimodal_frame_analysis" for row in registry["runs"])


def test_direct_vision_execute_gates_missing_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LECTURE_VISION_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
                "vision_execution": {"provider": "gemini", "model": "gemini-2.5-flash", "multimodal_limit": 1, "temporal_limit": 1, "frame_count": 5},
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
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 1,
            "visual_route": "mixed",
            "frame_paths": [str(frame)],
            "temporal_frame_paths": [str(frame), str(frame), str(frame), str(frame), str(frame)],
        }
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    explicit_legacy = {"provider": "gemini", "model": "gemini-2.5-flash"}
    multimodal = run_multimodal_frame_analysis(bundle, execute=True, provider_config=explicit_legacy)
    temporal = run_temporal_visual_analysis(bundle, execute=True, provider_config=explicit_legacy)
    updated_timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert multimodal["summary"]["status"] == "vision_provider_not_ready"
    assert multimodal["summary"]["error"] == "missing_api_key"
    assert multimodal["items"][0]["executed"] is False
    assert temporal["summary"]["status"] == "vision_provider_not_ready"
    assert temporal["summary"]["error"] == "missing_api_key"
    assert temporal["items"][0]["executed"] is False
    assert "visual_understanding" not in updated_timeline[0]
    assert "temporal_visual_understanding" not in updated_timeline[0]
    assert "vision_provider_not_ready" in Path(multimodal["report_path"]).read_text(encoding="utf-8")


def test_direct_vision_execute_requires_preflight_confirmation(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 1,
            "visual_route": "mixed",
            "frame_paths": [str(frame)],
            "temporal_frame_paths": [str(frame), str(frame), str(frame)],
        }
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal_mod
    import video_knowledge_pipeline.temporal_visual_analyzer as temporal_mod

    monkeypatch.setattr(multimodal_mod, "call_vision_model", lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should be gated")))
    monkeypatch.setattr(temporal_mod, "call_vision_model", lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should be gated")))

    multimodal = run_multimodal_frame_analysis(bundle, execute=True, provider_config={"provider": "gemini", "api_key": "secret"}, limit=1)
    temporal = run_temporal_visual_analysis(bundle, execute=True, provider_config={"provider": "gemini", "api_key": "secret"}, limit=1)

    assert multimodal["summary"]["status"] == "vision_confirmation_required"
    assert multimodal["summary"]["error"] == "confirm_vision_mismatch"
    assert multimodal["post_run_refresh"]["status"] == "skipped"
    assert multimodal["vision_restore_hint"]["status"] == "not_needed"
    assert multimodal["vision_restore_hint"]["reason"] == "no_timeline_updates"
    assert multimodal["summary"]["preflight_path"]
    assert multimodal["summary"]["expected_api_calls"] == 1
    assert multimodal["summary"]["expected_indexes"] == "1"
    assert multimodal["items"][0]["executed"] is False
    assert temporal["summary"]["status"] == "vision_confirmation_required"
    assert temporal["summary"]["error"] == "confirm_vision_mismatch"
    assert temporal["post_run_refresh"]["status"] == "skipped"
    assert temporal["vision_restore_hint"]["status"] == "not_needed"
    assert temporal["vision_restore_hint"]["reason"] == "no_timeline_updates"
    assert temporal["summary"]["preflight_path"]
    assert temporal["summary"]["expected_api_calls"] == 1
    assert temporal["summary"]["expected_indexes"] == "1"
    assert temporal["items"][0]["executed"] is False


def test_local_proxy_visual_tasks_use_unified_runtime_and_preserve_timeline_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame_a = assets / "frame-a.jpg"
    frame_b = assets / "frame-b.jpg"
    frame_a.write_bytes(b"fake-image-a")
    frame_b.write_bytes(b"fake-image-b")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 2,
            "visual_route": "mixed",
            "frame_paths": [str(frame_a)],
            "temporal_frame_paths": [str(frame_a), str(frame_b)],
        }
    ]
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal_mod
    import video_knowledge_pipeline.online_model_gateway as online_gateway
    import video_knowledge_pipeline.temporal_visual_analyzer as temporal_mod

    calls: list[dict[str, object]] = []

    def keep_explicit(model_type: str, explicit: dict[str, object] | None = None, **kwargs: object) -> dict[str, object]:
        return dict(explicit or {})

    def fake_runtime(task: str, **kwargs: object) -> dict[str, object]:
        calls.append({"task": task, **kwargs})
        if task == "semantic_frame":
            content = {
                "objects": ["slide"],
                "actions": ["presenting"],
                "interface_state": "title slide",
                "spatial_relations": ["speaker inset at lower right"],
                "instructor_focus": "chapter title",
                "non_text_information": ["orange circular layout"],
                "confidence": 0.95,
                "keep_image_reason": "layout matters",
                "evidence_frame_paths": [],
            }
        else:
            content = {
                "event_sequence": ["title remains visible", "speaker gestures"],
                "state_changes": ["speaker pose changes"],
                "operation_steps": [],
                "causal_links": [],
                "possible_missing_points": [],
                "confidence": 0.9,
                "evidence_frame_paths": [],
            }
        return {"ok": True, "status": "completed", "content": json.dumps(content, ensure_ascii=False)}

    def fake_preflight(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "ready_to_execute": True,
            "expected_api_calls": 1,
            "selected_indexes": {
                "semantic": [1] if kwargs.get("include_semantic") else [],
                "temporal": [1] if kwargs.get("include_temporal") else [],
            },
            "preflight_path": "",
            "preflight_json_path": "",
        }

    monkeypatch.setattr(online_gateway, "resolve_model_api_provider_config", keep_explicit)
    monkeypatch.setattr(online_gateway, "model_runtime_request", fake_runtime)
    monkeypatch.setattr(multimodal_mod, "vision_execution_preflight", fake_preflight)
    monkeypatch.setattr(
        temporal_mod,
        "_execution_control",
        lambda *args, **kwargs: (None, {"status": "ready"}),
    )
    provider = {
        "provider": "openai_compatible",
        "adapter_backend": "proxy",
        "base_url": "http://127.0.0.1:8776/v1",
        "model": "vkp-local-vision-test",
        "location": "local",
        "execution_location": "local",
        "route_id": "pool-local-vision",
        "route_revision": "a" * 64,
    }

    semantic = run_multimodal_frame_analysis(
        bundle,
        execute=True,
        provider_config=provider,
        limit=1,
        indexes=[1],
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
    )
    temporal = run_temporal_visual_analysis(
        bundle,
        execute=True,
        provider_config=provider,
        frame_count=2,
        limit=1,
        indexes=[1],
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
    )

    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]
    assert [row["task"] for row in calls] == ["semantic_frame", "temporal_sequence"]
    assert all(row["execution_location"] == "local" for row in calls)
    assert semantic["summary"]["updated"] == 1
    assert temporal["summary"]["updated"] == 1
    assert updated["visual_understanding"]["objects"] == ["slide"]
    assert updated["temporal_visual_understanding"]["event_sequence"][0] == "title remains visible"
    assert updated["visual_understanding"]["schema"].startswith("lecture_visual_understanding")
    assert updated["temporal_visual_understanding"]["schema"].startswith("lecture_temporal_visual_understanding")

def test_direct_vision_candidates_skip_existing_understanding_to_match_preflight(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 1,
            "visual_route": "mixed",
            "frame_paths": [str(frame)],
            "temporal_frame_paths": [str(frame), str(frame), str(frame)],
            "visual_understanding": {"objects": ["already done"]},
            "temporal_visual_understanding": {"event_sequence": ["already done"]},
        },
        {
            "index": 2,
            "start": 1,
            "end": 2,
            "visual_route": "mixed",
            "frame_paths": [str(frame)],
            "temporal_frame_paths": [str(frame), str(frame), str(frame)],
        },
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal_mod
    import video_knowledge_pipeline.temporal_visual_analyzer as temporal_mod

    monkeypatch.setattr(multimodal_mod, "call_vision_model", lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should be gated")))
    monkeypatch.setattr(temporal_mod, "call_vision_model", lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should be gated")))

    multimodal = run_multimodal_frame_analysis(bundle, execute=True, provider_config={"provider": "gemini", "api_key": "secret"}, limit=1)
    temporal = run_temporal_visual_analysis(bundle, execute=True, provider_config={"provider": "gemini", "api_key": "secret"}, limit=1)

    assert multimodal["summary"]["expected_indexes"] == "2"
    assert multimodal["items"][0]["index"] == 2
    assert temporal["summary"]["expected_indexes"] == "2"
    assert temporal["items"][0]["index"] == 2


def test_explicit_document_route_can_be_selected_for_semantic_conflict_analysis(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-document-semantic"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 1,
            "visual_route": "document_visual",
            "frame_paths": [str(frame)],
            "visual_text": "OCR text",
        }
    ]
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    preflight = vision_execution_preflight(
        bundle,
        provider_config={"provider": "local_qwen_vl"},
        semantic_limit=1,
        temporal_limit=0,
        include_temporal=False,
        semantic_indexes=[1],
    )
    preview = run_multimodal_frame_analysis(
        bundle,
        execute=False,
        provider_config={"provider": "local_qwen_vl"},
        limit=1,
        indexes=[1],
    )

    assert preflight["selected_indexes"]["semantic"] == [1]
    assert preflight["expected_api_calls"] == 1
    assert preview["items"][0]["index"] == 1
    assert preview["items"][0]["visual_route"] == "document_visual"


def test_direct_multimodal_indexes_are_reflected_in_preflight_confirmation(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {"index": 1, "start": 0, "end": 1, "visual_route": "semantic_frame", "frame_paths": [str(frame)]},
        {"index": 2, "start": 1, "end": 2, "visual_route": "semantic_frame", "frame_paths": [str(frame)]},
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal_mod

    calls: list[dict[str, object]] = []

    def fake_call_vision_model(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "error": "", "content": "{}"}

    monkeypatch.setattr(multimodal_mod, "call_vision_model", fake_call_vision_model)

    result = run_multimodal_frame_analysis(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "api_key": "secret"},
        limit=1,
        indexes=[2],
    )

    assert result["summary"]["status"] == "vision_confirmation_required"
    assert result["summary"]["expected_api_calls"] == 1
    assert result["summary"]["expected_indexes"] == "2"
    assert result["items"][0]["index"] == 2
    assert result["run_audit"]["record"]["execution_control"]["status"] == "vision_confirmation_required"
    assert result["run_audit"]["record"]["execution_control"]["confirmed"] is False
    preflight = json.loads(Path(result["summary"]["preflight_json_path"]).read_text(encoding="utf-8"))
    assert preflight["execution_profile"]["semantic_indexes"] == [2]
    assert preflight["selected_indexes"]["semantic"] == [2]
    assert calls == []

    confirmed = run_multimodal_frame_analysis(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "api_key": "secret"},
        limit=1,
        indexes=[2],
        confirm_vision_calls=1,
        confirm_vision_indexes="2",
    )

    assert len(calls) == 1
    assert confirmed["items"][0]["index"] == 2
    assert confirmed["items"][0]["executed"] is True
    assert confirmed["summary"]["updated"] == 1
    assert confirmed["post_run_refresh"]["status"] == "ok"
    assert Path(confirmed["post_run_refresh"]["knowledge_coverage_path"]).exists()
    assert Path(confirmed["post_run_refresh"]["bundle_status_report_path"]).exists()
    assert Path(confirmed["post_run_refresh"]["controlled_execution_check_path"]).exists()
    assert confirmed["vision_restore_hint"]["status"] == "ready"
    assert confirmed["vision_restore_hint"]["run_id"] == confirmed["run_audit"]["record"]["run_id"]
    assert "vision-analysis-restore-plan" in confirmed["vision_restore_hint"]["restore_plan_command"]
    assert "--confirm-run-id" in confirmed["vision_restore_hint"]["restore_apply_execute_command"]
    control = confirmed["run_audit"]["record"]["execution_control"]
    assert control["status"] == "confirmed"
    assert control["confirmed"] is True
    assert control["expected_api_calls"] == 1
    assert control["expected_indexes"] == "2"
    assert control["received_confirm_vision_calls"] == 1
    assert control["received_confirm_vision_indexes"] == "2"
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert "visual_understanding" not in updated[0]
    assert updated[1]["visual_understanding"]["validation_status"] == "incomplete"


def test_direct_temporal_indexes_are_reflected_in_preflight_confirmation(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {"index": 1, "start": 0, "end": 1, "visual_route": "temporal_sequence", "frame_paths": [str(frame)], "temporal_frame_paths": [str(frame), str(frame)]},
        {"index": 2, "start": 1, "end": 2, "visual_route": "temporal_sequence", "frame_paths": [str(frame)], "temporal_frame_paths": [str(frame), str(frame)]},
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    import video_knowledge_pipeline.temporal_visual_analyzer as temporal_mod

    calls: list[dict[str, object]] = []

    def fake_call_vision_model(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "error": "", "content": "{}"}

    monkeypatch.setattr(temporal_mod, "call_vision_model", fake_call_vision_model)

    result = run_temporal_visual_analysis(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "api_key": "secret"},
        limit=1,
        indexes=[2],
    )

    assert result["summary"]["status"] == "vision_confirmation_required"
    assert result["summary"]["expected_api_calls"] == 1
    assert result["summary"]["expected_indexes"] == "2"
    assert result["items"][0]["index"] == 2
    assert result["run_audit"]["record"]["execution_control"]["status"] == "vision_confirmation_required"
    assert result["run_audit"]["record"]["execution_control"]["confirmed"] is False
    preflight = json.loads(Path(result["summary"]["preflight_json_path"]).read_text(encoding="utf-8"))
    assert preflight["execution_profile"]["temporal_indexes"] == [2]
    assert preflight["selected_indexes"]["temporal"] == [2]
    assert calls == []

    confirmed = run_temporal_visual_analysis(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "api_key": "secret"},
        limit=1,
        indexes=[2],
        confirm_vision_calls=1,
        confirm_vision_indexes="2",
    )

    assert len(calls) == 1
    assert confirmed["items"][0]["index"] == 2
    assert confirmed["items"][0]["executed"] is True
    assert confirmed["summary"]["updated"] == 1
    assert confirmed["post_run_refresh"]["status"] == "ok"
    assert Path(confirmed["post_run_refresh"]["knowledge_coverage_path"]).exists()
    assert Path(confirmed["post_run_refresh"]["bundle_status_report_path"]).exists()
    assert Path(confirmed["post_run_refresh"]["controlled_execution_check_path"]).exists()
    assert confirmed["vision_restore_hint"]["status"] == "ready"
    assert confirmed["vision_restore_hint"]["run_id"] == confirmed["run_audit"]["record"]["run_id"]
    assert "vision-analysis-restore-plan" in confirmed["vision_restore_hint"]["restore_plan_command"]
    assert "--confirm-run-id" in confirmed["vision_restore_hint"]["restore_apply_execute_command"]
    control = confirmed["run_audit"]["record"]["execution_control"]
    assert control["status"] == "confirmed"
    assert control["confirmed"] is True
    assert control["expected_api_calls"] == 1
    assert control["expected_indexes"] == "2"
    assert control["received_confirm_vision_calls"] == 1
    assert control["received_confirm_vision_indexes"] == "2"
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert "temporal_visual_understanding" not in updated[0]
    assert updated[1]["temporal_visual_understanding"]["validation_status"] == "incomplete"


def test_model_output_validation_marks_incomplete_understanding(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 1,
            "visual_route": "mixed",
            "frame_paths": [str(frame)],
            "temporal_frame_paths": [str(frame), str(frame), str(frame), str(frame), str(frame)],
        }
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal_mod
    import video_knowledge_pipeline.temporal_visual_analyzer as temporal_mod

    def fake_call_vision_model(*, provider_config, prompt, image_paths, allowed_roots=None):
        return {"ok": True, "error": "", "content": "{}"}

    monkeypatch.setattr(multimodal_mod, "call_vision_model", fake_call_vision_model)
    monkeypatch.setattr(temporal_mod, "call_vision_model", fake_call_vision_model)

    multimodal = run_multimodal_frame_analysis(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "api_key": "secret"},
        limit=1,
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
    )
    temporal = run_temporal_visual_analysis(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "api_key": "secret"},
        limit=1,
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
    )
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]

    assert multimodal["summary"]["updated"] == 1
    assert temporal["summary"]["updated"] == 1
    assert multimodal["post_run_refresh"]["status"] == "ok"
    assert temporal["post_run_refresh"]["status"] == "ok"
    assert (bundle / "knowledge-coverage.json").exists()
    assert (bundle / "bundle-status.json").exists()
    assert (bundle / "controlled-execution-check.json").exists()
    assert updated["visual_understanding"]["validation_status"] == "incomplete"
    assert "missing_visual_content" in updated["visual_understanding"]["validation_issues"]
    assert updated["temporal_visual_understanding"]["validation_status"] == "incomplete"
    assert "missing_temporal_content" in updated["temporal_visual_understanding"]["validation_issues"]
    assert "visual_understanding_incomplete" in updated["quality_issues"]
    assert "temporal_understanding_incomplete" in updated["quality_issues"]
    multimodal_report = Path(multimodal["report_path"]).read_text(encoding="utf-8")
    temporal_report = Path(temporal["report_path"]).read_text(encoding="utf-8")
    assert "本次结果审核表" in multimodal_report
    assert "| Index | Time | Route | Frame | Executed | OK | Validation | Issues | Transcript | OCR / visual text | Model understanding | Keep image reason | Confidence | Evidence |" in multimodal_report
    assert "missing_visual_content" in multimodal_report
    assert "Model Understanding" in multimodal_report
    assert "本次结果审核表" in temporal_report
    assert "missing_temporal_content" in temporal_report
    assert "Model Understanding" in temporal_report
    assert Path(multimodal["run_audit"]["jsonl_path"]).exists()
    assert Path(multimodal["run_audit"]["markdown_path"]).exists()
    assert Path(temporal["run_audit"]["jsonl_path"]).exists()
    audit_rows = [
        json.loads(line)
        for line in Path(temporal["run_audit"]["jsonl_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["kind"] for row in audit_rows] == ["semantic_frame", "temporal_sequence"]
    assert audit_rows[0]["selected_count"] == 1
    assert audit_rows[0]["executed_count"] == 1
    assert audit_rows[0]["incomplete_count"] == 1
    assert audit_rows[1]["incomplete_count"] == 1
    assert audit_rows[0]["timeline_diff_count"] == 1
    assert audit_rows[1]["timeline_diff_count"] == 1
    assert audit_rows[0]["timeline_diff"][0]["index"] == 1
    assert "visual_understanding" in audit_rows[0]["timeline_diff"][0]["changed_fields"]
    assert "temporal_visual_understanding" in audit_rows[1]["timeline_diff"][0]["changed_fields"]
    first_change = audit_rows[0]["timeline_diff"][0]["changes"][0]
    assert "before_value" in first_change
    assert "after_value" in first_change
    assert "secret" not in json.dumps(audit_rows, ensure_ascii=False)
    assert "raw_model_output" not in json.dumps(audit_rows, ensure_ascii=False)
    audit_markdown = Path(temporal["run_audit"]["markdown_path"]).read_text(encoding="utf-8")
    assert "Vision Analysis Runs" in audit_markdown
    assert "semantic_frame" in audit_markdown
    assert "temporal_sequence" in audit_markdown
    assert "Timeline Diff" in audit_markdown
    assert "visual_understanding" in audit_markdown

    log = vision_analysis_run_log(bundle)
    assert log["count"] == 2
    assert log["last_run"]["kind"] == "temporal_sequence"
    assert log["last_run"]["timeline_diff_count"] == 1

    restore = vision_analysis_restore_plan(bundle, run_id=audit_rows[0]["run_id"])
    assert restore["status"] == "ready"
    assert restore["restorable_count"] == 1
    assert restore["operations"][0]["index"] == 1
    assert "visual_understanding" in restore["operations"][0]["changed_fields"]
    assert restore["operations"][0]["changes"][0]["action"] == "remove_field"
    assert Path(restore["json_path"]).exists()
    assert Path(restore["markdown_path"]).exists()
    restore_text = json.dumps(restore, ensure_ascii=False)
    assert "secret" not in restore_text
    assert "raw_model_output" not in restore_text

    dry_run = vision_analysis_apply_restore(bundle, plan_json=restore["json_path"])
    assert dry_run["summary"]["execute"] is False
    assert dry_run["summary"]["status"] == "ok"
    assert dry_run["summary"]["applied_count"] == 0
    assert json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0].get("visual_understanding")

    blocked = vision_analysis_apply_restore(bundle, plan_json=restore["json_path"], execute=True, confirm_run_id="wrong")
    assert blocked["summary"]["status"] == "restore_confirmation_required"
    assert blocked["summary"]["applied_count"] == 0
    assert json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0].get("visual_understanding")

    applied_restore = vision_analysis_apply_restore(
        bundle,
        plan_json=restore["json_path"],
        execute=True,
        confirm_run_id=audit_rows[0]["run_id"],
    )
    restored_timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert applied_restore["summary"]["status"] == "ok"
    assert applied_restore["summary"]["applied_count"] == 1
    assert "visual_understanding" not in restored_timeline[0]
    assert Path(applied_restore["audit"]["jsonl_path"]).exists()
    assert Path(applied_restore["audit"]["markdown_path"]).exists()


def test_temporal_understanding_maps_model_frame_basenames_to_candidate_paths(tmp_path: Path) -> None:
    frame_paths = [
        tmp_path / "temporal-frames" / "0012" / "frame_01_0000320000ms.jpg",
        tmp_path / "temporal-frames" / "0012" / "frame_02_0000321000ms.jpg",
    ]
    payload = {
        "event_sequence": ["cursor moves across the page"],
        "evidence_frame_paths": [frame_paths[0].name, frame_paths[1].name],
    }

    result = _normalise_temporal_understanding(payload, {"frame_paths": [str(path) for path in frame_paths]})

    assert result["validation_status"] == "ok"
    assert result["evidence_frame_paths"] == [str(path) for path in frame_paths]


def test_webui_export_preserves_controlled_execution_entrypoints(tmp_path: Path) -> None:
    root = tmp_path / "run"
    packages = root / "lecture-packages"
    packages.mkdir(parents=True)
    (packages / "lecture-package.json").write_text(
        json.dumps(
            {
                "title": "preserve controls",
                "coverage": {},
                "sources": [],
                "timeline": [{"start": 0, "end": 1, "transcript": "hello", "visual_route": "semantic_frame"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    exported = export_webui_bundle(root, output_dir=bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mcp_controlled_execution_check_args"] == "mcp-controlled-execution-check.args.json"
    assert (bundle / "mcp-controlled-execution-check.args.json").exists()
    assert manifest["mcp_controlled_execution_smoke_args"] == "mcp-controlled-execution-smoke.args.json"
    assert manifest["mcp_apply_review_notes_args"] == "mcp-apply-review-notes.args.json"
    assert manifest["mcp_acceptance_check_args"] == "mcp-acceptance-check.args.json"
    assert manifest["acceptance_check"] == "acceptance-check.md"
    assert manifest["review_session"] == "review-session.md"
    assert manifest["review_notes_template"] == "review-notes.template.json"
    assert manifest["review_fill_guide"] == "review-fill-guide.md"
    assert manifest["knowledge_note_extraction_audit_markdown"] == "exports/extraction-audit.md"
    assert (bundle / "mcp-acceptance-check.args.json").exists()
    assert (bundle / "review-notes.template.json").exists()
    assert (bundle / "review-notes.json").exists()
    assert (bundle / "review-fill-guide.md").exists()
    assert exported["mcp_apply_review_notes_args_path"] == str(bundle / "mcp-apply-review-notes.args.json")
    assert exported["review_notes_template_path"] == str(bundle / "review-notes.template.json")
    assert exported["review_fill_guide_path"] == str(bundle / "review-fill-guide.md")
    assert exported["mcp_acceptance_check_args_path"] == str(bundle / "mcp-acceptance-check.args.json")
    apply_args = json.loads((bundle / "mcp-apply-review-notes.args.json").read_text(encoding="utf-8"))
    assert apply_args["bundle_dir"] == str(bundle)
    assert apply_args["review_json"] == str(bundle / "review-notes.json")
    assert apply_args["write"] is True
    assert exported["mcp_controlled_execution_smoke_args_path"] == str(bundle / "mcp-controlled-execution-smoke.args.json")
    smoke_args = json.loads((bundle / "mcp-controlled-execution-smoke.args.json").read_text(encoding="utf-8"))
    assert smoke_args["provider_config"] == {"provider": "fixture"}
    assert smoke_args["execute"] is False
    readme = (bundle / "README.md").read_text(encoding="utf-8")
    assert "acceptance-check.md" in readme
    assert "extraction-audit.md" in readme
    assert "review-notes.template.json" in readme
    assert "review-fill-guide.md" in readme

    (bundle / "vision-execution-preflight.md").write_text("# preflight", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text("{}", encoding="utf-8")
    (bundle / "controlled-execution-smoke.md").write_text("# smoke", encoding="utf-8")
    (bundle / "controlled-execution-smoke.json").write_text("{}", encoding="utf-8")
    (bundle / "mcp-run-multimodal-frame-analysis-confirmed.args.json").write_text("{}", encoding="utf-8")
    manifest["vision_execution_preflight"] = "vision-execution-preflight.md"
    manifest["vision_execution_preflight_json"] = "vision-execution-preflight.json"
    manifest["controlled_execution_smoke"] = "controlled-execution-smoke.md"
    manifest["controlled_execution_smoke_json"] = "controlled-execution-smoke.json"
    manifest["mcp_multimodal_frame_analysis_confirmed_args"] = "mcp-run-multimodal-frame-analysis-confirmed.args.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    export_webui_bundle(root, output_dir=bundle)
    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert refreshed["vision_execution_preflight"] == "vision-execution-preflight.md"
    assert refreshed["vision_execution_preflight_json"] == "vision-execution-preflight.json"
    assert refreshed["controlled_execution_smoke"] == "controlled-execution-smoke.md"
    assert refreshed["controlled_execution_smoke_json"] == "controlled-execution-smoke.json"
    assert refreshed["mcp_multimodal_frame_analysis_confirmed_args"] == "mcp-run-multimodal-frame-analysis-confirmed.args.json"


def test_acceptance_bundle_execute_vision_requires_preflight_confirmation(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "run" / "webui-bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "visual_route": "semantic_frame", "frame_paths": [str(frame)]}], ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")

    import video_knowledge_pipeline.acceptance_run as acceptance_run

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(acceptance_run, "run_video_frame_router", lambda bundle_dir: {"report_path": str(bundle / "router.md")})
    monkeypatch.setattr(acceptance_run, "run_visual_structure_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "visual-structure-report.md")})
    monkeypatch.setattr(
        acceptance_run,
        "run_multimodal_frame_analysis",
        lambda bundle_dir, **kwargs: calls.append({"tool": "semantic", **kwargs}) or {"report_path": str(bundle / "multimodal-frame-analysis-report.md")},
    )
    monkeypatch.setattr(acceptance_run, "run_temporal_frame_groups", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "temporal-frame-groups-report.md")})
    monkeypatch.setattr(
        acceptance_run,
        "run_temporal_visual_analysis",
        lambda bundle_dir, **kwargs: calls.append({"tool": "temporal", **kwargs}) or {"report_path": str(bundle / "temporal-visual-analysis-report.md")},
    )
    monkeypatch.setattr(acceptance_run, "vision_acceptance_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "vision-acceptance-plan.md")})
    monkeypatch.setattr(acceptance_run, "audit_knowledge_coverage", lambda bundle_dir: {"coverage_markdown_path": str(bundle / "knowledge-coverage.md")})
    monkeypatch.setattr(acceptance_run, "export_knowledge_note", lambda bundle_dir, **kwargs: {"note_path": str(bundle / "exports" / "knowledge-note.md")})
    monkeypatch.setattr(acceptance_run, "bundle_status_report", lambda bundle_dir: {"status": "ready", "report_path": str(bundle / "bundle-status.md")})

    result = run_acceptance_bundle(
        bundle,
        output_dir=tmp_path / "acceptance",
        execute_vision=True,
        provider_config={"provider": "gemini", "model": "gemini-2.5-flash", "api_key": "secret-value"},
        semantic_limit=1,
        temporal_limit=1,
    )

    assert calls == []
    steps = {step["key"]: step for step in result["steps"]}
    assert steps["vision_execution_preflight"]["status"] == "ok"
    assert steps["multimodal_frame_analysis"]["status"] == "vision_confirmation_required"
    assert steps["multimodal_frame_analysis"]["result"]["summary"]["expected_api_calls"] == 1
    assert steps["multimodal_frame_analysis"]["result"]["summary"]["expected_indexes"] == "1"
    assert steps["multimodal_frame_analysis"]["result"]["summary"]["model_called"] is False
    assert "secret-value" not in json.dumps(result, ensure_ascii=False)
    assert "secret-value" not in Path(result["report_path"]).read_text(encoding="utf-8")


def test_acceptance_bundle_execute_vision_runs_after_matching_confirmation(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "run" / "webui-bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "visual_route": "semantic_frame", "frame_paths": [str(frame)]}], ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")

    import video_knowledge_pipeline.acceptance_run as acceptance_run

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(acceptance_run, "run_video_frame_router", lambda bundle_dir: {"report_path": str(bundle / "router.md")})
    monkeypatch.setattr(acceptance_run, "run_visual_structure_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "visual-structure-report.md")})
    monkeypatch.setattr(
        acceptance_run,
        "run_multimodal_frame_analysis",
        lambda bundle_dir, **kwargs: calls.append({"tool": "semantic", **kwargs}) or {"report_path": str(bundle / "multimodal-frame-analysis-report.md")},
    )
    monkeypatch.setattr(acceptance_run, "run_temporal_frame_groups", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "temporal-frame-groups-report.md")})
    monkeypatch.setattr(
        acceptance_run,
        "run_temporal_visual_analysis",
        lambda bundle_dir, **kwargs: calls.append({"tool": "temporal", **kwargs}) or {"report_path": str(bundle / "temporal-visual-analysis-report.md")},
    )
    monkeypatch.setattr(acceptance_run, "vision_acceptance_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "vision-acceptance-plan.md")})
    monkeypatch.setattr(acceptance_run, "audit_knowledge_coverage", lambda bundle_dir: {"coverage_markdown_path": str(bundle / "knowledge-coverage.md")})
    monkeypatch.setattr(acceptance_run, "export_knowledge_note", lambda bundle_dir, **kwargs: {"note_path": str(bundle / "exports" / "knowledge-note.md")})
    monkeypatch.setattr(acceptance_run, "bundle_status_report", lambda bundle_dir: {"status": "ready", "report_path": str(bundle / "bundle-status.md")})

    result = run_acceptance_bundle(
        bundle,
        output_dir=tmp_path / "acceptance",
        execute_vision=True,
        provider_config={"provider": "gemini", "model": "gemini-2.5-flash", "api_key": "secret-value"},
        semantic_limit=1,
        temporal_limit=1,
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
    )

    assert any(step["key"] == "vision_execution_preflight" for step in result["steps"])
    assert calls[0]["tool"] == "semantic"
    assert calls[0]["execute"] is True
    assert calls[0]["limit"] == 1
    assert calls[1]["tool"] == "temporal"
    assert calls[1]["execute"] is True
    assert calls[1]["limit"] == 1
    assert "secret-value" not in json.dumps(result, ensure_ascii=False)


def test_acceptance_bundle_preflights_after_temporal_frame_groups(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "run" / "webui-bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "visual_route": "semantic_frame", "frame_paths": [str(frame)]},
                {"index": 2, "visual_route": "temporal_sequence", "frame_paths": [str(frame)]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")

    import video_knowledge_pipeline.acceptance_run as acceptance_run

    monkeypatch.setattr(acceptance_run, "run_video_frame_router", lambda bundle_dir: {"report_path": str(bundle / "router.md")})
    monkeypatch.setattr(acceptance_run, "run_visual_structure_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "visual-structure-report.md")})
    monkeypatch.setattr(acceptance_run, "run_multimodal_frame_analysis", lambda bundle_dir, **kwargs: (_ for _ in ()).throw(AssertionError("vision should be gated")))
    monkeypatch.setattr(acceptance_run, "run_temporal_visual_analysis", lambda bundle_dir, **kwargs: (_ for _ in ()).throw(AssertionError("vision should be gated")))

    def fake_temporal_groups(bundle_dir, **kwargs):
        timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
        timeline[1]["temporal_frame_paths"] = [str(frame), str(frame)]
        (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
        return {"report_path": str(bundle / "temporal-frame-groups-report.md"), "summary": {"execute": kwargs["execute"], "updated": 1}}

    monkeypatch.setattr(acceptance_run, "run_temporal_frame_groups", fake_temporal_groups)
    monkeypatch.setattr(acceptance_run, "vision_acceptance_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "vision-acceptance-plan.md")})
    monkeypatch.setattr(acceptance_run, "audit_knowledge_coverage", lambda bundle_dir: {"coverage_markdown_path": str(bundle / "knowledge-coverage.md")})
    monkeypatch.setattr(acceptance_run, "export_knowledge_note", lambda bundle_dir, **kwargs: {"note_path": str(bundle / "exports" / "knowledge-note.md")})
    monkeypatch.setattr(acceptance_run, "bundle_status_report", lambda bundle_dir: {"status": "ready", "report_path": str(bundle / "bundle-status.md")})

    result = run_acceptance_bundle(
        bundle,
        output_dir=tmp_path / "acceptance",
        execute_temporal_groups=True,
        execute_vision=True,
        provider_config={"provider": "gemini", "model": "gemini-2.5-flash", "api_key": "secret-value"},
        semantic_limit=1,
        temporal_limit=1,
    )

    keys = [step["key"] for step in result["steps"]]
    assert keys.index("temporal_frame_groups") < keys.index("vision_execution_preflight")
    steps = {step["key"]: step for step in result["steps"]}
    assert steps["multimodal_frame_analysis"]["status"] == "vision_confirmation_required"
    assert steps["multimodal_frame_analysis"]["result"]["summary"]["expected_api_calls"] == 2
    assert steps["multimodal_frame_analysis"]["result"]["summary"]["expected_indexes"] == "1,2"
    assert steps["temporal_frame_groups"]["result"]["summary"]["execute"] is True


def test_temporal_frame_groups_limit_and_temporal_visual_requires_group(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"fake video")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"start": 0, "end": 4, "visual_route": "mixed", "video_key": str(video), "frame_paths": [str(frame)]},
                {"start": 4, "end": 8, "visual_route": "mixed", "video_key": str(video), "frame_paths": [str(frame)]},
                {"start": 8, "end": 12, "video_key": str(video), "frame_paths": [str(frame)]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    groups = run_temporal_frame_groups(bundle, limit=1)
    assert groups["summary"]["total"] == 1
    assert groups["summary"]["limit"] == 1

    indexed_groups = run_temporal_frame_groups(bundle, indexes=[2])
    assert indexed_groups["summary"]["total"] == 1
    assert indexed_groups["summary"]["indexes"] == [2]
    assert indexed_groups["items"][0]["index"] == 2

    explicit_non_temporal = run_temporal_frame_groups(bundle, indexes=[3])
    assert explicit_non_temporal["summary"]["total"] == 1
    assert explicit_non_temporal["items"][0]["index"] == 3
    assert explicit_non_temporal["items"][0]["visual_route"] == "unknown"
    assert "unknown" in explicit_non_temporal["summary"]["include_routes"]

    preview = run_temporal_visual_analysis(bundle)
    assert preview["summary"]["total"] == 0

    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["temporal_frame_paths"] = [str(frame), str(frame)]
    timeline[1]["temporal_frame_paths"] = [str(frame), str(frame)]
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    preview = run_temporal_visual_analysis(bundle, limit=1)
    assert preview["summary"]["total"] == 1
    assert preview["summary"]["limit"] == 1
    assert preview["items"][0]["index"] == 1
    assert preview["run_registry"]["run_type"] == "temporal_visual_analysis"
    assert preview["run_registry"]["status"] == "needs_execution"
    run = read_json(bundle / "runs" / "temporal-visual-analysis" / "run.json")
    assert run["retry_command"].startswith(".\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis")


def test_bundle_next_action_generates_temporal_groups_before_temporal_analysis(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"fake video")
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
                    "transcript": "老师演示连续操作。",
                    "visual_route": "temporal_sequence",
                    "visual_text": "操作界面",
                    "video_key": str(video),
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
    assert action["key"] == "temporal_frame_groups"
    assert action["mcp_tool"] == "run_temporal_frame_groups"
    assert action["human_required"] is False

    advanced = bundle_advance(bundle, execute=False, temporal_limit=1, frame_count=8)
    assert advanced["status"] == "advanced"
    assert advanced["action_result"]["summary"]["schema"] == "lecture_temporal_frame_groups_summary.v1"
    assert advanced["action_result"]["summary"]["total"] == 1
    assert advanced["action_result"]["summary"]["limit"] == 1


def test_vision_acceptance_plan_masks_key_and_selects_small_batch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = []
    for index in range(1, 25):
        route = "semantic_frame" if index <= 20 else "temporal_sequence"
        item = {"start": index, "end": index + 1, "visual_route": route, "frame_paths": [str(frame)]}
        if route == "temporal_sequence":
            item["temporal_frame_paths"] = [str(frame), str(frame)]
        timeline.append(item)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    plan = vision_acceptance_plan(
        bundle,
        provider_config={"provider": "gemini", "api_key": "secret-value", "model": "gemini-2.5-flash"},
        semantic_limit=19,
        temporal_limit=3,
    )

    assert plan["provider"]["api_key_configured"] is True
    assert "secret-value" not in json.dumps(plan, ensure_ascii=False)
    assert plan["candidate_counts"]["semantic_selected"] == 19
    assert plan["candidate_counts"]["temporal_selected"] == 3
    assert plan["selected_indexes"]["semantic"] == list(range(1, 20))
    assert plan["selected_indexes"]["temporal"] == [21, 22, 23]
    commands = plan["commands"]
    assert commands["test_provider"].startswith(".\\scripts\\video-knowledge.ps1 test-vision-provider")
    assert "python -m video_knowledge_pipeline.cli" not in json.dumps(commands, ensure_ascii=False)
    assert "set_agnes_env_example" in commands
    assert "set_openai_env_example" in commands
    assert "--confirm-vision-calls 19" in commands["run_semantic_acceptance"]
    assert "--confirm-vision-indexes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" in commands["run_semantic_acceptance"]
    assert "--confirm-vision-calls 3" in commands["run_temporal_acceptance"]
    assert "--confirm-vision-indexes 21,22,23" in commands["run_temporal_acceptance"]
    assert "secret-value" not in json.dumps(commands, ensure_ascii=False)


def test_vision_execution_preflight_checks_readiness_and_restore_chain(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {"index": 1, "visual_route": "semantic_frame", "assets": [{"path": str(frame)}]},
        {"index": 2, "visual_route": "semantic_frame", "frame_paths": [str(frame)], "visual_understanding": {"objects": ["done"]}},
        {"index": 3, "visual_route": "temporal_sequence", "frame_paths": [str(frame)], "temporal_frame_paths": [str(frame), str(frame)]},
        {"index": 4, "visual_route": "temporal_sequence", "frame_paths": [str(frame)]},
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    missing_key = vision_execution_preflight(bundle, provider_config={"provider": "gemini", "api_key": ""}, semantic_limit=1, temporal_limit=1, frame_count=8)
    assert missing_key["ready_to_execute"] is False
    assert missing_key["expected_api_calls"] == 2
    assert missing_key["candidate_counts"]["semantic_available"] == 1
    assert missing_key["candidate_counts"]["temporal_available"] == 1
    assert missing_key["candidate_counts"]["temporal_without_frame_groups"] == 1
    assert {item["key"] for item in missing_key["blockers"]} == {"missing_api_key"}
    assert Path(missing_key["preflight_path"]).exists()
    assert Path(missing_key["preflight_json_path"]).exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vision_execution_preflight"] == "vision-execution-preflight.md"
    assert manifest["mcp_vision_execution_preflight_args"] == "mcp-vision-execution-preflight.args.json"

    ready = vision_execution_preflight(
        bundle,
        provider_config={"provider": "gemini", "api_key": "secret-value", "model": "gemini-2.5-flash"},
        semantic_limit=1,
        temporal_limit=1,
    )
    assert ready["ready_to_execute"] is True
    assert ready["provider"]["api_key_configured"] is True
    assert ready["writes"]["semantic_fields"][0] == "visual_understanding"
    assert ready["restore_chain"]["diff_audit_available"] is True
    assert "secret-value" not in json.dumps(ready, ensure_ascii=False)
    assert "secret-value" not in Path(ready["preflight_json_path"]).read_text(encoding="utf-8")
    assert ready["confirmation"]["semantic_confirm_vision_calls"] == 1
    assert ready["confirmation"]["semantic_confirm_vision_indexes"] == "1"
    assert ready["confirmation"]["temporal_confirm_vision_calls"] == 1
    assert ready["confirmation"]["temporal_confirm_vision_indexes"] == "3"
    assert "--confirm-vision-calls 1" in ready["commands"]["confirmed_run_semantic"]
    assert '--confirm-vision-indexes "1"' in ready["commands"]["confirmed_run_semantic"]
    assert "--confirm-vision-calls 1" in ready["commands"]["confirmed_run_temporal"]
    assert '--confirm-vision-indexes "3"' in ready["commands"]["confirmed_run_temporal"]
    ready_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert "mcp_bundle_advance_confirmed_args" not in ready_manifest
    assert ready_manifest["mcp_multimodal_frame_analysis_confirmed_args"] == "mcp-run-multimodal-frame-analysis-confirmed.args.json"
    assert ready_manifest["mcp_temporal_visual_analysis_confirmed_args"] == "mcp-run-temporal-visual-analysis-confirmed.args.json"
    semantic_args = json.loads((bundle / "mcp-run-multimodal-frame-analysis-confirmed.args.json").read_text(encoding="utf-8"))
    temporal_args = json.loads((bundle / "mcp-run-temporal-visual-analysis-confirmed.args.json").read_text(encoding="utf-8"))
    assert semantic_args["provider_config"]["provider"] == "gemini"
    assert semantic_args["provider_config"]["model"] == "gemini-2.5-flash"
    assert "api_key" not in semantic_args["provider_config"]
    assert temporal_args["provider_config"]["provider"] == "gemini"
    assert temporal_args["provider_config"]["model"] == "gemini-2.5-flash"
    assert "api_key" not in temporal_args["provider_config"]
    assert semantic_args["indexes"] == [1]
    assert semantic_args["confirm_vision_calls"] == 1
    assert semantic_args["confirm_vision_indexes"] == "1"
    assert temporal_args["indexes"] == [3]
    assert temporal_args["confirm_vision_calls"] == 1
    assert temporal_args["confirm_vision_indexes"] == "3"
    assert "secret-value" not in json.dumps([semantic_args, temporal_args], ensure_ascii=False)
    assert ready["confirmed_mcp_args"]["mcp_multimodal_frame_analysis_confirmed_args"]["mcp_tool"] == "run_multimodal_frame_analysis"
    assert ready["confirmed_mcp_args"]["mcp_temporal_visual_analysis_confirmed_args"]["mcp_tool"] == "run_temporal_visual_analysis"
    ready_json = json.loads(Path(ready["preflight_json_path"]).read_text(encoding="utf-8"))
    ready_markdown = Path(ready["preflight_path"]).read_text(encoding="utf-8")
    assert ready_json["preflight_path"] == ready["preflight_path"]
    assert "Confirmed MCP Args" in ready_markdown
    assert "mcp-run-multimodal-frame-analysis-confirmed.args.json" in ready_markdown
    assert "mcp-run-temporal-visual-analysis-confirmed.args.json" in ready_markdown
    assert "secret-value" not in json.dumps(ready_json, ensure_ascii=False)
    assert "secret-value" not in ready_markdown

    custom_url = vision_execution_preflight(
        bundle,
        provider_config={
            "provider": "custom_openai_compatible",
            "api_key": "secret-value",
            "model": "custom-vlm",
            "base_url": "https://example.invalid/v1/chat/completions?key=secret-value",
        },
        semantic_limit=1,
        temporal_limit=0,
        include_temporal=False,
    )
    custom_args = json.loads((bundle / "mcp-run-multimodal-frame-analysis-confirmed.args.json").read_text(encoding="utf-8"))
    assert custom_url["ready_to_execute"] is True
    assert custom_args["provider_config"]["provider"] == "custom_openai_compatible"
    assert custom_args["provider_config"]["model"] == "custom-vlm"
    assert "base_url" not in custom_args["provider_config"]
    assert "secret-value" not in json.dumps(custom_args, ensure_ascii=False)

    indexed = vision_execution_preflight(
        bundle,
        provider_config={"provider": "gemini", "api_key": "secret-value", "model": "gemini-2.5-flash"},
        semantic_limit=0,
        temporal_limit=0,
        semantic_indexes=[1],
        temporal_indexes=[3],
    )
    assert indexed["execution_profile"]["semantic_indexes"] == [1]
    assert indexed["execution_profile"]["temporal_indexes"] == [3]
    assert indexed["selected_indexes"]["semantic"] == [1]
    assert indexed["selected_indexes"]["temporal"] == [3]
    assert indexed["confirmation"]["confirm_vision_indexes"] == "1,3"

    semantic_only = vision_execution_preflight(
        bundle,
        provider_config={"provider": "gemini", "api_key": "secret-value", "model": "gemini-2.5-flash"},
        semantic_limit=0,
        temporal_limit=0,
        include_temporal=False,
        semantic_indexes=[1],
    )
    assert semantic_only["execution_profile"]["include_temporal"] is False
    assert semantic_only["selected_indexes"]["semantic"] == [1]
    assert semantic_only["selected_indexes"]["temporal"] == []
    assert semantic_only["confirmation"]["confirm_vision_indexes"] == "1"


def test_bundle_status_reports_controlled_execution_chain(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "controlled",
                "vision_execution_preflight": "vision-execution-preflight.md",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
                "mcp_multimodal_frame_analysis_confirmed_args": "mcp-run-multimodal-frame-analysis-confirmed.args.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "vision-execution-preflight.md").write_text("# preflight", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text("{}", encoding="utf-8")
    (bundle / "mcp-run-multimodal-frame-analysis-confirmed.args.json").write_text(
        json.dumps({"bundle_dir": str(bundle), "execute": True, "indexes": [1]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps({"run_id": "semantic_frame-1", "kind": "semantic_frame"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (bundle / "bundle-advance-runs.jsonl").write_text(
        json.dumps(
            {
                "created_at": "2026-06-06T00:00:00",
                "status": "blocked",
                "blocked_reason": "vision execution confirmation required; match confirm_vision_calls and confirm_vision_indexes from preflight",
                "action_summary": {"blocker_keys": []},
                "action_artifacts": {
                    "preflight_path": str(bundle / "vision-execution-preflight.md"),
                    "preflight_json_path": str(bundle / "vision-execution-preflight.json"),
                    "vision_run_id": "semantic_frame-1",
                    "vision_restore_plan_command": "python -m video_knowledge_pipeline.cli vision-analysis-restore-plan bundle --run-id semantic_frame-1",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    status = bundle_status_report(bundle, refresh=False)

    controlled = status["controlled_execution"]
    assert controlled["status"] == "blocked"
    assert controlled["preflight_status"] == "present"
    assert controlled["confirmation_status"] == "required"
    assert controlled["audit_status"] == "present"
    assert controlled["latest_vision_run_id"] == "semantic_frame-1"
    assert "confirmation_required" in controlled["blockers"]
    assert "vision-analysis-restore-plan" in controlled["restore_plan_command"]
    assert any(
        row["key"] == "mcp_multimodal_frame_analysis_confirmed_args"
        and row["mcp_tool"] == "run_multimodal_frame_analysis"
        and row["exists"]
        for row in status["mcp_commands"]
    )
    markdown = Path(status["report_markdown_path"]).read_text(encoding="utf-8")
    assert "可控真实执行" in markdown
    assert "semantic_frame-1" in markdown

    check = controlled_execution_check(bundle, refresh=False)
    assert check["ready_for_real_vision_execution"] is False
    assert check["status"] == "blocked"
    assert any(item["key"] == "batch_confirmed_or_not_pending" and not item["ok"] for item in check["checklist"])
    assert Path(check["report_path"]).exists()
    assert Path(check["report_markdown_path"]).exists()
    assert "confirm_vision_calls" in " ".join(check["next_steps"])


def test_bundle_status_reports_direct_vision_confirmation_gate(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-direct-gate"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "direct-gate",
                "vision_execution_preflight": "vision-execution-preflight.md",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "vision-execution-preflight.md").write_text("# preflight", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text("{}", encoding="utf-8")
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "semantic_frame-direct-gate",
                "kind": "semantic_frame",
                "execute": True,
                "status": "vision_confirmation_required",
                "updated_count": 0,
                "execution_control": {
                    "status": "vision_confirmation_required",
                    "confirmed": False,
                    "expected_api_calls": 1,
                    "expected_indexes": "5",
                    "received_confirm_vision_calls": "",
                    "received_confirm_vision_indexes": "",
                    "preflight_path": str(bundle / "vision-execution-preflight.md"),
                    "preflight_json_path": str(bundle / "vision-execution-preflight.json"),
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    status = bundle_status_report(bundle, refresh=False)

    controlled = status["controlled_execution"]
    assert controlled["status"] == "blocked"
    assert status["mcp_args_audit"]["status"] == "ok"
    assert controlled["confirmation_status"] == "required"
    assert controlled["latest_execution_control_status"] == "vision_confirmation_required"
    assert controlled["latest_execution_control_confirmed"] is False
    assert controlled["latest_execution_control_expected_indexes"] == "5"
    assert controlled["latest_vision_run_id"] == "semantic_frame-direct-gate"
    assert "confirmation_required" in controlled["blockers"]
    markdown = Path(status["report_markdown_path"]).read_text(encoding="utf-8")
    assert "确认值：calls `1`，indexes `5`" in markdown
    assert "最近写入：updated `0`，changed `0`，recoverable `False`" in markdown

    check = controlled_execution_check(bundle, refresh=False)
    assert check["ready_for_real_vision_execution"] is False
    assert check["mcp_args_audit"]["status"] == "ok"
    assert any(
        item["key"] == "batch_confirmed_or_not_pending"
        and not item["ok"]
        and item["detail"] == "vision_confirmation_required"
        for item in check["checklist"]
    )
    assert "same vision execution command" in " ".join(check["next_steps"])
    check_markdown = Path(check["report_markdown_path"]).read_text(encoding="utf-8")
    assert "Confirm calls: `1`" in check_markdown
    assert "Confirm indexes: `5`" in check_markdown
    assert "Latest write: updated `0`, changed `0`" in check_markdown


def test_controlled_execution_check_ready_when_chain_has_recoverable_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-ready"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "vision_execution_preflight": "vision-execution-preflight.md",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "vision-execution-preflight.md").write_text("# preflight", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text("{}", encoding="utf-8")
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "semantic_frame-ready",
                "kind": "semantic_frame",
                "execute": True,
                "status": "ok",
                "updated_count": 1,
                "timeline_diff_count": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "vision-restore-plan.json").write_text("{}", encoding="utf-8")
    (bundle / "mcp-bundle-status-report.args.json").write_text(
        json.dumps({"bundle_dir": str(bundle), "refresh": False, "write": True}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "bundle-advance-runs.jsonl").write_text(
        json.dumps(
            {
                "created_at": "2026-06-06T00:00:00",
                "status": "advanced",
                "blocked_reason": "",
                "action_artifacts": {
                    "vision_run_id": "semantic_frame-ready",
                    "vision_restore_plan_command": "python -m video_knowledge_pipeline.cli vision-analysis-restore-plan bundle --run-id semantic_frame-ready",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    check = controlled_execution_check(bundle, refresh=False)

    assert check["ready_for_real_vision_execution"] is True
    assert check["mcp_args_audit"]["status"] == "ok"
    assert check["status"] == "ready"
    assert all(item["ok"] for item in check["checklist"])
    assert check["next_steps"] == ["Controlled execution chain is ready."]


def test_controlled_execution_check_blocks_bad_mcp_args(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-bad-mcp"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "vision_execution_preflight": "vision-execution-preflight.md",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
                "mcp_unknown_args": "mcp-unknown.args.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "vision-execution-preflight.md").write_text("# preflight", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text("{}", encoding="utf-8")
    (bundle / "mcp-unknown.args.json").write_text(json.dumps({"bundle_dir": str(bundle)}, ensure_ascii=False), encoding="utf-8")
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "semantic_frame-ready",
                "kind": "semantic_frame",
                "execute": True,
                "status": "ok",
                "updated_count": 1,
                "timeline_diff_count": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "vision-restore-plan.json").write_text("{}", encoding="utf-8")

    check = controlled_execution_check(bundle, refresh=False)

    assert check["ready_for_real_vision_execution"] is False
    assert check["mcp_args_audit"]["status"] == "blocked"
    assert any(item["key"] == "mcp_args_usable" and not item["ok"] for item in check["checklist"])
    assert "mcp-audit-bundle" in " ".join(check["next_steps"])


def test_controlled_execution_check_blocks_empty_confirmed_vision_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-empty-run"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "vision_execution_preflight": "vision-execution-preflight.md",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "vision-execution-preflight.md").write_text("# preflight", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text("{}", encoding="utf-8")
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "semantic_frame-empty",
                "kind": "semantic_frame",
                "execute": True,
                "status": "ok",
                "updated_count": 0,
                "timeline_diff_count": 0,
                "execution_control": {"status": "confirmed", "confirmed": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "vision-restore-plan.json").write_text("{}", encoding="utf-8")

    check = controlled_execution_check(bundle, refresh=False)

    assert check["ready_for_real_vision_execution"] is False
    assert "no_recoverable_vision_write" in check["controlled_execution"]["blockers"]
    assert any(
        item["key"] == "recoverable_vision_write_available" and not item["ok"] for item in check["checklist"]
    )
    assert "writes controlled timeline fields" in " ".join(check["next_steps"])


def test_controlled_execution_check_ready_with_direct_confirmed_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-direct-ready"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "vision_execution_preflight": "vision-execution-preflight.md",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "vision-execution-preflight.md").write_text("# preflight", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text("{}", encoding="utf-8")
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "semantic_frame-direct-ready",
                "kind": "semantic_frame",
                "execute": True,
                "status": "ok",
                "updated_count": 1,
                "execution_control": {
                    "status": "ready",
                    "confirmed": True,
                    "expected_api_calls": 1,
                    "expected_indexes": "5",
                    "received_confirm_vision_calls": 1,
                    "received_confirm_vision_indexes": "5",
                    "preflight_path": str(bundle / "vision-execution-preflight.md"),
                    "preflight_json_path": str(bundle / "vision-execution-preflight.json"),
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "vision-restore-plan.json").write_text("{}", encoding="utf-8")

    status = bundle_status_report(bundle, refresh=False)
    controlled = status["controlled_execution"]
    assert controlled["confirmation_status"] == "confirmed"
    assert controlled["latest_execution_control_confirmed"] is True
    assert controlled["latest_vision_run_id"] == "semantic_frame-direct-ready"
    assert controlled["status"] == "vision_run_recoverable"

    check = controlled_execution_check(bundle, refresh=False)
    assert check["ready_for_real_vision_execution"] is True
    assert all(item["ok"] for item in check["checklist"])


def test_acceptance_and_vision_plan_use_unified_execution_profile(monkeypatch, tmp_path: Path) -> None:
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
                    "multimodal_limit": 5,
                    "temporal_limit": 2,
                    "frame_count": 7,
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
    timeline = []
    for index in range(1, 12):
        route = "semantic_frame" if index <= 8 else "temporal_sequence"
        item = {"index": index, "start": index, "end": index + 1, "visual_route": route, "frame_paths": [str(frame)]}
        if route == "temporal_sequence":
            item["temporal_frame_paths"] = [str(frame), str(frame)]
        timeline.append(item)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    plan = vision_acceptance_plan(bundle)
    result = run_acceptance_bundle(bundle, output_dir=tmp_path / "acceptance")

    assert plan["targets"]["semantic_limit"] == 5
    assert plan["targets"]["temporal_limit"] == 2
    assert plan["targets"]["frame_count"] == 7
    assert plan["candidate_counts"]["semantic_selected"] == 5
    assert plan["candidate_counts"]["temporal_selected"] == 2
    assert result["execution_profile"]["semantic_limit"] == 5
    assert result["execution_profile"]["temporal_limit"] == 2
    assert result["execution_profile"]["frame_count"] == 7
    assert "GEMINI_API_KEY" not in json.dumps(result, ensure_ascii=False)





def test_vision_review_triage_default_selects_only_risky_items(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 5,
            "visual_route": "semantic_frame",
            "transcript": "这里看屏幕右边价格是399元，工具叫Playwright client。",
            "visual_text": "价格 299 元 Playwrite client",
            "frame_paths": [str(frame)],
        },
        {
            "index": 2,
            "start": 5,
            "end": 10,
            "visual_route": "temporal_sequence",
            "transcript": "现在打开后台，点击提交审核按钮。",
            "visual_text": "后台 审核",
            "frame_paths": [str(frame)],
            "quality_issues": ["temporal_sequence_without_analysis"],
        },
        {
            "index": 3,
            "start": 10,
            "end": 15,
            "visual_route": "document_visual",
            "transcript": "这个表格有三步。",
            "visual_text": "",
            "frame_paths": [str(frame)],
            "quality_issues": ["ocr_text_empty"],
        },
        {
            "index": 4,
            "start": 15,
            "end": 20,
            "visual_route": "semantic_frame",
            "transcript": "这一段只是普通讲述。",
            "visual_text": "普通讲述",
            "frame_paths": [str(frame)],
            "visual_understanding": {"summary": "already done"},
        },
    ]
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", timeline)

    result = vision_review_triage(bundle)

    assert result["mode"] == "triage"
    assert 1 in result["semantic_indexes"]
    assert 2 not in result["temporal_indexes"]
    assert 2 in result["temporal_recapture_indexes"]
    assert 3 in result["visual_structure_first_indexes"]
    assert 4 not in result["semantic_indexes"]
    assert (bundle / "vision-review-triage.json").exists()
    assert (bundle / "vision-review-triage.md").exists()
    assert (bundle / "mcp-vision-review-triage-preflight.args.json").exists()
    manifest = read_json(bundle / "manifest.json")
    assert manifest["mcp_vision_review_triage_args"] == "mcp-vision-review-triage.args.json"
    preflight = read_json(bundle / "mcp-vision-review-triage-preflight.args.json")
    assert preflight["semantic_indexes"] == result["semantic_indexes"]
    assert preflight["temporal_indexes"] == result["temporal_indexes"]


def test_vision_review_triage_full_mode_selects_all_frame_items(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {"index": 1, "visual_route": "semantic_frame", "transcript": "普通讲述", "visual_text": "普通", "frame_paths": [str(frame)]},
        {"index": 2, "visual_route": "document_visual", "transcript": "课件", "visual_text": "课件", "frame_paths": [str(frame)]},
        {"index": 3, "visual_route": "temporal_sequence", "transcript": "点击按钮", "visual_text": "按钮", "frame_paths": [str(frame)]},
        {"index": 4, "visual_route": "semantic_frame", "transcript": "无图", "visual_text": "无图"},
    ]
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", timeline)

    result = vision_review_triage(bundle, mode="full", write=False)

    assert result["mode"] == "full"
    assert result["semantic_indexes"] == [1, 2, 3]
    assert result["visual_structure_first_indexes"] == [1, 2, 3]
    assert result["temporal_indexes"] == [3]
    assert 4 not in result["semantic_indexes"]




def test_vision_review_triage_uses_external_tagger_json_weights(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    timeline = [
        {"index": 1, "start": 0, "end": 5, "visual_route": "semantic_frame", "transcript": "plain", "visual_text": "plain", "frame_paths": [str(frame)]},
        {"index": 2, "start": 10, "end": 15, "visual_route": "unknown", "transcript": "plain", "visual_text": "plain", "frame_paths": [str(frame)]},
        {"index": 3, "start": 30, "end": 35, "visual_route": "unknown", "transcript": "plain", "visual_text": "plain", "frame_paths": [str(frame)]},
        {"index": 4, "start": 40, "end": 45, "visual_route": "unknown", "transcript": "plain", "visual_text": "plain", "frame_paths": [str(frame)]},
    ]
    tagger = {
        "annotations": [
            {"index": 1, "tags": ["\u95f2\u804a", "\u8fc7\u6e21"]},
            {"index": 2, "tags": ["\u7591\u96be", "\u5de5\u5177\u540d"]},
            {"time": 32, "tags": ["\u64cd\u4f5c"]},
            {"index": 4, "tags": ["\u8868\u683c"]},
        ]
    }
    tagger_path = bundle / "qinglong-tags.json"
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", timeline)
    write_json(tagger_path, tagger)

    result = vision_review_triage(bundle, tagger_json=tagger_path, min_score=2)

    assert result["tagger_annotations_count"] == 4
    assert result["tagger_indexes"] == [1, 2, 3, 4]
    assert 1 not in result["semantic_indexes"]
    assert 2 in result["semantic_indexes"]
    assert 3 not in result["temporal_indexes"]
    assert 3 in result["temporal_recapture_indexes"]
    assert 4 in result["visual_structure_first_indexes"]
    row1 = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert "tagged_as_low_value_or_repetitive" in row1["tag_suppression_reasons"]


def test_import_tagger_annotations_updates_unified_timeline(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    timeline = [
        {"index": 1, "start": 0, "end": 5, "visual_route": "semantic_frame", "frame_paths": ["frame1.jpg"]},
        {"index": 2, "start": 10, "end": 20, "visual_route": "unknown", "frame_paths": ["frame2.jpg"]},
    ]
    tagger = {
        "annotations": [
            {"time": 12, "tags": ["\u64cd\u4f5c", "\u6b65\u9aa4"], "text": "\u6253\u5f00\u540e\u53f0\u5e76\u70b9\u51fb\u5ba1\u6838\u6309\u94ae"},
            {"index": 1, "tags": ["\u95f2\u804a", "\u8fc7\u6e21"], "text": "\u8fc7\u6e21\u8bed"},
        ]
    }
    tagger_path = bundle / "qinglong-tags.json"
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", timeline)
    write_json(tagger_path, tagger)

    result = import_tagger_annotations(bundle, tagger_path)

    assert result["updated_indexes"] == [1, 2]
    updated = read_json(bundle / "timeline.json")
    assert updated[0]["tagger_tags"] == ["\u95f2\u804a", "\u8fc7\u6e21"]
    assert updated[1]["tagger_tags"] == ["\u64cd\u4f5c", "\u6b65\u9aa4"]
    assert updated[1]["tagger_visual_summary"] == "\u6253\u5f00\u540e\u53f0\u5e76\u70b9\u51fb\u5ba1\u6838\u6309\u94ae"
    assert updated[1]["integrated_visual"]["tagger_visual_summary"] == updated[1]["tagger_visual_summary"]
    assert updated[1]["tagger_time_axis"][0]["time"] == 12
    manifest = read_json(bundle / "manifest.json")
    assert manifest["mcp_import_tagger_annotations_args"] == "mcp-import-tagger-annotations.args.json"
    assert (bundle / "tagger-import-report.md").exists()


def test_resolve_terms_reconciles_asr_ocr_metadata_and_glossary(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "frame1.jpg").write_bytes(b"test-frame")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 5,
            "transcript": "browser base 是唯一真神",
            "subtitle": "browser bees 是唯一真神",
            "visual_text": "Browserbase 控制已登录浏览器",
            "visual_route": "semantic_frame",
            "frame_paths": ["frame1.jpg"],
        },
        {
            "index": 2,
            "start": 5,
            "end": 10,
            "transcript": "u i tars 可以操作任意 app",
            "visual_text": "UI-TARS 7B",
            "structured_visual": [{"markdown": "| Tool | UI-TARS |"}],
            "tagger_tags": ["工具名"],
        },
    ]
    metadata = {"title": "浏览器自动化横评", "description": "Browserbase, UI-TARS, Playwright MCP"}
    glossary = {
        "terms": [
            {"canonical": "Browserbase", "aliases": ["browser base", "browser bees", "browserbase"]},
            {"canonical": "UI-TARS", "aliases": ["ui tars", "u i tars", "uitars"]},
            {"canonical": "Playwright MCP", "aliases": ["playwright mcp"]},
        ]
    }
    metadata_path = bundle / "metadata.json"
    glossary_path = bundle / "glossary.json"
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", timeline)
    write_json(metadata_path, metadata)
    write_json(glossary_path, glossary)

    result = resolve_terms(bundle, metadata_json=metadata_path, glossary_json=glossary_path)

    terms = {term["canonical_term"]: term for term in result["terms"]}
    assert "Browserbase" in terms
    assert "browser base" in [value.lower() for value in terms["Browserbase"]["raw_mentions"]]
    assert terms["Browserbase"]["needs_human_review"] is True
    assert "UI-TARS" in terms
    updated = read_json(bundle / "timeline.json")
    assert any(row["canonical_term"] == "Browserbase" for row in updated[0]["term_candidates"])
    assert any(row["canonical_term"] == "UI-TARS" for row in updated[1]["term_candidates"])
    assert any(row["canonical_term"] == "Browserbase" for row in updated[0]["integrated_visual"]["term_candidates"])
    assert updated[0]["corrected_transcript"] == "Browserbase 是唯一真神"
    assert updated[1]["corrected_transcript"] == "UI-TARS 可以操作任意 app"
    assert (bundle / "corrected-transcript.json").exists()
    assert (bundle / "corrected-transcript.md").exists()
    assert (bundle / "corrected-transcript.srt").exists()
    triage = vision_review_triage(bundle, min_score=2, write=False)
    assert 1 in triage["semantic_indexes"]
    exported = export_knowledge_note(bundle, title="Term Test")
    note = Path(exported["note_path"]).read_text(encoding="utf-8")
    transcript = Path(exported["full_transcript_path"]).read_text(encoding="utf-8")
    assert "Browserbase" in note
    assert "Browserbase 是唯一真神" in note
    assert "UI-TARS 可以操作任意 app" in note
    assert "browser base 是唯一真神" not in note
    assert "u i tars 可以操作任意 app" not in note
    assert "已自动采用" in note
    assert "术语仲裁" in transcript
    assert "Browserbase 是唯一真神" in transcript
    assert "UI-TARS 可以操作任意 app" in transcript
    scripts = Path(exported["content_assets"]["short_video_script_drafts_path"]).read_text(encoding="utf-8")
    assert "UI-TARS 可以操作任意 app" in scripts
    assert "u i tars 可以操作任意 app" not in scripts
    manifest = read_json(bundle / "manifest.json")
    assert manifest["term_resolution"] == "term-resolution.json"
    assert manifest["mcp_resolve_terms_args"] == "mcp-resolve-terms.args.json"
    assert (bundle / "term-glossary.md").exists()
    assert (bundle / "term-review.md").exists()
