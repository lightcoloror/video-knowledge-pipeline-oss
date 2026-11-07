from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from video_knowledge_pipeline.config import DEFAULT_LOCAL_FRAME_BUDGET, DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS, DEFAULT_LOCAL_FRAME_SAMPLING_MODE, asr_runtime_profile, config_status, ebook_pipeline_profile, model_api_settings_status, resolve_vision_execution_profile, service_url, set_asr_runtime_profile, set_vision_execution_profile, vision_execution_profile
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

def test_bundle_json_write_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "bundle" / "manifest.json"

    write_json(target, {"schema": "test.v1", "value": "初始"})
    write_json(target, {"schema": "test.v1", "value": "更新"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"schema": "test.v1", "value": "更新"}
    assert not list(target.parent.glob(".manifest.json.*.tmp"))


def test_bundle_write_lock_rejects_overlapping_writer(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / ".bundle-write.lock").write_text('{"pid":999999,"operation":"other"}', encoding="utf-8")

    try:
        with bundle_write_lock(bundle, operation="second"):
            raise AssertionError("second lock unexpectedly acquired")
    except RuntimeError as exc:
        assert "bundle_write_lock_busy" in str(exc)

    assert (bundle / ".bundle-write.lock").exists()


def test_unified_config_status_and_service_url(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 9876, "path": "/call"},
                    "openclaw_http": {"host": "127.0.0.1", "port": 8931, "path": "/call", "docker_host": "host.docker.internal"},
                    "mcp": {"transport": "stdio"},
                },
                "vision_execution": {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "multimodal_limit": 7,
                    "temporal_limit": 2,
                    "frame_count": 6,
                },
                "ebook_pipeline": {
                    "execute_default": True,
                    "include_routes": ["document_visual", "mixed"],
                    "limit": 12,
                    "timeout_seconds": 240,
                    "rapidocr_device": "cuda",
                    "rapidocr_cuda_device_id": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    status = config_status()

    assert status["ok"] is True
    assert status["config_path"] == str(config_path.resolve())
    assert status["service_urls"]["ebook_markdown_pipeline_http"] == "http://127.0.0.1:9876/call"
    assert status["service_urls"]["openclaw_http"] == "http://127.0.0.1:8931/call"
    assert status["service_urls"]["openclaw_http_docker"] == "http://host.docker.internal:8931/call"
    assert status["validation"]["ok"] is True
    assert service_url("ebook_markdown_pipeline_http") == "http://127.0.0.1:9876/call"
    assert service_url("openclaw_http") == "http://127.0.0.1:8931/call"
    assert status["vision_execution"]["provider"] == "gemini"
    assert status["vision_execution"]["multimodal_limit"] == 7
    assert vision_execution_profile()["frame_count"] == 6
    assert status["ebook_pipeline"]["execute_default"] is True
    assert ebook_pipeline_profile()["include_routes"] == ["document_visual", "mixed"]
    assert ebook_pipeline_profile()["limit"] == 12
    assert ebook_pipeline_profile()["rapidocr_device"] == "cuda"
    assert ebook_pipeline_profile()["rapidocr_cuda_device_id"] == 1

def test_config_status_explicit_path_and_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 70000, "path": "call"},
                    "openclaw_http": {"host": "", "port": 0, "path": "call"},
                    "mcp": {"transport": "stdio"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    status = config_status(config_path)

    assert status["ok"] is False
    assert status["config_path"] == str(config_path.resolve())
    issue_keys = {issue["key"] for issue in status["validation"]["issues"]}
    assert "missing_port" in issue_keys
    assert "missing_host" in issue_keys
    assert "path_without_slash" in issue_keys
    assert "ebook_markdown_pipeline_http" not in status["service_urls"]



def test_model_api_settings_status_and_profile_persistence(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 9876, "path": "/call"},
                    "openclaw_http": {"host": "127.0.0.1", "port": 8931, "path": "/call"},
                    "mcp": {"transport": "stdio"},
                },
                "vision_execution": {"provider": "gemini", "model": "gemini-2.5-flash"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_API_KEY", "local-test-key")

    status = model_api_settings_status(config_path)

    assert status["schema"] == "video_knowledge_pipeline.model_api_settings.v1"
    assert status["current_profile"]["provider"] == "gemini"
    volcengine = next(row for row in status["providers"] if row["provider"] == "volcengine_coding_plan")
    assert volcengine["api_key_configured"] is True
    assert "local-test-key" not in json.dumps(status, ensure_ascii=False)

    updated = set_vision_execution_profile(
        provider="volcengine_coding_plan",
        model="ark-code-latest",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        multimodal_limit=10,
        temporal_limit=3,
        frame_count=8,
        config_path=config_path,
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert updated["ok"] is True
    assert saved["vision_execution"]["provider"] == "volcengine_coding_plan"
    assert saved["vision_execution"]["multimodal_limit"] == 10
    assert saved["vision_execution"]["frame_count"] == 8

    with pytest.raises(ValueError, match="must not contain secrets"):
        set_vision_execution_profile(provider="gemini", base_url="https://example.com?api_key=bad", config_path=config_path)


def test_model_api_settings_treats_openai_compatible_ark_env_as_volcengine(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 9876, "path": "/call"},
                    "openclaw_http": {"host": "127.0.0.1", "port": 8931, "path": "/call"},
                    "mcp": {"transport": "stdio"},
                },
                "vision_execution": {"provider": "volcengine_coding_plan", "model": "ark-code-latest"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "shared-openai-compatible-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    status = model_api_settings_status(config_path)
    volcengine = next(row for row in status["providers"] if row["provider"] == "volcengine_coding_plan")
    assert volcengine["api_key_configured"] is False

    monkeypatch.setenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    status = model_api_settings_status(config_path)
    volcengine = next(row for row in status["providers"] if row["provider"] == "volcengine_coding_plan")
    assert volcengine["api_key_configured"] is True
    assert "shared-openai-compatible-key" not in json.dumps(status, ensure_ascii=False)

def test_local_sampling_defaults_are_separate_from_cloud_vision_limits() -> None:
    parser = build_parser()

    local_args = parser.parse_args(["prepare-local-video-run", "lesson.mp4", "run-dir"])
    acceptance_args = parser.parse_args(["acceptance-run", "lesson.mp4", "run-dir"])
    openclaw_args = parser.parse_args(["openclaw-video-ingest", "lesson.mp4"])
    handoff_args = parser.parse_args(["openclaw-video-ingest-vdo-handoff", "--handoff-path", "handoff.json"])
    vision_profile = resolve_vision_execution_profile()

    assert local_args.sample_interval == DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS
    assert local_args.max_frames == DEFAULT_LOCAL_FRAME_BUDGET
    assert acceptance_args.max_frames == DEFAULT_LOCAL_FRAME_BUDGET
    assert openclaw_args.max_frames == DEFAULT_LOCAL_FRAME_BUDGET
    assert handoff_args.max_frames == DEFAULT_LOCAL_FRAME_BUDGET
    assert vision_profile["multimodal_limit"] == 19
    assert vision_profile["temporal_limit"] == 3
    assert vision_profile["frame_count"] == 8
def test_run_visual_structure_cli_defaults_cover_all_visual_routes() -> None:
    args = build_parser().parse_args(["run-visual-structure", "bundle"])

    assert args.include_routes == "document_visual,mixed,semantic_frame,temporal_sequence"







def test_model_api_cli_commands_are_registered() -> None:
    parser = build_parser()

    settings_args = parser.parse_args(["model-api-settings"])
    profile_args = parser.parse_args([
        "set-vision-profile",
        "--provider",
        "volcengine_coding_plan",
        "--model",
        "ark-code-latest",
        "--base-url",
        "https://ark.cn-beijing.volces.com/api/coding/v3",
        "--multimodal-limit",
        "10",
        "--temporal-limit",
        "3",
        "--frame-count",
        "8",
    ])

    assert settings_args.command == "model-api-settings"
    assert profile_args.command == "set-vision-profile"
    assert profile_args.provider == "volcengine_coding_plan"
    assert profile_args.multimodal_limit == 10

def test_asr_runtime_profile_is_sanitized_and_cli_registered(tmp_path: Path) -> None:
    config_path = tmp_path / "vkp-config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 8765, "path": "/call"},
                    "openclaw_http": {"host": "127.0.0.1", "port": 8931, "path": "/call"},
                    "mcp": {"transport": "stdio"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = set_asr_runtime_profile(
        provider="speaches_openai_compatible",
        model="iic/SenseVoiceSmall",
        device="cuda_preferred",
        compute_type="float16",
        enable_diarization=True,
        service_base_url="http://127.0.0.1:8000/v1",
        service_model="Systran/faster-whisper-large-v3",
        config_path=config_path,
    )
    profile = asr_runtime_profile(config_path)
    status = config_status(config_path)
    settings = model_api_settings_status(config_path)
    parser = build_parser()
    args = parser.parse_args(["set-asr-runtime-profile", "--provider", "funasr_sensevoice", "--device", "cuda_preferred", "--enable-vad", "true"])

    assert result["ok"] is True
    assert profile["provider"] == "speaches_openai_compatible"
    assert profile["device"] == "cuda_preferred"
    assert profile["openai_compatible"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert status["validation"]["ok"] is True
    assert status["asr_runtime"]["provider"] == "speaches_openai_compatible"
    assert settings["asr_runtime"]["enable_diarization"] is True
    assert any(row["provider"] == "speaches_openai_compatible" for row in settings["asr_service_adapters"])
    assert "key" not in json.dumps(profile, ensure_ascii=False).lower()
    assert args.command == "set-asr-runtime-profile"
    assert args.enable_vad == "true"
