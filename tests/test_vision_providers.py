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
from video_knowledge_pipeline.vision_api import _gemini_endpoint, parse_model_json, provider_requires_api_key, resolve_provider_config, test_vision_provider as run_vision_provider_test
from video_knowledge_pipeline.vision_environment import vision_environment_status
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight
from video_knowledge_pipeline.vision_provider_smoke import rank_vision_providers, vision_provider_matrix, vision_provider_smoke
from video_knowledge_pipeline.vlm_preprocess import prepare_vlm_image_inputs
from video_knowledge_pipeline.webui_bridge import export_webui_bundle, refresh_bundle_review_html
import video_knowledge_pipeline.visual_structure as visual_structure
from video_knowledge_pipeline.visual_structure import run_visual_structure_plan



# Moved from test_video_pipeline_smoke.py during Phase 10 split.

def test_vision_execution_profile_env_overrides_config_before_explicit_provider(monkeypatch, tmp_path: Path) -> None:
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
                    "multimodal_limit": 7,
                    "temporal_limit": 2,
                    "frame_count": 6,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LECTURE_VISION_PROVIDER", "agnes")
    monkeypatch.setenv("LECTURE_VISION_MODEL", "agnes-1.5-flash")

    env_profile = resolve_vision_execution_profile(config_path=config_path)

    assert env_profile["provider_config"]["provider"] == "agnes"
    assert env_profile["provider_config"]["model"] == "agnes-1.5-flash"
    assert env_profile["multimodal_limit"] == 7

    explicit = resolve_vision_execution_profile(
        config_path=config_path,
        provider_config={"provider": "openai", "model": "gpt-4o-mini"},
    )

    assert explicit["provider_config"]["provider"] == "openai"
    assert explicit["provider_config"]["model"] == "gpt-4o-mini"


def test_vision_provider_matrix_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "vision-provider-matrix",
            "--providers",
            "fixture,agnes",
            "--bundle-dir",
            "bundle",
            "--timeout-seconds",
            "5",
            "--preferred-provider",
            "agnes",
            "--no-write",
        ]
    )

    assert args.command == "vision-provider-matrix"
    assert args.providers == "fixture,agnes"
    assert args.bundle_dir == "bundle"
    assert args.timeout_seconds == 5
    assert args.preferred_provider == "agnes"
    assert args.no_write is True


def test_vision_provider_smoke_and_matrix_cli_dispatch_contract(tmp_path: Path) -> None:
    smoke_code = cli_main(["vision-provider-smoke", "--provider", "fixture", "--output-dir", str(tmp_path / "smoke"), "--no-write"])
    matrix_code = cli_main(
        [
            "vision-provider-matrix",
            "--providers",
            "fixture",
            "--output-dir",
            str(tmp_path / "matrix"),
            "--preferred-provider",
            "fixture",
            "--no-write",
        ]
    )

    assert smoke_code == 0
    assert matrix_code == 0



def test_gemini_endpoint_keeps_api_key_out_of_url() -> None:
    endpoint = _gemini_endpoint(
        {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model": "gemini-3.5-flash",
            "api_key": "must-not-appear",
        }
    )
    assert endpoint == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"


def test_provider_config_supports_gemini_and_openai(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    gemini = resolve_provider_config({"provider": "gemini"})
    assert gemini["provider"] == "gemini"
    assert gemini["api_key"] == "gemini-key"
    assert gemini["model"] == "gemini-3.6-flash"

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    openai = resolve_provider_config({"provider": "openai_compatible"})
    assert openai["provider"] == "openai_compatible"
    assert openai["api_key"] == "openai-key"

    monkeypatch.setenv("AGNES_API_KEY", "agnes-key")
    agnes = resolve_provider_config({"provider": "agnes"})
    assert agnes["provider"] == "agnes"
    assert agnes["api_key"] == "agnes-key"
    assert "agnes-ai" in agnes["base_url"]
    assert agnes["model"] == "agnes-1.5-flash"

    monkeypatch.setenv("LLM_API_KEY", "ark-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    monkeypatch.setenv("LLM_MODEL", "ark-code-latest")
    volcengine = resolve_provider_config({"provider": "volcengine_coding_plan"})
    assert volcengine["provider"] == "volcengine_coding_plan"
    assert volcengine["api_key"] == "ark-key"
    assert volcengine["base_url"] == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert volcengine["model"] == "ark-code-latest"




def test_volcengine_profile_reuses_openai_compatible_env_only_for_ark(monkeypatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "openai-compatible-ark-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    cfg = resolve_provider_config({"provider": "volcengine_coding_plan"})
    assert cfg["api_key"] == "openai-compatible-ark-key"
    assert cfg["base_url"] == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert cfg["model"] == "ark-code-latest"

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    cfg = resolve_provider_config({"provider": "volcengine_coding_plan"})
    assert cfg["api_key"] == ""
    assert cfg["base_url"] == "https://ark.cn-beijing.volces.com/api/coding/v3"

def test_provider_config_supports_local_qwen_vl_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LECTURE_VISION_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_QWEN_VL_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_QWEN_VL_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_QWEN_VL_MODEL", raising=False)

    cfg = resolve_provider_config({"provider": "local_qwen_vl"})

    assert cfg["provider"] == "local_qwen_vl"
    assert cfg["api_key"] == ""
    assert cfg["base_url"] == "http://127.0.0.1:8000/v1"
    assert cfg["model"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert provider_requires_api_key(cfg) is False


def test_provider_config_supports_local_qwen_vl_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_QWEN_VL_BASE_URL", "http://127.0.0.1:8010/v1")
    monkeypatch.setenv("LOCAL_QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    monkeypatch.setenv("LOCAL_QWEN_VL_API_KEY", "local-secret")

    cfg = resolve_provider_config({"provider": "qwen_local"})

    assert cfg["provider"] == "local_qwen_vl"
    assert cfg["base_url"] == "http://127.0.0.1:8010/v1"
    assert cfg["model"] == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert cfg["api_key"] == "local-secret"
def test_explicit_provider_does_not_inherit_different_env_model(monkeypatch) -> None:
    monkeypatch.setenv("LECTURE_VISION_PROVIDER", "agnes")
    monkeypatch.setenv("LECTURE_VISION_MODEL", "agnes-1.5-flash")
    monkeypatch.setenv("LECTURE_VISION_BASE_URL", "https://apihub.agnes-ai.com/v1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    gemini = resolve_provider_config({"provider": "gemini"})
    openai = resolve_provider_config({"provider": "openai"})

    assert gemini["provider"] == "gemini"
    assert gemini["model"] == "gemini-3.6-flash"
    assert gemini["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert openai["provider"] == "openai"
    assert openai["model"] == "gpt-4o-mini"
    assert openai["base_url"] == "https://api.openai.com/v1/chat/completions"


def test_openai_compatible_endpoint_accepts_provider_base_url() -> None:
    from video_knowledge_pipeline.vision_api import _openai_compatible_chat_completions_url

    assert (
        _openai_compatible_chat_completions_url({"base_url": "https://apihub.agnes-ai.com/v1"})
        == "https://apihub.agnes-ai.com/v1/chat/completions"
    )
    assert (
        _openai_compatible_chat_completions_url({"base_url": "https://apihub.agnes-ai.com/v1/chat/completions"})
        == "https://apihub.agnes-ai.com/v1/chat/completions"
    )
    assert (
        _openai_compatible_chat_completions_url({"base_url": "https://example.test"})
        == "https://example.test/v1/chat/completions"
    )
    assert (
        _openai_compatible_chat_completions_url({"base_url": "https://ark.cn-beijing.volces.com/api/coding/v3"})
        == "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
    )


def test_vision_provider_defaults_to_unified_config(monkeypatch, tmp_path: Path) -> None:
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
                    "model": "gemini-2.5-flash-lite",
                    "multimodal_limit": 4,
                    "temporal_limit": 2,
                    "frame_count": 7,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))
    monkeypatch.delenv("LECTURE_VISION_PROVIDER", raising=False)
    monkeypatch.delenv("LECTURE_VISION_MODEL", raising=False)
    monkeypatch.delenv("LECTURE_VISION_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    cfg = resolve_provider_config()

    assert cfg["provider"] == "gemini"
    assert cfg["model"] == "gemini-2.5-flash-lite"
    assert cfg["api_key"] == ""


def test_direct_multimodal_retries_transient_provider_error(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-retry"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "visual_route": "semantic_frame",
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal_mod

    calls = {"count": 0}

    def flaky_call(*, provider_config, prompt, image_paths, allowed_roots=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"ok": False, "error": "SSL: UNEXPECTED_EOF_WHILE_READING", "content": ""}
        return {
            "ok": True,
            "error": "",
            "content": json.dumps(
                {
                    "objects": ["screen"],
                    "actions": ["presenter speaks"],
                    "confidence": 0.8,
                    "evidence_frame_paths": [str(frame)],
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr(multimodal_mod, "call_vision_model", flaky_call)

    result = run_multimodal_frame_analysis(
        bundle,
        execute=True,
        provider_config={"provider": "gemini", "api_key": "secret"},
        limit=1,
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
        vision_retries=2,
        vision_retry_delay_seconds=0,
    )
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]

    assert calls["count"] == 2
    assert result["summary"]["updated"] == 1
    assert result["items"][0]["attempt_count"] == 2
    assert result["items"][0]["attempts"][0]["ok"] is False
    assert result["items"][0]["attempts"][1]["ok"] is True
    assert updated["visual_understanding"]["schema"] == "lecture_visual_understanding.v1"


def test_fixture_vision_provider_executes_controlled_semantic_and_temporal_runs(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-fixture-provider"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame_a = assets / "frame-a.jpg"
    frame_b = assets / "frame-b.jpg"
    frame_a.write_bytes(b"fake image a")
    frame_b.write_bytes(b"fake image b")
    timeline = [
        {"index": 1, "start": 0, "end": 1, "visual_route": "semantic_frame", "frame_paths": [str(frame_a)]},
        {
            "index": 2,
            "start": 1,
            "end": 2,
            "visual_route": "temporal_sequence",
            "frame_paths": [str(frame_a)],
            "temporal_frame_paths": [str(frame_a), str(frame_b)],
        },
    ]
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    provider_config = {"provider": "fixture"}
    provider_test = run_vision_provider_test(provider_config, image_paths=[str(frame_a), str(frame_b)])
    assert provider_test["ok"] is True
    assert provider_test["provider"]["api_key_required"] is False

    semantic_preflight = vision_execution_preflight(
        bundle,
        provider_config=provider_config,
        semantic_limit=1,
        temporal_limit=0,
        include_semantic=True,
        include_temporal=False,
    )
    assert semantic_preflight["ready_to_execute"] is True
    assert semantic_preflight["confirmation"]["semantic_confirm_vision_calls"] == 1
    assert semantic_preflight["confirmation"]["semantic_confirm_vision_indexes"] == "1"

    semantic = run_multimodal_frame_analysis(
        bundle,
        execute=True,
        provider_config=provider_config,
        limit=1,
        indexes=[1],
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
    )
    assert semantic["summary"]["status"] == "ok"
    assert semantic["summary"]["updated"] == 1
    assert semantic["items"][0]["ok"] is True
    assert semantic["items"][0]["visual_understanding"]["validation_status"] == "ok"
    assert semantic["run_audit"]["record"]["execution_control"]["confirmed"] is True
    assert semantic["vision_restore_hint"]["status"] == "ready"

    temporal_preflight = vision_execution_preflight(
        bundle,
        provider_config=provider_config,
        semantic_limit=0,
        temporal_limit=1,
        frame_count=8,
        include_semantic=False,
        include_temporal=True,
    )
    assert temporal_preflight["ready_to_execute"] is True
    assert temporal_preflight["confirmation"]["temporal_confirm_vision_calls"] == 1
    assert temporal_preflight["confirmation"]["temporal_confirm_vision_indexes"] == "2"

    temporal = run_temporal_visual_analysis(
        bundle,
        execute=True,
        provider_config=provider_config,
        limit=1,
        indexes=[2],
        confirm_vision_calls=1,
        confirm_vision_indexes="2",
    )
    assert temporal["summary"]["status"] == "ok"
    assert temporal["summary"]["updated"] == 1
    assert temporal["items"][0]["ok"] is True
    assert temporal["items"][0]["temporal_visual_understanding"]["validation_status"] == "ok"
    assert temporal["run_audit"]["record"]["execution_control"]["confirmed"] is True
    assert temporal["vision_restore_hint"]["status"] == "ready"

    check = controlled_execution_check(bundle, refresh=False)
    assert check["ready_for_real_vision_execution"] is True
    assert check["controlled_execution"]["latest_vision_run_status"] == "ok"
    assert check["controlled_execution"]["latest_vision_write_recoverable"] is True
    assert all(item["ok"] for item in check["checklist"])

    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert updated[0]["visual_understanding"]["source"] == "fixture-vision:fixture"
    assert updated[1]["temporal_visual_understanding"]["source"] == "fixture-vision:fixture"


def test_vision_provider_test_does_not_leak_key_and_parse_repair(monkeypatch) -> None:
    monkeypatch.delenv("LECTURE_VISION_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing = run_vision_provider_test({"provider": "openai", "api_key": ""})
    assert missing["provider"]["api_key_configured"] is False
    assert "api_key" not in missing["provider"]
    assert missing["checks"][0]["error"] == "missing_api_key"

    parsed = parse_model_json("```json\n{\"ok\": true}\n```")
    assert parsed["ok"] is True
    failed = parse_model_json("not json")
    assert failed["_parse_failed"] is True
    assert "raw_content" in failed


def test_vision_provider_test_covers_single_and_multi_image(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")

    import video_knowledge_pipeline.vision_api as vision_api

    calls = []

    def fake_call_vision_model(*, provider_config, prompt, image_paths):
        calls.append(list(image_paths))
        return {"ok": True, "error": "", "content": "{\"ok\": true, \"seen\": %d}" % len(image_paths)}

    monkeypatch.setattr(vision_api, "call_vision_model", fake_call_vision_model)

    result = run_vision_provider_test({"provider": "openai", "api_key": "secret"}, image_paths=[str(first), str(second)])

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["safe_to_execute"] is True
    assert "api_key" not in result["provider"]
    assert [check["name"] for check in result["checks"]] == ["text_ping", "single_image_json", "multi_image_json"]
    assert calls == [[], [str(first)], [str(first), str(second)]]


def test_vision_provider_test_normalizes_transport_and_parse_failures(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    first.write_bytes(b"fake")

    import video_knowledge_pipeline.vision_api as vision_api

    def timeout_provider(*, provider_config, prompt, image_paths):
        return {"ok": False, "error": "The read operation timed out", "content": ""}

    monkeypatch.setattr(vision_api, "call_vision_model", timeout_provider)
    sentinel_key = "actual-key-123"  # gitleaks:allow synthetic redaction sentinel
    timeout = run_vision_provider_test({"provider": "openai", "api_key": sentinel_key}, image_paths=[str(first)])

    assert timeout["ok"] is False
    assert timeout["safe_to_execute"] is False
    assert timeout["status"] == "provider_unreachable"
    assert timeout["error_class"] == "provider_unreachable"
    assert sentinel_key not in json.dumps(timeout, ensure_ascii=False)

    def non_json_provider(*, provider_config, prompt, image_paths):
        return {"ok": True, "error": "", "content": "not json"}

    monkeypatch.setattr(vision_api, "call_vision_model", non_json_provider)
    parse_failed = run_vision_provider_test({"provider": "openai", "api_key": sentinel_key}, image_paths=[str(first)])

    assert parse_failed["ok"] is False
    assert parse_failed["safe_to_execute"] is False
    assert parse_failed["status"] == "model_output_parse_failed"
    assert parse_failed["error_class"] == "model_output_parse_failed"


def test_vision_provider_test_diagnoses_text_only_image_timeout(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"fake image one")
    second.write_bytes(b"fake image two")

    import video_knowledge_pipeline.vision_api as vision_api

    def text_ok_image_timeout(*, provider_config, prompt, image_paths):
        if not image_paths:
            return {"ok": True, "error": "", "content": "{\"ok\": true}"}
        return {"ok": False, "error": "The read operation timed out", "content": ""}

    monkeypatch.setattr(vision_api, "call_vision_model", text_ok_image_timeout)

    result = run_vision_provider_test({"provider": "agnes", "api_key": "actual-key-123"}, image_paths=[str(first), str(second)])  # gitleaks:allow synthetic redaction sentinel

    assert result["ok"] is False
    assert result["safe_to_execute"] is False
    assert result["status"] == "text_only_ok_image_timeout"
    assert result["error_class"] == "text_only_ok_image_timeout"
    assert result["failure_diagnosis"]["text_ping_ok"] is True
    assert result["failure_diagnosis"]["image_checks_failed"] == 2
    assert result["failure_diagnosis"]["failed_image_check_names"] == ["single_image_json", "multi_image_json"]
    assert result["checks"][1]["image_payload"]["total_bytes"] == len(b"fake image one")
    assert result["checks"][2]["image_payload"]["total_bytes"] == len(b"fake image one") + len(b"fake image two")
    assert "actual-key-123" not in json.dumps(result, ensure_ascii=False)


def test_vision_execution_preflight_can_block_on_provider_health(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "semantic_frame",
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import video_knowledge_pipeline.vision_api as vision_api

    monkeypatch.setattr(
        vision_api,
        "call_vision_model",
        lambda *, provider_config, prompt, image_paths: {"ok": False, "error": "SSL: UNEXPECTED_EOF_WHILE_READING", "content": ""},
    )

    preflight = vision_execution_preflight(
        bundle,
        provider_config={"provider": "openai", "api_key": "actual-key-123", "model": "gpt-4o-mini"},  # gitleaks:allow synthetic redaction sentinel
        semantic_limit=1,
        include_temporal=False,
        check_provider=True,
    )
    blocker_keys = {item["key"] for item in preflight["blockers"]}
    markdown = Path(preflight["preflight_path"]).read_text(encoding="utf-8")

    assert preflight["provider_health"]["status"] == "provider_transport_error"
    assert preflight["provider_health"]["safe_to_execute"] is False
    assert "provider_health_failed" in blocker_keys
    assert preflight["ready_to_execute"] is False
    assert "Provider Health" in markdown
    assert "provider_transport_error" in markdown
    assert "actual-key-123" not in json.dumps(preflight, ensure_ascii=False)


def test_vision_env_status_writes_no_secret_template_and_refuses_overwrite(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.vision_environment as vision_environment

    monkeypatch.setattr(vision_environment, "project_root", lambda: tmp_path)
    monkeypatch.delenv("AGNES_API_KEY", raising=False)
    template = tmp_path / ".local" / "vision.env"

    result = vision_environment_status(provider="agnes", model="agnes-1.5-flash", write_template=True)

    assert result["provider"]["provider"] == "agnes"
    assert result["provider"]["api_key_configured"] is False
    assert result["template_written"] is True
    text = template.read_text(encoding="utf-8")
    assert "AGNES_API_KEY=<your key>" in text
    assert "secret" not in text.lower()
    assert result["env_files"][0]["exists"] is True
    assert result["next_action"]["key"] == "fill_api_key"

    template.write_text("AGNES_API_KEY=keep-existing", encoding="utf-8")
    second = vision_environment_status(provider="agnes", write_template=True)

    assert second["template_written"] is False
    assert second["template_status"] == "exists"
    assert template.read_text(encoding="utf-8") == "AGNES_API_KEY=keep-existing"


def test_vision_env_status_reports_key_presence_without_leaking_value(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.vision_environment as vision_environment

    monkeypatch.setattr(vision_environment, "project_root", lambda: tmp_path)
    monkeypatch.setenv("LECTURE_VISION_PROVIDER", "agnes")
    monkeypatch.setenv("AGNES_API_KEY", "secret-value")
    monkeypatch.setenv("LECTURE_VISION_MODEL", "agnes-1.5-flash")

    result = vision_environment_status()

    encoded = json.dumps(result, ensure_ascii=False)
    assert result["provider"]["provider"] == "agnes"
    assert result["provider"]["api_key_configured"] is True
    assert result["sanitized_provider_config"]["provider"] == "agnes"
    assert "secret-value" not in encoded
    assert result["next_action"]["key"] == "run_vision_provider_matrix"


def test_vision_env_status_supports_local_qwen_without_key(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.vision_environment as vision_environment

    monkeypatch.setattr(vision_environment, "project_root", lambda: tmp_path)
    monkeypatch.delenv("LOCAL_QWEN_VL_API_KEY", raising=False)
    template = tmp_path / ".local" / "vision.env"

    result = vision_environment_status(provider="local_qwen_vl", write_template=True)
    text = template.read_text(encoding="utf-8")

    assert result["provider"]["provider"] == "local_qwen_vl"
    assert result["provider"]["api_key_required"] is False
    assert result["provider"]["api_key_configured"] is False
    assert result["next_action"]["key"] == "run_vision_provider_matrix"
    assert "LOCAL_QWEN_VL_API_KEY=<your key>" not in text
    assert "# LOCAL_QWEN_VL_API_KEY=<your local key>" in text
    assert "LECTURE_VISION_BASE_URL=http://127.0.0.1:8000/v1" in text
    assert "LECTURE_VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct" in text

def test_vision_env_status_writes_volcengine_template(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.vision_environment as vision_environment

    monkeypatch.setattr(vision_environment, "project_root", lambda: tmp_path)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    template = tmp_path / ".local" / "vision.env"

    result = vision_environment_status(provider="volcengine_coding_plan", write_template=True)
    text = template.read_text(encoding="utf-8")

    assert result["provider"]["provider"] == "volcengine_coding_plan"
    assert result["provider"]["model"] == "ark-code-latest"
    assert result["provider"]["api_key_configured"] is False
    assert "ARK_API_KEY=<your key>" in text
    assert "LECTURE_VISION_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3" in text
    assert "LECTURE_VISION_MODEL=ark-code-latest" in text


def test_vision_env_placeholder_key_is_not_configured(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.vision_environment as vision_environment

    monkeypatch.setattr(vision_environment, "project_root", lambda: tmp_path)
    local = tmp_path / ".local"
    local.mkdir()
    (local / "vision.env").write_text(
        "LECTURE_VISION_PROVIDER=agnes\nAGNES_API_KEY=<your key>\nLECTURE_VISION_MODEL=agnes-1.5-flash\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LECTURE_VISION_PROVIDER", "agnes")
    monkeypatch.setenv("AGNES_API_KEY", "<your key>")
    monkeypatch.setenv("LECTURE_VISION_MODEL", "agnes-1.5-flash")

    result = vision_environment_status()

    assert result["provider"]["provider"] == "agnes"
    assert result["provider"]["api_key_configured"] is False
    assert result["next_action"]["key"] == "fill_api_key"


def test_vision_execution_preflight_uses_matrix_recommended_provider_config(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-matrix-preflight"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "vision_provider_matrix_json": "vision-provider-matrix.json",
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
                    "visual_route": "semantic_frame",
                    "assets": [{"path": "assets/frame.jpg"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "vision-provider-matrix.json").write_text(
        json.dumps(
            {
                "schema": "lecture_vision_provider_matrix.v1",
                "status": "ok",
                "recommended_provider": "fixture",
                "recommended_provider_config": {
                    "provider": "fixture",
                    "model": "fixture-vision",
                    "base_url": "https://example.invalid/v1?key=<redacted>",
                    "timeout_seconds": 11,
                    "image_probe_max_edge": 512,
                    "image_probe_jpeg_quality": 55,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    preflight = vision_execution_preflight(bundle, semantic_limit=1, temporal_limit=0, include_temporal=False, write=False)

    assert preflight["execution_profile"]["provider_config_source"] == "vision_provider_matrix"
    assert preflight["provider"]["provider"] == "fixture"
    assert preflight["provider"]["model"] == "fixture-vision"
    assert preflight["provider"]["timeout_seconds"] == 11
    assert preflight["provider"]["base_url"] != "https://example.invalid/v1?key=<redacted>"
    assert preflight["recommended_provider_config"]["image_probe_max_edge"] == 512
    assert preflight["recommended_provider_config"]["image_probe_jpeg_quality"] == 55
    assert preflight["ready_to_execute"] is True
    assert "<redacted>" in json.dumps(preflight, ensure_ascii=False)


def test_controlled_execution_check_blocks_provider_health_failed_even_with_prior_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-provider-health"
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
    (bundle / "vision-execution-preflight.md").write_text("# preflight\n\n- Provider health: `provider_unreachable`", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text(
        json.dumps(
            {
                "ready_to_execute": False,
                "blockers": [{"key": "provider_health_failed"}],
                "provider_health": {
                    "status": "provider_unreachable",
                    "safe_to_execute": False,
                    "error_class": "provider_unreachable",
                    "secrets_redacted": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "semantic_frame-old-ok",
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

    check = controlled_execution_check(bundle, refresh=False)
    controlled = check["controlled_execution"]

    assert controlled["status"] == "blocked"
    assert "provider_health_failed" in controlled["blockers"]
    assert controlled["provider_health_status"] == "provider_unreachable"
    assert check["ready_for_real_vision_execution"] is False
    assert any(item["key"] == "provider_health_ok" and not item["ok"] for item in check["checklist"])
    assert "provider health check" in " ".join(check["next_steps"])


def test_acceptance_check_reports_provider_blocked_when_visual_gaps_need_unsafe_provider(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-acceptance-provider"
    bundle.mkdir()
    (bundle / "assets").mkdir()
    (bundle / "assets" / "frame.jpg").write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "provider blocked",
                "vision_execution_preflight": "vision-execution-preflight.md",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
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
                    "end": 1,
                    "transcript": "讲解一个界面状态",
                    "visual_route": "semantic_frame",
                    "visual_text": "Synthetic screen label",
                    "assets": [{"path": "assets/frame.jpg"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "vision-execution-preflight.md").write_text("# preflight\n", encoding="utf-8")
    (bundle / "vision-execution-preflight.json").write_text(
        json.dumps(
            {
                "ready_to_execute": False,
                "blockers": [{"key": "provider_health_failed"}],
                "provider_health": {
                    "status": "provider_unreachable",
                    "safe_to_execute": False,
                    "error_class": "provider_unreachable",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps({"run_id": "semantic_frame-1", "status": "ok", "execute": True, "updated_count": 1, "timeline_diff_count": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = acceptance_check(bundle, refresh=True)

    assert report["status"] == "provider_blocked"
    assert report["summary"]["semantic_missing"] == 1
    assert report["summary"]["provider_health"] == "provider_unreachable"
    assert any(item["key"] == "provider_health_failed" for item in report["blockers"])
    assert report["next_action"]["key"] == "provider_matrix_missing"
    assert report["next_action"]["mcp_tool"] == "vision_provider_matrix"
    assert report["provider_matrix"]["status"] == "missing"
    assert (bundle / "acceptance-check.json").exists()
    assert (bundle / "acceptance-check.md").exists()
    assert json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["mcp_acceptance_check_args"] == "mcp-acceptance-check.args.json"


def test_acceptance_check_reports_machine_action_when_provider_not_checked(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-acceptance-machine"
    bundle.mkdir()
    (bundle / "assets").mkdir()
    (bundle / "assets" / "frame.jpg").write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "machine action"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "transcript": "讲解一个界面状态",
                    "visual_route": "semantic_frame",
                    "visual_text": "Synthetic screen label",
                    "assets": [{"path": "assets/frame.jpg"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = acceptance_check(bundle, refresh=True)

    assert report["status"] == "machine_action_available"
    assert report["summary"]["semantic_missing"] == 1
    assert report["next_action"]["key"] in {"semantic_frame_understanding", "export_knowledge_note"}
    assert any(item["key"] == "semantic_frame_without_analysis" for item in report["blockers"])


def test_bundle_next_action_prioritizes_provider_repair_when_vision_provider_is_unsafe(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-next-provider-repair"
    bundle.mkdir()
    (bundle / "assets").mkdir()
    (bundle / "assets" / "frame.jpg").write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "provider repair",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
                "mcp_vision_execution_preflight_args": "mcp-vision-execution-preflight.args.json",
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
                    "end": 1,
                    "transcript": "讲解一个界面状态",
                    "visual_route": "semantic_frame",
                    "visual_text": "屏幕上有一个控制台",
                    "assets": [{"path": "assets/frame.jpg"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "vision-execution-preflight.json").write_text(
        json.dumps(
            {
                "ready_to_execute": False,
                "blockers": [{"key": "provider_health_failed"}],
                "provider_health": {
                    "status": "provider_unreachable",
                    "safe_to_execute": False,
                    "error_class": "provider_unreachable",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = bundle_next_action(bundle)
    action = result["next_action"]

    assert result["status"] == "provider_blocked"
    assert action["key"] == "provider_matrix_repair"
    assert action["mcp_tool"] == "vision_provider_matrix"
    assert action["for_blocked_action"]["mcp_tool"] == "run_multimodal_frame_analysis"
    assert action["provider_health"]["safe_to_execute"] is False
    assert (bundle / "mcp-vision-provider-matrix.args.json").exists()


def test_vision_provider_smoke_writes_no_secret_report_and_bundle_args(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smoke"
    bundle.mkdir()
    assets = bundle / "assets"
    assets.mkdir()
    first = assets / "first.jpg"
    second = assets / "second.jpg"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "visual_route": "temporal_sequence",
                    "assets": [{"path": "assets/first.jpg"}],
                    "temporal_frame_paths": ["assets/first.jpg", "assets/second.jpg"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import video_knowledge_pipeline.vision_api as vision_api

    def fake_call_vision_model(*, provider_config, prompt, image_paths):
        return {"ok": True, "error": "", "content": "{\"ok\": true, \"image_count\": %d}" % len(image_paths)}

    monkeypatch.setattr(vision_api, "call_vision_model", fake_call_vision_model)
    secret = "actual-key-123"  # gitleaks:allow synthetic redaction sentinel

    report = vision_provider_smoke(
        provider_config={
            "provider": "openai",
            "api_key": secret,
            "base_url": "https://api.example.test/v1?api_key=url-secret&foo=bar",
        },
        bundle_dir=bundle,
    )

    assert report["status"] == "ok"
    assert report["safe_to_execute"] is True
    assert len(report["image_paths"]) == 2
    assert report["diagnostics"]["endpoint_kind"] == "openai_chat_completions"
    assert report["diagnostics"]["base_url_host"] == "api.example.test"
    assert "url-secret" not in report["diagnostics"]["request_url"]
    assert "api_key=%3Credacted%3E" in report["diagnostics"]["request_url"]
    assert report["image_selection"]["has_multi_image_check"] is True
    assert (bundle / "vision-provider-smoke.json").exists()
    assert (bundle / "vision-provider-smoke.md").exists()
    report_text = (bundle / "vision-provider-smoke.json").read_text(encoding="utf-8")
    args_text = (bundle / "mcp-vision-provider-smoke.args.json").read_text(encoding="utf-8")
    assert secret not in report_text
    assert secret not in args_text
    assert "url-secret" not in report_text
    assert "url-secret" not in args_text
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mcp_vision_provider_smoke_args"] == "mcp-vision-provider-smoke.args.json"
    run = json.loads((bundle / "runs" / "vision-provider-smoke" / "run.json").read_text(encoding="utf-8"))
    assert run["run_type"] == "vision_provider_smoke"
    assert run["status"] == "completed"
    assert run["artifacts"][0]["path"].endswith("vision-provider-smoke.json")
    assert "actual-key" not in json.dumps(run, ensure_ascii=False)


def test_vision_provider_smoke_reports_missing_key_without_writing_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LECTURE_VISION_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "out"

    report = vision_provider_smoke(provider="openai", output_dir=out)

    assert report["status"] == "missing_api_key"
    assert report["safe_to_execute"] is False
    assert report["recovery_suggestion"].startswith("Missing API key")
    assert "api_key" not in report["provider"]
    assert (out / "vision-provider-smoke.md").exists()


def test_vision_provider_smoke_reports_text_only_image_timeout(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smoke-image-timeout"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    (assets / "first.jpg").write_bytes(b"first")
    (assets / "second.jpg").write_bytes(b"second")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "visual_route": "temporal_sequence",
                    "assets": [{"path": "assets/first.jpg"}],
                    "temporal_frame_paths": ["assets/first.jpg", "assets/second.jpg"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import video_knowledge_pipeline.vision_api as vision_api

    def text_ok_image_timeout(*, provider_config, prompt, image_paths):
        if not image_paths:
            return {"ok": True, "error": "", "content": "{\"ok\": true}"}
        return {"ok": False, "error": "The read operation timed out", "content": ""}

    monkeypatch.setattr(vision_api, "call_vision_model", text_ok_image_timeout)

    report = vision_provider_smoke(provider_config={"provider": "agnes", "api_key": "actual-key-123"}, bundle_dir=bundle, timeout_seconds=3)  # gitleaks:allow synthetic redaction sentinel
    markdown = (bundle / "vision-provider-smoke.md").read_text(encoding="utf-8")
    report_text = (bundle / "vision-provider-smoke.json").read_text(encoding="utf-8")

    assert report["status"] == "text_only_ok_image_timeout"
    assert report["error_class"] == "text_only_ok_image_timeout"
    assert report["failure_diagnosis"]["text_ping_ok"] is True
    assert report["failure_diagnosis"]["image_checks_failed"] == 2
    assert report["recovery_suggestion"].startswith("Text ping passed but image checks timed out")
    assert "Failure diagnosis: `text_only_ok_image_timeout`" in markdown
    assert "actual-key-123" not in report_text


def test_vision_provider_smoke_can_probe_resized_limited_images(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-smoke-probe"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "first.jpg"
    second = assets / "second.jpg"
    first.write_bytes(b"first original")
    second.write_bytes(b"second original")
    probe_dir = bundle / "vision-provider-smoke-probes"
    probe_dir.mkdir()
    probe = probe_dir / "01-first-probe.jpg"
    probe.write_bytes(b"probe")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "visual_route": "temporal_sequence",
                    "assets": [{"path": "assets/first.jpg"}],
                    "temporal_frame_paths": ["assets/first.jpg", "assets/second.jpg"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import video_knowledge_pipeline.vision_api as vision_api

    calls = []

    def fake_call_vision_model(*, provider_config, prompt, image_paths):
        calls.append(list(image_paths))
        return {"ok": True, "error": "", "content": "{\"ok\": true}"}

    monkeypatch.setattr(vision_api, "call_vision_model", fake_call_vision_model)

    report = vision_provider_smoke(
        provider_config={"provider": "openai", "api_key": "actual-key-123"},  # gitleaks:allow synthetic redaction sentinel
        bundle_dir=bundle,
        image_probe_max_edge=320,
        image_probe_jpeg_quality=55,
        max_images=1,
    )
    args = json.loads((bundle / "mcp-vision-provider-smoke.args.json").read_text(encoding="utf-8"))
    markdown = (bundle / "vision-provider-smoke.md").read_text(encoding="utf-8")

    assert report["safe_to_execute"] is True
    assert report["image_selection"]["source_image_count"] == 1
    assert report["image_selection"]["image_count"] == 1
    assert report["image_selection"]["has_multi_image_check"] is False
    assert report["image_probe"]["status"] in {"ok", "partial", "unavailable"}
    assert report["image_probe"]["max_edge"] == 320
    assert args["image_probe_max_edge"] == 320
    assert args["image_probe_jpeg_quality"] == 55
    assert args["max_images"] == 1
    assert "Image probe status" in markdown
    assert len(calls) == 2
    assert calls[0] == []
    assert len(calls[1]) == 1
    assert "actual-key-123" not in (bundle / "vision-provider-smoke.json").read_text(encoding="utf-8")


def test_rank_vision_providers_requires_key_image_checks_and_honors_preference() -> None:
    results = [
        {
            "provider": {"provider": "openai", "model": "gpt-4o-mini", "api_key_required": True, "api_key_configured": True, "timeout_seconds": 30},
            "safe_to_execute": True,
            "checks": [
                {"name": "text_ping", "ok": True},
                {"name": "single_image_json", "ok": True},
                {"name": "multi_image_json", "ok": True},
            ],
            "recommended_provider_config": {"provider": "openai", "model": "gpt-4o-mini"},
        },
        {
            "provider": {"provider": "agnes", "model": "agnes-1.5-flash", "api_key_required": True, "api_key_configured": True, "timeout_seconds": 30},
            "safe_to_execute": True,
            "checks": [
                {"name": "text_ping", "ok": True},
                {"name": "single_image_json", "ok": True},
                {"name": "multi_image_json", "ok": True},
            ],
            "recommended_provider_config": {"provider": "agnes", "model": "agnes-1.5-flash"},
        },
        {
            "provider": {"provider": "gemini", "model": "gemini-2.5-flash", "api_key_required": True, "api_key_configured": False, "timeout_seconds": 10},
            "safe_to_execute": True,
            "checks": [
                {"name": "text_ping", "ok": True},
                {"name": "single_image_json", "ok": True},
                {"name": "multi_image_json", "ok": True},
            ],
            "recommended_provider_config": {"provider": "gemini", "model": "gemini-2.5-flash"},
        },
        {
            "provider": {"provider": "custom_openai_compatible", "model": "text-only", "api_key_required": True, "api_key_configured": True, "timeout_seconds": 5},
            "safe_to_execute": False,
            "checks": [
                {"name": "text_ping", "ok": True},
                {"name": "single_image_json", "ok": False},
            ],
            "recommended_provider_config": {"provider": "custom_openai_compatible", "model": "text-only"},
        },
    ]

    ranking = rank_vision_providers(results, preferred_provider="agnes")

    assert ranking[0]["provider"] == "agnes"
    assert ranking[0]["ready"] is True
    assert ranking[0]["preferred"] is True
    rows = {row["provider"]: row for row in ranking}
    assert rows["gemini"]["ready"] is False
    assert rows["gemini"]["recommended_provider_config"] == {}
    assert rows["custom_openai_compatible"]["ready"] is False
    assert rows["custom_openai_compatible"]["recommended_provider_config"] == {}
    assert "api_key" not in json.dumps(ranking, ensure_ascii=False)


def test_vision_provider_matrix_recommends_first_ready_provider_and_writes_no_secret(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-matrix"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "visual_route": "semantic_frame",
                    "assets": [{"path": "assets/frame.jpg"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "matrix-secret-key")

    report = vision_provider_matrix(providers=["fixture", "openai"], bundle_dir=bundle, timeout_seconds=3)

    assert report["status"] == "ok"
    assert report["recommended_provider"] == "fixture"
    assert report["recommended_provider_config"]["provider"] == "fixture"
    assert [row["provider"]["provider"] for row in report["results"]] == ["fixture", "openai"]
    assert report["provider_ranking"][0]["provider"] == "fixture"
    assert report["results"][0]["safe_to_execute"] is True
    assert (bundle / "vision-provider-matrix.json").exists()
    assert (bundle / "vision-provider-matrix.md").exists()
    assert (bundle / "mcp-vision-provider-matrix.args.json").exists()
    text = (bundle / "vision-provider-matrix.json").read_text(encoding="utf-8")
    args_text = (bundle / "mcp-vision-provider-matrix.args.json").read_text(encoding="utf-8")
    assert "matrix-secret-key" not in text
    assert "matrix-secret-key" not in args_text
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vision_provider_matrix"] == "vision-provider-matrix.md"
    assert manifest["mcp_vision_provider_matrix_args"] == "mcp-vision-provider-matrix.args.json"
    run = json.loads((bundle / "runs" / "vision-provider-matrix" / "run.json").read_text(encoding="utf-8"))
    assert run["run_type"] == "vision_provider_matrix"
    assert run["status"] == "completed"
    assert run["artifacts"][0]["path"].endswith("vision-provider-matrix.json")
    assert "matrix-secret-key" not in json.dumps(run, ensure_ascii=False)


def test_vision_provider_matrix_does_not_select_agnes_when_image_probe_fails(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-matrix-agnes"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    (assets / "first.jpg").write_bytes(b"first")
    (assets / "second.jpg").write_bytes(b"second")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "visual_route": "temporal_sequence",
                    "assets": [{"path": "assets/first.jpg"}],
                    "temporal_frame_paths": ["assets/first.jpg", "assets/second.jpg"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGNES_API_KEY", "agnes-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    import video_knowledge_pipeline.vision_api as vision_api

    def fake_call_vision_model(*, provider_config, prompt, image_paths):
        if provider_config.get("provider") == "agnes" and image_paths:
            return {"ok": False, "error": "image probe failed", "content": ""}
        return {"ok": True, "error": "", "content": "{\"ok\": true}"}

    monkeypatch.setattr(vision_api, "call_vision_model", fake_call_vision_model)

    report = vision_provider_matrix(providers=["agnes", "openai"], bundle_dir=bundle, preferred_provider="agnes", timeout_seconds=3)

    assert report["status"] == "ok"
    assert report["recommended_provider"] == "openai"
    assert report["recommended_provider_config"]["provider"] == "openai"
    ranking = {row["provider"]: row for row in report["provider_ranking"]}
    assert ranking["agnes"]["ready"] is False
    assert ranking["agnes"]["single_image_json_ok"] is False
    assert ranking["agnes"]["recommended_provider_config"] == {}
    encoded = json.dumps(report, ensure_ascii=False)
    assert "agnes-secret" not in encoded
    assert "openai-secret" not in encoded


def test_controlled_execution_check_prefers_latest_direct_provider_not_ready_over_stale_confirmation(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-provider-missing"
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
    (bundle / "bundle-advance-runs.jsonl").write_text(
        json.dumps(
            {
                "created_at": "2026-06-06T00:00:00",
                "status": "blocked",
                "blocked_reason": "vision execution confirmation required; match confirm_vision_calls and confirm_vision_indexes from preflight",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "vision-analysis-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "semantic_frame-provider-missing",
                "kind": "semantic_frame",
                "execute": True,
                "status": "vision_provider_not_ready",
                "updated_count": 0,
                "timeline_diff_count": 0,
                "execution_control": {
                    "status": "vision_provider_not_ready",
                    "error": "missing_api_key",
                    "expected_api_calls": 1,
                    "expected_indexes": "5",
                    "received_confirm_vision_calls": 1,
                    "received_confirm_vision_indexes": "5",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    check = controlled_execution_check(bundle, refresh=False)

    controlled = check["controlled_execution"]
    assert "vision_provider_not_ready" in controlled["blockers"]
    assert "confirmation_required" not in controlled["blockers"]
    assert all(item["key"] != "batch_confirmed_or_not_pending" or item["ok"] for item in check["checklist"])
    next_steps = " ".join(check["next_steps"])
    assert "provider API key" in next_steps
    assert "confirm_vision_calls" not in next_steps


def test_vlm_preprocess_prepares_compatible_probe_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")

    disabled = prepare_vlm_image_inputs([str(image_path)], output_dir=tmp_path / "probes", max_edge=0)

    assert disabled["schema"] == "video_knowledge_pipeline.vlm_preprocess.v1"
    assert disabled["status"] == "disabled"
    assert disabled["prepared_image_paths"] == [str(image_path)]
    assert disabled["total_source_bytes"] == len(b"fake-image")
