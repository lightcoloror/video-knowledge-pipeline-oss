from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.local_production_preset import (
    install_local_production_preset,
)
from video_knowledge_pipeline.model_api_settings import (
    LOCAL_PRODUCTION_ROUTE_PRESET_ID,
    LOCAL_PRODUCTION_ROUTE_TASK_PROFILES,
    apply_model_api_route_preset,
    load_model_api_settings,
    resolve_model_api_route,
)
from video_knowledge_pipeline.smart_summary_section_llm import _profile_execution


def test_local_production_route_preset_installs_exact_local_models(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "model-api-settings.json"
    status = apply_model_api_route_preset(
        LOCAL_PRODUCTION_ROUTE_PRESET_ID,
        settings_path=settings_path,
        secrets_path=tmp_path / "model-api-secrets.json",
    )

    assert status["task_routes"] == LOCAL_PRODUCTION_ROUTE_TASK_PROFILES
    assert {row["location"] for row in status["profiles"]} == {"local"}
    assert {row["base_url"] for row in status["profiles"]} == {
        "http://127.0.0.1:1234/v1"
    }
    assert {row["model"] for row in status["profiles"]} == {
        "qwen/qwen3-vl-8b",
        "qwen/qwen3.5-9b",
    }
    profiles = {row["id"]: row for row in status["profiles"]}
    assert profiles["local-lmstudio-qwen3-vl-8b"]["adapter_backend"] == "builtin"
    text_profile = profiles["local-lmstudio-qwen3-5-9b"]
    assert text_profile["provider"] == "local_openai_compatible"
    assert text_profile["adapter_backend"] == "builtin"
    assert text_profile["provider_options"] == {
        "reasoning_effort": "none",
        "response_format": "text",
        "max_tokens": 1200,
    }
    assert {row["location"] for row in status["route_pools"]} == {"local_only"}
    assert all(
        row["default_location"] == "local"
        and row["remote_pool_id"] == ""
        for row in status["route_bindings"].values()
    )
    assert status["last_route_preset"]["remote_destinations"] == []
    assert status["last_route_preset"]["remote_profiles_selected"] is False
    assert status["last_route_preset"]["remote_requests_made"] is False
    assert status["last_route_preset"]["local_media_tasks"]["asr"]["primary"] == "sensevoice"
    assert (
        status["last_route_preset"]["local_media_tasks"]["ocr"]["primary"]
        == "ebook_markdown_pipeline"
    )

    for task in LOCAL_PRODUCTION_ROUTE_TASK_PROFILES:
        route = resolve_model_api_route(
            task,
            execution_location="local",
            settings_path=settings_path,
        )
        assert route["execution_location"] == "local"
        assert route["pool_location"] == "local_only"
        assert route["retry_policy"]["max_retries"] == 0


def test_local_production_installer_writes_isolated_secretless_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "local-production"
    result = install_local_production_preset(root)

    assert result["status"] == "installed"
    assert result["network_policy"]["remote_destinations"] == []
    assert result["network_policy"]["remote_requests_allowed"] is False
    assert result["network_policy"]["automatic_local_remote_fallback"] is False
    assert result["models"]["asr_primary"]["device"] == "cuda"
    assert result["models"]["ocr"]["provider"] == "ebook_markdown_pipeline"
    assert result["models"]["vision"]["model"] == "qwen/qwen3-vl-8b"
    assert result["models"]["text"]["model"] == "qwen/qwen3.5-9b"
    assert result["models"]["text"]["reasoning_effort"] == "none"
    assert result["models"]["text"]["response_format"] == "text"
    assert result["models"]["text"]["max_tokens"] == 1200
    assert result["models"]["text"]["adapter_backend"] == "builtin"
    assert not (root / "model-api-secrets.json").exists()

    settings = load_model_api_settings(root / "model-api-settings.json")
    assert settings["task_routes"] == LOCAL_PRODUCTION_ROUTE_TASK_PROFILES
    assert all(
        str(row["base_url"]).startswith("http://127.0.0.1:")
        for row in settings["profiles"]
    )
    pipeline = json.loads(
        (root / "video-knowledge-pipeline.json").read_text(encoding="utf-8")
    )
    assert pipeline["processing_profiles"]["default"] == LOCAL_PRODUCTION_ROUTE_PRESET_ID
    assert (
        pipeline["processing_profiles"][LOCAL_PRODUCTION_ROUTE_PRESET_ID][
            "data_export_allowed"
        ]
        is False
    )
    assert pipeline["ebook_pipeline"]["rapidocr_device"] == "cuda"
    assert pipeline["local_production"]["remote_requests_allowed"] is False


def test_local_summary_auto_execution_does_not_require_export_permission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "video_knowledge_pipeline.smart_summary_section_llm.processing_profile",
        lambda _name: {
            "text_llm_auto_execute": True,
            "data_export_allowed": False,
            "llm_preflight_call_threshold": 20,
            "llm_preflight_input_char_threshold": 120000,
        },
    )
    result = _profile_execution(
        tmp_path,
        profile_name=LOCAL_PRODUCTION_ROUTE_PRESET_ID,
        cfg={
            "provider": "openai_compatible",
            "adapter_backend": "proxy",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "qwen/qwen3.5-9b",
            "execution_location": "local",
        },
        candidates=[
            {
                "section_id": "s1",
                "title": "本地章节",
                "evidence": {"summary_sentences": ["本地证据"]},
            }
        ],
        max_prompt_chars=6000,
        explicit_execute=False,
        auto_from_profile=True,
    )

    assert result["status"] == "auto_execution_ready"
    assert result["auto_execute"] is True
    assert result["data_export_allowed"] is False
    assert result["local_execution"] is True
    assert result["data_export_required"] is False