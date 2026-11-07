from __future__ import annotations

from pathlib import Path

import pytest

from video_knowledge_pipeline.model_api_settings import (
    resolve_model_api_route,
    upsert_model_api_profile,
    validate_model_api_profile,
)
from video_knowledge_pipeline.model_gateway import render_litellm_config
from video_knowledge_pipeline.model_provider_catalog import (
    provider_catalog_status,
    provider_usage_scope,
    providers_for_capability,
    resolve_litellm_provider,
)


def _profile(
    profile_id: str,
    *,
    provider: str,
    model: str,
    base_url: str,
    capabilities: list[str],
    litellm_provider: str = "",
) -> dict[str, object]:
    return {
        "id": profile_id,
        "name": profile_id,
        "provider": provider,
        "litellm_provider": litellm_provider,
        "adapter_backend": "proxy",
        "location": "remote",
        "capabilities": capabilities,
        "base_url": base_url,
        "model": model,
        "timeout_seconds": 120,
        "enabled": True,
    }


def test_catalog_covers_major_online_families_and_has_generic_extension() -> None:
    result = provider_catalog_status()
    providers = {row["provider"]: row for row in result["providers"]}

    assert result["provider_count"] >= 30
    assert result["extension_provider"] == "litellm_native"
    assert result["secrets_in_catalog"] is False
    for provider in (
        "openai",
        "anthropic",
        "gemini",
        "deepseek",
        "dashscope",
        "volcengine_ark",
        "openrouter",
        "groq_asr",
        "deepgram_asr",
        "fireworks_asr",
        "mistral_asr",
        "mistral",
        "litellm_native",
        "local_openai_compatible",
        "speaches_openai_compatible",
    ):
        assert provider in providers
    assert providers["volcengine_coding_plan"]["default_model"] == "deepseek-v4-pro"
    assert providers["volcengine_coding_plan"]["default_capabilities"] == ["text"]
    assert providers["volcengine_coding_plan"]["allowed_provider_options"] == [
        "thinking_mode",
        "response_format",
        "max_tokens",
        "strip_reasoning_tags",
        "strip_json_fences",
    ]
    assert providers["volcengine_coding_plan"]["required_provider_options"] == []
    assert providers["volcengine_coding_plan"]["usage_scope"] == "general_model_api"
    assert providers["volcengine_ark"]["default_base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert providers["volcengine_ark"]["default_capabilities"] == ["text"]
    assert providers["volcengine_ark"]["allowed_provider_options"] == [
        "thinking_mode",
        "response_format",
        "max_tokens",
        "strip_reasoning_tags",
        "strip_json_fences",
    ]
    assert providers["volcengine_ark"]["usage_scope"] == "general_model_api"
    assert provider_usage_scope("volcengine_coding_plan") == "general_model_api"
    assert provider_usage_scope("volcengine_ark") == "general_model_api"
    assert providers["siliconflow"]["allowed_provider_options"] == [
        "enable_thinking",
        "thinking_budget",
        "reasoning_effort",
        "response_format",
        "max_tokens",
        "stream",
        "strip_reasoning_tags",
        "strip_json_fences",
    ]
    assert providers["siliconflow"]["required_provider_options"] == []
    assert providers["siliconflow"]["supported_capabilities"] == [
        "text",
        "vision",
        "asr",
    ]
    assert providers["groq"]["default_model"] == "qwen/qwen3.6-27b"
    assert providers["groq"]["default_capabilities"] == ["text", "vision"]
    assert providers["groq"]["allowed_provider_options"] == [
        "reasoning_effort",
        "reasoning_format",
        "strip_reasoning_tags",
        "strip_json_fences",
    ]
    assert providers["groq_asr"]["default_model"] == "whisper-large-v3-turbo"
    assert providers["groq_asr"]["litellm_provider"] == "openai"
    assert providers["groq_asr"]["allowed_provider_options"] == [
        "asr_timestamp_granularity"
    ]
    assert providers["deepgram_asr"]["supported_capabilities"] == ["asr"]
    assert providers["mistral"]["supported_capabilities"] == ["ocr"]


def test_groq_asr_legacy_profile_gets_hashed_word_timestamp_contract() -> None:
    profile = _profile(
        "groq-asr-legacy",
        provider="groq_asr",
        litellm_provider="groq",
        base_url="https://api.groq.com/openai/v1",
        model="whisper-large-v3-turbo",
        capabilities=["asr"],
    )

    validated = validate_model_api_profile(profile, [])

    assert validated["profile"]["litellm_provider"] == "openai"
    assert validated["profile"]["provider_options"] == {
        "asr_timestamp_granularity": "word"
    }

    profile["provider_options"] = {"asr_timestamp_granularity": "sentence"}
    with pytest.raises(ValueError, match="asr_timestamp_granularity"):
        validate_model_api_profile(profile, [])

def test_catalog_filters_profiles_by_capability_and_location() -> None:
    remote_asr = providers_for_capability("asr", location="remote")
    local_asr = providers_for_capability("asr", location="local")

    assert "groq_asr" in remote_asr
    assert "deepgram_asr" in remote_asr
    assert "openai_compatible" in remote_asr
    assert "speaches_openai_compatible" not in remote_asr
    assert local_asr == ["speaches_openai_compatible"]
    assert "mistral" in providers_for_capability("ocr", location="remote")
    with pytest.raises(ValueError, match="unsupported provider capability"):
        providers_for_capability("video")
    with pytest.raises(ValueError, match="unsupported provider location"):
        providers_for_capability("text", location="automatic")


def test_catalog_locks_preset_prefix_and_validates_native_extension() -> None:
    assert resolve_litellm_provider("anthropic") == "anthropic"
    with pytest.raises(ValueError, match="must match"):
        resolve_litellm_provider("anthropic", "openai")
    assert resolve_litellm_provider("mistral_asr") == "openai"
    assert resolve_litellm_provider("mistral_asr", "mistral") == "openai"
    assert resolve_litellm_provider("groq_asr") == "openai"
    assert resolve_litellm_provider("groq_asr", "groq") == "openai"
    with pytest.raises(ValueError, match="require a valid"):
        resolve_litellm_provider("litellm_native", "")
    assert resolve_litellm_provider("litellm_native", "ovhcloud") == "ovhcloud"


def test_profile_defaults_and_route_revision_include_litellm_provider(
    tmp_path: Path,
) -> None:
    validated = validate_model_api_profile(
        {
            "id": "anthropic-main",
            "name": "Anthropic main",
            "provider": "anthropic",
            "adapter_backend": "proxy",
            "location": "remote",
            "base_url": "https://api.anthropic.com",
            "model": "operator-selected-model",
            "enabled": True,
        },
        ["text_llm"],
    )["profile"]
    assert validated["litellm_provider"] == "anthropic"
    assert validated["capabilities"] == ["text"]

    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    profile = _profile(
        "native-main",
        provider="litellm_native",
        litellm_provider="anthropic",
        base_url="https://provider.example/v1",
        model="selected-model",
        capabilities=["text"],
    )
    upsert_model_api_profile(
        profile,
        tasks=["text_llm"],
        settings_path=settings,
        secrets_path=secrets,
    )
    first = resolve_model_api_route(
        "text_llm", execution_location="remote", settings_path=settings
    )
    profile["litellm_provider"] = "openrouter"
    upsert_model_api_profile(
        profile,
        tasks=["text_llm"],
        settings_path=settings,
        secrets_path=secrets,
    )
    second = resolve_model_api_route(
        "text_llm", execution_location="remote", settings_path=settings
    )

    assert first["deployments"][0]["litellm_provider"] == "anthropic"
    assert second["deployments"][0]["litellm_provider"] == "openrouter"
    assert first["route_revision"] != second["route_revision"]


def test_gateway_renders_catalog_prefixes_for_text_asr_and_ocr(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    output = tmp_path / "litellm.yaml"
    rows = (
        (
            _profile(
                "anthropic-text",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                model="operator-text-model",
                capabilities=["text"],
            ),
            ["text_llm"],
        ),
        (
            _profile(
                "groq-asr",
                provider="groq_asr",
                base_url="https://api.groq.com/openai/v1",
                model="whisper-large-v3",
                capabilities=["asr"],
            ),
            ["asr"],
        ),
        (
            _profile(
                "mistral-ocr",
                provider="mistral",
                base_url="https://api.mistral.ai/v1",
                model="mistral-ocr-latest",
                capabilities=["ocr"],
            ),
            ["ocr"],
        ),
    )
    for profile, tasks in rows:
        upsert_model_api_profile(
            profile,
            tasks=tasks,
            settings_path=settings,
            secrets_path=secrets,
        )

    result = render_litellm_config(
        settings_path=settings,
        secrets_path=secrets,
        output_path=output,
        write=True,
    )
    rendered = output.read_text(encoding="utf-8")

    assert result["model_count"] == 3
    assert 'model: "anthropic/operator-text-model"' in rendered
    assert 'model: "openai/whisper-large-v3"' in rendered
    assert 'model: "mistral/mistral-ocr-latest"' in rendered
    assert 'mode: "audio_transcription"' in rendered
    assert 'mode: "ocr"' in rendered


def test_gateway_renders_mistral_asr_with_openai_transcription_transport(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    output = tmp_path / "litellm.yaml"
    upsert_model_api_profile(
        _profile(
            "mistral-asr",
            provider="mistral_asr",
            base_url="https://api.mistral.ai/v1",
            model="voxtral-mini-2602",
            capabilities=["asr"],
        ),
        tasks=["asr"],
        settings_path=settings,
        secrets_path=secrets,
    )

    result = render_litellm_config(
        settings_path=settings,
        secrets_path=secrets,
        output_path=output,
        write=True,
    )
    rendered = output.read_text(encoding="utf-8")

    assert result["model_count"] == 1
    assert 'model: "openai/voxtral-mini-2602"' in rendered
    assert 'api_base: "https://api.mistral.ai/v1"' in rendered
    assert 'mode: "audio_transcription"' in rendered


def test_catalog_declares_advanced_cloud_auth_without_secret_values() -> None:
    providers = {
        row["provider"]: row for row in provider_catalog_status()["providers"]
    }

    assert providers["azure_openai"]["litellm_provider"] == "azure"
    assert providers["azure_openai"]["required_provider_options"] == ["api_version"]
    assert providers["azure_openai_entra"]["auth_mode"] == "external_environment"
    assert providers["vertex_ai"]["litellm_provider"] == "vertex_ai"
    assert providers["vertex_ai_ocr"]["supported_capabilities"] == ["ocr"]
    assert providers["bedrock"]["required_provider_options"] == ["aws_region_name"]
    bindings = providers["bedrock"]["environment_bindings"]
    assert {row["env"] for row in bindings} == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    }
    assert "secret_value" not in str(providers)


def test_provider_options_are_allowlisted_and_change_route_revision(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    profile = _profile(
        "azure-main",
        provider="azure_openai",
        base_url="https://example-resource.openai.azure.com",
        model="deployment-name",
        capabilities=["text"],
    )
    profile["provider_options"] = {"api_version": "2024-08-01-preview"}
    upsert_model_api_profile(
        profile,
        tasks=["text_llm"],
        settings_path=settings,
        secrets_path=secrets,
    )
    first = resolve_model_api_route(
        "text_llm", execution_location="remote", settings_path=settings
    )

    profile["provider_options"] = {"api_version": "2025-01-01-preview"}
    upsert_model_api_profile(
        profile,
        tasks=["text_llm"],
        settings_path=settings,
        secrets_path=secrets,
    )
    second = resolve_model_api_route(
        "text_llm", execution_location="remote", settings_path=settings
    )

    assert first["deployments"][0]["provider_options"]["api_version"] == "2024-08-01-preview"
    assert first["route_revision"] != second["route_revision"]
    with pytest.raises(ValueError, match="safe non-secret"):
        validate_model_api_profile(
            {
                **profile,
                "provider_options": {"api_key": "must-not-be-stored"},
            },
            ["text_llm"],
        )
    with pytest.raises(ValueError, match="catalog-controlled"):
        validate_model_api_profile(
            {
                **profile,
                "provider_options": {"api_version": "os.environ/UNTRUSTED"},
            },
            ["text_llm"],
        )


def test_volcengine_runtime_options_are_optional_and_strictly_validated() -> None:
    profile = _profile(
        "ark-minimax-m3",
        provider="volcengine_coding_plan",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="minimax-m3",
        capabilities=["text"],
    )
    profile["provider_options"] = {
        "thinking_mode": "disabled",
        "response_format": "json_object",
        "max_tokens": 1024,
    }

    validated = validate_model_api_profile(profile, ["summary_rewrite"])

    assert validated["profile"]["provider_options"] == {
        "max_tokens": 1024,
        "response_format": "json_object",
        "thinking_mode": "disabled",
    }
    assert validated["profile"]["required_provider_options"] == []
    with pytest.raises(ValueError, match="thinking_mode"):
        validate_model_api_profile(
            {**profile, "provider_options": {"thinking_mode": "sometimes"}},
            ["summary_rewrite"],
        )
    with pytest.raises(ValueError, match="response_format"):
        validate_model_api_profile(
            {**profile, "provider_options": {"response_format": "xml"}},
            ["summary_rewrite"],
        )


def test_siliconflow_legacy_thinking_mode_is_safely_migrated() -> None:
    profile = _profile(
        "sf-thinking",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V4-Pro",
        capabilities=["text"],
    )
    profile["provider_options"] = {"thinking_mode": "disabled"}

    validated = validate_model_api_profile(profile, [])

    assert validated["profile"]["provider_options"] == {"enable_thinking": False}


def test_siliconflow_rejects_ambiguous_legacy_thinking_mode() -> None:
    profile = _profile(
        "sf-thinking-auto",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V4-Pro",
        capabilities=["text"],
    )
    profile["provider_options"] = {"thinking_mode": "auto"}

    with pytest.raises(ValueError, match="enable_thinking boolean"):
        validate_model_api_profile(profile, [])




def test_siliconflow_runtime_options_are_locked_and_strictly_validated(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    profile = _profile(
        "siliconflow-qwen",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3.5-397B-A17B",
        capabilities=["text"],
    )
    profile["provider_options"] = {
        "enable_thinking": False,
        "thinking_budget": 256,
        "max_tokens": 256,
        "stream": True,
    }

    validated = validate_model_api_profile(profile, ["summary_rewrite"])
    assert validated["profile"]["provider_options"] == {
        "enable_thinking": False,
        "max_tokens": 256,
        "stream": True,
        "thinking_budget": 256,
    }
    upsert_model_api_profile(
        profile,
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )
    first = resolve_model_api_route(
        "summary_rewrite",
        execution_location="remote",
        settings_path=settings,
    )

    profile["provider_options"]["max_tokens"] = 128
    upsert_model_api_profile(
        profile,
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )
    second = resolve_model_api_route(
        "summary_rewrite",
        execution_location="remote",
        settings_path=settings,
    )

    assert first["route_revision"] != second["route_revision"]
    assert first["deployments"][0]["provider_options"]["max_tokens"] == 256
    with pytest.raises(ValueError, match="enable_thinking must be a boolean"):
        validate_model_api_profile(
            {**profile, "provider_options": {"enable_thinking": "false"}},
            ["summary_rewrite"],
        )
    with pytest.raises(ValueError, match="thinking_budget must be between"):
        validate_model_api_profile(
            {**profile, "provider_options": {"thinking_budget": 127}},
            ["summary_rewrite"],
        )


def test_gateway_keeps_siliconflow_request_options_runtime_only(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    output = tmp_path / "litellm.yaml"
    profile = _profile(
        "siliconflow-qwen",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3.5-397B-A17B",
        capabilities=["text"],
    )
    profile["provider_options"] = {
        "enable_thinking": False,
        "thinking_budget": 256,
        "max_tokens": 256,
        "stream": True,
    }
    upsert_model_api_profile(
        profile,
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )

    render_litellm_config(
        settings_path=settings,
        secrets_path=secrets,
        output_path=output,
    )
    rendered = output.read_text(encoding="utf-8")

    assert 'model: "openai/Qwen/Qwen3.5-397B-A17B"' in rendered
    assert "enable_thinking:" not in rendered
    assert "thinking_budget:" not in rendered
    assert "max_tokens:" not in rendered
    assert "stream:" not in rendered

def test_gateway_renders_external_environment_refs_and_blocks_missing_auth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    output = tmp_path / "litellm.yaml"
    profile = _profile(
        "vertex-main",
        provider="vertex_ai",
        base_url="https://us-central1-aiplatform.googleapis.com",
        model="gemini-2.5-flash",
        capabilities=["vision"],
    )
    profile["provider_options"] = {
        "vertex_project": "example-project",
        "vertex_location": "us-central1",
    }
    upsert_model_api_profile(
        profile,
        tasks=["semantic_frame"],
        settings_path=settings,
        secrets_path=secrets,
    )
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    blocked = render_litellm_config(
        settings_path=settings,
        secrets_path=secrets,
        output_path=output,
        write=True,
    )
    rendered = output.read_text(encoding="utf-8")

    assert blocked["ready_for_start"] is False
    assert blocked["credential_blockers"] == [
        {
            "profile_id": "vertex-main",
            "status": "missing_environment",
            "missing_environment": ["GOOGLE_APPLICATION_CREDENTIALS"],
        }
    ]
    assert 'vertex_project: "example-project"' in rendered
    assert 'vertex_credentials: "os.environ/GOOGLE_APPLICATION_CREDENTIALS"' in rendered
    assert "credential-json" not in rendered

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "credential-json")
    ready = render_litellm_config(
        settings_path=settings,
        secrets_path=secrets,
        output_path=output,
        write=False,
    )
    assert ready["ready_for_start"] is True
    assert ready["credential_blockers"] == []
