from __future__ import annotations

import video_knowledge_pipeline.online_model_gateway as online_model_gateway

from video_knowledge_pipeline.online_model_gateway import (
    MODEL_TYPES,
    _should_use_litellm,
    asr_transcriptions_url,
    call_litellm_chat,
    online_model_api_call,
    online_model_api_matrix,
)


def test_local_qwen_auto_backend_uses_native_openai_compatible_adapter() -> None:
    assert _should_use_litellm({"provider": "local_qwen_vl"}) is False
    assert _should_use_litellm({"provider": "local_qwen_vl", "adapter_backend": "litellm"}) is True


def test_missing_backend_uses_legacy_compatibility_without_implicit_fallback() -> None:
    config = {"provider": "openai_compatible"}

    assert online_model_gateway._adapter_backend(config) == "legacy"
    assert online_model_gateway._should_fallback_from_litellm(
        config,
        {"ok": False, "error": "litellm failed"},
    ) is False
    plan = online_model_gateway._adapter_reuse_plan(config)
    assert plan["fallback_backend"] == "none"
    assert plan["legacy_adapter"] == "built_in_openai_compatible_urllib"


def test_explicit_auto_backend_retains_auditable_same_route_fallback() -> None:
    config = {
        "provider": "openai_compatible",
        "adapter_backend": "auto",
    }

    assert online_model_gateway._adapter_backend(config) == "auto_litellm_then_builtin"
    assert online_model_gateway._should_fallback_from_litellm(
        config,
        {"ok": False, "error": "litellm failed"},
    ) is True
    assert (
        online_model_gateway._adapter_reuse_plan(config)["fallback_backend"]
        == "built_in_openai_compatible_urllib"
    )


def test_litellm_chat_rejects_empty_or_reasoning_only_output(monkeypatch) -> None:
    class FakeLiteLLM:
        @staticmethod
        def completion(**kwargs):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "reasoning_content": "private reasoning"},
                    }
                ]
            }

    monkeypatch.setitem(__import__("sys").modules, "litellm", FakeLiteLLM())
    result = call_litellm_chat(provider_config={"model": "local"}, messages=[])

    assert result["ok"] is False
    assert result["content"] == ""
    assert result["error"].startswith("empty_content_reasoning_only")


def test_new_gemini_models_omit_deprecated_temperature(monkeypatch) -> None:
    calls = []

    class FakeLiteLLM:
        @staticmethod
        def completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(__import__("sys").modules, "litellm", FakeLiteLLM())

    for model in ("gemini-3.6-flash", "gemini-3.5-flash-lite"):
        result = call_litellm_chat(
            provider_config={"provider": "gemini", "model": model},
            messages=[{"role": "user", "content": "test"}],
            temperature=0,
        )
        assert result["ok"] is True

    assert all("temperature" not in call for call in calls)


def test_online_model_api_matrix_lists_all_interfaces_without_execution() -> None:
    result = online_model_api_matrix(write=False)

    assert result["ok"] is True
    assert result["status"] == "planned"
    assert result["model_types"] == list(MODEL_TYPES)
    assert {row["model_type"] for row in result["providers"]} == set(MODEL_TYPES)
    assert all("api_key" not in row.get("provider", {}) for row in result["providers"])


def test_online_model_api_text_preview_does_not_execute() -> None:
    result = online_model_api_call("summary_rewrite", input_text="测试文本", write=False)

    assert result["ok"] is True
    assert result["status"] == "planned"
    assert result["execute"] is False
    assert result["request_plan"]["interface"] == "openai_chat_completions"
    assert result["request_plan"]["input_text_chars"] == 4


def test_online_model_api_vision_preview_accepts_image_paths_without_file_check() -> None:
    result = online_model_api_call("ocr", image_paths=["D:/missing-frame.png"], write=False)

    assert result["ok"] is True
    assert result["status"] == "planned"
    assert result["request_plan"]["interface"] == "vision_chat_completions_or_gemini"
    assert result["request_plan"]["image_count"] == 1


def test_online_model_api_asr_preview_builds_audio_transcription_plan() -> None:
    result = online_model_api_call("asr", audio_path="D:/missing.wav", write=False)

    assert result["ok"] is True
    assert result["status"] == "planned"
    assert result["request_plan"]["interface"] == "openai_audio_transcriptions"
    assert result["request_plan"]["audio_exists"] is False


def test_asr_transcriptions_url_normalizes_common_openai_compatible_bases() -> None:
    assert asr_transcriptions_url({"base_url": "https://api.example.com/v1"}) == "https://api.example.com/v1/audio/transcriptions"
    assert asr_transcriptions_url({"base_url": "https://api.example.com"}) == "https://api.example.com/v1/audio/transcriptions"
    assert asr_transcriptions_url({"base_url": "https://api.example.com/v1/audio/transcriptions"}) == "https://api.example.com/v1/audio/transcriptions"


def test_online_model_api_volcengine_profile_covers_vision_and_text(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "ark-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    monkeypatch.setenv("LLM_MODEL", "ark-code-latest")
    provider_config = {"provider": "volcengine_coding_plan"}

    text_result = online_model_api_call("summary_rewrite", provider_config=provider_config, input_text="测试", write=False)
    vision_result = online_model_api_call("semantic_frame", provider_config=provider_config, image_paths=["D:/frame.png"], write=False)

    assert text_result["request_plan"]["provider"]["provider"] == "volcengine_coding_plan"
    assert text_result["request_plan"]["provider"]["model"] == "ark-code-latest"
    assert text_result["request_plan"]["url"] == "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
    assert vision_result["request_plan"]["provider"]["provider"] == "volcengine_coding_plan"
    assert vision_result["request_plan"]["provider"]["model"] == "ark-code-latest"
    assert vision_result["request_plan"]["image_count"] == 1

def test_online_model_api_allows_coding_plan_to_reach_adapter(
    monkeypatch, tmp_path
) -> None:
    provider_config = {
        "provider": "volcengine_coding_plan",
        "api_key": "ark-key",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
    }
    adapter_called = False

    def allowed_adapter(**kwargs):
        nonlocal adapter_called
        adapter_called = True
        return {"ok": True, "status": "completed", "content": "adapter reached"}

    monkeypatch.setenv("VKP_MODEL_API_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(online_model_gateway, "call_litellm_chat", allowed_adapter)
    monkeypatch.setattr(online_model_gateway, "call_openai_compatible_text", allowed_adapter)

    result = online_model_api_call(
        "text_llm",
        provider_config=provider_config,
        input_text="测试",
        execute=True,
        write=False,
    )

    assert result["ok"] is True
    assert result["content"] == "adapter reached"
    assert adapter_called is True

def test_proxy_resolvers_preserve_route_metadata_without_reading_provider_env_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-main-proxy-config")
    proxy = {
        "provider": "openai_compatible",
        "adapter_backend": "proxy",
        "base_url": "https://provider.example/v1",
        "model": "remote-model",
        "location": "remote",
        "execution_location": "remote",
        "route_id": "pool-remote",
        "route_revision": "a" * 64,
        "virtual_model": "vkp-remote-text-test-aaaaaaaaaaaa",
        "consent_id": "consent-test",
    }

    resolved = [
        online_model_gateway.resolve_provider_config(proxy),
        online_model_gateway.resolve_text_provider_config(proxy),
        online_model_gateway.resolve_asr_provider_config(proxy),
    ]

    assert all(row["adapter_backend"] == "proxy" for row in resolved)
    assert all(row["route_revision"] == "a" * 64 for row in resolved)
    assert all(row["consent_id"] == "consent-test" for row in resolved)
    assert all(row.get("api_key") == "" for row in resolved)

def test_proxy_backend_never_falls_back_to_legacy_adapter(monkeypatch) -> None:
    legacy_called = False

    def fake_legacy(**kwargs):
        nonlocal legacy_called
        legacy_called = True
        return {"ok": True, "content": "legacy"}

    monkeypatch.setattr(
        online_model_gateway,
        "model_runtime_request",
        lambda *args, **kwargs: {
            "ok": False,
            "status": "gateway_unavailable",
            "content": "",
            "error": "proxy down",
        },
    )
    monkeypatch.setattr(online_model_gateway, "call_openai_compatible_text", fake_legacy)

    result = online_model_api_call(
        "summary_rewrite",
        provider_config={
            "provider": "openai_compatible",
            "adapter_backend": "proxy",
            "base_url": "https://provider.example/v1",
            "model": "remote-model",
        },
        input_text="source",
        execute=True,
        write=False,
    )

    assert result["status"] == "gateway_unavailable"
    assert result["fallback_from"] == ""
    assert legacy_called is False
