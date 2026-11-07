from __future__ import annotations

import io
import urllib.error

import pytest

from video_knowledge_pipeline.text_llm_gateway import (
    _build_openai_compatible_text_body,
    _http_error_detail,
    extract_json_document,
    normalize_provider_base_url,
    openai_compatible_chat_completions_url,
    resolve_openai_compatible_api_base_url,
    text_llm_provider_smoke,
)


def test_vsummary_reused_base_url_normalization() -> None:
    assert normalize_provider_base_url("https://api.example.com/v1/chat/completions") == "https://api.example.com/v1"
    assert resolve_openai_compatible_api_base_url("https://api.example.com") == "https://api.example.com/v1"
    assert resolve_openai_compatible_api_base_url("https://ark.cn-beijing.volces.com/api/coding/v3") == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert openai_compatible_chat_completions_url({"base_url": "https://api.example.com"}) == "https://api.example.com/v1/chat/completions"
    assert openai_compatible_chat_completions_url({"base_url": "https://ark.cn-beijing.volces.com/api/coding/v3"}).endswith("/api/coding/v3/chat/completions")


def test_vsummary_reused_json_extraction_modes() -> None:
    assert extract_json_document('{"ok": true}', require_object=True) == {"ok": True}
    assert extract_json_document('```json\n{"ok": true}\n```', require_object=True) == {"ok": True}
    assert extract_json_document('prefix {"items": [{"x": 1}]} suffix', require_object=True) == {"items": [{"x": 1}]}
    with pytest.raises(ValueError):
        extract_json_document('no json here', require_object=True)


def test_text_llm_provider_smoke_defaults_to_plan_without_secret_leak() -> None:
    result = text_llm_provider_smoke(
        {"provider": "volcengine_coding_plan", "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3?key=secret", "model": "ark-code-latest"},
        execute=False,
    )
    assert result["status"] == "planned"
    assert result["ok"] is True
    assert result["provider"]["api_key_configured"] is False
    assert "secret" not in result["request_plan"]["url"]
    assert result["source_reuse"]["project"] == "alpha03123/vsummary"

def test_volcengine_coding_plan_resolves_as_normal_text_llm(monkeypatch) -> None:
    from video_knowledge_pipeline.text_llm_gateway import resolve_text_provider_config

    monkeypatch.setenv("LLM_API_KEY", "ark-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    monkeypatch.setenv("LLM_MODEL", "ark-code-latest")

    cfg = resolve_text_provider_config({"provider": "volcengine_coding_plan"})

    assert cfg["provider"] == "volcengine_coding_plan"
    assert cfg["api_key"] == "ark-key"
    assert cfg["model"] == "ark-code-latest"
    assert cfg["base_url"] == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert openai_compatible_chat_completions_url(cfg) == "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"

def test_text_llm_preserves_volcengine_thinking_disabled(monkeypatch) -> None:
    from video_knowledge_pipeline.text_llm_gateway import resolve_text_provider_config

    monkeypatch.setenv("LLM_API_KEY", "ark-key")
    cfg = resolve_text_provider_config(
        {
            "provider": "volcengine_coding_plan",
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "model": "glm-latest",
            "thinking": {"type": "disabled"},
        }
    )

    body = _build_openai_compatible_text_body(
        cfg=cfg,
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
        response_format=None,
        max_tokens=32,
    )

    assert cfg["thinking"] == {"type": "disabled"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["model"] == "glm-latest"


def test_text_llm_disable_thinking_shorthand(monkeypatch) -> None:
    from video_knowledge_pipeline.text_llm_gateway import resolve_text_provider_config

    monkeypatch.setenv("LLM_API_KEY", "ark-key")
    cfg = resolve_text_provider_config(
        {
            "provider": "volcengine_coding_plan",
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "model": "kimi-k2.7-code",
            "disable_thinking": True,
        }
    )

    assert cfg["thinking"] == {"type": "disabled"}

def test_text_llm_merges_route_locked_reasoning_effort_into_request_body() -> None:
    from video_knowledge_pipeline.text_llm_gateway import resolve_text_provider_config

    cfg = resolve_text_provider_config(
        {
            "provider": "local_openai_compatible",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "qwen/qwen3.5-9b",
            "provider_options": {
                "reasoning_effort": "none",
                "response_format": "text",
                "max_tokens": 1200,
            },
        }
    )
    body = _build_openai_compatible_text_body(
        cfg=cfg,
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
        response_format=None,
        max_tokens=4096,
    )

    assert cfg["provider"] == "local_vlm"
    assert body["reasoning_effort"] == "none"
    assert body["response_format"] == {"type": "text"}
    assert body["max_tokens"] == 1200

def test_http_error_detail_keeps_provider_message_but_not_request_payload() -> None:
    exc = urllib.error.HTTPError(
        "http://127.0.0.1:1234/v1/chat/completions",
        400,
        "Bad Request",
        {},
        io.BytesIO(b'{"error":{"message":"response_format must be text\\n"}}'),
    )

    assert _http_error_detail(exc) == "http_400: response_format must be text"
