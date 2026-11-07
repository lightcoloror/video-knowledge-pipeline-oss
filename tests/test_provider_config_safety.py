from __future__ import annotations

import pytest

from video_knowledge_pipeline.provider_config_safety import secretless_provider_config


def test_accepts_normal_provider_configuration() -> None:
    source = {
        "provider": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "model": "example-model",
        "timeout_seconds": 90,
        "extra_body": {"thinking": {"type": "disabled"}},
        "provider_options": {
            "enable_thinking": False,
            "thinking_budget": 256,
            "max_tokens": 256,
            "stream": True,
        },
    }

    result = secretless_provider_config(source)

    assert result == source
    assert result is not source


@pytest.mark.parametrize(
    "provider_config",
    [
        {"api_key": "not-allowed"},
        {"transport": {"headers": {"Authorization": "Bearer not-allowed"}}},
        {"transport": {"x_api_key": "not-allowed"}},
        {"provider_options": {"access_token": "not-allowed"}},
        {"provider_options": {"token_value": "not-allowed"}},
        {"base_url": "https://user:password@example.invalid/v1"},
        {"base_url": "https://example.invalid/v1?access_token=not-allowed"},
        {"custom": "Bearer not-allowed"},
        {"items": [{"password": "not-allowed"}]},
    ],
)
def test_rejects_inline_credentials_at_any_nesting_level(provider_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="runtime environment variables"):
        secretless_provider_config(provider_config)


def test_allows_empty_secret_named_placeholders() -> None:
    assert secretless_provider_config({"api_key": "", "token": None}) == {
        "api_key": "",
        "token": None,
    }
