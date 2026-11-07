from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_knowledge_pipeline import model_api_settings as settings_module
from video_knowledge_pipeline import model_provider_probe as probe_module
from video_knowledge_pipeline.model_api_settings import install_model_api_onboarding_bundle
from video_knowledge_pipeline.model_provider_probe import probe_model_api_onboarding_bundle


class _Response:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]


@pytest.fixture
def fake_secret_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        settings_module,
        "_protect_secret",
        lambda value: "cipher:" + value.encode("utf-8").hex(),
    )
    monkeypatch.setattr(
        settings_module,
        "_unprotect_secret",
        lambda value: bytes.fromhex(value.removeprefix("cipher:")).decode("utf-8"),
    )


def test_catalog_probe_is_planned_without_network(
    tmp_path: Path,
    fake_secret_codec: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    install_model_api_onboarding_bundle(
        "modelscope",
        api_key="secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    monkeypatch.setattr(
        probe_module,
        "_open_catalog_request",
        lambda *_args, **_kwargs: pytest.fail("planned probe must not call the network"),
    )

    result = probe_model_api_onboarding_bundle(
        "modelscope",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    assert result["status"] == "planned"
    assert result["network_calls"] == 0
    assert result["model_inference_calls"] == 0
    assert result["artifact_reads"] == 0
    assert result["artifact_uploads"] == 0


def test_modelscope_catalog_probe_verifies_exact_models(
    tmp_path: Path,
    fake_secret_codec: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    install_model_api_onboarding_bundle(
        "modelscope",
        api_key="secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    monkeypatch.setattr(
        probe_module,
        "_open_catalog_request",
        lambda *_args, **_kwargs: _Response(
            {"data": [{"id": "ZhipuAI/GLM-5.2"}, {"id": "deepseek-ai/DeepSeek-V4-Pro"}]}
        ),
    )

    result = probe_model_api_onboarding_bundle(
        "modelscope",
        execute=True,
        include_model_ids=True,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    assert result["status"] == "verified"
    assert result["network_calls"] == 1
    assert result["model_inference_calls"] == 0
    assert all(row["visible"] is True for row in result["catalog_entries"])
    assert result["catalog_model_ids"] == [
        "deepseek-ai/DeepSeek-V4-Pro",
        "ZhipuAI/GLM-5.2",
    ]
    assert {row["client_endpoint"] for row in result["invocation_contracts"]} == {"/v1/chat/completions"}


def test_ark_catalog_probe_keeps_coding_plan_aliases_out_of_missing_models(
    tmp_path: Path,
    fake_secret_codec: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    install_model_api_onboarding_bundle(
        "ark_coding_plan",
        api_key="secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    monkeypatch.setattr(
        probe_module,
        "_open_catalog_request",
        lambda *_args, **_kwargs: _Response(
            {
                "data": [
                    {"id": "deepseek-v4-pro-260425"},
                    {"id": "deepseek-v4-flash-260425"},
                    {"id": "glm-5-2-260617"},
                ]
            }
        ),
    )

    result = probe_model_api_onboarding_bundle(
        "ark_coding_plan",
        execute=True,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    aliases = [row for row in result["catalog_entries"] if row["model"].startswith("kimi-")]
    assert result["status"] == "verified"
    assert result["enabled_models_missing"] == []
    assert aliases
    assert all(row["visible"] is False for row in aliases)
    assert all(row["catalog_visibility_required"] is False for row in aliases)

def test_gemini_catalog_probe_uses_header_not_query_key(
    tmp_path: Path,
    fake_secret_codec: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    install_model_api_onboarding_bundle(
        "google_gemini",
        api_key="gemini-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    captured: dict[str, Any] = {}

    def fake_open(request: Any, *, timeout_seconds: int) -> _Response:
        captured["url"] = request.full_url
        captured["key"] = request.get_header("X-goog-api-key")
        captured["timeout"] = timeout_seconds
        return _Response({"models": [{"name": "models/gemini-3.6-flash"}, {"name": "models/gemini-3.5-flash-lite"}]})

    monkeypatch.setattr(probe_module, "_open_catalog_request", fake_open)
    result = probe_model_api_onboarding_bundle(
        "google_gemini",
        execute=True,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    assert result["status"] == "verified"
    assert captured["key"] == "gemini-secret"
    assert "gemini-secret" not in captured["url"]
    assert result["invocation_contracts"][0]["provider_endpoint"].endswith(":generateContent")
