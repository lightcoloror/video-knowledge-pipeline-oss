from __future__ import annotations

from pathlib import Path

import pytest

from video_knowledge_pipeline.model_api_settings import upsert_model_api_profile
from video_knowledge_pipeline.trusted_model_connector import execute_local_model_task


def _configure_local_text_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = tmp_path / "model-api-settings.json"
    secrets = tmp_path / "model-api-secrets.json"
    monkeypatch.setenv("VKP_MODEL_API_SETTINGS_PATH", str(settings))
    monkeypatch.setenv("VKP_MODEL_API_SECRETS_PATH", str(secrets))
    upsert_model_api_profile(
        {
            "id": "local-text",
            "name": "Local text",
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:9001/v1",
            "model": "local-text-model",
            "location": "local",
            "capabilities": ["text"],
        },
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )


def test_local_execution_uses_only_loopback_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_local_text_route(monkeypatch, tmp_path)
    artifact = tmp_path / "source.md"
    artifact.write_text("local evidence", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_call(task: str, **kwargs: object) -> dict[str, object]:
        captured.update({"task": task, **kwargs})
        return {"ok": True, "status": "completed", "response": {"content": "local result"}}

    monkeypatch.setattr("video_knowledge_pipeline.trusted_model_connector.model_task_api_call", fake_call)
    result = execute_local_model_task(
        "smart_summary_rewrite",
        [artifact],
        instructions="summarise",
        write=False,
    )

    assert result["ok"] is True
    assert result["remote_requests_made"] is False
    assert result["route"]["execution_location"] == "local"
    assert captured["provider_config"]["base_url"] == "http://127.0.0.1:9001/v1"
    assert "local evidence" in str(captured["input_text"])


def test_local_execution_never_falls_back_to_remote(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VKP_MODEL_API_SETTINGS_PATH", str(tmp_path / "missing.json"))
    called = False

    def fake_call(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"ok": True}

    monkeypatch.setattr("video_knowledge_pipeline.trusted_model_connector.model_task_api_call", fake_call)
    artifact = tmp_path / "source.md"
    artifact.write_text("local evidence", encoding="utf-8")

    result = execute_local_model_task("smart_summary_rewrite", [artifact], write=False)

    assert result["status"] == "local_gateway_unavailable"
    assert result["remote_requests_made"] is False
    assert called is False