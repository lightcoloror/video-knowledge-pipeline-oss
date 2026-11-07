from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline import online_model_gateway
from video_knowledge_pipeline.model_task_gateway import (
    MODEL_TASKS,
    model_task_api_call,
    model_task_coverage_audit,
)


def test_model_task_coverage_writes_machine_and_human_reports(tmp_path: Path) -> None:
    result = model_task_coverage_audit(output_dir=tmp_path)

    assert result["status"] == "complete"
    assert result["counts"]["total"] == len(MODEL_TASKS)
    assert result["counts"]["unified"] >= 4
    assert result["counts"]["deferred"] == 0
    assert (tmp_path / "model-task-coverage.json").exists()
    assert (tmp_path / "model-task-coverage.md").exists()
    term = next(row for row in result["rows"] if row["task"] == "term_arbitration")
    assert term["gateway"] == "model_task_gateway"
    assert term["online_api_adapter"] is True


def test_model_task_gateway_preserves_structured_text_messages(monkeypatch) -> None:
    captured: dict = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "error": "", "content": '{"decisions": []}'}

    monkeypatch.setattr(online_model_gateway, "_should_use_litellm", lambda cfg: False)
    monkeypatch.setattr(online_model_gateway, "call_openai_compatible_text", fake_call)
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Review the evidence."},
    ]

    result = model_task_api_call(
        "term_arbitration",
        provider_config={"provider": "fixture", "model": "fixture-text"},
        messages=messages,
        execute=True,
        response_format={"type": "json_object"},
        max_tokens=1200,
    )

    assert result["ok"] is True
    assert result["task"] == "term_arbitration"
    assert captured["messages"] == messages
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 1200


def test_native_video_task_routes_to_provider_capability(monkeypatch) -> None:
    captured: dict = {}

    def fake_video(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "error": "", "content": "native video result"}

    monkeypatch.setattr(online_model_gateway, "call_gemini_video", fake_video)
    result = model_task_api_call(
        "native_video_segment",
        provider_config={"provider": "gemini", "model": "gemini-3.6-flash"},
        video_path="fixture.mp4",
        execute=True,
    )

    assert result["ok"] is True
    assert result["task_contract"]["execution"] == "online_provider_capability"
    assert captured["video_path"] == "fixture.mp4"
