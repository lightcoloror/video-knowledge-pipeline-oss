from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
)
from video_knowledge_pipeline.trusted_model_connector import (
    execute_consented_model_task,
    execute_local_model_task,
    trusted_model_connector_status,
)


PROVIDER = {
    "provider": "custom_openai_compatible",
    "base_url": "https://example.invalid/v1",
    "model": "test-model",
}


@pytest.mark.parametrize(
    ("task", "filenames", "expected_key"),
    [
        ("multimodal_frame_analysis", ["frame.png"], "image_paths"),
        ("temporal_visual_analysis", ["frame-1.jpg", "frame-2.jpg"], "image_paths"),
        ("cloud_asr", ["clip.wav"], "audio_path"),
    ],
)
def test_consented_online_modalities_reach_existing_gateway(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
    filenames: list[str],
    expected_key: str,
) -> None:
    artifacts = []
    for filename in filenames:
        path = tmp_path / filename
        path.write_bytes(b"fixture")
        artifacts.append(path)
    consent = create_model_connector_consent(
        tmp_path,
        task=task,
        artifact_paths=artifacts,
        provider_config=PROVIDER,
        instructions="Process only the authorised fixture.",
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    captured: dict[str, object] = {}

    def fake_call(task_name: str, **kwargs: object) -> dict[str, object]:
        captured.update({"task": task_name, **kwargs})
        return {
            "ok": True,
            "status": "completed",
            "response": {"content": "fixture-result"},
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER, write=False
    )
    assert result["ok"] is True
    assert captured["task"] == task
    assert expected_key in captured
    if expected_key == "image_paths":
        assert captured[expected_key] == [str(path.resolve()) for path in artifacts]
    else:
        assert captured[expected_key] == str(artifacts[0].resolve())


def test_online_ocr_executes_and_counts_each_artifact_independently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    images = [tmp_path / "slide-a.png", tmp_path / "slide-b.png"]
    for path in images:
        path.write_bytes(path.name.encode("utf-8"))
    consent = create_model_connector_consent(
        tmp_path,
        task="online_ocr",
        artifact_paths=images,
        provider_config=PROVIDER,
        max_calls=2,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    calls: list[list[str]] = []

    def fake_call(task_name: str, **kwargs: object) -> dict[str, object]:
        image_paths = list(kwargs["image_paths"])
        calls.append(image_paths)
        content = {
            "pages": [{"index": 0, "markdown": f"OCR {Path(image_paths[0]).name}"}]
        }
        return {
            "ok": True,
            "status": "completed",
            "content": json.dumps(content) if len(calls) == 2 else content,
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_consented_model_task(
        consent["consent_path"],
        provider_config=PROVIDER,
        write=False,
    )

    assert calls == [[str(images[0].resolve())], [str(images[1].resolve())]]
    assert result["ok"] is True
    assert result["model_result"]["call_count"] == 2
    assert result["model_result"]["completed_call_count"] == 2
    assert result["usage"]["calls_attempted"] == 2
    assert result["usage"]["calls_completed"] == 2
    assert result["cost_control"]["reported_cost_known"] is False
    assert result["usage"]["cost_committed_usd"] == 1.0
    assert result["usage"]["cost_unreported_calls"] == 2
    pages = result["model_result"]["content"]["pages"]
    assert [row["image_path"] for row in pages] == [
        str(path.resolve()) for path in images
    ]
    assert all(row["source_artifact_sha256"] for row in pages)


def test_temporal_consent_executes_each_frame_group_and_counts_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    groups = []
    artifacts = []
    for group_id in (6, 80):
        group = tmp_path / "temporal-frames" / f"{group_id:04d}"
        group.mkdir(parents=True)
        frames = []
        for index in range(1, 3):
            frame = group / f"frame_{index:02d}.jpg"
            frame.write_bytes(f"{group_id}-{index}".encode())
            frames.append(frame)
            artifacts.append(frame)
        groups.append(frames)
    consent = create_model_connector_consent(
        tmp_path,
        task="temporal_visual_analysis",
        artifact_paths=artifacts,
        provider_config=PROVIDER,
        max_calls=2,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    calls: list[list[str]] = []

    def fake_call(task_name: str, **kwargs: object) -> dict[str, object]:
        image_paths = list(kwargs["image_paths"])
        calls.append(image_paths)
        return {
            "schema": "video_knowledge_pipeline.model_runtime_result.v1",
            "ok": True,
            "status": "completed",
            "task": "temporal_sequence",
            "execution_location": "remote",
            "route_id": "direct-test",
            "route_revision": "a" * 64,
            "deployment": {"id": "fake", "provider": "fake"},
            "provider": "fake",
            "latency_ms": 10,
            "usage": {"total_tokens": 5},
            "estimated_cost": 0.01,
            "content": {"summary": Path(image_paths[0]).parent.name},
            "evidence": [],
            "consent_id": "fake",
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_consented_model_task(
        consent["consent_path"],
        provider_config=PROVIDER,
        write=False,
    )

    assert calls == [[str(path.resolve()) for path in group] for group in groups]
    assert result["ok"] is True
    assert (
        result["model_result"]["schema"]
        == "video_knowledge_pipeline.model_runtime_result.v1"
    )
    assert result["model_result"]["call_count"] == 2
    assert result["model_result"]["completed_call_count"] == 2
    assert result["model_result"]["usage"]["total_tokens"] == 10
    assert result["usage"]["calls_attempted"] == 2
    assert result["usage"]["calls_completed"] == 2
    assert result["cost_control"]["reported_cost_usd"] == 0.02
    assert result["usage"]["cost_committed_usd"] == 0.02
    assert result["cost_control"]["remaining_estimated_cost_usd"] == 0.98
    assert result["cost_control"]["cost_limit_exceeded"] is False
    assert [row["group_id"] for row in result["model_result"]["content"]["groups"]] == [
        "0006",
        "0080",
    ]


def test_temporal_consent_unwraps_gateway_runtime_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    group = tmp_path / "temporal-frames" / "0006"
    group.mkdir(parents=True)
    frames = []
    for index in range(1, 3):
        frame = group / f"frame_{index:02d}.jpg"
        frame.write_bytes(str(index).encode())
        frames.append(frame)
    consent = create_model_connector_consent(
        tmp_path,
        task="temporal_visual_analysis",
        artifact_paths=frames,
        provider_config=PROVIDER,
        max_calls=1,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    runtime = {
        "schema": "video_knowledge_pipeline.model_runtime_result.v1",
        "ok": True,
        "status": "completed",
        "task": "temporal_sequence",
        "execution_location": "remote",
        "route_id": "nested-route",
        "route_revision": "d" * 64,
        "deployment": {
            "id": "nested-deployment",
            "provider": "nested-provider",
            "model": "nested-model",
        },
        "provider": "nested-provider",
        "latency_ms": 321,
        "usage": {"total_tokens": 17},
        "estimated_cost": 0.02,
        "content": {"state_changes": ["hands moved"]},
    }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        lambda *args, **kwargs: {
            "schema": "video_knowledge_pipeline.model_task_gateway.v1",
            "ok": True,
            "status": "completed",
            "content": runtime["content"],
            "runtime_result": runtime,
        },
    )

    result = execute_consented_model_task(
        consent["consent_path"],
        provider_config=PROVIDER,
        write=False,
    )

    model_result = result["model_result"]
    assert model_result["latency_ms"] == 321
    assert model_result["usage"] == {"total_tokens": 17.0}
    assert model_result["estimated_cost"] == 0.02
    assert model_result["provider"] == "nested-provider"
    assert model_result["deployment"]["id"] == "nested-deployment"
    assert model_result["calls"][0]["latency_ms"] == 321
    assert model_result["content"]["groups"][0]["content"] == runtime["content"]


def test_temporal_consent_status_checks_group_call_allowance(tmp_path: Path) -> None:
    artifacts = []
    for group_id in (6, 80):
        group = tmp_path / f"{group_id:04d}"
        group.mkdir()
        frame = group / "frame_01.jpg"
        frame.write_bytes(str(group_id).encode())
        artifacts.append(frame)
    consent = create_model_connector_consent(
        tmp_path,
        task="temporal_visual_analysis",
        artifact_paths=artifacts,
        provider_config=PROVIDER,
        max_calls=1,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )

    status = trusted_model_connector_status(
        consent["consent_path"],
        provider_config=PROVIDER,
        expected_task="temporal_visual_analysis",
        expected_calls=2,
    )

    assert status["valid"] is False
    assert any(
        row["key"] == "consent_call_limit_exceeded" for row in status["blockers"]
    )


def test_local_temporal_execution_uses_the_same_grouped_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts = []
    for group_id in (6, 80):
        group = tmp_path / "temporal-frames" / f"{group_id:04d}"
        group.mkdir(parents=True)
        frame = group / "frame_01.jpg"
        frame.write_bytes(str(group_id).encode())
        artifacts.append(frame)
    route = {
        "route_id": "local-vision",
        "route_revision": "b" * 64,
        "virtual_model": "vkp-local-vision",
        "execution_location": "local",
        "deployments": [
            {
                "id": "local-vlm",
                "provider": "local_vlm",
                "adapter_backend": "proxy",
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "local-vlm",
            }
        ],
    }
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.resolve_model_api_route",
        lambda *args, **kwargs: route,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.resolve_model_api_provider_config",
        lambda *args, **kwargs: {
            "provider": "local_vlm",
            "adapter_backend": "proxy",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-vlm",
            "execution_location": "local",
        },
    )
    calls: list[list[str]] = []

    def fake_call(task_name: str, **kwargs: object) -> dict[str, object]:
        calls.append(list(kwargs["image_paths"]))
        return {
            "ok": True,
            "status": "completed",
            "content": {"summary": "local"},
            "latency_ms": 3,
            "usage": {},
            "estimated_cost": 0,
            "deployment": route["deployments"][0],
            "provider": "local_vlm",
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_local_model_task(
        "temporal_visual_analysis",
        artifacts,
        route_id="local-vision",
        write=False,
    )

    assert len(calls) == 2
    assert result["ok"] is True
    assert result["model_result"]["execution_location"] == "local"
    assert result["model_result"]["call_count"] == 2
    assert result["model_result"]["remote_requests_made"] is False
