from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.local_vlm_server_adapter as local_vlm
import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal
import video_knowledge_pipeline.vision_api as vision_api


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_openai_compatible_vision_surfaces_token_and_truncation(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"synthetic-image")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": '{"objects": ["screen"]}'},
                        "finish_reason": "length",
                    }
                ]
            }
        )

    monkeypatch.setattr(vision_api.urllib.request, "urlopen", fake_urlopen)

    result = vision_api.call_openai_compatible_vision(
        provider_config={
            "provider": "local_vlm",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "synthetic-model",
        },
        prompt="Return JSON",
        image_paths=[str(image)],
        max_tokens=64,
    )

    assert captured["body"]["max_tokens"] == 64
    assert result["request_max_tokens"] == 64
    assert result["request_max_tokens_omitted"] is False
    assert result["finish_reason"] == "length"
    assert result["truncated"] is True
    assert result["complete"] is False


def test_multimodal_local_vlm_is_one_frame_per_request_and_does_not_apply_truncated(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frames = []
    for index in range(1, 3):
        frame = assets / f"frame-{index}.jpg"
        frame.write_bytes(f"synthetic-{index}".encode("utf-8"))
        frames.append(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": index,
                    "start": index - 1,
                    "end": index,
                    "visual_route": "semantic_frame",
                    "frame_paths": [str(frame)],
                }
                for index, frame in enumerate(frames, start=1)
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_call(*, provider_config, prompt, image_paths, allowed_roots=None, max_tokens=None):
        calls.append({"image_paths": list(image_paths), "max_tokens": max_tokens})
        truncated = len(calls) == 1
        return {
            "ok": True,
            "status": "ok",
            "error": "",
            "content": json.dumps(
                {
                    "objects": ["screen"],
                    "actions": ["synthetic"],
                    "confidence": 0.8,
                    "evidence_frame_paths": list(image_paths),
                }
            ),
            "finish_reason": "length" if truncated else "stop",
            "truncated": truncated,
            "complete": not truncated,
            "request_max_tokens": max_tokens,
            "request_max_tokens_omitted": max_tokens is None,
        }

    monkeypatch.setattr(multimodal, "call_vision_model_with_broker_reservation", fake_call)
    monkeypatch.setattr(
        multimodal,
        "_refresh_post_vision_outputs",
        lambda *args, **kwargs: {"status": "synthetic_skipped"},
    )

    result = multimodal.run_multimodal_frame_analysis(
        bundle,
        execute=True,
        provider_config={
            "provider": "local_vlm",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "synthetic-model",
        },
        limit=2,
        confirm_vision_calls=2,
        confirm_vision_indexes="1,2",
        max_tokens=64,
    )

    assert [len(row["image_paths"]) for row in calls] == [1, 1]
    assert [row["max_tokens"] for row in calls] == [64, 64]
    assert result["summary"]["max_images_per_request"] == 1
    assert result["summary"]["complete_count"] == 1
    assert result["summary"]["truncated_count"] == 1
    assert result["summary"]["updated"] == 1
    assert result["items"][0]["request_mode"] == "single_frame"
    assert result["items"][0]["truncated"] is True
    assert result["items"][0]["complete"] is False
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert "visual_understanding" not in timeline[0]
    assert timeline[1]["visual_understanding"]["objects"] == ["screen"]


def test_lm_studio_1234_is_explicit_plan_only_loopback_candidate(tmp_path: Path) -> None:
    result = local_vlm.local_vlm_serving_smoke(
        provider="local_vlm",
        output_dir=str(tmp_path),
        base_url="http://127.0.0.1:1234/v1",
        model="synthetic-lm-studio-model",
        max_tokens=64,
        execute=False,
        write=False,
    )

    assert result["ok"] is True
    assert result["status"] == "plan_only"
    assert result["input_spec"]["base_url"] == "http://127.0.0.1:1234/v1"
    assert result["input_spec"]["model"] == "synthetic-lm-studio-model"
    assert result["input_spec"]["max_tokens"] == 64
    assert result["operator_boundary"]["provider_request_made"] is False


def test_local_vlm_rejects_non_loopback_override(tmp_path: Path) -> None:
    result = local_vlm.local_vlm_serving_smoke(
        provider="local_vlm",
        output_dir=str(tmp_path),
        base_url="https://example.test/v1",
        model="synthetic-model",
        execute=False,
        write=False,
    )

    assert result["ok"] is False
    assert result["status"] == "invalid_local_provider_url"
    assert result["operator_boundary"]["provider_request_made"] is False


def test_local_vlm_multi_image_truncation_is_not_complete(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.vision_provider_smoke as smoke_module

    captured = {}

    def fake_smoke(**kwargs):
        captured.update(kwargs)
        return {
            "ok": False,
            "status": "model_output_truncated",
            "checks": [
                {"name": "text_ping", "ok": True, "status": "ok", "image_count": 0},
                {"name": "single_image_json", "ok": True, "status": "ok", "image_count": 1},
                {
                    "name": "multi_image_json",
                    "ok": False,
                    "status": "truncated",
                    "image_count": 2,
                    "finish_reason": "length",
                    "truncated": True,
                    "complete": False,
                },
            ],
        }

    monkeypatch.setattr(
        smoke_module,
        "vision_provider_smoke",
        fake_smoke,
    )

    result = local_vlm.local_vlm_serving_smoke(
        provider="local_vlm",
        output_dir=str(tmp_path),
        base_url="http://127.0.0.1:1234/v1",
        model="synthetic-model",
        max_tokens=64,
        execute=True,
        write=False,
    )

    matrix = {row["key"]: row for row in result["capability_matrix"]}
    assert result["ok"] is False
    assert captured["provider_config"]["max_tokens"] == 64
    assert matrix["multi_image_json"]["status"] == "truncated"
    assert matrix["multi_image_json"]["complete"] is False
