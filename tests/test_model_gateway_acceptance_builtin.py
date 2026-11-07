from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_gateway_acceptance import (
    capture_model_gateway_lane_result,
)


def _runtime() -> dict[str, object]:
    return {
        "schema": "video_knowledge_pipeline.model_runtime_result.v1",
        "ok": True,
        "status": "completed",
        "task": "temporal_sequence",
        "execution_location": "local",
        "route_id": "pool-local-production-qwen3-vl",
        "route_revision": "a" * 64,
        "deployment": {"id": "local-lmstudio-qwen3-vl-8b"},
        "provider": "local_qwen_vl",
        "latency_ms": 900,
        "usage": {},
        "estimated_cost": 0,
        "content": {"groups": []},
        "evidence": [],
        "consent_id": "",
    }


def _source(tmp_path: Path, backend: str) -> Path:
    source = tmp_path / f"{backend}.json"
    source.write_text(
        json.dumps(
            {
                "route": {
                    "route_id": "pool-local-production-qwen3-vl",
                    "route_revision": "a" * 64,
                    "execution_location": "local",
                    "deployments": [
                        {
                            "id": "local-lmstudio-qwen3-vl-8b",
                            "adapter_backend": backend,
                            "base_url": "http://127.0.0.1:1234/v1",
                        }
                    ],
                },
                "model_result": _runtime(),
            }
        ),
        encoding="utf-8",
    )
    return source


def test_lane_c_accepts_local_production_builtin_route(tmp_path: Path) -> None:
    captured = capture_model_gateway_lane_result(
        "C", _source(tmp_path, "builtin"), output_dir=tmp_path / "report"
    )

    assert captured["status"] == "captured"
    saved = json.loads(
        (tmp_path / "report" / "lane-c-proxy-local.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["acceptance_capture"]["expected_runtime"] == "openai_compatible_local"


def test_lane_c_still_rejects_legacy_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="builtin or proxy"):
        capture_model_gateway_lane_result(
            "C", _source(tmp_path, "legacy"), output_dir=tmp_path / "report"
        )
