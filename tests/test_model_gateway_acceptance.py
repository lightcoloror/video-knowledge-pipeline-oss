from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_gateway_acceptance import (
    build_temporal_gateway_acceptance_manifest,
    capture_model_gateway_lane_result,
    compare_model_gateway_results,
)


def _result(path: Path, *, location: str, latency: int, cost: float | None, consent: str = "") -> Path:
    payload = {
        "schema": "video_knowledge_pipeline.model_runtime_result.v1",
        "ok": True,
        "status": "completed",
        "task": "temporal_sequence",
        "execution_location": location,
        "route_id": f"{location}-vision",
        "route_revision": "a" * 64,
        "deployment": {"id": f"{location}-vlm"},
        "provider": {"provider": "fake"},
        "latency_ms": latency,
        "usage": {"total_tokens": 10},
        "estimated_cost": cost,
        "content": {"summary": f"{location} result"},
        "evidence": [{"sha256": "b" * 64}],
        "consent_id": consent,
        "quality_gate": {"status": "pass"},
        "failure_recovery": "not_needed",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_offline_abc_comparison_does_not_copy_model_content(tmp_path: Path) -> None:
    result = compare_model_gateway_results(
        _result(tmp_path / "a.json", location="remote", latency=300, cost=None),
        _result(tmp_path / "b.json", location="remote", latency=220, cost=0.01, consent="consent-b"),
        _result(tmp_path / "c.json", location="local", latency=900, cost=0),
        output_dir=tmp_path / "report",
        sample_id="fixed-temporal-6",
    )

    assert result["status"] == "ready_for_review"
    assert result["comparison"]["schema_compatible"] is True
    assert result["comparison"]["latency_ms"] == {"A": 300.0, "B": 220.0, "C": 900.0}
    assert result["comparison"]["call_count"] == {"A": 1, "B": 1, "C": 1}
    assert all("content" not in row for row in result["lanes"])
    assert result["operator_boundary"]["does_not_call_models"] is True
    assert (tmp_path / "report" / "model-gateway-abc-comparison.md").is_file()


def test_temporal_manifest_requires_exact_frame_count_and_only_hashes_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    complete = bundle / "temporal-frames" / "0006"
    incomplete = bundle / "temporal-frames" / "0080"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    for index in range(1, 9):
        (complete / f"frame_{index:02d}.jpg").write_bytes(f"frame-{index}".encode())
    for index in range(1, 8):
        (incomplete / f"frame_{index:02d}.jpg").write_bytes(f"frame-{index}".encode())

    result = build_temporal_gateway_acceptance_manifest(
        bundle,
        indexes=[80, 6],
        output_dir=tmp_path / "report",
    )

    assert result["status"] == "incomplete"
    assert result["ready_group_count"] == 1
    assert result["failed_group_count"] == 1
    assert result["groups"][0]["frame_count"] == 8
    assert all(set(frame) == {"path", "bytes", "sha256"} for frame in result["groups"][0]["frames"])
    assert result["operator_boundary"]["model_calls_made"] == 0
    assert result["operator_boundary"]["images_uploaded"] is False

def test_lane_capture_requires_matching_route_backend_and_location(tmp_path: Path) -> None:
    runtime_path = _result(
        tmp_path / "runtime.json",
        location="remote",
        latency=220,
        cost=0.01,
        consent="consent-b",
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    source = tmp_path / "connector-execution.json"
    source.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.trusted_model_connector.v1",
                "route": {
                    "route_id": "remote-vision",
                    "route_revision": "a" * 64,
                    "execution_location": "remote",
                    "deployments": [{"id": "remote-vlm", "adapter_backend": "proxy"}],
                },
                "model_result": runtime,
            }
        ),
        encoding="utf-8",
    )

    captured = capture_model_gateway_lane_result(
        "B",
        source,
        output_dir=tmp_path / "report",
    )

    target = tmp_path / "report" / "lane-b-proxy-remote.json"
    assert captured["status"] == "captured"
    assert captured["remote_requests_made"] is False
    assert target.is_file()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["acceptance_capture"]["lane"] == "B"
    assert saved["content"] == runtime["content"]
    with pytest.raises(ValueError, match="execution_location=local"):
        capture_model_gateway_lane_result("C", source, output_dir=tmp_path / "report")
