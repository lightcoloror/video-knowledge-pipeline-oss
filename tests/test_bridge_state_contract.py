from __future__ import annotations

import json
import socket
from pathlib import Path

import video_knowledge_pipeline.openclaw_bridge_status as status_module
from video_knowledge_pipeline.bridge_state_contract import evaluate_bridge_state


def _evaluate(
    *,
    socket_state: str = "refused",
    listening: bool = False,
    health_checked: bool = False,
    health_ok: bool = False,
    task_state: str = "",
    pipeline_ready: bool = True,
):
    return evaluate_bridge_state(
        configured=True,
        socket_probe={"state": socket_state, "listening": listening},
        health_probe={"checked": health_checked, "ok": health_ok},
        scheduled_task={"scheduled_task_state": task_state},
        pipeline_capability={
            "ready": pipeline_ready,
            "offline_quality_route_discoverable": pipeline_ready,
        },
    )


def test_stopped_bridge_keeps_local_pipeline_available() -> None:
    result = _evaluate()

    assert result["bridge_state"] == "stopped_by_design"
    assert result["configured"] is True
    assert result["listening"] is False
    assert result["healthy"] is False
    assert result["bridge_call_execution_allowed"] is False
    assert result["pipeline_offline_ready"] is True
    assert result["availability"]["local_pipeline"]["bridge_required"] is False


def test_healthy_bridge_allows_http_calls() -> None:
    result = _evaluate(socket_state="listening", listening=True, health_checked=True, health_ok=True)

    assert result["bridge_state"] == "healthy"
    assert result["listening"] is True
    assert result["healthy"] is True
    assert result["bridge_call_execution_allowed"] is True


def test_stale_runtime_record_does_not_allow_http_calls() -> None:
    result = _evaluate(task_state="Running")

    assert result["bridge_state"] == "stale_runtime_record"
    assert result["stale_runtime_record"] is True
    assert result["bridge_call_execution_allowed"] is False


def test_timeout_is_distinct_from_stopped_by_design() -> None:
    result = _evaluate(socket_state="timeout")

    assert result["bridge_state"] == "probe_timeout"
    assert result["runtime_disposition"] == "stopped_by_design"
    assert result["pipeline_offline_ready"] is True


def test_missing_pipeline_capability_is_reported_independently() -> None:
    result = _evaluate(pipeline_ready=False)

    assert result["bridge_state"] == "stopped_by_design"
    assert result["pipeline_offline_ready"] is False


def test_review_page_presence_never_marks_content_reviewed() -> None:
    result = _evaluate()

    assert result["review_evidence_policy"]["review_page_exists_is_not_content_reviewed"] is True
    assert result["review_evidence_policy"]["content_reviewed"] is False


def test_socket_probe_preserves_timeout(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise socket.timeout("probe timed out")

    monkeypatch.setattr(status_module.socket, "create_connection", raise_timeout)
    result = status_module._socket_probe("127.0.0.1", 8931, 0.1)

    assert result["listening"] is False
    assert result["state"] == "timeout"

def test_bridge_state_fixture_matrix() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "bridge_state_contract" / "scenarios.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    for scenario in payload["scenarios"]:
        result = evaluate_bridge_state(
            configured=scenario["configured"],
            socket_probe=scenario["socket_probe"],
            health_probe=scenario["health_probe"],
            scheduled_task=scenario["scheduled_task"],
            pipeline_capability=scenario["pipeline_capability"],
        )
        for key, value in scenario["expected"].items():
            assert result[key] == value, scenario["name"]