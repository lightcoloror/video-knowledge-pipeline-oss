from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from video_knowledge_pipeline.consented_model_batch import (
    ConsentedModelBatchManager,
    classify_execution_result,
)
from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
)
from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)
from video_knowledge_pipeline.trusted_model_connector_remote_mcp import build_server


def _create_consent(root: Path, name: str, destination: str) -> Path:
    artifact = root / f"{name}.md"
    artifact.write_text(f"fixture {name}", encoding="utf-8")
    consent_path = root / "consents" / f"{name}.json"
    result = create_model_connector_consent(
        root,
        task="smart_summary_rewrite",
        artifact_paths=[artifact],
        provider_config={
            "provider": "custom_openai_compatible",
            "base_url": f"https://{destination}/v1",
            "model": "fixture-model",
        },
        output_path=consent_path,
        max_calls=1,
        max_estimated_cost_usd=0.01,
        max_cost_per_call_usd=0.01,
        max_retries_per_call=0,
        confirm_data_export=True,
    )
    return Path(result["consent_path"])


class RecordingExecutor:
    def __init__(self, *, delay: float = 0.04, rate_limited: set[str] | None = None):
        self.delay = delay
        self.rate_limited = rate_limited or set()
        self.lock = threading.Lock()
        self.active_global = 0
        self.max_active_global = 0
        self.active_destinations: dict[str, int] = {}
        self.max_active_destinations: dict[str, int] = {}
        self.calls: list[str] = []

    def __call__(
        self,
        consent_path: str,
        *,
        expected_route_revision: str,
        write: bool,
    ) -> dict[str, object]:
        payload = json.loads(Path(consent_path).read_text(encoding="utf-8"))
        destination = payload["authorized_destinations"][0]
        name = Path(consent_path).stem
        with self.lock:
            self.calls.append(name)
            self.active_global += 1
            self.max_active_global = max(self.max_active_global, self.active_global)
            active = self.active_destinations.get(destination, 0) + 1
            self.active_destinations[destination] = active
            self.max_active_destinations[destination] = max(
                self.max_active_destinations.get(destination, 0), active
            )
        try:
            time.sleep(self.delay)
            if name in self.rate_limited:
                return {
                    "ok": False,
                    "status": "provider_unavailable",
                    "error": "HTTP 429 rate limit",
                }
            return {
                "ok": True,
                "status": "completed",
                "artifacts": {
                    "execution_report": str(Path(consent_path).with_suffix(".report.json"))
                },
            }
        finally:
            with self.lock:
                self.active_global -= 1
                self.active_destinations[destination] -= 1


def _manager(
    root: Path,
    executor: RecordingExecutor,
    *,
    maximum_destination: int = 2,
) -> ConsentedModelBatchManager:
    policy = TrustedModelConnectorPolicy(
        (root.resolve(),), frozenset({"a.example", "b.example"})
    )
    return ConsentedModelBatchManager(
        project_root=root,
        policy=policy,
        executor=executor,
        global_parallel_limit=4,
        maximum_parallel_per_destination=maximum_destination,
    )


def test_batch_runs_different_destinations_in_parallel_and_persists_progress(
    tmp_path: Path,
) -> None:
    consents = [
        _create_consent(tmp_path, "a-1", "a.example"),
        _create_consent(tmp_path, "a-2", "a.example"),
        _create_consent(tmp_path, "b-1", "b.example"),
        _create_consent(tmp_path, "b-2", "b.example"),
    ]
    executor = RecordingExecutor()
    manager = _manager(
        tmp_path,
        executor,
        maximum_destination=1,
    )

    submitted = manager.submit([str(path) for path in consents], write=True)
    assert submitted["status"] == "accepted"
    assert Path(submitted["status_path"]).is_file()
    result = manager.wait(submitted["job_id"])

    assert result["status"] == "completed"
    assert result["summary"]["completed"] == 4
    assert executor.max_active_global >= 2
    assert max(executor.max_active_destinations.values()) == 1
    assert all("content" not in item for item in result["items"])


def test_batch_submit_is_idempotent_while_consent_usage_can_change(
    tmp_path: Path,
) -> None:
    consent = _create_consent(tmp_path, "slow", "a.example")
    executor = RecordingExecutor(delay=0.15)
    manager = _manager(tmp_path, executor)

    first = manager.submit([str(consent)])
    second = manager.submit([str(consent)])
    result = manager.wait(first["job_id"])

    assert second["status"] == "existing_result"
    assert second["job_id"] == first["job_id"]
    assert result["status"] == "completed"
    assert executor.calls == ["slow"]


def test_batch_records_rate_limit_without_retrying_or_fallback(
    tmp_path: Path,
) -> None:
    consents = [
        _create_consent(tmp_path, "limited", "a.example"),
        _create_consent(tmp_path, "healthy", "a.example"),
    ]
    executor = RecordingExecutor(rate_limited={"limited"})
    manager = _manager(tmp_path, executor)

    submitted = manager.submit([str(path) for path in consents])
    result = manager.wait(submitted["job_id"])

    assert result["status"] == "degraded"
    assert result["summary"]["completed"] == 1
    assert result["summary"]["failed"] == 1
    assert sorted(executor.calls) == ["healthy", "limited"]
    assert result["settings"]["retries_added_by_batch"] == 0
    assert result["settings"]["automatic_fallback_added_by_batch"] is False
    limited = next(item for item in result["items"] if item["consent_id"])
    assert any(item["outcome"] == "rate_limited" for item in result["items"])
    assert "content" not in limited


@pytest.mark.parametrize(
    ("result", "classification"),
    [
        ({"ok": False, "error": "HTTP 429"}, "rate_limited"),
        (
            {"ok": False, "model_result": {"error": "Server disconnected"}},
            "transient_provider_failure",
        ),
        ({"ok": False, "status": "contract_failed"}, "permanent_failure"),
    ],
)
def test_result_classification(result: dict[str, object], classification: str) -> None:
    assert classify_execution_result(result) == classification


def test_remote_mcp_exposes_submit_and_read_only_status_tools(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    manager = _manager(tmp_path, executor)
    policy = manager.policy
    server = build_server(
        policy=policy,
        host="127.0.0.1",
        port=8766,
        batch_manager=manager,
    )
    tools = asyncio.run(server.list_tools())
    rows = {tool.name: tool for tool in tools}

    submit = rows["submit_consented_model_batch_tool"]
    status = rows["consented_model_batch_status_tool"]
    assert submit.annotations.readOnlyHint is False
    assert submit.annotations.openWorldHint is True
    assert status.annotations.readOnlyHint is True
    assert status.annotations.openWorldHint is False
    assert "provider_config" not in submit.inputSchema["properties"]
    assert "consent_paths" in submit.inputSchema.get("required", [])
    assert "depends_on" in submit.inputSchema["properties"]
