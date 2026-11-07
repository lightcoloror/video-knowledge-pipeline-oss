from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

from video_knowledge_pipeline.consented_model_batch import (
    ConsentedModelBatchManager,
    classify_execution_result,
    list_consented_model_batches,
)
from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)
from video_knowledge_pipeline.trusted_model_connector_remote_mcp import build_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MemoryPolicy:
    def __init__(self, consents: dict[str, dict[str, Any]]) -> None:
        self.allowed_roots = (PROJECT_ROOT,)
        self.consents = consents

    def require_path(self, value: str | Path, *, label: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"{label} is outside the test root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def require_consent_scope(
        self, value: str | Path, *, require_execution_contract: bool
    ) -> dict[str, Any]:
        assert require_execution_contract is True
        return copy.deepcopy(self.consents[str(Path(value).resolve())])


class MemoryBatchManager(ConsentedModelBatchManager):
    def __init__(self, **kwargs: Any) -> None:
        self.memory: dict[str, dict[str, Any]] = {}
        super().__init__(**kwargs)

    def _read_existing(self, job_id: str) -> dict[str, Any] | None:
        payload = self.memory.get(job_id)
        return copy.deepcopy(payload) if payload is not None else None

    def _read_payload(self, job_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.memory[job_id])

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.memory[str(payload["job_id"])] = copy.deepcopy(payload)


class MemoryExecutor:
    def __init__(
        self,
        policy: MemoryPolicy,
        *,
        delay: float = 0.03,
        rate_limited: set[str] | None = None,
    ) -> None:
        self.policy = policy
        self.delay = delay
        self.rate_limited = rate_limited or set()
        self.lock = threading.Lock()
        self.calls: list[str] = []
        self.active_global = 0
        self.max_active_global = 0
        self.active_by_destination: dict[str, int] = {}
        self.max_active_by_destination: dict[str, int] = {}

    def __call__(
        self,
        consent_path: str,
        *,
        expected_route_revision: str,
        write: bool,
    ) -> dict[str, Any]:
        payload = self.policy.consents[str(Path(consent_path).resolve())]
        destination = payload["authorized_destinations"][0]
        consent_id = payload["consent_id"]
        assert expected_route_revision == payload["route"]["route_revision"]
        with self.lock:
            self.calls.append(consent_id)
            self.active_global += 1
            self.max_active_global = max(self.max_active_global, self.active_global)
            active = self.active_by_destination.get(destination, 0) + 1
            self.active_by_destination[destination] = active
            self.max_active_by_destination[destination] = max(
                self.max_active_by_destination.get(destination, 0), active
            )
        try:
            time.sleep(self.delay)
            if consent_id in self.rate_limited:
                return {
                    "ok": False,
                    "status": "provider_unavailable",
                    "error": "HTTP 429 rate limit",
                }
            return {"ok": True, "status": "completed"}
        finally:
            with self.lock:
                self.active_global -= 1
                self.active_by_destination[destination] -= 1


def _consent(consent_id: str, destination: str) -> dict[str, Any]:
    return {
        "consent_id": consent_id,
        "task": "smart_summary_rewrite",
        "authorized_destinations": [f"https://{destination}"],
        "authorized_deployments": [
            {
                "provider": "custom_openai_compatible",
                "base_url": f"https://{destination}/v1",
                "model": "fixture-model",
            }
        ],
        "route": {"route_revision": f"revision-{consent_id}"},
        "upload_manifest": {"manifest_sha256": f"manifest-{consent_id}"},
    }


def _manager(
    *,
    destinations: list[str],
    rate_limited: set[str] | None = None,
    delay: float = 0.03,
) -> tuple[MemoryBatchManager, MemoryExecutor, list[str]]:
    fixture_paths = [
        PROJECT_ROOT / "tests" / "test_consented_model_batch_memory.py",
        PROJECT_ROOT / "tests" / "test_consented_model_batch.py",
        PROJECT_ROOT / "tests" / "test_trusted_model_connector_policy.py",
        PROJECT_ROOT / "tests" / "test_model_connector_consent_v2.py",
    ][: len(destinations)]
    consents = {
        str(path.resolve()): _consent(f"consent-{index}", destination)
        for index, (path, destination) in enumerate(zip(fixture_paths, destinations))
    }
    policy = MemoryPolicy(consents)
    executor = MemoryExecutor(
        policy, delay=delay, rate_limited=rate_limited
    )
    manager = MemoryBatchManager(
        project_root=PROJECT_ROOT,
        policy=policy,  # type: ignore[arg-type]
        executor=executor,
        global_parallel_limit=4,
        maximum_parallel_per_destination=1,
    )
    return manager, executor, [str(path) for path in fixture_paths]


def test_memory_batch_parallelises_destinations_but_serialises_each_destination() -> None:
    manager, executor, paths = _manager(
        destinations=["a.example", "a.example", "b.example", "b.example"]
    )

    submitted = manager.submit(paths)
    result = manager.wait(submitted["job_id"])

    assert result["status"] == "completed"
    assert result["summary"]["completed"] == 4
    assert executor.max_active_global >= 2
    assert max(executor.max_active_by_destination.values()) == 1
    assert all("content" not in item for item in result["items"])


def test_memory_batch_is_idempotent_and_never_replays_an_active_job() -> None:
    manager, executor, paths = _manager(destinations=["a.example"], delay=0.1)

    first = manager.submit(paths)
    second = manager.submit(paths)
    result = manager.wait(first["job_id"])

    assert second["status"] == "existing_result"
    assert second["job_id"] == first["job_id"]
    assert result["status"] == "completed"
    assert executor.calls == ["consent-0"]


def test_memory_batch_degrades_on_429_without_retry_or_fallback() -> None:
    manager, executor, paths = _manager(
        destinations=["a.example", "a.example"],
        rate_limited={"consent-0"},
    )

    submitted = manager.submit(paths)
    result = manager.wait(submitted["job_id"])

    assert result["status"] == "degraded"
    assert result["summary"]["completed"] == 1
    assert result["summary"]["failed"] == 1
    assert sorted(executor.calls) == ["consent-0", "consent-1"]
    assert result["settings"]["retries_added_by_batch"] == 0
    assert result["settings"]["automatic_fallback_added_by_batch"] is False
    assert any(item["outcome"] == "rate_limited" for item in result["items"])


def test_dependency_failure_blocks_descendants_without_provider_calls() -> None:
    manager, executor, paths = _manager(
        destinations=["a.example", "b.example", "a.example"],
        rate_limited={"consent-0"},
    )

    submitted = manager.submit(paths, depends_on=[[], [0], [1]])
    result = manager.wait(submitted["job_id"])

    assert result["status"] == "failed"
    assert result["summary"]["failed"] == 1
    assert result["summary"]["dependency_blocked"] == 2
    assert result["summary"]["rate_limited"] == 1
    assert executor.calls == ["consent-0"]
    assert [item["state"] for item in result["items"]] == [
        "failed",
        "dependency_blocked",
        "dependency_blocked",
    ]


def test_dependency_cycle_is_rejected_before_execution() -> None:
    manager, executor, paths = _manager(
        destinations=["a.example", "b.example"]
    )

    try:
        manager.submit(paths, depends_on=[[1], [0]])
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cyclic dependency graph should be rejected")
    assert executor.calls == []


def test_litellm_owns_provider_rate_limit_and_error_classification() -> None:
    manager, _, _ = _manager(destinations=["a.example"])
    snapshot = manager.public_snapshot()
    assert snapshot["batch_adaptive_limiter_enabled"] is False
    assert snapshot["provider_rate_limit_owner"] == "litellm_proxy"
    assert classify_execution_result({"ok": False, "error": "HTTP 429"}) == "rate_limited"
    assert (
        classify_execution_result(
            {"ok": False, "model_result": {"error": "Server disconnected"}}
        )
        == "transient_provider_failure"
    )


def test_remote_mcp_declares_batch_submit_and_read_only_status() -> None:
    manager, _, _ = _manager(destinations=["a.example"])
    policy = TrustedModelConnectorPolicy(
        (PROJECT_ROOT,), frozenset({"a.example"})
    )
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


def test_batch_list_is_read_only_and_redacts_consent_and_content(tmp_path: Path) -> None:
    status_path = (
        tmp_path
        / ".local"
        / "model-connector-batches"
        / "model_batch_fixture"
        / "batch-execution.json"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.consented_model_batch.v1",
                "job_id": "model_batch_fixture",
                "status": "degraded",
                "terminal": True,
                "submitted_at": "2026-07-21T10:00:00+00:00",
                "updated_at": "2026-07-21T10:01:00+00:00",
                "completed_at": "2026-07-21T10:01:00+00:00",
                "summary": {
                    "total": 2,
                    "completed": 1,
                    "failed": 1,
                    "rate_limited": 1,
                },
                "settings": {
                    "max_parallel_global": 4,
                    "max_parallel_per_destination": 2,
                    "provider_rate_limit_owner": "litellm_proxy",
                    "retries_added_by_batch": 0,
                },
                "destination_controllers": {
                    "https://api.example": {
                        "completed_calls": 2,
                        "outcomes": {"success": 1, "rate_limited": 1},
                    }
                },
                "items": [
                    {
                        "task": "smart_summary_rewrite",
                        "destination": "https://api.example",
                        "consent_path": "D:/secret/consent.json",
                        "content": "private model input",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = list_consented_model_batches(tmp_path)
    encoded = json.dumps(result, ensure_ascii=False)

    assert result["count"] == 1
    assert result["items"][0]["summary"]["rate_limited"] == 1
    assert result["items"][0]["tasks"] == ["smart_summary_rewrite"]
    assert result["provider_rate_limit_owner"] == "litellm_proxy"
    assert result["batch_adaptive_limiter_enabled"] is False
    assert result["consent_paths_exposed"] is False
    assert "consent_path" not in result["items"][0]
    assert "private model input" not in encoded
    assert "D:/secret/consent.json" not in encoded
