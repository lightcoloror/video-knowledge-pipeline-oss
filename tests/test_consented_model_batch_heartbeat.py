from __future__ import annotations

import threading
import time
from pathlib import Path

from video_knowledge_pipeline.consented_model_batch import (
    ConsentedModelBatchManager,
)
from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
)
from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)


def _create_consent(root: Path) -> Path:
    artifact = root / "heartbeat-fixture.md"
    artifact.write_text("heartbeat fixture", encoding="utf-8")
    result = create_model_connector_consent(
        root,
        task="smart_summary_rewrite",
        artifact_paths=[artifact],
        provider_config={
            "provider": "custom_openai_compatible",
            "base_url": "https://heartbeat.example/v1",
            "model": "fixture-model",
        },
        output_path=root / "consents" / "heartbeat.json",
        max_calls=1,
        max_estimated_cost_usd=0.01,
        max_cost_per_call_usd=0.01,
        max_retries_per_call=0,
        confirm_data_export=True,
    )
    return Path(result["consent_path"])


class BlockingExecutor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(
        self,
        consent_path: str,
        *,
        expected_route_revision: str,
        write: bool,
    ) -> dict[str, object]:
        del consent_path, expected_route_revision, write
        self.started.set()
        if not self.release.wait(timeout=2):
            return {"ok": False, "status": "fixture_timeout"}
        return {"ok": True, "status": "completed", "artifacts": {}}


def test_running_batch_persists_heartbeat_without_copying_model_content(
    tmp_path: Path,
) -> None:
    executor = BlockingExecutor()
    manager = ConsentedModelBatchManager(
        project_root=tmp_path,
        policy=TrustedModelConnectorPolicy(
            (tmp_path.resolve(),), frozenset({"heartbeat.example"})
        ),
        executor=executor,
        heartbeat_interval_seconds=0.02,
        heartbeat_stale_multiplier=3,
    )
    submitted = manager.submit([str(_create_consent(tmp_path))])
    assert executor.started.wait(timeout=1)

    snapshot: dict[str, object] = {}
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        snapshot = manager.status(submitted["job_id"])
        item = snapshot["items"][0]  # type: ignore[index]
        if int(item.get("heartbeat_count") or 0) >= 2:  # type: ignore[union-attr]
            break
        time.sleep(0.01)

    item = snapshot["items"][0]  # type: ignore[index]
    assert item["state"] == "running"  # type: ignore[index]
    assert item["heartbeat_state"] == "active"  # type: ignore[index]
    assert int(item["heartbeat_count"]) >= 2  # type: ignore[index]
    assert snapshot["summary"]["heartbeat_alive"] == 1  # type: ignore[index]
    assert snapshot["summary"]["heartbeat_stale"] == 0  # type: ignore[index]
    assert "content" not in item

    executor.release.set()
    result = manager.wait(submitted["job_id"], timeout=2)
    completed = result["items"][0]
    assert result["status"] == "completed"
    assert completed["heartbeat_state"] == "stopped"
    assert completed["heartbeat_stopped_at"]
    assert result["summary"]["heartbeat_alive"] == 0
    assert result["summary"]["heartbeat_stale"] == 0


def test_summary_distinguishes_alive_and_stale_running_calls() -> None:
    now_ms = int(time.time() * 1000)
    summary = ConsentedModelBatchManager._summary(
        [
            {
                "state": "running",
                "heartbeat_at_unix_ms": now_ms,
                "heartbeat_stale_after_seconds": 1,
            },
            {
                "state": "running",
                "heartbeat_at_unix_ms": now_ms - 2_000,
                "heartbeat_stale_after_seconds": 1,
            },
        ]
    )

    assert summary["running"] == 2
    assert summary["heartbeat_alive"] == 1
    assert summary["heartbeat_stale"] == 1
