from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
    model_connector_consent_lock_path,
    record_model_connector_attempt,
    reserve_model_connector_attempt,
)
from video_knowledge_pipeline.storage import bundle_write_lock, read_json, write_json


PROVIDER = {
    "provider": "custom_openai_compatible",
    "base_url": "https://example.invalid/v1",
    "model": "test-model",
}


def _create_consent(root: Path, *, max_calls: int) -> dict[str, Any]:
    source = root / "transcript.md"
    source.write_text("Approved local fixture.", encoding="utf-8")
    return create_model_connector_consent(
        root,
        task="smart_summary_rewrite",
        artifact_paths=[source],
        provider_config=PROVIDER,
        max_calls=max_calls,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )


def _reserve_in_process(
    consent_path: str,
    start_event: Any,
    result_queue: Any,
) -> None:
    try:
        if not start_event.wait(timeout=15):
            result_queue.put({"error": "start_timeout"})
            return
        result = reserve_model_connector_attempt(
            consent_path,
            provider_config=PROVIDER,
            lock_timeout_seconds=10,
        )
        result_queue.put(
            {
                "reserved": bool(result.get("reserved")),
                "status": str(result.get("status") or ""),
                "blockers": result.get("blockers") or [],
            }
        )
    except (
        BaseException
    ) as exc:  # Child failures must be observable by the parent test.
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})


def test_cross_process_reservations_never_exceed_consent_limit(request: Any) -> None:
    process_root = Path(tempfile.mkdtemp(prefix="vkp-consent-process-"))
    request.addfinalizer(lambda: shutil.rmtree(process_root, ignore_errors=True))
    consent = _create_consent(process_root, max_calls=3)
    context = multiprocessing.get_context("spawn")
    try:
        start_event = context.Event()
        result_queue = context.Queue()
    except PermissionError as exc:
        pytest.skip(f"multiprocessing IPC is unavailable in this test environment: {exc}")
    processes = [
        context.Process(
            target=_reserve_in_process,
            args=(str(consent["consent_path"]), start_event, result_queue),
        )
        for _ in range(6)
    ]

    for process in processes:
        process.start()
    start_event.set()
    results = [result_queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)

    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)
    assert not [row for row in results if row.get("error")]
    assert sum(bool(row.get("reserved")) for row in results) == 3
    blocked = [row for row in results if not row.get("reserved")]
    assert all(
        any(
            item.get("key") == "consent_call_limit_exceeded"
            for item in row.get("blockers") or []
        )
        for row in blocked
    )

    payload = read_json(Path(consent["consent_path"]))
    assert payload["usage"]["calls_attempted"] == 3
    assert payload["usage"]["calls_completed"] == 0


def test_live_old_lock_is_not_recovered(tmp_path: Path) -> None:
    consent = _create_consent(tmp_path, max_calls=1)
    lock_path = model_connector_consent_lock_path(consent["consent_path"])
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "operation": "active"}), encoding="utf-8"
    )
    old = time.time() - 3600
    os.utime(lock_path, (old, old))

    result = reserve_model_connector_attempt(
        consent["consent_path"],
        provider_config=PROVIDER,
        lock_timeout_seconds=0,
        stale_after_seconds=1,
    )

    assert result["reserved"] is False
    assert result["status"] == "consent_busy"
    assert any(row["key"] == "consent_reservation_busy" for row in result["blockers"])
    assert lock_path.is_file()


def test_dead_stale_lock_is_recovered(tmp_path: Path) -> None:
    consent = _create_consent(tmp_path, max_calls=1)
    lock_path = model_connector_consent_lock_path(consent["consent_path"])
    lock_path.write_text(
        json.dumps({"pid": 2147483647, "operation": "abandoned"}), encoding="utf-8"
    )
    old = time.time() - 3600
    os.utime(lock_path, (old, old))

    result = reserve_model_connector_attempt(
        consent["consent_path"],
        provider_config=PROVIDER,
        lock_timeout_seconds=0,
        stale_after_seconds=1,
    )

    assert result["reserved"] is True
    assert result["usage"]["calls_attempted"] == 1
    assert not lock_path.exists()


def test_concurrent_completion_updates_are_not_lost(tmp_path: Path) -> None:
    consent = _create_consent(tmp_path, max_calls=20)
    reservation = reserve_model_connector_attempt(
        consent["consent_path"],
        provider_config=PROVIDER,
        expected_calls=20,
    )
    assert reservation["reserved"] is True
    barrier = Barrier(20)

    def complete() -> None:
        barrier.wait(timeout=10)
        record_model_connector_attempt(
            consent["consent_path"],
            completed=True,
            lock_timeout_seconds=10,
        )

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(complete) for _ in range(20)]
        for future in futures:
            future.result(timeout=30)

    payload = read_json(Path(consent["consent_path"]))
    assert payload["usage"]["calls_attempted"] == 20
    assert payload["usage"]["calls_completed"] == 20


def test_custom_lock_is_reentrant_only_for_owning_thread(tmp_path: Path) -> None:
    lock_path = tmp_path / ".custom.lock"
    with bundle_write_lock(tmp_path, lock_name=lock_path.name):
        with bundle_write_lock(tmp_path, lock_name=lock_path.name):
            assert lock_path.is_file()
    assert not lock_path.exists()


def test_atomic_json_replace_retries_transient_permission_error(
    monkeypatch: Any, tmp_path: Path
) -> None:
    target = tmp_path / "payload.json"
    real_replace = os.replace
    attempts = {"count": 0}

    def flaky_replace(
        source: str | bytes | Path, destination: str | bytes | Path
    ) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("transient Windows sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr("video_knowledge_pipeline.storage.os.replace", flaky_replace)
    write_json(target, {"ok": True})

    assert attempts["count"] == 3
    assert read_json(target) == {"ok": True}


def test_lock_creation_retries_transient_windows_permission_error(
    monkeypatch: Any, tmp_path: Path
) -> None:
    lock_path = tmp_path / ".transient.lock"
    real_open = os.open
    attempts = {"count": 0}

    def flaky_open(
        path: str | bytes | Path, flags: int, *args: Any, **kwargs: Any
    ) -> int:
        if str(path) == str(lock_path):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise PermissionError("transient Windows lock creation collision")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("video_knowledge_pipeline.storage.os.open", flaky_open)
    with bundle_write_lock(tmp_path, lock_name=lock_path.name, timeout_seconds=1):
        assert lock_path.is_file()

    assert attempts["count"] == 2
    assert not lock_path.exists()
