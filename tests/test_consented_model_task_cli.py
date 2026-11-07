from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_knowledge_pipeline import cli
from video_knowledge_pipeline.consented_model_task_cli import (
    EXIT_BLOCKED,
    EXIT_EXECUTION_FAILED,
    EXIT_INVALID,
    EXIT_SUCCESS,
    consented_model_task_exit_code,
    run_consented_model_task_cli,
)


class AllowingPolicy:
    def require_consent_scope(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "active"}


class BlockingPolicy:
    def require_consent_scope(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("provider destination is not allowlisted")


def _consent(path: Path) -> Path:
    path.write_text(json.dumps({"task": "smart_summary_rewrite"}), encoding="utf-8")
    return path


def test_cli_requires_exact_revision_and_explicit_write(
    tmp_path: Path,
) -> None:
    consent = _consent(tmp_path / "consent.json")
    result, code = run_consented_model_task_cli(
        consent,
        route_revision="",
        write=True,
        policy=AllowingPolicy(),  # type: ignore[arg-type]
    )
    assert code == EXIT_INVALID
    assert result["status"] == "invalid_input"
    result, code = run_consented_model_task_cli(
        consent,
        route_revision="route-rev",
        write=None,
        policy=AllowingPolicy(),  # type: ignore[arg-type]
    )
    assert code == EXIT_INVALID
    assert result["remote_requests_made"] is False


def test_cli_policy_block_prevents_executor_call(tmp_path: Path) -> None:
    consent = _consent(tmp_path / "consent.json")
    called = False

    def executor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"ok": True, "status": "completed"}

    result, code = run_consented_model_task_cli(
        consent,
        route_revision="route-rev",
        write=True,
        executor=executor,
        policy=BlockingPolicy(),  # type: ignore[arg-type]
    )
    assert code == EXIT_BLOCKED
    assert result["status"] == "policy_blocked"
    assert result["remote_requests_made"] is False
    assert called is False


def test_cli_passes_only_locked_inputs_to_existing_executor(tmp_path: Path) -> None:
    consent = _consent(tmp_path / "consent.json")
    captured: dict[str, Any] = {}

    def executor(consent_path: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update({"consent_path": consent_path, **kwargs})
        return {"ok": True, "status": "completed", "execution_id": "run-1"}

    result, code = run_consented_model_task_cli(
        consent,
        route_revision="route-rev",
        write=True,
        executor=executor,
        policy=AllowingPolicy(),  # type: ignore[arg-type]
    )
    assert code == EXIT_SUCCESS
    assert result["execution_id"] == "run-1"
    assert captured == {
        "consent_path": consent.resolve(),
        "expected_route_revision": "route-rev",
        "write": True,
    }


def test_cli_main_emits_full_json_and_blocked_exit_code(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    consent = _consent(tmp_path / "consent.json")
    expected = {
        "schema": "video_knowledge_pipeline.trusted_model_connector.v1",
        "ok": False,
        "status": "consent_required",
        "consent": {"blockers": [{"key": "consent_call_limit_exceeded"}]},
    }

    def fake_run(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], int]:
        return expected, EXIT_BLOCKED

    monkeypatch.setattr(cli, "run_consented_model_task_cli", fake_run)
    code = cli.main(
        [
            "execute-consented-model-task",
            str(consent),
            "--route-revision",
            "route-rev",
            "--write",
        ]
    )
    assert code == EXIT_BLOCKED
    assert json.loads(capsys.readouterr().out) == expected


def test_exit_code_contract() -> None:
    assert consented_model_task_exit_code({"ok": True, "status": "completed"}) == 0
    assert consented_model_task_exit_code({"ok": False, "status": "failed"}) == EXIT_EXECUTION_FAILED
    assert consented_model_task_exit_code({"ok": False, "status": "route_required"}) == EXIT_BLOCKED
    assert consented_model_task_exit_code({"ok": False, "status": "invalid_input"}) == EXIT_INVALID
