from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_chunk_batch_status import (
    SCHEMA,
    STATUS_TOOL,
    query_asr_chunk_batch_status,
)
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.consented_model_batch import SCHEMA as BATCH_SCHEMA
from video_knowledge_pipeline.trusted_broker_http_client import (
    require_loopback_mcp_url,
)


def _batch(job_id: str) -> dict[str, object]:
    return {
        "schema": BATCH_SCHEMA,
        "job_id": job_id,
        "status": "completed",
        "terminal": True,
        "status_path": f"D:/fixture/{job_id}/batch-execution.json",
        "summary": {
            "total": 2,
            "completed": 2,
            "failed": 0,
        },
        "items": [],
    }


def test_asr_chunk_batch_status_queries_existing_tool_and_can_save_raw_status(
    tmp_path: Path,
) -> None:
    job_id = "model_batch_fixture"
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(url: str, arguments: dict[str, object]) -> dict[str, object]:
        calls.append((url, arguments))
        return _batch(job_id)

    output = tmp_path / "terminal-batch-status.json"
    result = query_asr_chunk_batch_status(
        job_id,
        output_path=output,
        transport=transport,
    )

    assert result["schema"] == SCHEMA
    assert result["status"] == "completed"
    assert result["terminal"] is True
    assert result["broker_tool"] == STATUS_TOOL
    assert result["broker_control_requests_made"] == 1
    assert result["direct_provider_requests_made"] is False
    assert result["read_only"] is True
    assert result["saved_status_path"] == str(output.resolve())
    assert json.loads(output.read_text(encoding="utf-8")) == _batch(job_id)
    assert calls == [
        (
            "http://127.0.0.1:8766/mcp",
            {"job_id": job_id},
        )
    ]


def test_asr_chunk_batch_status_rejects_invalid_or_mismatched_job_id() -> None:
    with pytest.raises(ValueError, match="invalid"):
        query_asr_chunk_batch_status("../model_batch_fixture", transport=lambda *_: {})

    with pytest.raises(ValueError, match="does not match"):
        query_asr_chunk_batch_status(
            "model_batch_fixture",
            transport=lambda *_: _batch("model_batch_other"),
        )


def test_shared_broker_url_rejects_remote_credentials_query_and_fragment() -> None:
    for value in (
        "https://example.com/mcp",
        "http://user:secret@127.0.0.1:8766/mcp",
        "http://127.0.0.1:8766/mcp?token=secret",
        "http://127.0.0.1:8766/mcp#fragment",
    ):
        with pytest.raises(ValueError):
            require_loopback_mcp_url(value)


def test_asr_chunk_batch_status_cli_uses_read_only_front_door(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_query(
        job_id: str,
        *,
        broker_url: str,
        output_path: str | None,
    ) -> dict[str, object]:
        captured.update(
            job_id=job_id,
            broker_url=broker_url,
            output_path=output_path,
        )
        return {
            "schema": SCHEMA,
            "ok": True,
            "status": "running",
            "terminal": False,
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.cli.query_asr_chunk_batch_status",
        fake_query,
    )
    output = tmp_path / "status.json"
    assert (
        cli_main(
            [
                "asr-chunk-batch-status",
                "model_batch_fixture",
                "--output-path",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    assert captured == {
        "job_id": "model_batch_fixture",
        "broker_url": "http://127.0.0.1:8766/mcp",
        "output_path": str(output),
    }
