from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline.asr_chunk_batch_submit import (
    SCHEMA,
    submit_asr_chunk_batch_workflow,
)
from video_knowledge_pipeline.asr_chunk_batch_workflow import (
    SCHEMA as WORKFLOW_SCHEMA,
    SUBMISSION_TOOL,
)
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.trusted_broker_http_client import (
    decode_broker_tool_response,
)


def _workflow(tmp_path: Path) -> Path:
    path = tmp_path / "asr-chunk-batch-workflow.json"
    write_json(
        path,
        {
            "schema": WORKFLOW_SCHEMA,
            "ok": True,
            "status": "ready",
            "workflow_sha256": "fixed-workflow-identity",
            "chunk_count": 2,
            "chunk_manifest": str(tmp_path / "chunk-manifest.json"),
            "bundle_dir": str(tmp_path / "bundle"),
            "activity_audit": {},
            "nodes": [
                {"consent_path": str(tmp_path / "consent-1.json")},
                {"consent_path": str(tmp_path / "consent-2.json")},
            ],
            "submission": {
                "tool": SUBMISSION_TOOL,
                "arguments": {
                    "bundle_dir": str(tmp_path / "bundle"),
                    "nodes": [
                        {
                            "id": "asr-chunk-0001",
                            "consent_path": str(tmp_path / "consent-1.json"),
                            "depends_on": [],
                        },
                        {
                            "id": "asr-chunk-0002",
                            "consent_path": str(tmp_path / "consent-2.json"),
                            "depends_on": [],
                        },
                    ],
                    "write": True,
                    "max_parallel_global": 2,
                    "max_parallel_per_destination": 1,
                },
            },
        },
    )
    return path


def _unchanged(saved: dict[str, object], _: Path) -> dict[str, object]:
    return saved


def test_asr_chunk_submit_defaults_to_revalidated_preview(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)

    result = submit_asr_chunk_batch_workflow(
        workflow,
        revalidator=_unchanged,
    )

    assert result["schema"] == SCHEMA
    assert result["status"] == "ready"
    assert result["submission_performed"] is False
    assert result["broker_control_requests_made"] == 0
    assert result["direct_provider_requests_made"] is False
    assert result["provider_execution_delegated_to_broker"] is False


def test_asr_chunk_submit_calls_existing_loopback_broker_tool_once(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(url: str, arguments: dict[str, object]) -> dict[str, object]:
        calls.append((url, arguments))
        return {
            "status": "accepted",
            "job_id": "model_batch_fixture",
        }

    result = submit_asr_chunk_batch_workflow(
        workflow,
        execute=True,
        transport=transport,
        revalidator=_unchanged,
    )

    assert result["ok"] is True
    assert result["status"] == "accepted"
    assert result["submission_performed"] is True
    assert result["broker_control_requests_made"] == 1
    assert result["direct_provider_requests_made"] is False
    assert result["provider_execution_delegated_to_broker"] is True
    saved = json.loads(workflow.read_text(encoding="utf-8"))
    assert calls == [
        (
            "http://127.0.0.1:8766/mcp",
            saved["submission"]["arguments"],
        )
    ]


def test_asr_chunk_submit_rejects_non_loopback_even_in_preview(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)

    with pytest.raises(ValueError, match="loopback"):
        submit_asr_chunk_batch_workflow(
            workflow,
            broker_url="https://example.com/mcp",
            revalidator=_unchanged,
        )

    with pytest.raises(ValueError, match="credentials"):
        submit_asr_chunk_batch_workflow(
            workflow,
            broker_url="http://user:secret@127.0.0.1:8766/mcp",
            revalidator=_unchanged,
        )


def test_asr_chunk_submit_rejects_stale_workflow_identity(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)

    def changed(saved: dict[str, object], _: Path) -> dict[str, object]:
        return {**saved, "workflow_sha256": "changed"}

    with pytest.raises(ValueError, match="stale"):
        submit_asr_chunk_batch_workflow(workflow, revalidator=changed)


def test_decode_tool_response_reuses_structured_or_json_text() -> None:
    assert decode_broker_tool_response(
        SimpleNamespace(structuredContent={"ok": True}, content=[], isError=False)
    ) == {"ok": True}
    assert decode_broker_tool_response(
        SimpleNamespace(
            structuredContent=None,
            structured_content=None,
            content=[SimpleNamespace(text='{"ok": true, "status": "accepted"}')],
            isError=False,
        )
    ) == {"ok": True, "status": "accepted"}


def test_asr_chunk_submit_cli_defaults_to_preview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    captured: dict[str, object] = {}

    def fake_submit(
        workflow_path: str,
        *,
        broker_url: str,
        execute: bool,
    ) -> dict[str, object]:
        captured.update(
            workflow_path=workflow_path,
            broker_url=broker_url,
            execute=execute,
        )
        return {"ok": True, "status": "ready"}

    monkeypatch.setattr(
        "video_knowledge_pipeline.cli.submit_asr_chunk_batch_workflow",
        fake_submit,
    )
    assert cli_main(["asr-chunk-batch-submit", str(workflow)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert captured["broker_url"] == "http://127.0.0.1:8766/mcp"
    assert captured["execute"] is False
