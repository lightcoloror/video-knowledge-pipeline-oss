from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .consented_model_batch import SCHEMA as BATCH_SCHEMA
from .storage import write_json
from .trusted_broker_http_client import (
    call_loopback_broker_tool,
    require_loopback_mcp_url,
)


SCHEMA = "video_knowledge_pipeline.asr_chunk_batch_status_query.v1"
STATUS_TOOL = "consented_model_batch_status_tool"
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def query_asr_chunk_batch_status(
    job_id: str,
    *,
    broker_url: str = "http://127.0.0.1:8766/mcp",
    output_path: str | Path | None = None,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """Read one durable Broker batch status without touching a provider."""

    clean_job_id = _clean_job_id(job_id)
    clean_broker_url = require_loopback_mcp_url(broker_url)
    arguments = {"job_id": clean_job_id}
    batch = (
        transport(clean_broker_url, arguments)
        if transport is not None
        else call_loopback_broker_tool(clean_broker_url, STATUS_TOOL, arguments)
    )
    if not isinstance(batch, dict):
        raise ValueError("Broker batch status response must be a JSON object")
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError("Broker returned an unsupported batch status schema")
    if str(batch.get("job_id") or "") != clean_job_id:
        raise ValueError("Broker batch status job_id does not match the request")

    saved_path = ""
    if output_path:
        target = Path(output_path).expanduser().resolve()
        write_json(target, batch)
        saved_path = str(target)
    summary = batch.get("summary") if isinstance(batch.get("summary"), dict) else {}
    return {
        "schema": SCHEMA,
        "ok": True,
        "status": str(batch.get("status") or "unknown"),
        "terminal": bool(batch.get("terminal")),
        "job_id": clean_job_id,
        "summary": summary,
        "broker_url": clean_broker_url,
        "broker_tool": STATUS_TOOL,
        "broker_control_requests_made": 1,
        "direct_provider_requests_made": False,
        "read_only": True,
        "batch_status_path": str(batch.get("status_path") or ""),
        "saved_status_path": saved_path,
        "batch": batch,
    }


def _clean_job_id(value: str) -> str:
    clean = str(value or "").strip()
    if (
        not clean.startswith("model_batch_")
        or len(clean) > 160
        or Path(clean).name != clean
        or "/" in clean
        or "\\" in clean
    ):
        raise ValueError("invalid consented model batch job_id")
    return clean
