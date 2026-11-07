from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)
from video_knowledge_pipeline.trusted_model_connector_remote_mcp import build_server


class WorkflowToolManager:
    def public_snapshot(self) -> dict[str, Any]:
        return {"provider_rate_limit_owner": "litellm_proxy"}

    def submit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def submit_workflow(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def status(self, job_id: str) -> dict[str, Any]:
        return {"job_id": job_id}


def test_remote_mcp_exposes_named_workflow_without_provider_overrides() -> None:
    root = Path(__file__).resolve().parents[1]
    server = build_server(
        policy=TrustedModelConnectorPolicy(
            (root,), frozenset({"api.example"})
        ),
        host="127.0.0.1",
        port=8766,
        batch_manager=WorkflowToolManager(),  # type: ignore[arg-type]
    )
    tools = asyncio.run(server.list_tools())
    rows = {tool.name: tool for tool in tools}

    workflow = rows["submit_consented_model_workflow_tool"]
    assert workflow.annotations.readOnlyHint is False
    assert workflow.annotations.openWorldHint is True
    assert {"bundle_dir", "nodes"} <= set(workflow.inputSchema["required"])
    assert "provider_config" not in workflow.inputSchema["properties"]
    assert "route_revision" not in workflow.inputSchema["properties"]
