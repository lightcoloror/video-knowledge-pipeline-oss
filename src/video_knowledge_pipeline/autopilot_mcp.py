from __future__ import annotations

from pathlib import Path
from typing import Any

from .autopilot import autopilot_next, autopilot_start, autopilot_status, autopilot_submit


def build_server() -> Any:
    """Build the deliberately narrow, four-tool Autopilot MCP profile."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install VKP with the mcp extra to use the Autopilot MCP profile") from exc

    server = FastMCP("video-knowledge-pipeline-autopilot")

    @server.tool(name="autopilot_start")
    def start_tool(
        bundle_or_batch: str,
        runner: str = "agent-session",
        execution_route: str = "auto",
    ) -> dict[str, Any]:
        return autopilot_start(bundle_or_batch, runner=runner, execution_route=execution_route)

    @server.tool(name="autopilot_next")
    def next_tool(job_id: str, retry_provider: bool = False) -> dict[str, Any]:
        return autopilot_next(job_id, retry_provider=retry_provider)

    @server.tool(name="autopilot_submit")
    def submit_tool(job_id: str, input_json: str) -> dict[str, Any]:
        return autopilot_submit(job_id, input_json=Path(input_json))

    @server.tool(name="autopilot_status")
    def status_tool(job_id: str) -> dict[str, Any]:
        return autopilot_status(job_id)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
