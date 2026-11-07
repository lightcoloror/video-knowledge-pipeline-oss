from __future__ import annotations

from typing import Any

from .personal_vlog_runtime import (
    personal_vlog_prepare,
    personal_vlog_resume,
    personal_vlog_run,
    personal_vlog_status,
)


def build_server() -> Any:
    """Build the deliberately narrow four-tool personal Vlog MCP profile."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install VKP with the mcp extra to use personal Vlog MCP") from exc
    server = FastMCP("video-knowledge-pipeline-personal-vlog")

    @server.tool(name="personal_vlog_prepare")
    def prepare_tool(
        event_root: str,
        project_dir: str,
        max_depth: int = 2,
        max_items: int = 2000,
    ) -> dict[str, Any]:
        return personal_vlog_prepare(
            event_root,
            project_dir,
            max_depth=max_depth,
            max_items=max_items,
        )

    @server.tool(name="personal_vlog_run")
    def run_tool(
        project_dir: str,
        presets: str = "all",
        stop_after: str = "",
    ) -> dict[str, Any]:
        return personal_vlog_run(project_dir, presets=presets, stop_after=stop_after)

    @server.tool(name="personal_vlog_resume")
    def resume_tool(
        project_dir: str,
        run_id: str,
        stop_after: str = "",
    ) -> dict[str, Any]:
        return personal_vlog_resume(project_dir, run_id, stop_after=stop_after)

    @server.tool(name="personal_vlog_status")
    def status_tool(project_dir: str, run_id: str = "") -> dict[str, Any]:
        return personal_vlog_status(project_dir, run_id)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
