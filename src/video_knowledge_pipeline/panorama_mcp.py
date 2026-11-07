from __future__ import annotations

from typing import Any

from .panorama_cli import (
    panorama_import,
    panorama_prepare,
    panorama_restore,
    panorama_run,
    panorama_status,
)
from .panorama_preflight import panorama_doctor


def build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install VKP with the mcp extra to use panorama MCP") from exc
    server = FastMCP("video-knowledge-pipeline-panorama")

    server.tool(name="panorama_doctor")(panorama_doctor_tool)
    server.tool(name="panorama_prepare")(panorama_prepare_tool)
    server.tool(name="panorama_run")(panorama_run_tool)
    server.tool(name="panorama_import")(panorama_import_tool)
    server.tool(name="panorama_status")(panorama_status_tool)
    server.tool(name="panorama_restore")(panorama_restore_tool)
    return server


def panorama_doctor_tool(
    media_or_bundle: str = "",
    projection: str = "",
    input_fov: float = 0.0,
    lens_order: str = "",
) -> dict[str, Any]:
    return panorama_doctor(
        media_or_bundle or None,
        projection=projection or None,
        input_fov=input_fov or None,
        lens_order=lens_order or None,
    )


def panorama_prepare_tool(
    bundle_dir: str,
    projection: str,
    indexes: list[int] | None = None,
    frame_count: int = 8,
    limit: int = 16,
    face_size: int = 512,
    input_fov: float = 0.0,
    lens_order: str = "",
    run_id: str = "",
    write: bool = True,
) -> dict[str, Any]:
    return panorama_prepare(
        bundle_dir,
        projection=projection,
        indexes=indexes,
        frame_count=frame_count,
        limit=limit,
        face_size=face_size,
        input_fov=input_fov or None,
        lens_order=lens_order or None,
        run_id=run_id or None,
        write=write,
    )


def panorama_run_tool(bundle_dir: str, run_id: str, execute_projection: bool = False) -> dict[str, Any]:
    return panorama_run(bundle_dir, run_id=run_id, execute_projection=execute_projection)


def panorama_import_tool(bundle_dir: str, run_id: str, input_json: str, write: bool = True) -> dict[str, Any]:
    return panorama_import(bundle_dir, run_id=run_id, input_json=input_json, write=write)


def panorama_status_tool(bundle_dir: str, run_id: str = "") -> dict[str, Any]:
    return panorama_status(bundle_dir, run_id=run_id)


def panorama_restore_tool(
    bundle_dir: str,
    run_id: str,
    execute: bool = False,
    confirm_run_id: str = "",
) -> dict[str, Any]:
    return panorama_restore(
        bundle_dir,
        run_id=run_id,
        execute=execute,
        confirm_run_id=confirm_run_id,
    )


def main() -> None:
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
