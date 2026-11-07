from __future__ import annotations

from typing import Any

from .media_connector_consent import media_connector_preflight
from .trusted_model_connector import (
    execute_consented_bundle_vision,
    execute_consented_model_task,
    compact_model_execution_receipt,
    execute_local_model_task,
    resolve_legacy_bundle_vision_route,
    trusted_model_connector_capabilities,
    trusted_model_connector_status,
)


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("Install MCP support with: pip install -e .[mcp]") from exc

    server = FastMCP("vkp-trusted-online-model-connector")

    @server.tool()
    def model_connector_capabilities() -> dict[str, Any]:
        """List model tasks, candidate media capabilities, and consent boundaries."""
        return trusted_model_connector_capabilities()

    @server.tool()
    def model_connector_consent_status(
        consent_path: str,
        route_revision: str = "",
        expected_task: str = "",
        expected_calls: int = 1,
    ) -> dict[str, Any]:
        """Validate a human-created consent without sending any data."""
        return trusted_model_connector_status(
            consent_path,
            expected_task=expected_task,
            expected_route_revision=route_revision,
            expected_calls=expected_calls,
        )

    @server.tool()
    def media_connector_preflight_tool(
        consent_path: str,
        route_revision: str,
        expected_calls: int = 1,
        settings_path: str = "",
    ) -> dict[str, Any]:
        """Validate a MediaKit route and consent without uploading or executing."""
        return media_connector_preflight(
            consent_path,
            route_revision=route_revision,
            expected_calls=expected_calls,
            settings_path=settings_path or None,
        )

    @server.tool()
    def execute_consented_model_task_tool(
        consent_path: str,
        route_revision: str,
        write: bool = True,
        return_mode: str = "receipt",
    ) -> dict[str, Any]:
        """Execute one task using only the route and deployments locked by consent."""
        if return_mode not in {"receipt", "full"}:
            raise ValueError("return_mode must be receipt or full")
        execution = execute_consented_model_task(
            consent_path,
            expected_route_revision=route_revision,
            write=write,
        )
        return compact_model_execution_receipt(execution) if return_mode == "receipt" else execution

    @server.tool()
    def execute_local_model_task_tool(
        task: str,
        artifact_paths: list[str],
        route_id: str = "",
        instructions: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        """Execute a local-only task; configured deployments must remain loopback-only."""
        return execute_local_model_task(
            task,
            artifact_paths,
            route_id=route_id,
            instructions=instructions,
            write=write,
        )

    @server.tool()
    def execute_consented_semantic_vision(
        bundle_dir: str,
        indexes: list[int],
        export_consent: str,
        route_revision: str,
        image_max_edge: int = 512,
        image_jpeg_quality: int = 55,
    ) -> dict[str, Any]:
        """Run the configured singleton legacy visual route under its existing consent."""
        provider_config, _route = resolve_legacy_bundle_vision_route(
            "semantic", expected_route_revision=route_revision
        )
        return execute_consented_bundle_vision(
            bundle_dir,
            mode="semantic",
            indexes=indexes,
            export_consent=export_consent,
            provider_config=provider_config,
            image_max_edge=image_max_edge,
            image_jpeg_quality=image_jpeg_quality,
        )

    @server.tool()
    def execute_consented_temporal_vision(
        bundle_dir: str,
        indexes: list[int],
        export_consent: str,
        route_revision: str,
        frame_count: int = 8,
        image_max_edge: int = 512,
        image_jpeg_quality: int = 55,
    ) -> dict[str, Any]:
        """Run the configured singleton legacy temporal route under its existing consent."""
        provider_config, _route = resolve_legacy_bundle_vision_route(
            "temporal", expected_route_revision=route_revision
        )
        return execute_consented_bundle_vision(
            bundle_dir,
            mode="temporal",
            indexes=indexes,
            export_consent=export_consent,
            provider_config=provider_config,
            frame_count=frame_count,
            image_max_edge=image_max_edge,
            image_jpeg_quality=image_jpeg_quality,
        )

    server.run()


if __name__ == "__main__":
    main()