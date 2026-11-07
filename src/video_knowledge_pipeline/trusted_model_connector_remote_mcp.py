from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import os
import threading
from pathlib import Path
from typing import Any

from mcp.types import ToolAnnotations

from .consented_model_batch import ConsentedModelBatchManager
from .media_connector_consent import media_connector_preflight
from .model_business_authorization import (
    create_business_child_consent,
    validate_model_business_authorization,
)
from .trusted_model_connector import (
    execute_consented_bundle_vision,
    execute_consented_model_task,
    compact_model_execution_receipt,
    execute_local_model_task,
    resolve_legacy_bundle_vision_route,
    trusted_model_connector_capabilities,
    trusted_model_connector_status,
)
from .trusted_model_connector_policy import TrustedModelConnectorPolicy


READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
EXECUTION_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
LOCAL_WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


_PARITY_BACKGROUND_SCHEMA = (
    "video_knowledge_pipeline.coding_tool_provider_parity_submission.v1"
)
_PARITY_BACKGROUND_LOCK = threading.Lock()
_PARITY_BACKGROUND_EXECUTORS: dict[
    str, concurrent.futures.ThreadPoolExecutor
] = {}
_PARITY_BACKGROUND_FUTURES: dict[
    str, concurrent.futures.Future[dict[str, Any]]
] = {}


def _parity_plan_uses_background_execution(consent_path: str | Path) -> bool:
    consent_file = Path(consent_path).expanduser().resolve()
    plan_file = consent_file.parent.parent / "parity-plan.json"
    if not plan_file.is_file():
        return False
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    contract = (
        plan.get("comparison_contract")
        if isinstance(plan.get("comparison_contract"), dict)
        else {}
    )
    return str(contract.get("request_profile_id") or "") == "capability_ceiling_v1"


def _execute_parity_background(
    consent_path: Path,
    expected_route_revision: str,
    write: bool,
) -> dict[str, Any]:
    from .coding_tool_provider_parity import (
        execute_coding_tool_provider_parity_consent,
    )

    return execute_coding_tool_provider_parity_consent(
        consent_path,
        expected_route_revision=expected_route_revision,
        write=write,
    )


def _submit_parity_background(
    consent_path: str | Path,
    *,
    consent: dict[str, Any],
    expected_route_revision: str,
    write: bool,
) -> dict[str, Any]:
    consent_file = Path(consent_path).expanduser().resolve()
    result_path = consent_file.parent / "execution.json"
    candidate_id = consent_file.parent.name
    destinations = [
        str(value).strip()
        for value in consent.get("authorized_destinations") or []
        if str(value).strip()
    ]
    if len(destinations) != 1:
        raise ValueError("provider parity consent must authorize exactly one destination")
    destination = destinations[0]
    submission_key = str(consent_file).casefold()
    if result_path.is_file():
        return {
            "schema": _PARITY_BACKGROUND_SCHEMA,
            "status": "existing_result",
            "candidate_id": candidate_id,
            "destination": destination,
            "consent_path": str(consent_file),
            "result_path": str(result_path),
            "external_request_started_by_this_call": False,
        }

    with _PARITY_BACKGROUND_LOCK:
        existing = _PARITY_BACKGROUND_FUTURES.get(submission_key)
        if existing is not None:
            return {
                "schema": _PARITY_BACKGROUND_SCHEMA,
                "status": "completed" if existing.done() else "in_progress",
                "candidate_id": candidate_id,
                "destination": destination,
                "consent_path": str(consent_file),
                "result_path": str(result_path),
                "external_request_started_by_this_call": False,
            }
        executor = _PARITY_BACKGROUND_EXECUTORS.get(destination)
        if executor is None:
            safe_name = destination.replace("https://", "").replace(".", "-")
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"vkp-parity-{safe_name}",
            )
            _PARITY_BACKGROUND_EXECUTORS[destination] = executor
        future = executor.submit(
            _execute_parity_background,
            consent_file,
            expected_route_revision,
            write,
        )
        _PARITY_BACKGROUND_FUTURES[submission_key] = future

    return {
        "schema": _PARITY_BACKGROUND_SCHEMA,
        "status": "accepted",
        "candidate_id": candidate_id,
        "destination": destination,
        "consent_path": str(consent_file),
        "result_path": str(result_path),
        "external_request_started_by_this_call": False,
        "execution": "background_single_destination_queue",
        "poll": "read result_path for the terminal execution artifact",
    }


def _locked_route_revision(consent: dict[str, Any], requested: str = "") -> str:
    route = consent.get("route") if isinstance(consent.get("route"), dict) else {}
    locked = str(route.get("route_revision") or "").strip()
    explicit = str(requested or "").strip()
    if explicit and locked and explicit != locked:
        raise ValueError("requested route_revision differs from the consent-locked revision")
    resolved = explicit or locked
    if not resolved:
        raise ValueError("route_revision is required and is not locked by consent")
    return resolved


def build_server(
    *,
    policy: TrustedModelConnectorPolicy,
    host: str = "127.0.0.1",
    port: int = 8766,
    streamable_http_path: str = "/mcp",
    batch_manager: ConsentedModelBatchManager | None = None,
) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("Install MCP support with: pip install -e .[mcp]") from exc

    server = FastMCP(
        "vkp-trusted-capability-broker",
        host=host,
        port=int(port),
        streamable_http_path=_normalise_http_path(streamable_http_path),
    )
    manager = batch_manager or ConsentedModelBatchManager(
        project_root=policy.allowed_roots[0],
        policy=policy,
        global_parallel_limit=int(
            os.environ.get("VKP_MODEL_BATCH_GLOBAL_CONCURRENCY", "4")
        ),
        maximum_parallel_per_destination=int(
            os.environ.get("VKP_MODEL_BATCH_MAX_DESTINATION_CONCURRENCY", "2")
        ),
    )

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def model_connector_capabilities() -> dict[str, Any]:
        """List supported tasks, consent boundaries, and the active egress policy."""
        result = trusted_model_connector_capabilities()
        result["runtime_policy"] = policy.public_snapshot()
        result["transport"] = {
            "kind": "streamable-http",
            "loopback_only": True,
            "platform_access": "secure_mcp_tunnel",
        }
        result["batch_execution"] = manager.public_snapshot()
        return result

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def model_connector_consent_status(
        consent_path: str,
        route_revision: str = "",
        expected_task: str = "",
    ) -> dict[str, Any]:
        """Validate a human-created consent and its saved route without sending data."""
        policy.require_consent_scope(
            consent_path,
            expected_task=expected_task,
        )
        return trusted_model_connector_status(
            consent_path,
            expected_task=expected_task,
            expected_route_revision=route_revision,
        )

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def model_business_authorization_status_tool(
        authorization_path: str,
    ) -> dict[str, Any]:
        """Validate one confirmed business envelope without sending data."""
        policy.require_path(
            authorization_path, label="business authorization"
        )
        return validate_model_business_authorization(
            authorization_path, policy=policy
        )

    @server.tool(annotations=LOCAL_WRITE_ANNOTATIONS)
    def create_business_child_consent_tool(
        authorization_path: str,
        stage_id: str,
        artifact_paths: list[str],
        producer: str,
        lineage_input_paths: list[str],
        max_calls: int = 1,
    ) -> dict[str, Any]:
        """Mint exact consent v2 inside a pre-confirmed business envelope."""
        policy.require_path(
            authorization_path, label="business authorization"
        )
        for artifact_path in artifact_paths:
            policy.require_path(artifact_path, label="derived artifact")
        for input_path in lineage_input_paths:
            policy.require_path(input_path, label="lineage input")
        return create_business_child_consent(
            authorization_path,
            stage_id=stage_id,
            artifact_paths=artifact_paths,
            producer=producer,
            input_paths=lineage_input_paths,
            max_calls=max_calls,
            policy=policy,
            write=True,
        )

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def media_connector_preflight_tool(
        consent_path: str,
        route_revision: str,
        expected_calls: int = 1,
        settings_path: str = "",
    ) -> dict[str, Any]:
        """Validate saved MediaKit route/consent state without uploading or executing."""
        policy.require_path(consent_path, label="consent_path")
        if settings_path:
            policy.require_path(settings_path, label="settings_path")
        return media_connector_preflight(
            consent_path,
            route_revision=route_revision,
            expected_calls=expected_calls,
            settings_path=settings_path or None,
            policy=policy,
        )

    @server.tool(annotations=EXECUTION_ANNOTATIONS)
    def execute_consented_model_task_tool(
        consent_path: str,
        route_revision: str = "",
        write: bool = True,
        return_mode: str = "receipt",
    ) -> dict[str, Any]:
        """Execute one task using only the route and deployments locked by consent."""
        consent = policy.require_consent_scope(
            consent_path,
            require_execution_contract=True,
        )
        locked_route_revision = _locked_route_revision(consent, route_revision)
        if str(consent.get("task") or "") == "provider_task_benchmark":
            if _parity_plan_uses_background_execution(consent_path):
                return _submit_parity_background(
                    consent_path,
                    consent=consent,
                    expected_route_revision=locked_route_revision,
                    write=write,
                )
            from .coding_tool_provider_parity import (
                execute_coding_tool_provider_parity_consent,
            )

            return execute_coding_tool_provider_parity_consent(
                consent_path,
                expected_route_revision=locked_route_revision,
                write=write,
            )
        if return_mode not in {"receipt", "full"}:
            raise ValueError("return_mode must be receipt or full")
        execution = execute_consented_model_task(
            consent_path,
            expected_route_revision=locked_route_revision,
            write=write,
        )
        return compact_model_execution_receipt(execution) if return_mode == "receipt" else execution

    @server.tool(annotations=EXECUTION_ANNOTATIONS)
    def submit_consented_model_batch_tool(
        consent_paths: list[str],
        write: bool = True,
        max_parallel_global: int = 4,
        max_parallel_per_destination: int = 2,
        depends_on: list[list[int]] | None = None,
        node_ids: list[str] | None = None,
        bundle_dir: str = "",
    ) -> dict[str, Any]:
        """Validate and submit a consent-locked background batch without added retries."""
        return manager.submit(
            consent_paths,
            write=write,
            max_parallel_global=max_parallel_global,
            max_parallel_per_destination=max_parallel_per_destination,
            depends_on=depends_on,
            node_ids=node_ids,
            bundle_dir=bundle_dir,
        )


    @server.tool(annotations=EXECUTION_ANNOTATIONS)
    def submit_consented_model_workflow_tool(
        bundle_dir: str,
        nodes: list[dict[str, Any]],
        write: bool = True,
        max_parallel_global: int = 4,
        max_parallel_per_destination: int = 2,
    ) -> dict[str, Any]:
        """Submit named consent nodes through the existing batch DAG and run registry."""
        return manager.submit_workflow(
            bundle_dir=bundle_dir,
            nodes=nodes,
            write=write,
            max_parallel_global=max_parallel_global,
            max_parallel_per_destination=max_parallel_per_destination,
        )
    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def consented_model_batch_status_tool(job_id: str) -> dict[str, Any]:
        """Read durable progress and the terminal result for a submitted batch."""
        return manager.status(job_id)

    @server.tool(annotations=EXECUTION_ANNOTATIONS)
    def execute_local_model_task_tool(
        task: str,
        artifact_paths: list[str],
        route_id: str = "",
        instructions: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        """Execute a local-only model task without remote export consent."""
        for artifact_path in artifact_paths:
            policy.require_path(artifact_path, label="artifact_path")
        return execute_local_model_task(
            task,
            artifact_paths,
            route_id=route_id,
            instructions=instructions,
            write=write,
        )

    @server.tool(annotations=EXECUTION_ANNOTATIONS)
    def execute_consented_semantic_vision(
        bundle_dir: str,
        indexes: list[int],
        export_consent: str,
        route_revision: str,
        image_max_edge: int = 512,
        image_jpeg_quality: int = 55,
    ) -> dict[str, Any]:
        """Run a singleton legacy vision route without accepting caller provider URLs."""
        policy.require_path(bundle_dir, label="bundle_dir")
        policy.require_path(export_consent, label="export_consent")
        provider_config, route = resolve_legacy_bundle_vision_route(
            "semantic", expected_route_revision=route_revision
        )
        for deployment in route["deployments"]:
            policy.require_destination_identity(deployment)
        return execute_consented_bundle_vision(
            bundle_dir,
            mode="semantic",
            indexes=indexes,
            export_consent=export_consent,
            provider_config=provider_config,
            image_max_edge=image_max_edge,
            image_jpeg_quality=image_jpeg_quality,
        )

    @server.tool(annotations=EXECUTION_ANNOTATIONS)
    def execute_consented_temporal_vision(
        bundle_dir: str,
        indexes: list[int],
        export_consent: str,
        route_revision: str,
        frame_count: int = 8,
        image_max_edge: int = 512,
        image_jpeg_quality: int = 55,
    ) -> dict[str, Any]:
        """Run a singleton legacy temporal route without accepting caller provider URLs."""
        policy.require_path(bundle_dir, label="bundle_dir")
        policy.require_path(export_consent, label="export_consent")
        provider_config, route = resolve_legacy_bundle_vision_route(
            "temporal", expected_route_revision=route_revision
        )
        for deployment in route["deployments"]:
            policy.require_destination_identity(deployment)
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

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the VKP trusted capability broker"
    )
    parser.add_argument(
        "--host", default=os.environ.get("VKP_MODEL_CONNECTOR_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VKP_MODEL_CONNECTOR_PORT", "8766")),
    )
    parser.add_argument(
        "--path", default=os.environ.get("VKP_MODEL_CONNECTOR_PATH", "/mcp")
    )
    args = parser.parse_args(argv)
    _require_loopback_host(args.host)
    project_root = Path(__file__).resolve().parents[2]
    policy = TrustedModelConnectorPolicy.from_environment(default_root=project_root)
    server = build_server(
        policy=policy,
        host=args.host,
        port=args.port,
        streamable_http_path=args.path,
    )
    server.run(transport="streamable-http")


def _normalise_http_path(value: str) -> str:
    path = str(value or "/mcp").strip()
    if not path.startswith("/") or "?" in path or "#" in path:
        raise ValueError(
            "MCP HTTP path must be an absolute path without query or fragment"
        )
    return path.rstrip("/") or "/mcp"


def _require_loopback_host(value: str) -> None:
    host = str(value or "").strip().lower()
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(
        "streamable-http must bind to a loopback address; use Secure MCP Tunnel for platform access"
    )


if __name__ == "__main__":
    main()
