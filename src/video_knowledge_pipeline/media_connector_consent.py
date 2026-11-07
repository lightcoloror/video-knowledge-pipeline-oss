from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .media_capability_registry import media_capability
from .mediakit_cli_adapter import mediakit_cli_status
from .media_route_settings import (
    build_media_route_snapshot,
    media_route_settings_status,
    save_media_route_settings,
)
from .model_connector_consent import (
    create_model_connector_consent,
    reserve_model_connector_attempt,
    validate_model_connector_consent,
)
from .storage import read_json
from .trusted_model_connector_policy import TrustedModelConnectorPolicy


SCHEMA = "video_knowledge_pipeline.media_connector_preflight.v1"
DEFAULT_FILENAME = "media-connector-consent.json"


def create_media_connector_consent(
    root_dir: str | Path,
    *,
    task: str,
    artifact_paths: list[str | Path],
    settings_path: str | Path | None = None,
    instructions: str = "",
    purpose: str = "approved remote media capability task",
    expires_hours: float = 24,
    max_calls: int = 1,
    max_estimated_cost_usd: float,
    max_cost_per_call_usd: float | None = None,
    confirm_data_export: bool = False,
    output_path: str | Path | None = None,
    policy: TrustedModelConnectorPolicy | None = None,
    write: bool = True,
) -> dict[str, Any]:
    capability = media_capability(task)
    active_policy = policy or _default_policy()
    if settings_path:
        active_policy.require_path(settings_path, label="settings_path")
    route_status = build_media_route_snapshot(
        capability["task"],
        settings_path=settings_path,
    )

    root = active_policy.require_path(root_dir, label="root_dir", must_exist=False)
    artifacts = [active_policy.require_path(path, label="artifact_path") for path in artifact_paths]
    route = route_status["route"]
    _require_route_policy(active_policy, route)
    consent_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else root / DEFAULT_FILENAME
    )
    return create_model_connector_consent(
        root,
        task=capability["task"],
        artifact_paths=artifacts,
        route_snapshot=route,
        instructions=instructions,
        purpose=purpose,
        expires_hours=expires_hours,
        max_calls=max_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
        max_cost_per_call_usd=max_cost_per_call_usd,
        confirm_data_export=confirm_data_export,
        output_path=consent_path,
        write=write,
    )


def media_connector_consent_status(
    consent_path: str | Path,
    *,
    expected_route_revision: str = "",
    expected_calls: int = 1,
    settings_path: str | Path | None = None,
    policy: TrustedModelConnectorPolicy | None = None,
    route_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(consent_path).expanduser().resolve()
    try:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError("media consent must be a JSON object")
        task = media_capability(str(payload.get("task") or ""))["task"]
        active_policy = policy or _default_policy()
        if settings_path:
            active_policy.require_path(settings_path, label="settings_path")
        route_status = build_media_route_snapshot(task, settings_path=settings_path)

        route = route_snapshot or _consent_route_snapshot(payload) or route_status["route"]
        active_policy.require_consent_scope(
            path,
            expected_task=task,
            require_execution_contract=True,
        )
        _require_route_policy(active_policy, route)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema": SCHEMA,
            "status": "blocked",
            "valid": False,
            "ready_for_execution": False,
            "consent_path": str(path),
            "blockers": [{"key": "media_consent_preflight_failed", "message": str(exc)}],
            "network_calls_made": 0,
            "operator_boundary": _operator_boundary(),
        }
    result = validate_model_connector_consent(
        path,
        route_snapshot=route,
        expected_route_revision=expected_route_revision,
        expected_task=task,
        expected_calls=expected_calls,
    )
    credential_configured = bool(str(os.environ.get("MEDIAKIT_API_KEY") or "").strip())
    cli = mediakit_cli_status()
    blockers = list(result.get("blockers") or [])
    if not credential_configured:
        blockers.append(
            {
                "key": "mediakit_credential_missing",
                "message": "MEDIAKIT_API_KEY is not configured in the Broker process environment",
            }
        )
    if not cli.get("available"):
        blockers.append(
            {
                "key": "mediakit_cli_unavailable",
                "message": "Official MediaKit CLI is not installed or not on PATH; no media upload will be attempted.",
            }
        )
    ready = bool(result.get("valid")) and credential_configured and bool(cli.get("available"))
    return {
        **result,
        "schema": SCHEMA,
        "status": "ready_for_execution" if ready else "blocked",
        "valid": bool(result.get("valid")),
        "ready_for_execution": ready,
        "credential": {
            "env": "MEDIAKIT_API_KEY",
            "configured": credential_configured,
            "value_exposed": False,
        },
        "route_status": route_status,
        "execution_tool": cli,
        "blockers": blockers,
        "network_calls_made": 0,
        "operator_boundary": _operator_boundary(),
    }


def reserve_media_connector_attempt(
    consent_path: str | Path,
    *,
    expected_route_revision: str,
    expected_calls: int = 1,
    settings_path: str | Path | None = None,
    policy: TrustedModelConnectorPolicy | None = None,
    route_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(consent_path).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("media consent must be a JSON object")
    task = media_capability(str(payload.get("task") or ""))["task"]
    active_policy = policy or _default_policy()
    if settings_path:
        active_policy.require_path(settings_path, label="settings_path")
    route_status = build_media_route_snapshot(task, settings_path=settings_path)

    route = route_snapshot or _consent_route_snapshot(payload) or route_status["route"]
    if not str(os.environ.get("MEDIAKIT_API_KEY") or "").strip():
        raise ValueError(
            "MEDIAKIT_API_KEY must be configured before reserving a media connector attempt"
        )
    active_policy.require_consent_scope(
        path,
        expected_task=task,
        require_execution_contract=True,
    )
    _require_route_policy(active_policy, route)
    return reserve_model_connector_attempt(
        path,
        route_snapshot=route,
        expected_route_revision=expected_route_revision,
        expected_task=task,
        expected_calls=expected_calls,
    )


def media_connector_preflight(
    consent_path: str | Path,
    *,
    route_revision: str,
    expected_calls: int = 1,
    settings_path: str | Path | None = None,
    policy: TrustedModelConnectorPolicy | None = None,
    route_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return media_connector_consent_status(
        consent_path,
        expected_route_revision=route_revision,
        expected_calls=expected_calls,
        settings_path=settings_path,
        policy=policy,
        route_snapshot=route_snapshot,
    )


def _consent_route_snapshot(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Reuse the immutable route that was hash-bound into an existing consent."""
    route = payload.get("route")
    if not isinstance(route, dict):
        return None
    required = ("route_id", "route_revision", "execution_location", "deployments", "destinations")
    if not all(route.get(key) for key in required):
        return None
    return route

def _require_route_policy(
    policy: TrustedModelConnectorPolicy,
    route: dict[str, Any],
) -> None:
    for deployment in route.get("deployments") or []:
        policy.require_destination_identity(deployment)
    for destination in route.get("destinations") or []:
        policy.require_destination_identity({"base_url": destination})


def _default_policy() -> TrustedModelConnectorPolicy:
    return TrustedModelConnectorPolicy.from_environment(
        default_root=Path(__file__).resolve().parents[2]
    )


def _operator_boundary() -> dict[str, Any]:
    return {
        "read_only_preflight": False,
        "execute_tool_available": bool(mediakit_cli_status().get("available")),
        "real_remote_requests_allowed": "only_after_valid_consent_reservation",
        "consent_creation_exposed_by_mcp": False,
        "arbitrary_provider_urls_allowed": False,
        "automatic_upload_allowed": "only_for_explicitly_consented_artifacts",
        "silent_fallback_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configure and inspect consent-gated MediaKit routes; execution is available through trusted_model_connector after reservation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    configure = sub.add_parser("configure")
    configure.add_argument("--settings-path", default="")
    configure.add_argument("--upload-destination", action="append", default=[])
    configure.add_argument("--route-id", default="mediakit-remote-approved")
    configure.add_argument("--max-poll-attempts", type=int, default=12)
    configure.add_argument("--poll-interval-seconds", type=float, default=10)
    configure.add_argument("--timeout-seconds", type=int, default=900)
    configure.add_argument("--write", action="store_true")
    route_status = sub.add_parser("route-status")
    route_status.add_argument("--task", default="")
    route_status.add_argument("--settings-path", default="")
    create = sub.add_parser("create")
    create.add_argument("root_dir")
    create.add_argument("--task", required=True)
    create.add_argument("--artifact", action="append", required=True)
    create.add_argument("--settings-path", default="")
    create.add_argument("--instructions", default="")
    create.add_argument("--expires-hours", type=float, default=24)
    create.add_argument("--max-calls", type=int, default=1)
    create.add_argument("--max-estimated-cost-usd", type=float, required=True)
    create.add_argument("--max-cost-per-call-usd", type=float)
    create.add_argument("--confirm-data-export", action="store_true")
    create.add_argument("--output-path", default="")
    status = sub.add_parser("status")
    status.add_argument("consent_path")
    status.add_argument("--route-revision", required=True)
    status.add_argument("--expected-calls", type=int, default=1)
    status.add_argument("--settings-path", default="")
    args = parser.parse_args(argv)
    if args.command == "configure":
        result = save_media_route_settings(
            upload_destinations=args.upload_destination,
            route_id=args.route_id,
            max_poll_attempts=args.max_poll_attempts,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            settings_path=args.settings_path or None,
            write=args.write,
        )
    elif args.command == "route-status":
        result = media_route_settings_status(
            task=args.task,
            settings_path=args.settings_path or None,
        )
    elif args.command == "create":
        result = create_media_connector_consent(
            args.root_dir,
            task=args.task,
            artifact_paths=args.artifact,
            settings_path=args.settings_path or None,
            instructions=args.instructions,
            expires_hours=args.expires_hours,
            max_calls=args.max_calls,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            max_cost_per_call_usd=args.max_cost_per_call_usd,
            confirm_data_export=args.confirm_data_export,
            output_path=args.output_path or None,
        )
    else:
        result = media_connector_consent_status(
            args.consent_path,
            expected_route_revision=args.route_revision,
            expected_calls=args.expected_calls,
            settings_path=args.settings_path or None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") not in {"blocked", "invalid", "missing"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
