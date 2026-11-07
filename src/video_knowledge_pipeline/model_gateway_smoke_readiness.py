from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .model_api_settings import public_model_api_settings_status
from .model_gateway import model_gateway_doctor
from .model_gateway_acceptance import (
    LANE_FILENAMES,
    build_temporal_gateway_acceptance_manifest,
    compare_model_gateway_results,
)
from .models import now_iso

from .storage import read_json, write_json
from .trusted_model_connector import trusted_model_connector_status


SCHEMA = "video_knowledge_pipeline.model_gateway_smoke_readiness.v1"
REQUIRED_ROUTES = (
    ("remote_text", "summary_rewrite", "remote", "proxy", None, "smart_summary_rewrite"),
    ("remote_vision", "temporal_sequence", "remote", "proxy", None, "temporal_visual_analysis"),
    ("remote_asr", "asr", "remote", "proxy", {"openai_compatible_asr", "groq_asr", "mistral_asr"}, "cloud_asr"),
    ("remote_ocr", "ocr", "remote", "proxy", {"mistral", "mistral_compatible_ocr"}, "online_ocr"),
    ("local_vlm", "temporal_sequence", "local", "proxy", {"local_qwen_vl", "local_vlm", "openai_compatible"}, ""),
    ("local_speaches", "asr", "local", "proxy", {"openai_compatible_asr"}, ""),
)
ONLINE_PIPELINE_TASKS = (
    "asr",
    "ocr",
    "document_visual",
    "semantic_frame",
    "temporal_sequence",
    "text_llm",
    "summary_rewrite",
    "transcript_correction",
)


def model_gateway_smoke_readiness(
    bundle_dir: str | Path,
    *,
    indexes: list[int],
    frame_count: int = 8,
    consent_paths: list[str | Path] | None = None,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    gateway_config_path: str | Path | None = None,
    port_record_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    online_only: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else root / "exports" / "model-gateway-acceptance"
    settings = public_model_api_settings_status(settings_path, secrets_path)
    gateway = model_gateway_doctor(
        gateway_config_path=gateway_config_path,
        port_record_path=port_record_path,
        probe_http=True,
    )
    temporal = build_temporal_gateway_acceptance_manifest(
        root,
        indexes=indexes,
        frame_count=frame_count,
        write=False,
    )
    representative_routes = _route_readiness(settings)
    if online_only:
        routes = _online_pipeline_route_readiness(settings)
        configuration_warnings = []
        consent_routes = [
            row
            for row in representative_routes
            if row.get("execution_location") == "remote"
        ]
    else:
        routes = representative_routes
        configuration_warnings = _configuration_warnings(routes)
        consent_routes = routes
    consents = _consent_readiness(consent_paths or [], consent_routes, temporal)
    lane_paths = {lane: out / filename for lane, filename in LANE_FILENAMES.items()}
    comparison = compare_model_gateway_results(
        lane_paths["A"],
        lane_paths["B"],
        lane_paths["C"],
        output_dir=out,
        sample_id="temporal-fixed-groups",
        write=False,
    )
    checks = {
        str(row.get("key") or ""): row
        for row in gateway.get("checks") or []
        if isinstance(row, dict)
    }
    route_ready = all(bool(row["ready"]) for row in routes)
    port_ready = bool((checks.get("port_record") or {}).get("ok"))
    gateway_live = bool((checks.get("live_listener") or {}).get("ok"))
    gateway_healthy = gateway_live and gateway.get("http_status") == "ready"
    temporal_ready = temporal.get("status") == "ready"
    consent_ready = all(bool(row["valid"]) for row in consents)
    lane_rows = {
        str(row.get("lane") or ""): row
        for row in comparison.get("lanes") or []
        if isinstance(row, dict)
    }
    if online_only:
        lane_b = lane_rows.get("B") or {}
        lanes_ready = bool(lane_b.get("loaded")) and bool(
            (lane_b.get("schema_contract") or {}).get("compatible")
        )
    else:
        lanes_ready = (
            comparison.get("status") == "ready_for_review"
            and bool(comparison["comparison"]["schema_compatible"])
        )
    if not route_ready or not port_ready or not temporal_ready:
        status = "configuration_required"
    elif not gateway_healthy:
        status = "operator_start_required"
    elif not consent_ready:
        status = "operator_consent_required"
    elif not lanes_ready:
        status = "operator_smoke_required"
    else:
        status = "ready_for_final_review"
    result = {
        "schema": SCHEMA,
        "status": status,
        "mode": "online_only" if online_only else "hybrid_abc",
        "bundle_dir": str(root),
        "settings": {
            "settings_path": settings["settings_path"],
            "profile_count": len(settings.get("profiles") or []),
            "route_count": len(settings.get("route_status") or []),
            "settings_ui_url": settings["settings_ui_url"],
        },
        "gateway": {
            "status": gateway.get("status"),
            "live": gateway_live,
            "healthy": gateway_healthy,
            "port_recorded_and_owned": port_ready,
            "bind_available": bool((checks.get("bind_available") or {}).get("ok")),
            "host": gateway["gateway"]["host"],
            "port": gateway["gateway"]["port"],
        },
        "route_requirements": routes,
        "route_ready_count": sum(bool(row["ready"]) for row in routes),
        "route_required_count": len(routes),
        "configuration_warnings": configuration_warnings,
        "configuration_warning_count": len(configuration_warnings),
        "remote_consents": consents,
        "consent_ready_count": sum(bool(row["valid"]) for row in consents),
        "consent_required_count": len(consents),
        "temporal_sample": {
            "status": temporal.get("status"),
            "ready_group_count": temporal.get("ready_group_count"),
            "group_count": temporal.get("group_count"),
            "failed_group_count": temporal.get("failed_group_count"),
            "indexes": temporal.get("indexes"),
        },
        "lane_results": {
            "status": comparison.get("status"),
            "schema_compatible": bool(comparison["comparison"]["schema_compatible"]),
            "paths": {key: str(value) for key, value in lane_paths.items()},
            "loaded": {row["lane"]: bool(row["loaded"]) for row in comparison["lanes"]},
        },
        "next_actions": _next_actions(
            status,
            settings["settings_ui_url"],
            configuration_warnings,
            online_only=online_only,
            gateway_healthy=gateway_healthy,
        ),
        "operator_boundary": {
            "readiness_only": True,
            "does_not_start_gateway": True,
            "does_not_call_models": True,
            "does_not_upload_artifacts": True,
            "does_not_create_consent": True,
            "does_not_write_port_registry": True,
            "remote_requests_made": False,
        },
        "updated_at": now_iso(),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "model-gateway-smoke-readiness.json", result)
        (out / "model-gateway-smoke-readiness.md").write_text(
            _render_markdown(result), encoding="utf-8"
        )
        result["artifacts"] = {
            "json": str(out / "model-gateway-smoke-readiness.json"),
            "markdown": str(out / "model-gateway-smoke-readiness.md"),
        }
    return result


def _route_readiness(settings: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = {
        str(row.get("id") or ""): row
        for row in settings.get("profiles") or []
        if isinstance(row, dict)
    }
    rows = []
    for key, task, location, backend, providers, consent_task in REQUIRED_ROUTES:
        matches = []
        for route in settings.get("route_status") or []:
            if (
                not isinstance(route, dict)
                or route.get("task") != task
                or route.get("execution_location") != location
            ):
                continue
            deployments = [profiles.get(str(value)) for value in route.get("deployments") or []]
            if not deployments or any(not isinstance(profile, dict) for profile in deployments):
                continue
            if any(str(profile.get("adapter_backend") or "") != backend for profile in deployments):
                continue
            if providers and any(str(profile.get("provider") or "") not in providers for profile in deployments):
                continue
            matches.append(route)
        selected = matches[0] if matches else {}
        selected_profiles = [
            profiles.get(str(value)) for value in selected.get("deployments") or []
        ]
        selected_profiles = [
            profile for profile in selected_profiles if isinstance(profile, dict)
        ]
        allowlist_ready = (
            location == "local" or selected.get("allowlist_status") == "approved"
        )
        credentials_ready = location == "local" or (
            bool(selected_profiles)
            and all(
                bool(profile.get("api_key_configured"))
                for profile in selected_profiles
                if isinstance(profile, dict)
            )
        )
        blockers = []
        if not selected:
            blockers.append("route_missing")
        if selected and not allowlist_ready:
            blockers.append("remote_allowlist_not_approved")
        if selected and not credentials_ready:
            blockers.append("remote_credentials_missing")
        rows.append(
            {
                "key": key,
                "task": task,
                "execution_location": location,
                "adapter_backend": backend,
                "allowed_providers": sorted(providers or []),
                "ready": bool(selected) and allowlist_ready and credentials_ready,
                "allowlist_ready": allowlist_ready,
                "credentials_ready": credentials_ready,
                "blockers": blockers,
                "route_id": str(selected.get("route_id") or ""),
                "route_revision": str(selected.get("route_revision") or ""),
                "virtual_model": str(selected.get("virtual_model") or ""),
                "allowlist_status": str(selected.get("allowlist_status") or "missing"),
                "consent_task": consent_task,
                "deployments": [
                    str(profile.get("id") or "") for profile in selected_profiles
                ],
                "providers": sorted(
                    {str(profile.get("provider") or "") for profile in selected_profiles}
                ),
                "deployment_origins": sorted(
                    {
                        origin
                        for profile in selected_profiles
                        if (origin := _endpoint_origin(str(profile.get("base_url") or "")))
                    }
                ),
            }
        )
    return rows


def _online_pipeline_route_readiness(settings: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = {
        str(row.get("id") or ""): row
        for row in settings.get("profiles") or []
        if isinstance(row, dict)
    }
    routes = [
        row
        for row in settings.get("route_status") or []
        if isinstance(row, dict) and row.get("execution_location") == "remote"
    ]
    rows = []
    for task in ONLINE_PIPELINE_TASKS:
        selected = next((row for row in routes if row.get("task") == task), {})
        selected_profiles = [
            profiles.get(str(profile_id))
            for profile_id in selected.get("deployments") or []
        ]
        selected_profiles = [
            profile for profile in selected_profiles if isinstance(profile, dict)
        ]
        enabled = bool(selected_profiles) and all(
            bool(profile.get("enabled", True)) for profile in selected_profiles
        )
        proxy_only = bool(selected_profiles) and all(
            str(profile.get("adapter_backend") or "") == "proxy"
            for profile in selected_profiles
        )
        credentials_ready = bool(selected_profiles) and all(
            bool(profile.get("api_key_configured")) for profile in selected_profiles
        )
        allowlist_ready = selected.get("allowlist_status") == "approved"

        blockers = []
        if not selected:
            blockers.append("route_missing")
        if selected and not enabled:
            blockers.append("deployment_disabled")
        if selected and not proxy_only:
            blockers.append("non_proxy_deployment")
        if selected and not credentials_ready:
            blockers.append("remote_credentials_missing")
        if selected and not allowlist_ready:
            blockers.append("remote_allowlist_not_approved")

        rows.append(
            {
                "key": f"online_{task}",
                "task": task,
                "execution_location": "remote",
                "adapter_backend": "proxy",
                "allowed_providers": [],
                "ready": bool(selected) and not blockers,
                "allowlist_ready": allowlist_ready,
                "credentials_ready": credentials_ready,
                "blockers": blockers,
                "route_id": str(selected.get("route_id") or ""),
                "route_revision": str(selected.get("route_revision") or ""),
                "virtual_model": str(selected.get("virtual_model") or ""),
                "allowlist_status": str(selected.get("allowlist_status") or "missing"),
                "consent_task": "",
                "deployments": [
                    str(profile.get("id") or "") for profile in selected_profiles
                ],
                "providers": sorted(
                    {str(profile.get("provider") or "") for profile in selected_profiles}
                ),
                "deployment_origins": sorted(
                    {
                        origin
                        for profile in selected_profiles
                        if (origin := _endpoint_origin(str(profile.get("base_url") or "")))
                    }
                ),
            }
        )
    return rows
def _endpoint_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.hostname:
        return ""
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return f"{scheme}://{host}:{port}" if port is not None else f"{scheme}://{host}"


def _configuration_warnings(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {str(row.get("key") or ""): row for row in routes}
    vlm_origins = set(by_key.get("local_vlm", {}).get("deployment_origins") or [])
    asr_origins = set(by_key.get("local_speaches", {}).get("deployment_origins") or [])
    shared = sorted(vlm_origins.intersection(asr_origins))
    if not shared:
        return []
    return [
        {
            "key": "local_service_endpoint_shared",
            "severity": "warning",
            "origins": shared,
            "message": (
                "Local VLM and Speaches routes share an origin. This is valid only "
                "when one operator-managed service intentionally exposes both chat "
                "completions and audio transcriptions; otherwise configure distinct ports."
            ),
        }
    ]


def _consent_readiness(
    paths: list[str | Path],
    routes: list[dict[str, Any]],
    temporal: dict[str, Any],
) -> list[dict[str, Any]]:
    supplied: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        supplied.setdefault(str(payload.get("task") or ""), []).append((path, payload))
    temporal_paths = {
        str(Path(str(frame.get("path") or "")).expanduser().resolve())
        for group in temporal.get("groups") or []
        if isinstance(group, dict)
        for frame in group.get("frames") or []
        if isinstance(frame, dict) and str(frame.get("path") or "").strip()
    }
    temporal_calls = max(1, int(temporal.get("group_count") or 0))
    rows = []
    for route in routes:
        task = str(route.get("consent_task") or "")
        if not task:
            continue
        candidates = supplied.get(task) or []
        selected: dict[str, Any] = {}
        for path, payload in candidates:
            consent_artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
            expected_calls = (
                temporal_calls
                if route.get("key") == "remote_vision"
                else max(1, len(consent_artifacts))
                if task == "online_ocr"
                else 1
            )
            status = trusted_model_connector_status(
                path,
                expected_task=task,
                expected_route_revision=str(route.get("route_revision") or ""),
                expected_calls=expected_calls,
            )
            blockers = [
                str(row.get("key") or "")
                for row in status.get("blockers") or []
                if isinstance(row, dict)
            ]
            coverage_ready = True
            if route.get("key") == "remote_vision":
                consent_paths = {
                    str(Path(str(row.get("path") or "")).expanduser().resolve())
                    for row in consent_artifacts
                    if isinstance(row, dict) and str(row.get("path") or "").strip()
                }
                coverage_ready = bool(temporal_paths) and consent_paths == temporal_paths
                if not coverage_ready:
                    blockers.append("temporal_artifact_coverage_mismatch")
            compact = {
                "path": str(path),
                "valid": bool(status.get("valid")) and coverage_ready,
                "status": str(status.get("status") or ""),
                "remaining_calls": status.get("remaining_calls"),
                "expected_calls": expected_calls,
                "artifact_count": len(consent_artifacts),
                "artifact_coverage_ready": coverage_ready,
                "blockers": blockers,
            }
            if not selected or compact["valid"]:
                selected = compact
            if compact["valid"]:
                break
        rows.append(
            {
                "route_key": route["key"],
                "task": task,
                "route_revision": route.get("route_revision") or "",
                "valid": bool(selected.get("valid")),
                "consent_path": str(selected.get("path") or ""),
                "status": str(selected.get("status") or "missing"),
                "remaining_calls": selected.get("remaining_calls"),
                "expected_calls": selected.get("expected_calls") or (
                    temporal_calls if route.get("key") == "remote_vision" else 1
                ),
                "artifact_count": int(selected.get("artifact_count") or 0),
                "artifact_coverage_ready": bool(selected.get("artifact_coverage_ready")),
                "blockers": selected.get("blockers")
                or (["consent_missing"] if not candidates else []),
            }
        )
    return rows

def _next_actions(
    status: str,
    settings_ui_url: str,
    configuration_warnings: list[dict[str, Any]] | None = None,
    *,
    online_only: bool = False,
    gateway_healthy: bool = False,
) -> list[str]:
    actions = []
    actions.extend(
        str(row.get("message") or "")
        for row in configuration_warnings or []
        if str(row.get("message") or "").strip()
    )
    if status == "configuration_required":
        actions.extend(
            [
                (f"Configure the eight required online Proxy task routes in {settings_ui_url}; saving does not authorize export." if online_only else f"Configure the six required local/remote Proxy routes in {settings_ui_url}; saving does not authorize export."),
                "Register the configured LiteLLM loopback port to VKP LiteLLM Proxy only after explicit operator approval.",
            ]
        )
    if status == "operator_start_required" or (not gateway_healthy and status in {"operator_consent_required", "operator_smoke_required"}):
        actions.append(
            "Review `video-knowledge-model-gateway start` preview, then separately approve `start --execute`."
        )
    if status in {"operator_consent_required", "operator_smoke_required"}:
        actions.append(
            "Create route-revision-locked consent files for remote text, ASR, and OCR, plus one temporal vision consent that exactly covers all fixed groups and reserves one call per group."
        )
    if status == "operator_smoke_required":
        actions.append(
            "Run approved A/B/C smokes and save each runtime result at the declared lane path."
        )
    if status == "ready_for_final_review":
        actions.append(
            "Run the offline A/B/C comparator and inspect Bundle, Timeline, quality, latency, calls, cost, and recovery gates."
        )
    return actions


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# VKP Model Gateway Smoke Readiness",
        "",
        f"- Status: `{result['status']}`",
        f"- Routes: `{result['route_ready_count']}/{result['route_required_count']}`",
        f"- Remote consents: `{result['consent_ready_count']}/{result['consent_required_count']}`",
        f"- Temporal groups: `{result['temporal_sample']['ready_group_count']}/{result['temporal_sample']['group_count']}`",
        f"- Gateway live: `{result['gateway']['live']}`",
        f"- Gateway healthy: `{result['gateway']['healthy']}`",
        f"- Port recorded and owned: `{result['gateway']['port_recorded_and_owned']}`",
        f"- Configuration warnings: `{result['configuration_warning_count']}`",
        "",
        "| Requirement | Task | Location | Backend | Route | Allowlist | Ready |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["route_requirements"]:
        lines.append(
            f"| `{row['key']}` | `{row['task']}` | `{row['execution_location']}` | `{row['adapter_backend']}` | "
            f"`{row['route_id'] or 'missing'}` | `{row['allowlist_status']}` | `{row['ready']}` |"
        )
    if result["configuration_warnings"]:
        lines.extend(["", "## Configuration warnings", ""])
        lines.extend(
            f"- `{row['key']}`: {row['message']} Origins: `{', '.join(row['origins'])}`"
            for row in result["configuration_warnings"]
        )
    lines.extend(
        [
            "",
            "| Consent task | Calls | Artifacts | Exact coverage | Valid | Blockers |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in result["remote_consents"]:
        lines.append(
            f"| `{row['task']}` | {row['expected_calls']} | {row['artifact_count']} | "
            f"`{row['artifact_coverage_ready']}` | `{row['valid']}` | "
            f"`{', '.join(row['blockers']) or 'none'}` |"
        )
    lines.extend(["", "## Next actions", ""])
    lines.extend(f"- {action}" for action in result["next_actions"])
    lines.extend(
        [
            "",
            "> Readiness only: no gateway start, model call, upload, consent creation, or port-registry write was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit operator readiness for VKP hybrid model gateway smokes"
    )
    parser.add_argument("bundle_dir")
    parser.add_argument("--indexes", default="6,80,112,135,199,201")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--consent", action="append", default=[])
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--secrets-path", default="")
    parser.add_argument("--gateway-config-path", default="")
    parser.add_argument("--port-record-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--online-only", action="store_true")
    args = parser.parse_args(argv)
    indexes = [int(value.strip()) for value in str(args.indexes).split(",") if value.strip()]
    result = model_gateway_smoke_readiness(
        args.bundle_dir,
        indexes=indexes,
        frame_count=args.frame_count,
        consent_paths=args.consent,
        settings_path=args.settings_path or None,
        secrets_path=args.secrets_path or None,
        gateway_config_path=args.gateway_config_path or None,
        port_record_path=args.port_record_path or None,
        output_dir=args.output_dir or None,
        online_only=bool(args.online_only),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready_for_final_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
