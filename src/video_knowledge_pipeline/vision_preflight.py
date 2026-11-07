from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import resolve_vision_execution_profile
from .storage import read_json, read_jsonl, write_json
from .vision_api import provider_requires_api_key
from .vision_api import provider_runtime_diagnostics
from .vision_api import resolve_provider_config
from .vision_api import test_vision_provider
from .vision_gateway_profile import resolve_route_based_vision_gateway_profile
from .vision_gateway_readiness import route_based_gateway_provider_test


def vision_execution_preflight(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    semantic_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    include_semantic: bool = True,
    include_temporal: bool = True,
    semantic_indexes: list[int] | None = None,
    temporal_indexes: list[int] | None = None,
    check_provider: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    timeline = [item for item in timeline_data if isinstance(item, dict)] if isinstance(timeline_data, list) else []
    active_tasks = (["semantic_frame"] if include_semantic else []) + (["temporal_sequence"] if include_temporal else [])
    route_profiles = {task: resolve_route_based_vision_gateway_profile(task) for task in active_tasks}
    matrix_recommended_config = _matrix_recommended_provider_config(root, manifest) if provider_config is None and isinstance(manifest, dict) else {}
    matrix_runtime_config = _provider_runtime_config_from_recommendation(matrix_recommended_config)
    use_route_gateway = provider_config is None and not matrix_runtime_config
    task_configs = {task: dict(route_profile.get("provider_config") or {}) for task, route_profile in route_profiles.items() if use_route_gateway and route_profile.get("route_configured")}
    missing_gateway_routes = [task for task, route_profile in route_profiles.items() if use_route_gateway and not route_profile.get("route_configured")]
    primary_task = "temporal_sequence" if "temporal_sequence" in task_configs else ("semantic_frame" if "semantic_frame" in task_configs else (active_tasks[0] if active_tasks else "temporal_sequence"))
    if use_route_gateway and task_configs:
        runtime_config = task_configs[primary_task]
        provider_config_source = "route_based_gateway"
    else:
        runtime_config = provider_config or matrix_runtime_config or None
        provider_config_source = "explicit" if provider_config is not None else ("vision_provider_matrix" if matrix_runtime_config else "default_profile_legacy")
    profile = resolve_vision_execution_profile(
        provider_config=runtime_config,
        multimodal_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
    )
    cfg = resolve_provider_config(profile["provider_config"])
    if not task_configs:
        task_configs = {task: cfg for task in active_tasks}
    semantic_limit_value = int(profile["multimodal_limit"])
    temporal_limit_value = int(profile["temporal_limit"])
    frame_count_value = int(profile["frame_count"])
    semantic_candidates = _semantic_candidates(timeline, explicit_indexes=semantic_indexes) if include_semantic else []
    temporal_candidates = _temporal_candidates(timeline) if include_temporal else []
    temporal_missing_groups = _temporal_missing_groups(timeline) if include_temporal else []
    semantic_candidates_filtered = _filter_indexes(semantic_candidates, semantic_indexes)
    temporal_candidates_filtered = _filter_indexes(temporal_candidates, temporal_indexes)
    semantic_selected = _select(semantic_candidates, semantic_limit_value) if include_semantic else []
    temporal_selected = _select(temporal_candidates_filtered, temporal_limit_value) if include_temporal else []
    if include_semantic:
        semantic_selected = _select(semantic_candidates_filtered, semantic_limit_value)
    restore_ready = _restore_chain_status(root)
    active_gateway_config = dict(task_configs.get(primary_task) or {})
    provider_diagnostics = provider_runtime_diagnostics(cfg)
    provider_health = (
        route_based_gateway_provider_test(active_gateway_config, task=primary_task)
        if str(active_gateway_config.get("adapter_backend") or "").strip().lower() == "proxy"
        else test_vision_provider(
            cfg,
            image_paths=_provider_smoke_image_paths(
                root,
                timeline,
                semantic_selected=semantic_selected,
                temporal_selected=temporal_selected,
            ),
        )
        if check_provider
        else {
            "schema": "lecture_vision_provider_test.v1",
            "status": "not_checked",
            "safe_to_execute": None,
            "error_class": "",
            "error_summary": "",
            "secrets_redacted": True,
        }
    )
    provider_public = dict(provider_health.get("provider") or {}) or {"provider": cfg.get("provider"), "base_url": provider_diagnostics.get("base_url") or cfg.get("base_url"), "model": cfg.get("model"), "api_key_configured": bool(cfg.get("api_key")), "timeout_seconds": cfg.get("timeout_seconds")}
    provider_public["config_source"] = provider_config_source
    provider_public["effective"] = True
    expected_calls = len(semantic_selected) + len(temporal_selected)
    confirmation = {
        "confirm_vision_calls": expected_calls,
        "confirm_vision_indexes": _index_confirmation(semantic_selected, temporal_selected),
        "semantic_confirm_vision_calls": len(semantic_selected),
        "semantic_confirm_vision_indexes": _index_confirmation(semantic_selected, []),
        "temporal_confirm_vision_calls": len(temporal_selected),
        "temporal_confirm_vision_indexes": _index_confirmation([], temporal_selected),
    }
    blockers = _blockers(
        cfg=cfg,
        expected_calls=expected_calls,
        semantic_selected=semantic_selected,
        temporal_selected=temporal_selected,
        temporal_missing_groups=temporal_missing_groups,
        restore_ready=restore_ready,
        provider_health=provider_health,
    )
    for task in missing_gateway_routes:
        blockers.append({"key": "gateway_route_missing", "message": f"No route-based gateway profile is configured for {task}; legacy fallback is blocked."})
    ready_for_confirmed_execution = not blockers
    provider_health_verified = provider_health.get("safe_to_execute") is True
    route_profile_rows = []
    for item in route_profiles.values():
        row = {key: value for key, value in item.items() if key != "provider_config"}
        row["usage"] = (
            "active"
            if use_route_gateway
            else "unused_explicit_or_matrix_provider_selected"
        )
        route_profile_rows.append(row)
    # This is intentionally independent from API/provider readiness. Import is
    # lazy to avoid a module cycle with the existing analyzers reused by the
    # interactive session adapter.
    from .codex_vision_session import codex_session_preflight_summary

    interactive_session = codex_session_preflight_summary(
        root,
        timeline,
        semantic_selected=semantic_selected,
        temporal_selected=temporal_selected,
        temporal_missing_groups=temporal_missing_groups,
    )
    preflight = {
        "schema": "lecture_vision_execution_preflight.v1",
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "provider": provider_public,
        "effective_provider": provider_public,
        "recommended_provider_config": matrix_recommended_config,
        "route_based_gateway_profiles": route_profile_rows,
        "remote_fallback_disabled": bool(provider_config is not None),
        "provider_diagnostics": provider_diagnostics,
        "execution_profile": {
            "provider_config_source": provider_config_source,
            "semantic_limit": semantic_limit_value,
            "temporal_limit": temporal_limit_value,
            "frame_count": frame_count_value,
            "include_semantic": bool(include_semantic),
            "include_temporal": bool(include_temporal),
            "semantic_indexes": [int(index) for index in semantic_indexes or []],
            "temporal_indexes": [int(index) for index in temporal_indexes or []],
            "check_provider": bool(check_provider),
        },
        "candidate_counts": {
            "semantic_available": len(semantic_candidates),
            "semantic_after_index_filter": len(semantic_candidates_filtered),
            "semantic_selected": len(semantic_selected),
            "temporal_available": len(temporal_candidates),
            "temporal_after_index_filter": len(temporal_candidates_filtered),
            "temporal_selected": len(temporal_selected),
            "temporal_without_frame_groups": len(temporal_missing_groups),
        },
        "selected_indexes": {
            "semantic": semantic_selected,
            "temporal": temporal_selected,
            "temporal_missing_groups_sample": temporal_missing_groups[:10],
        },
        "expected_api_calls": expected_calls,
        "confirmation": confirmation,
        "writes": {
            "semantic_fields": ["visual_understanding", "visual_understanding_updated_at", "quality_issues", "integrated_visual"],
            "temporal_fields": ["temporal_visual_understanding", "temporal_visual_understanding_updated_at", "quality_issues", "integrated_visual"],
            "audit_artifacts": ["vision-analysis-runs.jsonl", "vision-analysis-runs.md"],
            "restore_artifacts": ["vision-restore-plan.json", "vision-restore-plan.md", "vision-restore-runs.jsonl", "vision-restore-runs.md"],
        },
        "restore_chain": restore_ready,
        "provider_health": provider_health,
        "provider_health_required": bool(check_provider),
        "provider_health_verified": provider_health_verified,
        "interactive_session": interactive_session,
        "ready_for_confirmed_execution": ready_for_confirmed_execution,
        "ready_to_execute": ready_for_confirmed_execution,
        "readiness_scope": (
            "configuration_confirmation_and_provider_health"
            if check_provider
            else "configuration_and_confirmation_only_provider_health_not_checked"
        ),
        "blockers": blockers,
        "commands": _commands(
            root,
            semantic_limit=semantic_limit_value,
            temporal_limit=temporal_limit_value,
            frame_count=frame_count_value,
            include_semantic=include_semantic,
            include_temporal=include_temporal,
            semantic_indexes=[int(index) for index in semantic_indexes or []],
            temporal_indexes=[int(index) for index in temporal_indexes or []],
            semantic_selected=semantic_selected,
            temporal_selected=temporal_selected,
            confirmation=confirmation,
        ),
        "confirmed_mcp_args": {},
    }
    if write:
        json_path = root / "vision-execution-preflight.json"
        markdown_path = root / "vision-execution-preflight.md"
        preflight["preflight_path"] = str(markdown_path)
        preflight["preflight_json_path"] = str(json_path)
        write_json(json_path, preflight)
        markdown_path.write_text(render_vision_execution_preflight_markdown(preflight), encoding="utf-8")
        if isinstance(manifest, dict):
            manifest["vision_execution_preflight"] = "vision-execution-preflight.md"
            manifest["vision_execution_preflight_json"] = "vision-execution-preflight.json"
            manifest["mcp_vision_execution_preflight_args"] = "mcp-vision-execution-preflight.args.json"
            for stale_key in (
                "mcp_bundle_advance_confirmed_args",
                "mcp_multimodal_frame_analysis_confirmed_args",
                "mcp_temporal_visual_analysis_confirmed_args",
            ):
                manifest.pop(stale_key, None)
            write_json(
                root / "mcp-vision-execution-preflight.args.json",
                {
                    "bundle_dir": str(root),
                    "semantic_limit": semantic_limit_value,
                    "temporal_limit": temporal_limit_value,
                    "frame_count": frame_count_value,
                    "include_semantic": bool(include_semantic),
                    "include_temporal": bool(include_temporal),
                    "semantic_indexes": [int(index) for index in semantic_indexes or []],
                    "temporal_indexes": [int(index) for index in temporal_indexes or []],
                    "check_provider": bool(check_provider),
                    "write": True,
                },
            )
            confirmed_args = _confirmed_mcp_args(
                root,
                cfg=cfg,
                semantic_limit=semantic_limit_value,
                temporal_limit=temporal_limit_value,
                frame_count=frame_count_value,
                task_configs=task_configs,
                semantic_selected=semantic_selected,
                temporal_selected=temporal_selected,
                confirmation=confirmation,
            )
            confirmed_summary: dict[str, dict[str, str]] = {}
            for key, value in confirmed_args.items():
                manifest[key] = value["path"]
                write_json(root / value["path"], value["args"])
                confirmed_summary[key] = {
                    "path": str(root / value["path"]),
                    "mcp_tool": _confirmed_mcp_tool(key),
                    "command": _mcp_call_command(root / value["path"], _confirmed_mcp_tool(key)),
                }
            preflight["confirmed_mcp_args"] = confirmed_summary
            write_json(manifest_path, manifest)
            write_json(json_path, preflight)
            markdown_path.write_text(render_vision_execution_preflight_markdown(preflight), encoding="utf-8")
    return preflight


def render_vision_execution_preflight_markdown(preflight: dict[str, Any]) -> str:
    provider = preflight.get("provider") if isinstance(preflight.get("provider"), dict) else {}
    recommended = preflight.get("recommended_provider_config") if isinstance(preflight.get("recommended_provider_config"), dict) else {}
    execution_profile = preflight.get("execution_profile") if isinstance(preflight.get("execution_profile"), dict) else {}
    counts = preflight.get("candidate_counts") if isinstance(preflight.get("candidate_counts"), dict) else {}
    selected = preflight.get("selected_indexes") if isinstance(preflight.get("selected_indexes"), dict) else {}
    writes = preflight.get("writes") if isinstance(preflight.get("writes"), dict) else {}
    restore = preflight.get("restore_chain") if isinstance(preflight.get("restore_chain"), dict) else {}
    interactive = preflight.get("interactive_session") if isinstance(preflight.get("interactive_session"), dict) else {}
    provider_health = preflight.get("provider_health") if isinstance(preflight.get("provider_health"), dict) else {}
    provider_diagnostics = preflight.get("provider_diagnostics") if isinstance(preflight.get("provider_diagnostics"), dict) else {}
    proxy_env = provider_diagnostics.get("proxy_env") if isinstance(provider_diagnostics.get("proxy_env"), dict) else {}
    lines = [
        "# Vision Execution Preflight",
        "",
        f"- Bundle: `{preflight.get('bundle_dir', '')}`",
        f"- Provider: `{provider.get('provider', '')}` / `{provider.get('model', '')}`",
        f"- Provider config source: `{execution_profile.get('provider_config_source', '')}`",
        f"- Effective provider: `{provider.get('provider', '')}` / `{provider.get('base_url', '')}`",
        f"- Remote fallback disabled: `{preflight.get('remote_fallback_disabled', False)}`",
        f"- Matrix recommended config: `{recommended}`",
        f"- API key configured: `{provider.get('api_key_configured', False)}`",
        f"- Route ID: `{provider.get('route_id', '')}` / revision: `{provider.get('route_revision', '')}`",
        f"- Gateway configured: `{provider.get('gateway_configured', False)}`",
        f"- Gateway ready: `{provider.get('gateway_ready', False)}`",
        f"- DPAPI credential ready: `{provider.get('credential_ready', False)}` / `{provider.get('credential_status', '')}`",
        f"- Provider health: `{provider_health.get('status', 'not_checked')}`",
        f"- Provider health required: `{preflight.get('provider_health_required', False)}`",
        f"- Provider health verified: `{preflight.get('provider_health_verified', False)}`",
        f"- Provider safe to execute: `{provider_health.get('safe_to_execute', None)}`",
        f"- Provider error class: `{provider_health.get('error_class', '')}`",
        f"- Endpoint kind: `{provider_diagnostics.get('endpoint_kind', '')}`",
        f"- Request URL: `{provider_diagnostics.get('request_url', '')}`",
        f"- Ready for confirmed execution: `{preflight.get('ready_for_confirmed_execution', False)}`",
        f"- Readiness scope: `{preflight.get('readiness_scope', '')}`",
        f"- Expected API calls: `{preflight.get('expected_api_calls', 0)}`",
        f"- Confirm calls: `{(preflight.get('confirmation') or {}).get('confirm_vision_calls', 0)}`",
        f"- Confirm indexes: `{(preflight.get('confirmation') or {}).get('confirm_vision_indexes', '')}`",
        f"- Semantic confirm: `{(preflight.get('confirmation') or {}).get('semantic_confirm_vision_calls', 0)}` / `{(preflight.get('confirmation') or {}).get('semantic_confirm_vision_indexes', '')}`",
        f"- Temporal confirm: `{(preflight.get('confirmation') or {}).get('temporal_confirm_vision_calls', 0)}` / `{(preflight.get('confirmation') or {}).get('temporal_confirm_vision_indexes', '')}`",
        f"- Codex interactive ready to prepare: `{interactive.get('ready_to_prepare', False)}`",
        f"- Codex API key required: `{interactive.get('api_key_required', False)}`",
        f"- Codex provider health check: `{interactive.get('provider_health_check', 'not_applicable')}`",
        "",
        "## Candidates",
        "",
        f"- Semantic available/selected: `{counts.get('semantic_available', 0)}` / `{counts.get('semantic_selected', 0)}`",
        f"- Temporal available/selected: `{counts.get('temporal_available', 0)}` / `{counts.get('temporal_selected', 0)}`",
        f"- Temporal without frame groups: `{counts.get('temporal_without_frame_groups', 0)}`",
        f"- Semantic indexes: `{selected.get('semantic', [])}`",
        f"- Temporal indexes: `{selected.get('temporal', [])}`",
        f"- Codex semantic/temporal ready: `{interactive.get('semantic_ready', 0)}` / `{interactive.get('temporal_ready', 0)}`",
        "",
        "## Writes",
        "",
        f"- Semantic fields: `{writes.get('semantic_fields', [])}`",
        f"- Temporal fields: `{writes.get('temporal_fields', [])}`",
        f"- Audit artifacts: `{writes.get('audit_artifacts', [])}`",
        f"- Restore artifacts: `{writes.get('restore_artifacts', [])}`",
        "",
        "## Restore Chain",
        "",
        f"- Diff audit available: `{restore.get('diff_audit_available', False)}`",
        f"- Restore plan available: `{restore.get('restore_plan_available', False)}`",
        f"- Restore apply audit available: `{restore.get('restore_apply_audit_available', False)}`",
        "",
        "## Provider Health",
        "",
        f"- Status: `{provider_health.get('status', 'not_checked')}`",
        f"- Safe to execute: `{provider_health.get('safe_to_execute', None)}`",
        f"- Error class: `{provider_health.get('error_class', '')}`",
        f"- Error summary: `{provider_health.get('error_summary', '')}`",
        f"- Request URL: `{provider_diagnostics.get('request_url', '')}`",
        f"- Proxy env: `HTTP={proxy_env.get('HTTP_PROXY', False)}` / `HTTPS={proxy_env.get('HTTPS_PROXY', False)}` / `ALL={proxy_env.get('ALL_PROXY', False)}` / `NO_PROXY={proxy_env.get('NO_PROXY', False)}`",
        f"- Secrets redacted: `{provider_health.get('secrets_redacted', True)}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else []
    lines.extend([f"- `{item.get('key', '')}`: {item.get('message', '')}" for item in blockers if isinstance(item, dict)] or ["- None"])
    lines.extend(["", "## Commands", ""])
    commands = preflight.get("commands") if isinstance(preflight.get("commands"), dict) else {}
    for key, command in commands.items():
        lines.extend([f"### {key}", "", "```powershell", str(command), "```", ""])
    confirmed = preflight.get("confirmed_mcp_args") if isinstance(preflight.get("confirmed_mcp_args"), dict) else {}
    if confirmed:
        lines.extend(["", "## Confirmed MCP Args", ""])
        for key, row in confirmed.items():
            if not isinstance(row, dict):
                continue
            lines.extend(
                [
                    f"### {key}",
                    "",
                    f"- Tool: `{row.get('mcp_tool', '')}`",
                    f"- Args: `{row.get('path', '')}`",
                    "",
                    "```powershell",
                    str(row.get("command") or ""),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _matrix_recommended_provider_config(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    matrix_rel = str(manifest.get("vision_provider_matrix_json") or "vision-provider-matrix.json").strip()
    if not matrix_rel:
        return {}
    matrix_path = Path(matrix_rel)
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    if not matrix_path.exists():
        return {}
    matrix = read_json(matrix_path)
    if not isinstance(matrix, dict) or str(matrix.get("status") or "") != "ok":
        return {}
    recommended = matrix.get("recommended_provider_config")
    if not isinstance(recommended, dict):
        return {}
    provider = str(recommended.get("provider") or matrix.get("recommended_provider") or "").strip()
    if not provider:
        return {}
    allowed: dict[str, Any] = {"provider": provider}
    for key in ("base_url", "model", "timeout_seconds", "image_probe_max_edge", "image_probe_jpeg_quality"):
        value = recommended.get(key)
        if value in (None, "", [], {}):
            continue
        allowed[key] = int(value) if key in {"timeout_seconds", "image_probe_max_edge", "image_probe_jpeg_quality"} else value
    return allowed


def _provider_runtime_config_from_recommendation(recommended: dict[str, Any]) -> dict[str, Any]:
    if not recommended:
        return {}
    runtime: dict[str, Any] = {}
    for key in ("provider", "model", "timeout_seconds"):
        value = recommended.get(key)
        if value not in (None, "", [], {}):
            runtime[key] = value
    base_url = str(recommended.get("base_url") or "").strip()
    if base_url and "<redacted>" not in base_url:
        runtime["base_url"] = base_url
    return runtime


def _semantic_candidates(
    timeline: list[dict[str, Any]],
    *,
    explicit_indexes: list[int] | None = None,
) -> list[int]:
    explicitly_requested = {int(value) for value in explicit_indexes or [] if int(value) > 0}
    return [
        index
        for index, item in enumerate(timeline, start=1)
        if (
            str(item.get("visual_route") or "") in {"semantic_frame", "mixed"}
            or index in explicitly_requested
        )
        and _has_frames(item)
        and not _has_valid_understanding(item.get("visual_understanding"))
    ]


def _temporal_candidates(timeline: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, item in enumerate(timeline, start=1)
        if str(item.get("visual_route") or "") in {"temporal_sequence", "mixed"}
        and _has_list(item.get("temporal_frame_paths"))
        and not _has_valid_understanding(item.get("temporal_visual_understanding"))
    ]


def _temporal_missing_groups(timeline: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, item in enumerate(timeline, start=1)
        if str(item.get("visual_route") or "") in {"temporal_sequence", "mixed"} and not _has_list(item.get("temporal_frame_paths"))
    ]


def _select(values: list[int], limit: int) -> list[int]:
    if limit <= 0:
        return values
    return values[:limit]


def _filter_indexes(values: list[int], indexes: list[int] | None) -> list[int]:
    wanted = {int(index) for index in indexes or [] if int(index) > 0}
    if not wanted:
        return values
    return [value for value in values if int(value) in wanted]


def _restore_chain_status(root: Path) -> dict[str, Any]:
    run_rows = read_jsonl(root / "vision-analysis-runs.jsonl") if (root / "vision-analysis-runs.jsonl").exists() else []
    restore_rows = read_jsonl(root / "vision-restore-runs.jsonl") if (root / "vision-restore-runs.jsonl").exists() else []
    latest_diff_has_values = False
    for row in reversed(run_rows):
        for diff in row.get("timeline_diff") or []:
            for change in diff.get("changes") or []:
                if isinstance(change, dict) and "before_value" in change:
                    latest_diff_has_values = True
                    break
            if latest_diff_has_values:
                break
        if latest_diff_has_values:
            break
    return {
        "diff_audit_available": True,
        "runs_logged": len(run_rows),
        "latest_diff_has_structured_values": latest_diff_has_values,
        "restore_plan_available": (root / "vision-restore-plan.json").exists(),
        "restore_apply_available": True,
        "restore_apply_audit_available": bool(restore_rows),
        "restore_apply_runs_logged": len(restore_rows),
    }


def _blockers(
    *,
    cfg: dict[str, Any],
    expected_calls: int,
    semantic_selected: list[int],
    temporal_selected: list[int],
    temporal_missing_groups: list[int],
    restore_ready: dict[str, Any],
    provider_health: dict[str, Any],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if provider_requires_api_key(cfg) and not cfg.get("api_key"):
        blockers.append({"key": "missing_api_key", "message": "Set the provider API key before execute=true."})
    if expected_calls <= 0:
        blockers.append({"key": "no_selected_candidates", "message": "No semantic or temporal candidates are selected for this execution profile."})
    if temporal_missing_groups and not temporal_selected:
        blockers.append({"key": "temporal_frame_groups_missing", "message": "Temporal candidates need temporal_frame_paths before temporal analysis can run."})
    if not restore_ready.get("diff_audit_available") or not restore_ready.get("restore_apply_available"):
        blockers.append({"key": "restore_chain_missing", "message": "Vision diff audit and restore apply support must be available before real execution."})
    if provider_health.get("safe_to_execute") is False:
        status = str(provider_health.get("status") or "provider_failed")
        error_class = str(provider_health.get("error_class") or status)
        blockers.append(
            {
                "key": "provider_health_failed",
                "message": f"Vision provider smoke test failed: {status} / {error_class}. Fix provider/network/model JSON output before execute=true.",
            }
        )
    return blockers


def _provider_smoke_image_paths(
    root: Path,
    timeline: list[dict[str, Any]],
    *,
    semantic_selected: list[int],
    temporal_selected: list[int],
) -> list[str]:
    paths: list[str] = []
    for index in semantic_selected:
        if 1 <= int(index) <= len(timeline):
            paths.extend(_item_frame_paths(root, timeline[int(index) - 1], temporal=False))
    for index in temporal_selected:
        if 1 <= int(index) <= len(timeline):
            paths.extend(_item_frame_paths(root, timeline[int(index) - 1], temporal=True))
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
        if len(result) >= 8:
            break
    return result


def _item_frame_paths(root: Path, item: dict[str, Any], *, temporal: bool) -> list[str]:
    keys = ("temporal_frame_paths", "frame_paths") if temporal else ("frame_paths",)
    paths: list[str] = []
    for key in keys:
        values = item.get(key) if isinstance(item.get(key), list) else []
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            path = Path(text)
            paths.append(str(path if path.is_absolute() else (root / path).resolve()))
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        text = str(asset.get("path") or asset.get("source") or "").strip()
        if not text:
            continue
        path = Path(text)
        paths.append(str(path if path.is_absolute() else (root / path).resolve()))
    return paths


def _commands(
    root: Path,
    *,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    include_semantic: bool,
    include_temporal: bool,
    semantic_indexes: list[int],
    temporal_indexes: list[int],
    semantic_selected: list[int],
    temporal_selected: list[int],
    confirmation: dict[str, Any],
) -> dict[str, str]:
    commands = {
        "test_provider": f".\\scripts\\video-knowledge.ps1 test-vision-provider --image-paths \"{root / 'assets' / '0001-001_0000000000ms.jpg'}\"",
        "inspect_runs": f".\\scripts\\video-knowledge.ps1 vision-analysis-run-log \"{root}\"",
        "plan_restore": f".\\scripts\\video-knowledge.ps1 vision-analysis-restore-plan \"{root}\" --run-id <run_id>",
    }
    if include_semantic and semantic_selected:
        index_arg = f" --indexes {','.join(str(index) for index in semantic_indexes)}" if semantic_indexes else ""
        selected_arg = f" --indexes {','.join(str(index) for index in semantic_selected)}"
        commands["run_semantic"] = f".\\scripts\\video-knowledge.ps1 run-multimodal-frame-analysis \"{root}\" --execute --limit {semantic_limit}{index_arg}"
        commands["confirmed_run_semantic"] = (
            f".\\scripts\\video-knowledge.ps1 run-multimodal-frame-analysis \"{root}\" --execute --limit {semantic_limit}{selected_arg}"
            f" --confirm-vision-calls {confirmation.get('semantic_confirm_vision_calls', 0)}"
            f" --confirm-vision-indexes \"{confirmation.get('semantic_confirm_vision_indexes', '')}\""
        )
    if include_temporal and temporal_selected:
        index_arg = f" --indexes {','.join(str(index) for index in temporal_indexes)}" if temporal_indexes else ""
        selected_arg = f" --indexes {','.join(str(index) for index in temporal_selected)}"
        commands["run_temporal"] = f".\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis \"{root}\" --execute --frame-count {frame_count} --limit {temporal_limit}{index_arg}"
        commands["confirmed_run_temporal"] = (
            f".\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis \"{root}\" --execute --frame-count {frame_count} --limit {temporal_limit}{selected_arg}"
            f" --confirm-vision-calls {confirmation.get('temporal_confirm_vision_calls', 0)}"
            f" --confirm-vision-indexes \"{confirmation.get('temporal_confirm_vision_indexes', '')}\""
        )
    return commands


def _confirmed_mcp_args(
    root: Path,
    *,
    cfg: dict[str, Any],
    task_configs: dict[str, dict[str, Any]] | None = None,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    semantic_selected: list[int],
    temporal_selected: list[int],
    confirmation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    task_configs = dict(task_configs or {})
    args: dict[str, dict[str, Any]] = {}
    semantic_config = task_configs.get("semantic_frame")
    if semantic_selected and semantic_config:
        args["mcp_multimodal_frame_analysis_confirmed_args"] = {
            "path": "mcp-run-multimodal-frame-analysis-confirmed.args.json",
            "args": {
                "bundle_dir": str(root),
                "execute": True,
                "provider_config": _sanitised_confirmed_provider_config(semantic_config),
                "limit": semantic_limit,
                "indexes": semantic_selected,
                "confirm_vision_calls": confirmation.get("semantic_confirm_vision_calls", 0),
                "confirm_vision_indexes": confirmation.get("semantic_confirm_vision_indexes", ""),
            },
        }
    temporal_config = task_configs.get("temporal_sequence")
    if temporal_selected and temporal_config:
        args["mcp_temporal_visual_analysis_confirmed_args"] = {
            "path": "mcp-run-temporal-visual-analysis-confirmed.args.json",
            "args": {
                "bundle_dir": str(root),
                "execute": True,
                "provider_config": _sanitised_confirmed_provider_config(temporal_config),
                "frame_count": frame_count,
                "limit": temporal_limit,
                "indexes": temporal_selected,
                "confirm_vision_calls": confirmation.get("temporal_confirm_vision_calls", 0),
                "confirm_vision_indexes": confirmation.get("temporal_confirm_vision_indexes", ""),
            },
        }
    return args


def _sanitised_confirmed_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    provider = str(cfg.get("provider") or "").strip()
    model = str(cfg.get("model") or "").strip()
    base_url = str(cfg.get("base_url") or "").strip()
    result: dict[str, Any] = {}
    if provider:
        result["provider"] = provider
    if model:
        result["model"] = model
    if base_url and not _url_has_secret_query(base_url):
        result["base_url"] = base_url
    if cfg.get("timeout_seconds"):
        result["timeout_seconds"] = int(cfg.get("timeout_seconds") or 60)
    for key in ("adapter_backend", "execution_location", "route_id", "route_revision", "virtual_model", "profile_id"):
        value = cfg.get(key)
        if value not in (None, "", [], {}):
            result[key] = value
    if str(cfg.get("adapter_backend") or "").strip().lower() == "proxy" and (not result.get("route_id") or not result.get("route_revision")):
        return {}
    return result


def _url_has_secret_query(value: str) -> bool:
    lowered = value.lower()
    secret_markers = ("api_key=", "apikey=", "key=", "token=", "access_token=", "authorization=", "bearer ")
    return any(marker in lowered for marker in secret_markers)


def _confirmed_mcp_tool(key: str) -> str:
    return {
        "mcp_multimodal_frame_analysis_confirmed_args": "run_multimodal_frame_analysis",
        "mcp_temporal_visual_analysis_confirmed_args": "run_temporal_visual_analysis",
    }.get(key, "")


def _mcp_call_command(args_path: Path, tool: str) -> str:
    return f".\\scripts\\video-knowledge.ps1 mcp-call {tool} '{args_path}'"


def _index_confirmation(semantic_selected: list[int], temporal_selected: list[int]) -> str:
    return ",".join(str(index) for index in [*semantic_selected, *temporal_selected])


def _has_mapping(value: object) -> bool:
    return isinstance(value, dict) and any(item not in (None, "", [], {}) for item in value.values())


def _has_valid_understanding(value: object) -> bool:
    if not _has_mapping(value):
        return False
    if not isinstance(value, dict):
        return False
    if value.get("parse_failed") is True:
        return False
    if str(value.get("validation_status") or "").strip().lower() == "incomplete":
        return False
    return True


def _has_list(value: object) -> bool:
    return isinstance(value, list) and any(item not in (None, "", [], {}) for item in value)


def _has_frames(item: dict[str, Any]) -> bool:
    return _has_list(item.get("frame_paths")) or _has_list(item.get("assets"))
