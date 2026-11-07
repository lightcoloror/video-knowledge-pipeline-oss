from __future__ import annotations

from pathlib import Path
from typing import Any

from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .vision_api import (
    provider_runtime_diagnostics,
    redact_url_secrets,
    resolve_provider_config,
    test_vision_provider,
)
from .vlm_preprocess import prepare_image_probe
from .vision_gateway_readiness import (
    configured_gateway_vision_profiles,
    route_based_gateway_provider_test,
)

SMOKE_SCHEMA = "lecture_vision_provider_smoke.v1"
MATRIX_SCHEMA = "lecture_vision_provider_matrix.v1"


def vision_provider_smoke(
    *,
    provider_config: dict[str, Any] | None = None,
    provider: str = "",
    model: str = "",
    base_url: str = "",
    timeout_seconds: int | None = None,
    bundle_dir: str | Path | None = None,
    single_image: str | Path | None = None,
    multi_image_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    image_probe_max_edge: int = 0,
    image_probe_jpeg_quality: int = 70,
    max_images: int = 8,
    write: bool = True,
) -> dict[str, Any]:
    cfg_input = dict(provider_config or {})
    if provider:
        cfg_input["provider"] = provider
    if model:
        cfg_input["model"] = model
    if base_url:
        cfg_input["base_url"] = base_url
    if timeout_seconds is not None:
        cfg_input["timeout_seconds"] = int(timeout_seconds)
    cfg = resolve_provider_config(cfg_input)
    bundle_root = Path(bundle_dir).expanduser().resolve() if bundle_dir else None
    out_root = Path(output_dir).expanduser().resolve() if output_dir else (bundle_root or Path.cwd())
    source_image_paths = _smoke_image_paths(bundle_root=bundle_root, single_image=single_image, multi_image_dir=multi_image_dir, max_images=max_images)
    image_probe = _prepare_image_probe(
        source_image_paths,
        output_dir=out_root / "vision-provider-smoke-probes",
        max_edge=image_probe_max_edge,
        jpeg_quality=image_probe_jpeg_quality,
    )
    image_paths = [str(path) for path in image_probe.get("image_paths") or source_image_paths if str(path)]
    diagnostics = provider_runtime_diagnostics(cfg)
    provider_test = (
        route_based_gateway_provider_test(
            cfg_input,
            task=str(cfg_input.get("task") or ""),
        )
        if str(cfg_input.get("adapter_backend") or "").strip().lower() == "proxy"
        else test_vision_provider(cfg, image_paths=image_paths)
    )
    report_path = out_root / "vision-provider-smoke.json"
    markdown_path = out_root / "vision-provider-smoke.md"
    args_path = out_root / "mcp-vision-provider-smoke.args.json"
    report = {
        "schema": SMOKE_SCHEMA,
        "checked_at": now_iso(),
        "bundle_dir": str(bundle_root) if bundle_root else "",
        "output_dir": str(out_root),
        "status": provider_test.get("status", "unknown"),
        "safe_to_execute": bool(provider_test.get("safe_to_execute")),
        "error_class": provider_test.get("error_class", ""),
        "error_summary": provider_test.get("error_summary", ""),
        "provider": provider_test.get("provider") if isinstance(provider_test.get("provider"), dict) else {},
        "diagnostics": diagnostics,
        "failure_diagnosis": provider_test.get("failure_diagnosis") if isinstance(provider_test.get("failure_diagnosis"), dict) else {},
        "image_selection": _image_selection_summary(image_paths, source_image_paths=source_image_paths, max_images=max_images),
        "image_probe": image_probe,
        "image_paths": image_paths,
        "checks": provider_test.get("checks") if isinstance(provider_test.get("checks"), list) else [],
        "recommended_provider_config": _recommended_provider_config(
            provider_test=provider_test,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
        ),
        "recovery_suggestion": _recovery_suggestion(provider_test),
        "secrets_redacted": True,
        "provider_test": provider_test,
        "report_path": str(report_path),
        "report_markdown_path": str(markdown_path),
        "mcp_args_path": str(args_path),
    }
    if write:
        out_root.mkdir(parents=True, exist_ok=True)
        if bundle_root:
            manifest_path = bundle_root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["vision_provider_smoke"] = _rel_to_bundle(bundle_root, markdown_path)
                manifest["vision_provider_smoke_json"] = _rel_to_bundle(bundle_root, report_path)
                manifest["mcp_vision_provider_smoke_args"] = _rel_to_bundle(bundle_root, args_path)
                write_json(manifest_path, manifest)
        write_json(report_path, report)
        markdown_path.write_text(render_vision_provider_smoke_markdown(report), encoding="utf-8")
        write_json(
            args_path,
            {
                "provider_config": _public_provider_args(cfg),
                "provider": "",
                "model": "",
                "base_url": "",
                "timeout_seconds": int(cfg.get("timeout_seconds") or 60),
                "bundle_dir": str(bundle_root) if bundle_root else "",
                "single_image": str(single_image or ""),
                "multi_image_dir": str(multi_image_dir or ""),
                "output_dir": str(out_root),
                "image_probe_max_edge": int(image_probe_max_edge or 0),
                "image_probe_jpeg_quality": int(image_probe_jpeg_quality or 70),
                "max_images": int(max_images or 8),
                "write": True,
            },
        )
        if bundle_root:
            register_bundle_run(
                bundle_root,
                run_type="vision_provider_smoke",
                run_id="vision-provider-smoke",
                status="completed" if report.get("safe_to_execute") else "needs_retry",
                title="Vision provider smoke",
                summary=f"Provider {report.get('status', 'unknown')} / safe_to_execute={bool(report.get('safe_to_execute'))}.",
                artifacts=[
                    {"key": "json", "path": report_path},
                    {"key": "markdown", "path": markdown_path},
                    {"key": "mcp_args", "path": args_path},
                ],
                failed_items=[] if report.get("safe_to_execute") else [{"reason": report.get("error_class") or report.get("status") or "provider_not_ready", "detail": report.get("error_summary") or report.get("recovery_suggestion") or "provider smoke not ready"}],
                retry_command=f".\\scripts\\video-knowledge.ps1 vision-provider-smoke --bundle-dir {bundle_root}",
                operator_boundary={
                    "read_only_report": True,
                    "does_not_modify_timeline": True,
                    "secrets_redacted": True,
                },
                write=True,
            )
    return report


def vision_provider_matrix(
    *,
    providers: list[str] | None = None,
    bundle_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    timeout_seconds: int | None = None,
    single_image: str | Path | None = None,
    multi_image_dir: str | Path | None = None,
    image_probe_max_edge: int = 0,
    image_probe_jpeg_quality: int = 70,
    max_images: int = 8,
    preferred_provider: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Run secret-safe smoke diagnostics across candidate providers."""
    provider_names = [str(value).strip() for value in providers or [] if str(value).strip()]
    if not provider_names:
        provider_names = ["local_qwen_vl", "volcengine_coding_plan", "gemini", "openai", "agnes"]
    bundle_root = Path(bundle_dir).expanduser().resolve() if bundle_dir else None
    gateway_candidates: list[tuple[str, dict[str, Any]]] = []
    if providers is None:
        for route_profile in configured_gateway_vision_profiles():
            if not route_profile.get("route_configured"):
                continue
            route_config = dict(route_profile.get("provider_config") or {})
            route_config["task"] = str(route_profile.get("task") or "")
            route_config["provider_config_source"] = "route_based_gateway"
            gateway_candidates.append((f"gateway:{route_config['task']}", route_config))
    candidate_configs = list(gateway_candidates)
    candidate_configs.extend((f"legacy:{name}", {"provider": name, "provider_config_source": "legacy_provider"}) for name in provider_names)

    out_root = Path(output_dir).expanduser().resolve() if output_dir else (bundle_root or Path.cwd())
    results: list[dict[str, Any]] = []
    for candidate_id, cfg in candidate_configs:
        provider_name = str(cfg.get("provider") or "")
        if timeout_seconds is not None:
            cfg["timeout_seconds"] = int(timeout_seconds)
        smoke = vision_provider_smoke(
            provider_config=cfg,
            bundle_dir=bundle_root,
            output_dir=out_root,
            single_image=single_image,
            multi_image_dir=multi_image_dir,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            max_images=max_images,
            write=False,
        )
        results.append(
            {
                "provider": smoke.get("provider", {}),
                "candidate_id": candidate_id,
                "provider_config_source": str(cfg.get("provider_config_source") or "legacy_provider"),
                "status": smoke.get("status", "unknown"),
                "safe_to_execute": bool(smoke.get("safe_to_execute")),
                "error_class": smoke.get("error_class", ""),
                "error_summary": smoke.get("error_summary", ""),
                "diagnostics": smoke.get("diagnostics", {}),
                "failure_diagnosis": smoke.get("failure_diagnosis", {}),
                "image_selection": smoke.get("image_selection", {}),
                "image_probe": smoke.get("image_probe", {}),
                "checks": smoke.get("checks", []),
                "recommended_provider_config": smoke.get("recommended_provider_config", {}),
                "recovery_suggestion": smoke.get("recovery_suggestion", ""),
            }
        )
    ranking = rank_vision_providers(results, preferred_provider=preferred_provider)
    gateway_rows = [row for row in ranking if row.get("provider_config_source") == "route_based_gateway"]
    eligible_rows = gateway_rows if gateway_rows else ranking
    recommended_row = next((row for row in eligible_rows if row.get("ready")), {})
    # A configured route never silently falls back to a legacy provider row.
    # The legacy rows remain diagnostics only until the operator selects one.
    recommended = str(recommended_row.get("provider") or "")
    recommended_config = dict(recommended_row.get("recommended_provider_config") or {})
    report_path = out_root / "vision-provider-matrix.json"
    markdown_path = out_root / "vision-provider-matrix.md"
    args_path = out_root / "mcp-vision-provider-matrix.args.json"
    report = {
        "schema": MATRIX_SCHEMA,
        "checked_at": now_iso(),
        "bundle_dir": str(bundle_root) if bundle_root else "",
        "output_dir": str(out_root),
        "providers_requested": provider_names,
        "status": "ok" if recommended else "no_provider_ready",
        "recommended_provider": recommended,
        "configured_gateway_routes": [{key: value for key, value in profile.items() if key != "provider_config"} for profile in configured_gateway_vision_profiles()],
        "recommended_provider_config": recommended_config,
        "provider_ranking": ranking,
        "results": results,
        "secrets_redacted": True,
        "report_path": str(report_path),
        "report_markdown_path": str(markdown_path),
        "mcp_args_path": str(args_path),
    }
    if write:
        out_root.mkdir(parents=True, exist_ok=True)
        if bundle_root:
            manifest_path = bundle_root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["vision_provider_matrix"] = _rel_to_bundle(bundle_root, markdown_path)
                manifest["vision_provider_matrix_json"] = _rel_to_bundle(bundle_root, report_path)
                manifest["mcp_vision_provider_matrix_args"] = _rel_to_bundle(bundle_root, args_path)
                write_json(manifest_path, manifest)
        write_json(report_path, report)
        markdown_path.write_text(render_vision_provider_matrix_markdown(report), encoding="utf-8")
        write_json(
            args_path,
            {
                "providers": provider_names,
                "bundle_dir": str(bundle_root) if bundle_root else "",
                "output_dir": str(out_root),
                "timeout_seconds": int(timeout_seconds or 0),
                "single_image": str(single_image or ""),
                "multi_image_dir": str(multi_image_dir or ""),
                "image_probe_max_edge": int(image_probe_max_edge or 0),
                "image_probe_jpeg_quality": int(image_probe_jpeg_quality or 70),
                "max_images": int(max_images or 8),
                "preferred_provider": str(preferred_provider or ""),
                "write": True,
            },
        )
        if bundle_root:
            register_bundle_run(
                bundle_root,
                run_type="vision_provider_matrix",
                run_id="vision-provider-matrix",
                status="completed" if report.get("recommended_provider") else "needs_retry",
                title="Vision provider matrix",
                summary=f"Recommended provider: {report.get('recommended_provider') or 'none'}.",
                artifacts=[
                    {"key": "json", "path": report_path},
                    {"key": "markdown", "path": markdown_path},
                    {"key": "mcp_args", "path": args_path},
                ],
                failed_items=[] if report.get("recommended_provider") else [{"reason": "no_provider_ready", "detail": "No provider passed the matrix smoke checks."}],
                retry_command=f".\\scripts\\video-knowledge.ps1 vision-provider-matrix --providers \"{(',').join(provider_names)}\" --bundle-dir {bundle_root}",
                operator_boundary={
                    "read_only_report": True,
                    "does_not_modify_timeline": True,
                    "secrets_redacted": True,
                },
                write=True,
            )
    return report


def render_vision_provider_smoke_markdown(report: dict[str, Any]) -> str:
    provider = report.get("provider") if isinstance(report.get("provider"), dict) else {}
    diagnostics = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}
    proxy_env = diagnostics.get("proxy_env") if isinstance(diagnostics.get("proxy_env"), dict) else {}
    image_selection = report.get("image_selection") if isinstance(report.get("image_selection"), dict) else {}
    failure_diagnosis = report.get("failure_diagnosis") if isinstance(report.get("failure_diagnosis"), dict) else {}
    image_probe = report.get("image_probe") if isinstance(report.get("image_probe"), dict) else {}
    lines = [
        "---",
        "type: lecture-vision-provider-smoke",
        f'created: "{report.get("checked_at", now_iso())}"',
        "---",
        "",
        "# Vision Provider Smoke",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Safe to execute: `{report.get('safe_to_execute', False)}`",
        f"- Error class: `{report.get('error_class', '')}`",
        f"- Recommended provider config: `{report.get('recommended_provider_config', {})}`",
        f"- Provider: `{provider.get('provider', '')}`",
        f"- Model: `{provider.get('model', '')}`",
        f"- Base URL: `{provider.get('base_url', '')}`",
        f"- API key configured: `{provider.get('api_key_configured', False)}`",
        f"- Timeout seconds: `{provider.get('timeout_seconds', '')}`",
        f"- Secrets redacted: `{report.get('secrets_redacted', True)}`",
        "",
        "## Runtime Diagnostics",
        "",
        f"- Endpoint kind: `{diagnostics.get('endpoint_kind', '')}`",
        f"- Base URL host: `{diagnostics.get('base_url_host', '')}`",
        f"- Request URL: `{diagnostics.get('request_url', '')}`",
        f"- Proxy env: `HTTP={proxy_env.get('HTTP_PROXY', False)}` / `HTTPS={proxy_env.get('HTTPS_PROXY', False)}` / `ALL={proxy_env.get('ALL_PROXY', False)}` / `NO_PROXY={proxy_env.get('NO_PROXY', False)}`",
        f"- Image count: `{image_selection.get('image_count', 0)}`",
        f"- Source image count: `{image_selection.get('source_image_count', 0)}`",
        f"- Max images: `{image_selection.get('max_images', 8)}`",
        f"- Has single-image check: `{image_selection.get('has_single_image_check', False)}`",
        f"- Has multi-image check: `{image_selection.get('has_multi_image_check', False)}`",
        f"- Image probe status: `{image_probe.get('status', '')}`",
        f"- Image probe total bytes: `{image_probe.get('total_source_bytes', 0)}` -> `{image_probe.get('total_probe_bytes', 0)}`",
        f"- Failure diagnosis: `{failure_diagnosis.get('status', '')}`",
        f"- Text ping OK: `{failure_diagnosis.get('text_ping_ok', False)}`",
        f"- Image checks failed: `{failure_diagnosis.get('image_checks_failed', 0)}`",
        "",
        "## Recovery",
        "",
        str(report.get("recovery_suggestion") or ""),
        "",
        "## Checks",
        "",
        "| Check | OK | Status | Error Class | Images | Error |",
        "|---|---:|---|---|---:|---|",
    ]
    for check in report.get("checks") or []:
        if not isinstance(check, dict):
            continue
        lines.append(
            "| `{name}` | {ok} | `{status}` | `{error_class}` | {images} | {error} |".format(
                name=check.get("name", ""),
                ok=check.get("ok", False),
                status=check.get("status", ""),
                error_class=check.get("error_class", ""),
                images=check.get("image_count", 0),
                error=_md_cell(str(check.get("error") or "")),
            )
        )
    lines.extend(["", "## Evidence Images", ""])
    images = [str(path) for path in report.get("image_paths") or [] if str(path)]
    if images:
        for path in images:
            lines.append(f"- `{path}`")
    else:
        lines.append("- No image smoke checks requested; text ping only.")
    return "\n".join(lines).rstrip() + "\n"


def render_vision_provider_matrix_markdown(report: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-vision-provider-matrix",
        f'created: "{report.get("checked_at", now_iso())}"',
        "---",
        "",
        "# Vision Provider Matrix",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Recommended provider: `{report.get('recommended_provider', '')}`",
        f"- Recommended provider config: `{report.get('recommended_provider_config', {})}`",
        f"- Secrets redacted: `{report.get('secrets_redacted', True)}`",
        "",
        "## Ranking",
        "",
        "| Rank | Provider | Ready | Score | Text | Single image | Multi image | Failures | Preference | Reason |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("provider_ranking") or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {rank} | {provider} | `{ready}` | {score} | `{text}` | `{single}` | `{multi}` | {failures} | `{preferred}` | {reason} |".format(
                rank=row.get("rank", ""),
                provider=_md_cell(str(row.get("provider", ""))),
                ready=bool(row.get("ready")),
                score=row.get("score", 0),
                text=bool(row.get("text_ping_ok")),
                single=bool(row.get("single_image_json_ok")),
                multi=bool(row.get("multi_image_json_ok")),
                failures=row.get("failure_count", 0),
                preferred=bool(row.get("preferred")),
                reason=_md_cell(str(row.get("reason") or "")),
            )
        )
    lines.extend(
        [
            "",
            "## Providers",
            "",
            "| Provider | Model | Ready | Status | Error Class | Endpoint | Proxy | Recovery |",
            "|---|---|---:|---|---|---|---|---|",
        ]
    )
    for row in report.get("results") or []:
        if not isinstance(row, dict):
            continue
        provider = row.get("provider") if isinstance(row.get("provider"), dict) else {}
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        proxy_env = diagnostics.get("proxy_env") if isinstance(diagnostics.get("proxy_env"), dict) else {}
        failure_diagnosis = row.get("failure_diagnosis") if isinstance(row.get("failure_diagnosis"), dict) else {}
        proxy = f"H={bool(proxy_env.get('HTTP_PROXY'))},S={bool(proxy_env.get('HTTPS_PROXY'))},A={bool(proxy_env.get('ALL_PROXY'))}"
        lines.append(
            "| {provider} | {model} | `{ready}` | `{status}` | `{error_class}` | `{endpoint}` | `{proxy}` | {recovery} |".format(
                provider=_md_cell(str(provider.get("provider", ""))),
                model=_md_cell(str(provider.get("model", ""))),
                ready=bool(row.get("safe_to_execute")),
                status=_md_cell(str(row.get("status", ""))),
                error_class=_md_cell(str(failure_diagnosis.get("status") or row.get("error_class", ""))),
                endpoint=_md_cell(str(diagnostics.get("request_url") or "")),
                proxy=_md_cell(proxy),
                recovery=_md_cell(str(row.get("recovery_suggestion") or "")),
            )
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
        ]
    )
    if report.get("recommended_provider"):
        lines.append("Run `vision-execution-preflight --check-provider` with the recommended provider profile, then execute only the confirmed batch.")
    else:
        lines.append("No provider is currently safe. Fix the provider/network/auth issue or complete the remaining visual gaps through reviewed `review-notes.json` import.")
    return "\n".join(lines).rstrip() + "\n"


def _smoke_image_paths(
    *,
    bundle_root: Path | None,
    single_image: str | Path | None,
    multi_image_dir: str | Path | None,
    max_images: int = 8,
) -> list[str]:
    paths: list[str] = []
    if single_image:
        path = _resolve_path(bundle_root, single_image)
        if path.exists():
            paths.append(str(path))
    if multi_image_dir:
        paths.extend(str(path) for path in _image_files(_resolve_path(bundle_root, multi_image_dir))[:8])
    if not paths and bundle_root:
        paths.extend(_bundle_smoke_images(bundle_root))
    return _dedupe(paths)[: max(int(max_images or 8), 1)]


def _prepare_image_probe(
    source_paths: list[str],
    *,
    output_dir: Path,
    max_edge: int = 0,
    jpeg_quality: int = 70,
) -> dict[str, Any]:
    return prepare_image_probe(
        source_paths,
        output_dir=output_dir,
        max_edge=max_edge,
        jpeg_quality=jpeg_quality,
        role="vision_provider_smoke",
    )

def _bundle_smoke_images(bundle_root: Path) -> list[str]:
    timeline_path = bundle_root / "timeline.json"
    data = read_json(timeline_path) if timeline_path.exists() else []
    timeline = data if isinstance(data, list) else []
    single = ""
    multi: list[str] = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        for candidate in _item_frame_paths(bundle_root, item):
            if not single and Path(candidate).exists():
                single = candidate
        temporal = [path for path in _item_temporal_paths(bundle_root, item) if Path(path).exists()]
        if len(temporal) >= 2:
            multi = temporal[:8]
            break
    result = [single] if single else []
    result.extend(multi)
    return result


def _item_frame_paths(bundle_root: Path, item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("frame_paths", "evidence_paths"):
        values = item.get(key)
        if isinstance(values, list):
            paths.extend(str(_resolve_path(bundle_root, value)) for value in values if str(value or ""))
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            value = asset.get("path") or asset.get("source")
            if value:
                paths.append(str(_resolve_path(bundle_root, value)))
    return paths


def _item_temporal_paths(bundle_root: Path, item: dict[str, Any]) -> list[str]:
    values = item.get("temporal_frame_paths")
    if not isinstance(values, list):
        return []
    return [str(_resolve_path(bundle_root, value)) for value in values if str(value or "")]


def _image_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([child for child in path.iterdir() if child.is_file() and child.suffix.lower() in suffixes])


def _resolve_path(bundle_root: Path | None, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or not bundle_root:
        return path.resolve()
    return (bundle_root / path).resolve()


def _recovery_suggestion(provider_test: dict[str, Any]) -> str:
    status = str(provider_test.get("status") or "")
    failure_diagnosis = provider_test.get("failure_diagnosis") if isinstance(provider_test.get("failure_diagnosis"), dict) else {}
    error_class = str(failure_diagnosis.get("status") or provider_test.get("error_class") or status)
    if provider_test.get("safe_to_execute"):
        return "Provider text/image JSON smoke checks passed. Rerun vision-execution-preflight with --check-provider, then execute the confirmed visual batch."
    if error_class == "text_only_ok_image_timeout":
        return "Text ping passed but image checks timed out. Increase timeout_seconds, reduce image count/size, or verify the model/endpoint supports vision payloads."
    if error_class == "text_only_ok_image_not_supported":
        return "Text ping passed but image payloads appear unsupported. Switch to a vision-capable model/profile before running real visual analysis."
    if error_class == "text_only_ok_image_payload_too_large":
        return "Text ping passed but image payloads appear too large. Reduce frame resolution/count or JPEG quality, then rerun this smoke check."
    if error_class == "text_only_ok_image_parse_failed":
        return "Text ping passed but image checks did not return parseable JSON. Adjust the JSON-only prompt or switch provider/model before real execution."
    if error_class == "missing_api_key":
        return "Missing API key. Set the provider-specific key in the local environment or explicit provider config, then rerun this smoke check."
    if error_class == "provider_unreachable":
        return "Provider is unreachable or timed out. Check base_url, local proxy/network, and timeout_seconds; then rerun this smoke check before real model execution."
    if error_class == "provider_dns_failed":
        return "Provider host could not be resolved. Check the base URL host, DNS, and proxy settings before rerunning this smoke check."
    if error_class == "provider_proxy_failed":
        return "The request appears to fail through the proxy path. Check HTTP_PROXY/HTTPS_PROXY/ALL_PROXY or temporarily switch provider/network."
    if error_class == "provider_connection_refused":
        return "Connection was refused by the endpoint. Check whether the base_url points to the correct API host and path."
    if error_class == "provider_transport_error":
        return "Transport/TLS failed before a valid model response. Check proxy/TLS path or switch provider."
    if error_class == "provider_auth_failed":
        return "Provider rejected authentication. Verify or rotate the API key without writing it into repo files."
    if error_class == "provider_rate_limited":
        return "Provider rate limit or quota was hit. Wait, reduce batch size, or switch provider."
    if error_class == "model_output_parse_failed":
        return "Provider responded but did not return parseable JSON. Try another model/profile or adjust the JSON-only prompt before real execution."
    return f"Provider smoke failed with `{status}`. Inspect checks and rerun with a known-good provider profile."


def _public_provider_args(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.get("provider"),
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "model": cfg.get("model"),
        "timeout_seconds": cfg.get("timeout_seconds"),
    }


def _image_selection_summary(image_paths: list[str], *, source_image_paths: list[str] | None = None, max_images: int = 8) -> dict[str, Any]:
    return {
        "image_count": len(image_paths),
        "source_image_count": len(source_image_paths or image_paths),
        "has_single_image_check": bool(image_paths),
        "has_multi_image_check": len(image_paths) >= 2,
        "max_images": int(max_images or 8),
    }


def rank_vision_providers(results: list[dict[str, Any]], *, preferred_provider: str = "") -> list[dict[str, Any]]:
    preferred = str(preferred_provider or "").strip().lower().replace("-", "_")
    rows: list[dict[str, Any]] = []
    for original_position, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
        diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
        checks = result.get("checks") if isinstance(result.get("checks"), list) else []
        check_status = _check_status_map(checks)
        provider_name = str(provider.get("provider") or diagnostics.get("provider") or "").strip()
        gateway_configured = bool(provider.get("gateway_configured"))
        gateway_ready = bool(provider.get("gateway_ready"))
        credential_ready = bool(provider.get("credential_ready"))
        key_configured = (not bool(provider.get("api_key_required", True))) or bool(provider.get("api_key_configured") or diagnostics.get("api_key_configured") or credential_ready)
        provider_config_source = str(result.get("provider_config_source") or "legacy_provider")
        text_ok = bool((check_status.get("text_ping") or {}).get("ok"))
        single_ok = bool((check_status.get("single_image_json") or {}).get("ok"))
        multi_check_present = "multi_image_json" in check_status
        multi_ok = bool((check_status.get("multi_image_json") or {}).get("ok")) if multi_check_present else False
        failure_count = sum(1 for check in checks if isinstance(check, dict) and not check.get("ok"))
        timeout_seconds = _int_value(provider.get("timeout_seconds") or diagnostics.get("timeout_seconds") or 0)
        ready = (
            bool(result.get("safe_to_execute")) and gateway_ready and credential_ready
            if gateway_configured
            else bool(result.get("safe_to_execute")) and key_configured and text_ok and single_ok and (multi_ok if multi_check_present else True)
        )
        reason = (
            "gateway configured and ready; no provider request was sent; consent is still required"
            if ready and gateway_configured
            else _ranking_reason(
                key_configured=key_configured,
                text_ok=text_ok,
                single_ok=single_ok,
                multi_check_present=multi_check_present,
                multi_ok=multi_ok,
                ready=ready,
                error_class=str(result.get("error_class") or ""),
            )
        )
        score = (
            (1000 if key_configured else 0)
            + (200 if gateway_configured else 0)
            + (100 if text_ok else 0)
            + (50 if single_ok else 0)
            + (25 if multi_ok else 0)
            - (failure_count * 10)
            - timeout_seconds / 1000
            + (1 if preferred and provider_name.lower().replace("-", "_") == preferred else 0)
        )
        recommended_config = dict(result.get("recommended_provider_config") or {})
        rows.append(
            {
                "rank": 0,
                "provider": provider_name,
                "model": str(provider.get("model") or diagnostics.get("model") or ""),
                "ready": ready,
                "score": round(score, 3),
                "key_configured": key_configured,
                "provider_config_source": provider_config_source,
                "gateway_configured": gateway_configured,
                "gateway_ready": gateway_ready,
                "credential_ready": credential_ready,
                "text_ping_ok": text_ok,
                "single_image_json_ok": single_ok,
                "multi_image_json_ok": multi_ok,
                "multi_image_check_present": multi_check_present,
                "failure_count": failure_count,
                "timeout_seconds": timeout_seconds,
                "preferred": bool(preferred and provider_name.lower().replace("-", "_") == preferred),
                "reason": reason,
                "recommended_provider_config": recommended_config if ready else {},
                "original_position": original_position,
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row["ready"]),
            not bool(row["key_configured"]),
            not bool(row["gateway_configured"]),
            not bool(row["text_ping_ok"]),
            not bool(row["single_image_json_ok"]),
            not bool(row["multi_image_json_ok"]),
            int(row["failure_count"]),
            int(row["timeout_seconds"] or 0),
            not bool(row["preferred"]),
            int(row["original_position"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row.pop("original_position", None)
    return rows


def _recommended_provider_config(
    *,
    provider_test: dict[str, Any],
    image_probe_max_edge: int,
    image_probe_jpeg_quality: int,
) -> dict[str, Any]:
    if not provider_test.get("safe_to_execute"):
        return {}
    provider = provider_test.get("provider") if isinstance(provider_test.get("provider"), dict) else {}
    if not provider.get("provider"):
        return {}
    config: dict[str, Any] = {
        "provider": provider.get("provider"),
        "model": provider.get("model"),
        "image_probe_max_edge": int(image_probe_max_edge or 0),
        "image_probe_jpeg_quality": int(image_probe_jpeg_quality or 70),
    }
    base_url = str(provider.get("base_url") or "").strip()
    if base_url:
        config["base_url"] = redact_url_secrets(base_url)
    if provider.get("timeout_seconds"):
        config["timeout_seconds"] = int(provider.get("timeout_seconds") or 60)
    return {key: value for key, value in config.items() if value not in (None, "", [], {})}

    for key in ("adapter_backend", "execution_location", "route_id", "route_revision", "virtual_model", "profile_id"):
        value = provider.get(key)
        if value not in (None, "", [], {}):
            config[key] = value
    if provider.get("gateway_configured"):
        config["provider_config_source"] = "route_based_gateway"


def _check_status_map(checks: list[Any]) -> dict[str, dict[str, Any]]:
    return {
        str(check.get("name") or ""): check
        for check in checks
        if isinstance(check, dict) and str(check.get("name") or "")
    }


def _ranking_reason(
    *,
    key_configured: bool,
    text_ok: bool,
    single_ok: bool,
    multi_check_present: bool,
    multi_ok: bool,
    ready: bool,
    error_class: str,
) -> str:
    if ready:
        return "provider passed key, text, and required image JSON checks"
    if not key_configured:
        return "missing API key"
    if not text_ok:
        return "text ping failed"
    if not single_ok:
        return "single-image JSON check failed or was not available"
    if multi_check_present and not multi_ok:
        return "multi-image JSON check failed"
    return error_class or "provider is not safe to execute"


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _rel_to_bundle(bundle_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(bundle_root.resolve()))
    except ValueError:
        return str(path)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _path_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() and path.is_file() else 0
    except OSError:
        return 0


def _total_bytes(paths: list[str]) -> int:
    return sum(_path_bytes(Path(path).expanduser()) for path in paths)
