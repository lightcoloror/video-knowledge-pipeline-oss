from __future__ import annotations

from pathlib import Path
from typing import Any

from .path_defaults import tool_source_review_root
from .media_async_client import _is_loopback_base_url
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json, write_text_atomic


SOURCE_ROOT = tool_source_review_root()


def local_vlm_adapter_plan(output_dir: str | None = None, write: bool = False) -> dict[str, Any]:
    """Return the planned local VLM adapter contracts without launching models."""
    repos = [
        {
            "name": "Qwen2.5-VL/Qwen3-VL",
            "local_path": str(SOURCE_ROOT / "Qwen2.5-VL"),
            "status": _repo_status(SOURCE_ROOT / "Qwen2.5-VL"),
            "recommended_mode": "openai_compatible_http",
            "why": "Supports image lists as video, local video paths, fps/num_frames controls, and qwen-vl-utils preprocessing.",
            "adapter_contract": {
                "input": "frame_paths or video_path plus prompt JSON",
                "output": "OpenAI-compatible chat completion content parsed by vision_api.parse_model_json",
                "evidence_required": "all frame paths or source video path must be echoed in the pipeline report",
            },
        },
        {
            "name": "InternVL",
            "local_path": str(SOURCE_ROOT / "InternVL"),
            "status": _repo_status(SOURCE_ROOT / "InternVL"),
            "recommended_mode": "subprocess_or_http_worker",
            "why": "README examples sample videos into num_segments frames with decord and model.chat, which matches temporal frame groups.",
            "adapter_contract": {
                "input": "ordered frame group or video segment path",
                "output": "JSON text describing objects, state changes, operation steps, uncertainty, and retained evidence",
                "evidence_required": "frame index/time metadata from temporal_frame_groups",
            },
        },
        {
            "name": "LLaVA-NeXT/LLaVA-OneVision",
            "local_path": str(SOURCE_ROOT / "LLaVA-NeXT"),
            "status": _repo_status(SOURCE_ROOT / "LLaVA-NeXT"),
            "recommended_mode": "sglang_http_or_subprocess_worker",
            "why": "Repository includes video sampling helpers and points to SGLang HTTP deployment for video models.",
            "adapter_contract": {
                "input": "short video path or sampled frames plus JSON prompt",
                "output": "JSON text parsed through the same multimodal/temporal normalizers",
                "evidence_required": "sampled frame list and source segment timestamps",
            },
        },
    ]
    result = {
        "ok": True,
        "schema": "video_knowledge_local_vlm_adapter_plan.v1",
        "default_recommendation": "Use the existing vision_api provider layer. Run local Qwen-VL behind an OpenAI-compatible HTTP server and select provider=local_qwen_vl; do not import model code into VKP.",
        "implemented_provider_profiles": {
            "local_qwen_vl": {
                "provider": "local_qwen_vl",
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "api_key_required": False,
                "env": [
                    "LECTURE_VISION_PROVIDER=local_qwen_vl",
                    "LECTURE_VISION_BASE_URL=http://127.0.0.1:8000/v1",
                    "LECTURE_VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct",
                    "LOCAL_QWEN_VL_API_KEY=<optional if server requires auth>",
                ],
            },
            "local_vlm": {
                "provider": "local_vlm",
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "local-vlm",
                "api_key_required": False,
            },
        },
        "shared_adapter_rules": [
            "Do not import model repositories inside the main pipeline package.",
            "Prefer HTTP/OpenAI-compatible serving so CLI, MCP, and WebUI use the same provider layer.",
            "Subprocess workers are allowed for smoke tests but must write raw output and never mutate timeline on parse failure.",
            "All local VLM output must preserve frame_paths or source segment evidence paths.",
        ],
        "repos": repos,
    }
    if write:
        out_dir = Path(output_dir or ".").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "local-vlm-adapter-plan.json"
        markdown_path = out_dir / "local-vlm-adapter-plan.md"
        result["artifacts"] = {"json": str(report_path), "markdown": str(markdown_path)}
        result["report_path"] = str(report_path)
        result["report_markdown_path"] = str(markdown_path)
        write_json(report_path, result)
        write_text_atomic(markdown_path, _render_markdown(result))
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Local VLM Adapter Plan",
        "",
        f"- Default recommendation: {result.get('default_recommendation')}",
        "- Boundary: do not import model repositories into VKP; serve local models through HTTP/OpenAI-compatible or an isolated worker.",
        "",
        "## Provider Profiles",
        "",
    ]
    profiles = result.get("implemented_provider_profiles") if isinstance(result.get("implemented_provider_profiles"), dict) else {}
    for key, profile in profiles.items():
        lines.extend(
            [
                f"### `{key}`",
                "",
                f"- Provider: `{profile.get('provider')}`",
                f"- Base URL: `{profile.get('base_url')}`",
                f"- Model: `{profile.get('model')}`",
                f"- API key required: `{profile.get('api_key_required')}`",
                "",
            ]
        )
        env = profile.get("env") if isinstance(profile.get("env"), list) else []
        if env:
            lines.append("Environment:")
            for item in env:
                lines.append(f"- `{item}`")
            lines.append("")
    lines.extend(["## Reviewed Local Model Repos", ""])
    for repo in result.get("repos") or []:
        status = repo.get("status") if isinstance(repo.get("status"), dict) else {}
        lines.extend(
            [
                f"### {repo.get('name')}",
                "",
                f"- Local path: `{repo.get('local_path')}`",
                f"- Exists: `{status.get('exists')}`",
                f"- Has README: `{status.get('has_readme')}`",
                f"- Recommended mode: `{repo.get('recommended_mode')}`",
                f"- Why: {repo.get('why')}",
                "",
                "Adapter contract:",
            ]
        )
        contract = repo.get("adapter_contract") if isinstance(repo.get("adapter_contract"), dict) else {}
        for key, value in contract.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(["## Shared Rules", ""])
    for rule in result.get("shared_adapter_rules") or []:
        lines.append(f"- {rule}")
    return "\n".join(lines).rstrip() + "\n"


def _repo_status(path: Path) -> dict[str, Any]:
    return {
        "exists": path.exists(),
        "path": str(path),
        "has_readme": (path / "README.md").exists(),
    }


def local_vlm_serving_smoke(
    *,
    provider: str = "local_qwen_vl",
    bundle_dir: str = "",
    output_dir: str = "",
    single_image: str = "",
    multi_image_dir: str = "",
    execute: bool = False,
    timeout_seconds: int = 30,
    max_images: int = 3,
    image_probe_max_edge: int = 512,
    image_probe_jpeg_quality: int = 70,
    frame_group_count: int = 8,
    base_url: str = "",
    model: str = "",
    max_tokens: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Plan or run a local VLM service capability smoke.

    The default is plan-only. Passing execute=True delegates to the existing
    provider smoke layer so Qwen/InternVL servers remain outside the main VKP
    process and never become default pipeline execution.
    """

    normalised = (provider or "local_qwen_vl").strip() or "local_qwen_vl"
    if normalised in {"qwen", "qwen2.5-vl", "qwen_vl", "qwen3-vl"}:
        normalised = "local_qwen_vl"
    if normalised in {"internvl", "local_internvl"}:
        normalised = "local_vlm"
    profiles = local_vlm_adapter_plan(write=False).get("implemented_provider_profiles", {})
    profile = dict(profiles.get(normalised) or profiles.get("local_vlm", {}))
    if base_url:
        profile["base_url"] = str(base_url)
    if model:
        profile["model"] = str(model)
    if max_tokens is not None and int(max_tokens) > 0:
        profile["max_tokens"] = int(max_tokens)
    out_dir = Path(output_dir or bundle_dir or ".").expanduser().resolve()
    bundle_root = Path(bundle_dir).expanduser().resolve() if bundle_dir else None
    frame_group = _select_short_frame_group(bundle_root, frame_group_count=frame_group_count) if bundle_root else {}
    operator_boundary = {
        "default_execute": False,
        "does_not_start_model_server": True,
        "does_not_modify_timeline": True,
        "no_cloud_call": True,
        "local_server_must_already_be_running": True,
        "provider_request_made": bool(execute),
        "reads_or_writes_pipeline_config": False,
    }
    if normalised in {"local_qwen_vl", "local_vlm"} and not _is_loopback_base_url(str(profile.get("base_url") or "")):
        return {
            "ok": False,
            "status": "invalid_local_provider_url",
            "schema": "video_knowledge_pipeline.local_vlm_serving_smoke.v1",
            "execute": bool(execute),
            "provider": normalised,
            "profile": profile,
            "input_spec": {
                "base_url": profile.get("base_url", ""),
                "model": profile.get("model", ""),
                "max_tokens": profile.get("max_tokens"),
            },
            "capability_matrix": [],
            "operator_boundary": {**operator_boundary, "provider_request_made": False},
            "error": "local VLM providers require an explicit loopback HTTP URL with a port",
        }
    result: dict[str, Any] = {
        "ok": True,
        "status": "plan_only" if not execute else "executing",
        "schema": "video_knowledge_pipeline.local_vlm_serving_smoke.v1",
        "execute": bool(execute),
        "provider": normalised,
        "profile": profile,
        "bundle_dir": str(bundle_root) if bundle_root else "",
        "output_dir": str(out_dir),
        "input_spec": {
            "transport": "openai_compatible_http",
            "provider": normalised,
            "base_url": profile.get("base_url", ""),
            "model": profile.get("model", ""),
            "max_tokens": profile.get("max_tokens"),
            "max_images": int(max_images or 0),
            "image_probe_max_edge": int(image_probe_max_edge or 0),
            "image_probe_jpeg_quality": int(image_probe_jpeg_quality or 70),
            "short_frame_group_target_count": int(frame_group_count or 0),
            "short_frame_group_found": bool(frame_group.get("frame_paths")),
            "short_frame_group_image_count": len(frame_group.get("frame_paths") or []),
        },
        "short_frame_group": frame_group,
        "capability_matrix": _local_vlm_capability_matrix(smoke=None, frame_group=frame_group, execute=execute),
        "command_examples": {
            "qwen_openai_server": "python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-3B-Instruct --host 127.0.0.1 --port 8000",
            "vkp_smoke_plan": ".\\scripts\\video-knowledge.ps1 local-vlm-serving-smoke --provider local_qwen_vl --bundle-dir <bundle>",
            "vkp_smoke_execute": ".\\scripts\\video-knowledge.ps1 local-vlm-serving-smoke --provider local_qwen_vl --bundle-dir <bundle> --execute",
            "lm_studio_plan_only": ".\\scripts\\video-knowledge.ps1 local-vlm-serving-smoke --provider local_vlm --base-url http://127.0.0.1:1234/v1 --model <loaded-model> --max-tokens 1200",
        },
        "operator_boundary": operator_boundary,
    }
    if execute:
        from .vision_provider_smoke import vision_provider_smoke

        result["smoke"] = vision_provider_smoke(
            provider_config=profile,
            timeout_seconds=timeout_seconds,
            bundle_dir=bundle_dir,
            single_image=single_image,
            multi_image_dir=multi_image_dir,
            output_dir=str(out_dir),
            max_images=max_images,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            write=write,
        )
        result["ok"] = bool(result["smoke"].get("ok"))
        result["status"] = "executed_ready" if result["ok"] else "executed_failed"
        result["capability_matrix"] = _local_vlm_capability_matrix(smoke=result["smoke"], frame_group=frame_group, execute=execute)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "local-vlm-serving-smoke.json"
        markdown_path = out_dir / "local-vlm-serving-smoke.md"
        args_path = out_dir / "mcp-local-vlm-serving-smoke.args.json"
        result["artifacts"] = {"json": str(json_path), "markdown": str(markdown_path)}
        result["mcp_args_path"] = str(args_path)
        write_json(json_path, result)
        write_text_atomic(markdown_path, _render_smoke_markdown(result))
        write_json(
            args_path,
            {
                "provider": normalised,
                "bundle_dir": str(bundle_root) if bundle_root else "",
                "output_dir": str(out_dir),
                "single_image": str(single_image or ""),
                "multi_image_dir": str(multi_image_dir or ""),
                "execute": bool(execute),
                "timeout_seconds": int(timeout_seconds or 30),
                "max_images": int(max_images or 3),
                "image_probe_max_edge": int(image_probe_max_edge or 512),
                "image_probe_jpeg_quality": int(image_probe_jpeg_quality or 70),
                "frame_group_count": int(frame_group_count or 8),
                "base_url": str(profile.get("base_url") or ""),
                "model": str(profile.get("model") or ""),
                "max_tokens": profile.get("max_tokens"),
                "write": True,
            },
        )
        if bundle_root:
            manifest_path = bundle_root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["local_vlm_serving_smoke"] = _rel_to_bundle(bundle_root, markdown_path)
                manifest["local_vlm_serving_smoke_json"] = _rel_to_bundle(bundle_root, json_path)
                manifest["mcp_local_vlm_serving_smoke_args"] = _rel_to_bundle(bundle_root, args_path)
                write_json(manifest_path, manifest)
                register_bundle_run(
                    bundle_root,
                    run_type="local_vlm_serving_smoke",
                    run_id="local-vlm-serving-smoke",
                    status="completed" if result.get("execute") and result.get("ok") else "needs_retry" if result.get("execute") else "needs_execution",
                    title="Local VLM serving smoke",
                    summary=f"Provider {result.get('provider', 'unknown')} / execute={bool(result.get('execute'))} / ok={bool(result.get('ok'))}.",
                    artifacts=[
                        {"key": "json", "path": json_path},
                        {"key": "markdown", "path": markdown_path},
                        {"key": "mcp_args", "path": args_path},
                    ],
                    failed_items=[] if result.get("ok") else [{"reason": "local_vlm_smoke_not_ready", "detail": "Local VLM service smoke has not completed successfully."}],
                    retry_command=f".\\scripts\\video-knowledge.ps1 local-vlm-serving-smoke --provider {normalised} --bundle-dir {bundle_root}",
                    operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {"does_not_start_model_server": True, "does_not_modify_timeline": True},
                    write=True,
                )
    return result


def _render_smoke_markdown(result: dict[str, Any]) -> str:
    input_spec = result.get("input_spec") if isinstance(result.get("input_spec"), dict) else {}
    lines = [
        "# Local VLM Serving Smoke",
        "",
        f"- Provider: `{result.get('provider')}`",
        f"- Execute: `{result.get('execute')}`",
        f"- OK: `{result.get('ok')}`",
        "- Boundary: this command never starts Qwen/InternVL servers and never mutates timeline.",
        f"- Base URL: `{input_spec.get('base_url', '')}`",
        f"- Model: `{input_spec.get('model', '')}`",
        f"- Max tokens: `{input_spec.get('max_tokens', '')}`",
        f"- Max images: `{input_spec.get('max_images', 0)}`",
        f"- Short frame group: `{input_spec.get('short_frame_group_image_count', 0)}` frames",
        "",
        "## Capability Matrix",
        "",
        "| Capability | Status | Evidence |",
        "|---|---|---|",
    ]
    for row in result.get("capability_matrix") or []:
        if not isinstance(row, dict):
            continue
        evidence = str(row.get("evidence", "")).replace("|", "\\|")
        lines.append(f"| `{row.get('key', '')}` | `{row.get('status', '')}` | {evidence} |")
    lines.extend(["", "## Commands", ""])
    for label, command in (result.get("command_examples") or {}).items():
        lines.append(f"- `{label}`: `{command}`")
    smoke = result.get("smoke") if isinstance(result.get("smoke"), dict) else {}
    if smoke:
        lines.extend(["", "## Smoke Result", "", f"- Schema: `{smoke.get('schema')}`", f"- OK: `{smoke.get('ok')}`"])
    frame_group = result.get("short_frame_group") if isinstance(result.get("short_frame_group"), dict) else {}
    if frame_group:
        lines.extend(["", "## Short Frame Group", ""])
        lines.append(f"- Timeline index: `{frame_group.get('timeline_index', '')}`")
        lines.append(f"- Time range: `{frame_group.get('start', '')}` - `{frame_group.get('end', '')}`")
        for frame_path in frame_group.get("frame_paths") or []:
            lines.append(f"- `{frame_path}`")
    return "\n".join(lines).rstrip() + "\n"


def _local_vlm_capability_matrix(*, smoke: dict[str, Any] | None, frame_group: dict[str, Any], execute: bool) -> list[dict[str, Any]]:
    checks = _check_map(smoke.get("checks") if isinstance(smoke, dict) else [])
    has_group = len(frame_group.get("frame_paths") or []) >= 2
    if not execute:
        return [
            {"key": "openai_compatible_endpoint", "status": "planned", "complete": False, "evidence": "provider profile is configured; server is not contacted in preview"},
            {"key": "text_json", "status": "planned", "complete": False, "evidence": "execute=true will run provider text ping"},
            {"key": "single_image_json", "status": "planned", "complete": False, "evidence": "fallback sends one prepared image per request"},
            {"key": "per_frame_aggregation", "status": "planned", "complete": False, "evidence": "semantic frame results are aggregated without multi-image fallback"},
            {"key": "multi_image_json", "status": "planned", "complete": False, "evidence": "diagnostic only; truncation never counts as complete fallback"},
            {"key": "short_frame_group_json", "status": "planned" if has_group else "missing_sample", "complete": False, "evidence": f"{len(frame_group.get('frame_paths') or [])} frame(s) selected from bundle"},
        ]
    single_complete = _check_complete(checks, "single_image_json")
    multi_complete = _check_complete(checks, "multi_image_json")
    return [
        {"key": "openai_compatible_endpoint", "status": "ok" if smoke and smoke.get("status") == "ok" else "check_report", "complete": bool(smoke and smoke.get("status") == "ok"), "evidence": str(smoke.get("status", "")) if smoke else "no smoke report"},
        {"key": "text_json", "status": _check_status(checks, "text_ping"), "complete": _check_complete(checks, "text_ping"), "evidence": _check_evidence(checks, "text_ping")},
        {"key": "single_image_json", "status": _check_status(checks, "single_image_json"), "complete": single_complete, "evidence": _check_evidence(checks, "single_image_json")},
        {"key": "per_frame_aggregation", "status": "ready" if single_complete else "incomplete", "complete": single_complete, "evidence": "single-frame fallback is the canonical local aggregation mode"},
        {"key": "multi_image_json", "status": _check_status(checks, "multi_image_json"), "complete": multi_complete, "evidence": _check_evidence(checks, "multi_image_json")},
        {
            "key": "short_frame_group_json",
            "status": _check_status(checks, "multi_image_json") if has_group else "missing_sample",
            "complete": multi_complete if has_group else False,
            "evidence": f"{len(frame_group.get('frame_paths') or [])} ordered frame(s); uses multi_image_json check as short-frame-group proxy",
        },
    ]


def _select_short_frame_group(bundle_root: Path | None, *, frame_group_count: int) -> dict[str, Any]:
    if not bundle_root:
        return {}
    timeline_path = bundle_root / "timeline.json"
    timeline = read_json(timeline_path) if timeline_path.exists() else []
    if not isinstance(timeline, list):
        return {}
    target_count = max(2, min(int(frame_group_count or 8), 12))
    best: dict[str, Any] = {}
    for item in timeline:
        if not isinstance(item, dict):
            continue
        paths = _existing_paths(bundle_root, item.get("temporal_frame_paths"))
        if len(paths) >= 2:
            selected = paths[:target_count]
            if len(selected) > len(best.get("frame_paths") or []):
                best = _frame_group_row(item, selected)
        if len(best.get("frame_paths") or []) >= target_count:
            break
    if best:
        return best
    for item in timeline:
        if not isinstance(item, dict):
            continue
        paths = _existing_paths(bundle_root, item.get("frame_paths"))
        if len(paths) >= 2:
            return _frame_group_row(item, paths[:target_count])
    return {}


def _existing_paths(bundle_root: Path, values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    paths: list[str] = []
    for value in values:
        if not str(value or ""):
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = bundle_root / path
        resolved = path.resolve()
        if resolved.exists():
            paths.append(str(resolved))
    return paths


def _frame_group_row(item: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    return {
        "timeline_index": item.get("index", ""),
        "start": item.get("start", item.get("start_seconds", "")),
        "end": item.get("end", item.get("end_seconds", "")),
        "visual_route": item.get("visual_route", ""),
        "frame_paths": paths,
    }


def _check_map(checks: Any) -> dict[str, dict[str, Any]]:
    return {str(row.get("name") or ""): row for row in checks or [] if isinstance(row, dict)}


def _check_status(checks: dict[str, dict[str, Any]], key: str) -> str:
    row = checks.get(key)
    if not row:
        return "not_run"
    if row.get("truncated"):
        return "truncated"
    return "ok" if row.get("ok") else str(row.get("status") or row.get("error_class") or "failed")


def _check_complete(checks: dict[str, dict[str, Any]], key: str) -> bool:
    row = checks.get(key)
    if not row:
        return False
    return bool(row.get("complete", row.get("ok") and not row.get("truncated")))


def _check_evidence(checks: dict[str, dict[str, Any]], key: str) -> str:
    row = checks.get(key)
    if not row:
        return "check was not run"
    return f"images={row.get('image_count', 0)}, status={row.get('status', '')}, error={row.get('error_class', '')}"


def _rel_to_bundle(bundle_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(bundle_root.resolve()))
    except ValueError:
        return str(path)
