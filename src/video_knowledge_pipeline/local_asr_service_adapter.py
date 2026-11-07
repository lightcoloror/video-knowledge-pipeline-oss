from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .asr_adapter import normalize_asr_output
from .config import asr_runtime_profile
from .models import new_id, now_iso
from .model_task_gateway import model_task_api_call
from .online_model_gateway import asr_transcriptions_url, resolve_asr_provider_config


def online_model_api_call(model_type, **kwargs):
    return model_task_api_call("local_asr_service", **kwargs)
from .run_artifact_registry import register_bundle_run
from .storage import append_jsonl, ensure_project_dirs, read_json, read_jsonl, write_json
from .vision_api import redact_url_secrets


LOCAL_ASR_SERVICE_PLAN_SCHEMA = "video_knowledge_pipeline.local_asr_service_plan.v1"
LOCAL_ASR_SERVICE_RUN_SCHEMA = "video_knowledge_pipeline.local_asr_service_run.v1"


def plan_local_asr_service_run(
    root: str | Path,
    media_path: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    model: str = "",
    language: str = "zh",
    prompt: str = "",
) -> dict[str, Any]:
    """Plan a Speaches/OpenAI-compatible ASR service run.

    This is intentionally preview-first. It reuses VKP's OpenAI-compatible ASR
    gateway, but defaults to the local service profile in config.asr_runtime.
    """
    root_path = Path(root).expanduser().resolve()
    paths = ensure_project_dirs(root_path)
    media = Path(media_path).expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(f"media not found: {media}")

    cfg = _runtime_service_config(provider_config=provider_config, model=model, language=language)
    resolved = resolve_asr_provider_config(cfg)
    local_service = _is_local_service_url(str(resolved.get("base_url") or ""))
    run_id = new_id("local_asr_service")
    output_dir = paths["transcripts"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / "raw-asr-output.json"
    plan_path = output_dir / "local-asr-service-plan.json"
    plan = {
        "schema": LOCAL_ASR_SERVICE_PLAN_SCHEMA,
        "run_id": run_id,
        "project": str(paths["root"]),
        "preset": "local-asr-service",
        "provider": "speaches_openai_compatible" if local_service else "custom_openai_compatible_asr",
        "media_path": str(media),
        "output_dir": str(output_dir),
        "expected_output_json": str(raw_output),
        "provider_config": _safe_provider_config(resolved),
        "language": language,
        "prompt": prompt,
        "execute": False,
        "local_service": local_service,
        "upload_required": not local_service,
        "default_upload": False,
        "request_plan": {
            "interface": "openai_audio_transcriptions",
            "url": redact_url_secrets(asr_transcriptions_url(resolved)),
            "model": resolved.get("model", ""),
            "language": resolved.get("language", ""),
            "audio_path": str(media),
            "audio_exists": media.exists(),
        },
        "operator_boundary": {
            "preview_first": True,
            "execute_required_for_service_call": True,
            "local_service_execution_allowed_by_default": local_service,
            "remote_service_requires_allow_remote": not local_service,
            "provider_config_runtime_only": True,
            "secrets_redacted": True,
            "reuse_source": "Speaches/OpenAI-compatible audio transcriptions contract via VKP online_model_gateway.",
        },
        "next_actions": _next_actions(local_service),
        "created_at": now_iso(),
    }
    write_json(plan_path, plan)
    return {**plan, "plan_path": str(plan_path)}


def run_local_asr_service_plan(
    plan_json: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    normalize: bool = True,
    allow_remote: bool = False,
) -> dict[str, Any]:
    plan_path = Path(plan_json).expanduser().resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("local ASR service plan must be a JSON object")
    project = Path(str(plan.get("project") or plan_path.parent)).expanduser().resolve()
    media = str(plan.get("media_path") or "")
    output_dir = Path(str(plan.get("output_dir") or plan_path.parent)).expanduser().resolve()
    expected_output = Path(str(plan.get("expected_output_json") or output_dir / "raw-asr-output.json")).expanduser().resolve()
    cfg = dict(plan.get("provider_config") if isinstance(plan.get("provider_config"), dict) else {})
    cfg.update(provider_config or {})
    resolved = resolve_asr_provider_config(cfg)
    local_service = _is_local_service_url(str(resolved.get("base_url") or ""))
    result: dict[str, Any] = {
        "schema": LOCAL_ASR_SERVICE_RUN_SCHEMA,
        "plan_path": str(plan_path),
        "project": str(project),
        "preset": "local-asr-service",
        "provider": "speaches_openai_compatible" if local_service else "custom_openai_compatible_asr",
        "media_path": media,
        "output_dir": str(output_dir),
        "expected_output_json": str(expected_output),
        "execute": bool(execute),
        "normalize": bool(normalize),
        "allow_remote": bool(allow_remote),
        "local_service": local_service,
        "started_at": now_iso(),
        "status": "preview",
        "service_call": {},
        "raw_output_json": str(expected_output) if expected_output.exists() else "",
        "normalized": None,
        "operator_boundary": {
            "preview_first": True,
            "execute_required_for_service_call": True,
            "remote_service_requires_allow_remote": True,
            "secrets_redacted": True,
        },
    }
    if not execute:
        result["finished_at"] = now_iso()
        return _write_local_service_log(project, result)
    if not local_service and not allow_remote:
        result.update({"status": "blocked_remote_asr_service", "error": "remote OpenAI-compatible ASR service requires allow_remote=true", "finished_at": now_iso()})
        return _write_local_service_log(project, result)

    output_dir.mkdir(parents=True, exist_ok=True)
    call = online_model_api_call(
        "asr",
        provider_config=resolved,
        audio_path=media,
        prompt=str(plan.get("prompt") or ""),
        execute=True,
        output_dir=output_dir,
        write=True,
    )
    result["service_call"] = _compact_service_call(call)
    if not call.get("ok"):
        result.update({"status": str(call.get("status") or call.get("error") or "local_asr_service_failed"), "finished_at": now_iso()})
        return _write_local_service_log(project, result)
    raw_payload = call.get("raw_response") if call.get("raw_response") is not None else {"text": call.get("content", "")}
    write_json(expected_output, raw_payload)
    result["raw_output_json"] = str(expected_output)
    result["status"] = "ok"
    if normalize:
        try:
            title = Path(media).stem if media else expected_output.stem
            result["normalized"] = normalize_asr_output(project, expected_output, provider="openai", title=title)
        except Exception as exc:  # noqa: BLE001 - surfaced in report.
            result["status"] = "normalize_failed"
            result["error"] = str(exc)
    result["finished_at"] = now_iso()
    return _write_local_service_log(project, result)


def _runtime_service_config(*, provider_config: dict[str, Any] | None, model: str, language: str) -> dict[str, Any]:
    runtime = asr_runtime_profile()
    service = dict(runtime.get("openai_compatible") if isinstance(runtime.get("openai_compatible"), dict) else {})
    cfg = {
        "provider": "openai_compatible_asr",
        "base_url": service.get("base_url") or "http://127.0.0.1:8000/v1",
        "model": service.get("model") or "Systran/faster-whisper-large-v3",
        "timeout_seconds": int(service.get("timeout_seconds") or 600),
        "language": language,
    }
    cfg.update(provider_config or {})
    if model:
        cfg["model"] = model
    return cfg


def _is_local_service_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _safe_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if key != "api_key"}


def _compact_service_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": call.get("schema", ""),
        "model_type": call.get("model_type", ""),
        "status": call.get("status", ""),
        "ok": bool(call.get("ok")),
        "error": call.get("error", ""),
        "request_plan": call.get("request_plan", {}),
        "artifacts": call.get("artifacts", {}),
        "secrets_redacted": True,
    }


def _write_local_service_log(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "local-asr-service-runs.jsonl"
    record = {
        "created_at": now_iso(),
        "preset": "local-asr-service",
        "provider": result.get("provider", ""),
        "execute": bool(result.get("execute")),
        "normalize": bool(result.get("normalize")),
        "local_service": bool(result.get("local_service")),
        "status": result.get("status", ""),
        "raw_output_json": result.get("raw_output_json", ""),
        "normalized_json": ((result.get("normalized") or {}).get("json_path") if isinstance(result.get("normalized"), dict) else ""),
        "error": result.get("error", "") or ((result.get("service_call") or {}).get("error") if isinstance(result.get("service_call"), dict) else ""),
    }
    append_jsonl(log_path, [record])
    rows = read_jsonl(log_path)
    markdown_path = paths["notes"] / "local-asr-service-runs.md"
    markdown_path.write_text(_render_log_markdown(root, rows), encoding="utf-8")
    report = _write_local_service_report(result, record)
    result["local_asr_service_log"] = {"record": record, "log_path": str(log_path), "markdown_path": str(markdown_path), "report_path": str(report), "count": len(rows)}
    registry = _register_local_service_run_if_bundle(root, result)
    if registry:
        result["run_registry"] = registry
    return result


def _write_local_service_report(result: dict[str, Any], record: dict[str, Any]) -> Path:
    output_dir = Path(str(result.get("output_dir") or ".")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "local-asr-service-run-report.md"
    lines = [
        "# Local ASR Service Run Report",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Execute: `{bool(result.get('execute'))}`",
        f"- Local service: `{bool(result.get('local_service'))}`",
        f"- Remote allowed: `{bool(result.get('allow_remote'))}`",
        f"- Raw output: `{result.get('raw_output_json', '')}`",
        f"- Normalized JSON: `{record.get('normalized_json', '')}`",
        f"- Started at: `{result.get('started_at', '')}`",
        f"- Finished at: `{result.get('finished_at', '')}`",
        "",
        "## Boundary",
        "",
        "This branch is for Speaches/OpenAI-compatible ASR services. Localhost services are allowed only with explicit execution; remote services additionally require `allow_remote=true`.",
        "",
        "## Error",
        "",
        "```text",
        str(record.get("error") or ""),
        "```",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _render_log_markdown(root: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Local ASR Service Runs",
        "",
        f"- Project: `{root}`",
        f"- Count: {len(rows)}",
        "- Boundary: localhost OpenAI-compatible ASR service only by default; remote endpoints require explicit allow_remote.",
        "",
        "| Time | Execute | Local | Normalize | Status | Output |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        execute = "yes" if row.get("execute") else "no"
        local = "yes" if row.get("local_service") else "no"
        normalize = "yes" if row.get("normalize") else "no"
        lines.append(f"| {row.get('created_at', '')} | {execute} | {local} | {normalize} | {row.get('status', '')} | `{row.get('normalized_json') or row.get('raw_output_json') or ''}` |")
    return "\n".join(lines).rstrip() + "\n"


def _next_actions(local_service: bool) -> list[str]:
    if local_service:
        return [
            "Start Speaches or another local OpenAI-compatible ASR server if it is not already running.",
            "Run run-local-asr-service-plan with --execute to transcribe through localhost.",
            "Use the normalized transcript as an evidence source; raw SenseVoice/FunASR remains available for comparison.",
        ]
    return [
        "This endpoint is not localhost. Review privacy and upload boundary first.",
        "Run run-local-asr-service-plan with --execute --allow-remote only after explicit approval.",
    ]


def _register_local_service_run_if_bundle(root: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    if not (root.expanduser().resolve() / "manifest.json").exists():
        return None
    return register_bundle_run(
        root,
        run_type="local_asr_service",
        title="Local ASR service run",
        status=str(result.get("status") or ""),
        summary=f"execute={bool(result.get('execute'))}, local_service={bool(result.get('local_service'))}",
        artifacts={
            "report": ((result.get("local_asr_service_log") or {}).get("report_path") if isinstance(result.get("local_asr_service_log"), dict) else ""),
            "raw_output_json": result.get("raw_output_json", ""),
            "normalized_json": ((result.get("normalized") or {}).get("json_path") if isinstance(result.get("normalized"), dict) else ""),
        },
        safety={
            "local_service": bool(result.get("local_service")),
            "remote_allowed": bool(result.get("allow_remote")),
            "secrets_redacted": True,
        },
    )
