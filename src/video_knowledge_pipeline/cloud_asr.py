from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .asr_adapter import normalize_asr_output
from .models import new_id, now_iso
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .model_task_gateway import model_task_api_call
from .online_model_gateway import resolve_asr_provider_config, asr_transcriptions_url
from .run_artifact_registry import register_bundle_run
from .storage import append_jsonl, ensure_project_dirs, read_json, read_jsonl, write_json
from .file_hash import sha256_file as _file_sha256
from .vision_api import redact_url_secrets


def online_model_api_call(model_type, **kwargs):
    return model_task_api_call("cloud_asr", **kwargs)


CLOUD_ASR_PLAN_SCHEMA = "video_knowledge_pipeline.cloud_asr_plan.v1"
CLOUD_ASR_RUN_SCHEMA = "video_knowledge_pipeline.cloud_asr_run.v1"
CLOUD_ASR_AUDIO_PREP_SCHEMA = "video_knowledge_pipeline.cloud_asr_audio_prep.v1"


def prepare_cloud_asr_audio(
    media_path: str | Path,
    *,
    output_path: str | Path | None = None,
    bitrate_kbps: int = 32,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    execute: bool = False,
    timeout_seconds: int = 1800,
    receipt_bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare a small speech-only MP3 using VKP's existing FFmpeg front door."""
    media = Path(media_path).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"media not found: {media}")
    if bitrate_kbps < 16 or bitrate_kbps > 128:
        raise ValueError("bitrate_kbps must be between 16 and 128")
    if sample_rate_hz not in {8000, 12000, 16000, 24000, 32000, 44100, 48000}:
        raise ValueError("unsupported sample_rate_hz")
    if channels not in {1, 2}:
        raise ValueError("channels must be 1 or 2")
    target = Path(output_path).expanduser().resolve() if output_path else media.with_name(f"{media.stem}.cloud-asr-{bitrate_kbps}k.mp3")
    ffmpeg = resolve_media_tool("ffmpeg")
    command = [ffmpeg or "ffmpeg", "-y", "-i", str(media), "-vn", "-map_metadata", "-1", "-ac", str(channels), "-ar", str(sample_rate_hz), "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", str(target)]
    source_bytes = media.stat().st_size
    result: dict[str, Any] = {
        "schema": CLOUD_ASR_AUDIO_PREP_SCHEMA,
        "status": "planned",
        "ok": True,
        "execute": bool(execute),
        "source_path": str(media),
        "source_bytes": source_bytes,
        "output_path": str(target),
        "profile": f"speech_mp3_{bitrate_kbps}k_candidate",
        "bitrate_kbps": bitrate_kbps,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "command": command,
        "network_call": False,
        "operator_boundary": {"local_ffmpeg_only": True, "candidate_does_not_replace_source": True, "provider_quality_ab_required_before_default_switch": True},
    }
    if not execute:
        return result
    if not ffmpeg:
        return {**result, "ok": False, "status": "ffmpeg_not_available"}
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=max(30, int(timeout_seconds)), check=False, env=local_tool_subprocess_env())
    if completed.returncode != 0 or not target.is_file():
        return {**result, "ok": False, "status": "ffmpeg_failed", "returncode": completed.returncode, "stderr_tail": completed.stderr[-2000:]}
    output_bytes = target.stat().st_size
    provenance = {
        **result,
        "status": "completed",
        "output_bytes": output_bytes,
        "bytes_saved": max(0, source_bytes - output_bytes),
        "byte_reduction_ratio": round(1 - output_bytes / source_bytes, 4) if source_bytes else 0.0,
        "source_sha256": _file_sha256(media),
        "output_sha256": _file_sha256(target),
        "completed_at": now_iso(),
    }
    provenance_path = target.with_suffix(target.suffix + ".provenance.json")
    provenance["provenance_path"] = str(provenance_path)
    write_json(provenance_path, provenance)
    if receipt_bundle_dir:
        from .local_media_contracts import build_ffmpeg_execution_receipt

        receipt_path = (
            Path(receipt_bundle_dir)
            / "exports/media-execution/ffmpeg-cloud-asr-audio-execution-receipt.json"
        )
        receipt = build_ffmpeg_execution_receipt(
            receipt_bundle_dir,
            operation="transcode",
            ffmpeg_path=ffmpeg,
            actual_argv=command,
            inputs=[{"role": "source_media", "path": media}],
            outputs=[{"role": "cloud_asr_audio_candidate", "path": target}],
            requested_backend="cpu",
            selected_backend="cpu",
            hardware_accelerated=False,
            fallback_used=False,
            allowed_roots=[media.parent, target.parent, Path(ffmpeg).resolve().parent],
            output_path=receipt_path,
            write=True,
        )
        provenance["ffmpeg_execution_receipt"] = {
            "path": str(receipt_path.resolve()),
            "sha256": receipt["receipt_sha256"],
        }
        write_json(provenance_path, provenance)
    return provenance


def plan_cloud_asr_run(
    root: str | Path,
    media_path: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    model: str = "",
    language: str = "zh",
    prompt: str = "",
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    paths = ensure_project_dirs(root_path)
    media = Path(media_path).expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(f"media not found: {media}")

    cfg = dict(provider_config or {})
    if model:
        cfg["model"] = model
    if language:
        cfg["language"] = language
    cfg.setdefault("model", "gpt-4o-transcribe")
    resolved = resolve_asr_provider_config(cfg)
    safe_cfg = _safe_provider_config(resolved)
    run_id = new_id("cloud_asr")
    output_dir = paths["transcripts"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / "raw-asr-output.json"
    plan_path = output_dir / "cloud-asr-plan.json"
    plan = {
        "schema": CLOUD_ASR_PLAN_SCHEMA,
        "run_id": run_id,
        "project": str(paths["root"]),
        "preset": "cloud-asr",
        "provider": "online_asr",
        "media_path": str(media),
        "output_dir": str(output_dir),
        "expected_output_json": str(raw_output),
        "provider_config": safe_cfg,
        "language": language,
        "prompt": prompt,
        "execute": False,
        "upload_required": True,
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
            "execute_required_for_network_call": True,
            "will_upload_audio_only_with_execute": True,
            "provider_config_runtime_only": True,
            "secrets_redacted": True,
            "local_asr_primary": "SenseVoice/FunASR remains the default VKP ASR path; this is an explicit cloud fallback/quality branch.",
        },
        "next_actions": [
            "Review this plan and privacy boundary before upload.",
            "Run run-cloud-asr-plan with --execute only when cloud ASR is approved for this media.",
            "Use plan-whisperx-alignment after local/cloud transcript if word-level timestamps are needed.",
        ],
        "created_at": now_iso(),
    }
    write_json(plan_path, plan)
    return {**plan, "plan_path": str(plan_path)}


def run_cloud_asr_plan(
    plan_json: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    normalize: bool = True,
) -> dict[str, Any]:
    plan_path = Path(plan_json).expanduser().resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("cloud ASR plan must be a JSON object")
    project = Path(str(plan.get("project") or plan_path.parent)).expanduser().resolve()
    media = str(plan.get("media_path") or "")
    output_dir = Path(str(plan.get("output_dir") or plan_path.parent)).expanduser().resolve()
    expected_output = Path(str(plan.get("expected_output_json") or output_dir / "raw-asr-output.json")).expanduser().resolve()
    cfg = dict(plan.get("provider_config") if isinstance(plan.get("provider_config"), dict) else {})
    cfg.update(provider_config or {})
    result: dict[str, Any] = {
        "schema": CLOUD_ASR_RUN_SCHEMA,
        "plan_path": str(plan_path),
        "project": str(project),
        "preset": "cloud-asr",
        "provider": "online_asr",
        "media_path": media,
        "output_dir": str(output_dir),
        "expected_output_json": str(expected_output),
        "execute": bool(execute),
        "normalize": bool(normalize),
        "started_at": now_iso(),
        "status": "preview",
        "cloud_call": {},
        "raw_output_json": str(expected_output) if expected_output.exists() else "",
        "normalized": None,
        "operator_boundary": {
            "preview_first": True,
            "execute_required_for_network_call": True,
            "will_upload_audio_only_with_execute": bool(execute),
            "secrets_redacted": True,
        },
    }
    if not execute:
        result["finished_at"] = now_iso()
        return _write_cloud_asr_log(project, result)

    output_dir.mkdir(parents=True, exist_ok=True)
    call = online_model_api_call(
        "asr",
        provider_config=cfg,
        audio_path=media,
        prompt=str(plan.get("prompt") or ""),
        execute=True,
        output_dir=output_dir,
        write=True,
    )
    result["cloud_call"] = _compact_cloud_call(call)
    if not call.get("ok"):
        result.update({"status": str(call.get("status") or call.get("error") or "cloud_asr_failed"), "finished_at": now_iso()})
        return _write_cloud_asr_log(project, result)
    raw_payload = call.get("raw_response") if call.get("raw_response") is not None else {"text": call.get("content", "")}
    write_json(expected_output, raw_payload)
    result["raw_output_json"] = str(expected_output)
    result["status"] = "ok"
    if normalize:
        try:
            title = Path(media).stem if media else expected_output.stem
            result["normalized"] = normalize_asr_output(project, expected_output, provider="openai", title=title)
        except Exception as exc:  # noqa: BLE001 - surfaced in run report.
            result["status"] = "normalize_failed"
            result["error"] = str(exc)
    result["finished_at"] = now_iso()
    return _write_cloud_asr_log(project, result)


def cloud_asr_run_log(root: str | Path) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "cloud-asr-runs.jsonl"
    markdown_path = paths["notes"] / "cloud-asr-runs.md"
    rows = read_jsonl(log_path)
    markdown_path.write_text(_render_cloud_asr_log_markdown(root, rows), encoding="utf-8")
    return {"project": str(root), "log_path": str(log_path), "markdown_path": str(markdown_path), "count": len(rows), "commands": rows, "last": rows[-1] if rows else {}}


def _safe_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if key != "api_key"}


def _compact_cloud_call(call: dict[str, Any]) -> dict[str, Any]:
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


def _write_cloud_asr_log(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "cloud-asr-runs.jsonl"
    record = {
        "created_at": now_iso(),
        "preset": "cloud-asr",
        "provider": "online_asr",
        "execute": bool(result.get("execute")),
        "normalize": bool(result.get("normalize")),
        "status": result.get("status", ""),
        "raw_output_json": result.get("raw_output_json", ""),
        "normalized_json": ((result.get("normalized") or {}).get("json_path") if isinstance(result.get("normalized"), dict) else ""),
        "error": result.get("error", "") or ((result.get("cloud_call") or {}).get("error") if isinstance(result.get("cloud_call"), dict) else ""),
    }
    append_jsonl(log_path, [record])
    log = cloud_asr_run_log(root)
    report = _write_cloud_asr_report(result, record)
    result["cloud_asr_log"] = {"record": record, "log_path": log["log_path"], "markdown_path": log["markdown_path"], "report_path": str(report), "count": log["count"]}
    registry = _register_cloud_run_if_bundle(root, result)
    if registry:
        result["run_registry"] = registry
    return result


def _write_cloud_asr_report(result: dict[str, Any], record: dict[str, Any]) -> Path:
    output_dir = Path(str(result.get("output_dir") or ".")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cloud-asr-run-report.md"
    lines = [
        "# Cloud ASR Run Report",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Execute: `{bool(result.get('execute'))}`",
        f"- Upload required: `{bool(result.get('execute'))}`",
        f"- Raw output: `{result.get('raw_output_json', '')}`",
        f"- Normalized JSON: `{record.get('normalized_json', '')}`",
        f"- Started at: `{result.get('started_at', '')}`",
        f"- Finished at: `{result.get('finished_at', '')}`",
        "",
        "## Boundary",
        "",
        "Cloud ASR is optional. SenseVoice/FunASR remains the local default. This command uploads audio only when `execute=true` / `--execute` is used.",
        "",
        "## Error",
        "",
        "```text",
        str(record.get("error") or ""),
        "```",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _render_cloud_asr_log_markdown(root: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Cloud ASR Runs",
        "",
        f"- Project: `{root}`",
        f"- Count: {len(rows)}",
        "- Boundary: cloud ASR uploads audio only on explicit execution; local SenseVoice/FunASR remains the default.",
        "",
        "| Time | Execute | Normalize | Status | Output |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        execute = "yes" if row.get("execute") else "no"
        normalize = "yes" if row.get("normalize") else "no"
        lines.append(f"| {row.get('created_at', '')} | {execute} | {normalize} | {row.get('status', '')} | `{row.get('normalized_json') or row.get('raw_output_json') or ''}` |")
    return "\n".join(lines).rstrip() + "\n"


def _register_cloud_run_if_bundle(root: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    if not (root.expanduser().resolve() / "manifest.json").exists():
        return None
    status = "completed" if result.get("status") == "ok" else "needs_execution" if result.get("status") == "preview" else "needs_retry"
    return register_bundle_run(
        root,
        run_type="cloud_asr_run",
        run_id="cloud-asr-run",
        status=status,
        title="Cloud ASR optional run",
        summary=f"execute={bool(result.get('execute'))}; status={result.get('status', '')}; local ASR remains primary.",
        inputs={"plan_path": result.get("plan_path", ""), "media_path": result.get("media_path", "")},
        parameters={"execute": bool(result.get("execute")), "normalize": bool(result.get("normalize"))},
        artifacts=[
            {"key": "run_report", "path": (result.get("cloud_asr_log") or {}).get("report_path", "") if isinstance(result.get("cloud_asr_log"), dict) else ""},
            {"key": "raw_output_json", "path": result.get("raw_output_json", "")},
            {"key": "normalized_transcript_json", "path": ((result.get("normalized") or {}).get("json_path") if isinstance(result.get("normalized"), dict) else "")},
        ],
        failed_items=[] if status in {"completed", "needs_execution"} else [{"item": "cloud_asr_run", "reason": result.get("status", ""), "detail": result.get("error", "")}],
        retry_command=f".\\scripts\\video-knowledge.ps1 run-cloud-asr-plan '{result.get('plan_path', '')}' --execute",
        next_actions=["Compare cloud transcript with local SenseVoice/FunASR before accepting replacements.", "Run WhisperX alignment if word-level timestamps are needed."],
        operator_boundary={"cloud_call": True, "execute_required": True, "local_asr_primary": True, "secrets_redacted": True},
        write=True,
    )
