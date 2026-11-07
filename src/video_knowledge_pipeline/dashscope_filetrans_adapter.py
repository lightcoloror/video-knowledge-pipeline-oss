from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .model_runtime_client import consume_consented_remote_runtime_grant
from .path_defaults import source_reviews_root


SCHEMA = "video_knowledge_pipeline.dashscope_filetrans_adapter.v1"
UPSTREAM_COMMIT = "949bc84058cdae1d9c021c50203e6d2742f9392c"
UPSTREAM_SCRIPT = "generate_subtitle_qwen_api.py"
DEFAULT_UPSTREAM_ROOT = Path(
    os.environ.get("VKP_MOYS_ASR_WORKFLOW_ROOT")
    or source_reviews_root() / "moys-asr-workflow-20260810"
)
SUPPORTED_MODELS = frozenset(
    {
        "qwen-audio-3.0-asr-flash-filetrans",
        "qwen3-asr-flash-filetrans",
        "fun-asr",
    }
)


def dashscope_filetrans_adapter_status(
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the fixed upstream CLI without importing or copying its provider code."""

    root = Path(source_root or DEFAULT_UPSTREAM_ROOT).expanduser().resolve()
    script = root / UPSTREAM_SCRIPT
    commit = ""
    error = ""
    if script.is_file():
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            if completed.returncode == 0:
                commit = str(completed.stdout or "").strip()
            else:
                error = "upstream_commit_unavailable"
        except (OSError, subprocess.SubprocessError):
            error = "upstream_commit_unavailable"
    else:
        error = "upstream_script_missing"
    ready = script.is_file() and commit == UPSTREAM_COMMIT
    if script.is_file() and commit and commit != UPSTREAM_COMMIT:
        error = "upstream_commit_mismatch"
    return {
        "schema": SCHEMA,
        "ready": ready,
        "source_root": str(root),
        "script_path": str(script),
        "expected_commit": UPSTREAM_COMMIT,
        "actual_commit": commit,
        "error": error,
        "provider_calls_made": False,
        "secrets_accessed": False,
        "reuse": "moys-asr-workflow fixed CLI; no DashScope protocol reimplementation",
    }


def build_dashscope_filetrans_plan(
    *,
    provider_config: dict[str, Any],
    audio_path: str | Path,
    output_srt: str | Path,
    prompt: str = "",
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    cfg = dict(provider_config or {})
    options = (
        dict(cfg.get("provider_options") or {})
        if isinstance(cfg.get("provider_options"), dict)
        else {}
    )
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise ValueError(f"audio_not_found: {audio}")
    model = str(cfg.get("model") or "").strip()
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported DashScope filetrans model: {model!r}")
    region = str(options.get("region") or "beijing").strip().lower()
    if region not in {"beijing", "singapore"}:
        raise ValueError("DashScope region must be beijing or singapore")
    base_url = str(cfg.get("base_url") or "").strip().rstrip("/")
    workspace_id = str(options.get("workspace_id") or "").strip()
    if not workspace_id:
        workspace_id = _workspace_id_from_base_url(base_url, region=region)
    if region == "singapore" and not workspace_id:
        raise ValueError("DashScope Singapore requires workspace_id")
    language = str(options.get("language") or "yue").strip().lower()
    if not language:
        language = "yue"
    speaker = bool(options.get("speaker_diarization", True))
    poll_interval = int(options.get("poll_interval_seconds") or 5)
    poll_timeout = int(options.get("poll_timeout_seconds") or 1800)
    if poll_interval < 1 or poll_interval > 60:
        raise ValueError("poll_interval_seconds must be between 1 and 60")
    if poll_timeout < 60 or poll_timeout > 7200:
        raise ValueError("poll_timeout_seconds must be between 60 and 7200")
    output = Path(output_srt).expanduser().resolve()
    root = Path(source_root or DEFAULT_UPSTREAM_ROOT).expanduser().resolve()
    command = [
        sys.executable,
        str(root / UPSTREAM_SCRIPT),
        str(audio),
        "--output",
        str(output),
        "--model",
        model,
        "--language",
        language,
        "--region",
        region,
        "--json",
        "--no-html",
    ]
    if speaker:
        command.append("--speaker")
    context = str(prompt or "").strip()[:400]
    if context:
        command.extend(["--context", context])
    return {
        "schema": SCHEMA,
        "provider": "dashscope_filetrans",
        "model": model,
        "base_url": base_url,
        "destination": str(urlsplit(base_url).hostname or ""),
        "audio_path": str(audio),
        "output_srt": str(output),
        "output_project": str(output.with_suffix(".mosp")),
        "command": command,
        "environment": {
            "DASHSCOPE_REGION": region,
            "DASHSCOPE_WORKSPACE_ID": workspace_id,
            "DASHSCOPE_DEFAULT_LANGUAGE": language,
            "DASHSCOPE_POLL_INTERVAL": str(poll_interval),
            "DASHSCOPE_POLL_TIMEOUT": str(poll_timeout),
        },
        "timeout_seconds": poll_timeout + 120,
        "speaker_diarization": speaker,
        "context_chars": len(context),
        "api_key_in_command": False,
        "automatic_fallback": False,
        "source_root": str(root),
        "upstream_commit": UPSTREAM_COMMIT,
    }


def call_dashscope_filetrans_asr(
    *,
    provider_config: dict[str, Any],
    audio_path: str,
    prompt: str = "",
    source_root: str | Path | None = None,
    _runner: Callable[..., Any] = subprocess.run,
    _source_status: dict[str, Any] | None = None,
    _grant_consumer: Callable[..., str] = consume_consented_remote_runtime_grant,
) -> dict[str, Any]:
    """Execute the fixed upstream CLI only inside a Broker reservation."""

    cfg = dict(provider_config or {})
    grant_error = _grant_consumer(
        consent_id=str(cfg.get("consent_id") or ""),
        route_revision=str(cfg.get("route_revision") or ""),
    )
    if grant_error:
        return {
            "ok": False,
            "status": "consent_required",
            "error": grant_error,
            "content": "",
            "remote_requests_made": False,
            "automatic_fallback": False,
        }
    api_key = str(cfg.get("api_key") or "").strip()
    if not api_key:
        return {
            "ok": False,
            "status": "missing_api_key",
            "error": "missing_api_key",
            "content": "",
            "remote_requests_made": False,
            "automatic_fallback": False,
        }
    status = dict(_source_status or dashscope_filetrans_adapter_status(source_root))
    if not status.get("ready"):
        return {
            "ok": False,
            "status": str(status.get("error") or "adapter_unavailable"),
            "error": str(
                status.get("error") or "dashscope_filetrans_adapter_unavailable"
            ),
            "content": "",
            "adapter_status": status,
            "remote_requests_made": False,
            "automatic_fallback": False,
        }
    try:
        with tempfile.TemporaryDirectory(prefix="vkp-dashscope-filetrans-") as temp_dir:
            output_srt = Path(temp_dir) / "transcript.srt"
            plan = build_dashscope_filetrans_plan(
                provider_config=cfg,
                audio_path=audio_path,
                output_srt=output_srt,
                prompt=prompt,
                source_root=status["source_root"],
            )
            environment = dict(os.environ)
            environment.update(plan["environment"])
            environment["DASHSCOPE_API_KEY"] = api_key
            completed = _runner(
                plan["command"],
                cwd=str(status["source_root"]),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=int(plan["timeout_seconds"]),
                check=False,
            )
            if int(completed.returncode) != 0:
                return {
                    "ok": False,
                    "status": "provider_failed",
                    "error": f"dashscope_filetrans_exit_{int(completed.returncode)}",
                    "content": "",
                    "provider_detail": _redact_detail(
                        str(completed.stderr or completed.stdout or ""), api_key
                    ),
                    "adapter_status": status,
                    "remote_requests_made": True,
                    "automatic_fallback": False,
                }
            project_path = Path(plan["output_project"])
            if not project_path.is_file():
                return {
                    "ok": False,
                    "status": "output_missing",
                    "error": "dashscope_filetrans_output_missing",
                    "content": "",
                    "adapter_status": status,
                    "remote_requests_made": True,
                    "automatic_fallback": False,
                }
            project = json.loads(project_path.read_text(encoding="utf-8"))
            payload = _normalise_project(project, model=str(plan["model"]))
            return {
                "ok": True,
                "status": "completed",
                "error": "",
                "content": payload["text"],
                "raw_response": payload,
                "adapter_status": status,
                "remote_requests_made": True,
                "automatic_fallback": False,
                "execution": {
                    "provider": "dashscope_filetrans",
                    "model": plan["model"],
                    "destination": plan["destination"],
                    "speaker_diarization": plan["speaker_diarization"],
                    "automatic_fallback": False,
                    "upstream_commit": UPSTREAM_COMMIT,
                },
            }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "status": "timeout",
            "error": "dashscope_filetrans_timeout",
            "content": "",
            "remote_requests_made": True,
            "automatic_fallback": False,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": "adapter_error",
            "error": f"dashscope_filetrans_adapter_error: {type(exc).__name__}: {exc}",
            "content": "",
            "remote_requests_made": False,
            "automatic_fallback": False,
        }


def _workspace_id_from_base_url(base_url: str, *, region: str) -> str:
    host = str(urlsplit(base_url).hostname or "").lower()
    suffix = (
        ".ap-southeast-1.maas.aliyuncs.com"
        if region == "singapore"
        else ".cn-beijing.maas.aliyuncs.com"
    )
    return host[: -len(suffix)] if host.endswith(suffix) else ""


def _normalise_project(project: Any, *, model: str) -> dict[str, Any]:
    if not isinstance(project, dict):
        raise ValueError("DashScope .mosp output must be an object")
    raw_segments = project.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("DashScope .mosp output is missing segments")
    segments: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError("DashScope segment must be an object")
        text = str(raw.get("text") or "").strip()
        start = float(raw.get("start") or 0)
        end = float(raw.get("end") or 0)
        if not text or end < start:
            raise ValueError(f"invalid DashScope segment at index {index}")
        row: dict[str, Any] = {
            "id": f"dashscope-{index + 1:06d}",
            "start": start,
            "end": end,
            "text": text,
            "words": list(raw.get("items") or []),
        }
        if raw.get("speaker") is not None:
            row["speaker"] = str(raw.get("speaker"))
        segments.append(row)
    return {
        "text": "".join(row["text"] for row in segments),
        "language": str(project.get("language") or ""),
        "model": model,
        "segments": segments,
        "provider": "dashscope_filetrans",
        "source_schema": str(project.get("schema") or "moys_asr_workflow.mosp"),
    }


def _redact_detail(value: str, api_key: str) -> str:
    text = str(value or "")
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return text[-2000:]
