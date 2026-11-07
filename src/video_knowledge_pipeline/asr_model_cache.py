from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .asr_runner import _resolve_python_executable
from .funasr_python_runner import (
    FUNASR_MODELSCOPE_ALIASES,
    _local_model_candidates,
    _resolve_local_model,
)
from .media_tools import local_tool_subprocess_env
from .models import now_iso
from .storage import ensure_project_dirs, write_json

SCHEMA = "video_knowledge_pipeline.asr_model_cache.v1"
DEFAULT_MODELS = ["iic/SenseVoiceSmall", "fsmn-vad", "ct-punc"]
OPTIONAL_MODELS = ["cam++"]


def asr_model_cache_status(
    root: str | Path,
    *,
    models: list[str] | None = None,
    include_optional: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    selected = _selected_models(models=models, include_optional=include_optional)
    rows = [_model_status(model) for model in selected]
    result = {
        "schema": SCHEMA,
        "root": str(root_path),
        "models": rows,
        "ready": all(row.get("ready") for row in rows),
        "missing": [row for row in rows if not row.get("ready")],
        "python_executable": _resolve_python_executable(),
        "operator_boundary": {
            "status_only": True,
            "does_not_download": True,
            "prepare_requires_execute_and_allow_download": True,
        },
        "source_policy": {
            "hub": "modelscope",
            "china_accessible": True,
            "uses_funasr_native_downloader": True,
            "arbitrary_download_url": False,
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_status(root_path, result)
    return result


def prepare_asr_model_cache(
    root: str | Path,
    *,
    models: list[str] | None = None,
    include_optional: bool = False,
    execute: bool = False,
    allow_download: bool = False,
    device: str = "auto",
    timeout_seconds: int = 1800,
    write: bool = True,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    status = asr_model_cache_status(root_path, models=models, include_optional=include_optional, write=False)
    selected = _selected_models(models=models, include_optional=include_optional)
    punc_model = "ct-punc" if "ct-punc" in selected else ""
    spk_model = "cam++" if "cam++" in selected else ""
    vad_model = "fsmn-vad" if "fsmn-vad" in selected else ""
    primary_model = next((model for model in selected if model not in {"fsmn-vad", "ct-punc", "cam++"}), "iic/SenseVoiceSmall")
    result: dict[str, Any] = {
        "schema": "video_knowledge_pipeline.asr_model_cache_prepare.v1",
        "root": str(root_path),
        "execute": bool(execute),
        "allow_download": bool(allow_download),
        "device": device,
        "before": status,
        "status": "preview",
        "command": _prepare_command(model=primary_model, vad_model=vad_model, punc_model=punc_model, spk_model=spk_model, device=device),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "after": {},
        "operator_boundary": {
            "may_download_models": bool(execute and allow_download),
            "requires_execute": True,
            "requires_allow_download": True,
            "audio_not_uploaded": True,
            "network_access": (
                "modelscope_download_allowed"
                if execute and allow_download
                else "disabled"
            ),
        },
        "source_policy": status["source_policy"],
        "updated_at": now_iso(),
    }
    if not execute:
        return _write_prepare(root_path, result, write=write)
    if status.get("ready"):
        result["status"] = "already_ready"
        result["after"] = status
        return _write_prepare(root_path, result, write=write)
    if not allow_download and not _allow_model_download_env():
        result["status"] = "download_not_allowed"
        result["stderr"] = "Missing models exist; pass --allow-download or set LECTURE_ASR_ALLOW_MODEL_DOWNLOAD=1 to allow first-run model download."
        return _write_prepare(root_path, result, write=write)
    env = local_tool_subprocess_env()
    env["LECTURE_ASR_ALLOW_MODEL_DOWNLOAD"] = "1"
    pythonpath = str(Path(__file__).resolve().parents[1])
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pythonpath + (f"{os.pathsep}{current}" if current else "")
    root_path.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            result["command"],
            cwd=str(root_path),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=int(timeout_seconds or 0) or None,
            check=False,
            env=env,
        )
        result.update({"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    except FileNotFoundError as exc:
        result.update({"status": "command_not_found", "stderr": str(exc)})
        return _write_prepare(root_path, result, write=write)
    except subprocess.TimeoutExpired as exc:
        result.update({"status": "timeout", "stdout": str(exc.output or ""), "stderr": str(exc.stderr or "")})
        return _write_prepare(root_path, result, write=write)
    after = asr_model_cache_status(root_path, models=models, include_optional=include_optional, write=False)
    result["after"] = after
    result["status"] = "ready" if after.get("ready") else "prepare_failed"
    return _write_prepare(root_path, result, write=write)


def _selected_models(*, models: list[str] | None, include_optional: bool) -> list[str]:
    if models:
        return [str(model).strip() for model in models if str(model).strip()]
    selected = list(DEFAULT_MODELS)
    if include_optional:
        selected.extend(OPTIONAL_MODELS)
    return selected


def _model_status(model: str) -> dict[str, Any]:
    resolved = _resolve_local_model(model)
    ready = bool(resolved and resolved != model and Path(resolved).exists()) or Path(str(model)).expanduser().exists()
    model_ids = list(FUNASR_MODELSCOPE_ALIASES.get(model, (model,)))
    return {
        "model": model,
        "hub": "modelscope",
        "official_model_ids": model_ids,
        "resolved": resolved,
        "ready": ready,
        "status": "ready" if ready else "missing_or_not_downloaded",
        "candidate_paths": [str(path) for path in _local_model_candidates(model)],
    }


def _prepare_command(*, model: str, vad_model: str, punc_model: str, spk_model: str, device: str) -> list[str]:
    command = [
        _resolve_python_executable(),
        "-m",
        "video_knowledge_pipeline.funasr_model_cache_prepare",
        "--model",
        model,
        "--device",
        device,
    ]
    if vad_model:
        command.extend(["--vad-model", vad_model])
    if punc_model:
        command.extend(["--punc-model", punc_model])
    if spk_model:
        command.extend(["--spk-model", spk_model])
    return command


def _allow_model_download_env() -> bool:
    return os.environ.get("LECTURE_ASR_ALLOW_MODEL_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def _write_status(root: Path, result: dict[str, Any]) -> None:
    paths = ensure_project_dirs(root)
    write_json(paths["notes"] / "asr-model-cache-status.json", result)
    (paths["notes"] / "asr-model-cache-status.md").write_text(_render_status_markdown(result), encoding="utf-8")


def _write_prepare(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    if write:
        paths = ensure_project_dirs(root)
        write_json(paths["notes"] / "asr-model-cache-prepare.json", result)
        (paths["notes"] / "asr-model-cache-prepare.md").write_text(_render_prepare_markdown(result), encoding="utf-8")
    return result


def _render_status_markdown(result: dict[str, Any]) -> str:
    lines = ["# ASR Model Cache Status", "", f"- Ready: `{bool(result.get('ready'))}`", f"- Python: `{result.get('python_executable', '')}`", "", "| Model | Ready | Resolved |", "| --- | --- | --- |"]
    for row in result.get("models") or []:
        lines.append(f"| `{row.get('model', '')}` | `{bool(row.get('ready'))}` | `{row.get('resolved', '')}` |")
    return "\n".join(lines).rstrip() + "\n"


def _render_prepare_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASR Model Cache Prepare",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Execute: `{bool(result.get('execute'))}`",
        f"- Allow download: `{bool(result.get('allow_download'))}`",
        f"- Return code: `{result.get('returncode')}`",
        "",
        "## Command",
        "",
        "```text",
        " ".join(str(part) for part in result.get("command") or []),
        "```",
        "",
        "## Stderr Tail",
        "",
        "```text",
        str(result.get("stderr") or "")[-4000:],
        "```",
    ]
    return "\n".join(lines).rstrip() + "\n"
