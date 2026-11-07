from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .asr_ab_plan import plan_asr_ab_sample
from .asr_adapter import normalize_asr_output
from .asr_execution import run_asr_plan
from .cloud_asr import plan_cloud_asr_run, run_cloud_asr_plan
from .funasr_python_runner import _resolve_local_model
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import now_iso
from .asr_runner import plan_asr_run
from .storage import read_json, write_json
from .whisperx_alignment import run_whisperx_alignment

SCHEMA = "video_knowledge_pipeline.asr_ab_sample_run.v1"


def run_asr_ab_sample(
    workspace_dir: str | Path,
    media_path: str | Path = "",
    *,
    plan_json: str | Path = "",
    sample_start_seconds: float = 0.0,
    duration_seconds: float = 300.0,
    language: str = "zh",
    execute_sample: bool = False,
    execute_local: bool = False,
    execute_cloud: bool = False,
    cloud_provider_config: dict[str, Any] | None = None,
    variants: list[str] | None = None,
    timeout_seconds: int = 1800,
    write: bool = True,
) -> dict[str, Any]:
    """Run a bounded ASR A/B sample without promoting any transcript.

    Local SenseVoice variants may run with execute_local. Cloud ASR uploads only
    the extracted sample and only when execute_cloud=True.
    """

    if plan_json:
        plan_path = Path(plan_json).expanduser().resolve()
        plan = read_json(plan_path)
        if not isinstance(plan, dict):
            raise ValueError("ASR A/B sample plan must be a JSON object")
    else:
        if not media_path:
            raise ValueError("media_path is required when plan_json is not provided")
        plan = plan_asr_ab_sample(
            workspace_dir,
            media_path,
            sample_start_seconds=sample_start_seconds,
            duration_seconds=duration_seconds,
            language=language,
            cloud_provider_config=cloud_provider_config,
            write=True,
        )
        plan_path = Path(str(plan.get("artifacts", {}).get("json") or "")).expanduser().resolve()
    sample_path = Path(str(plan.get("sample_media_path") or "")).expanduser().resolve()
    sample_result = _run_sample_extract(plan, execute=execute_sample, timeout_seconds=timeout_seconds)
    selected = {str(value).strip() for value in (variants or []) if str(value).strip()}
    rows = []
    for variant in plan.get("variants") or []:
        if not isinstance(variant, dict):
            continue
        key = str(variant.get("key") or "")
        if selected and key not in selected:
            continue
        rows.append(_run_variant(plan, variant, sample_path=sample_path, execute_local=execute_local, execute_cloud=execute_cloud, cloud_provider_config=cloud_provider_config, timeout_seconds=timeout_seconds))
    output_dir = sample_path.parent if sample_path.name else Path(str(plan_path)).parent
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "workspace_dir": str(Path(str(plan.get("workspace_dir") or workspace_dir)).expanduser().resolve()),
        "plan_path": str(plan_path) if plan_path else "",
        "sample_media_path": str(sample_path),
        "sample_extract": sample_result,
        "execute_sample": bool(execute_sample),
        "execute_local": bool(execute_local),
        "execute_cloud": bool(execute_cloud),
        "variants": rows,
        "operator_boundary": {
            "does_not_promote_any_transcript": True,
            "cloud_upload_sample_only_when_execute_cloud": bool(execute_cloud),
            "same_sample_required_for_fair_comparison": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "asr-ab-sample-run.json"
        result = _merge_previous_run(result, json_path=json_path, selected=selected, sample_result=sample_result)
        result["status"] = _overall_status(sample_result, [row for row in result.get("variants") or [] if isinstance(row, dict)])
        md_path = output_dir / "asr-ab-sample-run.md"
        write_json(json_path, result)
        md_path.write_text(_render_markdown(result), encoding="utf-8")
        result["json_path"] = str(json_path)
        result["markdown_path"] = str(md_path)
    else:
        result["status"] = _overall_status(sample_result, rows)
    return result


def _merge_previous_run(result: dict[str, Any], *, json_path: Path, selected: set[str], sample_result: dict[str, Any]) -> dict[str, Any]:
    if not selected or not json_path.exists():
        return result
    try:
        previous = read_json(json_path)
    except Exception:
        return result
    if not isinstance(previous, dict):
        return result
    previous_rows = [row for row in (previous.get("variants") or []) if isinstance(row, dict) and str(row.get("key") or "")]
    current_rows = [row for row in (result.get("variants") or []) if isinstance(row, dict) and str(row.get("key") or "")]
    merged: dict[str, dict[str, Any]] = {str(row.get("key")): row for row in previous_rows}
    for row in current_rows:
        merged[str(row.get("key"))] = row
    order = [
        "sensevoice_basic",
        "sensevoice_full_punc",
        "sensevoice_full_punc_campp",
        "sensevoice_full_punc_campp_oracle_2",
        "moss_transcribe_diarize",
        "dolphin",
        "whisperx_alignment",
        "openai_cloud_asr",
    ]
    rows = [merged.pop(key) for key in order if key in merged]
    rows.extend(merged.values())
    output = dict(result)
    output["variants"] = rows
    output["merged_previous_variants"] = True
    if not output.get("sample_extract"):
        output["sample_extract"] = sample_result
    return output


def _run_sample_extract(plan: dict[str, Any], *, execute: bool, timeout_seconds: int) -> dict[str, Any]:
    command = plan.get("sample_extract_command") if isinstance(plan.get("sample_extract_command"), list) else []
    sample_path = Path(str(plan.get("sample_media_path") or "")).expanduser().resolve()
    result: dict[str, Any] = {"execute": bool(execute), "command": command, "sample_media_path": str(sample_path), "status": "exists" if sample_path.exists() else "preview", "returncode": None}
    if not execute:
        return result
    if not command:
        result["status"] = "missing_command"
        return result
    try:
        completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=int(timeout_seconds or 0) or None, check=False, env=_project_python_env())
    except FileNotFoundError as exc:
        result.update({"status": "command_not_found", "stderr": str(exc)})
        return result
    except subprocess.TimeoutExpired as exc:
        result.update({"status": "timeout", "stdout": str(exc.output or ""), "stderr": str(exc.stderr or "")})
        return result
    result.update({"returncode": completed.returncode, "stdout_tail": _tail(completed.stdout), "stderr_tail": _tail(completed.stderr), "status": "ok" if completed.returncode == 0 and sample_path.exists() else "failed"})
    return result


def _run_variant(
    plan: dict[str, Any],
    variant: dict[str, Any],
    *,
    sample_path: Path,
    execute_local: bool,
    execute_cloud: bool,
    cloud_provider_config: dict[str, Any] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    key = str(variant.get("key") or "")
    if key.startswith("sensevoice"):
        return _run_local_variant(plan, variant, execute=execute_local, timeout_seconds=timeout_seconds)
    if key == "moss_transcribe_diarize":
        return _run_moss_variant(
            plan,
            variant,
            sample_path=sample_path,
            execute=execute_local,
            timeout_seconds=timeout_seconds,
        )
    if key == "dolphin":
        return _run_dolphin_variant(plan, variant, execute=execute_local, timeout_seconds=timeout_seconds)
    if key == "whisperx_alignment":
        return _run_whisperx_alignment_variant(plan, variant, sample_path=sample_path, execute=execute_local, timeout_seconds=timeout_seconds)
    if key == "openai_cloud_asr":
        return _run_cloud_variant(plan, variant, sample_path=sample_path, execute=execute_cloud, provider_config=cloud_provider_config, timeout_seconds=timeout_seconds)
    return {"key": key, "status": str(variant.get("status") or "not_runnable"), "execute": False, "reason": "adapter_not_implemented_or_not_selected", "metrics": {}}


def _run_local_variant(plan: dict[str, Any], variant: dict[str, Any], *, execute: bool, timeout_seconds: int) -> dict[str, Any]:
    key = str(variant.get("key") or "")
    output_path = Path(str(variant.get("expected_output_json") or "")).expanduser().resolve()
    run_dir = output_path.parent / key
    run_dir.mkdir(parents=True, exist_ok=True)
    adjunct = _adjunct_model_readiness(variant)
    operator_boundary = (
        dict(variant.get("operator_boundary"))
        if isinstance(variant.get("operator_boundary"), dict)
        else {}
    )
    if execute and not adjunct["ready"] and not _allow_model_download():
        return {
            "key": key,
            "status": "asr_model_not_ready",
            "execute": bool(execute),
            "reason": "adjunct_model_not_ready",
            "model_ready": adjunct,
            "raw_output_json": "",
            "normalized_json": "",
            "normalized_srt": "",
            "metrics": {},
            "operator_boundary": operator_boundary,
        }
    variant_plan = {
        "project": str(Path(str(plan.get("workspace_dir") or output_path.parents[2])).expanduser().resolve()),
        "preset": "sensevoice",
        "provider": "sensevoice",
        "media_path": str(plan.get("sample_media_path") or ""),
        "output_dir": str(run_dir),
        "expected_output_json": str(run_dir / "raw-asr-output.json"),
        "command": [str(part) for part in (variant.get("command") or [])],
        "available": True,
        "runner": "funasr_python",
        "model_ready": {"ready": True, "status": "not_checked_by_ab_runner"},
        "pythonpath": str(Path(__file__).resolve().parents[1]),
    }
    if "--output" in variant_plan["command"]:
        index = variant_plan["command"].index("--output") + 1
        if index < len(variant_plan["command"]):
            variant_plan["command"][index] = variant_plan["expected_output_json"]
    plan_path = run_dir / "asr-run-plan.json"
    write_json(plan_path, variant_plan)
    run = run_asr_plan(plan_path, execute=execute, timeout_seconds=timeout_seconds)
    normalized = run.get("normalized") if isinstance(run.get("normalized"), dict) else {}
    return {
        "key": key,
        "status": str(run.get("status") or "preview"),
        "execute": bool(execute),
        "plan_path": str(plan_path),
        "raw_output_json": run.get("raw_output_json", ""),
        "normalized_json": normalized.get("json_path", ""),
        "normalized_srt": normalized.get("srt_path", ""),
        "metrics": _transcript_metrics(normalized.get("json_path", "")),
        "model_ready": adjunct,
        "operator_boundary": operator_boundary,
    }


def _run_moss_variant(
    plan: dict[str, Any],
    variant: dict[str, Any],
    *,
    sample_path: Path,
    execute: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the pinned MOSS candidate through VKP's existing ASR front door.

    Intent: make the already-reviewed MOSS speaker-attributed ASR directly
    comparable with CAM++ on the same exact local sample.
    Decision: call ``plan_asr_run`` and ``run_asr_plan`` rather than rebuilding
    the ``mtd-subtitle`` command, model gate, normalizer, or execution log.
    Reason: one ASR front door preserves fail-closed runtime/model readiness and
    proves that a MOSS failure cannot silently select another provider.
    Evidence: OpenMOSS commit ``eda4b9f`` owns the CLI and raw segment parser;
    VKP's existing preset already preserves its start/end/text/speaker fields.
    Effective scope: the candidate workspace under this A/B sample only; no
    model download, remote call, transcript promotion, fallback, or role guess.
    """

    key = str(variant.get("key") or "moss_transcribe_diarize")
    operator_boundary = (
        dict(variant.get("operator_boundary"))
        if isinstance(variant.get("operator_boundary"), dict)
        else {}
    )
    base: dict[str, Any] = {
        "key": key,
        "provider": "moss-transcribe-diarize",
        "execute": bool(execute),
        "plan_path": "",
        "raw_output_json": "",
        "normalized_json": "",
        "normalized_srt": "",
        "metrics": {},
        "operator_boundary": operator_boundary,
    }
    if not sample_path.exists():
        return {
            **base,
            "status": "sample_media_missing" if execute else "preview",
            "reason": "sample_media_not_extracted",
        }

    candidate_workspace = Path(
        str(
            variant.get("candidate_workspace")
            or sample_path.parent / key / "workspace"
        )
    ).expanduser().resolve()
    candidate_workspace.mkdir(parents=True, exist_ok=True)
    moss_plan = plan_asr_run(
        candidate_workspace,
        sample_path,
        preset=str(variant.get("preset") or "moss-transcribe-diarize"),
        language=str(variant.get("language") or "zh"),
        model=str(variant.get("model") or "") or None,
    )
    availability = (
        dict(moss_plan.get("availability"))
        if isinstance(moss_plan.get("availability"), dict)
        else {}
    )
    model_ready = (
        dict(moss_plan.get("model_ready"))
        if isinstance(moss_plan.get("model_ready"), dict)
        else {}
    )
    base.update(
        {
            "plan_path": str(moss_plan.get("plan_path") or ""),
            "runtime_ready": bool(moss_plan.get("available")),
            "availability": availability,
            "model_ready": model_ready,
            "candidate_workspace": str(candidate_workspace),
        }
    )
    if not execute:
        return {**base, "status": "preview"}

    run = run_asr_plan(
        str(moss_plan.get("plan_path") or ""),
        execute=True,
        timeout_seconds=timeout_seconds,
    )
    normalized = run.get("normalized") if isinstance(run.get("normalized"), dict) else {}
    normalized_json = str(normalized.get("json_path") or "")
    return {
        **base,
        "status": str(run.get("status") or "failed"),
        "returncode": run.get("returncode"),
        "stdout_tail": _tail(str(run.get("stdout") or "")),
        "stderr_tail": _tail(str(run.get("stderr") or "")),
        "raw_output_json": str(run.get("raw_output_json") or ""),
        "normalized_json": normalized_json,
        "normalized_srt": str(normalized.get("srt_path") or ""),
        "metrics": _transcript_metrics(normalized_json),
    }


def _run_dolphin_variant(plan: dict[str, Any], variant: dict[str, Any], *, execute: bool, timeout_seconds: int) -> dict[str, Any]:
    key = str(variant.get("key") or "dolphin")
    output_path = Path(str(variant.get("expected_output_json") or "")).expanduser().resolve()
    run_dir = output_path.parent / key
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(part) for part in (variant.get("command") or [])]
    if "--output" in command:
        index = command.index("--output") + 1
        if index < len(command):
            command[index] = str(run_dir / "raw-asr-output.json")
    raw_output = Path(command[command.index("--output") + 1]).expanduser().resolve() if "--output" in command else run_dir / "raw-asr-output.json"
    base = {
        "key": key,
        "execute": bool(execute),
        "plan_path": "",
        "raw_output_json": str(raw_output) if raw_output.exists() else "",
        "normalized_json": "",
        "normalized_srt": "",
        "metrics": {},
        "operator_boundary": {"does_not_promote_any_transcript": True, "second_evidence_source_only": True},
    }
    plan_path = run_dir / "dolphin-run-plan.json"
    write_json(plan_path, {
        "schema": "video_knowledge_pipeline.dolphin_run_plan.v1",
        "project": str(Path(str(plan.get("workspace_dir") or output_path.parents[2])).expanduser().resolve()),
        "provider": "dolphin",
        "media_path": str(plan.get("sample_media_path") or ""),
        "output_dir": str(run_dir),
        "expected_output_json": str(raw_output),
        "command": command,
        "execute": False,
        "operator_boundary": base["operator_boundary"],
    })
    base["plan_path"] = str(plan_path)
    if not execute:
        return {**base, "status": "preview"}
    module = _python_module_ready(command[0] if command else "", "dolphin")
    if not module["ready"]:
        return {**base, "status": "asr_module_not_ready", "reason": "dolphin_python_package_missing", "module_ready": module}
    sample_media = Path(str(plan.get("sample_media_path") or "")).expanduser().resolve()
    audio_extract = _extract_audio_for_dolphin(sample_media, run_dir / "dolphin-input.wav", timeout_seconds=timeout_seconds)
    if audio_extract.get("status") != "ok":
        return {**base, "status": "audio_extract_failed", "audio_extract": audio_extract, "module_ready": module}
    if "--input" in command:
        input_index = command.index("--input") + 1
        if input_index < len(command):
            command[input_index] = str(audio_extract.get("audio_path") or "")
    try:
        completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=int(timeout_seconds or 0) or None, check=False, env=_project_python_env())
    except FileNotFoundError as exc:
        return {**base, "status": "command_not_found", "stderr_tail": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {**base, "status": "timeout", "stdout_tail": _tail(str(exc.output or "")), "stderr_tail": _tail(str(exc.stderr or ""))}
    status = "ok" if completed.returncode == 0 and raw_output.exists() else "failed"
    result = {**base, "status": status, "returncode": completed.returncode, "stdout_tail": _tail(completed.stdout), "stderr_tail": _tail(completed.stderr), "raw_output_json": str(raw_output) if raw_output.exists() else "", "audio_extract": audio_extract, "module_ready": module}
    if status == "ok":
        try:
            project = Path(str(plan.get("workspace_dir") or raw_output.parents[2])).expanduser().resolve()
            normalized = normalize_asr_output(project, raw_output, provider="dolphin", title="dolphin")
            result.update({"normalized_json": normalized.get("json_path", ""), "normalized_srt": normalized.get("srt_path", ""), "metrics": _transcript_metrics(normalized.get("json_path", ""))})
        except Exception as exc:  # noqa: BLE001 - captured in A/B report.
            result.update({"status": "normalize_failed", "error": str(exc)})
    return result


def _run_whisperx_alignment_variant(
    plan: dict[str, Any],
    variant: dict[str, Any],
    *,
    sample_path: Path,
    execute: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    key = str(variant.get("key") or "whisperx_alignment")
    workspace = Path(str(plan.get("workspace_dir") or sample_path.parent.parent)).expanduser().resolve()
    run_dir = sample_path.parent / key
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_workspace = run_dir / "workspace"
    sample_workspace.mkdir(parents=True, exist_ok=True)
    base = {
        "key": key,
        "execute": bool(execute),
        "plan_path": "",
        "raw_output_json": "",
        "normalized_json": "",
        "normalized_srt": "",
        "metrics": {},
        "workspace_dir": str(sample_workspace),
        "operator_boundary": {
            "does_not_promote_any_transcript": True,
            "alignment_evidence_only": True,
            "does_not_replace_primary_asr": True,
            "source_workspace_dir": str(workspace),
        },
    }
    if not sample_path.exists():
        if not execute:
            return {**base, "status": "preview", "reason": "sample_not_extracted_yet"}
        return {**base, "status": "sample_missing", "error": f"sample media not found: {sample_path}"}
    try:
        run = run_whisperx_alignment(
            sample_workspace,
            sample_path,
            language=str(variant.get("language") or plan.get("language") or "zh"),
            model=str(variant.get("model") or "large-v3"),
            execute=execute,
            timeout_seconds=timeout_seconds,
            write=True,
        )
    except Exception as exc:  # noqa: BLE001 - captured in A/B report.
        return {**base, "status": "alignment_failed", "error": str(exc)}
    status = str(run.get("status") or "preview")
    normalized_json = str(run.get("alignment_transcript_json") or "")
    normalized_srt = str(run.get("alignment_transcript_srt") or "")
    return {
        **base,
        "status": "ok" if status == "alignment_ready" else status,
        "plan_path": str(run.get("plan_path") or ""),
        "raw_output_json": str(run.get("raw_output_json") or ""),
        "normalized_json": normalized_json,
        "normalized_srt": normalized_srt,
        "metrics": _transcript_metrics(normalized_json),
        "word_level_alignment": bool(run.get("word_level_alignment")),
        "alignment_run_json": str(run.get("json_path") or ""),
        "alignment_report": str(run.get("markdown_path") or ""),
    }


def _extract_audio_for_dolphin(sample_media: Path, audio_path: Path, *, timeout_seconds: int) -> dict[str, Any]:
    if not sample_media.exists():
        return {"status": "sample_missing", "sample_media_path": str(sample_media), "audio_path": str(audio_path)}
    ffmpeg = resolve_media_tool("ffmpeg") or "ffmpeg"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(sample_media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=max(30, min(int(timeout_seconds or 0) or 300, 300)),
            check=False,
            env=_project_python_env(),
        )
    except FileNotFoundError as exc:
        return {"status": "ffmpeg_not_found", "command": command, "sample_media_path": str(sample_media), "audio_path": str(audio_path), "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "command": command, "sample_media_path": str(sample_media), "audio_path": str(audio_path), "stdout_tail": _tail(str(exc.output or "")), "stderr_tail": _tail(str(exc.stderr or ""))}
    return {
        "status": "ok" if completed.returncode == 0 and audio_path.exists() else "failed",
        "command": command,
        "sample_media_path": str(sample_media),
        "audio_path": str(audio_path),
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _project_python_env() -> dict[str, str]:
    env = local_tool_subprocess_env()
    src = str(Path(__file__).resolve().parents[1])
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (f"{os.pathsep}{current}" if current else "")
    return env


def _python_module_ready(python: str, module: str) -> dict[str, Any]:
    if not python:
        return {"ready": False, "python": "", "module": module, "error": "missing_python"}
    try:
        completed = subprocess.run(
            [python, "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
            check=False,
            env=_project_python_env(),
        )
    except Exception as exc:
        return {"ready": False, "python": python, "module": module, "error": str(exc)}
    return {"ready": completed.returncode == 0, "python": python, "module": module, "returncode": completed.returncode, "stderr_tail": _tail(completed.stderr, 1000)}


def _adjunct_model_readiness(variant: dict[str, Any]) -> dict[str, Any]:
    command = [str(part) for part in (variant.get("command") or [])]
    required: list[dict[str, Any]] = []
    for flag in ("--punc-model", "--spk-model"):
        if flag not in command:
            continue
        index = command.index(flag) + 1
        if index >= len(command):
            continue
        model = command[index]
        resolved = _resolve_local_model(model)
        local = bool(resolved and resolved != model and Path(resolved).exists())
        required.append({"flag": flag, "model": model, "resolved": resolved, "ready": local})
    return {"ready": all(item.get("ready") for item in required), "required": required}


def _allow_model_download() -> bool:
    return os.environ.get("LECTURE_ASR_ALLOW_MODEL_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def _run_cloud_variant(plan: dict[str, Any], variant: dict[str, Any], *, sample_path: Path, execute: bool, provider_config: dict[str, Any] | None, timeout_seconds: int) -> dict[str, Any]:
    cfg = dict(provider_config or {})
    request_plan = variant.get("request_plan") if isinstance(variant.get("request_plan"), dict) else {}
    if request_plan.get("model") and "model" not in cfg:
        cfg["model"] = request_plan.get("model")
    if request_plan.get("language") and "language" not in cfg:
        cfg["language"] = request_plan.get("language")
    base = {
        "key": str(variant.get("key") or "openai_cloud_asr"),
        "execute": bool(execute),
        "request_plan": request_plan,
        "raw_output_json": "",
        "normalized_json": "",
        "normalized_srt": "",
        "metrics": {},
        "operator_boundary": {"cloud_upload": bool(execute), "sample_only": True, "secrets_redacted": True},
    }
    if not execute:
        return {**base, "status": "preview", "plan_path": ""}
    if not sample_path.exists():
        return {**base, "status": "sample_missing", "plan_path": "", "error": f"sample media not found: {sample_path}"}
    cloud_plan = plan_cloud_asr_run(sample_path.parent, sample_path, provider_config=cfg, model=str(cfg.get("model") or "gpt-4o-transcribe"), language=str(cfg.get("language") or plan.get("language") or "zh"))
    run = run_cloud_asr_plan(cloud_plan["plan_path"], provider_config=provider_config, execute=True, normalize=True)
    normalized = run.get("normalized") if isinstance(run.get("normalized"), dict) else {}
    cloud_call = run.get("cloud_call") if isinstance(run.get("cloud_call"), dict) else {}
    actual_plan = (cloud_call.get("request_plan") if isinstance(cloud_call.get("request_plan"), dict) else {}) or cloud_plan.get("request_plan", {})
    return {
        **base,
        "status": str(run.get("status") or "preview"),
        "request_plan": actual_plan,
        "plan_path": cloud_plan.get("plan_path", ""),
        "raw_output_json": run.get("raw_output_json", ""),
        "normalized_json": normalized.get("json_path", ""),
        "normalized_srt": normalized.get("srt_path", ""),
        "metrics": _transcript_metrics(normalized.get("json_path", "")),
    }


def _transcript_metrics(path: str) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    try:
        data = read_json(source)
    except Exception:
        return {}
    segments = data.get("segments") if isinstance(data, dict) else []
    if not isinstance(segments, list):
        return {}
    text = "\n".join(str(row.get("text") or "") for row in segments if isinstance(row, dict))
    punctuation = sum(text.count(ch) for ch in "，。！？；：,.!?;:")
    duration = 0.0
    speaker_labels: set[str] = set()
    speaker_labeled_segment_count = 0
    speaker_labeled_duration = 0.0
    if segments:
        starts = [float(row.get("start") or 0.0) for row in segments if isinstance(row, dict)]
        ends = [float(row.get("end") or 0.0) for row in segments if isinstance(row, dict)]
        if starts and ends:
            duration = max(ends) - min(starts)
        for row in segments:
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            speaker = str(
                row.get("speaker")
                or row.get("speaker_id")
                or metadata.get("speaker")
                or metadata.get("speaker_id")
                or ""
            ).strip()
            if not speaker:
                continue
            speaker_labels.add(speaker)
            speaker_labeled_segment_count += 1
            start = float(row.get("start") or 0.0)
            end = float(row.get("end") or start)
            speaker_labeled_duration += max(0.0, end - start)
    return {
        "segment_count": len(segments),
        "char_count": len(text),
        "punctuation_count": punctuation,
        "duration_seconds": round(duration, 3),
        "speaker_count": len(speaker_labels),
        "speaker_labeled_segment_count": speaker_labeled_segment_count,
        "speaker_labeled_duration_seconds": round(speaker_labeled_duration, 3),
        "speaker_labeled_duration_ratio": round(
            speaker_labeled_duration / duration,
            6,
        )
        if duration > 0
        else 0.0,
    }


def _overall_status(sample: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if sample.get("status") in {"failed", "timeout", "command_not_found", "missing_command"}:
        return "sample_failed"
    executed = [row for row in rows if row.get("execute")]
    if not executed:
        return "preview"
    if any(row.get("status") not in {"ok", "preview"} for row in executed):
        return "partial_failed"
    return "completed"


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASR A/B Sample Run",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Sample: `{result.get('sample_media_path', '')}`",
        f"- Execute sample: `{bool(result.get('execute_sample'))}`",
        f"- Execute local: `{bool(result.get('execute_local'))}`",
        f"- Execute cloud: `{bool(result.get('execute_cloud'))}`",
        "",
        "## Variants",
        "",
        "| Key | Status | Execute | Segments | Chars | Punctuation | Speakers | Speaker duration | Output |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("variants") or []:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        lines.append(
            f"| `{row.get('key', '')}` | `{row.get('status', '')}` | `{bool(row.get('execute'))}` | "
            f"{metrics.get('segment_count', '')} | {metrics.get('char_count', '')} | {metrics.get('punctuation_count', '')} | {metrics.get('speaker_count', '')} | {metrics.get('speaker_labeled_duration_ratio', '')} | `{row.get('normalized_json') or row.get('raw_output_json') or ''}` |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "This A/B run does not promote any transcript. Cloud ASR uploads only the extracted sample and only when explicitly enabled.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _tail(text: str, limit: int = 4000) -> str:
    value = str(text or "")
    return value[-limit:] if len(value) > limit else value
