from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _quote_ps_path
from .asr_adapter import normalize_asr_output
from .file_hash import sha256_file
from .asr_runner import plan_asr_run
from .local_media_progress import LocalMediaProgress, ProgressCallback
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import now_iso
from .qwen3_asr_python_runner import (
    CHECKPOINT_SCHEMA as QWEN_CHECKPOINT_SCHEMA,
    SCHEMA as QWEN_RAW_OUTPUT_SCHEMA,
    qwen_checkpoint_execution_contract,
    qwen_checkpoint_matches,
    qwen_checkpoint_validation_errors,
)
from .run_artifact_registry import register_bundle_run
from .storage import append_jsonl, ensure_project_dirs, read_json, read_jsonl, write_json


ASR_SMOKE_SCHEMA = "lecture_asr_smoke.v1"


def asr_smoke(
    media_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    preset: str = "sensevoice",
    model: str = "",
    language: str = "zh",
    duration_seconds: int = 30,
    execute: bool = True,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Run or preview a local short-segment ASR smoke test.

    This never uploads audio. It only creates a short local clip with ffmpeg and
    then delegates to the existing ASR plan/execution gate.
    """
    media = Path(media_path).expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(f"media not found: {media}")
    root = Path(output_dir).expanduser().resolve() if output_dir else media.parent / "asr-smoke"
    root.mkdir(parents=True, exist_ok=True)
    clip_path = root / f"asr-smoke-{max(int(duration_seconds or 30), 1)}s.wav"
    ffmpeg = resolve_media_tool("ffmpeg")
    started_at = now_iso()
    result: dict[str, Any] = {
        "schema": ASR_SMOKE_SCHEMA,
        "started_at": started_at,
        "finished_at": "",
        "media_path": str(media),
        "output_dir": str(root),
        "clip_path": str(clip_path),
        "preset": preset,
        "model": model,
        "language": language,
        "duration_seconds": max(int(duration_seconds or 30), 1),
        "execute": bool(execute),
        "timeout_seconds": int(timeout_seconds or 0),
        "ffmpeg": {"path": ffmpeg, "available": bool(ffmpeg)},
        "privacy": "Local-only ASR smoke. Audio stays on this machine; no cloud upload is performed by this command.",
        "status": "preview",
        "clip_command": _clip_command(ffmpeg or "ffmpeg", media, clip_path, duration_seconds=max(int(duration_seconds or 30), 1)),
        "asr_plan": {},
        "asr_run": {},
    }
    if not execute:
        result["finished_at"] = now_iso()
        return _write_asr_smoke_report(result)
    if not ffmpeg:
        result.update({"status": "ffmpeg_missing", "finished_at": now_iso()})
        return _write_asr_smoke_report(result)
    try:
        clip_completed = subprocess.run(
            result["clip_command"],
            cwd=str(root),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=max(int(timeout_seconds or 600), 30),
            check=False,
            env=local_tool_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "status": "clip_timeout",
                "clip_returncode": None,
                "clip_stdout": _timeout_stream(exc.output),
                "clip_stderr": _timeout_stderr(exc, timeout_seconds=timeout_seconds),
                "finished_at": now_iso(),
            }
        )
        return _write_asr_smoke_report(result)
    result.update(
        {
            "clip_returncode": clip_completed.returncode,
            "clip_stdout": clip_completed.stdout,
            "clip_stderr": clip_completed.stderr,
        }
    )
    if clip_completed.returncode != 0 or not clip_path.exists():
        result.update({"status": "clip_failed", "finished_at": now_iso()})
        return _write_asr_smoke_report(result)
    plan = plan_asr_run(root, clip_path, preset=preset, language=language, model=model or None)
    asr_run = run_asr_plan(plan["plan_path"], execute=True, timeout_seconds=timeout_seconds)
    result["asr_plan"] = _compact_asr_plan(plan)
    result["asr_run"] = _compact_asr_run(asr_run)
    result["status"] = "ok" if asr_run.get("status") == "ok" else str(asr_run.get("status") or "asr_failed")
    result["finished_at"] = now_iso()
    return _write_asr_smoke_report(result)


def run_asr_plan(
    plan_json: str | Path,
    *,
    execute: bool = False,
    normalize: bool = True,
    timeout_seconds: int = 0,
    resume: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Preview or run the ASR command from an ASR or lecture pipeline plan."""
    plan_path = Path(plan_json).expanduser().resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("ASR plan must be a JSON object")
    asr_plan = _extract_asr_plan(plan)
    normalize_requested = bool(normalize)
    asr_mode = str(asr_plan.get("asr_mode") or "").strip().lower()
    alignment_sidecar = asr_mode == "alignment" or str(asr_plan.get("preset") or "") == "qwen3-forced-aligner"
    normalize = bool(normalize_requested and not alignment_sidecar)
    project = Path(str(plan.get("project") or asr_plan.get("project") or "")).expanduser()
    if not project:
        project = plan_path.parent
    command = asr_plan.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) and part for part in command):
        raise ValueError("ASR plan command must be a non-empty string list")

    output_dir = Path(str(asr_plan.get("output_dir") or "")).expanduser()
    expected_output = Path(str(asr_plan.get("expected_output_json") or "")).expanduser()
    provider = str(asr_plan.get("provider") or "auto")
    result: dict[str, Any] = {
        "plan_path": str(plan_path),
        "project": str(project),
        "preset": str(asr_plan.get("preset") or ""),
        "provider": provider,
        "command": command,
        "execute": execute,
        "normalize": normalize,
        "normalize_requested": normalize_requested,
        "asr_mode": asr_mode,
        "normalization_skipped_reason": "alignment_sidecar" if alignment_sidecar and normalize_requested else ("caller_disabled" if not normalize_requested else ""),
        "timeout_seconds": int(timeout_seconds or 0),
        "resume": bool(resume),
        "resumed_from_checkpoint": False,
        "started_at": now_iso(),
        "output_dir": str(output_dir),
        "expected_output_json": str(expected_output),
        "status": "preview",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "raw_output_json": str(expected_output) if expected_output.exists() else "",
        "normalized": None,
    }

    if not execute:
        return _write_asr_log(project, result)

    progress = LocalMediaProgress(
        pipeline="local_asr_plan",
        snapshot_path=output_dir / "asr-plan-progress.json",
        events_path=output_dir / "asr-plan-progress.jsonl",
        callback=progress_callback,
    )
    result["_progress_recorder"] = progress
    progress.emit(
        stage="preflight",
        percent=0,
        message="Validating local ASR plan",
        output_paths=[expected_output],
        report_paths=[output_dir / "asr-run-report.md"],
    )

    effective_command = _qwen_resume_command(command, resume=resume)
    if resume:
        checkpoint_details = _qwen_checkpoint_resume_details(expected_output, effective_command)
        result.update(checkpoint_details)
        if checkpoint_details.get("checkpoint_complete"):
            output_reused = _qwen_completed_output_matches(
                expected_output,
                effective_command,
                checkpoint_details,
            )
            if output_reused:
                checkpoint_recovery = "raw_output_reused"
                skipped_reason = "output_already_complete"
            else:
                _restore_qwen_output_from_checkpoint(
                    expected_output,
                    effective_command,
                    checkpoint_details,
                )
                checkpoint_recovery = "raw_output_rebuilt"
                skipped_reason = "checkpoint_complete"
            result.update(
                {
                    "status": "ok",
                    "returncode": 0,
                    "planned_command": command,
                    "command": effective_command,
                    "execution_attempts": [],
                    "execution_skipped": True,
                    "execution_skipped_reason": skipped_reason,
                    "checkpoint_recovery": checkpoint_recovery,
                    "raw_output_json": str(expected_output.resolve()),
                    "finished_at": now_iso(),
                }
            )
            progress.emit(
                stage="resume",
                percent=85,
                message=(
                    "Reusing completed Qwen ASR output"
                    if output_reused
                    else "Rebuilding Qwen ASR output from completed checkpoint"
                ),
                output_paths=[expected_output],
                report_paths=[Path(str(checkpoint_details["checkpoint_path"]))],
            )
            if normalize:
                title = Path(str(asr_plan.get("media_path") or expected_output)).stem
                try:
                    result["normalized"] = normalize_asr_output(
                        project,
                        expected_output,
                        provider=provider,
                        title=title,
                    )
                except Exception as exc:
                    result["status"] = "normalize_failed"
                    result["normalization_error"] = str(exc)
                    result["stderr"] = _append_message(result.get("stderr", ""), str(exc))
            return _write_asr_log(project, result)

    if not asr_plan.get("available"):
        availability = asr_plan.get("availability") if isinstance(asr_plan.get("availability"), dict) else {}
        detail = "ASR runner is not marked available in the plan"
        if availability:
            command_path = availability.get("command_path") or ""
            module = availability.get("module") or ""
            module_available = availability.get("module_available")
            runtime_probe_value = availability.get("runtime_probe")
            runtime_probe = runtime_probe_value if isinstance(runtime_probe_value, dict) else {}
            runtime_blocker = runtime_probe.get("blocker") or ""
            blockers = list(availability.get("blockers") or [])
            detail = f"ASR runner is not marked available in the plan; command_path={command_path!r}; module={module!r}; module_available={module_available!r}; runtime_blocker={runtime_blocker!r}; blockers={blockers!r}"
        result.update({"status": "blocked", "stderr": detail, "availability": availability})
        return _write_asr_log(project, result)
    model_ready = asr_plan.get("model_ready") if isinstance(asr_plan.get("model_ready"), dict) else {}
    if (
        str(asr_plan.get("runner") or "") in {"funasr_python", "faster_whisper_python", "moss_transcribe_diarize_cli"}
        and model_ready
        and model_ready.get("ready") is False
        and not _allow_model_download()
    ):
        result.update(
            {
                "status": "asr_model_not_ready",
                "stderr": "Local ASR model cache is missing or unknown; set LECTURE_ASR_ALLOW_MODEL_DOWNLOAD=1 to allow first-run model download.",
                "finished_at": now_iso(),
                "model_ready": model_ready,
            }
        )
        return _write_asr_log(project, result)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress.emit(stage="execution", percent=10, message=f"Running local ASR provider {provider}")
    try:
        completed, execution_attempts = _run_command_with_cuda_oom_recovery(
            effective_command,
            asr_plan=asr_plan,
            cwd=output_dir,
            timeout_seconds=timeout_seconds,
            pythonpath=str(asr_plan.get("pythonpath") or ""),
            progress=progress,
        )
    except FileNotFoundError as exc:
        result.update(
            {
                "status": "command_not_found",
                "returncode": None,
                "stderr": str(exc),
                "finished_at": now_iso(),
            }
        )
        return _write_asr_log(project, result)
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "status": "timeout",
                "returncode": None,
                "stdout": _timeout_stream(exc.output),
                "stderr": _timeout_stderr(exc, timeout_seconds=timeout_seconds),
                "finished_at": now_iso(),
            }
        )
        result.update(_qwen_checkpoint_timeout_details(expected_output, command))
        return _write_asr_log(project, result)
    result.update(
        {
            "status": "ok" if completed.returncode == 0 else ("degraded" if completed.returncode == 2 else "failed"),
            "returncode": completed.returncode,
            "planned_command": command,
            "command": list(execution_attempts[-1]["command"]),
            "execution_attempts": execution_attempts,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "finished_at": now_iso(),
        }
    )
    if len(execution_attempts) > 1:
        result["recovery"] = "cuda_oom_recovered" if completed.returncode == 0 else "cuda_oom_recovery_exhausted"
    progress.emit(
        stage="collect_output",
        percent=85,
        message="Collecting local ASR output",
        details={"returncode": completed.returncode},
    )
    usable_returncode = completed.returncode in {0, 2}
    raw_output = _find_asr_output(expected_output, output_dir) if usable_returncode else None
    if usable_returncode and not raw_output:
        raw_output = _write_stdout_asr_output(completed.stdout, expected_output)
    if usable_returncode and raw_output and expected_output and raw_output.resolve() != expected_output.resolve():
        raw_output = _mirror_asr_output(raw_output, expected_output)
    result["raw_output_json"] = str(raw_output) if raw_output else ""
    try:
        raw_payload = read_json(raw_output) if raw_output and raw_output.exists() else {}
    except Exception as exc:
        raw_payload = {}
        result["stderr"] = _append_message(result.get("stderr", ""), f"Unable to inspect ASR status payload: {exc}")
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("runtime_metrics"), dict):
        # Intent: surface child telemetry through the stable ASR plan result.
        # Decision: copy only the bounded metrics summary, never chunk text.
        # Reason: operators need elapsed/GPU headroom without opening raw JSON.
        # Evidence: FunASR parent payload now aggregates native PyTorch counters.
        # Effective scope: local report/log metadata; inference is unchanged.
        result["runtime_metrics"] = dict(raw_payload["runtime_metrics"])
    if isinstance(raw_payload, dict) and str(raw_payload.get("status") or "") == "degraded":
        result["status"] = "degraded"
        result["failed_chunks"] = list(raw_payload.get("failed_chunks") or [])
        result["gaps"] = list(raw_payload.get("gaps") or [])
        result["retry_commands"] = list(raw_payload.get("retry_commands") or [])
    if (
        usable_returncode
        and alignment_sidecar
        and str(asr_plan.get("preset") or "") == "qwen3-forced-aligner"
        and raw_output
    ):
        result["alignment_sidecar_registration"] = _register_qwen3_alignment_sidecar(
            project, raw_output, asr_plan=asr_plan, raw_payload=raw_payload
        )
    if usable_returncode and not raw_output:
        result["status"] = "output_missing"
        result["stderr"] = _append_message(
            result.get("stderr", ""),
            f"ASR command completed successfully but no JSON output was found in {output_dir}",
        )
    if usable_returncode and normalize and raw_output:
        title = Path(str(asr_plan.get("media_path") or raw_output)).stem
        try:
            result["normalized"] = normalize_asr_output(project, raw_output, provider=provider, title=title)
        except Exception as exc:
            if result["status"] != "degraded":
                result["status"] = "normalize_failed"
            result["normalization_error"] = str(exc)
            result["stderr"] = _append_message(result.get("stderr", ""), str(exc))
    return _write_asr_log(project, result)


def asr_run_log(root: str | Path) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "asr-command-runs.jsonl"
    markdown_path = paths["notes"] / "asr-command-runs.md"
    rows = read_jsonl(log_path)
    markdown_path.write_text(_render_asr_log_markdown(root, rows), encoding="utf-8")
    return {
        "project": str(root),
        "log_path": str(log_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "commands": rows,
        "last": rows[-1] if rows else {},
    }


def _register_qwen3_alignment_sidecar(
    project: Path,
    raw_output: Path,
    *,
    asr_plan: dict[str, Any],
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    """Register validated word timing as read-only Bundle evidence."""

    manifest_path = project / "manifest.json"
    result: dict[str, Any] = {
        "status": "workspace_only",
        "path": str(raw_output),
        "canonical_transcript_path": str(asr_plan.get("alignment_transcript") or ""),
        "does_not_replace_canonical_transcript": True,
    }
    if not manifest_path.exists():
        return result
    expected_value = str(asr_plan.get("alignment_transcript") or "").strip()
    actual_value = str(raw_payload.get("transcript_path") or "").strip()
    if not expected_value or not actual_value:
        result["status"] = "blocked_transcript_identity_missing"
        return result
    expected = Path(expected_value).expanduser().resolve()
    actual = Path(actual_value).expanduser().resolve()
    if actual != expected or not expected.exists():
        result["status"] = "blocked_transcript_identity_mismatch"
        return result
    words = raw_payload.get("words") if isinstance(raw_payload.get("words"), list) else []
    if (
        str(raw_payload.get("schema") or "")
        != "video_knowledge_pipeline.qwen3_forced_aligner_output.v1"
        or str(raw_payload.get("status") or "") != "completed"
        or not words
        or raw_payload.get("timestamps_monotonic") is not True
    ):
        result["status"] = "alignment_not_ready"
        return result
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        result["status"] = "manifest_invalid"
        return result
    try:
        stored_path = str(raw_output.resolve().relative_to(project.resolve()))
    except ValueError:
        stored_path = str(raw_output.resolve())
    entry = {
        "preset": "qwen3-forced-aligner",
        "path": stored_path,
        "provider": str(raw_payload.get("provider") or "qwen3-forced-aligner"),
        "model": str(raw_payload.get("model") or ""),
        "status": "completed",
        "word_count": len(words),
        "transcript_path": str(expected),
        "transcript_sha256": _asr_artifact_sha256(expected),
        "sidecar_sha256": _asr_artifact_sha256(raw_output),
        "registered_at": now_iso(),
        "does_not_replace_canonical_transcript": True,
    }
    existing = [
        row
        for row in manifest.get("asr_alignment_sidecars") or []
        if isinstance(row, dict)
        and not (
            str(row.get("preset") or "") == entry["preset"]
            and str(row.get("transcript_path") or "") == entry["transcript_path"]
        )
    ]
    manifest["qwen3_forced_alignment_json"] = stored_path
    manifest["asr_alignment_sidecars"] = [*existing, entry]
    write_json(manifest_path, manifest)
    result.update(entry)
    result["status"] = "registered_read_only_sidecar"
    return result


def _asr_artifact_sha256(path: Path) -> str:
    return sha256_file(path)


def _extract_asr_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan.get("asr_plan"), dict):
        merged = dict(plan["asr_plan"])
        merged.setdefault("project", plan.get("project", ""))
        merged.setdefault("media_path", plan.get("media_path", ""))
        return merged
    return plan


def _run_command(command: list[str], *, cwd: Path, timeout_seconds: int, pythonpath: str = "") -> subprocess.CompletedProcess[str]:
    timeout = int(timeout_seconds or 0) or None
    env = local_tool_subprocess_env()
    if pythonpath:
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = pythonpath + (f"{os.pathsep}{current}" if current else "")
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _run_command_with_cuda_oom_recovery(
    command: list[str],
    *,
    asr_plan: dict[str, Any],
    cwd: Path,
    timeout_seconds: int,
    pythonpath: str,
    progress: LocalMediaProgress,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    """Run a local FunASR command with auditable OOM-only recovery."""
    command = _upgrade_funasr_command_to_chunked(command, asr_plan)
    completed = _run_command(command, cwd=cwd, timeout_seconds=timeout_seconds, pythonpath=pythonpath)
    attempts = [_execution_attempt("initial", command, completed)]
    if not _should_recover_cuda_oom(asr_plan, command, completed):
        return completed, attempts
    gpu_retry = _oom_recovery_command(command, device="cuda")
    progress.emit(stage="execution", percent=35, message="CUDA OOM: retrying local ASR on GPU with smaller batch and VAD segments")
    completed = _run_command(gpu_retry, cwd=cwd, timeout_seconds=timeout_seconds, pythonpath=pythonpath)
    attempts.append(_execution_attempt("gpu_oom_tuned_retry", gpu_retry, completed))
    if not _is_cuda_oom(completed):
        return completed, attempts
    cpu_retry = _replace_or_append_command_option(gpu_retry, "--device", "cpu")
    progress.emit(stage="execution", percent=60, message="CUDA OOM persisted after tuned GPU retry: retrying local ASR on CPU")
    completed = _run_command(cpu_retry, cwd=cwd, timeout_seconds=timeout_seconds, pythonpath=pythonpath)
    attempts.append(_execution_attempt("cpu_after_second_cuda_oom", cpu_retry, completed))
    return completed, attempts


def _should_recover_cuda_oom(asr_plan: dict[str, Any], command: list[str], completed: subprocess.CompletedProcess[str]) -> bool:
    if str(asr_plan.get("runner") or "") != "funasr_python":
        return False
    if str(asr_plan.get("preset") or "") not in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"}:
        return False
    device = str(_command_option(command, "--device") or asr_plan.get("local_asr_device") or "").lower()
    return device in {"cuda", "auto"} and _is_cuda_oom(completed)


def _is_cuda_oom(completed: subprocess.CompletedProcess[str]) -> bool:
    detail = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    return "cuda" in detail and "out of memory" in detail


def _oom_recovery_command(command: list[str], *, device: str) -> list[str]:
    batch_size_s = max(1, min(_command_int_option(command, "--batch-size-s", default=60) - 1, 10))
    vad_max = max(1, min(_command_int_option(command, "--vad-max-single-segment-time-ms", default=30000) - 1, 15000))
    recovered = _replace_or_append_command_option(command, "--batch-size-s", str(batch_size_s))
    recovered = _replace_or_append_command_option(recovered, "--vad-max-single-segment-time-ms", str(vad_max))
    return _replace_or_append_command_option(recovered, "--device", device)


def _replace_or_append_command_option(command: list[str], option: str, value: str) -> list[str]:
    updated = list(command)
    try:
        index = updated.index(option)
    except ValueError:
        updated.extend([option, value])
        return updated
    if index + 1 == len(updated):
        updated.append(value)
    else:
        updated[index + 1] = value
    return updated


def _command_option(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError:
        return ""
    return command[index + 1] if index + 1 < len(command) else ""


def _command_int_option(command: list[str], option: str, *, default: int) -> int:
    try:
        return max(1, int(_command_option(command, option) or default))
    except (TypeError, ValueError):
        return max(1, int(default))


def _execution_attempt(stage: str, command: list[str], completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {"stage": stage, "command": list(command), "returncode": completed.returncode, "stdout_tail": _tail(completed.stdout), "stderr_tail": _tail(completed.stderr)}


def _upgrade_funasr_command_to_chunked(command: list[str], asr_plan: dict[str, Any]) -> list[str]:
    """Upgrade old persisted SenseVoice plans without invalidating their artifacts."""
    if str(asr_plan.get("runner") or "") != "funasr_python":
        return list(command)
    if str(asr_plan.get("preset") or "") not in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"}:
        return list(command)
    upgraded = list(command)
    try:
        index = upgraded.index("video_knowledge_pipeline.funasr_python_runner")
    except ValueError:
        return upgraded
    upgraded[index] = "video_knowledge_pipeline.funasr_chunked_runner"
    if "--chunk-seconds" not in upgraded:
        upgraded.extend(["--chunk-seconds", "300"])
    if "--chunk-overlap-seconds" not in upgraded:
        upgraded.extend(["--chunk-overlap-seconds", "5"])
    return upgraded


def _find_asr_output(expected_output: Path, output_dir: Path) -> Path | None:
    if expected_output.exists():
        return expected_output.resolve()
    if output_dir.exists():
        nested_outputs = sorted(output_dir.rglob("raw-asr-output.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in nested_outputs:
            if path.resolve() != expected_output.resolve():
                return path.resolve()
        files = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in files:
            if (
                path.name != "asr-run-plan.json"
                and not path.name.endswith("-progress.json")
                and not path.name.endswith("-chunk-report.json")
            ):
                return path.resolve()
    return None


def _write_stdout_asr_output(stdout: str, expected_output: Path) -> Path | None:
    parsed = _parse_stdout_asr_records(stdout)
    if parsed is None:
        return None
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(expected_output, parsed)
    return expected_output.resolve()


def _mirror_asr_output(source: Path, expected_output: Path) -> Path:
    """Copy provider-named JSON output to the stable project output path."""
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    data = read_json(source)
    write_json(expected_output, data)
    return expected_output.resolve()


def _parse_stdout_asr_records(stdout: str) -> Any | None:
    for line in reversed(str(stdout or "").splitlines()):
        clean = line.strip()
        if not (clean.startswith("[{") and clean.endswith("}]")):
            continue
        try:
            parsed = ast.literal_eval(clean)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
    return None


def _write_asr_log(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    progress = result.pop("_progress_recorder", None)
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "asr-command-runs.jsonl"
    record = {
        "created_at": now_iso(),
        "preset": result.get("preset", ""),
        "provider": result.get("provider", ""),
        "execute": bool(result.get("execute")),
        "normalize": bool(result.get("normalize")),
        "resume": bool(result.get("resume")),
        "resumed_from_checkpoint": bool(result.get("resumed_from_checkpoint")),
        "status": result.get("status", ""),
        "returncode": result.get("returncode"),
        "command": result.get("command", []),
        "raw_output_json": result.get("raw_output_json", ""),
        "normalized_json": ((result.get("normalized") or {}).get("json_path") if isinstance(result.get("normalized"), dict) else ""),
        "stdout_tail": _tail(result.get("stdout", "")),
        "stderr_tail": _tail(result.get("stderr", "")),
        "execution_attempts": result.get("execution_attempts", []),
        "execution_skipped_reason": result.get("execution_skipped_reason", ""),
        "checkpoint_path": result.get("checkpoint_path", ""),
        "checkpoint_successful_chunk_count": int(
            result.get("checkpoint_successful_chunk_count") or 0
        ),
        "runtime_metrics": result.get("runtime_metrics", {}),
    }
    append_jsonl(log_path, [record])
    log = asr_run_log(root)
    report_path = _write_asr_run_report(result, record)
    result["asr_log"] = {
        "record": record,
        "log_path": log["log_path"],
        "markdown_path": log["markdown_path"],
        "report_path": str(report_path) if report_path else "",
        "count": log["count"],
    }
    if isinstance(progress, LocalMediaProgress):
        status = str(result.get("status") or "")
        terminal = "completed" if status == "ok" else ("degraded" if status == "degraded" else "failed")
        output_paths = [
            value
            for value in (
                result.get("raw_output_json"),
                ((result.get("normalized") or {}).get("json_path") if isinstance(result.get("normalized"), dict) else ""),
            )
            if value
        ]
        progress.emit(
            stage="finalize",
            percent=100,
            message=(
                "Local ASR plan completed"
                if terminal == "completed"
                else "Local ASR plan completed with missing chunks"
                if terminal == "degraded"
                else f"Local ASR plan ended with status {status or 'failed'}"
            ),
            status=terminal,
            output_paths=output_paths,
            report_paths=[report_path] if report_path else [],
            details={
                "returncode": result.get("returncode"),
                "failed_chunk_count": len(result.get("failed_chunks") or []),
            },
        )
        result["progress"] = progress.artifacts()
    run_registry = _register_run_if_bundle(root, result)
    if run_registry:
        result["run_registry"] = run_registry
    return result



def _register_run_if_bundle(root: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    bundle_root = root.expanduser().resolve()
    if not (bundle_root / "manifest.json").exists():
        return None
    status = _run_registry_status(result)
    failed_items = _run_failed_items(result)
    asr_log = result.get("asr_log") if isinstance(result.get("asr_log"), dict) else {}
    retry_command = f".\\scripts\\video-knowledge.ps1 run-asr-plan {_quote_ps_path(result.get('plan_path', ''))} --execute"
    return register_bundle_run(
        bundle_root,
        run_type="asr_run_plan",
        run_id="asr-run-plan",
        status=status,
        title="ASR run plan",
        summary=(
            f"Preset={result.get('preset', '')}; provider={result.get('provider', '')}; "
            f"execute={bool(result.get('execute'))}; status={result.get('status', '')}."
        ),
        inputs={
            "plan_path": result.get("plan_path", ""),
            "expected_output_json": result.get("expected_output_json", ""),
            "raw_output_json": result.get("raw_output_json", ""),
        },
        parameters={
            "execute": bool(result.get("execute")),
            "normalize": bool(result.get("normalize")),
            "resume": bool(result.get("resume")),
            "timeout_seconds": int(result.get("timeout_seconds") or 0),
            "preset": result.get("preset", ""),
            "provider": result.get("provider", ""),
        },
        artifacts=[
            {"key": "run_report", "path": asr_log.get("report_path", "")},
            {"key": "run_log", "path": asr_log.get("markdown_path", "")},
            {"key": "raw_output_json", "path": result.get("raw_output_json", "")},
            {"key": "checkpoint_json", "path": result.get("checkpoint_path", "")},
            {"key": "normalized_transcript_json", "path": ((result.get("normalized") or {}).get("json_path") if isinstance(result.get("normalized"), dict) else "")},
            {"key": "normalized_transcript_srt", "path": ((result.get("normalized") or {}).get("srt_path") if isinstance(result.get("normalized"), dict) else "")},
        ],
        failed_items=failed_items,
        retry_command=retry_command,
        next_actions=_run_next_actions(status),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "audio_stays_local": True,
            "preview_first": True,
            "model_download_requires_env": True,
            "purpose": "Expose local ASR plan/run status, artifacts, failures, and retry commands to VKP workbench/task console.",
        },
        write=True,
    )


def _run_registry_status(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "")
    if status == "preview":
        return "needs_execution"
    if status == "ok":
        return "completed"
    if status == "degraded":
        return "needs_review"
    if status in {"blocked", "asr_model_not_ready", "command_not_found"}:
        return "needs_input"
    if status in {"failed", "timeout", "output_missing", "normalize_failed", "clip_failed", "clip_timeout"}:
        return "needs_retry"
    return "needs_retry" if status else "unknown"


def _run_failed_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    status = str(result.get("status") or "")
    if status in {"", "ok", "preview"}:
        return []
    if status == "degraded" and result.get("failed_chunks"):
        return [
            {
                "item": f"chunk-{row.get('chunk_index')}",
                "reason": row.get("reason") or "chunk_failed",
                "detail": row.get("detail") or "",
                "retry_command": row.get("retry_command") or {},
            }
            for row in result.get("failed_chunks") or []
            if isinstance(row, dict)
        ]
    return [
        {
            "item": "asr_run",
            "reason": status,
            "detail": _tail(result.get("stderr", "") or result.get("stdout", "")),
        }
    ]


def _run_next_actions(status: str) -> list[str]:
    if status == "needs_execution":
        return ["Run ASR with --execute after confirming this is local ASR and the model cache is ready."]
    if status == "needs_input":
        return ["Fix local ASR environment/model cache/command availability, then rerun the ASR plan."]
    if status == "needs_retry":
        return ["Inspect asr-run-report.md and retry the same plan after fixing the failure."]
    if status == "completed":
        return ["Use normalized transcript as an input to transcript arbitration and smart summary."]
    if status == "needs_review":
        return ["Use the preserved successful transcript, review reported gaps, and run only the listed chunk retry commands."]
    return ["Inspect ASR run status before continuing."]


def _allow_model_download() -> bool:
    return str(os.environ.get("LECTURE_ASR_ALLOW_MODEL_DOWNLOAD", "")).strip().lower() in {"1", "true", "yes", "on"}


def _write_asr_run_report(result: dict[str, Any], record: dict[str, Any]) -> Path | None:
    output_dir = Path(str(result.get("output_dir") or "")).expanduser()
    if not output_dir:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "asr-run-report.md"
    lines = [
        "# ASR Run Report",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Preset: `{result.get('preset', '')}`",
        f"- Provider: `{result.get('provider', '')}`",
        f"- Execute: `{bool(result.get('execute'))}`",
        f"- Resume enabled: `{bool(result.get('resume'))}`",
        f"- Resumed from checkpoint: `{bool(result.get('resumed_from_checkpoint'))}`",
        f"- Checkpoint: `{result.get('checkpoint_path', '')}`",
        f"- Checkpoint successful chunks: `{result.get('checkpoint_successful_chunk_count', 0)}`",
        f"- Execution skipped reason: `{result.get('execution_skipped_reason', '')}`",
        f"- Return code: `{result.get('returncode')}`",
        f"- Raw output: `{result.get('raw_output_json', '')}`",
        f"- Normalized JSON: `{record.get('normalized_json', '')}`",
        f"- Started at: `{result.get('started_at', '')}`",
        f"- Finished at: `{result.get('finished_at', '')}`",
        "",
        "## Command",
        "",
        "```text",
        " ".join(str(part) for part in result.get("command", [])),
        "```",
        "",
        "## Stderr Tail",
        "",
        "```text",
        _tail(result.get("stderr", ""), limit=2000),
        "```",
    ]
    runtime_metrics = result.get("runtime_metrics") if isinstance(result.get("runtime_metrics"), dict) else {}
    if runtime_metrics:
        lines.extend(
            [
                "",
                "## Runtime Metrics",
                "",
                f"- Measured chunks: `{runtime_metrics.get('measured_chunk_count', 0)}`",
                f"- Total child elapsed seconds: `{runtime_metrics.get('total_child_elapsed_seconds', '')}`",
                f"- Max CUDA peak allocated MiB: `{runtime_metrics.get('max_cuda_peak_memory_allocated_mib', '')}`",
                f"- Max CUDA peak reserved MiB: `{runtime_metrics.get('max_cuda_peak_memory_reserved_mib', '')}`",
                f"- Missing metric chunks: `{runtime_metrics.get('missing_chunk_indexes', [])}`",
            ]
        )
    attempts = result.get("execution_attempts") if isinstance(result.get("execution_attempts"), list) else []
    if attempts:
        lines.extend(["", "## Execution Attempts", ""])
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            lines.append(
                f"- `{attempt.get('stage', '')}`: return `{attempt.get('returncode')}`; "
                f"device=`{_command_option(list(attempt.get('command') or []), '--device')}`; "
                f"batch_size_s=`{_command_option(list(attempt.get('command') or []), '--batch-size-s')}`; "
                f"vad_max_single_segment_time_ms=`{_command_option(list(attempt.get('command') or []), '--vad-max-single-segment-time-ms')}`"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _render_asr_log_markdown(root: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ASR Command Runs",
        "",
        f"- Project: `{root}`",
        f"- Count: {len(rows)}",
        "",
        "| Time | Preset | Execute | Normalize | Status | Return | Output |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        execute = "yes" if row.get("execute") else "no"
        normalize = "yes" if row.get("normalize") else "no"
        lines.append(
            f"| {row.get('created_at', '')} | `{row.get('preset', '')}` | {execute} | {normalize} | {row.get('status', '')} | {row.get('returncode', '')} | `{row.get('normalized_json') or row.get('raw_output_json') or ''}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _tail(value: Any, *, limit: int = 4000) -> str:
    return str(value or "")[-limit:]


def _is_qwen_asr_command(command: list[str]) -> bool:
    return "video_knowledge_pipeline.qwen3_asr_python_runner" in command


def _qwen_resume_command(command: list[str], *, resume: bool) -> list[str]:
    if not _is_qwen_asr_command(command):
        return list(command)
    updated = [part for part in command if part != "--no-resume"]
    if not resume:
        updated.append("--no-resume")
    return updated


def _qwen_command_contract(
    command: list[str],
    *,
    window_plan_revision: str | None = None,
) -> dict[str, Any] | None:
    if not _is_qwen_asr_command(command):
        return None
    input_value = _command_option(command, "--input")
    model = _command_option(command, "--model") or "Qwen/Qwen3-ASR-1.7B"
    if not input_value:
        return None
    media = Path(input_value).expanduser().resolve()
    if not media.is_file():
        return None
    forced_aligner = "" if "--no-timestamps" in command else (
        _command_option(command, "--forced-aligner") or "Qwen/Qwen3-ForcedAligner-0.6B"
    )
    raw_indexes = _command_option(command, "--chunk-indexes")
    try:
        chunk_indexes = [int(value.strip()) for value in raw_indexes.split(",") if value.strip()]
    except ValueError:
        return None
    vad_value = _command_option(command, "--vad-json")
    vad_sha256 = ""
    if vad_value:
        vad_path = Path(vad_value).expanduser().resolve()
        if not vad_path.is_file():
            return None
        vad_sha256 = sha256_file(vad_path)
    return qwen_checkpoint_execution_contract(
        media=media,
        model=model,
        forced_aligner=forced_aligner,
        language=_command_option(command, "--language") or "Chinese",
        context=_command_option(command, "--context"),
        chunk_seconds=_command_int_option(command, "--chunk-seconds", default=300),
        max_new_tokens=_command_int_option(command, "--max-new-tokens", default=1024),
        dtype_name=(
            _command_option(command, "--dtype")
            or os.environ.get("VKP_QWEN_ASR_DTYPE", "auto")
        ),
        chunk_indexes=chunk_indexes,
        window_plan_revision=window_plan_revision,
        timestamp_basis=_command_option(command, "--timestamp-basis") or "chunk_relative",
        vad_evidence_sha256=vad_sha256,
    )


def _qwen_checkpoint_resume_details(expected_output: Path, command: list[str]) -> dict[str, Any]:
    checkpoint = expected_output.with_name(f"{expected_output.stem}-checkpoint.json")
    if not checkpoint.is_file():
        return {}
    try:
        payload = read_json(checkpoint)
    except Exception as exc:
        return {
            "checkpoint_path": str(checkpoint),
            "checkpoint_error": str(exc),
            "checkpoint_resume_eligible": False,
        }
    schema = str(payload.get("schema") or "") if isinstance(payload, dict) else ""
    if not isinstance(payload, dict) or schema not in {
        QWEN_CHECKPOINT_SCHEMA,
        "video_knowledge_pipeline.qwen3_asr_checkpoint.v1",
    }:
        return {
            "checkpoint_path": str(checkpoint),
            "checkpoint_resume_eligible": False,
            "checkpoint_resume_rejected_reason": "checkpoint_schema_mismatch",
        }
    stored_contract = payload.get("execution_contract")
    window_plan_revision = (
        str(stored_contract.get("window_plan_revision") or "")
        if isinstance(stored_contract, dict)
        else ""
    )
    contract = _qwen_command_contract(
        command,
        window_plan_revision=window_plan_revision or None,
    )
    if contract is None:
        return {
            "checkpoint_path": str(checkpoint),
            "checkpoint_resume_eligible": False,
            "checkpoint_resume_rejected_reason": "qwen_command_contract_unavailable",
        }
    if not qwen_checkpoint_matches(payload, contract):
        return {
            "checkpoint_path": str(checkpoint),
            "checkpoint_resume_eligible": False,
            "checkpoint_resume_rejected_reason": "execution_contract_mismatch",
        }
    if schema == QWEN_CHECKPOINT_SCHEMA:
        window_plan_path = expected_output.with_name(f"{expected_output.stem}-window-plan.json")
        try:
            window_plan = read_json(window_plan_path)
        except Exception:
            return {
                "checkpoint_path": str(checkpoint),
                "checkpoint_resume_eligible": False,
                "checkpoint_resume_rejected_reason": "window_plan_unreadable",
            }
        errors = qwen_checkpoint_validation_errors(
            payload,
            expected_contract=contract,
            expected_window_plan=window_plan,
        )
        if errors:
            return {
                "checkpoint_path": str(checkpoint),
                "checkpoint_resume_eligible": False,
                "checkpoint_resume_rejected_reason": errors[0],
            }
    declared_successful = sorted(
        {int(value) for value in payload.get("successful_chunk_indexes") or []}
    )
    result_successful = sorted(
        {
            int(row.get("chunk_index") or 0)
            for row in payload.get("results") or []
            if isinstance(row, dict)
        }
    )
    if declared_successful and declared_successful != result_successful:
        return {
            "checkpoint_path": str(checkpoint),
            "checkpoint_resume_eligible": False,
            "checkpoint_resume_rejected_reason": "checkpoint_result_index_mismatch",
        }
    successful = result_successful
    failed_count = int(payload.get("failed_chunk_count") or len(payload.get("failed_chunks") or []))
    requested_count = int(payload.get("requested_chunk_count") or 0)
    successful_count = len(successful)
    complete = bool(
        requested_count > 0
        and failed_count == 0
        and successful_count >= requested_count
        and len(successful) >= requested_count
    )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_resume_eligible": True,
        "checkpoint_status": str(payload.get("status") or "running"),
        "checkpoint_requested_chunk_count": requested_count,
        "checkpoint_successful_chunk_count": successful_count,
        "checkpoint_successful_chunk_indexes": successful,
        "checkpoint_failed_chunk_count": failed_count,
        "checkpoint_window_plan_revision": window_plan_revision,
        "checkpoint_complete": complete,
        "resumed_from_checkpoint": bool(successful or failed_count),
        "partial_output_preserved": bool(successful),
    }


def _qwen_completed_output_matches(
    output: Path,
    command: list[str],
    checkpoint_details: dict[str, Any],
) -> bool:
    if not output.is_file():
        return False
    contract = _qwen_command_contract(
        command,
        window_plan_revision=(
            str(checkpoint_details.get("checkpoint_window_plan_revision") or "")
            or None
        ),
    )
    if contract is None:
        return False
    try:
        payload = read_json(output)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    expected_indexes = checkpoint_details.get("checkpoint_successful_chunk_indexes") or []
    actual_indexes = sorted({int(value) for value in payload.get("successful_chunk_indexes") or []})
    return bool(
        str(payload.get("schema") or "") == QWEN_RAW_OUTPUT_SCHEMA
        and str(payload.get("status") or "") == "completed"
        and str(payload.get("input_path") or "") == str(contract["input_identity"]["path"])
        and str(payload.get("model") or "") == str(contract["model"])
        and int(payload.get("chunk_seconds") or 0) == int(contract["chunk_seconds"])
        and int(payload.get("failed_chunk_count") or 0) == 0
        and actual_indexes == expected_indexes
    )


def _restore_qwen_output_from_checkpoint(
    output: Path,
    command: list[str],
    checkpoint_details: dict[str, Any],
) -> None:
    checkpoint_path = Path(str(checkpoint_details["checkpoint_path"]))
    checkpoint = read_json(checkpoint_path)
    stored_contract = (
        checkpoint.get("execution_contract")
        if isinstance(checkpoint, dict)
        else None
    )
    contract = _qwen_command_contract(
        command,
        window_plan_revision=(
            str(stored_contract.get("window_plan_revision") or "")
            if isinstance(stored_contract, dict)
            else None
        ),
    )
    if not isinstance(checkpoint, dict) or contract is None:
        raise ValueError("Qwen checkpoint cannot be restored without a matching execution contract")
    if not qwen_checkpoint_matches(checkpoint, contract):
        raise ValueError("Qwen checkpoint execution contract changed before recovery")
    rows = [dict(row) for row in checkpoint.get("results") or [] if isinstance(row, dict)]
    failed_chunks = [
        dict(row) for row in checkpoint.get("failed_chunks") or [] if isinstance(row, dict)
    ]
    successful = sorted({int(row.get("chunk_index") or 0) for row in rows})
    requested_count = int(checkpoint.get("requested_chunk_count") or 0)
    if requested_count <= 0 or len(successful) < requested_count or failed_chunks:
        raise ValueError("Qwen checkpoint is not complete")
    payload = {
        "schema": QWEN_RAW_OUTPUT_SCHEMA,
        "provider": "qwen3-asr",
        "model": contract["model"],
        "forced_aligner": contract["forced_aligner"],
        "chunk_seconds": contract["chunk_seconds"],
        "chunk_count": requested_count,
        "max_chunk_attempts": _command_int_option(command, "--max-chunk-attempts", default=2),
        "retry_exhausted_chunk_count": 0,
        "checkpoint_path": str(checkpoint_path),
        "window_plan_path": str(output.with_name(f"{output.stem}-window-plan.json")),
        "window_plan": checkpoint.get("window_plan") or {},
        "window_plan_revision": str(checkpoint.get("window_plan_revision") or ""),
        "input_sha256": str(contract["input_identity"].get("sha256") or ""),
        "timestamp_input_basis": str(contract.get("timestamp_input_basis") or "chunk_relative"),
        "timestamp_output_basis": "media_absolute",
        "vad_evidence_sha256": str(contract.get("vad_evidence_sha256") or ""),
        "execution_order": [
            int(row["index"])
            for row in sorted(
                checkpoint.get("window_plan", {}).get("windows", []),
                key=lambda row: (int(row.get("execution_order", 0)), int(row.get("index", 0))),
            )
        ],
        "resumed_from_checkpoint": True,
        "checkpointed_successful_chunk_count": len(successful),
        "successful_chunk_count": len(successful),
        "failed_chunk_count": 0,
        "successful_chunk_indexes": successful,
        "device": "not_reexecuted",
        "dtype": contract["dtype"],
        "input_path": contract["input_identity"]["path"],
        "input_identity": contract["input_identity"],
        "ok": True,
        "usable": True,
        "status": "completed",
        "results": rows,
        "segments": [segment for row in rows for segment in row.get("segments") or []],
        "text": "\n".join(str(row.get("text") or "") for row in rows).strip(),
        "failed_chunks": [],
        "gaps": [],
        "retry_commands": [],
        "fallback": {"recommended_model": "Qwen/Qwen3-ASR-0.6B", "automatic": False},
        "checkpoint_recovery": "raw_output_rebuilt_without_model_execution",
    }
    write_json(output, payload)


def _qwen_checkpoint_timeout_details(expected_output: Path, command: list[str]) -> dict[str, Any]:
    details = _qwen_checkpoint_resume_details(expected_output, command)
    if not details:
        return {}
    details["resume_command"] = list(command)
    return details

def _timeout_stream(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _timeout_stderr(exc: subprocess.TimeoutExpired, *, timeout_seconds: int) -> str:
    stderr = _timeout_stream(exc.stderr)
    message = f"timeout after {int(timeout_seconds or 0)}s: {exc}"
    return (stderr + "\n" + message).strip() if stderr else message


def _append_message(value: Any, message: str) -> str:
    current = str(value or "").strip()
    clean_message = str(message or "").strip()
    if current and clean_message:
        return f"{current}\n{clean_message}"
    return current or clean_message


def _clip_command(ffmpeg: str, media: Path, clip_path: Path, *, duration_seconds: int) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-t",
        str(max(int(duration_seconds or 30), 1)),
        "-i",
        str(media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(clip_path),
    ]


def _write_asr_smoke_report(result: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(result.get("output_dir") or ".")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "asr-smoke.json"
    markdown_path = root / "asr-smoke.md"
    result["json_path"] = str(json_path)
    result["report_path"] = str(markdown_path)
    write_json(json_path, result)
    markdown_path.write_text(render_asr_smoke_markdown(result), encoding="utf-8")
    return result


def render_asr_smoke_markdown(result: dict[str, Any]) -> str:
    asr_run = result.get("asr_run") if isinstance(result.get("asr_run"), dict) else {}
    lines = [
        "# ASR Smoke Report",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Execute: `{bool(result.get('execute'))}`",
        f"- Media: `{result.get('media_path', '')}`",
        f"- Clip: `{result.get('clip_path', '')}`",
        f"- Duration seconds: `{result.get('duration_seconds', '')}`",
        f"- Preset/model: `{result.get('preset', '')}` / `{result.get('model', '')}`",
        f"- Language: `{result.get('language', '')}`",
        f"- FFmpeg available: `{(result.get('ffmpeg') or {}).get('available', False)}`",
        f"- Privacy: {result.get('privacy', '')}",
        "",
        "## Clip Command",
        "",
        "```text",
        " ".join(str(part) for part in result.get("clip_command") or []),
        "```",
        "",
        "## ASR Result",
        "",
        f"- ASR status: `{asr_run.get('status', '')}`",
        f"- Normalized JSON: `{asr_run.get('normalized_json', '')}`",
        f"- Raw output: `{asr_run.get('raw_output_json', '')}`",
        f"- Report: `{asr_run.get('report_path', '')}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _compact_asr_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_path": plan.get("plan_path", ""),
        "preset": plan.get("preset", ""),
        "provider": plan.get("provider", ""),
        "runner": plan.get("runner", ""),
        "available": bool(plan.get("available")),
        "model_ready": plan.get("model_ready", {}),
        "expected_output_json": plan.get("expected_output_json", ""),
    }


def _compact_asr_run(run: dict[str, Any]) -> dict[str, Any]:
    normalized = run.get("normalized") if isinstance(run.get("normalized"), dict) else {}
    log = run.get("asr_log") if isinstance(run.get("asr_log"), dict) else {}
    return {
        "status": run.get("status", ""),
        "returncode": run.get("returncode"),
        "raw_output_json": run.get("raw_output_json", ""),
        "normalized_json": normalized.get("json_path", ""),
        "report_path": log.get("report_path", ""),
    }
