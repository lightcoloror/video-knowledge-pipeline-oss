from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from .audio_chunk_manifest import extract_audio_chunk_window, fixed_chunk_boundaries
from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .local_media_progress import LocalMediaProgress, stderr_progress_callback
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .asr_runner import _model_ready
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.qwen3_asr_raw_output.v1"
CHECKPOINT_SCHEMA = "video_knowledge_pipeline.qwen3_asr_checkpoint.v2"
WINDOW_PLAN_SCHEMA = "video_knowledge_pipeline.qwen_asr_window_plan.v2"
TIMESTAMP_BASES = {"chunk_relative", "media_absolute"}
VAD_CONFIDENCE_THRESHOLD = 0.5


def _read_pcm_chunk(path: Path):
    """Use Qwen's native waveform interface, avoiding librosa/numba cache I/O.

    extract_audio_chunk_window already writes mono PCM at 16 kHz; decoding it
    directly also avoids a second, implicit resampler.
    """
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != 16000 or audio.ndim != 1 or not audio.size:
        raise ValueError("qwen_chunk_requires_nonempty_mono_16khz_pcm")
    return audio, rate


class QwenCheckpointRejected(ValueError):
    """A present checkpoint failed identity or structural validation."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen3-asr-python-runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--forced-aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--context", default="")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.environ.get("LECTURE_ASR_DEVICE", "auto"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=os.environ.get("VKP_QWEN_ASR_DTYPE", "auto"))
    parser.add_argument("--chunk-seconds", type=int, default=300)
    parser.add_argument("--chunk-indexes", default="", help="Comma-separated zero-based chunks to retry; all chunks are processed by default")
    parser.add_argument("--max-chunk-attempts", type=int, default=2, help="Bounded attempts per chunk across checkpoint resumes")
    parser.add_argument("--timestamp-basis", choices=sorted(TIMESTAMP_BASES), default="chunk_relative")
    parser.add_argument("--vad-json", default="", help="Optional local speech-VAD evidence used only to prioritize windows")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Ignore a matching per-chunk checkpoint and start all requested chunks again")
    args = parser.parse_args(argv)
    result = run_qwen3_asr(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        forced_aligner="" if args.no_timestamps else args.forced_aligner,
        language=args.language,
        context=args.context,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        chunk_seconds=args.chunk_seconds,
        dtype_name=args.dtype,
        chunk_indexes=_chunk_indexes(args.chunk_indexes),
        max_chunk_attempts=args.max_chunk_attempts,
        timestamp_basis=args.timestamp_basis,
        vad_json=args.vad_json or None,
        resume=not args.no_resume,
        progress_callback=stderr_progress_callback,
    )
    print(json.dumps(result, ensure_ascii=False))
    if result.get("status") == "degraded":
        return 2
    return 0 if result.get("ok") else 1


def run_qwen3_asr(
    *,
    input_path: str,
    output_path: str,
    model: str,
    forced_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    language: str = "Chinese",
    context: str = "",
    device: str = "auto",
    max_new_tokens: int = 1024,
    chunk_seconds: int = 300,
    dtype_name: str = "auto",
    chunk_indexes: list[int] | None = None,
    max_chunk_attempts: int = 2,
    timestamp_basis: str = "chunk_relative",
    vad_json: str | Path | None = None,
    resume: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    media = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = LocalMediaProgress(
        pipeline="local_qwen3_asr",
        snapshot_path=output.with_name(f"{output.stem}-progress.json"),
        events_path=output.with_name(f"{output.stem}-progress.jsonl"),
        callback=progress_callback,
    )
    report_path = output.with_name(f"{output.stem}-chunk-report.json")
    checkpoint_path = output.with_name(f"{output.stem}-checkpoint.json")
    progress.emit(
        stage="preflight",
        percent=0,
        message="Validating local ASR input and runtime",
        output_paths=[output],
        report_paths=[report_path, checkpoint_path],
    )
    if not media.exists():
        payload = _failure(output, "input_not_found", f"input not found: {media}", model=model)
        return _finalize_payload(payload, output, report_path, progress)
    input_timestamp_basis = str(timestamp_basis or "chunk_relative").strip().lower()
    if input_timestamp_basis not in TIMESTAMP_BASES:
        payload = _failure(
            output,
            "qwen3_asr_timestamp_basis_invalid",
            f"unsupported timestamp basis: {input_timestamp_basis}",
            model=model,
        )
        return _finalize_payload(payload, output, report_path, progress)
    try:
        input_sha256 = sha256_file(media)
        vad_intervals, vad_evidence_sha256 = _load_vad_priority_evidence(vad_json)
    except Exception as exc:
        payload = _failure(output, "qwen3_asr_vad_preflight_failed", str(exc), model=model)
        return _finalize_payload(payload, output, report_path, progress)
    preset = "qwen3-forced-aligner" if "ForcedAligner" in model else (
        "qwen3-asr-0.6b" if "0.6B" in model else "qwen3-asr-1.7b"
    )
    readiness = _model_ready(preset=preset, model=model)
    if not readiness.get("ready"):
        payload = _failure(
            output,
            "qwen3_asr_model_not_ready",
            f"local Qwen snapshot is not complete: {readiness.get('status')}",
            model=model,
        )
        payload["model_readiness"] = readiness
        return _finalize_payload(payload, output, report_path, progress)
    model_source = str((readiness.get("cache_matches") or [model])[0])
    if forced_aligner:
        aligner_ready = _model_ready(preset="qwen3-forced-aligner", model=forced_aligner)
        if not aligner_ready.get("ready"):
            payload = _failure(output, "qwen3_aligner_not_ready", "Transcription model is ready; optional aligner is missing. Replan with --no-timestamps for chunk-level text evidence.", model=model)
            payload["transcription_model_ready"] = True
            payload["alignment_model_ready"] = False
            return _finalize_payload(payload, output, report_path, progress)
        forced_aligner = str((aligner_ready.get("cache_matches") or [forced_aligner])[0])
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except Exception as exc:
        payload = _failure(output, "qwen_asr_module_not_ready", str(exc), model=model)
        return _finalize_payload(payload, output, report_path, progress)
    selected_device = _device(device, torch)
    dtype = _torch_dtype(dtype_name, selected_device, torch)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": "cuda:0" if selected_device.startswith("cuda") else "cpu",
        "low_cpu_mem_usage": True,
        "max_inference_batch_size": 1,
        "max_new_tokens": max(128, int(max_new_tokens or 1024)),
    }
    if forced_aligner:
        kwargs["forced_aligner"] = forced_aligner
        kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": kwargs["device_map"]}
    try:
        progress.emit(stage="model_load", percent=5, message=f"Loading {model} on {selected_device}")
        runtime = Qwen3ASRModel.from_pretrained(model_source, **kwargs)
    except RuntimeError as exc:
        code = "qwen3_asr_cuda_oom" if "out of memory" in str(exc).lower() else "qwen3_asr_runtime_failed"
        payload = _failure(output, code, str(exc), model=model)
        payload["fallback"] = {"recommended_model": "Qwen/Qwen3-ASR-0.6B", "automatic": False}
        return _finalize_payload(payload, output, report_path, progress)
    except Exception as exc:
        payload = _failure(output, "qwen3_asr_runtime_failed", str(exc), model=model)
        return _finalize_payload(payload, output, report_path, progress)

    chunk_duration = max(30, int(chunk_seconds or 300))
    attempt_limit = max(1, int(max_chunk_attempts or 1))
    selected_indexes = sorted(set(chunk_indexes or []))
    window_plan_path = output.with_name(f"{output.stem}-window-plan.json")
    try:
        media_duration = _media_duration_seconds(media)
        boundaries = fixed_chunk_boundaries(media_duration, chunk_duration)
        missing = sorted(set(selected_indexes) - set(range(len(boundaries))))
        if missing:
            raise ValueError(f"requested chunk indexes not found: {missing}")
        contract_seed = qwen_checkpoint_execution_contract(
            media=media,
            model=model,
            forced_aligner=forced_aligner,
            language=language,
            context=context,
            chunk_seconds=chunk_duration,
            max_new_tokens=max_new_tokens,
            dtype_name=dtype_name,
            chunk_indexes=chunk_indexes,
            input_sha256=input_sha256,
            timestamp_basis=input_timestamp_basis,
            vad_evidence_sha256=vad_evidence_sha256,
        )
        window_plan = build_qwen_window_plan(
            duration_seconds=media_duration,
            chunk_seconds=chunk_duration,
            execution_contract=contract_seed,
            chunk_indexes=selected_indexes,
            vad_intervals=vad_intervals,
        )
    except Exception as exc:
        payload = _failure(output, "qwen3_asr_chunking_failed", str(exc), model=model)
        return _finalize_payload(payload, output, report_path, progress)

    execution_contract = qwen_checkpoint_execution_contract(
        media=media,
        model=model,
        forced_aligner=forced_aligner,
        language=language,
        context=context,
        chunk_seconds=chunk_duration,
        max_new_tokens=max_new_tokens,
        dtype_name=dtype_name,
        chunk_indexes=chunk_indexes,
        window_plan_revision=str(window_plan["revision"]),
        input_sha256=input_sha256,
        timestamp_basis=input_timestamp_basis,
        vad_evidence_sha256=vad_evidence_sha256,
    )
    if resume and window_plan_path.is_file():
        try:
            existing_plan = read_json(window_plan_path)
        except Exception as exc:
            payload = _failure(output, "qwen3_asr_window_plan_rejected", str(exc), model=model)
            payload["window_plan_rejected_reason"] = "window_plan_unreadable"
            return _finalize_payload(payload, output, report_path, progress)
        if existing_plan != window_plan:
            if checkpoint_path.is_file():
                payload = _failure(
                    output,
                    "qwen3_asr_checkpoint_rejected",
                    "execution_contract_mismatch",
                    model=model,
                )
                payload["checkpoint_resume_rejected_reason"] = "execution_contract_mismatch"
                payload["checkpoint_path"] = str(checkpoint_path)
            else:
                payload = _failure(output, "qwen3_asr_window_plan_rejected", "window plan drift", model=model)
                payload["window_plan_rejected_reason"] = "window_plan_drift"
            return _finalize_payload(payload, output, report_path, progress)
    write_json(window_plan_path, window_plan)
    try:
        checkpoint = _load_checkpoint(
            checkpoint_path,
            expected_contract=execution_contract,
            window_plan=window_plan,
            resume=resume,
        )
    except QwenCheckpointRejected as exc:
        payload = _failure(output, "qwen3_asr_checkpoint_rejected", exc.reason, model=model)
        payload.update(
            checkpoint_path=str(checkpoint_path),
            checkpoint_resume_rejected_reason=exc.reason,
            window_plan_path=str(window_plan_path),
            window_plan=window_plan,
        )
        return _finalize_payload(payload, output, report_path, progress)
    rows: list[dict[str, Any]] = checkpoint["results"]
    failed_chunks: list[dict[str, Any]] = checkpoint["failed_chunks"]
    resumed_from_checkpoint = bool(checkpoint["resumed"])
    planned_windows = sorted(
        [dict(row) for row in window_plan["windows"]],
        key=lambda row: (int(row["execution_order"]), int(row["index"])),
    )
    requested_chunk_count = len(planned_windows)
    try:
        progress.emit(
            stage="chunking",
            percent=10,
            message=f"Planning {requested_chunk_count} recoverable audio windows",
        )
        # Keep derived chunks alongside the hash-bound checkpoint. Windows
        # private TEMP ACLs may reject the ffmpeg child even after mkdir works.
        # No broad cleanup: these are inspectable, resumable run artifacts.
        chunk_workspace = output.with_name(f"{output.stem}-audio-chunks")
        chunk_workspace.mkdir(parents=True, exist_ok=True)
        with nullcontext(str(chunk_workspace)) as temp_dir:
            completed_indexes = {int(row.get("chunk_index") or 0) for row in rows}
            pending_windows = [
                row for row in planned_windows if int(row["index"]) not in completed_indexes
            ]
            if completed_indexes:
                progress.emit(stage="resume", percent=10, message=f"Resuming after {len(completed_indexes)} checkpointed chunks")
            total = len(pending_windows)
            failed_dir = output.with_name(f"{output.stem}-failed-chunks")
            for current, window in enumerate(pending_windows, start=1):
                chunk_index = int(window["index"])
                window_id = str(window["window_id"])
                start = float(window["start_ms"]) / 1000.0
                end = float(window["end_ms"]) / 1000.0
                previous_failure = next(
                    (row for row in failed_chunks if int(row.get("chunk_index", -1)) == chunk_index),
                    None,
                )
                previous_attempt_count = int((previous_failure or {}).get("attempt_count") or 0)
                if previous_failure and (bool(previous_failure.get("retry_exhausted")) or previous_attempt_count >= attempt_limit):
                    progress.emit(
                        stage="retry_exhausted",
                        percent=10 + (80 * current / max(1, total)),
                        current_item=current,
                        total_items=total,
                        message=f"Skipping chunk {chunk_index}: retry limit reached",
                        details={"chunk_index": chunk_index, "attempt_count": previous_attempt_count, "max_chunk_attempts": attempt_limit},
                    )
                    continue
                failed_chunks = [
                    row
                    for row in failed_chunks
                    if int(row.get("chunk_index", -1)) != chunk_index
                ]
                attempt_count = previous_attempt_count + 1
                progress.emit(
                    stage="extraction",
                    percent=10 + (80 * (current - 1) / max(1, total)),
                    current_item=current,
                    total_items=total,
                    message=f"Extracting chunk {chunk_index}",
                )
                retry = _retry_command(
                    media=media,
                    output=output.with_name(f"{output.stem}-retry-chunk-{chunk_index:04d}.json"),
                    model=model,
                    forced_aligner=forced_aligner,
                    language=language,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    dtype_name=dtype_name,
                    chunk_seconds=chunk_duration,
                    chunk_index=chunk_index,
                    timestamp_basis=input_timestamp_basis,
                    vad_json=Path(vad_json).expanduser().resolve() if vad_json else None,
                    context=context,
                )
                try:
                    extracted = extract_audio_chunk_window(
                        media,
                        Path(temp_dir),
                        index=chunk_index,
                        start_seconds=start,
                        end_seconds=end,
                    )
                    chunk = Path(str(extracted["path"]))
                except Exception as exc:
                    failed_chunks.append(
                        {
                            "chunk_index": chunk_index,
                            "window_id": window_id,
                            "window_input_sha256": window["window_input_sha256"],
                            "start": float(start),
                            "end": float(end),
                            "reason": "chunk_extraction_failed",
                            "detail": str(exc),
                            "artifact_path": "",
                            "artifact_copy_error": "",
                            "retry_command": retry,
                            "attempt_count": attempt_count,
                            "retry_exhausted": attempt_count >= attempt_limit,
                        }
                    )
                    failed_chunks.sort(key=lambda row: int(row.get("chunk_index", -1)))
                    _write_checkpoint(
                        checkpoint_path,
                        media=media,
                        model=model,
                        chunk_seconds=chunk_duration,
                        forced_aligner=forced_aligner,
                        language=language,
                        results=rows,
                        failed_chunks=failed_chunks,
                        requested_chunk_count=requested_chunk_count,
                        execution_contract=execution_contract,
                        window_plan=window_plan,
                    )
                    progress.emit(
                        stage="extraction_failed",
                        percent=10 + (80 * current / max(1, total)),
                        current_item=current,
                        total_items=total,
                        message=f"Failed chunk {chunk_index} extraction; continuing",
                        details={"chunk_index": chunk_index, "failed_chunks": len(failed_chunks)},
                    )
                    continue

                progress.emit(
                    stage="transcription",
                    percent=10 + (80 * (current - 0.5) / max(1, total)),
                    current_item=current,
                    total_items=total,
                    message=f"Transcribing chunk {chunk_index}",
                )
                try:
                    values = list(
                        runtime.transcribe(
                            audio=_read_pcm_chunk(chunk),
                            context=context or None,
                            language=_language(language),
                            return_time_stamps=bool(forced_aligner),
                        )
                    )
                    if not values:
                        raise RuntimeError("empty_asr_result")
                    window_texts: list[str] = []
                    window_timestamps: list[dict[str, Any]] = []
                    window_segments: list[dict[str, Any]] = []
                    window_languages: list[str] = []
                    offset_applied = False
                    for value_index, value in enumerate(values, start=1):
                        text = str(getattr(value, "text", "") or "").strip()
                        normalized = normalize_qwen_window_timestamps(
                            _timestamps(getattr(value, "time_stamps", None)),
                            input_basis=input_timestamp_basis,
                            window_start_seconds=float(start),
                            window_end_seconds=float(end),
                        )
                        timestamps = normalized["timestamps"]
                        offset_applied = offset_applied or bool(normalized["offset_applied"])
                        window_texts.append(text)
                        window_languages.append(
                            str(getattr(value, "language", "") or language)
                        )
                        window_timestamps.extend(timestamps)
                        window_segments.extend(
                            _segments(
                                timestamps,
                                text,
                                fallback_start=float(start),
                                fallback_end=float(end),
                                segment_prefix=f"chunk-{chunk_index:04d}-result-{value_index:04d}",
                            )
                        )
                    rows.append(
                        {
                            "chunk_index": chunk_index,
                            "window_id": window_id,
                            "window_input_sha256": window["window_input_sha256"],
                            "chunk_offset_seconds": float(start),
                            "window_start_seconds": float(start),
                            "window_end_seconds": float(end),
                            "timestamp_input_basis": input_timestamp_basis,
                            "timestamp_basis": "media_absolute",
                            "timestamp_offset_applied": offset_applied,
                            "text": "\n".join(value for value in window_texts if value).strip(),
                            "language": window_languages[0] if window_languages else language,
                            "timestamps": window_timestamps,
                            "segments": window_segments,
                        }
                    )
                except Exception as exc:
                    code = "qwen3_asr_cuda_oom" if "out of memory" in str(exc).lower() else "chunk_transcription_failed"
                    durable_path = ""
                    copy_error = ""
                    try:
                        failed_dir.mkdir(parents=True, exist_ok=True)
                        durable = failed_dir / chunk.name
                        shutil.copy2(chunk, durable)
                        durable_path = str(durable)
                    except Exception as copy_exc:  # pragma: no cover - defensive filesystem evidence.
                        copy_error = str(copy_exc)
                    failed_chunks.append(
                        {
                            "chunk_index": chunk_index,
                            "window_id": window_id,
                            "window_input_sha256": window["window_input_sha256"],
                            "start": float(start),
                            "end": float(end),
                            "reason": code,
                            "detail": str(exc),
                            "artifact_path": durable_path,
                            "artifact_copy_error": copy_error,
                            "retry_command": retry,
                            "attempt_count": attempt_count,
                            "retry_exhausted": attempt_count >= attempt_limit,
                        }
                    )
                rows.sort(key=lambda row: int(row.get("chunk_index", -1)))
                failed_chunks.sort(key=lambda row: int(row.get("chunk_index", -1)))
                _write_checkpoint(
                    checkpoint_path,
                    media=media,
                    model=model,
                    chunk_seconds=chunk_duration,
                    forced_aligner=forced_aligner,
                    language=language,
                    results=rows,
                    failed_chunks=failed_chunks,
                    requested_chunk_count=requested_chunk_count,
                    execution_contract=execution_contract,
                    window_plan=window_plan,
                )
                chunk.unlink(missing_ok=True)
                progress.emit(
                    stage="transcription",
                    percent=10 + (80 * current / max(1, total)),
                    current_item=current,
                    total_items=total,
                    message=f"Finished chunk {chunk_index}",
                    details={"failed_chunks": len(failed_chunks), "successful_results": len(rows)},
                )
    except Exception as exc:
        payload = _failure(output, "qwen3_asr_chunking_failed", str(exc), model=model)
        return _finalize_payload(payload, output, report_path, progress)

    successful_chunks = sorted({int(row.get("chunk_index") or 0) for row in rows})
    status = "degraded" if failed_chunks and rows else ("failed" if failed_chunks else "completed")
    payload = {
        "schema": SCHEMA,
        "provider": "qwen3-asr",
        "model": model,
        "forced_aligner": forced_aligner,
        "chunk_seconds": chunk_duration,
        "chunk_count": requested_chunk_count,
        "max_chunk_attempts": attempt_limit,
        "retry_exhausted_chunk_count": sum(1 for row in failed_chunks if bool(row.get("retry_exhausted"))),
        "checkpoint_path": str(checkpoint_path),
        "window_plan_path": str(window_plan_path),
        "window_plan": window_plan,
        "window_plan_revision": str(window_plan["revision"]),
        "input_sha256": input_sha256,
        "timestamp_input_basis": input_timestamp_basis,
        "timestamp_output_basis": "media_absolute",
        "vad_evidence_sha256": vad_evidence_sha256,
        "execution_order": [int(row["index"]) for row in planned_windows],
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "checkpointed_successful_chunk_count": len(checkpoint.get("successful_chunk_indexes") or []),
        "successful_chunk_count": len(successful_chunks),
        "failed_chunk_count": len(failed_chunks),
        "successful_chunk_indexes": successful_chunks,
        "device": selected_device,
        "dtype": str(dtype).replace("torch.", ""),
        "input_path": str(media),
        "ok": status == "completed",
        "usable": bool(rows),
        "status": status,
        "results": rows,
        "segments": [segment for row in rows for segment in row.get("segments") or []],
        "text": "\n".join(str(row.get("text") or "") for row in rows).strip(),
        "failed_chunks": failed_chunks,
        "gaps": [
            {
                "chunk_index": row["chunk_index"],
                "start": row["start"],
                "end": row["end"],
                "reason": row["reason"],
            }
            for row in failed_chunks
        ],
        "retry_commands": [row["retry_command"] for row in failed_chunks],
        "fallback": {"recommended_model": "Qwen/Qwen3-ASR-0.6B", "automatic": False},
    }
    _write_checkpoint(
        checkpoint_path,
        media=media,
        model=model,
        chunk_seconds=chunk_duration,
        forced_aligner=forced_aligner,
        language=language,
        results=rows,
        failed_chunks=failed_chunks,
        requested_chunk_count=requested_chunk_count,
        execution_contract=execution_contract,
        window_plan=window_plan,
        status=status,
    )
    return _finalize_payload(payload, output, report_path, progress)


def _checkpoint_default() -> dict[str, Any]:
    return {
        "resumed": False,
        "results": [],
        "failed_chunks": [],
        "successful_chunk_indexes": [],
    }


def _load_checkpoint(
    path: Path,
    *,
    expected_contract: dict[str, Any] | None = None,
    window_plan: dict[str, Any] | None = None,
    media: Path | None = None,
    model: str = "",
    chunk_seconds: int = 300,
    forced_aligner: str = "",
    language: str = "Chinese",
    context: str = "",
    max_new_tokens: int = 1024,
    dtype_name: str = "auto",
    chunk_indexes: list[int] | None = None,
    window_plan_revision: str | None = None,
    resume: bool,
) -> dict[str, Any]:
    default = _checkpoint_default()
    if not resume or not path.is_file():
        return default
    try:
        payload = read_json(path)
    except Exception as exc:
        if expected_contract is not None:
            raise QwenCheckpointRejected("checkpoint_unreadable") from exc
        return default
    if not isinstance(payload, dict):
        if expected_contract is not None:
            raise QwenCheckpointRejected("checkpoint_not_object")
        return default
    if expected_contract is not None:
        if window_plan is None:
            raise QwenCheckpointRejected("window_plan_missing")
        if str(payload.get("schema") or "") == "video_knowledge_pipeline.qwen3_asr_checkpoint.v1":
            if not qwen_checkpoint_matches(payload, expected_contract):
                raise QwenCheckpointRejected("execution_contract_mismatch")
            rows = [dict(row) for row in payload.get("results") or [] if isinstance(row, dict)]
            failures = [dict(row) for row in payload.get("failed_chunks") or [] if isinstance(row, dict)]
            indexes = [int(row.get("chunk_index", -1)) for row in rows]
            if len(set(indexes)) != len(indexes):
                raise QwenCheckpointRejected("checkpoint_duplicate_result")
            if indexes != sorted(indexes):
                raise QwenCheckpointRejected("checkpoint_result_order_invalid")
            plan_by_index = {int(row["index"]): row for row in window_plan.get("windows") or []}
            if any(index not in plan_by_index for index in indexes):
                raise QwenCheckpointRejected("checkpoint_result_index_out_of_range")
            for row in rows:
                planned = plan_by_index[int(row["chunk_index"])]
                row.setdefault("window_id", planned["window_id"])
                row.setdefault("window_input_sha256", planned["window_input_sha256"])
                row.setdefault("window_start_seconds", float(planned["start_ms"]) / 1000.0)
                row.setdefault("window_end_seconds", float(planned["end_ms"]) / 1000.0)
                row.setdefault("timestamp_input_basis", "legacy_media_absolute")
                row.setdefault("timestamp_basis", "media_absolute")
                row.setdefault("timestamp_offset_applied", True)
            successful = sorted(indexes)
            return {
                "resumed": bool(rows or failures),
                "results": rows,
                "failed_chunks": failures,
                "successful_chunk_indexes": successful,
                "legacy_checkpoint_migrated": True,
            }
        errors = qwen_checkpoint_validation_errors(
            payload,
            expected_contract=expected_contract,
            expected_window_plan=window_plan,
        )
        if errors:
            raise QwenCheckpointRejected(errors[0])
    else:
        if media is None:
            raise ValueError("media is required for legacy checkpoint validation")
        legacy_contract = qwen_checkpoint_execution_contract(
            media=media,
            model=model,
            forced_aligner=forced_aligner,
            language=language,
            context=context,
            chunk_seconds=chunk_seconds,
            max_new_tokens=max_new_tokens,
            dtype_name=dtype_name,
            chunk_indexes=chunk_indexes,
            window_plan_revision=window_plan_revision,
        )
        if not qwen_checkpoint_matches(payload, legacy_contract):
            return default
    rows = [dict(row) for row in payload.get("results") or [] if isinstance(row, dict)]
    failures = [dict(row) for row in payload.get("failed_chunks") or [] if isinstance(row, dict)]
    successful = sorted({int(row.get("chunk_index") or 0) for row in rows})
    return {
        "resumed": bool(rows or failures),
        "results": rows,
        "failed_chunks": failures,
        "successful_chunk_indexes": successful,
    }


def _write_checkpoint(
    path: Path,
    *,
    media: Path,
    model: str,
    chunk_seconds: int,
    forced_aligner: str,
    language: str,
    results: list[dict[str, Any]],
    failed_chunks: list[dict[str, Any]],
    requested_chunk_count: int,
    execution_contract: dict[str, Any],
    window_plan: dict[str, Any],
    status: str = "running",
) -> None:
    successful = sorted({int(row.get("chunk_index") or 0) for row in results})
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "status": status,
        "input_identity": dict(execution_contract["input_identity"]),
        "execution_contract": dict(execution_contract),
        "window_plan_revision": str(window_plan["revision"]),
        "window_plan": window_plan,
        "model": model,
        "forced_aligner": forced_aligner,
        "language": language,
        "chunk_seconds": int(chunk_seconds),
        "requested_chunk_count": int(requested_chunk_count),
        "successful_chunk_indexes": successful,
        "successful_chunk_count": len(successful),
        "failed_chunk_count": len(failed_chunks),
        "results": results,
        "failed_chunks": failed_chunks,
    }
    payload["checkpoint_sha256"] = qwen_checkpoint_payload_sha256(payload)
    write_json(path, payload)


def qwen_checkpoint_execution_contract(
    *,
    media: Path,
    model: str,
    forced_aligner: str,
    language: str,
    context: str,
    chunk_seconds: int,
    max_new_tokens: int,
    dtype_name: str,
    chunk_indexes: list[int] | None,
    window_plan_revision: str | None = None,
    input_sha256: str | None = None,
    timestamp_basis: str = "chunk_relative",
    vad_evidence_sha256: str = "",
) -> dict[str, Any]:
    """Return the semantic identity used to accept checkpointed ASR text."""
    resolved = media.expanduser().resolve()
    stat = resolved.stat()
    digest = str(input_sha256 or sha256_file(resolved)).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("input_sha256_invalid")
    basis = str(timestamp_basis or "chunk_relative").strip().lower()
    if basis not in TIMESTAMP_BASES:
        raise ValueError("timestamp_basis_invalid")
    contract = {
        "input_identity": {
            "path": str(resolved),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest,
        },
        "model": str(model),
        "forced_aligner": str(forced_aligner),
        "language": str(language),
        "context_sha256": hashlib.sha256(str(context).encode("utf-8")).hexdigest(),
        "chunk_seconds": int(chunk_seconds),
        "max_new_tokens": max(128, int(max_new_tokens or 1024)),
        "dtype": str(dtype_name or "auto").strip().lower(),
        "chunk_indexes": sorted({int(value) for value in chunk_indexes or []}),
        "timestamp_input_basis": basis,
        "vad_evidence_sha256": str(vad_evidence_sha256 or ""),
    }
    if window_plan_revision:
        contract["window_plan_revision"] = str(window_plan_revision)
    return contract


def build_qwen_window_plan(
    *,
    duration_seconds: float,
    chunk_seconds: int,
    execution_contract: dict[str, Any],
    chunk_indexes: list[int] | None = None,
    vad_intervals: list[dict[str, Any]] | None = None,
    audio_activity_intervals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, source-ordered plan with a separate execution order."""

    duration = float(duration_seconds)
    chunk_duration = int(chunk_seconds)
    if not math.isfinite(duration) or duration <= 0 or chunk_duration <= 0:
        raise ValueError("window_plan_duration_invalid")
    identity = execution_contract.get("input_identity") or {}
    input_sha256 = str(identity.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("window_plan_input_sha256_invalid")
    all_boundaries = fixed_chunk_boundaries(duration, chunk_duration)
    selected = sorted({int(value) for value in chunk_indexes or []})
    unknown = sorted(set(selected) - set(range(len(all_boundaries))))
    if unknown:
        raise ValueError(f"requested chunk indexes not found: {unknown}")
    parameter_contract = {
        key: value
        for key, value in execution_contract.items()
        if key != "window_plan_revision"
    }
    parameter_revision = canonical_json_sha256(parameter_contract)
    vad_rows = _normalized_priority_intervals(vad_intervals or [])
    activity_rows = _normalized_priority_intervals(audio_activity_intervals or [])
    windows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(all_boundaries):
        if selected and index not in selected:
            continue
        start_ms = int(round(float(start) * 1000))
        end_ms = int(round(float(end) * 1000))
        speech_support = [row for row in vad_rows if _intervals_overlap(start, end, row["start"], row["end"])]
        activity_support = [row for row in activity_rows if _intervals_overlap(start, end, row["start"], row["end"])]
        high_confidence = any(float(row["confidence"]) >= VAD_CONFIDENCE_THRESHOLD for row in speech_support)
        low_confidence = bool(speech_support) and not high_confidence
        if high_confidence:
            priority = "vad_speech_first"
            priority_rank = 0
            review_required = False
        elif low_confidence:
            priority = "low_confidence_review_keep"
            priority_rank = 1
            review_required = True
        elif activity_support:
            priority = "audio_activity_review_keep"
            priority_rank = 2
            review_required = True
        else:
            priority = "source_order_keep"
            priority_rank = 3
            review_required = False
        identity_seed = {
            "schema": WINDOW_PLAN_SCHEMA,
            "input_sha256": input_sha256,
            "index": index,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "parameter_revision": parameter_revision,
        }
        window_id = f"qwen-window-{canonical_json_sha256(identity_seed)[:24]}"
        windows.append(
            {
                "index": index,
                "window_id": window_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "input_sha256": input_sha256,
                "window_input_sha256": canonical_json_sha256(identity_seed),
                "parameter_revision": parameter_revision,
                "dependencies": [
                    {"kind": "input", "sha256": input_sha256},
                    {"kind": "parameters", "sha256": parameter_revision},
                ],
                "priority": priority,
                "priority_rank": priority_rank,
                "review_required": review_required,
                "retention": "keep",
                "speech_proven": high_confidence,
                "vad_support": speech_support,
                "audio_activity_support": activity_support,
            }
        )
    execution_rows = sorted(windows, key=lambda row: (int(row["priority_rank"]), int(row["index"])))
    execution_order = {int(row["index"]): position for position, row in enumerate(execution_rows)}
    for row in windows:
        row["execution_order"] = execution_order[int(row["index"])]
        row.pop("priority_rank", None)
    plan_without_revision = {
        "schema": WINDOW_PLAN_SCHEMA,
        "duration_ms": int(round(duration * 1000)),
        "chunk_seconds": chunk_duration,
        "input_sha256": input_sha256,
        "parameter_revision": parameter_revision,
        "timestamp_input_basis": str(execution_contract.get("timestamp_input_basis") or "chunk_relative"),
        "vad_evidence_sha256": str(execution_contract.get("vad_evidence_sha256") or ""),
        "selected_indexes": selected,
        "windows": windows,
    }
    return {**plan_without_revision, "revision": canonical_json_sha256(plan_without_revision)}


def _normalized_priority_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for position, value in enumerate(rows):
        if not isinstance(value, dict):
            raise ValueError("priority_interval_not_object")
        start = float(value.get("start", value.get("start_seconds", 0.0)) or 0.0)
        end = float(value.get("end", value.get("end_seconds", start)) or start)
        confidence = float(value.get("confidence", 0.0) or 0.0)
        if not all(math.isfinite(item) for item in (start, end, confidence)) or start < 0 or end <= start:
            raise ValueError("priority_interval_invalid")
        normalized.append(
            {
                "segment_id": str(value.get("segment_id") or value.get("id") or f"interval-{position:04d}"),
                "start": round(start, 6),
                "end": round(end, 6),
                "confidence": round(max(0.0, min(1.0, confidence)), 6),
                "candidate_only": bool(value.get("candidate_only", False)),
            }
        )
    return sorted(normalized, key=lambda row: (row["start"], row["end"], row["segment_id"]))


def _intervals_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return max(float(left_start), float(right_start)) < min(float(left_end), float(right_end))


def _load_vad_priority_evidence(path: str | Path | None) -> tuple[list[dict[str, Any]], str]:
    if not path:
        return [], ""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"vad evidence not found: {source}")
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("vad evidence must be an object")
    rows = payload.get("segments", payload.get("intervals", []))
    if not isinstance(rows, list):
        raise ValueError("vad evidence segments must be a list")
    return _normalized_priority_intervals(rows), sha256_file(source)


def qwen_checkpoint_payload_sha256(payload: dict[str, Any]) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != "checkpoint_sha256"})


def qwen_checkpoint_validation_errors(
    payload: dict[str, Any],
    *,
    expected_contract: dict[str, Any],
    expected_window_plan: dict[str, Any],
) -> list[str]:
    """Validate the complete durable checkpoint before any resumed inference."""

    if str(payload.get("schema") or "") != CHECKPOINT_SCHEMA:
        return ["checkpoint_schema_mismatch"]
    if str(payload.get("checkpoint_sha256") or "") != qwen_checkpoint_payload_sha256(payload):
        return ["checkpoint_hash_mismatch"]
    if payload.get("execution_contract") != expected_contract:
        return ["execution_contract_mismatch"]
    if payload.get("window_plan") != expected_window_plan:
        return ["window_plan_mismatch"]
    if str(payload.get("window_plan_revision") or "") != str(expected_window_plan.get("revision") or ""):
        return ["window_plan_revision_mismatch"]
    plan_rows = expected_window_plan.get("windows") or []
    expected_by_index = {int(row["index"]): row for row in plan_rows}
    if int(payload.get("requested_chunk_count") or -1) != len(plan_rows):
        return ["checkpoint_requested_count_mismatch"]
    results = payload.get("results")
    failures = payload.get("failed_chunks")
    if not isinstance(results, list) or not isinstance(failures, list):
        return ["checkpoint_rows_invalid"]
    result_indexes = [int(row.get("chunk_index", -1)) for row in results if isinstance(row, dict)]
    if len(result_indexes) != len(results):
        return ["checkpoint_result_invalid"]
    if len(set(result_indexes)) != len(result_indexes):
        return ["checkpoint_duplicate_result"]
    if result_indexes != sorted(result_indexes):
        return ["checkpoint_result_order_invalid"]
    declared_success = [int(value) for value in payload.get("successful_chunk_indexes") or []]
    if declared_success != result_indexes:
        return ["checkpoint_success_index_mismatch"]
    if int(payload.get("successful_chunk_count") or 0) != len(result_indexes):
        return ["checkpoint_success_count_mismatch"]
    for row in results:
        index = int(row["chunk_index"])
        expected = expected_by_index.get(index)
        if expected is None:
            return ["checkpoint_result_index_out_of_range"]
        if str(row.get("window_id") or "") != str(expected["window_id"]):
            return ["checkpoint_result_window_id_mismatch"]
        if str(row.get("window_input_sha256") or "") != str(expected["window_input_sha256"]):
            return ["checkpoint_result_input_hash_mismatch"]
        if str(row.get("timestamp_basis") or "") != "media_absolute":
            return ["checkpoint_result_timestamp_basis_invalid"]
    failure_indexes = [int(row.get("chunk_index", -1)) for row in failures if isinstance(row, dict)]
    if len(failure_indexes) != len(failures) or len(set(failure_indexes)) != len(failure_indexes):
        return ["checkpoint_failure_rows_invalid"]
    if failure_indexes != sorted(failure_indexes):
        return ["checkpoint_failure_order_invalid"]
    if set(failure_indexes) & set(result_indexes):
        return ["checkpoint_success_failure_overlap"]
    if any(index not in expected_by_index for index in failure_indexes):
        return ["checkpoint_failure_index_out_of_range"]
    if int(payload.get("failed_chunk_count") or 0) != len(failure_indexes):
        return ["checkpoint_failure_count_mismatch"]
    return []


def qwen_window_plan_revision(
    *,
    media: Path,
    chunk_seconds: int,
    boundaries: list[tuple[float, float]],
) -> str:
    """Bind checkpoint identity to the deterministic fixed-window plan."""

    resolved = media.expanduser().resolve()
    stat = resolved.stat()
    return canonical_json_sha256(
        {
            "schema": "video_knowledge_pipeline.qwen_window_plan.v1",
            "input_identity": {
                "path": str(resolved),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            },
            "chunk_seconds": int(chunk_seconds),
            "windows": [
                {
                    "index": index,
                    "start_seconds": round(float(start), 6),
                    "end_seconds": round(float(end), 6),
                }
                for index, (start, end) in enumerate(boundaries)
            ],
        }
    )


def qwen_checkpoint_matches(payload: dict[str, Any], expected_contract: dict[str, Any]) -> bool:
    """Validate new exact contracts while retaining bounded legacy resume."""
    stored_contract = payload.get("execution_contract")
    if isinstance(stored_contract, dict):
        if str(payload.get("schema") or "") == "video_knowledge_pipeline.qwen3_asr_checkpoint.v1":
            projected: dict[str, Any] = {}
            for key, value in stored_contract.items():
                expected_value = expected_contract.get(key)
                if key == "input_identity" and isinstance(value, dict) and isinstance(expected_value, dict):
                    projected[key] = {nested: expected_value.get(nested) for nested in value}
                else:
                    projected[key] = expected_value
            return stored_contract == projected
        return stored_contract == expected_contract

    identity = payload.get("input_identity")
    expected_identity = expected_contract["input_identity"]
    if not isinstance(identity, dict):
        return False
    if (
        str(identity.get("path") or "") != str(expected_identity["path"])
        or int(identity.get("bytes") or -1) != int(expected_identity["bytes"])
        or str(payload.get("model") or "") != str(expected_contract["model"])
        or int(payload.get("chunk_seconds") or 0) != int(expected_contract["chunk_seconds"])
    ):
        return False
    if "forced_aligner" in payload and str(payload.get("forced_aligner") or "") != str(expected_contract["forced_aligner"]):
        return False
    if "language" in payload and str(payload.get("language") or "") != str(expected_contract["language"]):
        return False
    return True

def _torch_dtype(value: str, device: str, torch: Any) -> Any:
    requested = str(value or "auto").strip().lower()
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if str(device).startswith("cuda") else torch.float32

def _timestamps(values: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values or []:
        if isinstance(value, dict):
            text = value.get("text")
            start = value.get("start_time")
            end = value.get("end_time")
        else:
            text = getattr(value, "text", "")
            start = getattr(value, "start_time", 0.0)
            end = getattr(value, "end_time", start)
        rows.append({"text": str(text or ""), "start": float(start or 0.0), "end": float(end or start or 0.0)})
    return rows


def _segments(
    timestamps: list[dict[str, Any]],
    text: str,
    *,
    fallback_start: float = 0.0,
    fallback_end: float | None = None,
    segment_prefix: str = "segment",
) -> list[dict[str, Any]]:
    if not timestamps:
        return [
            {
                "segment_id": f"{segment_prefix}-0001",
                "source_segment_ids": [f"{segment_prefix}-0001"],
                "start": fallback_start,
                "end": fallback_end if fallback_end is not None else fallback_start,
                "timestamp_precision": "audio_window_only",
                "text": text,
            }
        ] if text else []
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in timestamps:
        current.append(row)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        token = str(row.get("text") or "")
        if duration >= 15.0 or re.search(r"[。！？!?]$", token):
            segments.append(_merge(current))
            current = []
    if current:
        segments.append(_merge(current))
    for index, segment in enumerate(segments, start=1):
        segment_id = f"{segment_prefix}-{index:04d}"
        segment["segment_id"] = segment_id
        segment["source_segment_ids"] = [segment_id]
    return segments


def _media_duration_seconds(media: Path) -> float:
    ffprobe = str(resolve_media_tool("ffprobe") or "")
    if not ffprobe:
        raise RuntimeError("ffprobe_not_ready_for_qwen3_asr_window_planning")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env=local_tool_subprocess_env(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "qwen3_asr_duration_probe_failed: "
            f"returncode={completed.returncode}; stderr={completed.stderr[-500:]}"
        )
    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("qwen3_asr_duration_probe_invalid") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("qwen3_asr_duration_probe_invalid")
    return duration


def _audio_chunks(media: Path, output_dir: Path, *, chunk_seconds: int) -> list[Path]:
    """Compatibility helper for FunASR; Qwen execution no longer calls it."""

    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_ready_for_qwen3_asr_chunking")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "chunk-%04d.wav"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "segment",
        "-segment_time", str(chunk_seconds), "-reset_timestamps", "1", str(pattern),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    chunks = sorted(output_dir.glob("chunk-*.wav"))
    if completed.returncode != 0 or not chunks:
        raise RuntimeError(f"qwen3_asr_audio_chunking_failed: {completed.stderr[-500:]}")
    return chunks


def _offset_timestamps(rows: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    """Legacy relative-timestamp helper retained for external callers."""
    return [
        {
            **row,
            "start": round(float(row.get("start") or 0.0) + offset, 6),
            "end": round(float(row.get("end") or row.get("start") or 0.0) + offset, 6),
        }
        for row in rows
    ]


def normalize_qwen_window_timestamps(
    rows: list[dict[str, Any]],
    *,
    input_basis: str,
    window_start_seconds: float,
    window_end_seconds: float,
) -> dict[str, Any]:
    """Normalize Qwen timestamps to absolute media time exactly once."""

    basis = str(input_basis or "").strip().lower()
    if basis not in TIMESTAMP_BASES:
        raise ValueError("timestamp_basis_invalid")
    window_start = float(window_start_seconds)
    window_end = float(window_end_seconds)
    if not math.isfinite(window_start) or not math.isfinite(window_end) or window_start < 0 or window_end < window_start:
        raise ValueError("timestamp_window_invalid")
    normalized: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"timestamp_row_invalid:{position}")
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", row.get("start", 0.0)) or start)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError(f"timestamp_value_invalid:{position}")
        if basis == "chunk_relative":
            start += window_start
            end += window_start
        if start < window_start - 0.001 or end > window_end + 0.001:
            raise ValueError(f"timestamp_outside_window:{position}")
        normalized.append(
            {
                **row,
                "start": round(start, 6),
                "end": round(end, 6),
            }
        )
    return {
        "timestamps": normalized,
        "input_basis": basis,
        "output_basis": "media_absolute",
        "offset_applied": basis == "chunk_relative" and bool(rows),
    }


def _merge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start": float(rows[0]["start"]),
        "end": float(rows[-1]["end"]),
        "text": "".join(str(row.get("text") or "") for row in rows).strip(),
        "words": rows,
    }


def _device(value: str, torch: Any) -> str:
    requested = str(value or "auto").lower()
    if requested == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _language(value: str) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "auto":
        return None
    # Intent: make the existing VKP/FunASR ``yue`` contract select Qwen's
    # reviewed Cantonese decoder instead of passing an unsupported alias.
    # Decision: keep aliases at the adapter boundary and forward Qwen's
    # canonical language name to the upstream package.
    # Reason: quality-benchmark and plan-asr historically share ISO-style
    # language values; Qwen3-ASR validates canonical display names.
    # Evidence: the pinned qwen-asr 0.0.6 model config lists ``Cantonese`` and
    # its ``transcribe`` API validates the supplied language before inference.
    # Effective scope: local Qwen3-ASR requests only; SenseVoice and transcript
    # text remain unchanged.
    aliases = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "cn": "Chinese",
        "yue": "Cantonese",
        "zh-yue": "Cantonese",
        "cantonese": "Cantonese",
        "en": "English",
    }
    return aliases.get(text.lower(), text)


def _chunk_indexes(value: str) -> list[int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    indexes = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if any(value < 0 for value in indexes):
        raise ValueError("chunk indexes must be zero or greater")
    return indexes


def _retry_command(
    *,
    media: Path,
    output: Path,
    model: str,
    forced_aligner: str,
    language: str,
    device: str,
    max_new_tokens: int,
    dtype_name: str,
    chunk_seconds: int,
    chunk_index: int,
    timestamp_basis: str = "chunk_relative",
    vad_json: Path | None = None,
    context: str = "",
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "video_knowledge_pipeline.qwen3_asr_python_runner",
        "--input",
        str(media),
        "--output",
        str(output),
        "--model",
        model,
        "--language",
        language,
        "--device",
        device,
        "--max-new-tokens",
        str(max_new_tokens),
        "--dtype",
        dtype_name,
        "--chunk-seconds",
        str(chunk_seconds),
        "--chunk-indexes",
        str(chunk_index),
        "--timestamp-basis",
        str(timestamp_basis),
    ]
    if forced_aligner:
        command.extend(["--forced-aligner", forced_aligner])
    else:
        command.append("--no-timestamps")
    if vad_json is not None:
        command.extend(["--vad-json", str(vad_json)])
    if str(context or "").strip():
        command.extend(["--context", str(context).strip()])
    return {"chunk_index": chunk_index, "command": command, "powershell": subprocess.list2cmdline(command)}


def _finalize_payload(
    payload: dict[str, Any],
    output: Path,
    report_path: Path,
    progress: LocalMediaProgress,
) -> dict[str, Any]:
    status = str(payload.get("status") or ("completed" if payload.get("ok") else "failed"))
    payload["status"] = status
    payload["output_path"] = str(output)
    payload["report_path"] = str(report_path)
    payload["progress"] = progress.artifacts()
    write_json(output, payload)
    report = {
        "schema": "video_knowledge_pipeline.local_asr_chunk_report.v1",
        "status": status,
        "input_path": payload.get("input_path", ""),
        "output_path": str(output),
        "successful_chunk_count": int(payload.get("successful_chunk_count") or 0),
        "failed_chunk_count": int(payload.get("failed_chunk_count") or 0),
        "failed_chunks": payload.get("failed_chunks") or [],
        "gaps": payload.get("gaps") or [],
        "retry_commands": payload.get("retry_commands") or [],
        "raw_successful_content_preserved": bool(payload.get("results")),
    }
    write_json(report_path, report)
    terminal = status if status in {"completed", "failed", "degraded"} else "failed"
    progress.emit(
        stage="finalize",
        percent=100,
        current_item=int(payload.get("successful_chunk_count") or 0) + int(payload.get("failed_chunk_count") or 0),
        total_items=int(payload.get("chunk_count") or 0),
        message=(
            "Local ASR completed"
            if terminal == "completed"
            else "Local ASR completed with missing chunks"
            if terminal == "degraded"
            else "Local ASR failed"
        ),
        status=terminal,
        output_paths=[output],
        report_paths=[report_path],
        details={
            "successful_chunk_count": int(payload.get("successful_chunk_count") or 0),
            "failed_chunk_count": int(payload.get("failed_chunk_count") or 0),
        },
    )
    return payload


def _failure(output: Path, code: str, error: str, *, model: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "provider": "qwen3-asr",
        "model": model,
        "ok": False,
        "usable": False,
        "status": "failed",
        "error_code": code,
        "error": error,
        "output_path": str(output),
    }


if __name__ == "__main__":
    raise SystemExit(main())
