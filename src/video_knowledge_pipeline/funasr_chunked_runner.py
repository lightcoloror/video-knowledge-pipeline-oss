from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .asr_local_agreement import (
    measure_boundary_lcs_dedup,
    measure_local_agreement,
)
from .canonical_json import canonical_json_sha256
from .audio_chunk_manifest import (
    build_fixed_chunk_manifest,
    prepare_fixed_overlap_chunks,
    prepare_silence_snapped_chunks,
)
from .local_media_progress import LocalMediaProgress, stderr_progress_callback
from .qwen3_asr_python_runner import _audio_chunks
from .speaker_global_alignment import (
    DEFAULT_OVERLAP_AGREEMENT_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    align_chunk_speaker_records,
    write_alignment_artifacts,
)
from .storage import read_json, write_json


SCHEMA = "video_knowledge_funasr_chunked_raw_output.v1"
CHECKPOINT_SCHEMA = "video_knowledge_pipeline.funasr_chunked_asr_checkpoint.v2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="funasr-chunked-runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default="sensevoice")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="")
    parser.add_argument("--spk-model", default="")
    parser.add_argument("--speaker-merge-threshold", type=float)
    parser.add_argument("--preset-speaker-count", type=int)
    parser.add_argument(
        "--speaker-global-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="FunASR-compatible cosine threshold for chunk-global anonymous IDs",
    )
    parser.add_argument("--language", default="zh")
    parser.add_argument("--hotword", default="")
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--use-itn", action="store_true", dest="use_itn")
    parser.add_argument("--no-use-itn", action="store_false", dest="use_itn")
    parser.set_defaults(use_itn=True)
    parser.add_argument("--merge-vad", action="store_true", dest="merge_vad")
    parser.add_argument("--no-merge-vad", action="store_false", dest="merge_vad")
    parser.set_defaults(merge_vad=True)
    parser.add_argument("--merge-length-s", type=int, default=15)
    parser.add_argument("--vad-max-single-segment-time-ms", type=int, default=30000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--chunk-seconds", type=int, default=300)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=5.0)
    parser.add_argument(
        "--chunk-boundary-mode",
        choices=["fixed_duration", "silence_snap"],
        default="fixed_duration",
    )
    parser.add_argument("--silence-noise-floor-db", type=float, default=-30.0)
    parser.add_argument("--minimum-silence-seconds", type=float, default=0.5)
    parser.add_argument("--silence-max-offset-seconds", type=float, default=10.0)
    parser.add_argument("--chunk-indexes", default="")
    parser.add_argument("--max-chunk-attempts", type=int, default=3)
    parser.add_argument("--rebuild-from-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    result = run_funasr_chunked(
        input_path=args.input,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        spk_model=args.spk_model,
        speaker_merge_threshold=args.speaker_merge_threshold,
        preset_speaker_count=args.preset_speaker_count,
        speaker_global_threshold=args.speaker_global_threshold,
        language=args.language,
        hotword=args.hotword,
        batch_size_s=args.batch_size_s,
        use_itn=args.use_itn,
        merge_vad=args.merge_vad,
        merge_length_s=args.merge_length_s,
        vad_max_single_segment_time_ms=args.vad_max_single_segment_time_ms,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        chunk_boundary_mode=args.chunk_boundary_mode,
        silence_noise_floor_db=args.silence_noise_floor_db,
        minimum_silence_seconds=args.minimum_silence_seconds,
        silence_max_offset_seconds=args.silence_max_offset_seconds,
        chunk_indexes=_chunk_indexes(args.chunk_indexes),
        max_chunk_attempts=args.max_chunk_attempts,
        rebuild_from_checkpoint=args.rebuild_from_checkpoint,
        progress_callback=stderr_progress_callback,
    )
    # Intent: keep child-process transport bounded for long transcripts.
    # Decision: print only an execution receipt; retain full ASR in output JSON.
    # Reason: echoing full results duplicated megabytes in memory, logs, and UI.
    # Evidence: asr_execution already resolves the declared output path first.
    # Effective scope: CLI stdout only; artifact schema/content are unchanged.
    print(json.dumps(_cli_result_summary(result), ensure_ascii=False))
    return 0 if result.get("status") in {"completed", "partial_targeted_completed"} else 2 if result.get("status") == "degraded" else 1


def _cli_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.funasr_chunked_runner_result.v1",
        "status": str(result.get("status") or ""),
        "ok": bool(result.get("ok")),
        "usable": bool(result.get("usable")),
        "quality_status": str(result.get("quality_status") or ""),
        "output_path": str(result.get("output_path") or ""),
        "report_path": str(result.get("report_path") or ""),
        "successful_chunk_count": int(result.get("successful_chunk_count") or 0),
        "failed_chunk_count": int(result.get("failed_chunk_count") or 0),
        "unresolved_chunk_indexes": list(result.get("unresolved_chunk_indexes") or []),
        "progress": dict(result.get("progress") or {}),
    }

def _chunk_runtime_summary(chunks_dir: Path, indexes: list[int]) -> dict[str, Any]:
    """Aggregate child telemetry without copying transcript content.

    Intent: expose duration and peak GPU pressure at the parent run level.
    Decision: read only ``runtime_metrics`` from successful child artifacts.
    Reason: parent checkpoints intentionally retain normalized records, not the
    child model telemetry envelope.
    Evidence: FunASR child payloads emit PyTorch elapsed/peak counters locally.
    Effective scope: run report telemetry only; transcript and retry semantics
    are unchanged.
    """

    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for index in indexes:
        path = chunks_dir / f"chunk-{int(index):04d}.json"
        if not path.is_file():
            missing.append(int(index))
            continue
        payload = read_json(path)
        metrics = payload.get("runtime_metrics") if isinstance(payload, dict) else None
        if not isinstance(metrics, dict):
            missing.append(int(index))
            continue
        rows.append({"chunk_index": int(index), **dict(metrics)})
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in rows]
    allocated = [
        float(row["cuda_peak_memory_allocated_mib"])
        for row in rows
        if row.get("cuda_peak_memory_allocated_mib") is not None
    ]
    reserved = [
        float(row["cuda_peak_memory_reserved_mib"])
        for row in rows
        if row.get("cuda_peak_memory_reserved_mib") is not None
    ]
    return {
        "status": "available" if rows else "unavailable",
        "measured_chunk_count": len(rows),
        "missing_chunk_indexes": missing,
        "total_child_elapsed_seconds": round(sum(elapsed), 3),
        "max_cuda_peak_memory_allocated_mib": round(max(allocated), 3) if allocated else None,
        "max_cuda_peak_memory_reserved_mib": round(max(reserved), 3) if reserved else None,
        "chunks": rows,
    }

def run_funasr_chunked(
    *,
    input_path: str,
    output_path: str,
    provider: str,
    model: str,
    vad_model: str = "fsmn-vad",
    punc_model: str = "",
    spk_model: str = "",
    speaker_merge_threshold: float | None = None,
    preset_speaker_count: int | None = None,
    speaker_global_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    language: str = "zh",
    hotword: str = "",
    batch_size_s: int = 60,
    use_itn: bool = True,
    merge_vad: bool = True,
    merge_length_s: int = 15,
    vad_max_single_segment_time_ms: int = 30000,
    device: str = "auto",
    chunk_seconds: int = 300,
    chunk_overlap_seconds: float = 0.0,
    chunk_boundary_mode: str = "fixed_duration",
    silence_noise_floor_db: float = -30.0,
    minimum_silence_seconds: float = 0.5,
    silence_max_offset_seconds: float = 10.0,
    chunk_indexes: list[int] | None = None,
    max_chunk_attempts: int = 3,
    rebuild_from_checkpoint: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run SenseVoice/FunASR as isolated, resumable local media chunks."""
    media = (
        Path(input_path).expanduser().absolute()
        if rebuild_from_checkpoint
        else Path(input_path).expanduser().resolve()
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_seconds = max(30, int(chunk_seconds or 300))
    chunk_overlap_seconds = float(chunk_overlap_seconds or 0.0)
    if not 0.0 < float(speaker_global_threshold) <= 1.0:
        raise ValueError("speaker_global_threshold must be in (0, 1]")
    checkpoint_path = output.with_name(f"{output.stem}-checkpoint.json")
    report_path = output.with_name(f"{output.stem}-chunk-report.json")
    chunks_dir = output.with_name(f"{output.stem}-chunks")
    progress = LocalMediaProgress(
        pipeline="local_funasr_chunked_asr",
        snapshot_path=output.with_name(f"{output.stem}-progress.json"),
        events_path=output.with_name(f"{output.stem}-progress.jsonl"),
        callback=progress_callback,
    )
    chunks_dir.mkdir(parents=True, exist_ok=True)
    progress.emit(stage="preflight", percent=0, message="Validating chunked local FunASR input", output_paths=[output], report_paths=[checkpoint_path, report_path])
    if not rebuild_from_checkpoint and not media.is_file():
        return _finalize(_failure(media, output, provider, model, "input_not_found"), report_path, progress)

    boundary_mode = str(chunk_boundary_mode or "fixed_duration").strip().lower()
    if (
        boundary_mode not in {"fixed_duration", "silence_snap"}
        or chunk_overlap_seconds < 0
        or chunk_overlap_seconds >= chunk_seconds / 2
    ):
        return _finalize(
            _failure(
                media,
                output,
                provider,
                model,
                "invalid_chunk_strategy",
                f"mode={boundary_mode}; overlap={chunk_overlap_seconds:g}",
            ),
            report_path,
            progress,
        )
    selected = sorted(set(chunk_indexes or []))
    chunk_manifest_path = output.with_name(f"{output.stem}-chunk-manifest.json")
    try:
        if rebuild_from_checkpoint:
            # Intent: rebuild canonical overlap output without re-running ASR.
            # Decision: reuse only the exact existing manifest and v2 checkpoint.
            # Reason: removable-media duration probes can fail after every chunk
            # already completed, while merge-only fixes need no media decode.
            # Evidence: the long-01 checkpoint retained all 13 successful chunks
            # when a later E: duration probe returned zero.
            # Effective scope: explicit --rebuild-from-checkpoint only; normal
            # execution still probes, chunks, and validates media as before.
            chunk_manifest = read_json(chunk_manifest_path)
            progress.emit(
                stage="checkpoint_rebuild",
                percent=5,
                message="Rebuilding canonical ASR from the exact saved checkpoint",
            )
        else:
            duration_seconds = _media_duration_seconds(media)
            progress.emit(
                stage="chunking",
                percent=5,
                message=(
                    f"Splitting media with {boundary_mode} boundaries near "
                    f"{chunk_seconds} seconds"
                ),
            )
            if boundary_mode == "silence_snap":
                chunk_manifest = prepare_silence_snapped_chunks(
                    media,
                    chunks_dir,
                    target_chunk_seconds=chunk_seconds,
                    media_duration_seconds=duration_seconds,
                    noise_floor_db=silence_noise_floor_db,
                    minimum_silence_seconds=minimum_silence_seconds,
                    max_offset_seconds=silence_max_offset_seconds,
                    overlap_seconds=chunk_overlap_seconds,
                )
            elif chunk_overlap_seconds > 0:
                chunk_manifest = prepare_fixed_overlap_chunks(
                    media,
                    chunks_dir,
                    target_chunk_seconds=chunk_seconds,
                    media_duration_seconds=duration_seconds,
                    overlap_seconds=chunk_overlap_seconds,
                )
            else:
                chunks = _audio_chunks(media, chunks_dir, chunk_seconds=chunk_seconds)
                chunk_manifest = build_fixed_chunk_manifest(
                    media,
                    chunks,
                    target_chunk_seconds=chunk_seconds,
                    media_duration_seconds=duration_seconds,
                )
            write_json(chunk_manifest_path, chunk_manifest)
    except Exception as exc:
        return _finalize(
            _failure(
                media,
                output,
                provider,
                model,
                "audio_chunking_failed",
                str(exc),
            ),
            report_path,
            progress,
        )

    execution_contract_revision = _execution_contract_revision(
        provider=provider,
        model=model,
        vad_model=vad_model,
        punc_model=punc_model,
        spk_model=spk_model,
        speaker_merge_threshold=speaker_merge_threshold,
        preset_speaker_count=preset_speaker_count,
        speaker_global_threshold=speaker_global_threshold,
        language=language,
        hotword=hotword,
        batch_size_s=batch_size_s,
        use_itn=use_itn,
        merge_vad=merge_vad,
        merge_length_s=merge_length_s,
        vad_max_single_segment_time_ms=vad_max_single_segment_time_ms,
        device=device,
    )
    checkpoint = _load_checkpoint(
        checkpoint_path,
        media=media,
        model=model,
        chunk_seconds=chunk_seconds,
        chunk_manifest_revision=str(chunk_manifest["revision"]),
        execution_contract_revision=execution_contract_revision,
        allow_legacy_fixed=boundary_mode == "fixed_duration",
        allow_unavailable_media=rebuild_from_checkpoint,
    )
    if rebuild_from_checkpoint and not checkpoint["resumed"]:
        return _finalize(
            _failure(
                media,
                output,
                provider,
                model,
                "checkpoint_not_reusable",
                "exact v2 checkpoint/manifest identity did not match",
            ),
            report_path,
            progress,
        )
    results = list(checkpoint["results"])
    failures = list(checkpoint["failed_chunks"])
    indexed = [
        (
            int(row["index"]),
            Path(str(row["artifact_path"])),
            dict(row),
        )
        for row in chunk_manifest["chunks"]
    ]
    available_indexes = {index for index, _path, _row in indexed}
    expected_indexes = sorted(available_indexes)
    if selected:
        missing = sorted(set(selected) - available_indexes)
        if missing:
            return _finalize(_failure(media, output, provider, model, "requested_chunk_missing", f"requested chunk indexes not found: {missing}"), report_path, progress)
        indexed = [row for row in indexed if row[0] in selected]
        # Intent: make an explicit repair command replace a previously
        # misclassified "successful" chunk instead of silently skipping it.
        # Decision: remove selected checkpoint rows/failures before execution.
        # Reason: historical empty chunks were recorded as successful.
        # Evidence: the 600-1200 second production gap needs exact reprocessing.
        # Effective scope: explicit --chunk-indexes repair runs only.
        selected_set = set(selected)
        results = [
            row for row in results if _chunk_number(row) not in selected_set
        ]
        failures = [
            row for row in failures if _chunk_number(row) not in selected_set
        ]
    successful = {_chunk_number(row) for row in results}
    todo = [
        (index, path, manifest_row)
        for index, path, manifest_row in indexed
        if index not in successful
    ]
    if rebuild_from_checkpoint:
        todo = []
    if checkpoint["resumed"]:
        progress.emit(stage="resume", percent=10, message=f"Resuming after {len(successful)} completed local ASR chunks")
    max_attempts = max(1, int(max_chunk_attempts or 1))
    for current, (index, chunk, chunk_manifest_row) in enumerate(todo, start=1):
        previous = next((row for row in failures if _chunk_number(row) == index), {})
        attempts = int(previous.get("attempt_count") or 0)
        if not selected and attempts >= max_attempts:
            continue
        failures = [row for row in failures if _chunk_number(row) != index]
        child_output = chunks_dir / f"chunk-{index:04d}.json"
        # Intent: prevent a failed child process from reusing stale JSON
        # left by a previous attempt.
        # Decision: remove only this deterministic child output before launch.
        # Reason: process failure plus an old file can otherwise look successful.
        # Evidence: explicit repair runs reuse the same chunk output path.
        # Effective scope: the selected local ASR chunk only.
        child_output.unlink(missing_ok=True)
        command = _child_command(
            chunk=chunk,
            output=child_output,
            provider=provider,
            model=model,
            vad_model=vad_model,
            punc_model=punc_model,
            spk_model=spk_model,
            speaker_merge_threshold=speaker_merge_threshold,
            preset_speaker_count=preset_speaker_count,
            language=language,
            hotword=hotword,
            batch_size_s=batch_size_s,
            use_itn=use_itn,
            merge_vad=merge_vad,
            merge_length_s=merge_length_s,
            vad_max_single_segment_time_ms=vad_max_single_segment_time_ms,
            device=device,
        )
        progress.emit(stage="transcription", percent=10 + (80 * (current - 1) / max(1, len(todo))), current_item=current, total_items=len(todo), message=f"Transcribing local ASR chunk {index}")
        completed = subprocess.run(command, cwd=str(chunks_dir), text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
        if completed.returncode == 0 and child_output.is_file():
            try:
                child = read_json(child_output)
                records = _result_records(child)
                if not records:
                    raise ValueError("empty_chunk_asr_result")
                if not _records_have_content(records):
                    raise ValueError("unverified_empty_chunk")
                offset = float(chunk_manifest_row["start_seconds"])
                for record_index, record in enumerate(records):
                    shifted = _offset_record(record, offset)
                    shifted["chunk_index"] = index
                    shifted["record_index"] = record_index
                    shifted["chunk_end_seconds"] = float(
                        chunk_manifest_row["end_seconds"]
                    )
                    shifted["chunk_core_start_seconds"] = float(
                        chunk_manifest_row.get("core_start_seconds", chunk_manifest_row["start_seconds"])
                    )
                    shifted["chunk_core_end_seconds"] = float(
                        chunk_manifest_row.get("core_end_seconds", chunk_manifest_row["end_seconds"])
                    )
                    shifted["chunk_duration_seconds"] = float(
                        chunk_manifest_row["duration_seconds"]
                    )
                    shifted["chunk_overlap_before_seconds"] = float(
                        chunk_manifest_row["overlap_before_seconds"]
                    )
                    shifted["chunk_overlap_after_seconds"] = float(
                        chunk_manifest_row["overlap_after_seconds"]
                    )
                    results.append(shifted)
            except Exception as exc:
                detail = str(exc)
                failure = _failure_row(
                    index,
                    chunk,
                    command,
                    attempts + 1,
                    detail,
                    start_seconds=float(chunk_manifest_row["start_seconds"]),
                    end_seconds=float(chunk_manifest_row["end_seconds"]),
                )
                if detail in {"empty_chunk_asr_result", "unverified_empty_chunk"}:
                    failure["reason"] = detail
                failure["retry_command"] = _parent_retry_command(
                    command,
                    media,
                    output,
                    chunk_seconds,
                    index,
                    chunk_boundary_mode=boundary_mode,
                    chunk_overlap_seconds=chunk_overlap_seconds,
                    silence_noise_floor_db=silence_noise_floor_db,
                    minimum_silence_seconds=minimum_silence_seconds,
                    silence_max_offset_seconds=silence_max_offset_seconds,
                )
                failures.append(failure)
        else:
            detail = _child_failure_detail(completed.stdout, completed.stderr)
            failure = _failure_row(
                    index,
                    chunk,
                    command,
                    attempts + 1,
                    detail,
                    start_seconds=float(chunk_manifest_row["start_seconds"]),
                    end_seconds=float(chunk_manifest_row["end_seconds"]),
                )
            failure["retry_command"] = _parent_retry_command(
                    command,
                    media,
                    output,
                    chunk_seconds,
                    index,
                    chunk_boundary_mode=boundary_mode,
                    chunk_overlap_seconds=chunk_overlap_seconds,
                    silence_noise_floor_db=silence_noise_floor_db,
                    minimum_silence_seconds=minimum_silence_seconds,
                    silence_max_offset_seconds=silence_max_offset_seconds,
                )
            failures.append(failure)
        results.sort(key=lambda row: (int(row.get("chunk_index") or 0), int(row.get("record_index") or 0)))
        _write_checkpoint(
            checkpoint_path,
            media=media,
            model=model,
            chunk_seconds=chunk_seconds,
            chunk_manifest_revision=str(chunk_manifest["revision"]),
            execution_contract_revision=execution_contract_revision,
            results=results,
            failed_chunks=failures,
        )
        progress.emit(stage="transcription", percent=10 + (80 * current / max(1, len(todo))), current_item=current, total_items=len(todo), message=f"Finished local ASR chunk {index}", details={"successful_chunk_count": len({_chunk_number(row) for row in results}), "failed_chunk_count": len(failures)})

    successful_indexes = sorted({_chunk_number(row) for row in results if _chunk_number(row) >= 0})
    successful_set = set(successful_indexes)
    failed_index_set = {
        _chunk_number(row) for row in failures if _chunk_number(row) >= 0
    }
    unresolved_indexes = sorted(set(expected_indexes) - successful_set)
    canonical_complete = (
        not failures and successful_set == set(expected_indexes)
    )
    targeted_repair_completed = (
        bool(selected)
        and set(selected).issubset(successful_set)
        and not (set(selected) & failed_index_set)
    )
    status = (
        "completed"
        if canonical_complete
        else (
            "partial_targeted_completed"
            if targeted_repair_completed
            else ("degraded" if results else "failed")
        )
    )
    aligned_results = results
    speaker_global_alignment: dict[str, Any] = {
        "schema": "video_knowledge_pipeline.speaker_global_alignment.v1",
        "status": "not_requested" if not spk_model else "unavailable",
        "candidate_only": True,
        "privacy": {
            "anonymous_ids_only": True,
            "person_identity_inferred": False,
            "embedding_vectors_in_public_artifact": False,
        },
    }
    if spk_model:
        # Intent: turn chunk-local CAM++ labels into recording-global anonymous IDs.
        # Decision: adapt FunASR HybridSpeakerTracker after successful chunks are
        # collected and before canonical overlap ownership is selected.
        # Reason: every isolated child starts its labels again at zero, which made
        # one real person appear as several speakers in long recordings.
        # Evidence: the pinned upstream mapping and VKP overlap LocalAgreement are
        # both already source-reviewed and tested.
        # Effective scope: candidate ``speaker_global_id`` plus a local-private
        # centroid sidecar. Text, local ``spk``, roles and identities do not change.
        aligned_results, public_alignment, private_alignment = align_chunk_speaker_records(
            results,
            similarity_threshold=speaker_global_threshold,
            overlap_agreement_threshold=DEFAULT_OVERLAP_AGREEMENT_THRESHOLD,
        )
        speaker_global_alignment = write_alignment_artifacts(
            output,
            public_alignment,
            private_alignment,
            write=True,
        )
    canonical_result, overlap_merge = _canonicalize_overlap_records(
        aligned_results,
        chunk_manifest,
    )
    runtime_metrics = _chunk_runtime_summary(chunks_dir, successful_indexes)
    quality_status = (
        "completed"
        if canonical_complete and overlap_merge["status"] in {"completed", "not_requested"}
        else "degraded"
    )
    payload = {
        "schema": SCHEMA,
        "provider": provider,
        "output_path": str(output),
        "model": model,
        "speaker_merge_threshold": speaker_merge_threshold,
        "preset_speaker_count": preset_speaker_count,
        "speaker_global_threshold": float(speaker_global_threshold),
        "input": str(media),
        "duration_seconds": float(chunk_manifest["source"]["duration_seconds"]),
        "device": device,
        "chunk_seconds": chunk_seconds,
        "chunk_overlap_seconds": chunk_overlap_seconds,
        "chunk_boundary_mode": boundary_mode,
        "chunk_manifest_path": str(chunk_manifest_path),
        "chunk_manifest_revision": str(chunk_manifest["revision"]),
        "execution_contract_revision": execution_contract_revision,
        "chunk_count": len(chunk_manifest["chunks"]),
        "expected_chunk_indexes": expected_indexes,
        "requested_chunk_indexes": selected,
        "canonical_complete": canonical_complete,
        "targeted_repair_completed": targeted_repair_completed,
        "unresolved_chunk_indexes": unresolved_indexes,
        "quality_status": quality_status,
        "overlap_merge": overlap_merge,
        "speaker_global_alignment": speaker_global_alignment,
        "runtime_metrics": runtime_metrics,
        "checkpoint_path": str(checkpoint_path),
        "chunk_directory": str(chunks_dir),
        "resumed_from_checkpoint": checkpoint["resumed"],
        "source_freshness": (
            "not_revalidated_checkpoint_only"
            if rebuild_from_checkpoint
            else "revalidated"
        ),
        "successful_chunk_indexes": successful_indexes,
        "successful_chunk_count": len(successful_indexes),
        "failed_chunk_count": len(failures),
        "failed_chunks": failures,
        "gaps": [{"chunk_index": row["chunk_index"], "start": row["start"], "end": row["end"], "reason": row["reason"]} for row in failures],
        "retry_commands": [row["retry_command"] for row in failures],
        "result": canonical_result,
        "chunk_results": _public_chunk_results(aligned_results),
        "status": status,
        "ok": status in {"completed", "partial_targeted_completed"},
        "usable": bool(results),
    }
    if not rebuild_from_checkpoint:
        _write_checkpoint(
            checkpoint_path,
            media=media,
            model=model,
            chunk_seconds=chunk_seconds,
            chunk_manifest_revision=str(chunk_manifest["revision"]),
            execution_contract_revision=execution_contract_revision,
            results=results,
            failed_chunks=failures,
            status=status,
        )
    return _finalize(payload, report_path, progress)


def _child_command(**kwargs: Any) -> list[str]:
    command = [sys.executable, "-m", "video_knowledge_pipeline.funasr_python_runner"]
    pairs = (("chunk", "--input"), ("output", "--output"), ("provider", "--provider"), ("model", "--model"), ("vad_model", "--vad-model"), ("language", "--language"), ("device", "--device"), ("batch_size_s", "--batch-size-s"), ("merge_length_s", "--merge-length-s"), ("vad_max_single_segment_time_ms", "--vad-max-single-segment-time-ms"))
    for key, option in pairs:
        command.extend([option, str(kwargs[key])])
    for key, option in (("punc_model", "--punc-model"), ("spk_model", "--spk-model"), ("hotword", "--hotword")):
        if str(kwargs[key] or "").strip():
            command.extend([option, str(kwargs[key]).strip()])
    # Intent: preserve the upstream FunASR ClusterBackend controls through VKP's
    # resumable chunk wrapper.
    # Decision: forward only explicitly supplied values to the existing child
    # runner instead of reimplementing clustering in VKP.
    # Reason: the A/B plan already emits these controls; dropping them here made
    # the real execution disagree with the reviewed plan.
    # Evidence: FunASR AutoModel accepts ``spk_kwargs.cb_kwargs.merge_thr`` and
    # ``generate(preset_spk_num=...)``; funasr_python_runner owns that adapter.
    # Effective scope: local FunASR chunks only; defaults and production routing
    # remain unchanged.
    for key, option in (
        ("speaker_merge_threshold", "--speaker-merge-threshold"),
        ("preset_speaker_count", "--preset-speaker-count"),
    ):
        if kwargs.get(key) is not None:
            command.extend([option, str(kwargs[key])])
    command.append("--use-itn" if kwargs["use_itn"] else "--no-use-itn")
    command.append("--merge-vad" if kwargs["merge_vad"] else "--no-merge-vad")
    return command


def _result_records(payload: Any) -> list[dict[str, Any]]:
    values = payload.get("result") if isinstance(payload, dict) else None
    return [dict(row) for row in values if isinstance(row, dict)] if isinstance(values, list) else []


def _child_failure_detail(stdout: str, stderr: str) -> str:
    """Prefer the child runner's structured model error over noisy progress output.

    Intent: retain the actionable model/runtime failure from a failed ASR chunk.
    Decision: reuse the child runner's existing JSON stdout contract before stderr.
    Reason: FunASR progress and warnings on stderr can otherwise hide the exception.
    Evidence: the first real CAM++ GPU trial returned JSON ``error`` on stdout while
    stderr contained only progress bars and a punctuation/timestamp warning.
    Effective scope: failure diagnostics only; retry, model and transcript behavior
    are unchanged.
    """

    stdout_text = str(stdout or "").strip()
    for line in reversed(stdout_text.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and str(payload.get("error") or "").strip():
            error = str(payload["error"]).strip()
            error_traceback = str(payload.get("error_traceback") or "").strip()
            return f"{error}\n{error_traceback}" if error_traceback else error
    return str(stderr or stdout or "chunk_asr_failed").strip()


def _records_have_content(records: list[dict[str, Any]]) -> bool:
    """Reject an empty response until existing local VAD/activity evidence proves silence.

    Intent: prevent empty five-minute responses from being counted as completed.
    Decision: preserve the chunk and use the existing exact retry path.
    Reason: a successful process exit proves execution, not speech coverage.
    Evidence: VKP already has chunk reports, activity audit and targeted retry.
    Effective scope: future FunASR chunk runs; prior artifacts stay intact.
    """

    for record in records:
        if str(record.get("text") or "").strip():
            return True
        for key in ("sentence_info", "words"):
            rows = record.get(key)
            if not isinstance(rows, list):
                continue
            if any(
                isinstance(row, dict)
                and str(row.get("text") or row.get("word") or "").strip()
                for row in rows
            ):
                return True
    return False


def _offset_record(record: dict[str, Any], offset_seconds: float) -> dict[str, Any]:
    offset_ms = round(offset_seconds * 1000.0, 6)
    value = json.loads(json.dumps(record, ensure_ascii=False))
    for row in value.get("sentence_info") or []:
        if isinstance(row, dict):
            _offset_time_fields(row, offset_ms)
    timestamps = value.get("timestamp")
    if isinstance(timestamps, list):
        value["timestamp"] = [[_number(pair[0]) + offset_ms, _number(pair[1]) + offset_ms] if isinstance(pair, list) and len(pair) >= 2 else pair for pair in timestamps]
    value["chunk_offset_seconds"] = offset_seconds
    return value


def _offset_time_fields(row: dict[str, Any], offset_ms: float) -> None:
    for key in ("start", "end", "start_time", "end_time", "begin", "finish"):
        if key in row:
            row[key] = _number(row[key]) + offset_ms
    timestamp = row.get("timestamp")
    if isinstance(timestamp, list):
        row["timestamp"] = [[_number(pair[0]) + offset_ms, _number(pair[1]) + offset_ms] if isinstance(pair, list) and len(pair) >= 2 else pair for pair in timestamp]


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _canonicalize_overlap_records(
    records: list[dict[str, Any]],
    chunk_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep one core owner while retaining overlap as review evidence.

    Intent: use overlap to protect boundary words without duplicating them.
    Decision: select timestamped sentences by core-window midpoint and emit
    SimulStreaming local-agreement evidence for each adjacent shared window.
    Reason: canonical text needs deterministic ownership; disagreements remain
    visible instead of being silently resolved by a second transcript owner.
    Evidence: CrispASR overlap-save plus WhisperStreaming LocalAgreement.
    Effective scope: normalized FunASR result only; raw chunk results are kept.
    """

    strategy = chunk_manifest.get("strategy")
    overlap = float(strategy.get("overlap_seconds") or 0.0) if isinstance(strategy, dict) else 0.0
    if overlap <= 0:
        return (
            [_strip_chunk_metadata(row) for row in records],
            {
                "status": "not_requested",
                "excluded_padding_sentence_count": 0,
                "untimed_record_count": 0,
                "boundary_review_required_count": 0,
                "boundaries": [],
            },
        )
    chunks = {
        int(row["index"]): dict(row)
        for row in chunk_manifest.get("chunks") or []
        if isinstance(row, dict) and "index" in row
    }
    boundaries = _overlap_boundary_evidence(records, chunks)
    boundary_by_right = {
        int(row["right_chunk_index"]): row
        for row in boundaries
        if "right_chunk_index" in row
    }
    canonical: list[dict[str, Any]] = []
    excluded = 0
    untimed = 0
    untimed_deduplicated = 0
    for record in records:
        value = json.loads(json.dumps(record, ensure_ascii=False))
        chunk_index = int(value.get("chunk_index") or 0)
        chunk = chunks.get(chunk_index, {})
        rows = value.get("sentence_info")
        if not isinstance(rows, list):
            untimed += 1
            text = str(value.get("text") or "")
            boundary = boundary_by_right.get(chunk_index, {})
            lcs = boundary.get("boundary_lcs")
            if isinstance(lcs, dict) and lcs.get("automatic_merge_allowed"):
                removed = int(lcs.get("right_prefix_character_count") or 0)
                value["text"] = text[removed:].lstrip()
                value["overlap_deduplication"] = {
                    "method": "crispasr_nemo_boundary_lcs",
                    "removed_prefix_characters": removed,
                    "matched_unit_count": int(lcs.get("matched_unit_count") or 0),
                    "confidence": float(lcs.get("confidence") or 0.0),
                    "raw_chunk_preserved": True,
                }
                untimed_deduplicated += 1
            if str(value.get("text") or "").strip():
                canonical.append(_strip_chunk_metadata(value))
            continue
        core_start = float(chunk.get("core_start_seconds", chunk.get("start_seconds", 0.0)))
        core_end = float(chunk.get("core_end_seconds", chunk.get("end_seconds", core_start)))
        is_last = int(value.get("chunk_index") or 0) == max(chunks, default=0)
        kept: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            cropped = _crop_sentence_to_core(
                raw,
                core_start_seconds=core_start,
                core_end_seconds=core_end,
                include_end=is_last,
            )
            if cropped is not None:
                if cropped:
                    kept.append(cropped)
                if str(cropped.get("text") or "") != str(raw.get("text") or ""):
                    excluded += 1
                continue
            start, end = _sentence_time_seconds(raw)
            midpoint = (start + end) / 2.0
            owns = core_start <= midpoint and (midpoint <= core_end if is_last else midpoint < core_end)
            if owns:
                kept.append(raw)
            else:
                excluded += 1
        value["sentence_info"] = kept
        value["text"] = "".join(str(row.get("text") or "") for row in kept).strip()
        if kept or value["text"]:
            canonical.append(_strip_chunk_metadata(value))
    review_count = sum(bool(row.get("requires_human_review")) for row in boundaries)
    return canonical, {
        "status": "review_required" if review_count else "completed",
        "excluded_padding_sentence_count": excluded,
        "untimed_record_count": untimed,
        "untimed_deduplicated_boundary_count": untimed_deduplicated,
        "boundary_review_required_count": review_count,
        "boundaries": boundaries,
    }


def _crop_sentence_to_core(
    row: dict[str, Any],
    *,
    core_start_seconds: float,
    core_end_seconds: float,
    include_end: bool,
) -> dict[str, Any] | None:
    """Crop a boundary-spanning sentence with FunASR character timestamps.

    Intent: prevent long sentence records from duplicating an overlap window.
    Decision: reuse FunASR's aligned character timestamps before the existing
    whole-sentence midpoint fallback.
    Reason: a 30-second sentence can cross a 5-second chunk boundary even when
    its midpoint belongs wholly to the right chunk.
    Evidence: the fixed long-01 sample exposed a 830.214-860.184 sentence whose
    first five seconds duplicated the preceding chunk's timestamped sentences.
    Effective scope: canonical overlap output only; raw chunk results and
    boundary-review evidence remain byte-for-byte available.
    """

    text = str(row.get("text") or "")
    timestamps = row.get("timestamp")
    if not text or not isinstance(timestamps, list) or len(timestamps) != len(text):
        return None
    kept_text: list[str] = []
    kept_timestamps: list[list[Any]] = []
    core_start_ms = core_start_seconds * 1000.0
    core_end_ms = core_end_seconds * 1000.0
    for character, pair in zip(text, timestamps, strict=True):
        if not isinstance(pair, list) or len(pair) < 2:
            return None
        start_ms = _number(pair[0])
        end_ms = max(start_ms, _number(pair[1]))
        midpoint_ms = (start_ms + end_ms) / 2.0
        owns = core_start_ms <= midpoint_ms and (
            midpoint_ms <= core_end_ms if include_end else midpoint_ms < core_end_ms
        )
        if owns:
            kept_text.append(character)
            kept_timestamps.append(pair)
    if not kept_text:
        return {}
    value = json.loads(json.dumps(row, ensure_ascii=False))
    value["text"] = "".join(kept_text)
    if "sentence" in value:
        value["sentence"] = value["text"]
    value["timestamp"] = kept_timestamps
    value["start"] = _number(kept_timestamps[0][0])
    value["end"] = _number(kept_timestamps[-1][1])
    return value


def _overlap_boundary_evidence(
    records: list[dict[str, Any]],
    chunks: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_chunk.setdefault(int(record.get("chunk_index") or 0), []).append(record)
    evidence: list[dict[str, Any]] = []
    for index in sorted(chunks):
        if index + 1 not in chunks:
            continue
        left = chunks[index]
        right = chunks[index + 1]
        start = max(float(left.get("start_seconds") or 0.0), float(right.get("start_seconds") or 0.0))
        end = min(float(left.get("end_seconds") or 0.0), float(right.get("end_seconds") or 0.0))
        if end <= start:
            continue
        left_records = by_chunk.get(index, [])
        right_records = by_chunk.get(index + 1, [])
        left_text = _sentences_in_window(left_records, start, end)
        right_text = _sentences_in_window(right_records, start, end)
        agreement = measure_local_agreement(left_text, right_text, language="zh")
        boundary_lcs = measure_boundary_lcs_dedup(
            "".join(str(row.get("text") or "") for row in left_records),
            "".join(str(row.get("text") or "") for row in right_records),
            language="zh",
        )
        if left_text and right_text:
            # Intent: avoid false degradation when ASR sentence cuts differ.
            # Decision: accept a high-confidence boundary LCS as corroboration.
            # Reason: timestamp midpoint ownership already prevents duplication.
            # Evidence: CrispASR/NeMo boundary LCS plus preserved raw chunks.
            # Effective scope: overlap review status; canonical text is unchanged.
            requires_review = (
                agreement["agreement_over_shorter"] < 0.2
                and not boundary_lcs["automatic_merge_allowed"]
            )
        else:
            requires_review = bool(boundary_lcs["requires_human_review"])
        evidence.append(
            {
                "left_chunk_index": index,
                "right_chunk_index": index + 1,
                "overlap_start": round(start, 6),
                "overlap_end": round(end, 6),
                "local_agreement": agreement,
                "boundary_lcs": boundary_lcs,
                "requires_human_review": requires_review,
            }
        )
    return evidence


def _sentences_in_window(
    records: list[dict[str, Any]],
    start: float,
    end: float,
) -> str:
    texts: list[str] = []
    for record in records:
        for row in record.get("sentence_info") or []:
            if not isinstance(row, dict):
                continue
            row_start, row_end = _sentence_time_seconds(row)
            midpoint = (row_start + row_end) / 2.0
            if start <= midpoint < end:
                texts.append(str(row.get("text") or "").strip())
    return "".join(texts)


def _sentence_time_seconds(row: dict[str, Any]) -> tuple[float, float]:
    start_ms = _number(row.get("start", row.get("start_time", row.get("begin", 0.0))))
    end_ms = _number(row.get("end", row.get("end_time", row.get("finish", start_ms))))
    return start_ms / 1000.0, max(start_ms, end_ms) / 1000.0

def _strip_chunk_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        not in {
            "chunk_index",
            "record_index",
            "_speaker_embedding_centers",
        }
    }


def _public_chunk_results(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip biometric vectors while retaining anonymous mapping evidence."""

    return [
        {
            key: value
            for key, value in record.items()
            if key != "_speaker_embedding_centers"
        }
        for record in records
    ]


def _parent_retry_command(
    child_command: list[str],
    media: Path,
    output: Path,
    chunk_seconds: int,
    chunk_index: int,
    *,
    chunk_boundary_mode: str,
    chunk_overlap_seconds: float,
    silence_noise_floor_db: float,
    minimum_silence_seconds: float,
    silence_max_offset_seconds: float,
) -> dict[str, Any]:
    command = list(child_command)
    module_index = command.index("video_knowledge_pipeline.funasr_python_runner")
    command[module_index] = "video_knowledge_pipeline.funasr_chunked_runner"
    command = _replace_option(command, "--input", str(media))
    command = _replace_option(command, "--output", str(output))
    command.extend(
        [
            "--chunk-seconds",
            str(chunk_seconds),
            "--chunk-boundary-mode",
            str(chunk_boundary_mode),
            "--chunk-overlap-seconds",
            str(chunk_overlap_seconds),
            "--silence-noise-floor-db",
            str(silence_noise_floor_db),
            "--minimum-silence-seconds",
            str(minimum_silence_seconds),
            "--silence-max-offset-seconds",
            str(silence_max_offset_seconds),
            "--chunk-indexes",
            str(chunk_index),
        ]
    )
    return {
        "chunk_index": chunk_index,
        "command": command,
        "powershell": subprocess.list2cmdline(command),
    }


def _replace_option(command: list[str], option: str, value: str) -> list[str]:
    updated = list(command)
    index = updated.index(option)
    updated[index + 1] = value
    return updated


def _failure_row(
    index: int,
    chunk: Path,
    command: list[str],
    attempts: int,
    detail: str,
    *,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    return {
        "chunk_index": index,
        "start": float(start_seconds),
        "end": float(end_seconds),
        "reason": "chunk_asr_failed",
        "detail": detail[-4000:],
        "artifact_path": str(chunk),
        "attempt_count": attempts,
        "retry_command": {
            "chunk_index": index,
            "command": command,
            "powershell": subprocess.list2cmdline(command),
        },
    }


def _execution_contract_revision(**values: Any) -> str:
    """Bind reusable checkpoints to the exact local ASR execution contract.

    Intent: keep resumability without silently mixing results from different
    runner behavior or decoding parameters.
    Decision: reuse VKP's canonical JSON hash and include the pinned upstream
    timestamp contract plus every child-inference option.
    Reason: a code/config change can alter sentence boundaries even when media,
    model id, and chunk manifest are unchanged.
    Evidence: FunASR 1.3.30's official CLI timestamp flags changed the same
    SenseVoice sample from one coarse cue to 11 timed sentences.
    Effective scope: local checkpoint reuse only; old raw files remain on disk,
    and no network, model download, or provider routing is introduced.
    """

    payload = dict(values)
    payload["hotword"] = canonical_json_sha256(str(values.get("hotword") or ""))
    return canonical_json_sha256(
        {
            "runner_contract": "funasr_python_runner.sentence_timestamps_and_speaker_centers.v3",
            **payload,
        }
    )

def _load_checkpoint(
    path: Path,
    *,
    media: Path,
    model: str,
    chunk_seconds: int,
    chunk_manifest_revision: str,
    execution_contract_revision: str,
    allow_legacy_fixed: bool,
    allow_unavailable_media: bool = False,
) -> dict[str, Any]:
    default = {"resumed": False, "results": [], "failed_chunks": []}
    if not path.is_file():
        return default
    try:
        value = read_json(path)
    except Exception:
        return default
    if not isinstance(value, dict) or str(value.get("schema") or "") != CHECKPOINT_SCHEMA:
        return default
    identity = value.get("input_identity") if isinstance(value, dict) else None
    if (
        not isinstance(identity, dict)
        or str(identity.get("path") or "") != str(media)
        or str(value.get("model") or "") != model
        or int(value.get("chunk_seconds") or 0) != chunk_seconds
    ):
        return default
    if not allow_unavailable_media:
        if int(identity.get("bytes") or -1) != media.stat().st_size:
            return default
    saved_revision = str(value.get("chunk_manifest_revision") or "")
    if saved_revision:
        if saved_revision != str(chunk_manifest_revision):
            return default
    elif not allow_legacy_fixed:
        return default
    if str(value.get("execution_contract_revision") or "") != str(
        execution_contract_revision
    ):
        return default
    return {"resumed": True, "results": [dict(row) for row in value.get("results") or [] if isinstance(row, dict)], "failed_chunks": [dict(row) for row in value.get("failed_chunks") or [] if isinstance(row, dict)]}

def _chunk_number(row: dict[str, Any]) -> int:
    try:
        value = row.get("chunk_index")
        return int(value) if value is not None else -1
    except (TypeError, ValueError):
        return -1



def _write_checkpoint(
    path: Path,
    *,
    media: Path,
    model: str,
    chunk_seconds: int,
    chunk_manifest_revision: str,
    execution_contract_revision: str,
    results: list[dict[str, Any]],
    failed_chunks: list[dict[str, Any]],
    status: str = "running",
) -> None:
    write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "status": status,
            "input_identity": {
                "path": str(media),
                "bytes": media.stat().st_size,
            },
            "model": model,
            "chunk_seconds": chunk_seconds,
            "chunk_manifest_revision": str(chunk_manifest_revision),
            "execution_contract_revision": str(execution_contract_revision),
            "successful_chunk_indexes": sorted(
                {
                    _chunk_number(row)
                    for row in results
                    if _chunk_number(row) >= 0
                }
            ),
            "results": results,
            "failed_chunks": failed_chunks,
        },
    )


def _finalize(payload: dict[str, Any], report_path: Path, progress: LocalMediaProgress) -> dict[str, Any]:
    output = Path(str(payload.get("output_path") or "")) if payload.get("output_path") else None
    if output is None:
        output = Path(str(payload.get("input") or report_path)).with_name("raw-asr-output.json")
    payload["output_path"] = str(output)
    payload["report_path"] = str(report_path)
    payload["progress"] = progress.artifacts()
    write_json(output, payload)
    write_json(report_path, {"schema": "video_knowledge_pipeline.local_asr_chunk_report.v1", "status": payload.get("status"), "quality_status": payload.get("quality_status", payload.get("status")), "canonical_complete": payload.get("canonical_complete", False), "targeted_repair_completed": payload.get("targeted_repair_completed", False), "unresolved_chunk_indexes": payload.get("unresolved_chunk_indexes", []), "successful_chunk_count": payload.get("successful_chunk_count", 0), "failed_chunk_count": payload.get("failed_chunk_count", 0), "failed_chunks": payload.get("failed_chunks", []), "gaps": payload.get("gaps", []), "retry_commands": payload.get("retry_commands", []), "raw_successful_content_preserved": bool(payload.get("result"))})
    terminal = str(payload.get("status") or "failed")
    if terminal == "partial_targeted_completed":
        progress_status = "completed"
        progress_message = "Local ASR targeted subset completed; canonical media remains incomplete"
    else:
        progress_status = terminal if terminal in {"completed", "degraded", "failed"} else "failed"
        progress_message = "Local ASR completed" if terminal == "completed" else "Local ASR completed with missing chunks" if terminal == "degraded" else "Local ASR failed"
    progress.emit(stage="finalize", percent=100, message=progress_message, status=progress_status, output_paths=[output], report_paths=[report_path])
    return payload


def _failure(media: Path, output: Path, provider: str, model: str, reason: str, detail: str = "") -> dict[str, Any]:
    return {"schema": SCHEMA, "provider": provider, "model": model, "input": str(media), "output_path": str(output), "status": "failed", "ok": False, "usable": False, "error_code": reason, "error": detail, "result": [], "failed_chunks": [], "gaps": [], "retry_commands": []}


def _media_duration_seconds(media: Path) -> float:
    from .funasr_python_runner import _media_duration_seconds as duration
    return duration(media)


def _chunk_indexes(value: str) -> list[int] | None:
    values = sorted({int(part.strip()) for part in str(value or "").split(",") if part.strip()})
    if any(index < 0 for index in values):
        raise ValueError("chunk indexes must be zero or greater")
    return values or None


if __name__ == "__main__":
    raise SystemExit(main())
