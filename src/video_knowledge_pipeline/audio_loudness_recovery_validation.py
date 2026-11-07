from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .audio_chunk_manifest import (
    SCHEMA as CHUNK_MANIFEST_SCHEMA,
    compute_audio_chunk_manifest_revision,
)
from .audio_loudness_recovery import SCHEMA as RECOVERY_SCHEMA
from .file_hash import sha256_file
from .interval_coverage import merge_intervals
from .models import now_iso
from .silero_vad_candidate import (
    SCHEMA as SILERO_VAD_SCHEMA,
    run_silero_vad_candidate,
)
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.audio_loudness_recovery_validation.v1"
RETRY_PLAN_SCHEMA = "video_knowledge_pipeline.audio_loudness_recovery_retry_plan.v1"
FASTER_WHISPER_COMMIT = "ed9a06cd89a93e47838f564998a6c09b655d7f43"


def validate_low_level_audio_candidate(
    recovery_report_path: str | Path,
    *,
    vad_report_path: str | Path | None = None,
    chunk_manifest_path: str | Path | None = None,
    chunk_index: int | None = None,
    output_path: str | Path | None = None,
    execute_vad: bool = False,
    minimum_speech_seconds: float = 0.5,
    minimum_speech_ratio: float = 0.01,
    write: bool = True,
) -> dict[str, Any]:
    """Validate a normalized sidecar before planning a targeted local ASR retry."""

    recovery_path = Path(recovery_report_path).expanduser().resolve()
    if not recovery_path.is_file():
        raise FileNotFoundError(f"recovery report not found: {recovery_path}")
    if execute_vad and not write:
        raise ValueError("execute_vad=True requires write=True for auditable VAD evidence")
    _validate_thresholds(
        minimum_speech_seconds=minimum_speech_seconds,
        minimum_speech_ratio=minimum_speech_ratio,
    )
    if (chunk_manifest_path is None) != (chunk_index is None):
        raise ValueError("chunk_manifest_path and chunk_index must be provided together")
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else recovery_path.with_name(
            f"{recovery_path.stem}.speech-validation.json"
        )
    )
    recovery = _read_object(recovery_path, "recovery report")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "status": "validating",
        "recovery_report": _artifact_identity(recovery_path),
        "vad_report": {},
        "chunk_lineage": {},
        "speech_evidence": {
            "passed": False,
            "segment_count": 0,
            "speech_seconds": 0.0,
            "speech_ratio": 0.0,
        },
        "targeted_retry_recommended": False,
        "targeted_retry_plan": {},
        "network_call": False,
        "decision_record": {
            "intent": "allow local ASR retry only after recovered audio contains independent speech evidence",
            "decision": "reuse faster-whisper Silero VAD and exact chunk-manifest lineage",
            "reason": "loudness normalization and FFmpeg non-silence do not prove speech",
            "evidence": (
                f"SYSTRAN/faster-whisper@{FASTER_WHISPER_COMMIT} "
                "VadOptions/get_speech_timestamps plus VKP content-addressed artifacts"
            ),
            "effective_scope": "candidate retry planning only; no ASR execution or transcript merge",
        },
        "operator_boundary": {
            "local_only": True,
            "model_download_allowed": False,
            "remote_upload_allowed": False,
            "automatic_asr_execution": False,
            "automatic_transcript_merge": False,
            "automatic_fallback": False,
            "canonical_transcript_modified": False,
            "chunk_manifest_modified": False,
        },
        "output_path": str(target),
        "updated_at": now_iso(),
    }
    recovery_errors, candidate, source = _validate_recovery_report(recovery)
    result["recovery_lineage"] = {
        "candidate": _optional_artifact_identity(candidate),
        "source": _optional_artifact_identity(source),
        "errors": recovery_errors,
    }
    if recovery_errors:
        return _finish(
            result,
            target,
            write=write,
            ok=False,
            status="invalid_recovery_lineage",
            errors=recovery_errors,
        )

    vad_path = (
        Path(vad_report_path).expanduser().resolve()
        if vad_report_path
        else candidate.with_name(f"{candidate.stem}.silero-vad-candidate.json")
    )
    if execute_vad:
        vad = run_silero_vad_candidate(
            candidate,
            output_path=vad_path,
            execute=True,
            write=True,
        )
    elif vad_path.is_file():
        vad = _read_object(vad_path, "Silero VAD report")
    else:
        result["vad_report"] = {
            "path": str(vad_path),
            "exists": False,
            "execute_vad": False,
        }
        result["next_actions"] = [
            "Run this validator with --execute-vad to reuse the installed faster-whisper Silero VAD.",
            "Do not start ASR from the recovered candidate until this report confirms speech evidence.",
        ]
        return _finish(
            result,
            target,
            write=write,
            ok=True,
            status="vad_required",
        )

    result["vad_report"] = {
        **(_optional_artifact_identity(vad_path) if vad_path.is_file() else {}),
        "execute_vad": bool(execute_vad),
        "status": str(vad.get("status") or ""),
    }
    vad_errors, intervals = _validate_vad_report(
        vad,
        candidate=candidate,
        candidate_sha256=str(
            (recovery.get("candidate_output") or {}).get("sha256") or ""
        ),
        duration_seconds=float(recovery.get("duration_seconds") or 0.0),
    )
    if vad_errors:
        return _finish(
            result,
            target,
            write=write,
            ok=False,
            status="invalid_vad_lineage",
            errors=vad_errors,
        )

    duration = float(recovery.get("duration_seconds") or 0.0)
    speech_seconds = round(sum(end - start for start, end in intervals), 6)
    speech_ratio = speech_seconds / duration if duration > 0 else 0.0
    speech_passed = (
        bool(intervals)
        and speech_seconds >= float(minimum_speech_seconds)
        and speech_ratio >= float(minimum_speech_ratio)
    )
    result["speech_evidence"] = {
        "passed": speech_passed,
        "source": "faster_whisper_bundled_silero_v5",
        "upstream_commit": FASTER_WHISPER_COMMIT,
        "segment_count": len(intervals),
        "speech_seconds": speech_seconds,
        "speech_ratio": round(speech_ratio, 6),
        "minimum_speech_seconds": float(minimum_speech_seconds),
        "minimum_speech_ratio": float(minimum_speech_ratio),
        "local_intervals": [
            {"start": start, "end": end, "duration_seconds": round(end - start, 6)}
            for start, end in intervals
        ],
        "candidate_only": True,
    }

    chunk_lineage: dict[str, Any] = {}
    chunk_errors: list[str] = []
    if chunk_manifest_path is not None and chunk_index is not None:
        chunk_lineage, chunk_errors = _resolve_chunk_lineage(
            chunk_manifest_path,
            chunk_index=int(chunk_index),
            recovery_source=source,
            recovery_source_sha256=str(
                (recovery.get("source_media") or {}).get("sha256") or ""
            ),
            recovery_duration_seconds=duration,
        )
        result["chunk_lineage"] = chunk_lineage
        if chunk_errors:
            return _finish(
                result,
                target,
                write=write,
                ok=False,
                status="invalid_chunk_lineage",
                errors=chunk_errors,
            )

    if not speech_passed:
        result["next_actions"] = [
            "Do not retry ASR automatically; the independent VAD did not meet the speech-evidence gate.",
            "If the chunk is materially important, perform a local human listen before classifying it as silence.",
        ]
        status = "no_speech_detected" if not intervals else "speech_evidence_below_threshold"
        return _finish(
            result,
            target,
            write=write,
            ok=True,
            status=status,
        )

    if not chunk_lineage:
        result["next_actions"] = [
            "Speech candidate confirmed. Supply the exact audio_chunk_manifest and chunk index to map it safely to the parent timeline.",
            "Do not merge any ASR text without parent timing and the existing secondary-evidence quality gate.",
        ]
        return _finish(
            result,
            target,
            write=write,
            ok=True,
            status="speech_candidate_confirmed",
        )

    chunk = chunk_lineage["chunk"]
    offset = float(chunk["start_seconds"])
    global_intervals = [
        {
            "start": round(offset + start, 6),
            "end": round(offset + end, 6),
            "duration_seconds": round(end - start, 6),
        }
        for start, end in intervals
    ]
    plan = {
        "schema": RETRY_PLAN_SCHEMA,
        "status": "planned",
        "input_audio": _artifact_identity(candidate),
        "input_role": "loudness_recovery_candidate",
        "parent_source": chunk_lineage["parent_source"],
        "chunk_manifest": chunk_lineage["manifest"],
        "chunk_index": int(chunk["index"]),
        "source_start_seconds": float(chunk["start_seconds"]),
        "source_end_seconds": float(chunk["end_seconds"]),
        "global_offset_seconds": offset,
        "speech_intervals_local": result["speech_evidence"]["local_intervals"],
        "speech_intervals_global": global_intervals,
        "requires_local_asr_execution": True,
        "requires_secondary_evidence_registration": True,
        "requires_quality_gate_before_merge": True,
        "automatic_execution": False,
        "canonical_transcript_modified": False,
    }
    result.update(
        {
            "targeted_retry_recommended": True,
            "targeted_retry_plan": plan,
            "next_actions": [
                "Run the configured local ASR explicitly on input_audio and preserve the listed global offset.",
                "Register the result through the existing secondary-ASR evidence path, then rerun transcript completeness and quality gates.",
            ],
        }
    )
    return _finish(
        result,
        target,
        write=write,
        ok=True,
        status="targeted_retry_planned",
    )


def _validate_recovery_report(
    recovery: Mapping[str, Any],
) -> tuple[list[str], Path, Path]:
    errors: list[str] = []
    if str(recovery.get("schema") or "") != RECOVERY_SCHEMA:
        errors.append("recovery_schema_mismatch")
    if not bool(recovery.get("ok")):
        errors.append("recovery_not_ok")
    if str(recovery.get("status") or "") != "candidate_requires_speech_vad":
        errors.append("recovery_status_not_candidate_requires_speech_vad")
    if not bool(recovery.get("candidate_only")):
        errors.append("recovery_candidate_only_missing")
    if bool(recovery.get("asr_retry_authorized")):
        errors.append("recovery_unexpected_asr_authorization")
    candidate_row = _mapping(recovery.get("candidate_output"))
    source_row = _mapping(recovery.get("source_media"))
    candidate = Path(str(candidate_row.get("path") or ".")).expanduser().resolve()
    source = Path(str(source_row.get("path") or ".")).expanduser().resolve()
    errors.extend(
        _artifact_errors(candidate, candidate_row, prefix="candidate")
    )
    errors.extend(_artifact_errors(source, source_row, prefix="source"))
    duration = _number(recovery.get("duration_seconds"))
    if duration <= 0:
        errors.append("recovery_duration_invalid")
    return _unique(errors), candidate, source


def _validate_vad_report(
    vad: Mapping[str, Any],
    *,
    candidate: Path,
    candidate_sha256: str,
    duration_seconds: float,
) -> tuple[list[str], list[tuple[float, float]]]:
    errors: list[str] = []
    if str(vad.get("schema") or "") != SILERO_VAD_SCHEMA:
        errors.append("vad_schema_mismatch")
    if not bool(vad.get("ok")) or str(vad.get("status") or "") != "completed":
        errors.append("vad_not_completed")
    if not bool(vad.get("candidate_only")):
        errors.append("vad_candidate_only_missing")
    source = _mapping(vad.get("source_media"))
    if not _same_path(Path(str(source.get("path") or ".")), candidate):
        errors.append("vad_candidate_path_mismatch")
    if str(source.get("sha256") or "") != candidate_sha256:
        errors.append("vad_candidate_sha256_mismatch")
    errors.extend(_artifact_errors(candidate, source, prefix="vad_source"))
    rows: list[tuple[float, float]] = []
    for position, row in enumerate(vad.get("segments") or []):
        if not isinstance(row, Mapping):
            errors.append(f"vad_segment_{position}_invalid")
            continue
        start = _number(row.get("start"))
        end = _number(row.get("end"))
        if start < 0 or end <= start or end > duration_seconds + 0.25:
            errors.append(f"vad_segment_{position}_out_of_bounds")
            continue
        rows.append((max(0.0, start), min(duration_seconds, end)))
    intervals = [
        (round(start, 6), round(end, 6))
        for start, end in merge_intervals(rows)
    ]
    if int(vad.get("segment_count") or 0) != len(vad.get("segments") or []):
        errors.append("vad_segment_count_mismatch")
    return _unique(errors), intervals


def _resolve_chunk_lineage(
    manifest_path: str | Path,
    *,
    chunk_index: int,
    recovery_source: Path,
    recovery_source_sha256: str,
    recovery_duration_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    path = Path(manifest_path).expanduser().resolve()
    errors: list[str] = []
    if not path.is_file():
        return {"manifest": {"path": str(path), "exists": False}}, [
            "chunk_manifest_not_found"
        ]
    manifest = _read_object(path, "chunk manifest")
    if str(manifest.get("schema") or "") != CHUNK_MANIFEST_SCHEMA:
        errors.append("chunk_manifest_schema_mismatch")
    recorded_revision = str(manifest.get("revision") or "")
    computed_revision = compute_audio_chunk_manifest_revision(manifest)
    if not recorded_revision:
        errors.append("chunk_manifest_revision_missing")
    elif recorded_revision != computed_revision:
        errors.append("chunk_manifest_revision_mismatch")

    chunks = [row for row in manifest.get("chunks") or [] if isinstance(row, Mapping)]
    chunk: dict[str, Any] | None = None
    for row in chunks:
        if "index" not in row:
            continue
        try:
            row_index = int(row["index"])
        except (TypeError, ValueError):
            continue
        if row_index == chunk_index:
            chunk = dict(row)
            break
    if chunk is None:
        errors.append("chunk_index_not_found")
        chunk = {}

    chunk_path_value = str(chunk.get("artifact_path") or "")
    chunk_path = (
        Path(chunk_path_value).expanduser().resolve()
        if chunk_path_value
        else recovery_source
    )
    if chunk:
        if not chunk_path_value:
            errors.append("manifest_chunk_path_missing")
        elif not _same_path(chunk_path, recovery_source):
            errors.append("recovery_source_not_manifest_chunk")
        manifest_chunk_sha256 = str(chunk.get("sha256") or "")
        if not manifest_chunk_sha256:
            errors.append("manifest_chunk_sha256_missing")
        elif manifest_chunk_sha256 != recovery_source_sha256:
            errors.append("manifest_chunk_recovery_sha256_mismatch")
        errors.extend(
            _artifact_errors(
                chunk_path,
                chunk,
                prefix="manifest_chunk",
                require_bytes=True,
            )
        )

    chunk_start = _number(chunk.get("start_seconds"))
    chunk_end = _number(chunk.get("end_seconds"))
    chunk_duration = _number(chunk.get("duration_seconds"))
    if chunk:
        if "start_seconds" not in chunk or chunk_start < 0:
            errors.append("chunk_start_invalid")
        if "end_seconds" not in chunk or chunk_end <= chunk_start:
            errors.append("chunk_end_invalid")
        if "duration_seconds" not in chunk or chunk_duration <= 0:
            errors.append("chunk_duration_invalid")
        elif abs((chunk_end - chunk_start) - chunk_duration) > 0.01:
            errors.append("chunk_boundary_duration_mismatch")
    if chunk and abs(chunk_duration - recovery_duration_seconds) > 0.25:
        errors.append("recovery_duration_chunk_mismatch")

    source = _mapping(manifest.get("source"))
    parent_path_value = str(source.get("path") or "")
    parent = (
        Path(parent_path_value).expanduser().resolve()
        if parent_path_value
        else path.parent
    )
    parent_sha256 = str(source.get("sha256") or "")
    if not parent_path_value:
        errors.append("parent_source_path_missing")
    else:
        errors.extend(
            _artifact_errors(
                parent,
                source,
                prefix="parent_source",
                require_bytes=True,
            )
        )

    lineage = {
        "manifest": {
            **_artifact_identity(path),
            "revision": recorded_revision,
            "computed_revision": computed_revision,
            "schema": str(manifest.get("schema") or ""),
        },
        "parent_source": {
            "path": str(parent),
            "bytes": parent.stat().st_size if parent.is_file() else None,
            "sha256": parent_sha256,
        },
        "chunk": {
            "index": int(chunk["index"]) if chunk and "index" in chunk else chunk_index,
            "artifact_path": str(chunk_path),
            "bytes": chunk.get("bytes"),
            "sha256": str(chunk.get("sha256") or ""),
            "artifact_sha256": recovery_source_sha256,
            "start_seconds": chunk_start,
            "end_seconds": chunk_end,
            "duration_seconds": chunk_duration,
        },
        "errors": _unique(errors),
    }
    return lineage, _unique(errors)


def _artifact_errors(
    path: Path,
    expected: Mapping[str, Any],
    *,
    prefix: str,
    require_bytes: bool = True,
) -> list[str]:
    if not path.is_file():
        return [f"{prefix}_not_found"]
    errors: list[str] = []
    expected_sha = str(expected.get("sha256") or "")
    if not expected_sha:
        errors.append(f"{prefix}_sha256_missing")
    elif sha256_file(path) != expected_sha:
        errors.append(f"{prefix}_sha256_changed")
    if require_bytes:
        expected_bytes = expected.get("bytes")
        if expected_bytes is None:
            errors.append(f"{prefix}_bytes_missing")
        elif int(expected_bytes) != path.stat().st_size:
            errors.append(f"{prefix}_bytes_changed")
    return errors


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _optional_artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "bytes": None, "sha256": ""}
    return {**_artifact_identity(path), "exists": True}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.expanduser().resolve())) == os.path.normcase(
        str(right.expanduser().resolve())
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _validate_thresholds(
    *,
    minimum_speech_seconds: float,
    minimum_speech_ratio: float,
) -> None:
    if not math.isfinite(float(minimum_speech_seconds)) or minimum_speech_seconds < 0:
        raise ValueError("minimum_speech_seconds must be finite and non-negative")
    if not math.isfinite(float(minimum_speech_ratio)) or not 0 <= minimum_speech_ratio <= 1:
        raise ValueError("minimum_speech_ratio must be between 0 and 1")


def _finish(
    result: dict[str, Any],
    target: Path,
    *,
    write: bool,
    ok: bool,
    status: str,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    result.update({"ok": bool(ok), "status": status, "updated_at": now_iso()})
    if errors:
        result["errors"] = _unique(errors)
    if write:
        write_json(target, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Stable local validation front door; it never invokes ASR."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m video_knowledge_pipeline.audio_loudness_recovery_validation"
    )
    parser.add_argument("recovery_report")
    parser.add_argument("--vad-report", default="")
    parser.add_argument("--chunk-manifest", default="")
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--execute-vad", action="store_true")
    parser.add_argument("--minimum-speech-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-speech-ratio", type=float, default=0.01)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = validate_low_level_audio_candidate(
        args.recovery_report,
        vad_report_path=args.vad_report or None,
        chunk_manifest_path=args.chunk_manifest or None,
        chunk_index=args.chunk_index,
        output_path=args.output_path or None,
        execute_vad=args.execute_vad,
        minimum_speech_seconds=args.minimum_speech_seconds,
        minimum_speech_ratio=args.minimum_speech_ratio,
        write=not args.no_write,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
