from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import now_iso
from .storage import read_json, write_json
from .file_hash import sha256_file as _file_sha256


SCHEMA = "video_knowledge_pipeline.asr_vad_chunk_manifest.v1"
DEFAULT_BITRATE_KBPS = 64
DEFAULT_MAX_REQUEST_SECONDS = 180.0
DEFAULT_CONTEXT_PADDING_SECONDS = 1.5


def prepare_asr_vad_chunks(
    media_path: str | Path,
    vad_json: str | Path,
    output_dir: str | Path,
    *,
    bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    max_request_seconds: float = DEFAULT_MAX_REQUEST_SECONDS,
    context_padding_seconds: float = DEFAULT_CONTEXT_PADDING_SECONDS,
    execute: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Build VAD-aligned ASR request artifacts without calling an ASR provider."""

    media = Path(media_path).expanduser().resolve()
    vad_path = Path(vad_json).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"media not found: {media}")
    if not vad_path.is_file():
        raise FileNotFoundError(f"VAD JSON not found: {vad_path}")
    if bitrate_kbps < 16 or bitrate_kbps > 128:
        raise ValueError("bitrate_kbps must be between 16 and 128")
    if sample_rate_hz not in {8000, 12000, 16000, 24000, 32000, 44100, 48000}:
        raise ValueError("unsupported sample_rate_hz")
    if channels not in {1, 2}:
        raise ValueError("channels must be 1 or 2")
    if max_request_seconds < 10:
        raise ValueError("max_request_seconds must be at least 10")
    if context_padding_seconds < 0 or context_padding_seconds > 10:
        raise ValueError("context_padding_seconds must be between 0 and 10")

    vad_payload = read_json(vad_path)
    intervals = _vad_intervals(vad_payload)
    if not intervals:
        raise ValueError("VAD JSON does not contain positive speech intervals")
    groups = _group_intervals(intervals, max_request_seconds=max_request_seconds)
    ffmpeg = resolve_media_tool("ffmpeg")
    chunks: list[dict[str, Any]] = []
    for position, group in enumerate(groups, start=1):
        core_start = float(group[0]["start"])
        core_end = float(group[-1]["end"])
        artifact_start = max(0.0, core_start - context_padding_seconds)
        artifact_end = core_end + context_padding_seconds
        target = target_dir / f"asr-chunk-{position:04d}.mp3"
        command = [
            ffmpeg or "ffmpeg",
            "-y",
            "-ss",
            f"{artifact_start:.3f}",
            "-i",
            str(media),
            "-t",
            f"{max(0.0, artifact_end - artifact_start):.3f}",
            "-vn",
            "-map_metadata",
            "-1",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate_hz),
            "-c:a",
            "libmp3lame",
            "-b:a",
            f"{bitrate_kbps}k",
            str(target),
        ]
        chunks.append(
            {
                "chunk_id": f"asr-chunk-{position:04d}",
                "position": position,
                "status": "planned",
                "core_start": round(core_start, 6),
                "core_end": round(core_end, 6),
                "artifact_start": round(artifact_start, 6),
                "artifact_end": round(artifact_end, 6),
                "context_padding_seconds": float(context_padding_seconds),
                "source_vad_segment_ids": [str(row["segment_id"]) for row in group],
                "output_path": str(target),
                "command": command,
            }
        )

    manifest_path = target_dir / "asr-vad-chunk-manifest.json"
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "planned",
        "ok": True,
        "execute": bool(execute),
        "source_path": str(media),
        "source_bytes": media.stat().st_size,
        "source_sha256": _file_sha256(media),
        "vad_json": str(vad_path),
        "vad_sha256": _file_sha256(vad_path),
        "vad_schema": str(vad_payload.get("schema") or "")
        if isinstance(vad_payload, dict)
        else "",
        "vad_interval_count": len(intervals),
        "chunk_count": len(chunks),
        "completed_chunk_count": 0,
        "failed_chunk_count": 0,
        "bitrate_kbps": int(bitrate_kbps),
        "sample_rate_hz": int(sample_rate_hz),
        "channels": int(channels),
        "max_request_seconds": float(max_request_seconds),
        "context_padding_seconds": float(context_padding_seconds),
        "chunks": chunks,
        "manifest_path": str(manifest_path),
        "network_call": False,
        "remote_execution": {
            "performed": False,
            "existing_scheduler": "consented_model_batch",
            "recommended_max_parallel_global": 4,
            "recommended_max_parallel_per_destination": 2,
            "automatic_retry": False,
            "automatic_fallback": False,
        },
        "operator_boundary": {
            "local_ffmpeg_only": True,
            "candidate_artifacts_do_not_replace_source": True,
            "exact_chunk_hash_consent_required_before_remote_call": True,
            "merge_uses_existing_targeted_retry_pipeline": True,
        },
        "created_at": now_iso(),
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    if not execute:
        write_json(manifest_path, manifest)
        return manifest
    if not ffmpeg:
        manifest.update({"ok": False, "status": "ffmpeg_not_available"})
        write_json(manifest_path, manifest)
        return manifest

    completed_count = 0
    failed_count = 0
    for row in chunks:
        output = Path(row["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                row["command"],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=max(30, int(timeout_seconds)),
                check=False,
                env=local_tool_subprocess_env(),
            )
        except Exception as exc:  # noqa: BLE001 - recorded per chunk.
            row.update({"status": "failed", "error": str(exc)})
            failed_count += 1
            continue
        if completed.returncode != 0 or not output.is_file():
            row.update(
                {
                    "status": "failed",
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
            failed_count += 1
            continue
        row.update(
            {
                "status": "completed",
                "output_bytes": output.stat().st_size,
                "output_sha256": _file_sha256(output),
            }
        )
        completed_count += 1

    status = (
        "completed"
        if completed_count == len(chunks)
        else ("degraded" if completed_count else "failed")
    )
    manifest.update(
        {
            "ok": status == "completed",
            "status": status,
            "completed_chunk_count": completed_count,
            "failed_chunk_count": failed_count,
            "completed_at": now_iso(),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def read_vad_intervals(vad_json: str | Path) -> list[dict[str, Any]]:
    """Read the existing FunASR-compatible VAD interval contract."""

    path = Path(vad_json).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VAD JSON not found: {path}")
    intervals = _vad_intervals(read_json(path))
    if not intervals:
        raise ValueError("VAD JSON does not contain positive speech intervals")
    return intervals


def _vad_intervals(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("segments")
    if not isinstance(rows, list):
        rows = payload.get("intervals")
    if not isinstance(rows, list):
        return []
    intervals: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        start = _number(row.get("start"), _number(row.get("start_ms"), 0.0) / 1000)
        end = _number(row.get("end"), _number(row.get("end_ms"), start * 1000) / 1000)
        if end <= start:
            continue
        intervals.append(
            {
                "segment_id": str(row.get("segment_id") or row.get("id") or f"vad-{position:04d}"),
                "start": start,
                "end": end,
            }
        )
    return sorted(intervals, key=lambda item: (item["start"], item["end"]))


def _group_intervals(
    intervals: list[dict[str, Any]],
    *,
    max_request_seconds: float,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in intervals:
        if not groups:
            groups.append([row])
            continue
        current = groups[-1]
        if float(row["end"]) - float(current[0]["start"]) <= max_request_seconds:
            current.append(row)
        else:
            groups.append([row])
    return groups



def _number(value: Any, default: float) -> float:
    if isinstance(value, bool) or value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
