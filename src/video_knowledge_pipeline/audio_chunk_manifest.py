from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .media_tools import local_tool_subprocess_env, resolve_media_tool


SCHEMA = "video_knowledge_pipeline.audio_chunk_manifest.v1"
SUBTITLE_EDIT_COMMIT = "1517bb5c23e1c4072ea829edbc8d08e27cf79289"
SUBTITLE_EDIT_SOURCE = (
    "src/ui/Features/Video/SpeechToText/OpenAiCompatible/OpenAiSttChunker.cs"
)
CRISPASR_COMMIT = "9deefe8f47273722415e4b4be5d87361b96177c9"
CRISPASR_OVERLAP_SOURCE = "src/core/crispasr_lcs.h"
_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def fixed_chunk_boundaries(
    total_seconds: float,
    target_chunk_seconds: float,
) -> list[tuple[float, float]]:
    """Return deterministic contiguous fixed-duration source windows."""

    total = float(total_seconds)
    target = float(target_chunk_seconds)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("total_seconds must be positive and finite")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_chunk_seconds must be positive and finite")
    return [
        (index * target, min(total, (index + 1) * target))
        for index in range(math.ceil(total / target))
    ]


def extract_audio_chunk_window(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    index: int,
    start_seconds: float,
    end_seconds: float,
    ffmpeg_path: str | Path | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Extract exactly one audio window without touching sibling artifacts."""

    media = Path(media_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    chunk_index = int(index)
    start = float(start_seconds)
    end = float(end_seconds)
    if not media.is_file():
        raise FileNotFoundError(media)
    if chunk_index < 0:
        raise ValueError("index must be zero or greater")
    if not math.isfinite(start) or start < 0:
        raise ValueError("start_seconds must be nonnegative and finite")
    if not math.isfinite(end) or end <= start:
        raise ValueError("end_seconds must be finite and greater than start_seconds")
    ffmpeg = str(ffmpeg_path or resolve_media_tool("ffmpeg") or "")
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_available_for_audio_chunk_window")

    output.mkdir(parents=True, exist_ok=True)
    chunk_path = output / f"chunk-{chunk_index:04d}.wav"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(media),
        "-t",
        f"{end - start:.6f}",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(chunk_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_seconds)),
        check=False,
        env=local_tool_subprocess_env(),
    )
    if (
        completed.returncode != 0
        or not chunk_path.is_file()
        or chunk_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "audio_chunk_window_extraction_failed: "
            f"chunk_index={chunk_index}; returncode={completed.returncode}; "
            f"stderr={completed.stderr[-500:]}"
        )
    return {
        "index": chunk_index,
        "path": str(chunk_path),
        "bytes": chunk_path.stat().st_size,
        "start_seconds": round(start, 6),
        "end_seconds": round(end, 6),
        "duration_seconds": round(end - start, 6),
        "command": command,
    }


def parse_subtitle_edit_silence_intervals(output: str) -> list[tuple[float, float]]:
    """Adapt Subtitle Edit's tested FFmpeg silence parser.

    Intent: use mature boundary evidence instead of inventing a second parser.
    Decision: retain Subtitle Edit's pairing and trailing-start rejection.
    Reason: an unmatched trailing silence is not a reliable reusable cut point.
    Evidence: pinned ``OpenAiSttChunker.ParseSilenceIntervals`` and its tests.
    Effective scope: local audio chunk planning only; no ASR or transcript edit.
    """

    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in str(output or "").splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END.search(line)
        if end_match and pending_start is not None:
            end = float(end_match.group(1))
            if end > pending_start:
                intervals.append((pending_start, end))
            pending_start = None
    return intervals


def compute_silence_adjusted_boundaries(
    total_seconds: float,
    chunk_count: int,
    silence_intervals: Iterable[tuple[float, float]],
    *,
    max_offset_seconds: float = 10.0,
) -> list[tuple[float, float]]:
    """Adapt Subtitle Edit's nearest-unused-silence boundary algorithm."""

    total = float(total_seconds)
    count = int(chunk_count)
    max_offset = float(max_offset_seconds)
    if total <= 0:
        raise ValueError("total_seconds must be positive")
    if count <= 0:
        raise ValueError("chunk_count must be positive")
    if max_offset < 0:
        raise ValueError("max_offset_seconds must be nonnegative")
    if count == 1:
        return [(0.0, total)]

    midpoints = sorted(
        (float(start) + float(end)) / 2.0
        for start, end in silence_intervals
        if float(end) > float(start)
        and 0.0 < (float(start) + float(end)) / 2.0 < total
    )
    cuts: list[float] = []
    last_cut = 0.0
    next_silence_index = 0
    for index in range(count - 1):
        target = total * (index + 1) / count
        best_cut = target
        best_offset = math.inf
        best_index = -1
        while (
            next_silence_index < len(midpoints)
            and midpoints[next_silence_index] <= last_cut
        ):
            next_silence_index += 1
        for candidate_index in range(next_silence_index, len(midpoints)):
            midpoint = midpoints[candidate_index]
            if midpoint > target + max_offset:
                break
            offset = abs(midpoint - target)
            if offset <= max_offset and offset < best_offset:
                best_offset = offset
                best_cut = midpoint
                best_index = candidate_index
        if best_cut <= last_cut:
            remaining = count - index
            best_cut = last_cut + (total - last_cut) / remaining
            best_index = -1
        cuts.append(best_cut)
        last_cut = best_cut
        if best_index >= 0:
            next_silence_index = best_index + 1

    boundaries: list[tuple[float, float]] = []
    start = 0.0
    for cut in cuts:
        boundaries.append((start, cut))
        start = cut
    boundaries.append((start, total))
    return boundaries


def expand_chunk_boundaries(
    core_boundaries: Iterable[tuple[float, float]],
    *,
    total_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    """Expand core chunks with bounded context windows.

    Intent: preserve words that straddle long-media chunk boundaries.
    Decision: adapt CrispASR's overlap-save windows while retaining exactly
    one non-overlapping core owner for every source-media timestamp.
    Reason: overlap is decoding context, not permission to duplicate text.
    Evidence: pinned CrispASR ``crispasr_lcs.h`` and VKP local-agreement.
    Effective scope: local chunk extraction only; canonical merge stays gated.
    """

    total = float(total_seconds)
    overlap = float(overlap_seconds)
    if total <= 0:
        raise ValueError("total_seconds must be positive")
    if overlap < 0:
        raise ValueError("overlap_seconds must be nonnegative")
    cores = [(float(start), float(end)) for start, end in core_boundaries]
    if any(end <= start for start, end in cores):
        raise ValueError("core chunk duration must be positive")
    return [
        (max(0.0, start - overlap), min(total, end + overlap))
        for start, end in cores
    ]


def prepare_fixed_overlap_chunks(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    target_chunk_seconds: float,
    media_duration_seconds: float,
    overlap_seconds: float,
    ffmpeg_path: str | Path | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Extract fixed chunks with CrispASR-style context overlap."""

    media = Path(media_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    duration = float(media_duration_seconds)
    target = float(target_chunk_seconds)
    overlap = float(overlap_seconds)
    if not media.is_file():
        raise FileNotFoundError(media)
    if duration <= 0 or target <= 0:
        raise ValueError("media duration and target chunk seconds must be positive")
    if overlap <= 0 or overlap >= target / 2:
        raise ValueError("overlap_seconds must be positive and less than half the chunk size")
    ffmpeg = str(ffmpeg_path or resolve_media_tool("ffmpeg") or "")
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_available_for_overlapped_chunking")
    core_boundaries = fixed_chunk_boundaries(duration, target)
    extraction_boundaries = expand_chunk_boundaries(
        core_boundaries,
        total_seconds=duration,
        overlap_seconds=overlap,
    )
    chunks, commands = _extract_chunk_windows(
        media,
        output,
        extraction_boundaries,
        ffmpeg=ffmpeg,
        timeout_seconds=timeout_seconds,
        failure_prefix="overlapped_audio_chunking_failed",
    )
    manifest = _manifest(
        media,
        chunks,
        extraction_boundaries,
        core_boundaries=core_boundaries,
        target_chunk_seconds=target,
        mode="fixed_overlap",
        boundary_source="crispasr_overlap_save",
        silence_detection={"status": "not_requested", "interval_count": 0},
    )
    manifest["extraction_commands"] = commands
    return manifest

def build_fixed_chunk_manifest(
    media_path: str | Path,
    chunk_paths: Iterable[str | Path],
    *,
    target_chunk_seconds: float,
    media_duration_seconds: float,
) -> dict[str, Any]:
    paths = [Path(path).expanduser().resolve() for path in chunk_paths]
    target = float(target_chunk_seconds)
    if target <= 0:
        raise ValueError("target_chunk_seconds must be positive")
    duration = float(media_duration_seconds)
    if duration <= 0:
        duration = target * len(paths)
    boundaries = (
        fixed_chunk_boundaries(duration, target)[: len(paths)] if paths else []
    )
    return _manifest(
        media_path,
        paths,
        boundaries,
        target_chunk_seconds=target,
        mode="fixed_duration",
        boundary_source="fixed_duration",
        silence_detection={"status": "not_requested", "interval_count": 0},
    )


def prepare_silence_snapped_chunks(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    target_chunk_seconds: float,
    media_duration_seconds: float,
    noise_floor_db: float = -30.0,
    minimum_silence_seconds: float = 0.5,
    max_offset_seconds: float = 10.0,
    overlap_seconds: float = 0.0,
    ffmpeg_path: str | Path | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Create exact local chunks using Subtitle Edit-compatible boundaries.

    The adapter invokes VKP's resolved FFmpeg directly and returns a manifest;
    it does not run ASR, retry a model, or create another pipeline owner.
    """

    media = Path(media_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    duration = float(media_duration_seconds)
    target = float(target_chunk_seconds)
    if not media.is_file():
        raise FileNotFoundError(media)
    if duration <= 0:
        raise ValueError("media_duration_seconds must be positive")
    if target <= 0:
        raise ValueError("target_chunk_seconds must be positive")
    ffmpeg = str(ffmpeg_path or resolve_media_tool("ffmpeg") or "")
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_available_for_silence_snapped_chunking")
    output.mkdir(parents=True, exist_ok=True)

    detect_command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(media),
        "-af",
        (
            f"silencedetect=noise={float(noise_floor_db):g}dB:"
            f"d={float(minimum_silence_seconds):g}"
        ),
        "-f",
        "null",
        "-",
    ]
    detected = subprocess.run(
        detect_command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_seconds)),
        check=False,
        env=local_tool_subprocess_env(),
    )
    intervals = (
        parse_subtitle_edit_silence_intervals(detected.stderr)
        if detected.returncode == 0
        else []
    )
    count = max(1, math.ceil(duration / target))
    boundaries = compute_silence_adjusted_boundaries(
        duration,
        count,
        intervals,
        max_offset_seconds=max_offset_seconds,
    )
    core_boundaries = boundaries
    extraction_boundaries = expand_chunk_boundaries(
        core_boundaries,
        total_seconds=duration,
        overlap_seconds=float(overlap_seconds),
    )
    chunks, extraction_commands = _extract_chunk_windows(
        media,
        output,
        extraction_boundaries,
        ffmpeg=ffmpeg,
        timeout_seconds=timeout_seconds,
        failure_prefix="silence_snapped_audio_chunking_failed",
    )
    detection_status = "completed" if detected.returncode == 0 else "ffmpeg_failed"
    boundary_source = (
        "subtitle_edit_nearest_silence"
        if intervals
        else "subtitle_edit_even_time_fallback"
    )
    manifest = _manifest(
        media,
        chunks,
        extraction_boundaries,
        core_boundaries=core_boundaries,
        target_chunk_seconds=target,
        mode="silence_snap",
        boundary_source=boundary_source,
        silence_detection={
            "status": detection_status,
            "interval_count": len(intervals),
            "noise_floor_db": float(noise_floor_db),
            "minimum_silence_seconds": float(minimum_silence_seconds),
            "max_offset_seconds": float(max_offset_seconds),
            "command": detect_command,
            "returncode": detected.returncode,
        },
    )
    manifest["extraction_commands"] = extraction_commands

    return manifest


def _extract_chunk_windows(
    media: Path,
    output: Path,
    boundaries: list[tuple[float, float]],
    *,
    ffmpeg: str,
    timeout_seconds: int,
    failure_prefix: str,
) -> tuple[list[Path], list[list[str]]]:
    """Use VKP's single resolved FFmpeg outlet for exact audio windows."""

    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("chunk-*.wav"):
        stale.unlink(missing_ok=True)
    chunks: list[Path] = []
    commands: list[list[str]] = []
    for index, (start, end) in enumerate(boundaries):
        try:
            extracted = extract_audio_chunk_window(
                media,
                output,
                index=index,
                start_seconds=start,
                end_seconds=end,
                ffmpeg_path=ffmpeg,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"{failure_prefix}: {exc}") from exc
        chunks.append(Path(str(extracted["path"])))
        commands.append(list(extracted["command"]))
    return chunks, commands

def compute_audio_chunk_manifest_revision(manifest: Mapping[str, Any]) -> str:
    """Hash stable source, strategy, detection, boundary and chunk identities."""

    source = manifest.get("source") if isinstance(manifest.get("source"), Mapping) else {}
    strategy = (
        manifest.get("strategy")
        if isinstance(manifest.get("strategy"), Mapping)
        else {}
    )
    silence = (
        manifest.get("silence_detection")
        if isinstance(manifest.get("silence_detection"), Mapping)
        else {}
    )
    chunks = [
        row for row in manifest.get("chunks") or [] if isinstance(row, Mapping)
    ]
    revision_payload = {
        "schema": str(manifest.get("schema") or ""),
        "source": dict(source),
        "strategy": dict(strategy),
        "silence_detection": {
            key: value
            for key, value in silence.items()
            if key not in {"command", "returncode"}
        },
        "chunks": [
            {key: value for key, value in row.items() if key != "artifact_path"}
            for row in chunks
        ],
    }
    return canonical_json_sha256(revision_payload)


def _manifest(
    media_path: str | Path,
    chunk_paths: list[Path],
    boundaries: list[tuple[float, float]],
    *,
    core_boundaries: list[tuple[float, float]] | None = None,
    target_chunk_seconds: float,
    mode: str,
    boundary_source: str,
    silence_detection: dict[str, Any],
) -> dict[str, Any]:
    if len(chunk_paths) != len(boundaries):
        raise ValueError("chunk paths and boundaries must have the same length")
    cores = list(core_boundaries or boundaries)
    if len(cores) != len(boundaries):
        raise ValueError("core and extraction boundaries must have the same length")
    media = Path(media_path).expanduser().resolve()
    chunks = []
    for index, (path, boundary, core) in enumerate(zip(chunk_paths, boundaries, cores)):
        start, end = map(float, boundary)
        core_start, core_end = map(float, core)
        if end <= start:
            raise ValueError("chunk boundary duration must be positive")
        chunks.append(
            {
                "index": index,
                "artifact_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "duration_seconds": round(end - start, 6),
                "core_start_seconds": round(core_start, 6),
                "core_end_seconds": round(core_end, 6),
                "overlap_before_seconds": round(core_start - start, 6),
                "overlap_after_seconds": round(end - core_end, 6),
                "boundary_source": boundary_source,
            }
        )
    source_stat = media.stat()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "path": str(media),
            "bytes": source_stat.st_size,
            "sha256": sha256_file(media),
            "mtime_ns": source_stat.st_mtime_ns,
            "duration_seconds": round(boundaries[-1][1], 6) if boundaries else 0.0,
        },
        "strategy": {
            "mode": mode,
            "target_chunk_seconds": float(target_chunk_seconds),
            "overlap_seconds": round(
                max(
                    (
                        max(
                            float(row["overlap_before_seconds"]),
                            float(row["overlap_after_seconds"]),
                        )
                        for row in chunks
                    ),
                    default=0.0,
                ),
                6,
            ),
            "boundary_semantics": "requested_source_media_window",
            "upstream": {
                "project": "Subtitle Edit",
                "commit": SUBTITLE_EDIT_COMMIT,
                "source": SUBTITLE_EDIT_SOURCE,
            }
            if mode == "silence_snap"
            else None,
        },
        "silence_detection": silence_detection,
        "chunks": chunks,
    }
    manifest["revision"] = compute_audio_chunk_manifest_revision(manifest)
    return manifest
