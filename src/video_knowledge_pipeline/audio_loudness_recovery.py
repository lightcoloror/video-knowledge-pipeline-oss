from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping

from .audio_silence_probe import probe_audio_silence
from .file_hash import sha256_file
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import now_iso
from .storage import replace_file_with_retry, write_json
from .video import probe_video


SCHEMA = "video_knowledge_pipeline.audio_loudness_recovery.v1"
UPSTREAM_COMMIT = "0a5dcf5f21f7f40ca77bc38ea6d1d3fd52e32c26"
TARGET_SAMPLE_RATE_HZ = 16000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"
_LOUDNESS_KEYS = (
    "input_i",
    "input_tp",
    "input_lra",
    "input_thresh",
    "target_offset",
)


def build_loudnorm_analysis_command(
    media_path: str | Path,
    *,
    target_lufs: float = -23.0,
    true_peak_db: float = -1.0,
    loudness_range: float = 7.0,
    ffmpeg_path: str | Path | None = None,
) -> list[str]:
    """Build the first pass of NarratoAI's two-pass EBU R128 workflow."""

    media = Path(media_path).expanduser().resolve()
    ffmpeg = str(ffmpeg_path or resolve_media_tool("ffmpeg") or "ffmpeg")
    audio_filter = (
        "aformat=channel_layouts=mono,"
        f"loudnorm=I={float(target_lufs):g}:TP={float(true_peak_db):g}:"
        f"LRA={float(loudness_range):g}:print_format=json"
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(media),
        "-vn",
        "-af",
        audio_filter,
        "-f",
        "null",
        "-",
    ]


def build_loudnorm_render_command(
    media_path: str | Path,
    output_path: str | Path,
    measurements: Mapping[str, float],
    *,
    target_lufs: float = -23.0,
    true_peak_db: float = -1.0,
    loudness_range: float = 7.0,
    ffmpeg_path: str | Path | None = None,
) -> list[str]:
    """Build a measured second pass that writes a fixed local ASR sidecar."""

    media = Path(media_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    values = _validated_measurements(measurements)
    ffmpeg = str(ffmpeg_path or resolve_media_tool("ffmpeg") or "ffmpeg")
    audio_filter = (
        "aformat=channel_layouts=mono,"
        f"loudnorm=I={float(target_lufs):g}:TP={float(true_peak_db):g}:"
        f"LRA={float(loudness_range):g}:"
        f"measured_I={values['input_i']:g}:"
        f"measured_LRA={values['input_lra']:g}:"
        f"measured_TP={values['input_tp']:g}:"
        f"measured_thresh={values['input_thresh']:g}:"
        f"offset={values['target_offset']:g}:linear=true:print_format=summary"
    )
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-nostats",
        "-i",
        str(media),
        "-vn",
        "-map_metadata",
        "-1",
        "-af",
        audio_filter,
        "-ar",
        str(TARGET_SAMPLE_RATE_HZ),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        TARGET_CODEC,
        "-f",
        "wav",
        str(target),
    ]


def parse_loudnorm_measurements(output: str) -> dict[str, float]:
    """Extract the complete loudnorm JSON object from mixed FFmpeg output."""

    decoder = json.JSONDecoder()
    text = str(output or "")
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not all(key in value for key in _LOUDNESS_KEYS):
            continue
        return _validated_measurements(value)
    raise ValueError("FFmpeg loudnorm output did not contain complete finite measurements")


def prepare_low_level_audio_candidate(
    media_path: str | Path,
    *,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    target_lufs: float = -23.0,
    true_peak_db: float = -1.0,
    loudness_range: float = 7.0,
    low_level_lufs_threshold: float = -35.0,
    silence_noise_db: float = -60.0,
    minimum_activity_seconds: float = 2.0,
    minimum_activity_ratio: float = 0.01,
    execute: bool = False,
    render_candidate: bool = False,
    overwrite_candidate: bool = False,
    timeout_seconds: int = 3600,
    write: bool = True,
) -> dict[str, Any]:
    """Prepare a candidate-only loudness sidecar without authorizing ASR retry."""

    media = Path(media_path).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"media not found: {media}")
    _validate_settings(
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
        loudness_range=loudness_range,
        low_level_lufs_threshold=low_level_lufs_threshold,
        minimum_activity_seconds=minimum_activity_seconds,
        minimum_activity_ratio=minimum_activity_ratio,
        timeout_seconds=timeout_seconds,
    )
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else media.with_name(f"{media.stem}.loudness-recovery-candidate.wav")
    )
    report = (
        Path(report_path).expanduser().resolve()
        if report_path
        else media.with_name(f"{media.stem}.loudness-recovery-report.json")
    )
    if target.suffix.lower() != ".wav":
        raise ValueError("audio recovery candidate output must use the .wav suffix")
    if media in {target, report} or target == report:
        raise ValueError("source, candidate output, and report paths must be distinct")

    source_identity = _artifact_identity(media)
    ffmpeg = resolve_media_tool("ffmpeg")
    analysis_command = build_loudnorm_analysis_command(
        media,
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
        loudness_range=loudness_range,
        ffmpeg_path=ffmpeg or None,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": True,
        "status": "planned",
        "execute": bool(execute),
        "render_candidate": bool(render_candidate),
        "overwrite_candidate": bool(overwrite_candidate),
        "write": bool(write),
        "source_media": source_identity,
        "candidate_output": {
            **_optional_artifact_identity(target),
            "produced_this_run": False,
            "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "channels": TARGET_CHANNELS,
            "codec": TARGET_CODEC,
        },
        "report_path": str(report),
        "settings": {
            "target_lufs": float(target_lufs),
            "true_peak_db": float(true_peak_db),
            "loudness_range": float(loudness_range),
            "low_level_lufs_threshold": float(low_level_lufs_threshold),
            "silence_noise_db": float(silence_noise_db),
            "minimum_activity_seconds": float(minimum_activity_seconds),
            "minimum_activity_ratio": float(minimum_activity_ratio),
        },
        "analysis_command": analysis_command,
        "upstream": {
            "project": "linyqh/NarratoAI",
            "repository": "https://github.com/linyqh/NarratoAI",
            "commit": UPSTREAM_COMMIT,
            "module": "app/services/audio_normalizer.py",
            "adapted_algorithm": "two_pass_ffmpeg_ebu_r128_loudnorm",
            "rejected_runtime": ["moviepy", "pydub", "numpy"],
            "rejected_behavior": ["simple_gain_fallback", "mp3_reencode_fallback"],
        },
        "decision_record": {
            "intent": "recover intelligibility candidates from low-level local audio",
            "decision": "render only a measured 16 kHz mono PCM sidecar after a non-silence gate",
            "reason": "loudness is not evidence of speech and gain can amplify noise",
            "evidence": "NarratoAI two-pass loudnorm plus VKP FFmpeg silence probing",
            "effective_scope": "local candidate artifact only; source and canonical ASR remain unchanged",
        },
        "candidate_only": True,
        "speech_proven": False,
        "asr_retry_authorized": False,
        "network_call": False,
        "operator_boundary": {
            "source_overwrite_allowed": False,
            "model_download_allowed": False,
            "remote_upload_allowed": False,
            "automatic_asr_retry": False,
            "automatic_fallback": False,
            "speech_vad_or_human_confirmation_required": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        if write:
            write_json(report, result)
        return result
    if not ffmpeg:
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="ffmpeg_not_available",
            error="registered FFmpeg runtime was not found",
        )

    try:
        duration = float(probe_video(media).duration_seconds)
    except Exception as exc:  # noqa: BLE001 - report a local media probe failure.
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="media_probe_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    if duration <= 0:
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="media_duration_invalid",
            error="media duration must be positive",
        )
    result["duration_seconds"] = round(duration, 6)
    silence_probe = probe_audio_silence(
        media,
        start=0.0,
        end=duration,
        noise_db=silence_noise_db,
        minimum_silence_seconds=0.3,
        timeout_seconds=max(90, min(int(timeout_seconds), int(duration / 4) + 30)),
    )
    result["silence_probe"] = silence_probe
    if not silence_probe.get("ok"):
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="silence_probe_failed",
            error=str(silence_probe.get("status") or "silence probe failed"),
        )

    activity_seconds = round(
        sum(
            max(0.0, float(row.get("duration_seconds") or 0.0))
            for row in silence_probe.get("activity_intervals") or []
            if isinstance(row, dict)
        ),
        6,
    )
    activity_ratio = activity_seconds / duration
    result["activity_gate"] = {
        "activity_seconds": activity_seconds,
        "activity_ratio": round(activity_ratio, 6),
        "passed": (
            activity_seconds >= float(minimum_activity_seconds)
            and activity_ratio >= float(minimum_activity_ratio)
        ),
        "non_silent_audio_is_not_proven_speech": True,
    }
    if not result["activity_gate"]["passed"]:
        result["recommended_action"] = (
            "keep the original chunk as a verified near-silence candidate; "
            "do not amplify it or retry ASR without speech evidence"
        )
        return _finish(
            result,
            report,
            write=write,
            ok=True,
            status="near_silence_blocked",
        )

    try:
        completed = subprocess.run(
            analysis_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout_seconds)),
            check=False,
            env=local_tool_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="loudness_analysis_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    result["analysis_returncode"] = completed.returncode
    if completed.returncode != 0:
        result["analysis_stderr_tail"] = (completed.stderr or "")[-1000:]
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="loudness_analysis_failed",
            error="FFmpeg loudnorm first pass failed",
        )
    try:
        measurements = parse_loudnorm_measurements(
            (completed.stderr or "") + "\n" + (completed.stdout or "")
        )
    except ValueError as exc:
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="loudness_measurements_invalid",
            error=str(exc),
        )
    result["loudness_measurements"] = measurements
    if measurements["input_i"] >= float(low_level_lufs_threshold):
        result["recommended_action"] = "use the original audio; loudness recovery is not needed"
        return _finish(
            result,
            report,
            write=write,
            ok=True,
            status="normalization_not_needed",
        )
    if not render_candidate:
        result["recommended_action"] = (
            "rerun with explicit candidate rendering, then require speech VAD or human confirmation"
        )
        return _finish(
            result,
            report,
            write=write,
            ok=True,
            status="low_level_candidate_detected",
        )

    if target.is_file() and not overwrite_candidate:
        result["recommended_action"] = (
            "choose a new candidate path or pass --overwrite-candidate explicitly"
        )
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="candidate_output_exists",
            error="candidate output already exists and was not overwritten",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_name(
        f".{target.stem}.{os.getpid()}.tmp{target.suffix}"
    )
    temporary_target.unlink(missing_ok=True)
    render_command = build_loudnorm_render_command(
        media,
        temporary_target,
        measurements,
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
        loudness_range=loudness_range,
        ffmpeg_path=ffmpeg,
    )
    result["render_command"] = render_command
    try:
        rendered = subprocess.run(
            render_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout_seconds)),
            check=False,
            env=local_tool_subprocess_env(),
        )
        result["render_returncode"] = rendered.returncode
        if rendered.returncode != 0:
            result["render_stderr_tail"] = (rendered.stderr or "")[-1000:]
            return _finish(
                result,
                report,
                write=write,
                ok=False,
                status="candidate_render_failed",
                error="FFmpeg loudnorm second pass failed",
            )
        if not temporary_target.is_file() or temporary_target.stat().st_size <= 44:
            return _finish(
                result,
                report,
                write=write,
                ok=False,
                status="candidate_render_failed",
                error="FFmpeg did not produce a non-empty WAV candidate",
            )
        format_report = _wav_contract(temporary_target)
        result["candidate_format_validation"] = format_report
        if not format_report["valid"]:
            return _finish(
                result,
                report,
                write=write,
                ok=False,
                status="candidate_format_invalid",
                error="rendered candidate is not 16 kHz mono 16-bit PCM WAV",
            )
        source_after = _artifact_identity(media)
        result["source_media_after"] = source_after
        if source_after["sha256"] != source_identity["sha256"]:
            return _finish(
                result,
                report,
                write=write,
                ok=False,
                status="source_identity_changed",
                error="source media changed during candidate rendering",
            )
        replace_file_with_retry(temporary_target, target)
        result["candidate_output"] = {
            **_artifact_identity(target),
            "exists": True,
            "produced_this_run": True,
            "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "channels": TARGET_CHANNELS,
            "codec": TARGET_CODEC,
        }
        result["recommended_action"] = (
            "run an independent speech-specific VAD or human review on this candidate; "
            "only confirmed speech may authorize a targeted ASR retry"
        )
        return _finish(
            result,
            report,
            write=write,
            ok=True,
            status="candidate_requires_speech_vad",
        )
    except (OSError, subprocess.TimeoutExpired, wave.Error) as exc:
        return _finish(
            result,
            report,
            write=write,
            ok=False,
            status="candidate_render_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        temporary_target.unlink(missing_ok=True)


def _validated_measurements(value: Mapping[str, Any]) -> dict[str, float]:
    measurements: dict[str, float] = {}
    for key in _LOUDNESS_KEYS:
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid loudnorm measurement: {key}") from exc
        if not math.isfinite(number):
            raise ValueError(f"non-finite loudnorm measurement: {key}")
        measurements[key] = number
    return measurements


def _validate_settings(
    *,
    target_lufs: float,
    true_peak_db: float,
    loudness_range: float,
    low_level_lufs_threshold: float,
    minimum_activity_seconds: float,
    minimum_activity_ratio: float,
    timeout_seconds: int,
) -> None:
    for name, value in (
        ("target_lufs", target_lufs),
        ("true_peak_db", true_peak_db),
        ("loudness_range", loudness_range),
        ("low_level_lufs_threshold", low_level_lufs_threshold),
        ("minimum_activity_seconds", minimum_activity_seconds),
        ("minimum_activity_ratio", minimum_activity_ratio),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if target_lufs >= 0 or true_peak_db > 0 or low_level_lufs_threshold >= 0:
        raise ValueError("loudness targets and thresholds must not exceed 0 dB")
    if loudness_range <= 0:
        raise ValueError("loudness_range must be positive")
    if minimum_activity_seconds < 0:
        raise ValueError("minimum_activity_seconds must not be negative")
    if not 0 <= minimum_activity_ratio <= 1:
        raise ValueError("minimum_activity_ratio must be between 0 and 1")
    if int(timeout_seconds) <= 0:
        raise ValueError("timeout_seconds must be positive")


def _wav_contract(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        sample_rate = int(source.getframerate())
        channels = int(source.getnchannels())
        sample_width = int(source.getsampwidth())
        frame_count = int(source.getnframes())
    return {
        "valid": (
            sample_rate == TARGET_SAMPLE_RATE_HZ
            and channels == TARGET_CHANNELS
            and sample_width == 2
            and frame_count > 0
        ),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
    }


def _optional_artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "bytes": None, "sha256": ""}
    return {**_artifact_identity(path), "exists": True}


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _finish(
    result: dict[str, Any],
    report: Path,
    *,
    write: bool,
    ok: bool,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    result.update({"ok": bool(ok), "status": status, "updated_at": now_iso()})
    if error:
        result["error"] = error
    if write:
        write_json(report, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Stable local front door; neither command authorizes ASR execution."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m video_knowledge_pipeline.audio_loudness_recovery"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare"):
        child = sub.add_parser(command)
        child.add_argument("media_path")
        child.add_argument("--output-path", default="")
        child.add_argument("--report-path", default="")
        child.add_argument("--target-lufs", type=float, default=-23.0)
        child.add_argument("--low-level-lufs-threshold", type=float, default=-35.0)
        child.add_argument("--silence-noise-db", type=float, default=-60.0)
        child.add_argument("--minimum-activity-seconds", type=float, default=2.0)
        child.add_argument("--minimum-activity-ratio", type=float, default=0.01)
        child.add_argument("--timeout-seconds", type=int, default=3600)
        child.add_argument("--overwrite-candidate", action="store_true")
        child.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_low_level_audio_candidate(
        args.media_path,
        output_path=args.output_path or None,
        report_path=args.report_path or None,
        target_lufs=args.target_lufs,
        low_level_lufs_threshold=args.low_level_lufs_threshold,
        silence_noise_db=args.silence_noise_db,
        minimum_activity_seconds=args.minimum_activity_seconds,
        minimum_activity_ratio=args.minimum_activity_ratio,
        timeout_seconds=args.timeout_seconds,
        execute=args.command == "prepare",
        render_candidate=args.command == "prepare",
        overwrite_candidate=args.overwrite_candidate,
        write=not args.no_write,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
