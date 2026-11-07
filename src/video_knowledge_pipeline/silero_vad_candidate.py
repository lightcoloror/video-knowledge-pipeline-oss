from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Callable

from .file_hash import sha256_file
from .models import now_iso
from .storage import write_json


SCHEMA = "video_knowledge_pipeline.silero_vad_candidate.v1"
SAMPLE_RATE_HZ = 16000


def run_silero_vad_candidate(
    media_path: str | Path,
    *,
    output_path: str | Path | None = None,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_duration_ms: int = 0,
    max_speech_duration_seconds: float | None = None,
    min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 400,
    execute: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Run faster-whisper's bundled Silero VAD as candidate-only evidence."""

    media = Path(media_path).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"media not found: {media}")
    _validate_settings(
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_seconds=max_speech_duration_seconds,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else media.with_name(f"{media.stem}.silero-vad-candidate.json")
    )
    runtime = _runtime_status()
    source_media = {
        "path": str(media),
        "bytes": media.stat().st_size,
        "sha256": sha256_file(media),
    }
    settings = {
        "threshold": float(threshold),
        "neg_threshold": float(neg_threshold) if neg_threshold is not None else None,
        "min_speech_duration_ms": int(min_speech_duration_ms),
        "max_speech_duration_seconds": (
            float(max_speech_duration_seconds)
            if max_speech_duration_seconds is not None
            else None
        ),
        "min_silence_duration_ms": int(min_silence_duration_ms),
        "speech_pad_ms": int(speech_pad_ms),
        "sample_rate_hz": SAMPLE_RATE_HZ,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": True,
        "status": "planned",
        "execute": bool(execute),
        "write": bool(write),
        "source_media": source_media,
        "upstream": {
            "project": "SYSTRAN/faster-whisper",
            "distribution": "faster-whisper",
            "installed_version": runtime["installed_version"],
            "api": "faster_whisper.vad.get_speech_timestamps",
            "model": "bundled_silero_not_loaded",
        },
        "runtime": runtime,
        "vad_settings": settings,
        "evidence_profile": "candidate-independent",
        "candidate_only": True,
        "segments": [],
        "segment_count": 0,
        "network_call": False,
        "operator_boundary": {
            "uses_existing_faster_whisper_runtime": True,
            "model_download_allowed": False,
            "candidate_evidence_only": True,
            "authoritative_vad_modified": False,
            "chunk_manifest_modified": False,
            "canonical_transcript_modified": False,
            "automatic_remote_retry": False,
            "automatic_fallback": False,
        },
        "output_path": str(target),
        "updated_at": now_iso(),
    }
    if not execute:
        if write:
            write_json(target, result)
        return result
    if not runtime["available"]:
        result.update(
            {
                "ok": False,
                "status": "runtime_unavailable",
                "error": "faster-whisper is not installed in the active Python runtime",
                "updated_at": now_iso(),
            }
        )
        if write:
            write_json(target, result)
        return result

    try:
        decode_audio, get_speech_timestamps, vad_options_type, assets = (
            _load_faster_whisper_vad_api()
        )
        result["model_assets"] = assets
        result["upstream"]["model"] = (
            "silero_v6_bundled_onnx" if any(asset["name"] == "silero_vad_v6.onnx" for asset in assets)
            else "silero_v5_bundled_onnx"
        )
        audio = decode_audio(str(media), sampling_rate=SAMPLE_RATE_HZ)
        if len(audio) == 0:
            raise ValueError("decoded audio is empty; no-speech inference is unavailable")
        options = vad_options_type(
            threshold=float(threshold),
            neg_threshold=float(neg_threshold) if neg_threshold is not None else None,
            min_speech_duration_ms=int(min_speech_duration_ms),
            max_speech_duration_s=(
                float(max_speech_duration_seconds)
                if max_speech_duration_seconds is not None
                else math.inf
            ),
            min_silence_duration_ms=int(min_silence_duration_ms),
            speech_pad_ms=int(speech_pad_ms),
        )
        raw_segments = get_speech_timestamps(
            audio,
            vad_options=options,
            sampling_rate=SAMPLE_RATE_HZ,
        )
        segments = _normalise_segments(raw_segments)
        if any(row['end_sample'] > len(audio) for row in segments):
            raise ValueError('Silero interval exceeds decoded audio length')
        result.update(
            {
                "ok": True,
                "status": "completed",
                "segments": segments,
                "segment_count": len(segments),
                "model_assets": assets,
                "decoded_sample_count": len(audio),
                "decoded_duration_seconds": round(len(audio) / SAMPLE_RATE_HZ, 6),
                "updated_at": now_iso(),
            }
        )
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "status": "failed",
                "error": f"silero_vad_failed: {exc}",
                "updated_at": now_iso(),
            }
        )
    if write:
        write_json(target, result)
    return result


def _runtime_status() -> dict[str, Any]:
    available = importlib.util.find_spec("faster_whisper") is not None
    version = ""
    if available:
        try:
            version = importlib.metadata.version("faster-whisper")
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "available": available,
        "installed_version": version,
        "python_executable": sys.executable,
        "bundled_model_expected": "Silero ONNX assets selected by the installed faster-whisper VAD factory",
        "download_required": False,
    }


def _load_faster_whisper_vad_api() -> tuple[
    Callable[..., Any], Callable[..., list[dict[str, Any]]], type[Any], list[dict[str, Any]]
]:
    from faster_whisper.audio import decode_audio  # type: ignore
    from faster_whisper.utils import get_assets_path  # type: ignore
    from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model  # type: ignore
    from inspect import unwrap

    asset_root = Path(get_assets_path()).resolve()
    assets: list[dict[str, Any]] = []
    factory = unwrap(get_vad_model)
    names = tuple(value for value in factory.__code__.co_consts
                  if isinstance(value, str) and value.endswith(".onnx"))
    supported = {frozenset({"silero_vad_v6.onnx"}),
                 frozenset({"silero_encoder_v5.onnx", "silero_decoder_v5.onnx"})}
    if frozenset(names) not in supported:
        raise ValueError(f"unsupported faster-whisper bundled VAD layout: {names}")
    for name in names:
        path = asset_root / name
        if not path.is_file():
            raise FileNotFoundError(f"bundled Silero VAD asset not found: {path}")
        assets.append(
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return decode_audio, get_speech_timestamps, VadOptions, assets


def _normalise_segments(rows: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        raise ValueError("Silero speech timestamps must be a list")
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("Silero speech timestamp must be an object")
        start_sample = _sample_number(row.get("start"))
        end_sample = _sample_number(row.get("end"))
        if end_sample <= start_sample:
            raise ValueError("Silero returned an invalid speech interval")
        result.append(
            {
                "segment_id": f"silero-vad-{position:04d}",
                "start": round(start_sample / SAMPLE_RATE_HZ, 6),
                "end": round(end_sample / SAMPLE_RATE_HZ, 6),
                "start_sample": start_sample,
                "end_sample": end_sample,
            }
        )
    return result


def _sample_number(value: Any) -> int:
    from numbers import Real
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value < 0 or int(value) != value):
        raise ValueError('Silero sample offsets must be nonnegative finite integers')
    return int(value)


def _validate_settings(
    *,
    threshold: float,
    neg_threshold: float | None,
    min_speech_duration_ms: int,
    max_speech_duration_seconds: float | None,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> None:
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if neg_threshold is not None and not 0.0 < float(neg_threshold) < 1.0:
        raise ValueError("neg_threshold must be between 0 and 1")
    if int(min_speech_duration_ms) < 0:
        raise ValueError("min_speech_duration_ms must not be negative")
    if max_speech_duration_seconds is not None and float(max_speech_duration_seconds) <= 0:
        raise ValueError("max_speech_duration_seconds must be positive")
    if int(min_silence_duration_ms) < 0:
        raise ValueError("min_silence_duration_ms must not be negative")
    if int(speech_pad_ms) < 0:
        raise ValueError("speech_pad_ms must not be negative")
