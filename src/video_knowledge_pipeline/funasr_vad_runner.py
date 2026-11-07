from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .storage import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="funasr-vad-runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="fsmn-vad")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-single-segment-time-ms", type=int, default=30000)
    parser.add_argument("--speech-noise-threshold", type=float, default=0.6)
    parser.add_argument("--max-end-silence-time-ms", type=int, default=800)
    parser.add_argument(
        "--evidence-profile",
        choices=["authoritative", "candidate-permissive"],
        default="authoritative",
    )
    args = parser.parse_args(argv)
    result = run_vad(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        device=args.device,
        max_single_segment_time_ms=args.max_single_segment_time_ms,
        speech_noise_threshold=args.speech_noise_threshold,
        max_end_silence_time_ms=args.max_end_silence_time_ms,
        evidence_profile=args.evidence_profile,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def run_vad(
    *,
    input_path: str,
    output_path: str,
    model: str = "fsmn-vad",
    device: str = "auto",
    max_single_segment_time_ms: int = 30000,
    speech_noise_threshold: float = 0.6,
    max_end_silence_time_ms: int = 800,
    evidence_profile: str = "authoritative",
) -> dict[str, Any]:
    media = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not media.exists():
        return {"ok": False, "error": f"input not found: {media}", "segments": []}
    if not 0.0 <= float(speech_noise_threshold) <= 1.0:
        return {
            "ok": False,
            "error": "speech_noise_threshold must be between 0 and 1",
            "segments": [],
        }
    if int(max_end_silence_time_ms) <= 0:
        return {
            "ok": False,
            "error": "max_end_silence_time_ms must be positive",
            "segments": [],
        }
    if int(max_single_segment_time_ms) <= 0:
        return {
            "ok": False,
            "error": "max_single_segment_time_ms must be positive",
            "segments": [],
        }
    if evidence_profile not in {"authoritative", "candidate-permissive"}:
        return {
            "ok": False,
            "error": f"unsupported evidence_profile: {evidence_profile}",
            "segments": [],
        }
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "error": f"funasr import failed: {exc}",
            "segments": [],
        }

    selected_device = _select_device(device)
    from .funasr_python_runner import _resolve_local_model

    resolved_model = _resolve_local_model(model)
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "model_revision": "v2.0.4",
        "disable_update": True,
        "speech_noise_thres": float(speech_noise_threshold),
        "max_end_silence_time": int(max_end_silence_time_ms),
    }
    if selected_device in {"cuda", "cpu"}:
        kwargs["device"] = selected_device
    try:
        vad = AutoModel(**kwargs)
        generated = vad.generate(
            input=str(media),
            cache={},
            is_final=True,
            max_single_segment_time=int(max_single_segment_time_ms or 30000),
        )
        segments = _segments(generated)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"funasr vad failed: {exc}",
            "segments": [],
        }

    vad_settings = {
        "max_single_segment_time_ms": int(max_single_segment_time_ms),
        "speech_noise_threshold": float(speech_noise_threshold),
        "max_end_silence_time_ms": int(max_end_silence_time_ms),
    }
    payload = {
        "schema": "video_knowledge_pipeline.funasr_vad_segments.v1",
        "input": str(media),
        "model": model,
        "resolved_model": resolved_model,
        "device": selected_device,
        "model_revision": "v2.0.4",
        "evidence_profile": evidence_profile,
        "candidate_only": evidence_profile == "candidate-permissive",
        "vad_settings": vad_settings,
        "segments": segments,
    }
    write_json(output, payload)
    return {
        "ok": True,
        "output_path": str(output),
        "segment_count": len(segments),
        "evidence_profile": evidence_profile,
        "candidate_only": evidence_profile == "candidate-permissive",
        "vad_settings": vad_settings,
        "segments": segments,
    }


def _segments(value: Any) -> list[dict[str, float]]:
    rows: list[Any] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                candidate = (
                    item.get("value") or item.get("segments") or item.get("timestamp")
                )
                if isinstance(candidate, list):
                    rows.extend(candidate)
    elif isinstance(value, dict):
        candidate = (
            value.get("value") or value.get("segments") or value.get("timestamp")
        )
        if isinstance(candidate, list):
            rows.extend(candidate)
    result: list[dict[str, float]] = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            start_ms, end_ms = float(row[0]), float(row[1])
        elif isinstance(row, dict):
            start_ms = float(row.get("start") or row.get("start_ms") or 0)
            end_ms = float(row.get("end") or row.get("end_ms") or start_ms)
        else:
            continue
        if end_ms > start_ms:
            result.append(
                {
                    "start": round(start_ms / 1000.0, 3),
                    "end": round(end_ms / 1000.0, 3),
                }
            )
    return result


def _select_device(value: str) -> str:
    requested = str(value or "auto").strip().lower()
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
