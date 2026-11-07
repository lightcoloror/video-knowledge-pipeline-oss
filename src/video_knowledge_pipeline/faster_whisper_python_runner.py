from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .storage import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run faster-whisper and write raw ASR JSON")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default=os.environ.get("LECTURE_ASR_DEVICE", "auto"))
    parser.add_argument("--compute-type", default=os.environ.get("LECTURE_ASR_COMPUTE_TYPE", "auto"))
    parser.add_argument("--vad-filter", action="store_true")
    args = parser.parse_args(argv)

    result = run_faster_whisper(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=args.vad_filter,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def run_faster_whisper(
    *,
    input_path: str,
    output_path: str,
    model: str,
    language: str,
    device: str = "auto",
    compute_type: str = "auto",
    vad_filter: bool = True,
) -> dict[str, Any]:
    media = Path(input_path).expanduser()
    output = Path(output_path).expanduser()
    if not media.exists():
        result = {"ok": False, "error": f"input_not_found: {media}", "segments": []}
        _write(output, result)
        return result
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        result = {"ok": False, "error": f"import_failed: {exc}", "segments": []}
        _write(output, result)
        return result

    try:
        selected_device = _select_device(device)
        selected_compute_type = _select_compute_type(compute_type, selected_device)
        whisper_model = WhisperModel(model, device=selected_device, compute_type=selected_compute_type)
        segments_iter, info = whisper_model.transcribe(
            str(media),
            language=language or None,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
        )
        segments = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text).strip(),
            }
            for segment in segments_iter
        ]
        result = {
            "schema": "video_knowledge_faster_whisper_raw_output.v1",
            "ok": True,
            "provider": "faster-whisper",
            "model": model,
            "language": getattr(info, "language", language),
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "device": selected_device,
            "compute_type": selected_compute_type,
            "vad_filter": vad_filter,
            "condition_on_previous_text": False,
            "input": str(media),
            "segments": segments,
        }
    except Exception as exc:
        result = {"ok": False, "error": f"transcribe_failed: {exc}", "segments": []}
    _write(output, result)
    return result



def _select_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch  # type: ignore

        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


def _select_compute_type(compute_type: str, selected_device: str) -> str:
    requested = str(compute_type or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    return "float16" if selected_device == "cuda" else "int8"


def _write(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
