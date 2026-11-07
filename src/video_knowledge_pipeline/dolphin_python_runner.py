from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .storage import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dolphin-python-runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("LECTURE_ASR_DOLPHIN_MODEL", "small"))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--region", default="CN")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--hotword-file", default="")
    parser.add_argument("--word-timestamp", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.environ.get("LECTURE_ASR_DEVICE", "auto"))
    args = parser.parse_args(argv)

    result = run_dolphin(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        language=args.language,
        region=args.region,
        hotwords=_hotwords(args.hotwords, args.hotword_file),
        word_timestamp=args.word_timestamp,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def run_dolphin(
    *,
    input_path: str,
    output_path: str,
    model: str = "small",
    language: str = "zh",
    region: str = "CN",
    hotwords: list[str] | None = None,
    word_timestamp: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    media = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not media.exists():
        result = {"ok": False, "status": "input_not_found", "error": f"input not found: {media}", "output_path": str(output)}
        _write_output(output, result)
        return result
    try:
        import dolphin  # type: ignore
        from dolphin import transcribe  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency.
        result = {"ok": False, "status": "dolphin_import_failed", "error": f"dolphin import failed: {exc}", "output_path": str(output)}
        _write_output(output, result)
        return result

    selected_device = _select_device(device)
    try:
        asr_model = dolphin.load_model(model, device=selected_device)
        kwargs: dict[str, Any] = {}
        if language:
            kwargs["lang_sym"] = language
        if region:
            kwargs["region_sym"] = region
        if hotwords:
            kwargs["hotwords"] = hotwords
            kwargs["use_deep_biasing"] = True
            kwargs["use_two_stage_filter"] = True
        if word_timestamp:
            kwargs["word_timestamp"] = True
        try:
            generated = transcribe(asr_model, str(media), **kwargs)
        except TypeError:
            generated = transcribe(asr_model, str(media))
    except Exception as exc:  # pragma: no cover - optional model runtime.
        result = {"ok": False, "status": "dolphin_generate_failed", "error": f"dolphin generate failed: {exc}", "output_path": str(output)}
        _write_output(output, result)
        return result

    payload = {
        "schema": "video_knowledge_dolphin_raw_output.v1",
        "provider": "dolphin",
        "model": model,
        "language": language,
        "region": region,
        "device": selected_device,
        "word_timestamp": bool(word_timestamp),
        "hotword_count": len(hotwords or []),
        "input": str(media),
        "duration_seconds": _media_duration_seconds(media),
        "result": _jsonable(generated),
    }
    _write_output(output, payload)
    return {"ok": True, "status": "ok", "output_path": str(output), "records": _record_count(payload["result"])}


def _write_output(output: Path, payload: dict[str, Any]) -> None:
    write_json(output, payload)


def _hotwords(inline: str, file_path: str) -> list[str]:
    rows: list[str] = []
    if inline:
        rows.extend(part.strip() for part in inline.split(",") if part.strip())
    if file_path:
        path = Path(file_path).expanduser()
        if path.exists():
            rows.extend(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    return rows


def _select_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch  # type: ignore

        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


def _media_duration_seconds(media: Path) -> float:
    ffprobe = os.environ.get("FFPROBE") or "ffprobe"
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(media)],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )
        if completed.returncode == 0:
            return max(float(completed.stdout.strip() or 0.0), 0.0)
    except Exception:
        pass
    return 0.0


def _record_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("segments", "result", "words"):
            if isinstance(value.get(key), list):
                return len(value[key])
        return 1
    if value in (None, ""):
        return 0
    return 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
