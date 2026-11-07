from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .media_tools import resolve_media_tool
from .storage import write_json
from .file_hash import sha256_file
from .transcript import parse_transcript


SCHEMA = "video_knowledge_pipeline.qwen3_forced_aligner_output.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen3-forced-aligner-runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.environ.get("LECTURE_ASR_DEVICE", "auto"))
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=os.environ.get("VKP_QWEN_ALIGNER_DTYPE", "auto"))
    parser.add_argument("--chunk-seconds", type=int, default=300)
    parser.add_argument("--work-dir", default=os.environ.get("VKP_QWEN_ALIGNER_WORK_DIR", ""))
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager"], default=os.environ.get("VKP_QWEN_ALIGNER_ATTN", "sdpa"))
    args = parser.parse_args(argv)
    result = run_qwen3_forced_alignment(
        input_path=args.input,
        transcript_path=args.transcript,
        output_path=args.output,
        model=args.model,
        language=args.language,
        device=args.device,
        dtype_name=args.dtype,
        chunk_seconds=args.chunk_seconds,
        work_dir=args.work_dir or None,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def run_qwen3_forced_alignment(
    *,
    input_path: str,
    transcript_path: str,
    output_path: str,
    model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    language: str = "Chinese",
    device: str = "auto",
    dtype_name: str = "auto",
    chunk_seconds: int = 300,
    work_dir: str | Path | None = None,
    attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    media = Path(input_path).expanduser().resolve()
    transcript = Path(transcript_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not media.exists():
        return _write_failure(output, "input_not_found", f"input not found: {media}", model=model)
    if not transcript.exists():
        return _write_failure(output, "transcript_not_found", f"transcript not found: {transcript}", model=model)
    cues = parse_transcript(transcript)
    if not cues:
        return _write_failure(output, "transcript_empty", "transcript has no cues", model=model)
    input_sha256 = sha256_file(media)
    transcript_sha256 = sha256_file(transcript)

    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except Exception as exc:
        return _write_failure(output, "qwen_forced_aligner_module_not_ready", str(exc), model=model)

    selected_device = _device(device, torch)
    dtype = _torch_dtype(dtype_name, selected_device, torch)
    try:
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": "cuda:0" if selected_device.startswith("cuda") else "cpu",
            "low_cpu_mem_usage": True,
        }
        if attn_implementation and attn_implementation != "auto":
            load_kwargs["attn_implementation"] = attn_implementation
        aligner = Qwen3ForcedAligner.from_pretrained(model, **load_kwargs)
        chunks = _cue_chunks(cues, max_seconds=max(30, int(chunk_seconds or 300)))
        output.parent.mkdir(parents=True, exist_ok=True)
        explicit_work = Path(work_dir).expanduser().resolve() if work_dir else None
        work_root = explicit_work or (output.parent / ".qwen3-aligner-work")
        work_root.mkdir(parents=True, exist_ok=True)
        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        chunk_rows: list[dict[str, Any]] = []
        # Windows TemporaryDirectory uses restrictive creation semantics that
        # can reject the decoder child. Use a unique owned directory without
        # changing ACLs; keep its derived clips as recovery evidence.
        persistent_work = explicit_work is not None or sys.platform == "win32"
        owned_work = explicit_work or (work_root / f"run-{uuid.uuid4().hex}")
        if persistent_work:
            owned_work.mkdir(parents=True, exist_ok=explicit_work is not None)
        work_context = nullcontext(str(owned_work)) if persistent_work else tempfile.TemporaryDirectory(prefix="run-", dir=str(work_root))
        with work_context as temp_dir:
            temp = Path(temp_dir)
            for chunk_index, chunk in enumerate(chunks):
                clip = temp / f"chunk-{chunk_index:04d}.wav"
                _extract_audio_window(media, clip, start=chunk["start"], end=chunk["end"])
                # Qwen3-ASR accepts ``(numpy.ndarray, sample_rate)`` directly. Passing a
                # Windows file path makes the upstream helper call ``librosa.load``;
                # that path can spend minutes in the current Windows/Numba runtime even
                # for a one-second WAV. The clip is already mono 16 kHz, so SoundFile
                # is lossless and avoids that unstable decoder path.
                aligned = aligner.align(audio=_read_audio_tuple(clip), text=chunk["text"], language=_language(language))
                items = list(aligned[0]) if aligned else []
                chunk_words = [
                    {
                        "text": str(getattr(item, "text", "") or ""),
                        "start": round(float(getattr(item, "start_time", 0.0)) + chunk["start"], 3),
                        "end": round(float(getattr(item, "end_time", 0.0)) + chunk["start"], 3),
                        "chunk_index": chunk_index,
                    }
                    for item in items
                ]
                words.extend(chunk_words)
                if chunk_words:
                    segment_start = float(chunk_words[0]["start"])
                    segment_end = float(chunk_words[-1]["end"])
                else:
                    segment_start = float(chunk["start"])
                    segment_end = float(chunk["end"])
                segments.append(
                    {
                        "index": len(segments) + 1,
                        "start": segment_start,
                        "end": segment_end,
                        "text": chunk["text"],
                        "words": chunk_words,
                        "source_cue_indexes": chunk["cue_indexes"],
                        "alignment": "word_level" if chunk_words else "unavailable",
                    }
                )
                chunk_rows.append(
                    {
                        "chunk_index": chunk_index,
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "source_cue_indexes": chunk["cue_indexes"],
                        "word_count": len(chunk_words),
                    }
                )
        if not persistent_work:
            shutil.rmtree(work_root, ignore_errors=True)
        if sha256_file(media) != input_sha256 or sha256_file(transcript) != transcript_sha256:
            raise RuntimeError("alignment_input_changed_during_execution")
        monotonic = _timestamps_monotonic(words)
        payload = {
            "schema": SCHEMA,
            "provider": "qwen3-forced-aligner",
            "model": model,
            "input_path": str(media),
            "input_sha256": input_sha256,
            "transcript_path": str(transcript),
            "transcript_sha256": transcript_sha256,
            "timing_resolution_seconds": float(aligner.timestamp_segment_time) / 1000,
            "device": selected_device,
            "dtype": str(dtype).replace("torch.", ""),
            "attn_implementation": attn_implementation,
            "language": _language(language),
            "chunk_seconds": max(30, int(chunk_seconds or 300)),
            "chunk_count": len(chunk_rows),
            "work_dir": str(temp_dir),
            "work_dir_preserved": persistent_work,
            "ok": bool(words) and monotonic,
            "status": "completed" if words and monotonic else "alignment_failed",
            "word_count": len(words),
            "timestamp_coverage": round(len([row for row in words if float(row["end"]) >= float(row["start"])]) / max(1, len(words)), 4),
            "timestamps_monotonic": monotonic,
            "chunks": chunk_rows,
            "words": words,
            "segments": segments,
            "operator_boundary": {
                "local_only": True,
                "transcript_text_preserved": True,
                "does_not_replace_asr": True,
                "no_cloud_call": True,
            },
        }
    except RuntimeError as exc:
        code = "qwen3_forced_aligner_cuda_oom" if "out of memory" in str(exc).lower() else "qwen3_forced_aligner_runtime_failed"
        payload = _failure(code, str(exc), model=model)
    except Exception as exc:
        payload = _failure("qwen3_forced_aligner_runtime_failed", str(exc), model=model)
    write_json(output, payload)
    return payload


def _cue_chunks(cues: list[Any], *, max_seconds: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[tuple[int, Any]] = []
    for index, cue in enumerate(cues):
        if current and float(cue.end) - float(current[0][1].start) > max_seconds:
            chunks.append(_cue_chunk(current))
            current = []
        current.append((index, cue))
    if current:
        chunks.append(_cue_chunk(current))
    return chunks


def _cue_chunk(rows: list[tuple[int, Any]]) -> dict[str, Any]:
    return {
        "start": max(0.0, float(rows[0][1].start)),
        "end": max(float(rows[-1][1].end), float(rows[0][1].start) + 0.1),
        "text": "".join(str(cue.text or "") for _, cue in rows).strip(),
        "cue_indexes": [index for index, _ in rows],
    }


def _extract_audio_window(media: Path, output: Path, *, start: float, end: float) -> None:
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_ready_for_forced_alignment")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(max(0.0, start)),
        "-t",
        str(max(0.1, end - start)),
        "-i",
        str(media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0 or not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(f"forced_aligner_audio_extract_failed: {completed.stderr[-500:]}")


def _read_audio_tuple(path: Path) -> tuple[Any, int]:
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError(f"soundfile_not_ready_for_forced_alignment: {exc}") from exc
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, int(sample_rate)


def _timestamps_monotonic(words: list[dict[str, Any]]) -> bool:
    previous = -1.0
    for row in words:
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        if start < previous - 0.05 or end < start:
            return False
        previous = end
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


def _device(value: str, torch: Any) -> str:
    requested = str(value or "auto").lower()
    if requested == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _language(value: str) -> str:
    text = str(value or "").strip()
    aliases = {"zh": "Chinese", "zh-cn": "Chinese", "cn": "Chinese", "en": "English"}
    return aliases.get(text.lower(), text or "Chinese")


def _write_failure(output: Path, code: str, error: str, *, model: str) -> dict[str, Any]:
    payload = _failure(code, error, model=model)
    write_json(output, payload)
    return payload


def _failure(code: str, error: str, *, model: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "provider": "qwen3-forced-aligner",
        "model": model,
        "ok": False,
        "status": "failed",
        "error_code": code,
        "error": error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
