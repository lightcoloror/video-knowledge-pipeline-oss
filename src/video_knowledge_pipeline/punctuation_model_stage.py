from __future__ import annotations

import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

from .asr_adapter import render_srt
from .asr_runner import _resolve_python_executable
from .funasr_python_runner import _resolve_local_model, _select_device
from .models import TranscriptCue, now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import parse_transcript


SCHEMA = "video_knowledge_pipeline.punctuation_model_stage.v1"
TERMINAL = "。！？!?"
ALL_PUNCTUATION = "，。！？；：、,.!?;:"
_CONTENT_DROP_RE = re.compile(r"[\s，。！？；：、,.!?;:]+")


def run_punctuation_model_stage(
    bundle_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    model: str = "ct-punc",
    device: str = "auto",
    block_chars: int = 480,
    execute: bool = False,
    promote: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    if execute and not write:
        raise ValueError("punctuation model execution requires write=true for audit artifacts")
    source = _resolve_input(root, input_path)
    cues = parse_transcript(source)
    blocks = _group_cues(cues, max_chars=max(80, int(block_chars or 480)))
    output_json = root / "punctuated-transcript.json"
    output_srt = root / "punctuated-transcript.srt"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "input_path": str(source),
        "model": model,
        "resolved_model": _resolve_local_model(model),
        "device": _select_device(device),
        "execute": bool(execute),
        "promote_requested": bool(promote),
        "status": "planned",
        "block_count": len(blocks),
        "processed_block_count": 0,
        "char_lock_passed": False,
        "quality_gate_passed": False,
        "promoted": False,
        "artifacts": {
            "json": str(output_json),
            "srt": str(output_srt),
        },
        "operator_boundary": {
            "local_only": True,
            "does_not_modify_raw_asr": True,
            "promotion_requires_char_lock": True,
            "no_cloud_call": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        result["next_actions"] = [
            f".\\scripts\\video-knowledge.ps1 punctuation-model-stage '{root}' --execute",
        ]
        _write_report(root, result, write=write)
        return result

    try:
        generated_blocks, runner = _run_model_blocks(root, blocks, model=model, device=device)
        result["runner"] = runner
    except Exception as exc:
        result.update({"status": "model_not_ready", "error": f"{type(exc).__name__}: {exc}"})
        _write_report(root, result, write=write)
        return result

    block_results: list[dict[str, Any]] = []
    output_cues: list[TranscriptCue] = []
    for index, block in enumerate(blocks, start=1):
        source_text = "".join(str(cue.text or "").strip() for cue in block)
        generated_text = str(generated_blocks[index - 1] or "") if index <= len(generated_blocks) else ""
        error = "" if generated_text else "empty_model_output"
        lock_passed = bool(generated_text) and _content_only(source_text) == _content_only(generated_text)
        if lock_passed:
            block_cues = _punctuated_text_to_cues(generated_text, float(block[0].start), float(block[-1].end))
            output_cues.extend(block_cues)
        block_results.append(
            {
                "block_index": index,
                "start": float(block[0].start),
                "end": float(block[-1].end),
                "source_chars": len(source_text),
                "output_chars": len(generated_text),
                "char_lock_passed": lock_passed,
                "error": error,
                "source_content_fingerprint": _content_only(source_text),
                "output_content_fingerprint": _content_only(generated_text),
            }
        )

    result["processed_block_count"] = sum(1 for row in block_results if not row["error"])
    result["blocks"] = block_results
    char_lock = bool(block_results) and all(row["char_lock_passed"] for row in block_results)
    metrics = _quality_metrics(output_cues)
    quality_gate = char_lock and metrics["punctuation_per_100_chars"] >= 2.0 and metrics["max_unpunctuated_span_chars"] <= 120
    result.update(
        {
            "char_lock_passed": char_lock,
            "quality_metrics": metrics,
            "quality_gate_passed": quality_gate,
            "status": "completed" if quality_gate else ("char_lock_failed" if not char_lock else "quality_gate_failed"),
        }
    )

    if write and output_cues and char_lock:
        payload = {
            "schema": "video_knowledge_pipeline.punctuated_transcript.v1",
            "provider": "funasr_ct_punc",
            "model": model,
            "resolved_model": result["resolved_model"],
            "source_path": str(source),
            "character_lock": {
                "passed": True,
                "source_content_chars": sum(len(row["source_content_fingerprint"]) for row in block_results),
                "output_content_chars": sum(len(row["output_content_fingerprint"]) for row in block_results),
            },
            "segments": [
                {
                    "index": index,
                    "start": cue.start,
                    "end": cue.end,
                    "text": cue.text,
                    "metadata": {"source": "ct_punc", "input_path": str(source)},
                }
                for index, cue in enumerate(output_cues, start=1)
            ],
        }
        write_json(output_json, payload)
        output_srt.write_text(render_srt(output_cues), encoding="utf-8")
        if promote and quality_gate:
            _promote(root, payload, output_cues, result)
    _write_report(root, result, write=write)
    return result


def _load_punctuation_model(model: str, device: str) -> Any:
    from funasr import AutoModel  # type: ignore

    kwargs: dict[str, Any] = {"model": model}
    if device in {"cuda", "cpu"}:
        kwargs["device"] = device
    return AutoModel(**kwargs)


def _run_model_blocks(
    root: Path,
    blocks: list[list[TranscriptCue]],
    *,
    model: str,
    device: str,
) -> tuple[list[str], dict[str, Any]]:
    request_path = root / "punctuation-model-request.json"
    output_path = root / "punctuation-model-runner-output.json"
    write_json(
        request_path,
        {
            "schema": "video_knowledge_pipeline.punctuation_model_request.v1",
            "blocks": [
                {"block_index": index, "text": "".join(str(cue.text or "").strip() for cue in block)}
                for index, block in enumerate(blocks, start=1)
            ],
        },
    )
    python_executable = _resolve_python_executable()
    command = [
        python_executable,
        "-m",
        "video_knowledge_pipeline.punctuation_model_runner",
        "--request-json",
        str(request_path),
        "--output-json",
        str(output_path),
        "--model",
        model,
        "--device",
        device,
    ]
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
        env=env,
    )
    if not output_path.exists():
        raise RuntimeError(
            f"punctuation runner produced no output; returncode={completed.returncode}; "
            f"stderr={completed.stderr[-500:]}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8-sig"))
    if completed.returncode != 0 or not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or completed.stderr[-500:] or "punctuation runner failed"))
    outputs = [str(value or "") for value in payload.get("outputs") or []]
    if len(outputs) != len(blocks):
        raise RuntimeError(f"punctuation runner block mismatch: expected {len(blocks)}, got {len(outputs)}")
    return outputs, {
        "python_executable": python_executable,
        "command_module": "video_knowledge_pipeline.punctuation_model_runner",
        "returncode": completed.returncode,
        "output_json": str(output_path),
    }


def _generate_text(model: Any, text: str) -> str:
    generated = model.generate(input=text)
    value: Any = generated
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        for key in ("text", "text_postprocessed", "sentence", "value"):
            if value.get(key):
                return str(value[key]).strip()
        nested = value.get("result")
        if isinstance(nested, list) and nested:
            return _generate_text_result(nested[0])
    return str(value or "").strip()


def _generate_text_result(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "text_postprocessed", "sentence", "value"):
            if value.get(key):
                return str(value[key]).strip()
    return str(value or "").strip()


def _group_cues(cues: list[TranscriptCue], *, max_chars: int) -> list[list[TranscriptCue]]:
    groups: list[list[TranscriptCue]] = []
    current: list[TranscriptCue] = []
    current_chars = 0
    for cue in cues:
        text = str(cue.text or "").strip()
        if not text:
            continue
        if current and current_chars + len(text) > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(TranscriptCue(start=cue.start, end=cue.end, text=text))
        current_chars += len(text)
    if current:
        groups.append(current)
    return groups


def _punctuated_text_to_cues(text: str, start: float, end: float) -> list[TranscriptCue]:
    chunks = [chunk.strip() for chunk in re.findall(r".+?(?:[。！？!?]+|$)", text, flags=re.DOTALL) if chunk.strip()]
    if not chunks:
        return []
    total = sum(max(len(_content_only(chunk)), 1) for chunk in chunks)
    duration = max(0.001, end - start)
    rows: list[TranscriptCue] = []
    cursor = start
    for index, chunk in enumerate(chunks):
        weight = max(len(_content_only(chunk)), 1) / total
        chunk_end = end if index == len(chunks) - 1 else min(end, cursor + duration * weight)
        rows.append(TranscriptCue(start=round(cursor, 3), end=round(chunk_end, 3), text=chunk))
        cursor = chunk_end
    return rows


def _quality_metrics(cues: list[TranscriptCue]) -> dict[str, Any]:
    text = "".join(cue.text for cue in cues)
    punctuation_count = sum(1 for char in text if char in ALL_PUNCTUATION)
    terminal_count = sum(1 for char in text if char in TERMINAL)
    spans = [len(_content_only(value)) for value in re.split(r"[。！？!?]+", text)]
    return {
        "character_count": len(text),
        "punctuation_count": punctuation_count,
        "terminal_count": terminal_count,
        "punctuation_per_100_chars": round(punctuation_count * 100 / max(len(text), 1), 4),
        "max_unpunctuated_span_chars": max(spans, default=0),
        "segment_count": len(cues),
    }


def _content_only(text: str) -> str:
    return "".join(
        char
        for char in str(text or "")
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _resolve_input(root: Path, input_path: str | Path | None) -> Path:
    if input_path:
        path = Path(input_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    for key in ("source_arbitrated_transcript_json", "normalized_transcript_json", "asr_transcript_json"):
        value = str(manifest.get(key) or "").strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            if path.exists():
                return path.resolve()
    for name in ("source-arbitrated-transcript.json", "normalized-transcript.json"):
        path = root / name
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"no ASR transcript found: {root}")


def _promote(root: Path, payload: dict[str, Any], cues: list[TranscriptCue], result: dict[str, Any]) -> None:
    corrected_json = root / "corrected-transcript.json"
    corrected_srt = root / "corrected-transcript.srt"
    write_json(corrected_json, {**payload, "schema": "video_knowledge_pipeline.corrected_transcript.v1"})
    corrected_srt.write_text(render_srt(cues), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest["punctuated_transcript_json"] = "punctuated-transcript.json"
    manifest["punctuated_transcript_srt"] = "punctuated-transcript.srt"
    manifest["corrected_transcript_json"] = "corrected-transcript.json"
    manifest["corrected_transcript_srt"] = "corrected-transcript.srt"
    manifest["transcript_json"] = "corrected-transcript.json"
    manifest["transcript_srt"] = "corrected-transcript.srt"
    manifest["punctuation_model_stage"] = {
        "status": result["status"],
        "model": result["model"],
        "char_lock_passed": result["char_lock_passed"],
        "quality_gate_passed": result["quality_gate_passed"],
    }
    write_json(manifest_path, manifest)
    result["promoted"] = True
    result["artifacts"]["corrected_json"] = str(corrected_json)
    result["artifacts"]["corrected_srt"] = str(corrected_srt)


def _write_report(root: Path, result: dict[str, Any], *, write: bool) -> None:
    if not write:
        return
    write_json(root / "punctuation-model-stage.json", result)
    lines = [
        "# Punctuation Model Stage",
        "",
        f"- Status: {result.get('status')}",
        f"- Model: {result.get('model')}",
        f"- Device: {result.get('device')}",
        f"- Blocks: {result.get('processed_block_count')}/{result.get('block_count')}",
        f"- Character lock: {result.get('char_lock_passed')}",
        f"- Quality gate: {result.get('quality_gate_passed')}",
        f"- Promoted: {result.get('promoted')}",
    ]
    (root / "punctuation-model-stage.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    run = register_bundle_run(
        root,
        run_type="punctuation_model_stage",
        run_id="punctuation-model-stage",
        status="completed" if result.get("promoted") else ("needs_execution" if result.get("status") == "planned" else "needs_review"),
        title="FunASR ct-punc punctuation stage",
        summary=f"status={result.get('status')}; char_lock={result.get('char_lock_passed')}; promoted={result.get('promoted')}",
        inputs={"input_path": result.get("input_path")},
        parameters={"model": result.get("model"), "device": result.get("device"), "execute": result.get("execute")},
        artifacts=[{"key": key, "path": value} for key, value in (result.get("artifacts") or {}).items()],
        failed_items=[] if result.get("promoted") else [{"id": "punctuation", "reason": result.get("status"), "detail": result.get("error") or "not promoted"}],
        retry_command=f".\\scripts\\video-knowledge.ps1 punctuation-model-stage '{root}' --execute --promote",
        next_actions=["Run export-knowledge-note after promotion."],
        operator_boundary=result["operator_boundary"],
        write=True,
    )
    result["run_artifact"] = run
    write_json(root / "punctuation-model-stage.json", result)