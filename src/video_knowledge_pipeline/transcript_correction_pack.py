from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .bilinote_summary_tools import (
    apply_transcript_corrections,
    build_transcript_correction_messages,
    correction_stats,
    parse_transcript_correction_json,
    split_transcript_for_mind_map,
    transcript_segments_to_text,
)
from .models import TranscriptCue, now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .model_task_gateway import model_task_api_call
from .transcript import format_timestamp, parse_transcript

SCHEMA = "video_knowledge_pipeline.transcript_correction_pack.v1"
CORRECTED_SCHEMA = "video_knowledge_pipeline.llm_corrected_transcript.v1"


def build_transcript_correction_pack(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    max_segments: int = 0,
    max_chunk_chars: int = 5000,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    title = str(manifest.get("title") or root.name)
    cues = _load_best_transcript(root, manifest)
    segments = _segments_from_cues(cues)
    if max_segments and max_segments > 0:
        segments = segments[: int(max_segments)]
    transcript_text = transcript_segments_to_text(cues[: len(segments)])
    chunks = split_transcript_for_mind_map(transcript_text, max_chars=max_chunk_chars)
    messages = build_transcript_correction_messages(title=title, segments=segments)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": title,
        "status": "planned",
        "execute": bool(execute),
        "source_reuse": {
            "project": "PrideWood/bilinote",
            "commit_reviewed": "25ee9ba",
            "reused_patterns": [
                "transcript correction prompt with fixed segment indexes",
                "transcript chunking for full-length mind map coverage",
                "JSON-only correction response contract",
            ],
        },
        "segment_count": len(segments),
        "chunk_count": len(chunks),
        "max_chunk_chars": max_chunk_chars,
        "artifacts": {
            "pack_json": "exports/transcript-correction-pack.json",
            "pack_markdown": "exports/transcript-correction-pack.md",
            "messages_json": "exports/transcript-correction-llm-messages.json",
            "corrected_json": "llm-corrected-transcript.json",
            "corrected_srt": "llm-corrected-transcript.srt",
            "corrected_markdown": "llm-corrected-transcript.md",
        },
        "operator_boundary": "preview by default; execute=true may call a text LLM provider; imported corrections do not overwrite term-resolution corrected transcript",
        "updated_at": now_iso(),
    }
    correction_payload: dict[str, Any] | None = None
    if input_json:
        correction_payload = _load_correction_payload(input_json)
        result["status"] = "imported"
        result["input_json"] = str(Path(input_json).expanduser().resolve())
    elif execute:
        if not provider_config:
            result["status"] = "missing_provider_config"
            result["ok"] = False
        else:
            response = model_task_api_call(
                "transcript_correction_pack", provider_config=provider_config, messages=messages,
                execute=True, temperature=0.1, response_format={"type": "json_object"},
                max_tokens=4000, write=False,
            )
            result["provider_call"] = {"ok": bool(response.get("ok")), "error": response.get("error", "")}
            if response.get("ok"):
                correction_payload = parse_transcript_correction_json(str(response.get("content") or ""))
                result["status"] = "executed"
            else:
                result["status"] = "provider_failed"
                result["ok"] = False
    if correction_payload is not None:
        corrected = apply_transcript_corrections(segments, correction_payload)
        stats = correction_stats(segments, corrected)
        corrected_payload = {
            "schema": CORRECTED_SCHEMA,
            "bundle_dir": str(root),
            "title": title,
            "source": "bilinote_style_llm_correction",
            "summary": stats,
            "segments": corrected,
            "updated_at": now_iso(),
        }
        result["correction_summary"] = stats
        result["corrected_transcript"] = {
            "json_path": str(root / "llm-corrected-transcript.json"),
            "srt_path": str(root / "llm-corrected-transcript.srt"),
            "markdown_path": str(root / "llm-corrected-transcript.md"),
        }
        if write:
            write_json(root / "llm-corrected-transcript.json", corrected_payload)
            (root / "llm-corrected-transcript.srt").write_text(_render_srt(corrected), encoding="utf-8")
            (root / "llm-corrected-transcript.md").write_text(_render_markdown(corrected_payload), encoding="utf-8")
            manifest["llm_corrected_transcript_json"] = "llm-corrected-transcript.json"
            manifest["llm_corrected_transcript_srt"] = "llm-corrected-transcript.srt"
            manifest["llm_corrected_transcript_markdown"] = "llm-corrected-transcript.md"
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "transcript-correction-llm-messages.json", {"messages": messages})
        write_json(exports / "transcript-correction-pack.json", {**result, "segments": segments, "chunks": chunks})
        (exports / "transcript-correction-pack.md").write_text(_render_pack_markdown(result, segments, chunks), encoding="utf-8")
        manifest["transcript_correction_pack_json"] = "exports/transcript-correction-pack.json"
        manifest["transcript_correction_pack_markdown"] = "exports/transcript-correction-pack.md"
        manifest["transcript_correction_messages_json"] = "exports/transcript-correction-llm-messages.json"
        manifest["transcript_correction_summary"] = {
            "status": result.get("status"),
            "segment_count": len(segments),
            "chunk_count": len(chunks),
            "updated_at": result.get("updated_at"),
        }
        write_json(root / "manifest.json", manifest)
    result["run_registry"] = _register_run(root, result, write=write)
    return result


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    artifacts = [
        {"key": "pack_json", "path": root / "exports" / "transcript-correction-pack.json"},
        {"key": "pack_markdown", "path": root / "exports" / "transcript-correction-pack.md"},
        {"key": "messages_json", "path": root / "exports" / "transcript-correction-llm-messages.json"},
        {"key": "corrected_json", "path": root / "llm-corrected-transcript.json"},
        {"key": "corrected_srt", "path": root / "llm-corrected-transcript.srt"},
        {"key": "corrected_markdown", "path": root / "llm-corrected-transcript.md"},
    ]
    failed_items: list[dict[str, Any]] = []
    segment_count = int(result.get("segment_count") or 0)
    result_status = str(result.get("status") or "planned")
    if segment_count <= 0:
        failed_items.append({"id": "transcript", "reason": "transcript_missing", "detail": "No transcript segments were available for correction."})
    if result_status == "missing_provider_config":
        failed_items.append({"id": "provider_config", "reason": "missing_provider_config", "detail": "execute=true requires provider_config."})
    if result_status == "provider_failed":
        provider_call = result.get("provider_call") if isinstance(result.get("provider_call"), dict) else {}
        failed_items.append({"id": "provider_call", "reason": "provider_failed", "detail": str(provider_call.get("error") or "text LLM provider failed")})
    correction_summary = result.get("correction_summary") if isinstance(result.get("correction_summary"), dict) else {}
    if result_status in {"imported", "executed"} and int(correction_summary.get("corrected_segments") or 0) <= 0:
        failed_items.append({"id": "correction_summary", "reason": "no_corrections_applied", "detail": "Correction payload was accepted but did not change any segment."})
    if segment_count <= 0 or result_status == "missing_provider_config":
        status = "needs_input"
    elif result_status == "provider_failed":
        status = "needs_retry"
    elif result_status == "planned":
        status = "needs_execution"
    elif failed_items:
        status = "needs_review"
    else:
        status = "completed"
    return register_bundle_run(
        root,
        run_type="transcript_correction_pack",
        run_id="transcript-correction-pack",
        status=status,
        title="Transcript correction pack",
        summary=f"Prepared {segment_count} transcript segments in {int(result.get('chunk_count') or 0)} chunk(s); status={result_status}.",
        inputs={
            "bundle_dir": str(root),
            "input_json": str(result.get("input_json") or ""),
        },
        parameters={
            "execute": bool(result.get("execute")),
            "segment_count": segment_count,
            "chunk_count": int(result.get("chunk_count") or 0),
            "max_chunk_chars": int(result.get("max_chunk_chars") or 0),
            "result_status": result_status,
            "corrected_segments": int(correction_summary.get("corrected_segments") or 0),
        },
        artifacts=artifacts,
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 transcript-correction-pack {_ps_quote(str(root))}",
        next_actions=_run_next_actions(status),
        operator_boundary={
            "local_only_by_default": True,
            "execute_may_call_text_llm": True,
            "no_video_download": True,
            "does_not_overwrite_term_resolution": True,
            "human_review_required_before_promoting_low_confidence_terms": True,
            "source_reuse": "PrideWood/BiliNote transcript correction prompt and chunking workflow.",
        },
        write=write,
    )


def _run_next_actions(status: str) -> list[str]:
    if status == "needs_execution":
        return ["Review exports/transcript-correction-pack.md, then import reviewed corrections with --input-json or rerun with --execute and provider_config."]
    if status == "needs_input":
        return ["Add a normalized/source transcript or provider_config, then rerun transcript-correction-pack."]
    if status == "needs_retry":
        return ["Inspect provider_call.error and rerun transcript-correction-pack with a working text LLM provider."]
    if status == "needs_review":
        return ["Review llm-corrected-transcript.md before using it as a corrected transcript source."]
    return ["Open llm-corrected-transcript.md or exports/transcript-correction-pack.md for human review."]


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("manifest.json must be a JSON object")
    return data


def _load_best_transcript(root: Path, manifest: dict[str, Any]) -> list[TranscriptCue]:
    keys = (
        "corrected_transcript_json",
        "normalized_transcript_json",
        "transcript_json",
        "source_transcript",
        "transcript_path",
        "corrected_transcript_srt",
        "normalized_transcript_srt",
        "transcript_srt",
    )
    candidates: list[Path] = []
    for key in keys:
        value = str(manifest.get(key) or "").strip()
        if value:
            candidates.append(_bundle_path(root, value))
    candidates.extend(
        [
            root / "corrected-transcript.json",
            root / "normalized-transcript.json",
            root / "transcript.json",
            root / "corrected-transcript.srt",
            root / "normalized-transcript.srt",
            root / "transcript.srt",
            root / "timeline-transcript.json",
            root / "timeline-transcript.srt",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        cues = parse_transcript(path)
        if cues:
            return cues
    timeline = root / "timeline.json"
    if timeline.exists():
        data = read_json(timeline)
        if isinstance(data, list):
            cues = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("corrected_transcript") or item.get("transcript") or item.get("text") or "").strip()
                if text:
                    cues.append(TranscriptCue(start=_seconds(item.get("start")), end=_seconds(item.get("end")), text=text))
            if cues:
                return cues
    return []


def _segments_from_cues(cues: list[TranscriptCue]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "start": cue.start,
            "end": cue.end,
            "timestamp": format_timestamp(cue.start),
            "text": cue.text,
        }
        for index, cue in enumerate(cues)
        if str(cue.text or "").strip()
    ]


def _load_correction_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    text = target.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = parse_transcript_correction_json(text)
    if not isinstance(data, dict):
        raise ValueError("input_json must be a JSON object")
    if "segments" not in data:
        raise ValueError("input_json must contain segments")
    return {"segments": [row for row in data.get("segments") or [] if isinstance(row, dict)]}


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for position, segment in enumerate(segments, start=1):
        text = str(segment.get("corrected_text") or segment.get("text") or "").strip()
        if not text:
            continue
        blocks.extend(
            [
                str(position),
                f"{format_timestamp(_seconds(segment.get('start'))).replace('.', ',')} --> {format_timestamp(_seconds(segment.get('end'))).replace('.', ',')}",
                text,
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + ("\n" if blocks else "")


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# LLM Corrected Transcript",
        "",
        f"- Source: `{payload.get('source')}`",
        f"- Segments: `{summary.get('segments', 0)}`",
        f"- Corrected segments: `{summary.get('corrected_segments', 0)}`",
        f"- Indexes preserved: `{summary.get('indexes_preserved')}`",
        "",
    ]
    for segment in payload.get("segments") or []:
        changed = " changed" if segment.get("changed") else ""
        lines.extend([f"## {segment.get('index')}. {segment.get('timestamp')}{changed}", "", str(segment.get("corrected_text") or segment.get("text") or "").strip(), ""])
        if segment.get("changed"):
            lines.extend(["原文：", "", str(segment.get("raw_text") or "").strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_pack_markdown(result: dict[str, Any], segments: list[dict[str, Any]], chunks: list[str]) -> str:
    lines = [
        "# Transcript Correction Pack",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Segments: `{len(segments)}`",
        f"- Chunks: `{len(chunks)}`",
        f"- Source reuse: `PrideWood/bilinote@25ee9ba`",
        "",
        "## Boundary",
        "",
        "Preview by default. Provider execution is explicit. Imported corrections write `llm-corrected-transcript.*` and do not overwrite term-resolution corrected transcript.",
        "",
        "## First Segments",
        "",
    ]
    for segment in segments[:20]:
        lines.append(f"- `{segment.get('timestamp')}` {segment.get('text')}")
    lines.extend(["", "## Chunk Plan", ""])
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"- Chunk {index}: `{len(chunk)}` chars")
    return "\n".join(lines).rstrip() + "\n"


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except Exception:
        return 0.0
