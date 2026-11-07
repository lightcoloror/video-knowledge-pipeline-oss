from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .asr_adapter import render_srt
from .models import TranscriptCue, now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import format_timestamp, parse_transcript
from .transcript_readable_llm import (
    _agent_substitute_polish_text,
    _readable_semantic_guard_issues,
)
from .transcript_speakers import speaker_display_name, speaker_label_map, speaker_payload

SCHEMA = "video_knowledge_pipeline.agent_readable_transcript_rewrite.v1"
OUTPUT_SCHEMA = "video_knowledge_pipeline.agent_readable_transcript.v1"
TASK_SCHEMA = "video_knowledge_pipeline.agent_readable_transcript_task.v1"
SUPPORTED_AGENTS = ["codex", "workbuddy", "opencode", "hermes_agent", "openclaw", "custom_local_agent"]


def run_agent_readable_transcript_rewrite(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    agent_name: str = "local_agent",
    source_path: str | Path | None = None,
    promote: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Create an agent-readable transcript pass without cloud calls.

    This command is the stable handoff point for Codex/OpenClaw/Hermes/WorkBuddy
    style local agents. By default it writes a task pack plus a conservative local
    rewrite. If a reviewed agent output is supplied with input_json, the same
    validator/import path is used.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    source = _resolve_source(root, manifest, source_path)
    cues = parse_transcript(source)
    agent = _normalise_agent_name(agent_name)
    task = _build_task(root, source, cues, agent_name=agent)
    if input_json:
        imported = _load_import(input_json)
        rows, rejected = _apply_import_rows(cues, imported)
        status = "imported" if not rejected else "imported_with_rejections"
        source_label = "agent_import"
    else:
        rows = _local_agent_rows(cues)
        rejected = []
        status = "agent_substitute_executed"
        source_label = "local_agent_substitute"
    output = _build_output(root, source, cues, rows, source_label=source_label, agent_name=agent)
    quality = _quality_summary(cues, output.get("segments") or [])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "source_path": str(source),
        "status": status,
        "ok": bool(rows) and not rejected,
        "agent_name": agent,
        "compatible_agent_runtimes": SUPPORTED_AGENTS,
        "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
        "segment_count": len(cues),
        "rewritten_segment_count": len(output.get("segments") or []),
        "rejected_count": len(rejected),
        "rejected_rows": rejected,
        "quality": quality,
        "artifacts": {
            "task_json": "agent-readable-transcript-task.json",
            "task_markdown": "agent-readable-transcript-task.md",
            "json": "agent-readable-transcript.json",
            "srt": "agent-readable-transcript.srt",
            "markdown": "agent-readable-transcript.md",
            "report_json": "agent-readable-transcript-rewrite.json",
            "report_markdown": "agent-readable-transcript-rewrite.md",
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "does_not_modify_raw_asr": True,
            "fact_correction_out_of_scope": True,
            "surface_character_lock_required": True,
            "semantic_correction_should_use": "transcript-evidence-correction-pipeline",
            "input_json_allows_external_agent_review": True,
            "promote_requires_flag": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_task(root, task)
        _write_output(root, output, manifest, promote=promote)
        write_json(root / "agent-readable-transcript-rewrite.json", result)
        (root / "agent-readable-transcript-rewrite.md").write_text(_render_report(result), encoding="utf-8")
        manifest["agent_readable_transcript_rewrite_json"] = "agent-readable-transcript-rewrite.json"
        manifest["agent_readable_transcript_rewrite_markdown"] = "agent-readable-transcript-rewrite.md"
        manifest["agent_readable_transcript_task_json"] = "agent-readable-transcript-task.json"
        manifest["agent_readable_transcript_task_markdown"] = "agent-readable-transcript-task.md"
        write_json(root / "manifest.json", manifest)
    result["run_artifact"] = _register_run(root, result, write=write)
    return result


def _normalise_agent_name(agent_name: str) -> str:
    name = str(agent_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not name or name == "local_agent":
        return "local_agent"
    aliases = {"hermes": "hermes_agent", "hermesagent": "hermes_agent", "open_code": "opencode", "open_claw": "openclaw"}
    return aliases.get(name, name)


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    data = read_json(path) if path.exists() else {}
    return data if isinstance(data, dict) else {}


def _resolve_source(root: Path, manifest: dict[str, Any], source_path: str | Path | None) -> Path:
    if source_path:
        path = Path(source_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"source transcript not found: {path}")
    keys = [
        "source_arbitrated_transcript_json",
        "human_corrected_transcript_json",
        "llm_corrected_transcript_json",
        "corrected_transcript_json",
        "readable_transcript_json",
        "postprocessed_transcript_json",
        "normalized_transcript_json",
        "transcript_json",
    ]
    for key in keys:
        value = str(manifest.get(key) or "").strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
    for name in ("source-arbitrated-transcript.json", "corrected-transcript.json", "postprocessed-transcript.json", "normalized-transcript.json"):
        path = root / name
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"no transcript sidecar found for bundle: {root}")


def _build_task(root: Path, source: Path, cues: list[TranscriptCue], *, agent_name: str) -> dict[str, Any]:
    return {
        "schema": TASK_SCHEMA,
        "bundle_dir": str(root),
        "source_path": str(source),
        "agent_name": agent_name,
        "instructions": [
            "Polish punctuation, whitespace, line breaks, and paragraph boundaries only.",
            "Do not add, delete, replace, or reorder any non-punctuation character; use evidence correction for wording or facts.",
            "Keep every item tied to its original index/start/end; merge only when you can preserve source_indexes.",
            "Return JSON: {segments:[{index,start,end,text,source_indexes?}]}",
        ],
        "segments": [{"index": idx, "start": cue.start, "end": cue.end, "text": cue.text} for idx, cue in enumerate(cues)],
        "created_at": now_iso(),
    }


def _write_task(root: Path, task: dict[str, Any]) -> None:
    write_json(root / "agent-readable-transcript-task.json", task)
    lines = ["# Agent Readable Transcript Task", "", "## Instructions", ""]
    lines.extend(f"- {item}" for item in task.get("instructions") or [])
    lines += ["", "## Segments", ""]
    for row in task.get("segments") or []:
        lines.append(f"- `{row.get('index')}` `{format_timestamp(float(row.get('start') or 0))} - {format_timestamp(float(row.get('end') or 0))}` {row.get('text') or ''}")
    (root / "agent-readable-transcript-task.md").write_text("\n".join(lines), encoding="utf-8")


def _local_agent_rows(cues: list[TranscriptCue]) -> list[dict[str, Any]]:
    rows = []
    for index, cue in enumerate(cues):
        text = _agent_substitute_polish_text(str(cue.text or ""))
        if text.strip():
            rows.append({"index": index, "text": text, "original_text": cue.text, "source_indexes": [index]})
    return rows


def _load_import(input_json: str | Path) -> dict[str, Any]:
    path = Path(input_json).expanduser().resolve()
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".md":
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("agent readable import must be a JSON object")
    return data


def _apply_import_rows(cues: list[TranscriptCue], imported: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[int] = set()
    claimed_source_indexes: set[int] = set()
    for item in imported.get("segments") or imported.get("rows") or []:
        if not isinstance(item, dict):
            rejected.append({"reason": "invalid_row", "detail": str(item)[:120]})
            continue
        try:
            index = int(item.get("index"))
        except Exception:
            rejected.append({"reason": "invalid_index", "detail": str(item)[:120]})
            continue
        if index < 0 or index >= len(cues) or index in seen:
            rejected.append({"index": index, "reason": "index_out_of_range_or_duplicate"})
            continue
        source_indexes_raw = item.get("source_indexes")
        if source_indexes_raw is None:
            source_indexes = [index]
        elif not isinstance(source_indexes_raw, list) or not source_indexes_raw:
            rejected.append({"index": index, "reason": "invalid_source_indexes"})
            continue
        else:
            try:
                source_indexes = [int(value) for value in source_indexes_raw]
            except Exception:
                rejected.append({"index": index, "reason": "invalid_source_indexes"})
                continue
        expected_indexes = list(range(source_indexes[0], source_indexes[-1] + 1))
        if (
            source_indexes != expected_indexes
            or source_indexes[0] != index
            or any(value < 0 or value >= len(cues) for value in source_indexes)
        ):
            rejected.append(
                {
                    "index": index,
                    "reason": "invalid_source_indexes",
                    "detail": "source_indexes must be an ordered contiguous range beginning at index",
                }
            )
            continue
        overlap = sorted(claimed_source_indexes.intersection(source_indexes))
        if overlap:
            rejected.append(
                {
                    "index": index,
                    "reason": "overlapping_source_indexes",
                    "source_indexes": source_indexes,
                    "overlap": overlap,
                }
            )
            continue
        text = str(item.get("text") or "").strip()
        original = "".join(str(cues[value].text or "") for value in source_indexes)
        if not text:
            rejected.append({"index": index, "reason": "empty_text"})
            continue
        if len(text) > max(24, len(original) * 2.4) or len(text) < max(1, int(len(original) * 0.3)):
            rejected.append({"index": index, "reason": "length_ratio_out_of_bounds", "original_chars": len(original), "text_chars": len(text)})
            continue
        semantic_issues = _readable_semantic_guard_issues(original, text)
        if semantic_issues:
            rejected.append(
                {
                    "index": index,
                    "reason": "readable_semantic_guard_failed",
                    "detail": "semantic content changed; route the edit through transcript evidence correction",
                    "source_indexes": source_indexes,
                    "issues": semantic_issues,
                }
            )
            continue
        seen.add(index)
        claimed_source_indexes.update(source_indexes)
        rows.append({"index": index, "text": text, "original_text": original, "source_indexes": source_indexes})
    return rows, rejected


def _build_output(root: Path, source: Path, cues: list[TranscriptCue], rows: list[dict[str, Any]], *, source_label: str, agent_name: str) -> dict[str, Any]:
    by_index = {int(row["index"]): row for row in rows if "index" in row}
    consumed_indexes = {
        int(source_index)
        for row in rows
        for source_index in (row.get("source_indexes") or [row.get("index")])
        if source_index is not None and int(source_index) != int(row.get("index"))
    }
    segments = []
    speaker_labels = speaker_label_map(cues)
    for index, cue in enumerate(cues):
        if index in consumed_indexes:
            continue
        row = by_index.get(index)
        text = str(row.get("text") if row else cue.text)
        source_indexes = [int(value) for value in (row.get("source_indexes") if row else [index])]
        source_cues = [cues[value] for value in source_indexes]
        segments.append({
            "start": source_cues[0].start,
            "end": source_cues[-1].end,
            "text": text,
            **speaker_payload(cue, speaker_labels),
            "metadata": {
                **dict(cue.metadata),
                "source": "agent_readable_transcript_rewrite" if row else "source_transcript_passthrough",
                "agent_name": agent_name,
                "source_label": source_label,
                "source_path": str(source),
                "original_text": cue.text,
                "segment_index": index,
                "source_indexes": source_indexes,
            },
        })
    return {"schema": OUTPUT_SCHEMA, "bundle_dir": str(root), "source_path": str(source), "created_at": now_iso(), "segments": segments}


def _write_output(root: Path, output: dict[str, Any], manifest: dict[str, Any], *, promote: bool) -> None:
    cues = [
        TranscriptCue(
            start=float(row.get("start") or 0),
            end=float(row.get("end") or 0),
            text=str(row.get("text") or ""),
            speaker=str(row.get("speaker") or ""),
            speaker_role=str(row.get("speaker_role") or ""),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in output.get("segments") or []
    ]
    write_json(root / "agent-readable-transcript.json", output)
    (root / "agent-readable-transcript.srt").write_text(render_srt(cues), encoding="utf-8")
    (root / "agent-readable-transcript.md").write_text(_render_transcript_markdown(output), encoding="utf-8")
    manifest["agent_readable_transcript_json"] = "agent-readable-transcript.json"
    manifest["agent_readable_transcript_srt"] = "agent-readable-transcript.srt"
    manifest["agent_readable_transcript_markdown"] = "agent-readable-transcript.md"
    if promote:
        write_json(root / "corrected-transcript.json", output)
        (root / "corrected-transcript.srt").write_text(render_srt(cues), encoding="utf-8")
        manifest["corrected_transcript_json"] = "corrected-transcript.json"
        manifest["corrected_transcript_srt"] = "corrected-transcript.srt"
        manifest["transcript_json"] = "corrected-transcript.json"
        manifest["transcript_srt"] = "corrected-transcript.srt"
        manifest["corrected_transcript_source"] = "agent_readable_transcript_rewrite"


def _render_transcript_markdown(output: dict[str, Any]) -> str:
    lines = ["# Agent Readable Transcript", ""]
    segments = [row for row in output.get("segments") or [] if isinstance(row, dict)]
    labels = speaker_label_map(segments)
    for row in segments:
        start = format_timestamp(float(row.get("start") or 0))
        end = format_timestamp(float(row.get("end") or 0))
        text = str(row.get("text") or "").strip()
        speaker = speaker_display_name(row, labels)
        prefix = f"**{speaker}** " if speaker else ""
        lines.append(f"- `{start} - {end}` {prefix}{text}")
    lines.append("")
    return "\n".join(lines)


def _quality_summary(cues: list[TranscriptCue], segments: list[dict[str, Any]]) -> dict[str, Any]:
    source_text = "\n".join(str(cue.text or "") for cue in cues)
    out_text = "\n".join(str(row.get("text") or "") for row in segments)
    punct = "。！？；：，,.!?;:"
    return {
        "source_chars": len(source_text),
        "output_chars": len(out_text),
        "source_punctuation_density_per_1000_chars": round(sum(ch in punct for ch in source_text) * 1000 / max(1, len(source_text)), 2),
        "output_punctuation_density_per_1000_chars": round(sum(ch in punct for ch in out_text) * 1000 / max(1, len(out_text)), 2),
    }


def _render_report(result: dict[str, Any]) -> str:
    q = result.get("quality") or {}
    return "\n".join([
        "# Agent Readable Transcript Rewrite",
        "",
        f"- Status: `{result.get('status')}`",
        f"- OK: `{result.get('ok')}`",
        f"- Agent: `{result.get('agent_name')}`",
        f"- Source: `{result.get('source_path')}`",
        f"- Segments: `{result.get('rewritten_segment_count')}/{result.get('segment_count')}`",
        f"- Punctuation density: `{q.get('source_punctuation_density_per_1000_chars')}` -> `{q.get('output_punctuation_density_per_1000_chars')}` / 1000 chars",
        f"- Rejected rows: `{result.get('rejected_count')}`",
        "",
        "## Boundary",
        "",
        "- Local only, no cloud call.",
        "- Does not perform factual or terminology arbitration.",
    ]) + "\n"


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    if not write:
        return {}
    artifacts = [value for value in (result.get("artifacts") or {}).values() if isinstance(value, str) and value]
    return register_bundle_run(
        root,
        run_type="agent-readable-transcript-rewrite",
        status=str(result.get("status") or "unknown"),
        title="Agent readable transcript rewrite",
        summary=f"Agent readable transcript rewrite {result.get('status')} for {result.get('rewritten_segment_count', 0)} segments.",
        parameters={"agent_name": result.get("agent_name"), "promote_requires_flag": True},
        artifacts=artifacts,
        failed_items=result.get("rejected_rows") or [],
        next_actions=["Run transcript-quality-gate before using this transcript for final exports."],
        operator_boundary=result.get("operator_boundary") or {},
    )
