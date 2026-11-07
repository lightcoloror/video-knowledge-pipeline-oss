from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import TranscriptCue, now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json
from .term_text import apply_high_confidence_term_replacements, high_confidence_term_replacements
from .transcript import format_timestamp, parse_transcript
from .transcript_speakers import speaker_label_map, speaker_payload

SCHEMA = "video_knowledge_pipeline.transcript_source_arbitration.v1"
CORRECTED_SCHEMA = "video_knowledge_pipeline.source_arbitrated_transcript.v1"

_FILE_ASR_SOURCE_TYPES = {"asr_explicit", "asr_normalized", "asr", "human_corrected", "llm_corrected"}
_DERIVED_TRANSCRIPT_MARKERS = (
    "corrected-transcript",
    "source-arbitrated-transcript",
    "transcript-source-arbitration",
    "agent-readable-transcript",
    "llm-readable-transcript",
    "readable-transcript",
    "postprocessed-transcript",
)


@dataclass
class TranscriptSource:
    source_id: str
    source_type: str
    weight: float
    path: str
    cues: list[TranscriptCue]


def arbitrate_transcript_sources(
    bundle_dir: str | Path,
    *,
    platform_subtitle: str | Path | None = None,
    subtitle: str | Path | None = None,
    asr_json: str | Path | None = None,
    glossary_json: str | Path | None = None,
    min_confidence: float = 0.72,
    promote: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Build a corrected transcript by voting across ASR, subtitles, and term evidence.

    This is a local arbitration layer. It does not call an LLM and it keeps raw
    transcript sources intact. High-confidence changes are written to a separate
    source-arbitrated transcript sidecar; low-confidence conflicts become review
    rows rather than silent replacements.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    explicit_paths = {
        "explicit_asr": asr_json,
        "explicit_platform_subtitle": platform_subtitle,
        "explicit_subtitle": subtitle,
    }
    sources = _load_sources(root, manifest, explicit_paths=explicit_paths)
    base = _choose_base_source(sources)
    term_item = _load_term_item(root, glossary_json=glossary_json)
    previous_corrected = {
        "corrected_transcript_json": manifest.get("corrected_transcript_json", ""),
        "corrected_transcript_srt": manifest.get("corrected_transcript_srt", ""),
        "corrected_transcript_markdown": manifest.get("corrected_transcript_markdown", ""),
    }

    if base is None:
        result = {
            "schema": SCHEMA,
            "bundle_dir": str(root),
            "status": "no_transcript_sources",
            "ok": False,
            "source_count": len(sources),
            "sources": [_source_row(source) for source in sources],
            "updated_at": now_iso(),
        }
        if write:
            _write_report_only(root, manifest, result)
        return result

    upstream_asr_quality = _source_asr_quality(base)
    segments = _arbitrate_segments(base, sources, term_item=term_item, min_confidence=min_confidence)
    _apply_upstream_asr_quality(segments, upstream_asr_quality)
    changed_count = sum(1 for row in segments if row.get("changed"))
    review_rows = [row for row in segments if row.get("needs_human_review")]
    quality_summary = _arbitration_quality_summary(
        segments,
        sources=sources,
        min_confidence=min_confidence,
        promote=promote,
    )
    status = "completed" if not review_rows else "completed_with_review_items"
    corrected_payload = {
        "schema": CORRECTED_SCHEMA,
        "bundle_dir": str(root),
        "source": "transcript_source_arbitration",
        "base_source": _source_row(base),
        "upstream_asr_quality": upstream_asr_quality,
        "updated_at": now_iso(),
        "summary": {
            "segments": len(segments),
            "changed_segments": changed_count,
            "review_segments": len(review_rows),
            "source_count": len(sources),
            "promoted_to_corrected_transcript": bool(promote),
            "quality_status": quality_summary.get("status"),
            "average_confidence": quality_summary.get("average_confidence"),
            "high_confidence_term_replacements": quality_summary.get("high_confidence_term_replacements"),
            "low_confidence_conflicts": quality_summary.get("low_confidence_conflicts"),
        },
        "quality_summary": quality_summary,
        "segments": segments,
    }
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": status,
        "ok": True,
        "base_source": _source_row(base),
        "upstream_asr_quality": upstream_asr_quality,
        "source_count": len(sources),
        "sources": [_source_row(source) for source in sources],
        "summary": corrected_payload["summary"],
        "quality_summary": quality_summary,
        "previous_corrected_manifest": previous_corrected,
        "artifacts": {
            "json": str(root / "transcript-source-arbitration.json"),
            "markdown": str(root / "transcript-source-arbitration.md"),
            "corrected_json": str(root / "source-arbitrated-transcript.json"),
            "corrected_srt": str(root / "source-arbitrated-transcript.srt"),
            "corrected_markdown": str(root / "source-arbitrated-transcript.md"),
            "mcp_args": str(root / "mcp-transcript-source-arbitration.args.json"),
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "does_not_modify_raw_sources": True,
            "low_confidence_conflicts_go_to_review": True,
        },
        "updated_at": now_iso(),
    }

    if write:
        with bundle_write_lock(root, operation="transcript_source_arbitration", timeout_seconds=1.0):
            write_json(root / "source-arbitrated-transcript.json", corrected_payload)
            (root / "source-arbitrated-transcript.srt").write_text(_render_srt(segments), encoding="utf-8")
            (root / "source-arbitrated-transcript.md").write_text(_render_corrected_markdown(corrected_payload), encoding="utf-8")
            write_json(root / "transcript-source-arbitration.json", {**result, "review_rows": review_rows})
            (root / "transcript-source-arbitration.md").write_text(_render_report(result, segments, review_rows), encoding="utf-8")
            write_json(
                root / "mcp-transcript-source-arbitration.args.json",
                {
                    "bundle_dir": str(root),
                    "platform_subtitle": str(Path(platform_subtitle).expanduser().resolve()) if platform_subtitle else "",
                    "subtitle": str(Path(subtitle).expanduser().resolve()) if subtitle else "",
                    "asr_json": str(Path(asr_json).expanduser().resolve()) if asr_json else "",
                    "glossary_json": str(Path(glossary_json).expanduser().resolve()) if glossary_json else "",
                    "min_confidence": min_confidence,
                    "promote": promote,
                    "write": True,
                },
            )
            manifest["transcript_source_arbitration_json"] = "transcript-source-arbitration.json"
            manifest["transcript_source_arbitration_markdown"] = "transcript-source-arbitration.md"
            manifest["source_arbitrated_transcript_json"] = "source-arbitrated-transcript.json"
            manifest["source_arbitrated_transcript_srt"] = "source-arbitrated-transcript.srt"
            manifest["source_arbitrated_transcript_markdown"] = "source-arbitrated-transcript.md"
            manifest["mcp_transcript_source_arbitration_args"] = "mcp-transcript-source-arbitration.args.json"
            manifest["transcript_source_arbitration_summary"] = corrected_payload["summary"]
            manifest["transcript_source_arbitration_quality"] = quality_summary
            if promote:
                manifest["corrected_transcript_json"] = "source-arbitrated-transcript.json"
                manifest["corrected_transcript_srt"] = "source-arbitrated-transcript.srt"
                manifest["corrected_transcript_markdown"] = "source-arbitrated-transcript.md"
                manifest["corrected_transcript_source"] = "transcript_source_arbitration"
            write_json(manifest_path, manifest)
            run_status = "completed" if not review_rows else "needs_review"
            register_bundle_run(
                root,
                run_type="transcript_source_arbitration",
                status=run_status,
                title="字幕/ASR 多源仲裁",
                summary=f"{changed_count} changed segments; {len(review_rows)} review segments; quality={quality_summary.get('status')}.",
                inputs={"base_source": base.source_id, "source_count": len(sources)},
                parameters={"min_confidence": min_confidence, "promote": promote},
                artifacts=[
                    {"key": "report", "path": root / "transcript-source-arbitration.md"},
                    {"key": "corrected_transcript", "path": root / "source-arbitrated-transcript.json"},
                    {"key": "mcp_args", "path": root / "mcp-transcript-source-arbitration.args.json"},
                ],
                failed_items=_review_failed_items(root, review_rows),
                retry_command=f".\\scripts\\video-knowledge.ps1 transcript-source-arbitration {root}",
                next_actions=["Review low-confidence transcript conflicts before treating them as facts."] if review_rows else [],
                operator_boundary=result["operator_boundary"],
                write=True,
            )
    return result



def _source_asr_quality(source: TranscriptSource) -> dict[str, Any]:
    path = Path(source.path)
    if path.suffix.lower() != ".json" or not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    execution = payload.get("source_execution")
    if not isinstance(execution, dict):
        return {}
    quality = execution.get("asr_quality")
    return dict(quality) if isinstance(quality, dict) else {}


def _apply_upstream_asr_quality(
    segments: list[dict[str, Any]],
    quality: dict[str, Any],
) -> None:
    if not quality:
        return
    for issue_type, key in (
        ("review", "review_chunks"),
        ("failed", "failed_chunks"),
    ):
        chunks = quality.get(key)
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            row = _match_upstream_asr_quality_segment(segments, chunk)
            if row is None:
                continue
            try:
                position = int(chunk.get("position") or 0)
            except (TypeError, ValueError):
                position = 0
            reasons = [
                str(reason)
                for reason in (chunk.get("reasons") or [])
                if str(reason).strip()
            ]
            reason = f"upstream_asr_{issue_type}"
            if reasons:
                reason = f"{reason}:{','.join(reasons)}"
            previous = str(row.get("review_reason") or "").strip()
            row["needs_human_review"] = True
            row["review_reason"] = f"{previous}; {reason}" if previous else reason
            row["upstream_asr_quality"] = {
                "status": str(quality.get("status") or ""),
                "issue_type": issue_type,
                "segment_id": str(chunk.get("segment_id") or ""),
                "position": position,
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "reasons": reasons,
                "preserve_original_text": bool(
                    chunk.get("preserve_original_text", True)
                ),
            }


def _match_upstream_asr_quality_segment(
    segments: list[dict[str, Any]],
    chunk: dict[str, Any],
) -> dict[str, Any] | None:
    segment_id = str(chunk.get("segment_id") or "").strip()
    if segment_id:
        for row in segments:
            row_ids = {
                str(row.get("segment_id") or "").strip(),
                *(str(value).strip() for value in (row.get("source_segment_ids") or [])),
            }
            if segment_id in row_ids:
                return row

    try:
        start = float(chunk.get("start"))
        end = float(chunk.get("end"))
    except (TypeError, ValueError):
        start = end = -1.0
    if end > start:
        best_row: dict[str, Any] | None = None
        best_overlap = 0.0
        for row in segments:
            row_start = _seconds(row.get("start"))
            row_end = _seconds(row.get("end"))
            overlap = max(0.0, min(end, row_end) - max(start, row_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_row = row
        if best_row is not None:
            return best_row

    try:
        position = int(chunk.get("position") or 0)
    except (TypeError, ValueError):
        position = 0
    if 1 <= position <= len(segments):
        return segments[position - 1]
    return None



def _review_failed_items(root: Path, review_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in review_rows[:50]:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        rows.append(
            {
                "index": index,
                "reason": "low_confidence_conflict",
                "detail": row.get("review_reason", ""),
                "time_range": f"{format_timestamp(_seconds(row.get('start')))} - {format_timestamp(_seconds(row.get('end')))}",
                "confidence": row.get("confidence"),
                "suggested_next_tool": "prepare_transcript_edit_session",
                "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 prepare-transcript-edit-session '{root}' --limit 0",
                "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session '{root}' --limit 0 --group-by reason",
            }
        )
    return rows

def _write_report_only(root: Path, manifest: dict[str, Any], result: dict[str, Any]) -> None:
    write_json(root / "transcript-source-arbitration.json", result)
    (root / "transcript-source-arbitration.md").write_text(_render_report(result, [], []), encoding="utf-8")
    write_json(root / "mcp-transcript-source-arbitration.args.json", {"bundle_dir": str(root), "write": True})
    manifest["transcript_source_arbitration_json"] = "transcript-source-arbitration.json"
    manifest["transcript_source_arbitration_markdown"] = "transcript-source-arbitration.md"
    manifest["mcp_transcript_source_arbitration_args"] = "mcp-transcript-source-arbitration.args.json"
    write_json(root / "manifest.json", manifest)


def _load_sources(root: Path, manifest: dict[str, Any], *, explicit_paths: dict[str, str | Path | None]) -> list[TranscriptSource]:
    candidates: list[tuple[str, str | Path, str, float]] = []
    for source_id, value in explicit_paths.items():
        if value:
            source_type = "asr_explicit" if source_id == "explicit_asr" else "platform_subtitle"
            candidates.append((source_id, value, source_type, 3.5))
    manifest_candidates = [
        ("human_corrected_transcript_json", "human_corrected", 8.0),
        ("human_corrected_transcript_srt", "human_corrected", 8.0),
        ("llm_corrected_transcript_json", "llm_corrected", 5.0),
        ("llm_corrected_transcript_srt", "llm_corrected", 5.0),
        ("normalized_transcript_json", "asr_normalized", 2.5),
        ("normalized_transcript_srt", "asr_normalized", 2.5),
        ("transcript_json", "asr", 2.0),
        ("transcript_srt", "asr", 2.0),
        ("source_transcript", "asr", 2.0),
        ("transcript_path", "asr", 2.0),
        ("platform_subtitle", "platform_subtitle", 3.0),
        ("platform_subtitle_path", "platform_subtitle", 3.0),
        ("source_subtitle", "platform_subtitle", 3.0),
        ("source_subtitle_path", "platform_subtitle", 3.0),
        ("subtitle_path", "platform_subtitle", 3.0),
        ("bilibili_subtitle", "platform_subtitle", 3.0),
        ("subtitle_json", "platform_subtitle", 3.0),
        ("subtitle_srt", "platform_subtitle", 3.0),
    ]
    for key, source_type, weight in manifest_candidates:
        value = manifest.get(key)
        if value:
            candidates.append((key, str(value), source_type, weight))
    root_candidates = [
        ("normalized-transcript.json", "asr_normalized", 2.5),
        ("normalized-transcript.srt", "asr_normalized", 2.5),
        ("transcript.json", "asr", 2.0),
        ("transcript.srt", "asr", 2.0),
        ("platform-subtitle.json", "platform_subtitle", 3.0),
        ("platform-subtitle.srt", "platform_subtitle", 3.0),
        ("platform-subtitle.vtt", "platform_subtitle", 3.0),
        ("subtitle.json", "platform_subtitle", 3.0),
        ("subtitle.srt", "platform_subtitle", 3.0),
        ("subtitle.vtt", "platform_subtitle", 3.0),
        ("source-subtitle.json", "platform_subtitle", 3.0),
        ("source-subtitle.srt", "platform_subtitle", 3.0),
        ("source-subtitle.vtt", "platform_subtitle", 3.0),
    ]
    for name, source_type, weight in root_candidates:
        candidates.append((name, name, source_type, weight))

    sources: list[TranscriptSource] = []
    seen: set[Path] = set()
    for source_id, value, source_type, weight in candidates:
        path = _bundle_path(root, str(value))
        if not path.exists() or path.resolve() in seen:
            continue
        if _should_ignore_derived_transcript(source_id, source_type, path):
            continue
        try:
            cues = parse_transcript(path)
        except Exception:
            continue
        if not cues:
            continue
        seen.add(path.resolve())
        candidate = TranscriptSource(
            source_id=source_id,
            source_type=source_type,
            weight=weight,
            path=str(path),
            cues=cues,
        )
        _append_unique_source(sources, candidate)

    timeline_sources = _timeline_sources(root)
    has_file_asr = any(source.source_type in _FILE_ASR_SOURCE_TYPES for source in sources)
    for source in timeline_sources:
        if source.source_id == "timeline_asr" and has_file_asr:
            continue
        sources.append(source)
    return sources


def _should_ignore_derived_transcript(source_id: str, source_type: str, path: Path) -> bool:
    if source_id == "explicit_asr" or source_type in {"human_corrected", "llm_corrected"}:
        return False
    if source_type not in {"asr", "asr_normalized"}:
        return False
    return _is_derived_transcript_artifact(path)


def _is_derived_transcript_artifact(path: Path) -> bool:
    name = _artifact_marker_text(path.name)
    if any(marker in name for marker in _DERIVED_TRANSCRIPT_MARKERS):
        return True
    if path.suffix.lower() != ".json":
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    metadata = " ".join(
        str(payload.get(key) or "")
        for key in ("schema", "source", "stage", "generator")
    )
    marker_text = _artifact_marker_text(metadata)
    return any(marker in marker_text for marker in _DERIVED_TRANSCRIPT_MARKERS)


def _artifact_marker_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def _append_unique_source(sources: list[TranscriptSource], candidate: TranscriptSource) -> None:
    group = _representation_group(candidate.source_type)
    for index, existing in enumerate(sources):
        if _representation_group(existing.source_type) != group:
            continue
        if not _cues_equivalent(existing.cues, candidate.cues):
            continue
        if _source_representation_rank(candidate) < _source_representation_rank(existing):
            sources[index] = candidate
        return
    sources.append(candidate)


def _representation_group(source_type: str) -> str:
    return str(source_type or "")


def _source_representation_rank(source: TranscriptSource) -> tuple[int, int]:
    explicit_rank = 0 if source.source_id == "explicit_asr" else 1
    suffix_rank = {".json": 0, ".srt": 1, ".vtt": 2}.get(Path(source.path).suffix.lower(), 3)
    return explicit_rank, suffix_rank


def _cues_equivalent(left: list[TranscriptCue], right: list[TranscriptCue]) -> bool:
    if len(left) != len(right):
        return False
    for left_cue, right_cue in zip(left, right):
        if abs(float(left_cue.start) - float(right_cue.start)) > 0.02:
            return False
        if abs(float(left_cue.end) - float(right_cue.end)) > 0.02:
            return False
        if _cue_text(left_cue.text) != _cue_text(right_cue.text):
            return False
    return True


def _cue_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _timeline_sources(root: Path) -> list[TranscriptSource]:
    path = root / "timeline.json"
    if not path.exists():
        return []
    try:
        timeline = read_json(path)
    except Exception:
        return []
    if not isinstance(timeline, list):
        return []
    asr: list[TranscriptCue] = []
    subtitle: list[TranscriptCue] = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        start = _seconds(item.get("start"))
        end = _seconds(item.get("end"), default=start)
        transcript = _text(item.get("transcript") or item.get("asr_text") or item.get("text"))
        caption = _text(item.get("subtitle") or item.get("caption") or item.get("original_subtitle"))
        if transcript:
            asr.append(TranscriptCue(start=start, end=end, text=transcript))
        if caption:
            subtitle.append(TranscriptCue(start=start, end=end, text=caption))
    sources: list[TranscriptSource] = []
    if asr:
        sources.append(TranscriptSource("timeline_asr", "asr_timeline", 1.8, str(path), asr))
    if subtitle:
        sources.append(TranscriptSource("timeline_subtitle", "platform_subtitle", 2.6, str(path), subtitle))
    return sources


def _choose_base_source(sources: list[TranscriptSource]) -> TranscriptSource | None:
    if not sources:
        return None
    priority = {
        "asr_explicit": 0,
        "human_corrected": 1,
        "asr_normalized": 2,
        "asr": 3,
        "asr_timeline": 4,
        "llm_corrected": 5,
        "platform_subtitle": 6,
    }
    return sorted(sources, key=lambda source: (priority.get(source.source_type, 9), -len(source.cues)))[0]


def _arbitrate_segments(
    base: TranscriptSource,
    sources: list[TranscriptSource],
    *,
    term_item: dict[str, Any],
    min_confidence: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    speaker_labels = speaker_label_map(base.cues)
    for index, cue in enumerate(base.cues, start=1):
        candidates = _candidate_texts(cue, base, sources, term_item=term_item)
        chosen = _choose_candidate(cue.text, candidates, min_confidence=min_confidence)
        corrected = chosen["text"]
        changed = corrected != cue.text
        row = {
                "index": index,
                "segment_id": cue.segment_id,
                "source_segment_ids": list(
                    cue.source_segment_ids
                    or ([cue.segment_id] if cue.segment_id else [])
                ),
                "start": cue.start,
                "end": cue.end,
                "timestamp": format_timestamp(cue.start),
                **speaker_payload(cue, speaker_labels),
                "metadata": dict(cue.metadata),
                "text": corrected,
                "corrected_text": corrected,
                "original_text": cue.text,
                "raw_text": cue.text,
                "changed": changed,
                "chosen_source": chosen.get("source_id"),
                "chosen_source_type": chosen.get("source_type"),
                "confidence": chosen.get("confidence"),
                "needs_human_review": chosen.get("needs_human_review", False),
                "review_reason": chosen.get("review_reason", ""),
                "arbitration_reasons": chosen.get("reasons", []),
                "term_replacements": chosen.get("term_replacements", []),
                # Intent: preserve source timing precision across local arbitration.
                # Decision: pass through the existing TranscriptCue transformations.
                # Reason: dropping coarse-timing provenance recreates false density retries.
                # Evidence: the 2026-07-24 Bundle regressed after arbitration/export.
                # Effective scope: source sidecar metadata only; text voting is unchanged.
                "transformations": [
                    dict(value)
                    for value in cue.transformations
                    if isinstance(value, dict)
                ],
                "alternatives": candidates[:8],
            }
        if not cue.text:
            row["empty_text_preserved"] = True
        segments.append(row)
    return segments


def _arbitration_quality_summary(
    segments: list[dict[str, Any]],
    *,
    sources: list[TranscriptSource],
    min_confidence: float,
    promote: bool,
) -> dict[str, Any]:
    total = len(segments)
    changed = [row for row in segments if row.get("changed")]
    review = [row for row in segments if row.get("needs_human_review")]
    confidences: list[float] = []
    high_confidence_term_replacements = 0
    review_reason_counts: dict[str, int] = {}
    chosen_source_type_counts: dict[str, int] = {}
    arbitration_reason_counts: dict[str, int] = {}
    for row in segments:
        try:
            confidences.append(float(row.get("confidence")))
        except Exception:
            pass
        high_confidence_term_replacements += len(row.get("term_replacements") or [])
        source_type = str(row.get("chosen_source_type") or "unknown")
        chosen_source_type_counts[source_type] = chosen_source_type_counts.get(source_type, 0) + 1
        if row.get("needs_human_review"):
            reason = str(row.get("review_reason") or "unknown")
            review_reason_counts[reason] = review_reason_counts.get(reason, 0) + 1
        for reason in row.get("arbitration_reasons") or []:
            key = str(reason or "unknown")
            arbitration_reason_counts[key] = arbitration_reason_counts.get(key, 0) + 1
    source_type_counts: dict[str, int] = {}
    for source in sources:
        source_type_counts[source.source_type] = source_type_counts.get(source.source_type, 0) + 1
    average_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    low_confidence_conflicts = len(review)
    if total <= 0:
        status = "missing_transcript"
    elif review:
        status = "needs_review"
    elif changed:
        status = "changed_clean"
    else:
        status = "clean"
    guidance: list[str] = []
    if high_confidence_term_replacements:
        guidance.append("High-confidence term replacements can be used in corrected transcript and smart-summary.")
    if review:
        guidance.append("Low-confidence transcript conflicts remain; do not treat those segments as confirmed facts until reviewed.")
    if not review and promote:
        guidance.append("Source-arbitrated transcript is safe as the default corrected transcript input.")
    review_refs = _review_segment_refs(review)
    trusted_indexes = [row.get("index") for row in segments if not row.get("needs_human_review")][:120]
    policy = _summary_input_policy(status=status, review_count=len(review), total=total, promote=promote)
    return {
        "status": status,
        "summary_input_policy": policy,
        "review_required": bool(review),
        "safe_segment_count": max(0, total - len(review)),
        "trusted_segment_indexes": trusted_indexes,
        "review_segment_refs": review_refs,
        "total_segments": total,
        "changed_segments": len(changed),
        "review_segments": len(review),
        "source_count": len(sources),
        "source_type_counts": source_type_counts,
        "chosen_source_type_counts": chosen_source_type_counts,
        "arbitration_reason_counts": arbitration_reason_counts,
        "review_reason_counts": review_reason_counts,
        "average_confidence": average_confidence,
        "min_confidence": min_confidence,
        "high_confidence_term_replacements": high_confidence_term_replacements,
        "low_confidence_conflicts": low_confidence_conflicts,
        "promoted_to_corrected_transcript": bool(promote),
        "can_use_as_summary_input": bool(total > 0 and not review),
        "changed_segment_indexes": [row.get("index") for row in changed[:80]],
        "review_segment_indexes": [row.get("index") for row in review[:80]],
        "smart_summary_guidance": guidance,
    }


def _review_segment_refs(review_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for row in review_rows[:80]:
        if not isinstance(row, dict):
            continue
        refs.append(
            {
                "index": row.get("index"),
                "start": row.get("start"),
                "end": row.get("end"),
                "time_range": f"{format_timestamp(_seconds(row.get('start')))} - {format_timestamp(_seconds(row.get('end')))}",
                "reason": row.get("review_reason") or "low_confidence_conflict",
                "confidence": row.get("confidence"),
                "chosen_source": row.get("chosen_source"),
                "original_text": row.get("original_text") or "",
                "corrected_text": row.get("corrected_text") or "",
            }
        )
    return refs


def _summary_input_policy(*, status: str, review_count: int, total: int, promote: bool) -> dict[str, Any]:
    if total <= 0:
        return {
            "mode": "missing_transcript",
            "can_use_corrected_transcript": False,
            "must_exclude_review_segments": True,
            "guidance": "No transcript segments are available; smart-summary needs ASR, subtitles, or manual transcript input.",
        }
    if review_count > 0:
        return {
            "mode": "partial_with_review_gaps",
            "can_use_corrected_transcript": False,
            "must_exclude_review_segments": True,
            "guidance": "Use source-arbitrated text only outside review_segment_refs; disputed segments need transcript review before factual use.",
        }
    if status == "changed_clean":
        return {
            "mode": "corrected_clean",
            "can_use_corrected_transcript": bool(promote),
            "must_exclude_review_segments": False,
            "guidance": "High-confidence corrected transcript can be used as the preferred smart-summary input.",
        }
    return {
        "mode": "clean",
        "can_use_corrected_transcript": bool(promote),
        "must_exclude_review_segments": False,
        "guidance": "Transcript sources are clean enough for smart-summary input.",
    }

def _candidate_texts(
    cue: TranscriptCue,
    base: TranscriptSource,
    sources: list[TranscriptSource],
    *,
    term_item: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        overlapping = _overlapping_cues(cue, source.cues)
        if not overlapping and source is not base:
            continue
        if source is base:
            text = cue.text
            start = cue.start
            end = cue.end
            overlap = 1.0
        else:
            text = " ".join(row.text for row in overlapping).strip()
            start = min((row.start for row in overlapping), default=cue.start)
            end = max((row.end for row in overlapping), default=cue.end)
            overlap = _overlap_ratio(cue.start, cue.end, start, end)
        if not text:
            continue
        candidate_item = {"term_candidates": term_item.get("term_candidates") or []}
        corrected = apply_high_confidence_term_replacements(text, candidate_item)
        replacements = high_confidence_term_replacements(candidate_item)
        similarity = _similarity(cue.text, corrected)
        if source is not base and similarity < 0.18:
            continue
        score = source.weight + overlap + _punctuation_bonus(corrected) + (0.8 if corrected != text else 0.0)
        rows.append(
            {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "path": source.path,
                "text": corrected,
                "raw_text": text,
                "start": start,
                "end": end,
                "overlap": round(overlap, 3),
                "similarity_to_base": round(similarity, 3),
                "score": round(score, 3),
                "term_replacements": replacements if corrected != text else [],
            }
        )
    rows.sort(key=lambda row: (-float(row.get("score") or 0), str(row.get("source_id") or "")))
    return rows


def _choose_candidate(base_text: str, candidates: list[dict[str, Any]], *, min_confidence: float) -> dict[str, Any]:
    if not candidates:
        return {
            "text": base_text,
            "source_id": "base",
            "source_type": "base",
            "confidence": 1.0,
            "reasons": ["no_alternatives"],
            "term_replacements": [],
        }
    base_candidate = next((row for row in candidates if row.get("raw_text") == base_text), candidates[-1])
    top = candidates[0]
    base_score = float(base_candidate.get("score") or 0.0)
    top_score = float(top.get("score") or 0.0)
    top_similarity = float(top.get("similarity_to_base") or 0.0)
    reasons: list[str] = []
    selected = base_candidate
    if top.get("term_replacements") and top_similarity >= 0.45:
        selected = top
        reasons.append("high_confidence_term_replacement")
    elif top is not base_candidate and top_score >= base_score + 1.0 and top_similarity >= 0.55:
        selected = top
        reasons.append("higher_weight_overlapping_subtitle")
    else:
        reasons.append("kept_base_asr")
    confidence = _confidence(selected, base_candidate, candidates)
    conflict = _has_close_conflict(selected, candidates)
    selected = {**selected}
    selected["confidence"] = round(confidence, 3)
    selected["needs_human_review"] = bool(conflict or confidence < min_confidence)
    selected["review_reason"] = "close_source_conflict" if conflict else ("low_arbitration_confidence" if confidence < min_confidence else "")
    selected["reasons"] = reasons
    return selected


def _confidence(selected: dict[str, Any], base_candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> float:
    selected_score = float(selected.get("score") or 0.0)
    base_score = float(base_candidate.get("score") or 0.0)
    top_score = max(float(row.get("score") or 0.0) for row in candidates) if candidates else selected_score
    ratio = selected_score / max(1.0, top_score)
    if selected is base_candidate:
        ratio = max(ratio, min(0.92, base_score / max(1.0, top_score)))
    return min(0.98, max(0.5, 0.55 + ratio * 0.35 + min(0.08, selected_score / 100)))


def _has_close_conflict(selected: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    selected_norm = _normalise_text(selected.get("text"))
    selected_score = float(selected.get("score") or 0.0)
    for row in candidates:
        if row is selected:
            continue
        if _normalise_text(row.get("text")) == selected_norm:
            continue
        if float(row.get("score") or 0.0) >= selected_score - 0.5 and float(row.get("similarity_to_base") or 0.0) >= 0.45:
            return True
    return False


def _overlapping_cues(cue: TranscriptCue, cues: list[TranscriptCue]) -> list[TranscriptCue]:
    midpoint = (cue.start + cue.end) / 2
    rows = []
    for other in cues:
        if other.end > cue.start and other.start < cue.end:
            rows.append(other)
        elif other.start <= midpoint <= other.end:
            rows.append(other)
    return rows


def _overlap_ratio(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    left_end = max(left_end, left_start + 0.01)
    right_end = max(right_end, right_start + 0.01)
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return overlap / max(0.01, left_end - left_start)


def _load_term_item(root: Path, *, glossary_json: str | Path | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    term_path = root / "term-resolution.json"
    if term_path.exists():
        try:
            data = read_json(term_path)
        except Exception:
            data = {}
        if isinstance(data, dict):
            for term in data.get("terms") or []:
                if not isinstance(term, dict):
                    continue
                rows.append(
                    {
                        "canonical_term": term.get("canonical_term"),
                        "raw_mentions": term.get("raw_mentions") or [],
                        "confidence": term.get("confidence"),
                    }
                )
    glossary_paths = _term_glossary_paths(root, glossary_json=glossary_json)
    for path in glossary_paths:
        rows.extend(_glossary_rows(path))
    return {"term_candidates": _dedupe_term_rows(rows), "glossary_paths": [str(path) for path in glossary_paths]}


def _term_glossary_paths(root: Path, *, glossary_json: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    if glossary_json:
        paths.append(Path(glossary_json).expanduser().resolve())
    entity_lexicon_path = root / "entity-lexicon.json"
    if entity_lexicon_path.exists():
        paths.append(entity_lexicon_path)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            value = str(manifest.get("term_arbitration_glossary_json") or "").strip()
            if value:
                candidate = Path(value)
                paths.append(candidate if candidate.is_absolute() else root / candidate)
    default_path = root / "term-arbitration-glossary.json"
    if default_path.exists():
        paths.append(default_path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _dedupe_term_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        canonical = _text(row.get("canonical_term"))
        raw_mentions = [str(value or "").strip() for value in row.get("raw_mentions") or [] if str(value or "").strip()]
        key = (canonical.lower(), "\u241f".join(sorted(value.lower() for value in raw_mentions)))
        if not canonical or not raw_mentions or key in seen:
            continue
        seen.add(key)
        out.append({**row, "canonical_term": canonical, "raw_mentions": raw_mentions})
    return out

def _glossary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = read_json(path)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        terms = data.get("terms") if isinstance(data.get("terms"), list) else []
        for item in terms:
            if not isinstance(item, dict):
                continue
            if bool(item.get("review_required")):
                continue
            canonical = _text(
                item.get("canonical")
                or item.get("canonical_term")
                or item.get("term")
                or item.get("name")
            )
            aliases = [
                str(alias)
                for alias in (item.get("aliases") or item.get("raw_mentions") or [])
                if str(alias).strip()
            ]
            try:
                confidence = float(item.get("confidence") if item.get("confidence") is not None else 1.0)
            except (TypeError, ValueError):
                confidence = 1.0
            if canonical and aliases:
                rows.append({"canonical_term": canonical, "raw_mentions": aliases, "confidence": confidence})
        if terms:
            return rows
        for key, value in data.items():
            if key == "terms":
                continue
            if isinstance(value, str):
                rows.append({"canonical_term": value, "raw_mentions": [key], "confidence": 1.0})
            elif isinstance(value, list):
                rows.append({"canonical_term": key, "raw_mentions": value, "confidence": 1.0})
    return rows


def _render_report(result: dict[str, Any], segments: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    quality = result.get("quality_summary") if isinstance(result.get("quality_summary"), dict) else {}
    lines = [
        "# Transcript Source Arbitration",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Sources: `{result.get('source_count', 0)}`",
        f"- Segments: `{summary.get('segments', 0)}`",
        f"- Changed segments: `{summary.get('changed_segments', 0)}`",
        f"- Review segments: `{summary.get('review_segments', 0)}`",
        "",
        "## Arbitration Quality",
        "",
        f"- Quality status: `{quality.get('status', summary.get('quality_status', ''))}`",
        f"- Average confidence: `{quality.get('average_confidence', summary.get('average_confidence', 0))}`",
        f"- High-confidence term replacements: `{quality.get('high_confidence_term_replacements', summary.get('high_confidence_term_replacements', 0))}`",
        f"- Low-confidence conflicts: `{quality.get('low_confidence_conflicts', summary.get('low_confidence_conflicts', 0))}`",
        f"- Can use as smart-summary input: `{quality.get('can_use_as_summary_input', False)}`",
        f"- Summary input mode: `{(quality.get('summary_input_policy') or {}).get('mode', '')}`",
        f"- Review reason counts: `{quality.get('review_reason_counts') or {}}`",
        "",
        "## Sources",
        "",
        "| Source | Type | Cues | Weight | Path |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for source in result.get("sources") or []:
        lines.append(f"| `{source.get('source_id')}` | `{source.get('source_type')}` | {source.get('cue_count')} | {source.get('weight')} | `{source.get('path')}` |")
    lines.extend(["", "## Changed Segments", "", "| # | Time | Confidence | Source | Original | Corrected |", "| ---: | --- | ---: | --- | --- | --- |"])
    changed = [row for row in segments if row.get("changed")]
    if changed:
        for row in changed[:80]:
            lines.append(
                f"| {row.get('index')} | `{format_timestamp(_seconds(row.get('start')))}` | {row.get('confidence')} | `{row.get('chosen_source')}` | {_md(row.get('original_text'))} | {_md(row.get('corrected_text'))} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")
    lines.extend(["", "## Review Needed", "", "| # | Time | Reason | Chosen | Alternatives |", "| ---: | --- | --- | --- | --- |"])
    if review_rows:
        for row in review_rows[:100]:
            alternatives = "; ".join(f"{alt.get('source_id')}:{alt.get('text')}" for alt in (row.get("alternatives") or [])[:3])
            lines.append(f"| {row.get('index')} | `{format_timestamp(_seconds(row.get('start')))}` | `{row.get('review_reason')}` | {_md(row.get('corrected_text'))} | {_md(alternatives)} |")
    else:
        lines.append("| - | - | - | - | - |")
    return "\n".join(lines).rstrip() + "\n"


def _render_corrected_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    quality = payload.get("quality_summary") if isinstance(payload.get("quality_summary"), dict) else {}
    lines = [
        "# Source-Arbitrated Transcript",
        "",
        f"- Source: `{payload.get('source')}`",
        f"- Segments: `{summary.get('segments', 0)}`",
        f"- Changed segments: `{summary.get('changed_segments', 0)}`",
        f"- Review segments: `{summary.get('review_segments', 0)}`",
        f"- Quality status: `{quality.get('status', summary.get('quality_status', ''))}`",
        f"- Average confidence: `{quality.get('average_confidence', summary.get('average_confidence', 0))}`",
        f"- Smart-summary input safe: `{quality.get('can_use_as_summary_input', False)}`",
        "",
    ]
    for segment in payload.get("segments") or []:
        badge = " changed" if segment.get("changed") else ""
        review = " review" if segment.get("needs_human_review") else ""
        lines.extend([f"## {segment.get('index')}. {format_timestamp(_seconds(segment.get('start')))}{badge}{review}", "", str(segment.get("corrected_text") or segment.get("text") or "").strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for position, segment in enumerate(segments, start=1):
        text = _text(segment.get("corrected_text") or segment.get("text"))
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(position),
                    f"{format_timestamp(_seconds(segment.get('start'))).replace('.', ',')} --> {format_timestamp(_seconds(segment.get('end'))).replace('.', ',')}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks).rstrip() + ("\n" if blocks else "")


def _source_row(source: TranscriptSource) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_type": source.source_type,
        "weight": source.weight,
        "path": source.path,
        "cue_count": len(source.cues),
    }


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _seconds(value: Any, *, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return max(0.0, float(value))
    except Exception:
        return default


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalise_text(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _similarity(left: str, right: str) -> float:
    left_norm = _normalise_text(left)
    right_norm = _normalise_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def _punctuation_bonus(value: str) -> float:
    text = str(value or "")
    return min(0.35, len(re.findall(r"[，。！？、,.!?;；:：]", text)) * 0.03)


def _md(value: Any) -> str:
    text = str(value or "-").replace("\n", " ").replace("|", "\\|")
    return text[:220] + "..." if len(text) > 220 else text
