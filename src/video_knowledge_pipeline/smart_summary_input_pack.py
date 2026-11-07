from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .content_scope import (
    apply_content_scope_to_timeline,
    content_scope_projection_summary,
    load_content_scope,
)
from .file_hash import sha256_file as _file_sha256
from .models import now_iso
from .companion_courseware_text import load_companion_courseware_text
from .courseware_material import build_courseware_material, render_courseware_material
from .page_metadata import load_page_metadata, page_metadata_context
from .local_media_progress import LocalMediaProgress, ProgressCallback
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .storage import read_json_object_or_empty as _read_optional_mapping
from .term_impact_gate import load_term_correction_impact_gate
from .term_text import apply_high_confidence_term_replacements, high_confidence_term_replacements, is_high_confidence_term_candidate
from .transcript import format_timestamp, parse_transcript
from .transcript_projection import project_transcript_cues
from .transcript_sidecar import ensure_review_transcript_sidecar
from .video_moment_index import build_video_moment_index
from .visual_evidence import continuous_tag_evidence_state, temporal_evidence_state, visual_evidence_state
from .visual_evidence import understanding_source_selector, visual_text_source_selector, visual_recognition_provenance
from .evidence_locator import JsonEvidenceSnapshot, TextEvidenceSnapshot

SCHEMA = "video_knowledge_pipeline.smart_summary_input_pack.v1"

BOUNDARY_CHARS = "。！？!?；;"
SOFT_BOUNDARY_TERMS = (
    "所以", "但是", "然后", "接下来", "首先", "第二", "第三", "最后", "比如", "也就是说", "换句话说", "这里", "注意",
)


def build_smart_summary_input_pack(
    bundle_dir: str | Path,
    *,
    title: str = "",
    write: bool = True,
    max_visual_items: int = 80,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build a cleaner local input layer for Codex smart-summary rewriting.

    The pack keeps evidence local and deterministic: corrected transcript text,
    punctuation/paragraph hints, term arbitration, and visual courseware digest.
    It is not an online model call and does not mutate raw ASR evidence.
    """

    root = Path(bundle_dir).expanduser().resolve()
    progress = (
        LocalMediaProgress(
            pipeline="local_smart_summary_input",
            snapshot_path=root / "exports" / "smart-summary-input-progress.json",
            events_path=root / "exports" / "smart-summary-input-progress.jsonl",
            callback=progress_callback,
        )
        if write
        else None
    )
    if progress:
        progress.emit(stage="load", percent=0, message="Loading local transcript and timeline evidence")
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_snapshot = JsonEvidenceSnapshot.read(root, 'timeline.json')
    timeline = timeline_snapshot.data
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be a JSON array")
    rows = [item for item in timeline if isinstance(item, dict)]
    from .media_applicability import project_timeline_for_media, resolve_media_applicability
    applicability = resolve_media_applicability(root, manifest, rows)
    positions = [pos for pos, item in enumerate(timeline) if isinstance(item, dict)]
    rows = project_timeline_for_media(rows, applicability)
    source_positions = {id(item): pos for item, pos in zip(rows, positions)}
    note_title = title or str(manifest.get("title") or root.name)
    sidecar = ensure_review_transcript_sidecar(root, manifest, rows, title=note_title, write=write)
    transcript_path = _transcript_path(root, manifest, sidecar)
    transcript_snapshot = None
    if transcript_path and transcript_path.suffix.lower() in {'.json', '.srt', '.vtt'}:
        # Strict source decoding happens outside the legacy parse/fallback catch.
        # Malformed, out-of-Bundle or ambiguous JSON must not become fresh proof.
        snapshot_type = JsonEvidenceSnapshot if transcript_path.suffix.lower() == '.json' else TextEvidenceSnapshot
        transcript_snapshot = snapshot_type.read(root, transcript_path.relative_to(root).as_posix())
    cues = []
    transcript_error = ""
    if transcript_path:
        try:
            cues = parse_transcript(transcript_path, source_snapshot=transcript_snapshot)
        except Exception as exc:  # pragma: no cover - defensive for malformed external ASR.
            if transcript_snapshot is not None:
                raise ValueError('source_locator transcript parse failed') from exc
            transcript_error = str(exc)
            cues = []
    if not cues:
        cues = _timeline_cues(rows, snapshot=timeline_snapshot, source_positions=source_positions)
        transcript_path = None
    _attach_transcript_source_indexes(cues)
    content_scope = load_content_scope(root, transcript_path=transcript_path)
    summary_cues, cue_scope_report = project_transcript_cues(
        cues,
        content_scope,
        projection="smart_summary",
    )
    summary_rows, timeline_scope_report = apply_content_scope_to_timeline(
        rows,
        content_scope,
        projection="smart_summary",
    )
    content_scope_summary = content_scope_projection_summary(
        content_scope,
        projection="smart_summary",
        cue_report=cue_scope_report,
        timeline_report=timeline_scope_report,
    )
    transcript_projection_review_gaps = [
        dict(event)
        for event in content_scope_summary.get("projection_audit") or []
        if isinstance(event, dict) and event.get("disposition") == "review_gap"
    ]
    if progress:
        progress.emit(
            stage="transcript",
            percent=30,
            current_item=len(summary_cues),
            total_items=len(cues),
            message="Preserving in-scope transcript order while retaining excluded evidence in the full transcript",
        )
    transcript_source_label = _transcript_source_label(transcript_path, manifest)
    transcript_source_decision = _transcript_source_decision(root, manifest, transcript_path, transcript_source_label)
    term_resolution = _load_term_resolution(root)
    term_summary = _term_summary(term_resolution, summary_rows)
    term_arbitration_codex = _term_arbitration_codex_summary(root, manifest)
    transcript_segments = _transcript_segments(summary_cues, summary_rows)
    semantic_correction_trace = _semantic_correction_trace(
        root,
        manifest,
        transcript_segments,
    )
    transcript_arbitration = _transcript_arbitration_quality(root, manifest)
    transcript_quality_policy = _transcript_quality_policy(transcript_arbitration)
    transcript_quality_gate = _transcript_quality_gate_summary(root, manifest)
    term_correction_impact = load_term_correction_impact_gate(root, manifest)
    transcript_semantic = _transcript_semantic_correction_summary(root, manifest)
    visual_digest = _visual_digest(summary_rows, max_items=max_visual_items)
    visual_digest["status"] = applicability["visual_status"]
    moment_index = _load_or_build_moment_index(root)
    companion_courseware = _companion_courseware_summary(root, manifest)
    courseware_material = build_courseware_material(root, write=write)
    evidence_trace = _evidence_trace(
        summary_rows,
        transcript_segments,
        moment_index,
        transcript_path=transcript_path,
        manifest=manifest,
        semantic_correction_trace=semantic_correction_trace,
        snapshot=timeline_snapshot,
        source_positions=source_positions,
    )
    source_context = _source_context_summary(root, manifest)
    quality_notes = _quality_notes(transcript_segments, term_summary, term_arbitration_codex, visual_digest, transcript_arbitration, transcript_quality_gate, term_correction_impact, transcript_semantic)
    if content_scope_summary.get("active"):
        quality_notes.append(
            "Operator-confirmed content scope excludes warm-up/non-body intervals from Smart Summary; "
            "the same source cues remain available in the full transcript and intermediate evidence."
        )
    if transcript_projection_review_gaps:
        quality_notes.append(
            "Transcript projection contains review gaps. Do not turn those spans into facts or infer missing words; "
            "keep them in the final review-gap section."
        )
    if source_context.get("available"):
        quality_notes.append("Webpage/source metadata is untrusted weak context: use it for topic, entity, and chapter hints only; never let it override transcript or visual evidence.")
    if progress:
        progress.emit(
            stage="evidence",
            percent=70,
            current_item=len(transcript_segments),
            total_items=len(transcript_segments),
            message="Assembled local summary evidence without merging transcript segments",
        )
    result = {
        "schema": SCHEMA,
        "media_applicability": applicability,
        "bundle_dir": str(root),
        "title": note_title,
        "created_at": now_iso(),
        "transcript_source": str(transcript_path or "timeline_fallback"),
        "transcript_source_sha256": _file_sha256(transcript_path) if transcript_path else "",
        "transcript_source_label": transcript_source_label,
        "transcript_source_decision": transcript_source_decision,
        "transcript_error": transcript_error,
        "transcript_segments": transcript_segments,
        "segment_policy": "preserve",
        "content_scope": content_scope_summary,
        "transcript_projection_review_gaps": transcript_projection_review_gaps,
        "segment_transformations": [
            transformation
            for segment in transcript_segments
            for transformation in segment.get("transformations") or []
            if isinstance(transformation, dict)
        ],
        "term_summary": term_summary,
        "term_arbitration_codex": term_arbitration_codex,
        "transcript_arbitration": transcript_arbitration,
        "transcript_quality_policy": transcript_quality_policy,
        "transcript_quality_gate": transcript_quality_gate,
        "term_correction_impact_gate": term_correction_impact,
        "transcript_semantic_correction": transcript_semantic,
        "source_context": source_context,
        "visual_digest": visual_digest,
        "evidence_trace": evidence_trace,
        "quality_notes": quality_notes,
        "companion_courseware": companion_courseware,
        "courseware_material": courseware_material,
        "artifacts": {
            "json": str(root / "exports" / "smart-summary-input-pack.json"),
            "markdown": str(root / "exports" / "smart-summary-input-pack.md"),
        },
        "write": bool(write),
    }
    # Bind to the observations actually parsed, not hashes reread after a race.
    for snapshot in (timeline_snapshot, transcript_snapshot):
        if snapshot is not None and _file_sha256(root / snapshot.artifact_path) != snapshot.artifact_sha256:
            raise ValueError('source_locator identity drift during summary input build')
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "smart-summary-input-pack.json", result)
        (exports / "smart-summary-input-pack.md").write_text(_render_pack_markdown(result), encoding="utf-8")
        manifest["smart_summary_input_pack"] = "exports/smart-summary-input-pack.json"
        manifest["smart_summary_input_pack_markdown"] = "exports/smart-summary-input-pack.md"
        manifest["mcp_build_smart_summary_input_pack_args"] = "mcp-build-smart-summary-input-pack.args.json"
        write_json(
            root / "mcp-build-smart-summary-input-pack.args.json",
            {"bundle_dir": str(root), "title": note_title, "write": True, "max_visual_items": max_visual_items},
        )
        write_json(manifest_path, manifest)
        result["run_registry"] = _register_run(root, result, write=write)
        if progress:
            review_gaps = (result.get("evidence_trace") or {}).get("review_gaps") or []
            terminal = "failed" if not transcript_segments else ("degraded" if review_gaps else "completed")
            progress.emit(
                stage="finalize",
                percent=100,
                current_item=len(transcript_segments),
                total_items=len(transcript_segments),
                message=(
                    "Smart summary input pack has no transcript segments"
                    if terminal == "failed"
                    else "Smart summary input pack completed with review gaps"
                    if terminal == "degraded"
                    else "Smart summary input pack completed"
                ),
                status=terminal,
                output_paths=[exports / "smart-summary-input-pack.json"],
                report_paths=[exports / "smart-summary-input-pack.md"],
                details={
                    "segment_policy": "preserve",
                    "segment_count": len(transcript_segments),
                    "content_scope_excluded_cues": content_scope_summary.get("excluded_transcript_cue_count", 0),
                    "review_gap_count": len(review_gaps),
                },
            )
            result["progress"] = progress.artifacts()
            write_json(exports / "smart-summary-input-pack.json", result)
    return result



def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    trace = result.get("evidence_trace") if isinstance(result.get("evidence_trace"), dict) else {}
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    segment_count = int(summary.get("transcript_segment_count") or len(result.get("transcript_segments") or []))
    review_gaps = trace.get("review_gaps") if isinstance(trace.get("review_gaps"), list) else []
    failed_items: list[dict[str, Any]] = []
    if segment_count <= 0:
        failed_items.append({"id": "transcript", "reason": "missing_transcript", "detail": "Smart summary input pack has no transcript segments."})
    for gap in review_gaps[:80]:
        if not isinstance(gap, dict):
            continue
        failed_items.append(
            {
                "index": gap.get("timeline_index"),
                "reason": "review_gap",
                "detail": ",".join(str(item) for item in gap.get("reasons") or []),
            }
        )
    if segment_count <= 0:
        status = "needs_input"
    elif review_gaps:
        status = "needs_review"
    else:
        status = "completed"
    return register_bundle_run(
        root,
        run_type="smart_summary_input_pack",
        run_id="smart-summary-input-pack",
        status=status,
        title="Smart summary input pack",
        summary=(
            f"Prepared {segment_count} transcript segments; "
            f"OCR/ebook={summary.get('ocr_or_ebook_items') or 0}, "
            f"visual={summary.get('visual_understanding_items') or 0}, "
            f"temporal={summary.get('temporal_understanding_items') or 0}, "
            f"review_gaps={summary.get('review_gaps') or 0}."
        ),
        inputs={"transcript_source": result.get("transcript_source"), "bundle_dir": str(root)},
        parameters={"max_visual_items": len((result.get("visual_digest") or {}).get("items") or []) if isinstance(result.get("visual_digest"), dict) else 0},
        artifacts=[
            {"key": "input_pack_json", "path": str(root / "exports" / "smart-summary-input-pack.json")},
            {"key": "input_pack_markdown", "path": str(root / "exports" / "smart-summary-input-pack.md")},
            {"key": "mcp_args", "path": str(root / "mcp-build-smart-summary-input-pack.args.json")},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 build-smart-summary-input-pack {_ps_quote(str(root))}",
        next_actions=_run_next_actions(status),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "does_not_modify_raw_transcript": True,
            "purpose": "Prepare evidence-traced input for Codex/LLM smart-summary generation.",
        },
        write=write,
    )


def _run_next_actions(status: str) -> list[str]:
    if status == "needs_input":
        return ["Run local ASR or import a reviewed transcript before treating smart-summary output as complete."]
    if status == "needs_review":
        return ["Review listed evidence gaps, then rerun build-smart-summary-input-pack before final summary generation."]
    return ["Run build-smart-summary-chapters or generate-smart-summary-with-codex to refresh summary outputs."]


def select_canonical_transcript_path(
    bundle_dir: str | Path,
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    """Expose the established summary-input transcript precedence.

    Intent: keep read-only derived projections on the same canonical transcript
    selected by Smart Summary. Decision: delegate to the existing selector
    instead of copying its precedence table. Effective scope: path selection
    only; no transcript is changed or created.
    """

    root = Path(bundle_dir).expanduser().resolve()
    current = manifest
    if current is None:
        value = read_json(root / "manifest.json")
        current = value if isinstance(value, dict) else {}
    return _transcript_path(root, current, {})


def _transcript_path(root: Path, manifest: dict[str, Any], sidecar: dict[str, Any]) -> Path | None:
    candidates: list[str] = []
    for key in (
        "source_arbitrated_transcript_json",
        "human_corrected_transcript_json",
        "llm_readable_transcript_json",
        "agent_readable_transcript_json",
        "readable_transcript_json",
        "llm_corrected_transcript_json",
        "corrected_transcript_json",
    ):
        value = manifest.get(key)
        if value:
            candidates.append(str(value))
    candidates.extend([
        "source-arbitrated-transcript.json",
        "human-corrected-transcript.json",
        "llm-readable-transcript.json",
        "agent-readable-transcript.json",
        "readable-transcript.json",
        "llm-corrected-transcript.json",
        "corrected-transcript.json",
    ])
    for key in ("json_path", "path", "source_path"):
        value = sidecar.get(key)
        if value:
            candidates.append(str(value))
    for key in (
        "normalized_transcript_json",
        "source_transcript",
        "transcript_json",
        "transcript_path",
    ):
        value = manifest.get(key)
        if value:
            candidates.append(str(value))
    candidates.extend(["normalized-transcript.json", "transcript.json"])
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.is_file():
            return path.resolve()
    return None

def _timeline_cues(timeline: list[dict[str, Any]], *, snapshot=None, source_positions=None) -> list[Any]:
    cues = []
    for item in timeline:
        text = str(item.get("corrected_transcript") or item.get("transcript") or item.get("text") or "").strip()
        if not text:
            continue
        segment_id = str(item.get("segment_id") or item.get("id") or f"timeline-{len(cues) + 1:06d}")
        cues.append(
            type(
                "SummaryInputCue",
                (),
                {
                    "start": _seconds(item.get("start")),
                    "end": _seconds(item.get("end")),
                    "text": text,
                    "segment_id": segment_id,
                    "source_segment_ids": list(item.get("source_segment_ids") or [segment_id]),
                    "transformations": list(item.get("transformations") or []),
                    "metadata": {
                        'source_locators': [snapshot.locator([source_positions[id(item)], next(
                            key for key in ('corrected_transcript', 'transcript', 'text') if item.get(key))])]
                        if snapshot is not None else [],
                        'source_observation_text': text,
                        'source_locator_scope': 'timeline_transcript_not_original_audio',
                    },
                },
            )()
        )
    return cues


def _transcript_segments(cues: list[Any], timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The matching function uses time overlap only. Numeric labels are not keys.
    by_index = dict(enumerate(timeline))
    segments: list[dict[str, Any]] = []
    for pos, cue in enumerate(cues, start=1):
        start = _seconds(getattr(cue, "start", 0.0))
        end = _seconds(getattr(cue, "end", start))
        raw = str(getattr(cue, "text", "") or "").strip()
        if not raw:
            continue
        item = _matching_timeline_item(by_index, start, end) or {}
        corrected = str(item.get("corrected_transcript") or "").strip() or apply_high_confidence_term_replacements(raw, item)
        punctuated = _light_punctuate(corrected)
        segment_id = str(getattr(cue, "segment_id", "") or f"segment-{pos:06d}")
        source_ids = list(getattr(cue, "source_segment_ids", []) or [segment_id])
        projection_metadata = (
            dict(getattr(cue, "metadata", {}) or {})
            if isinstance(getattr(cue, "metadata", {}), dict)
            else {}
        )
        source_segment_indexes = _source_segment_indexes(projection_metadata)
        transformations = [
            dict(value)
            for value in (getattr(cue, "transformations", []) or [])
            if isinstance(value, dict)
        ]
        if punctuated != corrected:
            transformations.append(
                {
                    "type": "summary_text_hint",
                    "source_segment_ids": source_ids,
                    "boundary_changed": False,
                }
            )
        segments.append(
            {
                "index": pos,
                "segment_id": segment_id,
                "source_segment_ids": source_ids,
                "source_segment_index": (
                    source_segment_indexes[0] if len(source_segment_indexes) == 1 else None
                ),
                "source_segment_indexes": source_segment_indexes,
                "timeline_index": item.get("index") or "",
                "start": start,
                "end": end,
                "start_time": format_timestamp(start),
                "end_time": format_timestamp(end),
                "raw_text": raw,
                "source_locators": projection_metadata.get('source_locators') or [],
                "source_observation_text": projection_metadata.get('source_observation_text', ''),
                "source_locator_scope": projection_metadata.get('source_locator_scope') or 'unlocated_projection',
                "corrected_text": corrected,
                "punctuated_text": punctuated,
                "transformations": transformations,
                "projection_status": str(
                    projection_metadata.get("projection_disposition") or "retained_semantic"
                ),
                "projection_reasons": [
                    str(value)
                    for value in (projection_metadata.get("projection_reasons") or [])
                    if str(value)
                ],
                "term_replacements": high_confidence_term_replacements(item),
                "visual_route": item.get("visual_route") or "",
                "visual_digest_ref": _item_index(item, 0) if item else "",
                "evidence_inputs": _timeline_evidence_inputs(item) if item else {},
            }
        )
    return segments


def _light_punctuate(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return ""
    if any(char in value for char in BOUNDARY_CHARS):
        return value
    # Heuristic only: keep raw text intact enough for evidence, but make long ASR
    # chunks readable for Codex/human review.
    for term in SOFT_BOUNDARY_TERMS:
        value = value.replace(term, f"。{term}")
    value = re.sub(r"^。+", "", value)
    if len(value) > 90:
        value = _chunk_chinese_like(value, max_chars=90)
    if value and value[-1] not in BOUNDARY_CHARS:
        value += "。"
    return value


def _chunk_chinese_like(text: str, *, max_chars: int) -> str:
    parts: list[str] = []
    buf = ""
    for token in re.split(r"(。)", text):
        if not token:
            continue
        buf += token
        if len(buf) >= max_chars or token == "。":
            parts.append(buf.strip("。 "))
            buf = ""
    if buf.strip():
        parts.append(buf.strip("。 "))
    return "。".join(part for part in parts if part) + "。"


def _load_term_resolution(root: Path) -> dict[str, Any]:
    path = root / "term-resolution.json"
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _transcript_quality_gate_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    candidates: list[str] = []
    value = manifest.get("transcript_quality_gate_json")
    if value:
        candidates.append(str(value))
    candidates.append("transcript-quality-gate.json")
    for value in candidates:
        path = _bundle_path(root, value)
        if not path.exists() or not path.is_file():
            continue
        data = _read_optional_mapping(path)
        if data:
            return {
                "exists": True,
                "path": str(path.resolve()),
                "status": str(data.get("status") or "unknown"),
                "ok": bool(data.get("ok")),
                "fail_count": _safe_int(data.get("fail_count")),
                "warning_count": _safe_int(data.get("warning_count")),
                "punctuation_density_per_1000_chars": data.get("punctuation_density_per_1000_chars", ""),
                "source_path": str(data.get("source_path") or ""),
                "next_actions": data.get("next_actions") if isinstance(data.get("next_actions"), list) else [],
            }
    summary = manifest.get("transcript_quality_gate_summary") if isinstance(manifest.get("transcript_quality_gate_summary"), dict) else {}
    if summary:
        return {
            "exists": True,
            "path": "manifest",
            "status": str(summary.get("status") or "unknown"),
            "ok": bool(summary.get("ok")),
            "fail_count": _safe_int(summary.get("fail_count")),
            "warning_count": _safe_int(summary.get("warning_count")),
            "punctuation_density_per_1000_chars": summary.get("punctuation_density_per_1000_chars", ""),
            "source_path": str(summary.get("source_path") or ""),
            "next_actions": summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else [],
        }
    return {"exists": False, "path": "", "status": "missing", "ok": False, "fail_count": 0, "warning_count": 0, "next_actions": []}


def _transcript_arbitration_quality(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    candidates: list[str] = []
    for key in ("transcript_source_arbitration_json", "transcript_source_arbitration_report"):
        value = manifest.get(key)
        if value:
            candidates.append(str(value))
    candidates.append("transcript-source-arbitration.json")
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.exists() or not path.is_file():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        quality = data.get("quality_summary") if isinstance(data.get("quality_summary"), dict) else {}
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        if not quality and isinstance(summary.get("quality_summary"), dict):
            quality = summary.get("quality_summary")
        return {
            "exists": True,
            "path": str(path.resolve()),
            "status": data.get("status") or summary.get("quality_status") or quality.get("status") or "unknown",
            "quality_summary": quality,
            "summary": summary,
            "review_rows": data.get("review_rows") if isinstance(data.get("review_rows"), list) else [],
        }
    manifest_quality = manifest.get("transcript_source_arbitration_quality") if isinstance(manifest.get("transcript_source_arbitration_quality"), dict) else {}
    manifest_summary = manifest.get("transcript_source_arbitration_summary") if isinstance(manifest.get("transcript_source_arbitration_summary"), dict) else {}
    if manifest_quality or manifest_summary:
        return {
            "exists": True,
            "path": "manifest",
            "status": manifest_quality.get("status") or manifest_summary.get("quality_status") or "unknown",
            "quality_summary": manifest_quality,
            "summary": manifest_summary,
            "review_rows": [],
        }
    return {
        "exists": False,
        "path": "",
        "status": "missing",
        "quality_summary": {},
        "summary": {},
        "review_rows": [],
    }

def _transcript_semantic_correction_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    pack_path = _bundle_path(root, str(manifest.get("transcript_semantic_correction_pack_json") or "transcript-semantic-correction-pack.json"))
    validation_path = _bundle_path(root, str(manifest.get("transcript_semantic_correction_validation_json") or "transcript-semantic-correction-validation.json"))
    closure_path = _bundle_path(root, str(manifest.get("transcript_semantic_correction_closure_json") or "transcript-semantic-correction-closure.json"))
    readable_path = _bundle_path(root, str(manifest.get("transcript_semantic_readable_impact_report_json") or "transcript-semantic-readable-impact-report.json"))
    summary_path = _bundle_path(root, str(manifest.get("transcript_semantic_summary_impact_report_json") or "transcript-semantic-summary-impact-report.json"))
    status_path = _bundle_path(root, str(manifest.get("transcript_semantic_correction_status_json") or "transcript-semantic-correction-status.json"))
    corrected_path = _bundle_path(root, str(manifest.get("corrected_transcript_json") or manifest.get("source_arbitrated_transcript_json") or "corrected-transcript.json"))
    pack = _read_optional_mapping(pack_path)
    validation = _read_optional_mapping(validation_path)
    closure = _read_optional_mapping(closure_path)
    readable = _read_optional_mapping(readable_path)
    summary = _read_optional_mapping(summary_path)
    status = _read_optional_mapping(status_path)
    candidates = pack.get("candidates") if isinstance(pack.get("candidates"), list) else []
    attention = status.get("semantic_attention_items") if isinstance(status.get("semantic_attention_items"), list) else []
    accepted = validation.get("accepted_decisions") if isinstance(validation.get("accepted_decisions"), list) else []
    review_required = validation.get("review_required") if isinstance(validation.get("review_required"), list) else []
    rejected = validation.get("rejected_decisions") if isinstance(validation.get("rejected_decisions"), list) else []
    candidate_type_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for row in candidates:
        if not isinstance(row, dict):
            continue
        ctype = str(row.get("correction_type") or "unknown")
        risk = str(row.get("risk_level") or "unknown")
        candidate_type_counts[ctype] = candidate_type_counts.get(ctype, 0) + 1
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    closure_status = str(closure.get("status") or "missing") if closure else "missing"
    readable_status = str(readable.get("status") or "missing") if readable else "missing"
    summary_status = str(summary.get("status") or "missing") if summary else "missing"
    pack_status = str(pack.get("status") or "missing") if pack else "missing"
    validation_status = str(validation.get("status") or "missing") if validation else "missing"
    if not pack:
        final_status = "not_started"
    elif candidates and not validation:
        final_status = "needs_codex_or_llm_review"
    elif review_required:
        final_status = "needs_human_review"
    elif accepted and closure_status not in {"completed", "completed_no_text_changes"}:
        final_status = "needs_closure"
    elif accepted and readable_status not in {"passed", "not_required", "no_accepted_decisions", "no_evaluable_replacements"}:
        final_status = "needs_readable_export_fix"
    elif accepted and summary_status not in {"passed", "no_accepted_decisions", "no_evaluable_replacements"}:
        final_status = "needs_smart_summary_refresh"
    elif candidates:
        final_status = "ready_for_summary_input"
    else:
        final_status = "no_candidates"
    return {
        "exists": bool(pack),
        "final_status": final_status,
        "pack_path": str(pack_path),
        "validation_path": str(validation_path),
        "closure_path": str(closure_path),
        "readable_impact_path": str(readable_path),
        "summary_impact_path": str(summary_path),
        "corrected_transcript_path": str(corrected_path) if corrected_path.exists() else "",
        "pack_status": pack_status,
        "validation_status": validation_status,
        "closure_status": closure_status,
        "readable_impact_status": readable_status,
        "summary_impact_status": summary_status,
        "candidate_count": _safe_int(pack.get("candidate_count")) if pack else 0,
        "candidate_type_counts": candidate_type_counts,
        "risk_level_counts": risk_counts,
        "semantic_attention_count": len(attention),
        "accepted_decision_count": len([row for row in accepted if isinstance(row, dict)]),
        "review_required_count": len([row for row in review_required if isinstance(row, dict)]),
        "rejected_decision_count": len([row for row in rejected if isinstance(row, dict)]),
        "applied_correction_count": _safe_int(closure.get("applied_correction_count")) if closure else 0,
        "changed_segment_count": _safe_int(closure.get("changed_segment_count")) if closure else 0,
        "summary_residual_original_total": _safe_int(summary.get("summary_residual_original_total")) if summary else 0,
        "summary_absorption_rate": summary.get("summary_absorption_rate", 0) if summary else 0,
        "next_actions": _transcript_semantic_next_actions(final_status, root),
    }


def _transcript_semantic_next_actions(status: str, root: Path) -> list[str]:
    q = _ps_quote(str(root))
    if status == "not_started":
        return [f"Run .\\scripts\\video-knowledge.ps1 transcript-semantic-correction-pack {q} to discover ASR/subtitle semantic correction candidates."]
    if status == "needs_codex_or_llm_review":
        return [f"Run .\\scripts\\video-knowledge.ps1 transcript-semantic-correction-llm-draft {q} --limit 80, or use Codex to fill transcript-semantic-correction-result.codex.md."]
    if status == "needs_human_review":
        return ["Review transcript-semantic-correction-review.md, then import reviewed decisions before closure."]
    if status == "needs_closure":
        return [f"Run .\\scripts\\video-knowledge.ps1 transcript-semantic-correction-closure {q} to write source-arbitrated transcript."]
    if status == "needs_readable_export_fix":
        return [f"Run export-knowledge-note and transcript-semantic-readable-impact-report for {q}."]
    if status == "needs_smart_summary_refresh":
        return [f"Regenerate smart-summary from source-arbitrated transcript, then run transcript-semantic-summary-impact-report {q}."]
    if status == "ready_for_summary_input":
        return ["Use source-arbitrated transcript and listed semantic corrections as preferred smart-summary input."]
    return []


def _term_arbitration_codex_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    pack_path = _bundle_path(root, str(manifest.get("term_arbitration_codex_pack_json") or "term-arbitration-codex-pack.json"))
    result_path = _bundle_path(root, str(manifest.get("term_arbitration_codex_result_json") or "term-arbitration-codex-result.json"))
    glossary_path = _bundle_path(root, str(manifest.get("term_arbitration_glossary_json") or "term-arbitration-glossary.json"))
    prompt_path = _bundle_path(root, str(manifest.get("term_arbitration_codex_prompt_markdown") or "term-arbitration-codex-prompt.md"))
    report_path = _bundle_path(root, str(manifest.get("term_arbitration_codex_markdown") or "term-arbitration-codex.md"))
    pack = _read_optional_mapping(pack_path)
    result = _read_optional_mapping(result_path)
    glossary = _read_optional_mapping(glossary_path)
    pack_status = str(pack.get("status") or "") if pack else ""
    decisions = result.get("decisions") if isinstance(result.get("decisions"), list) else []
    draft_decisions = pack.get("draft_decisions") if isinstance(pack.get("draft_decisions"), list) else []
    terms = glossary.get("terms") if isinstance(glossary.get("terms"), list) else []
    semantic = pack.get("llm_semantic_arbitration") if isinstance(pack.get("llm_semantic_arbitration"), dict) else {}
    if result_path.exists() and decisions:
        status = "imported"
    elif pack_path.exists():
        status = pack_status or "draft_ready"
    elif prompt_path.exists() or report_path.exists():
        status = "artifacts_present"
    else:
        status = "missing"
    imported_decisions = [row for row in decisions if isinstance(row, dict)]
    accepted_imported = [row for row in imported_decisions if str(row.get("action") or "replace") == "replace" and not bool(row.get("needs_human_review"))]
    return {
        "exists": status != "missing",
        "status": status,
        "pack_path": str(pack_path),
        "prompt_path": str(prompt_path),
        "result_path": str(result_path),
        "glossary_path": str(glossary_path),
        "report_path": str(report_path),
        "candidate_count": _safe_int(pack.get("candidate_count")) if pack else 0,
        "draft_decision_count": len(draft_decisions),
        "imported_decision_count": len(imported_decisions),
        "accepted_decision_count": len(accepted_imported),
        "glossary_term_count": len([row for row in terms if isinstance(row, dict)]),
        "codex_review_required": status in {"draft_ready", "artifacts_present"} and not imported_decisions,
        "ready_for_transcript_arbitration": bool(terms),
        "llm_semantic_arbitration": semantic,
        "semantic_review_status": str(semantic.get("review_status") or ""),
        "semantic_strategy": str(semantic.get("strategy") or ""),
        "rule_draft_is_not_semantic_confirmation": bool(semantic.get("rule_draft_is_not_semantic_confirmation", True)),
        "next_actions": _term_arbitration_next_actions(status, root),
    }


def _term_arbitration_next_actions(status: str, root: Path) -> list[str]:
    if status == "missing":
        return [f"Run .\\scripts\\video-knowledge.ps1 term-arbitration-codex {_ps_quote(str(root))} when tool names or domain terms need semantic judgment."]
    if status in {"draft_ready", "artifacts_present"}:
        return ["Review term-arbitration-codex-prompt.md with Codex, save term-arbitration-codex-result.json, then import it."]
    if status == "imported":
        return ["Run transcript-source-arbitration with term-arbitration-glossary.json, then run term-correction-impact-report."]
    return []

def _term_summary(term_resolution: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    terms = term_resolution.get("terms") if isinstance(term_resolution.get("terms"), list) else []
    high_conf = []
    needs_review = []
    for term in terms:
        if not isinstance(term, dict):
            continue
        row = {
            "canonical_term": str(term.get("canonical_term") or ""),
            "raw_mentions": term.get("raw_mentions") if isinstance(term.get("raw_mentions"), list) else [],
            "confidence": term.get("confidence"),
            "source_counts": term.get("source_counts") if isinstance(term.get("source_counts"), dict) else {},
            "needs_human_review": bool(term.get("needs_human_review")),
        }
        if is_high_confidence_term_candidate(term):
            high_conf.append(row)
        if term.get("needs_human_review"):
            needs_review.append(row)
    timeline_replacements = []
    for item in timeline:
        for replacement in high_confidence_term_replacements(item):
            timeline_replacements.append({"timeline_index": item.get("index"), **replacement})
    return {
        "source": "term-resolution.json" if term_resolution else "timeline_term_candidates",
        "resolved_terms": len(terms),
        "high_confidence_terms": high_conf[:50],
        "needs_review_terms": needs_review[:80],
        "timeline_replacements": timeline_replacements[:200],
    }


def _visual_digest(timeline: list[dict[str, Any]], *, max_items: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for pos, item in enumerate(timeline, start=1):
        route = str(item.get("visual_route") or "unknown")
        counts[route] = counts.get(route, 0) + 1
        visual_text = _first_text(item.get("human_corrected_visual_text"), item.get("visual_text"), item.get("ocr_text"))
        structured = ("" if visual_text_source_selector(item) == ['human_corrected_visual_text']
                      else _structured_visual_text(item.get("structured_visual")))
        visual_state = visual_evidence_state(item)
        temporal_state = temporal_evidence_state(item)
        frame_understanding = _qualified_evidence_text(visual_state)
        temporal = _qualified_evidence_text(temporal_state)
        tag_state = continuous_tag_evidence_state(item)
        tag_weak = str(tag_state.get("coarse_summary") or "") if tag_state.get("consumable_weak") else ""
        if not any((visual_text, structured, frame_understanding, temporal, tag_weak)):
            continue
        rows.append(
            {
                "timeline_index": item.get("index") or pos,
                "start": _seconds(item.get("start")),
                "end": _seconds(item.get("end")),
                "start_time": format_timestamp(_seconds(item.get("start"))),
                "end_time": format_timestamp(_seconds(item.get("end"))),
                "visual_route": route,
                "visual_text": _clip(visual_text, 240),
                "visual_text_role": str(item.get("visual_text_role") or "unclassified"),
                "structured_visual": _clip(structured, 360),
                "visual_understanding": _clip(frame_understanding, 360),
                "temporal_visual_understanding": _clip(temporal, 360),
                "continuous_tag_weak_evidence": _clip(tag_weak, 360),
                "visual_evidence_level": visual_state["evidence_level"],
                "temporal_evidence_level": temporal_state["evidence_level"],
                "visual_inference_basis": visual_state.get("inference_basis") or "",
                "temporal_inference_basis": temporal_state.get("inference_basis") or "",
                "evidence_paths": _evidence_paths(item),
            }
        )
    return {"route_counts": counts, "items": rows[: max(0, int(max_items or 0))], "total_items_with_visual_digest": len(rows),
            "purpose": "browsing_preview_not_complete_material", "item_limit": max(0, int(max_items or 0)),
            "items_truncated": len(rows) > max(0, int(max_items or 0)), "text_may_be_abbreviated": True}



def _companion_courseware_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    payload = load_companion_courseware_text(root, manifest)
    if not payload:
        return {"status": "not_present"}
    text = str(payload.get("text") or "").strip()
    limit = 6000
    return {
        "status": "covered_by_external_courseware",
        "title": str(payload.get("title") or ""),
        "source_sha256": str(payload.get("source_sha256") or ""),
        "bundle_copy_path": str(payload.get("bundle_copy_path") or ""),
        "evidence_scope": "external_courseware_not_video_frame",
        "screen_text_coverage": "covered_by_external_courseware",
        "structured_visual_coverage": "covered_by_external_courseware",
        "content_excerpt": text if len(text) <= limit else text[:limit].rstrip() + "\n[VKP: companion courseware excerpt truncated]",
        "truncated": len(text) > limit,
        "timestamp_mapping": "not_available",
    }

def _load_or_build_moment_index(root: Path) -> dict[str, Any]:
    path = root / "exports" / "video-moment-index.json"
    if path.exists():
        try:
            data = read_json(path)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    try:
        data = build_video_moment_index(root, write=False)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _evidence_trace(
    timeline: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    moment_index: dict[str, Any],
    *,
    transcript_path: Path | None,
    manifest: dict[str, Any],
    semantic_correction_trace: dict[str, Any] | None = None,
    snapshot=None,
    source_positions=None,
) -> dict[str, Any]:
    indexed_rows = [(_item_index(item, pos), item) for pos, item in enumerate(timeline, start=1)]
    timeline_indexes = _unique_ints([_safe_int(row.get("timeline_index")) for row in transcript_segments])
    ocr_items: list[dict[str, Any]] = []
    visual_items: list[dict[str, Any]] = []
    tile_items: list[dict[str, Any]] = []
    temporal_items: list[dict[str, Any]] = []
    continuous_tag_items: list[dict[str, Any]] = []
    review_gaps: list[dict[str, Any]] = []
    for idx, item in indexed_rows:
        location = {'snapshot': snapshot, 'source_position': source_positions[id(item)]} if snapshot is not None else {}
        inputs = _timeline_evidence_inputs(item)
        if inputs.get("has_ocr_or_ebook"):
            ocr_items.append(_compact_evidence_item(idx, item, kind="ocr_or_ebook", **location))
            from .visual_evidence import has_complementary_structured_visual
            if has_complementary_structured_visual(item):
                ocr_items.append(_compact_evidence_item(idx, item, kind="structured_visual", **location))
        if inputs.get("has_visual_understanding"):
            visual_items.append(_compact_evidence_item(idx, item, kind="visual_understanding", **location))
        if inputs.get("has_high_res_tile"):
            tile_items.append(_compact_evidence_item(idx, item, kind="high_res_tile", **location))
        if inputs.get("has_temporal_understanding"):
            temporal_items.append(_compact_evidence_item(idx, item, kind="temporal_visual_understanding", **location))
        if inputs.get("has_continuous_tag_weak_evidence"):
            continuous_tag_items.append(_compact_evidence_item(idx, item, kind="continuous_tag_weak_evidence", **location))
        reasons = _review_gap_reasons(item)
        if reasons:
            review_gaps.append(
                {
                    "timeline_index": idx,
                    "start": _seconds(item.get("start")),
                    "end": _seconds(item.get("end")),
                    "start_time": format_timestamp(_seconds(item.get("start"))),
                    "end_time": format_timestamp(_seconds(item.get("end"))),
                    "reasons": reasons,
                    "evidence_paths": _evidence_paths(item),
                }
            )
    moment_chunks = _moment_chunks_for_indexes(moment_index, timeline_indexes or [idx for idx, _ in indexed_rows])
    from .citation_digest import check_record_budget
    check_record_budget([*ocr_items, *visual_items, *tile_items, *temporal_items,
                         *continuous_tag_items, *moment_chunks, *review_gaps])
    return {
        "transcript_source": _transcript_source_label(transcript_path, manifest),
        "transcript_path": str(transcript_path) if transcript_path else "timeline_fallback",
        "transcript_sha256": _file_sha256(transcript_path) if transcript_path else "",
        "semantic_correction": semantic_correction_trace or {
            "status": "not_started",
            "artifact_hashes": {},
            "applied_segment_count": 0,
        },
        "transcript_segment_count": len(transcript_segments),
        "timeline_indexes": timeline_indexes,
        "ocr_items": ocr_items,
        "visual_items": visual_items,
        "tile_items": tile_items,
        "temporal_items": temporal_items,
        "continuous_tag_items": continuous_tag_items,
        "moment_chunks": moment_chunks,
        "review_gaps": review_gaps,
        "summary": {
            "timeline_items": len(timeline),
            "timeline_indexes_with_transcript": len(timeline_indexes),
            "ocr_or_ebook_items": len(ocr_items),
            "high_res_tile_items": len(tile_items),
            "visual_understanding_items": len(visual_items),
            "temporal_understanding_items": len(temporal_items),
            "continuous_tag_weak_evidence_items": len(continuous_tag_items),
            "moment_chunks": len(moment_chunks),
            "review_gaps": len(review_gaps),
        },
    }


def _semantic_correction_trace(
    root: Path,
    manifest: dict[str, Any],
    transcript_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind accepted semantic decisions to summary segments without rewriting text.

    The correction pipeline remains the only writer of corrected transcript text.
    This adapter only carries its already-validated identities and artifact hashes
    into the summary/citation evidence chain.
    """

    artifact_specs = {
        "pack": str(
            manifest.get("transcript_semantic_correction_pack_json")
            or "transcript-semantic-correction-pack.json"
        ),
        "validation": str(
            manifest.get("transcript_semantic_correction_validation_json")
            or "transcript-semantic-correction-validation.json"
        ),
        "closure": str(
            manifest.get("transcript_semantic_correction_closure_json")
            or "transcript-semantic-correction-closure.json"
        ),
        "decision_ledger": str(
            manifest.get("transcript_semantic_correction_decision_ledger_json")
            or "transcript-semantic-correction-decision-ledger.json"
        ),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for key, value in artifact_specs.items():
        path = _bundle_path(root, value)
        if not path.is_file():
            continue
        artifacts[key] = {"path": str(path), "sha256": _file_sha256(path)}
        payload = _read_optional_mapping(path)
        if payload:
            payloads[key] = payload

    pack = payloads.get("pack", {})
    validation = payloads.get("validation", {})
    closure = payloads.get("closure", {})
    ledger = payloads.get("decision_ledger", {})
    candidates = {
        str(row.get("candidate_id")): row
        for row in (pack.get("candidates") or [])
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    }
    accepted_rows = [
        row
        for row in (validation.get("accepted_decisions") or [])
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    ]
    ledger_rows = ledger.get("decisions") if isinstance(ledger.get("decisions"), list) else []
    if ledger_rows:
        accepted_rows = [
            row
            for row in ledger_rows
            if isinstance(row, dict)
            and str(row.get("candidate_id") or "")
            and str(row.get("action") or "replace") in {"replace", "keep_original"}
        ]

    by_segment: dict[int, list[dict[str, Any]]] = {}
    for decision in accepted_rows:
        candidate_id = str(decision.get("candidate_id") or "")
        candidate = candidates.get(candidate_id, {})
        segment_value = (
            decision.get("segment_index")
            if decision.get("segment_index") is not None
            else candidate.get("segment_index")
        )
        segment_index = _optional_nonnegative_int(segment_value)
        if segment_index is None:
            continue
        evidence_ids = _unique_texts(
            [*(candidate.get("evidence_ids") or []), *(decision.get("evidence_ids") or [])]
        )
        by_segment.setdefault(segment_index, []).append(
            {
                "candidate_id": candidate_id,
                "decision_id": str(decision.get("decision_id") or candidate_id),
                "action": str(decision.get("action") or "replace"),
                "correction_type": str(
                    decision.get("correction_type")
                    or candidate.get("correction_type")
                    or ""
                ),
                "evidence_ids": evidence_ids,
                "human_confirmed": bool(decision.get("human_confirmed")),
            }
        )

    applied_segment_count = 0
    for segment in transcript_segments:
        source_indexes = [
            value
            for value in (
                _optional_nonnegative_int(item)
                for item in (segment.get("source_segment_indexes") or [])
            )
            if value is not None
        ]
        if not source_indexes:
            source_index = _optional_nonnegative_int(segment.get("source_segment_index"))
            if source_index is not None:
                source_indexes = [source_index]
        decisions = [
            row
            for source_index in source_indexes
            for row in by_segment.get(source_index, [])
        ]
        if not decisions:
            continue
        applied_segment_count += 1
        segment["semantic_correction_provenance"] = {
            "status": "validated",
            "candidate_ids": _unique_texts(
                [row.get("candidate_id") for row in decisions]
            ),
            "decision_ids": _unique_texts(
                [row.get("decision_id") for row in decisions]
            ),
            "correction_types": _unique_texts(
                [row.get("correction_type") for row in decisions]
            ),
            "evidence_ids": _unique_texts(
                [
                    evidence_id
                    for row in decisions
                    for evidence_id in row.get("evidence_ids") or []
                ]
            ),
            "human_confirmed": any(
                bool(row.get("human_confirmed")) for row in decisions
            ),
            "artifact_hashes": artifacts,
        }

    if not artifacts:
        status = "not_started"
    elif accepted_rows and str(closure.get("status") or "") in {
        "completed",
        "completed_no_text_changes",
    }:
        status = "closed"
    elif accepted_rows:
        status = "validated_not_closed"
    else:
        status = "candidate_or_review_pending"
    return {
        "status": status,
        "artifact_hashes": artifacts,
        "accepted_decision_count": len(accepted_rows),
        "applied_segment_count": applied_segment_count,
        "candidate_ids": _unique_texts(
            [str(row.get("candidate_id") or "") for row in accepted_rows]
        ),
    }


def _attach_transcript_source_indexes(cues: list[Any]) -> None:
    """Preserve pre-projection cue identity without mutating source files."""

    for source_index, cue in enumerate(cues):
        metadata = (
            dict(getattr(cue, "metadata", {}) or {})
            if isinstance(getattr(cue, "metadata", {}), dict)
            else {}
        )
        if (
            metadata.get("source_segment_index") is None
            and not metadata.get("source_segment_indexes")
        ):
            metadata["source_segment_index"] = source_index
        cue.metadata = metadata


def _source_segment_indexes(metadata: dict[str, Any]) -> list[int]:
    values = metadata.get("source_segment_indexes")
    if not isinstance(values, list):
        values = [metadata.get("source_segment_index")]
    indexes = [
        value
        for value in (_optional_nonnegative_int(item) for item in values)
        if value is not None
    ]
    return list(dict.fromkeys(indexes))


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _unique_texts(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _timeline_evidence_inputs(item: dict[str, Any]) -> dict[str, Any]:
    visual_text = _first_text(item.get("human_corrected_visual_text"), item.get("visual_text"), item.get("ocr_text"))
    structured = _structured_visual_text(item.get("structured_visual"))
    visual_state = visual_evidence_state(item)
    temporal_state = temporal_evidence_state(item)
    visual = _qualified_evidence_text(visual_state)
    temporal = _qualified_evidence_text(temporal_state)
    tag_state = continuous_tag_evidence_state(item)
    tile_merges = item.get("tile_result_merges") if isinstance(item.get("tile_result_merges"), list) else []
    return {
        "has_timeline_item": bool(item),
        "visual_route": item.get("visual_route") or "",
        "has_ocr_or_ebook": bool(visual_text or structured),
        "has_high_res_tile": any(isinstance(row, dict) and str(row.get("action") or "") == "merge" for row in tile_merges),
        "has_visual_understanding": bool(visual),
        "has_temporal_understanding": bool(temporal),
        "has_continuous_tag_weak_evidence": bool(tag_state.get("consumable_weak")),
        "visual_evidence_level": visual_state["evidence_level"],
        "temporal_evidence_level": temporal_state["evidence_level"],
        "has_evidence_paths": bool(_evidence_paths(item)),
        "review_gap_reasons": _review_gap_reasons(item),
    }


def _tile_merge_text(item: dict[str, Any]) -> str:
    rows = item.get("tile_result_merges") if isinstance(item.get("tile_result_merges"), list) else []
    parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("action") or "") != "merge":
            continue
        tile_id = str(row.get("tile_id") or "").strip()
        confidence = row.get("confidence")
        evidence = str(row.get("evidence_path") or "").strip()
        label = "High-res tile"
        if tile_id:
            label += f" {tile_id}"
        if confidence is not None:
            label += f" confidence={confidence}"
        if evidence:
            label += f" evidence={evidence}"
        parts.append(label)
    return " | ".join(parts)

def _compact_evidence_item(index: int, item: dict[str, Any], *, kind: str,
                           snapshot=None, source_position=None) -> dict[str, Any]:
    if kind == "continuous_tag_weak_evidence":
        state = continuous_tag_evidence_state(item)
        evidence_level = "candidate_weak" if state.get('consumable_weak') else "unavailable"
        excerpt_source = str(state.get("coarse_summary") or "") if state.get('consumable_weak') else ""
    else:
        state = temporal_evidence_state(item) if kind == "temporal_visual_understanding" else visual_evidence_state(item)
        evidence_level = state["evidence_level"] if kind in {"visual_understanding", "temporal_visual_understanding"} else "candidate_evidence"
        excerpt_source = ""
    if kind in {"visual_understanding", "temporal_visual_understanding"}:
        excerpt_source = _qualified_evidence_text(state)
    elif kind == "structured_visual":
        excerpt_source = _structured_visual_text(item.get("structured_visual"))
    elif kind != "continuous_tag_weak_evidence":
        excerpt_source = _first_text(
            item.get("human_corrected_visual_text"),
            item.get("visual_text"),
            item.get("ocr_text"),
            _structured_visual_text(item.get("structured_visual")),
        )
    selector = []
    if kind in {'visual_understanding', 'temporal_visual_understanding'}:
        selector = understanding_source_selector(item, temporal=kind == 'temporal_visual_understanding')
    elif kind == 'continuous_tag_weak_evidence':
        selector = ['temporal_tag_delta']
    elif kind == 'structured_visual':
        selector = ['structured_visual']
    else:
        selector = visual_text_source_selector(item)
    locators = [snapshot.locator([source_position, *selector])] if snapshot is not None and selector else []
    return {
        "timeline_index": index,
        "kind": kind,
        'source_locators': locators,
        'recognition_provenance': visual_recognition_provenance(item, selector, text=excerpt_source),
        'source_context': {'timeline_array_position': source_position,
                           'selected_field': selector,
                           'scope': 'timeline_observation_not_original_media',
                           'interpretation_accuracy': 'not_assessed'},
        "start": _seconds(item.get("start")),
        "end": _seconds(item.get("end")),
        "start_time": format_timestamp(_seconds(item.get("start"))),
        "end_time": format_timestamp(_seconds(item.get("end"))),
        "visual_route": item.get("visual_route") or "",
        "evidence_level": evidence_level,
        "inference_basis": (state.get("inference_basis") or "") if evidence_level == "inferred" else "",
        "excerpt": excerpt_source,
        "evidence_paths": _evidence_paths(item),
    }


def _moment_chunks_for_indexes(moment_index: dict[str, Any], timeline_indexes: list[int]) -> list[dict[str, Any]]:
    wanted = set(timeline_indexes)
    chunks = moment_index.get("chunks") if isinstance(moment_index.get("chunks"), list) else []
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        indexes = [_safe_int(value) for value in (chunk.get("timeline_indexes") or [])]
        overlap = [value for value in indexes if value in wanted]
        if not overlap:
            continue
        out.append(
            {
                "chunk_index": chunk.get("chunk_index"),
                "start": _seconds(chunk.get("start")),
                "end": _seconds(chunk.get("end")),
                "start_time": chunk.get("start_time") or format_timestamp(_seconds(chunk.get("start"))),
                "end_time": chunk.get("end_time") or format_timestamp(_seconds(chunk.get("end"))),
                "timeline_indexes": overlap,
                "has_visual_evidence": bool(chunk.get("has_visual_evidence")),
                "has_temporal_evidence": bool(chunk.get("has_temporal_evidence")),
                "keywords": (chunk.get("keywords") or []) if isinstance(chunk.get("keywords"), list) else [],
                "evidence_paths": (chunk.get("evidence_paths") or []) if isinstance(chunk.get("evidence_paths"), list) else [],
            }
        )
    return out


def _review_gap_reasons(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in (
        "review_reason",
        "blocked_reason",
        "quality_gap",
        "quality_gaps",
        "quality_issues",
        "missing_reason",
        "ocr_issue",
        "screen_text_issue",
    ):
        value = item.get(key)
        if isinstance(value, list):
            reasons.extend(str(part) for part in value if str(part).strip())
        elif value:
            reasons.append(str(value))
    for flag in (
        "needs_human_review",
        "needs_review",
        "missing_visual_text",
        "screen_text_low_confidence",
        "ocr_text_empty",
        "semantic_frame_without_analysis",
        "temporal_sequence_without_analysis",
    ):
        if item.get(flag):
            reasons.append(flag)
    tag_state = continuous_tag_evidence_state(item)
    if tag_state.get("decision") == "review_gap":
        reasons.append("continuous_tag_review_gap")
    review_status = str(item.get("review_status") or "").strip()
    if review_status and review_status not in {"closed", "accepted", "accepted_known_gap"}:
        reasons.append(f"review_status:{review_status}")
    for label, state in (("visual", visual_evidence_state(item)), ("temporal", temporal_evidence_state(item))):
        if state["evidence_level"] == "unavailable":
            reason = str(state.get("unavailable_reason") or "unable to determine after observation")
            reasons.append(f"{label}_seen_unavailable:{reason}")
    return _unique(reasons)[:12]


def _transcript_source_label(transcript_path: Path | None, manifest: dict[str, Any]) -> str:
    if transcript_path:
        name = transcript_path.name.lower()
        if "human-corrected" in name or manifest.get("human_corrected_transcript_json"):
            return "human_corrected_transcript"
        if "llm-readable" in name:
            return "llm_readable_transcript"
        if "agent-readable" in name:
            return "agent_readable_transcript"
        if "readable" in name:
            return "readable_transcript"
        if "llm-corrected" in name:
            return "llm_corrected_transcript"
        if "corrected" in name:
            return "corrected_transcript"
        if "source-arbitrated" in name or manifest.get("corrected_transcript_source") == "transcript_source_arbitration":
            return "source_arbitrated_transcript"
        if "normalized" in name:
            return "normalized_transcript"
        return "transcript_sidecar"
    return "timeline_transcript"



def _transcript_source_decision(root: Path, manifest: dict[str, Any], transcript_path: Path | None, label: str) -> dict[str, Any]:
    selected = transcript_path.resolve() if transcript_path else None
    corrected_candidates = _transcript_candidate_paths(
        root,
        manifest,
        (
            "source_arbitrated_transcript_json",
            "human_corrected_transcript_json",
            "llm_readable_transcript_json",
            "agent_readable_transcript_json",
            "readable_transcript_json",
            "llm_corrected_transcript_json",
            "corrected_transcript_json",
        ),
        (
            "source-arbitrated-transcript.json",
            "human-corrected-transcript.json",
            "llm-readable-transcript.json",
            "agent-readable-transcript.json",
            "readable-transcript.json",
            "llm-corrected-transcript.json",
            "corrected-transcript.json",
        ),
    )
    raw_candidates = _transcript_candidate_paths(
        root,
        manifest,
        ("normalized_transcript_json", "transcript_json", "source_transcript", "transcript_path"),
        ("normalized-transcript.json", "transcript.json"),
    )
    uses_corrected = label in {"source_arbitrated_transcript", "human_corrected_transcript", "llm_corrected_transcript", "corrected_transcript", "llm_readable_transcript", "agent_readable_transcript", "readable_transcript"}
    uses_readable = label in {"llm_readable_transcript", "agent_readable_transcript", "readable_transcript"}
    priority = "readable_corrected_transcript_preferred" if uses_readable else ("corrected_transcript_preferred" if uses_corrected else ("raw_asr_fallback" if selected else "timeline_fallback"))
    if uses_readable:
        priority_reason = "readable transcript is preferred after semantic correction/postprocess for final notes."
    elif label == "source_arbitrated_transcript":
        priority_reason = "source-arbitrated transcript is preferred after term/tool-name arbitration."
    elif label in {"human_corrected_transcript", "llm_corrected_transcript", "corrected_transcript"}:
        priority_reason = "reviewed/corrected transcript is preferred over source-arbitrated and normalized ASR."
    elif selected:
        priority_reason = "no corrected transcript was available; normalized/raw transcript was used."
    else:
        priority_reason = "no transcript sidecar was available; timeline text was used."
    return {
        "selected_label": label,
        "selected_path": str(selected) if selected else "timeline_fallback",
        "uses_corrected_transcript": uses_corrected,
        "uses_readable_transcript": uses_readable,
        "priority": priority,
        "priority_reason": priority_reason,
        "corrected_candidates": [str(path) for path in corrected_candidates],
        "raw_asr_candidates": [str(path) for path in raw_candidates],
        "raw_asr_path": str(raw_candidates[0]) if raw_candidates else "",
    }


def _transcript_candidate_paths(root: Path, manifest: dict[str, Any], keys: tuple[str, ...], defaults: tuple[str, ...]) -> list[Path]:
    values: list[str] = []
    for key in keys:
        value = manifest.get(key)
        if value:
            values.append(str(value))
    values.extend(defaults)
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = _bundle_path(root, value).resolve()
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_file():
            paths.append(path)
    return paths


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(str(value or "")).expanduser()
    return path if path.is_absolute() else root / path


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _transcript_quality_policy(transcript_arbitration: dict[str, Any]) -> dict[str, Any]:
    quality = transcript_arbitration.get("quality_summary") if isinstance(transcript_arbitration.get("quality_summary"), dict) else {}
    policy = quality.get("summary_input_policy") if isinstance(quality.get("summary_input_policy"), dict) else {}
    review_refs = quality.get("review_segment_refs") if isinstance(quality.get("review_segment_refs"), list) else []
    trusted = quality.get("trusted_segment_indexes") if isinstance(quality.get("trusted_segment_indexes"), list) else []
    if not transcript_arbitration.get("exists"):
        return {
            "status": "missing_arbitration",
            "mode": "raw_transcript_unmerged",
            "can_use_corrected_transcript": False,
            "must_exclude_review_segments": False,
            "trusted_segment_indexes": [],
            "review_segment_refs": [],
            "guidance": "Transcript arbitration has not run; use transcript wording as unmerged evidence and keep term claims cautious.",
        }
    return {
        "status": quality.get("status", transcript_arbitration.get("status", "unknown")),
        "mode": policy.get("mode", "unknown"),
        "can_use_corrected_transcript": bool(policy.get("can_use_corrected_transcript")),
        "must_exclude_review_segments": bool(policy.get("must_exclude_review_segments")),
        "safe_segment_count": quality.get("safe_segment_count", 0),
        "review_segment_count": quality.get("review_segments", 0),
        "trusted_segment_indexes": trusted[:120],
        "review_segment_refs": review_refs[:80],
        "guidance": policy.get("guidance") or "; ".join(str(x) for x in quality.get("smart_summary_guidance") or []),
    }

def _source_context_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = load_page_metadata(root, manifest)
    pointer = str(manifest.get("page_metadata_json") or "").strip()
    artifact_path = root / pointer if pointer else root / "source" / "page-metadata.json"
    available = bool(metadata and any(metadata.get(key) for key in ("title", "description", "author", "uploader", "tags", "chapters")))
    return {
        "available": available,
        "trust": "untrusted_weak_context",
        "cannot_override_transcript_or_visual_evidence": True,
        "artifact_path": str(artifact_path) if artifact_path.is_file() else "",
        "artifact_sha256": _file_sha256(artifact_path) if artifact_path.is_file() else str(metadata.get("artifact_sha256") or ""),
        "title": str(metadata.get("title") or ""),
        "platform": str(metadata.get("platform") or ""),
        "author": str(metadata.get("author") or metadata.get("uploader") or ""),
        "published_at": str(metadata.get("published_at") or ""),
        "tags": [str(item) for item in (metadata.get("tags") or [])[:30]],
        "chapters": [item for item in (metadata.get("chapters") or [])[:40] if isinstance(item, dict)],
        "context": page_metadata_context(metadata, max_chars=1600),
    }

def _quality_notes(
    transcript_segments: list[dict[str, Any]],
    term_summary: dict[str, Any],
    term_arbitration_codex: dict[str, Any],
    visual_digest: dict[str, Any],
    transcript_arbitration: dict[str, Any],
    transcript_quality_gate: dict[str, Any],
    term_correction_impact: dict[str, Any],
    transcript_semantic: dict[str, Any],
) -> list[str]:
    notes = [
        "Use punctuated_text for readability, but keep raw_text/corrected_text as evidence references.",
        "High-confidence term replacements can be used directly in final human-readable files.",
    ]
    gate_status = str(transcript_quality_gate.get("status") or "missing") if isinstance(transcript_quality_gate, dict) else "missing"
    if gate_status == "passed":
        notes.append("Transcript quality gate passed; corrected transcript can feed full-transcript and smart-summary input.")
    elif gate_status not in {"missing", "not_required"}:
        notes.append("Transcript quality gate has not passed; run agent-readable-transcript-rewrite or review transcript-quality-gate.md before final use.")
    arbitration_quality = transcript_arbitration.get("quality_summary") if isinstance(transcript_arbitration.get("quality_summary"), dict) else {}
    arbitration_policy = _transcript_quality_policy(transcript_arbitration)
    if not transcript_arbitration.get("exists"):
        notes.append("Transcript source arbitration has not run; final summary should treat ASR/subtitle wording as unmerged evidence.")
    elif arbitration_policy.get("must_exclude_review_segments") or arbitration_quality.get("review_segments") or arbitration_quality.get("low_confidence_conflicts"):
        notes.append("Transcript arbitration still has low-confidence conflicts; use corrected transcript only outside the listed review segments.")
    elif arbitration_quality.get("high_confidence_term_replacements"):
        notes.append("Transcript arbitration completed with high-confidence term replacements; corrected transcript is preferred for smart-summary.")
    if term_arbitration_codex.get("codex_review_required"):
        notes.append("Codex term arbitration pack exists but reviewed decisions are not imported; do not treat ambiguous tool/domain terms as final.")
    elif term_arbitration_codex.get("ready_for_transcript_arbitration"):
        notes.append("Codex term arbitration has imported glossary terms; use them for transcript-source arbitration and final wording checks.")
    if term_correction_impact.get("required") and not term_correction_impact.get("passed"):
        notes.append("Term correction impact gate has not passed; fix residual aliases before treating smart-summary as final.")
    elif term_correction_impact.get("passed"):
        notes.append("Term correction impact gate passed; final human-readable exports are clean for the checked aliases.")
    semantic_status = str(transcript_semantic.get("final_status") or "not_started") if isinstance(transcript_semantic, dict) else "not_started"
    if semantic_status in {"needs_codex_or_llm_review", "needs_human_review", "needs_closure"}:
        notes.append("General ASR/subtitle semantic correction is not closed; keep suspected misrecognitions out of final claims or list them as review points.")
    elif semantic_status == "needs_smart_summary_refresh":
        notes.append("Semantic corrections are closed but smart-summary has not absorbed them; regenerate summary from source-arbitrated transcript before final use.")
    elif semantic_status == "ready_for_summary_input":
        notes.append("General ASR/subtitle semantic correction is ready; source-arbitrated transcript should be preferred for smart-summary wording.")
    if term_summary.get("needs_review_terms"):
        notes.append("Some terms still need human review; do not treat them as confirmed facts.")
    if visual_digest.get("status") == "not_applicable":
        notes.append("Visual/courseware evidence is not applicable to this verified audio-only source without supplemental materials.")
    elif not visual_digest.get("items"):
        notes.append("No reliable visual/courseware digest was available; keep visual evidence boundary in smart-summary.")
    else:
        notes.append("Visual/courseware digest is supplementary; do not override ASR unless evidence is explicit.")
    if len(transcript_segments) == 0:
        notes.append("Transcript is empty; smart-summary cannot be complete without manual input.")
    return notes


def _render_pack_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Smart Summary Input Pack: {result.get('title')}",
        "",
        f"- Schema: `{result.get('schema')}`",
        f"- Created: `{result.get('created_at')}`",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Transcript source: `{result.get('transcript_source')}`",
        *_transcript_source_decision_lines(result.get("transcript_source_decision") if isinstance(result.get("transcript_source_decision"), dict) else {}),
        "",
        "## 使用规则",
        "",
    ]
    for note in result.get("quality_notes") or []:
        lines.append(f"- {note}")
    content_scope = result.get("content_scope") if isinstance(result.get("content_scope"), dict) else {}
    lines.extend(
        [
            "",
            "## 内容范围",
            "",
            f"- Status: `{content_scope.get('status', 'not_configured')}`",
            f"- Active: `{bool(content_scope.get('active'))}`",
            f"- Excluded transcript cues: `{content_scope.get('excluded_transcript_cue_count', 0)}`",
            f"- Projection-omitted cues: `{content_scope.get('projection_omitted_cue_count', 0)}`",
            f"- Excluded timeline items: `{content_scope.get('excluded_timeline_item_count', 0)}`",
            f"- Dispositions: `{json.dumps(content_scope.get('disposition_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
            "- Boundary: excluded material stays in the full transcript and intermediate evidence, but must not enter Smart Summary.",
            "",
        ]
    )
    for interval in content_scope.get("excluded_intervals") or []:
        if not isinstance(interval, dict):
            continue
        lines.append(
            f"- Exclude `{format_timestamp(_seconds(interval.get('start')))}` - "
            f"`{format_timestamp(_seconds(interval.get('end')))}`: `{interval.get('reason', '')}`"
        )
    if content_scope.get("excluded_intervals"):
        lines.append("")
    projection_review_gaps = [
        row
        for row in result.get("transcript_projection_review_gaps") or []
        if isinstance(row, dict)
    ]
    if projection_review_gaps:
        lines.extend(["### 转写投影待复核", ""])
        for row in projection_review_gaps[:80]:
            lines.append(
                f"- `{format_timestamp(_seconds(row.get('start')))}` - "
                f"`{format_timestamp(_seconds(row.get('end')))}` · "
                f"`{','.join(str(value) for value in row.get('reasons') or [])}` · 不得作为事实或猜补原文"
            )
        lines.append("")
    source_context = result.get("source_context") if isinstance(result.get("source_context"), dict) else {}
    lines.extend(["", "## 网页来源上下文（低权重）", "", f"- Available: `{bool(source_context.get('available'))}`", f"- Trust: `{source_context.get('trust', 'missing')}`", f"- Artifact: `{source_context.get('artifact_path', '')}`", f"- SHA-256: `{source_context.get('artifact_sha256', '')}`", f"- Title: {source_context.get('title', '')}", f"- Platform / author: `{source_context.get('platform', '')}` / {source_context.get('author', '')}", "- Boundary: may suggest topic/entities/chapters; cannot override transcript or visual evidence.", ""])
    if source_context.get("context"):
        lines.extend([str(source_context.get("context")), ""])
    trace = result.get("evidence_trace") if isinstance(result.get("evidence_trace"), dict) else {}
    trace_summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    lines.extend([
        "",
        "## 证据追踪",
        "",
        f"- Transcript source: `{trace.get('transcript_source') or result.get('transcript_source')}`",
        f"- Transcript segments: `{trace_summary.get('transcript_segment_count') or len(result.get('transcript_segments') or [])}`",
        f"- Timeline indexes with transcript: `{trace_summary.get('timeline_indexes_with_transcript') or 0}`",
        f"- OCR/ebook items: `{trace_summary.get('ocr_or_ebook_items') or 0}`",
        f"- High-res tile items: `{trace_summary.get('high_res_tile_items') or 0}`",
        f"- Visual understanding items: `{trace_summary.get('visual_understanding_items') or 0}`",
        f"- Temporal understanding items: `{trace_summary.get('temporal_understanding_items') or 0}`",
        f"- Moment chunks: `{trace_summary.get('moment_chunks') or 0}`",
        f"- Review gaps: `{trace_summary.get('review_gaps') or 0}`",
        "",
    ])
    if trace.get("moment_chunks"):
        lines.extend(["### Moment evidence", "", "| Chunk | Time | Timeline indexes | Visual | Temporal | Keywords |", "| ---: | --- | --- | --- | --- | --- |"])
        for chunk in (trace.get("moment_chunks") or [])[:20]:
            lines.append(
                f"| {chunk.get('chunk_index')} | {_md(str(chunk.get('start_time')) + ' - ' + str(chunk.get('end_time')))} | {_md(', '.join(str(x) for x in chunk.get('timeline_indexes') or []))} | {bool(chunk.get('has_visual_evidence'))} | {bool(chunk.get('has_temporal_evidence'))} | {_md(', '.join(str(x) for x in chunk.get('keywords') or []))} |"
            )
        lines.append("")
    if trace.get("review_gaps"):
        lines.extend(["### Review gaps", "", "| Timeline | Time | Reasons |", "| ---: | --- | --- |"])
        for gap in (trace.get("review_gaps") or [])[:30]:
            lines.append(f"| {gap.get('timeline_index')} | {_md(str(gap.get('start_time')) + ' - ' + str(gap.get('end_time')))} | {_md(', '.join(str(x) for x in gap.get('reasons') or []))} |")
        lines.append("")
    arbitration = result.get("transcript_arbitration") if isinstance(result.get("transcript_arbitration"), dict) else {}
    arbitration_quality = arbitration.get("quality_summary") if isinstance(arbitration.get("quality_summary"), dict) else {}
    arbitration_summary = arbitration.get("summary") if isinstance(arbitration.get("summary"), dict) else {}
    term_gate = result.get("term_correction_impact_gate") if isinstance(result.get("term_correction_impact_gate"), dict) else {}
    term_arbitration = result.get("term_arbitration_codex") if isinstance(result.get("term_arbitration_codex"), dict) else {}
    transcript_semantic = result.get("transcript_semantic_correction") if isinstance(result.get("transcript_semantic_correction"), dict) else {}
    transcript_quality_gate = result.get("transcript_quality_gate") if isinstance(result.get("transcript_quality_gate"), dict) else {}
    lines.extend([
        "",
        "## Codex 术语/工具名语义仲裁",
        "",
        f"- Exists: `{bool(term_arbitration.get('exists'))}`",
        f"- Status: `{term_arbitration.get('status', '')}`",
        f"- Candidate count: `{term_arbitration.get('candidate_count', 0)}`",
        f"- Draft decisions: `{term_arbitration.get('draft_decision_count', 0)}`",
        f"- Imported decisions: `{term_arbitration.get('imported_decision_count', 0)}`",
        f"- Accepted decisions: `{term_arbitration.get('accepted_decision_count', 0)}`",
        f"- Glossary terms: `{term_arbitration.get('glossary_term_count', 0)}`",
        f"- Codex review required: `{bool(term_arbitration.get('codex_review_required'))}`",
        f"- Semantic strategy: `{term_arbitration.get('semantic_strategy', '')}`",
        f"- Semantic review status: `{term_arbitration.get('semantic_review_status', '')}`",
        f"- Rule draft is semantic confirmation: `{not bool(term_arbitration.get('rule_draft_is_not_semantic_confirmation', True))}`",
        f"- Ready for transcript arbitration: `{bool(term_arbitration.get('ready_for_transcript_arbitration'))}`",
        f"- Prompt: `{term_arbitration.get('prompt_path', '')}`",
        f"- Result JSON: `{term_arbitration.get('result_path', '')}`",
        "",
    ])
    for action in term_arbitration.get("next_actions") or []:
        lines.append(f"- Next: {action}")
    if term_arbitration.get("next_actions"):
        lines.append("")
    lines.extend([
        "",
        "## 逐字稿质量门禁",
        "",
        f"- Exists: `{bool(transcript_quality_gate.get('exists'))}`",
        f"- Status: `{transcript_quality_gate.get('status', 'missing')}`",
        f"- OK: `{bool(transcript_quality_gate.get('ok'))}`",
        f"- Fail / warning: `{transcript_quality_gate.get('fail_count', 0)}` / `{transcript_quality_gate.get('warning_count', 0)}`",
        f"- Punctuation density: `{transcript_quality_gate.get('punctuation_density_per_1000_chars', '')}` / 1000 chars",
        f"- Source: `{transcript_quality_gate.get('source_path', '')}`",
        "",
        "## ASR/字幕通用语义纠错",
        "",
        f"- Exists: `{bool(transcript_semantic.get('exists'))}`",
        f"- Final status: `{transcript_semantic.get('final_status', 'not_started')}`",
        f"- Pack status: `{transcript_semantic.get('pack_status', '')}`",
        f"- Validation status: `{transcript_semantic.get('validation_status', '')}`",
        f"- Closure status: `{transcript_semantic.get('closure_status', '')}`",
        f"- Readable impact status: `{transcript_semantic.get('readable_impact_status', '')}`",
        f"- Smart-summary impact status: `{transcript_semantic.get('summary_impact_status', '')}`",
        f"- Candidate count: `{transcript_semantic.get('candidate_count', 0)}`",
        f"- Semantic attention count: `{transcript_semantic.get('semantic_attention_count', 0)}`",
        f"- Accepted decisions: `{transcript_semantic.get('accepted_decision_count', 0)}`",
        f"- Review required: `{transcript_semantic.get('review_required_count', 0)}`",
        f"- Applied corrections: `{transcript_semantic.get('applied_correction_count', 0)}`",
        f"- Changed segments: `{transcript_semantic.get('changed_segment_count', 0)}`",
        f"- Summary residual originals: `{transcript_semantic.get('summary_residual_original_total', 0)}`",
        f"- Summary absorption rate: `{transcript_semantic.get('summary_absorption_rate', 0)}`",
        f"- Corrected transcript: `{transcript_semantic.get('corrected_transcript_path', '')}`",
        "",
    ])
    if transcript_semantic.get("candidate_type_counts"):
        lines.append("- Candidate types: " + ", ".join(f"`{key}`={value}" for key, value in sorted((transcript_semantic.get("candidate_type_counts") or {}).items())))
    for action in transcript_semantic.get("next_actions") or []:
        lines.append(f"- Next: {action}")
    lines.extend([
        "",
        "## 术语纠错影响门禁",
        "",
        f"- Exists: `{bool(term_gate.get('exists'))}`",
        f"- Required: `{bool(term_gate.get('required'))}`",
        f"- Passed: `{bool(term_gate.get('passed'))}`",
        f"- Status: `{term_gate.get('status', '')}`",
        f"- Final export alias total: `{term_gate.get('final_export_alias_total', 0)}`",
        f"- Detail: {term_gate.get('detail', '')}",
        "",
        "## 字幕/ASR 仲裁质量",
        "",
        f"- Exists: `{bool(arbitration.get('exists'))}`",
        f"- Status: `{arbitration.get('status', 'missing')}`",
        f"- Source: `{arbitration.get('path', '')}`",
        f"- Quality status: `{arbitration_quality.get('status', arbitration_summary.get('quality_status', ''))}`",
        f"- Changed segments: `{arbitration_quality.get('changed_segments', arbitration_summary.get('changed_segments', 0))}`",
        f"- Review segments: `{arbitration_quality.get('review_segments', arbitration_summary.get('review_segments', 0))}`",
        f"- Average confidence: `{arbitration_quality.get('average_confidence', arbitration_summary.get('average_confidence', 0))}`",
        f"- High-confidence term replacements: `{arbitration_quality.get('high_confidence_term_replacements', arbitration_summary.get('high_confidence_term_replacements', 0))}`",
        f"- Low-confidence conflicts: `{arbitration_quality.get('low_confidence_conflicts', arbitration_summary.get('low_confidence_conflicts', 0))}`",
    ])
    policy = result.get("transcript_quality_policy") if isinstance(result.get("transcript_quality_policy"), dict) else {}
    lines.extend([
        f"- Summary input mode: `{policy.get('mode', '')}`",
        f"- Can use corrected transcript: `{policy.get('can_use_corrected_transcript', False)}`",
        f"- Must exclude review segments: `{policy.get('must_exclude_review_segments', False)}`",
        f"- Safe segment count: `{policy.get('safe_segment_count', 0)}`",
        f"- Review segment count: `{policy.get('review_segment_count', 0)}`",
        f"- Policy guidance: {policy.get('guidance', '')}",
    ])
    review_refs = policy.get("review_segment_refs") if isinstance(policy.get("review_segment_refs"), list) else []
    if review_refs:
        lines.extend(["", "### 仲裁待复核片段", "", "| # | Time | Reason | Confidence | Text |", "| ---: | --- | --- | ---: | --- |"])
        for row in review_refs[:30]:
            lines.append(f"| {row.get('index')} | {_md(str(row.get('time_range') or ''))} | `{row.get('reason', '')}` | {row.get('confidence', '')} | {_md(str(row.get('corrected_text') or row.get('original_text') or ''))} |")
    guidance = arbitration_quality.get("smart_summary_guidance") if isinstance(arbitration_quality.get("smart_summary_guidance"), list) else []
    for note in guidance[:8]:
        lines.append(f"- Guidance: {note}")
    lines.extend(["", "## 术语纠错", ""])
    term_summary = result.get("term_summary") if isinstance(result.get("term_summary"), dict) else {}
    high_conf = term_summary.get("high_confidence_terms") if isinstance(term_summary.get("high_confidence_terms"), list) else []
    if high_conf:
        lines.extend(["| Canonical | Raw mentions | Confidence | Sources |", "| --- | --- | ---: | --- |"])
        for row in high_conf[:30]:
            lines.append(f"| {_md(row.get('canonical_term'))} | {_md(', '.join(str(x) for x in row.get('raw_mentions') or []))} | {row.get('confidence')} | {_md(str(row.get('source_counts') or {}))} |")
    else:
        lines.append("（暂无高置信术语纠错。）")
    needs_review = term_summary.get("needs_review_terms") if isinstance(term_summary.get("needs_review_terms"), list) else []
    if needs_review:
        lines.extend(["", "### 待人工复核术语", ""])
        for row in needs_review[:30]:
            lines.append(f"- `{row.get('canonical_term')}` <- `{', '.join(str(x) for x in row.get('raw_mentions') or [])}` confidence={row.get('confidence')}")
    lines.extend(["", "## 纠正版转写导航", ""])
    for seg in (result.get("transcript_segments") or [])[:160]:
        lines.extend(
            [
                f"### {seg.get('start_time')} - {seg.get('end_time')}",
                "",
                f"- Corrected: {seg.get('corrected_text') or '（空）'}",
                f"- Punctuated: {seg.get('punctuated_text') or '（空）'}",
                "",
            ]
        )
    visual = result.get("visual_digest") if isinstance(result.get("visual_digest"), dict) else {}
    lines.extend(["", "## 视觉/课件证据摘要", "", f"- Route counts: `{visual.get('route_counts') or {}}`", f"- Items with visual digest: `{visual.get('total_items_with_visual_digest') or 0}`", ""])
    for item in (visual.get("items") or [])[:80]:
        lines.extend([f"### #{item.get('timeline_index')} {item.get('start_time')} - {item.get('end_time')} `{item.get('visual_route')}`", ""])
        for label, key in (("OCR/屏幕文字", "visual_text"), ("图文结构", "structured_visual"), ("单帧理解", "visual_understanding"), ("连续片段", "temporal_visual_understanding")):
            if item.get(key):
                lines.append(f"- {label}: {item.get(key)}")
        if item.get("evidence_paths"):
            lines.append(f"- Evidence: `{'; '.join(item.get('evidence_paths') or [])}`")
        lines.append("")
    lines.append(render_courseware_material(result.get("courseware_material", {"active": False})))
    return "\n".join(lines).rstrip() + "\n"



def _transcript_source_decision_lines(decision: dict[str, Any]) -> list[str]:
    if not decision:
        return []
    lines = [
        f"- Transcript selected label: `{decision.get('selected_label') or 'unknown'}`",
        f"- Uses corrected transcript: `{bool(decision.get('uses_corrected_transcript'))}`",
        f"- Transcript priority: `{decision.get('priority') or 'unknown'}`",
        f"- Transcript priority reason: {decision.get('priority_reason') or ''}",
    ]
    raw = str(decision.get("raw_asr_path") or "").strip()
    if raw:
        lines.append(f"- Raw ASR fallback: `{raw}`")
    corrected = decision.get("corrected_candidates") if isinstance(decision.get("corrected_candidates"), list) else []
    if corrected:
        lines.append("- Corrected transcript candidates: " + "；".join(f"`{path}`" for path in corrected[:5]))
    return lines
def _matching_timeline_item(by_index: dict[int, dict[str, Any]], start: float, end: float) -> dict[str, Any] | None:
    best = None
    best_overlap = 0.0
    for item in by_index.values():
        left = _seconds(item.get("start"))
        right = _seconds(item.get("end"))
        overlap = max(0.0, min(end, right) - max(start, left))
        if overlap > best_overlap:
            best = item
            best_overlap = overlap
    return best


def _item_index(item: dict[str, Any], fallback: int) -> int:
    try:
        return int(item.get("index") or fallback)
    except Exception:
        return fallback


def _structured_visual_text(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key in ("text", "markdown", "summary", "tables", "formulas", "code", "layout"):
            if value.get(key):
                parts.append(f"{key}: {_mapping_text(value.get(key))}")
        return " | ".join(parts)
    return _mapping_text(value)


def _mapping_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "；".join(_mapping_text(item) for item in value if _mapping_text(item))
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            text = _mapping_text(val)
            if text:
                parts.append(f"{key}: {text}")
        return "；".join(parts)
    return str(value).strip()


def _qualified_evidence_text(state: dict[str, Any]) -> str:
    if not state.get("consumable"):
        return ""
    understanding = state.get("understanding")
    if isinstance(understanding, dict):
        # Use the same content contract as correction, not transport/model metadata.
        from .visual_evidence import VISUAL_CONTENT_FIELDS, TEMPORAL_CONTENT_FIELDS
        fields = (*VISUAL_CONTENT_FIELDS, *TEMPORAL_CONTENT_FIELDS, 'summary', 'text', 'description')
        understanding = {key: value for key, value in understanding.items() if key in fields}
    text = _mapping_text(understanding)
    if not text:
        return ""
    if state.get("evidence_level") == "inferred":
        basis = _mapping_text(state.get("inference_basis"))
        suffix = f"；推断依据：{basis}" if basis else ""
        return f"推断：画面可能显示{text}{suffix}"
    return text


def _first_text(*values: Any) -> str:
    for value in values:
        text = _mapping_text(value)
        if text:
            return text
    return ""


def _evidence_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("frame_path", "frame_paths", "evidence_frame_paths", "temporal_frame_paths"):
        value = item.get(key)
        if isinstance(value, list):
            paths.extend(str(path) for path in value if str(path).strip())
        elif value:
            paths.append(str(value))
    visual = item.get("visual_understanding") if isinstance(item.get("visual_understanding"), dict) else {}
    for key in ("evidence_frame_paths", "frame_paths"):
        value = visual.get(key)
        if isinstance(value, list):
            paths.extend(str(path) for path in value if str(path).strip())
    for row in item.get("tile_result_merges") or []:
        if isinstance(row, dict) and row.get("evidence_path"):
            paths.append(str(row.get("evidence_path")))
    for row in item.get("tile_review_targets") or []:
        if isinstance(row, dict) and row.get("evidence_path"):
            paths.append(str(row.get("evidence_path")))
    return _unique(paths)[:8]


def _clip(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except Exception:
        return 0.0


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
