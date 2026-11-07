from __future__ import annotations

from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .smart_summary_chapters import build_smart_summary_chapter_pack
from .smart_summary_codex import smart_summary_quality_check
from .smart_summary_input_pack import build_smart_summary_input_pack
from .storage import read_json, read_jsonl, write_json
from .video_moment_index import build_video_moment_index
from .video_rag_pack import build_video_rag_pack
from .courseware_material import assert_material_fresh

SCHEMA = "video_knowledge_pipeline.smart_summary_section_workflow.v1"


def build_smart_summary_section_workflow(
    bundle_dir: str | Path,
    *,
    title: str = "",
    write: bool = True,
    target_chapters: int = 8,
) -> dict[str, Any]:
    """Build an editable section-level workflow for smart-summary iteration.

    This is a local planning/state layer inspired by BiliNote section editing and
    vsummary staged generation. It does not call an LLM. It prepares stable
    section rows that Codex or a future provider can rewrite one at a time.
    """

    root = Path(bundle_dir).expanduser().resolve()
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    chapters = _load_or_build_chapters(root, title=title, target_chapters=target_chapters, write=write)
    quality = smart_summary_quality_check(root, require_codex=False, write=write)
    moment_index = _load_or_build_moment_index(root, write=write)
    video_rag_chunks = _load_or_build_video_rag_chunks(root, write=write)
    input_pack = _load_or_build_input_pack(root, title=title or str(chapters.get("title") or root.name), write=write)
    semantic_context = _semantic_correction_context(root, input_pack)
    material = assert_material_fresh(root, input_pack.get("courseware_material", {}))
    assert_material_fresh(root, chapters.get("courseware_material", {}))
    sections = _build_sections(root, chapters, quality, moment_index, video_rag_chunks, semantic_context)
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": title or str(chapters.get("title") or root.name),
        "created_at": now_iso(),
        "section_count": len(sections),
        "sections_needing_rewrite": sum(1 for row in sections if row.get("status") != "ready"),
        "quality_status": quality.get("status", ""),
        "quality_passed": bool(quality.get("passed")),
        "transcript_semantic_correction": semantic_context.get("summary", {}),
        "semantic_attention_candidate_count": len(semantic_context.get("attention_candidates") or []),
        "sections": sections,
        "courseware_material": material,
        "citation_source": _citation_source(moment_index, video_rag_chunks),
        "artifacts": {
            "json": str(exports / "smart-summary-section-workflow.json"),
            "markdown": str(exports / "smart-summary-section-workflow.md"),
            "todo_json": str(exports / "smart-summary-section-todo.json"),
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_summary_writeback": True,
            "purpose": "Prepare section-level smart-summary rewrite state and prompts for Codex or a future LLM provider.",
        },
    }
    if write:
        todo = _todo_payload(root, result)
        write_json(exports / "smart-summary-section-workflow.json", result)
        (exports / "smart-summary-section-workflow.md").write_text(_render_markdown(result), encoding="utf-8")
        write_json(exports / "smart-summary-section-todo.json", todo)
        write_json(root / "mcp-smart-summary-section-workflow.args.json", {"bundle_dir": str(root), "title": title, "write": True, "target_chapters": target_chapters})
        manifest_path = root / "manifest.json"
        manifest = _read_mapping(manifest_path)
        manifest["smart_summary_section_workflow"] = "exports/smart-summary-section-workflow.json"
        manifest["smart_summary_section_workflow_markdown"] = "exports/smart-summary-section-workflow.md"
        manifest["smart_summary_section_todo"] = "exports/smart-summary-section-todo.json"
        manifest["mcp_smart_summary_section_workflow_args"] = "mcp-smart-summary-section-workflow.args.json"
        write_json(manifest_path, manifest)
        result["run_artifact"] = _register_run(root, result, write=True)
        write_json(exports / "smart-summary-section-workflow.json", result)
        (exports / "smart-summary-section-workflow.md").write_text(_render_markdown(result), encoding="utf-8")
    return result


def _load_or_build_chapters(root: Path, *, title: str, target_chapters: int, write: bool) -> dict[str, Any]:
    path = root / "exports" / "smart-summary-chapters.json"
    if path.exists():
        value = read_json(path)
        if isinstance(value, dict) and value.get("chapters"):
            try:
                assert_material_fresh(root, value.get("courseware_material", {}))
                return value
            except ValueError:
                pass  # Rebuild stale projections; never keep stale courseware.
    return build_smart_summary_chapter_pack(root, title=title, target_chapters=target_chapters, write=write)



def _load_or_build_input_pack(root: Path, *, title: str, write: bool) -> dict[str, Any]:
    path = root / "exports" / "smart-summary-input-pack.json"
    if path.exists():
        value = _read_mapping(path)
        if isinstance(value, dict) and value.get("schema"):
            try:
                assert_material_fresh(root, value.get("courseware_material", {}))
                return value
            except ValueError:
                pass
    try:
        value = build_smart_summary_input_pack(root, title=title, write=write)
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {
            "schema": "video_knowledge_pipeline.smart_summary_input_pack.unavailable.v1",
            "available": False,
            "error": str(exc),
            "transcript_semantic_correction": {"final_status": "input_pack_unavailable"},
        }


def _load_or_build_moment_index(root: Path, *, write: bool) -> dict[str, Any]:
    path = root / "exports" / "video-moment-index.json"
    if path.exists():
        value = _read_mapping(path)
        if isinstance(value.get("chunks"), list):
            return value
    try:
        value = build_video_moment_index(root, write=write)
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {
            "schema": "video_knowledge_pipeline.video_moment_index.unavailable.v1",
            "available": False,
            "error": str(exc),
            "chunks": [],
        }


def _load_or_build_video_rag_chunks(root: Path, *, write: bool) -> list[dict[str, Any]]:
    path = root / "exports" / "video-rag-chunks.jsonl"
    if path.exists():
        return _read_video_rag_chunks(path)
    try:
        build_video_rag_pack(root, write=write)
    except Exception:
        return []
    return _read_video_rag_chunks(path)


def _read_video_rag_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        rows = read_jsonl(path)
    except Exception:
        return []
    return [row for row in rows if isinstance(row, dict)]


def _citation_source(moment_index: dict[str, Any], video_rag_chunks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    summary = moment_index.get("summary") if isinstance(moment_index.get("summary"), dict) else {}
    artifacts = moment_index.get("artifacts") if isinstance(moment_index.get("artifacts"), dict) else {}
    rag_chunks = video_rag_chunks or []
    chunks_by_kind: dict[str, int] = {}
    for row in rag_chunks:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        kind = str(metadata.get("chunk_kind") or "unknown")
        chunks_by_kind[kind] = chunks_by_kind.get(kind, 0) + 1
    return {
        "available": bool(moment_index.get("chunks") or rag_chunks),
        "schema": str(moment_index.get("schema") or ""),
        "chunk_count": len(moment_index.get("chunks") or []),
        "video_rag_chunk_count": len(rag_chunks),
        "video_rag_chunks_by_kind": chunks_by_kind,
        "duration_seconds": summary.get("duration_seconds", 0.0),
        "json": artifacts.get("json", ""),
        "markdown": artifacts.get("markdown", ""),
        "video_rag_jsonl": str(Path(str(artifacts.get("json") or "")).parent / "video-rag-chunks.jsonl") if artifacts.get("json") else "",
        "error": str(moment_index.get("error") or ""),
    }


def _section_citations(
    moment_index: dict[str, Any],
    video_rag_chunks: list[dict[str, Any]] | None,
    start: float,
    end: float,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    section_start = float(start or 0.0)
    section_end = float(end or section_start)
    if section_end < section_start:
        section_start, section_end = section_end, section_start
    moment_citations = _moment_citations(moment_index, section_start, section_end, limit=limit)
    remaining = max(0, limit - len(moment_citations))
    rag_citations = _video_rag_citations(video_rag_chunks or [], section_start, section_end, limit=remaining or 2)
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for citation in [*moment_citations, *rag_citations]:
        citation_id = str(citation.get("citation_id") or "")
        if citation_id and citation_id in seen:
            continue
        if citation_id:
            seen.add(citation_id)
        merged.append(citation)
        if len(merged) >= limit:
            break
    return merged


def _moment_citations(moment_index: dict[str, Any], section_start: float, section_end: float, *, limit: int) -> list[dict[str, Any]]:
    chunks = [row for row in moment_index.get("chunks") or [] if isinstance(row, dict)]
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for chunk in chunks:
        chunk_start = float(chunk.get("start") or 0.0)
        chunk_end = float(chunk.get("end") or chunk_start)
        overlap = _time_overlap(section_start, section_end, chunk_start, chunk_end)
        if overlap <= 0 and not (chunk_start <= section_start <= chunk_end):
            continue
        score = overlap
        if chunk.get("has_visual_evidence"):
            score += 0.25
        if chunk.get("has_temporal_evidence"):
            score += 0.25
        scored.append((score, chunk_start, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    citations: list[dict[str, Any]] = []
    for _, _, chunk in scored[: max(0, limit)]:
        chunk_index = int(chunk.get("chunk_index") or 0)
        citations.append(
            {
                "citation_id": f"moment-{chunk_index:04d}",
                "chunk_index": chunk_index,
                "chunk_kind": "moment",
                "source_kind": ('asr' if chunk.get('transcript_text') else 'visual'
                                if chunk.get('visual_text') else 'temporal_visual'
                                if chunk.get('temporal_text') else 'review_gap'),
                "time_range": f"{chunk.get('start_time', '')} - {chunk.get('end_time', '')}",
                "start": float(chunk.get("start") or 0.0),
                "end": float(chunk.get("end") or 0.0),
                "timeline_indexes": chunk.get("timeline_indexes") if isinstance(chunk.get("timeline_indexes"), list) else [],
                "snippet": _clip_text(str(chunk.get("transcript_text") or chunk.get("visual_text") or chunk.get("temporal_text") or ""), 360),
                "visual_snippet": _clip_text(str(chunk.get("visual_text") or ""), 240),
                "temporal_snippet": _clip_text(str(chunk.get("temporal_text") or ""), 240),
                "evidence_paths": (chunk.get("evidence_paths") if isinstance(chunk.get("evidence_paths"), list) else [])[:8],
                "has_visual_evidence": bool(chunk.get("has_visual_evidence")),
                "has_temporal_evidence": bool(chunk.get("has_temporal_evidence")),
                "source": "video_moment_index",
                "source_locators_by_kind": chunk.get("source_locators_by_kind", {}),
                **({'source_observations_by_kind': chunk['source_observations_by_kind']}
                   if 'source_observations_by_kind' in chunk else {}),
                "fact_status": "candidate_evidence",
            }
        )
    return citations


def _video_rag_citations(chunks: list[dict[str, Any]], section_start: float, section_end: float, *, limit: int) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, float, dict[str, Any]]] = []
    priority = {"visual_evidence": 0, "moment": 1, "review_gap": 2, "content_asset": 3}
    for row in chunks:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        kind = str(metadata.get("chunk_kind") or "unknown")
        chunk_start = float(metadata.get("start") or 0.0)
        chunk_end = float(metadata.get("end") or chunk_start)
        overlap = _time_overlap(section_start, section_end, chunk_start, chunk_end)
        timeline_indexes = metadata.get("timeline_indexes") if isinstance(metadata.get("timeline_indexes"), list) else []
        timeless_asset = kind == "content_asset" and not timeline_indexes and chunk_start == 0.0 and chunk_end == 0.0
        if overlap <= 0 and not timeless_asset and not (chunk_start <= section_start <= chunk_end):
            continue
        score = overlap
        if metadata.get("has_visual_evidence"):
            score += 0.2
        if metadata.get("has_temporal_evidence"):
            score += 0.2
        if kind == "review_gap":
            score += 0.1
        if timeless_asset:
            score = 0.01
        scored.append((score, priority.get(kind, 9), chunk_start, row))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    citations: list[dict[str, Any]] = []
    for _, _, _, row in scored[: max(0, limit)]:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        kind = str(metadata.get("chunk_kind") or "unknown")
        raw_id = str(row.get("id") or f"video-rag:{len(citations) + 1}")
        start = float(metadata.get("start") or 0.0)
        end = float(metadata.get("end") or start)
        citations.append(
            {
                "citation_id": _citation_id(raw_id, prefix=f"rag-{kind}"),
                "chunk_id": raw_id,
                "chunk_kind": kind,
                "time_range": f"{metadata.get('start_time', '')} - {metadata.get('end_time', '')}",
                "start": start,
                "end": end,
                "timeline_indexes": metadata.get("timeline_indexes") if isinstance(metadata.get("timeline_indexes"), list) else [],
                "snippet": _clip_text(str(row.get("text") or ""), 420),
                "visual_snippet": _clip_text(str(row.get("text") or ""), 240) if kind == "visual_evidence" else "",
                "temporal_snippet": _clip_text(str(row.get("text") or ""), 240) if metadata.get("has_temporal_evidence") else "",
                "evidence_paths": (metadata.get("evidence_paths") if isinstance(metadata.get("evidence_paths"), list) else [])[:8],
                "tags": metadata.get("tags") if isinstance(metadata.get("tags"), list) else [],
                "keywords": metadata.get("keywords") if isinstance(metadata.get("keywords"), list) else [],
                "has_visual_evidence": bool(metadata.get("has_visual_evidence")),
                "has_temporal_evidence": bool(metadata.get("has_temporal_evidence")),
                "source": "video_rag_chunks",
                "fact_status": "review_gap_not_fact" if kind == "review_gap" else "candidate_evidence",
            }
        )
    return citations


def _time_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a < start_a:
        start_a, end_a = end_a, start_a
    if end_b < start_b:
        start_b, end_b = end_b, start_b
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _citation_id(raw_id: str, *, prefix: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw_id).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return f"{prefix}-{safe[:72] or 'chunk'}"


def _clip_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."

def _semantic_correction_context(root: Path, input_pack: dict[str, Any]) -> dict[str, Any]:
    summary = input_pack.get("transcript_semantic_correction") if isinstance(input_pack.get("transcript_semantic_correction"), dict) else {}
    pack_path = _bundle_path(root, str(summary.get("pack_path") or "transcript-semantic-correction-pack.json"))
    status_path = _bundle_path(root, "transcript-semantic-correction-status.json")
    closure_path = _bundle_path(root, str(summary.get("closure_path") or "transcript-semantic-correction-closure.json"))
    pack = _read_mapping(pack_path)
    status = _read_mapping(status_path)
    closure = _read_mapping(closure_path)
    attention_rows = status.get("semantic_attention_items") if isinstance(status.get("semantic_attention_items"), list) else []
    attention_by_id = {str(row.get("candidate_id") or ""): row for row in attention_rows if isinstance(row, dict)}
    applied_rows = closure.get("applied_corrections") if isinstance(closure.get("applied_corrections"), list) else []
    applied_by_id = {str(row.get("candidate_id") or ""): row for row in applied_rows if isinstance(row, dict)}
    candidates: list[dict[str, Any]] = []
    for row in pack.get("candidates") or []:
        if not isinstance(row, dict):
            continue
        candidate = dict(row)
        candidate_id = str(candidate.get("candidate_id") or "")
        attention = attention_by_id.get(candidate_id)
        if attention:
            candidate["semantic_attention"] = True
            candidate["priority_score"] = attention.get("priority_score", candidate.get("priority_score"))
        applied = applied_by_id.get(candidate_id)
        if applied:
            candidate["semantic_correction_status"] = "applied"
            candidate["corrected_text"] = applied.get("corrected_text") or candidate.get("candidate_text") or candidate.get("suggested_text")
            candidate["correction_confidence"] = applied.get("confidence")
            candidate["correction_rationale"] = applied.get("rationale")
        candidates.append(candidate)
    attention_candidates = [row for row in candidates if row.get("semantic_attention")]
    return {"summary": summary, "candidates": candidates, "attention_candidates": attention_candidates}


def _section_semantic_items(context: dict[str, Any], start: float, end: float, *, limit: int = 8) -> list[dict[str, Any]]:
    if end < start:
        start, end = end, start
    status = str((context.get("summary") if isinstance(context.get("summary"), dict) else {}).get("final_status") or "")
    if status in {"ready_for_summary_input", "no_candidates", "not_started"}:
        return []
    rows: list[tuple[int, float, dict[str, Any]]] = []
    for candidate in context.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        c_start = _float(candidate.get("start"), -1.0)
        c_end = _float(candidate.get("end"), c_start)
        if c_start < 0 and c_end < 0:
            continue
        overlap = _time_overlap(start, end, c_start, c_end)
        if overlap <= 0 and not (start <= c_start <= end):
            continue
        has_suggestion = bool(str(candidate.get("candidate_text") or candidate.get("suggested_text") or candidate.get("corrected_text") or "").strip())
        is_applied = str(candidate.get("semantic_correction_status") or "") == "applied"
        if not is_applied and not has_suggestion:
            continue
        priority = int(candidate.get("priority_score") or 0)
        if is_applied:
            priority += 100
        if candidate.get("semantic_attention"):
            priority += 50
        rows.append((priority, c_start, _compact_semantic_candidate(candidate)))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in rows[:limit]]


def _compact_semantic_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "correction_type": candidate.get("correction_type"),
        "risk_level": candidate.get("risk_level"),
        "start": candidate.get("start"),
        "end": candidate.get("end"),
        "time_range": candidate.get("time_range"),
        "original_text": _clip_text(str(candidate.get("original_text") or ""), 160),
        "candidate_text": _clip_text(str(candidate.get("candidate_text") or candidate.get("suggested_text") or ""), 160),
        "corrected_text": _clip_text(str(candidate.get("corrected_text") or ""), 160),
        "correction_status": candidate.get("semantic_correction_status") or "candidate",
        "correction_confidence": candidate.get("correction_confidence"),
        "correction_rationale": _clip_text(str(candidate.get("correction_rationale") or ""), 180),
        "reason": candidate.get("reason"),
        "needs_human_review": bool(candidate.get("needs_human_review")),
        "semantic_attention": bool(candidate.get("semantic_attention")),
        "priority_score": candidate.get("priority_score", 0),
        "evidence_ids": candidate.get("evidence_ids") if isinstance(candidate.get("evidence_ids"), list) else [],
    }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _build_sections(root: Path, chapters: dict[str, Any], quality: dict[str, Any], moment_index: dict[str, Any], video_rag_chunks: list[dict[str, Any]] | None = None, semantic_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    chapter_rows = [row for row in chapters.get("chapters") or [] if isinstance(row, dict)]
    failed_quality = [str(row.get("key") or "") for row in quality.get("checks") or [] if isinstance(row, dict) and not row.get("passed")]
    sections: list[dict[str, Any]] = []
    semantic_context = semantic_context if isinstance(semantic_context, dict) else {"summary": {}, "candidates": [], "attention_candidates": []}
    for chapter in chapter_rows:
        section_start = float(chapter.get("start") or 0.0)
        section_end = float(chapter.get("end") or 0.0)
        semantic_items = _section_semantic_items(semantic_context, section_start, section_end)
        status, reasons = _section_status(chapter, failed_quality, semantic_context.get("summary") if isinstance(semantic_context.get("summary"), dict) else {}, semantic_items)
        section_id = f"chapter-{int(chapter.get('index') or len(sections) + 1):04d}"
        citations = _section_citations(moment_index, video_rag_chunks or [], section_start, section_end)
        evidence = _section_evidence(chapter, citations, semantic_items)
        sections.append(
            {
                "section_id": section_id,
                "chapter_index": chapter.get("index"),
                "title": str(chapter.get("title") or section_id),
                "start": float(chapter.get("start") or 0.0),
                "end": float(chapter.get("end") or 0.0),
                "start_time": str(chapter.get("start_time") or ""),
                "end_time": str(chapter.get("end_time") or ""),
                "status": status,
                "reasons": reasons,
                "evidence": evidence,
                "citations": citations,
                "semantic_correction_items": semantic_items,
                "rewrite_prompt": _rewrite_prompt(root, chapter, failed_quality, evidence, citations, semantic_context.get("summary") if isinstance(semantic_context.get("summary"), dict) else {}),
                "retry_command": f".\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex {_ps_quote(str(root))}",
                "artifacts": {
                    "chapter_pack": "exports/smart-summary-chapters.md",
                    "input_pack": "exports/smart-summary-input-pack.md",
                    "long_memory": "exports/long-video-memory-pack.md",
                    "quality": "exports/smart-summary-quality.md",
                },
            }
        )
    return sections


def _section_status(chapter: dict[str, Any], failed_quality: list[str], semantic_summary: dict[str, Any] | None = None, semantic_items: list[dict[str, Any]] | None = None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not chapter.get("summary_sentences"):
        reasons.append("missing_chapter_summary")
    if not chapter.get("key_points"):
        reasons.append("missing_key_points")
    if not chapter.get("actions"):
        reasons.append("missing_actions")
    if "visual_boundary" in failed_quality and not chapter.get("visual_notes"):
        reasons.append("visual_boundary_needs_explicit_note")
    if any(key in failed_quality for key in ("overview_readable", "segment_not_asr_dump", "balanced_sections")):
        reasons.append("global_summary_quality_failed")
    semantic_status = str((semantic_summary or {}).get("final_status") or "")
    if semantic_items:
        reasons.append("transcript_semantic_correction_pending")
    if semantic_status == "needs_smart_summary_refresh":
        reasons.append("semantic_correction_summary_refresh_required")
    if semantic_status == "needs_readable_export_fix":
        reasons.append("semantic_correction_readable_export_fix_required")
    return ("needs_rewrite" if reasons else "ready", reasons)


def _section_evidence(chapter: dict[str, Any], citations: list[dict[str, Any]] | None = None, semantic_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "summary_sentences": [str(value) for value in chapter.get("summary_sentences") or []][:6],
        "key_points": _row_texts(chapter.get("key_points")),
        "actions": _row_texts(chapter.get("actions")),
        "reusable_expressions": _row_texts(chapter.get("reusable_expressions")),
        "visual_notes": _row_texts(chapter.get("visual_notes")),
        "courseware_material": chapter.get("courseware_material", {"active": False}),
        "segment_count": int(chapter.get("segment_count") or 0),
        "citations": citations or [],
        "semantic_correction_items": semantic_items or [],
    }


def _row_texts(value: Any) -> list[str]:
    out: list[str] = []
    for row in value or []:
        if isinstance(row, dict):
            text = str(row.get("text") or row.get("title") or row.get("summary") or "").strip()
            time = str(row.get("time") or "").strip()
            if text:
                out.append((f"{time} {text}" if time else text).strip())
        elif row:
            out.append(str(row))
    return out[:8]


def _rewrite_prompt(root: Path, chapter: dict[str, Any], failed_quality: list[str], evidence: dict[str, Any], citations: list[dict[str, Any]] | None = None, semantic_summary: dict[str, Any] | None = None) -> str:
    title = str(chapter.get("title") or "").strip()
    start = str(chapter.get("start_time") or "").strip()
    end = str(chapter.get("end_time") or "").strip()
    lines = [
        f"请基于 VKP bundle `{root}` 的证据，为 `{start} - {end}` 这一节重写 smart-summary 的一个章节。",
        f"章节标题：{title}",
        "要求：压缩 ASR，不复制流水账；保留时间戳；明确视觉证据边界；输出可直接放进 `smart-summary.md` 的 Markdown 小节。",
        "忠实还原说话人的原意，而不是做外部事实裁判；主观或产品评价应归因给说话人。只有音频不清、来源冲突或模型新增内容才标待复核。",
    ]
    if failed_quality:
        lines.append("当前全局质量缺口：" + ", ".join(failed_quality))
    semantic_status = str((semantic_summary or {}).get("final_status") or "not_started")
    if semantic_status:
        lines.append(f"ASR/字幕语义纠错状态：{semantic_status}。未闭合候选只能写入待复核点；已闭合时优先采用 source-arbitrated transcript。")
    for citation in (citations or [])[:4]:
        lines.append('证据引用：{source}/{kind} / {time} / timeline={indexes} / {snippet}'.format(source=str(citation.get("source") or ""), kind=str(citation.get("chunk_kind") or ""), time=str(citation.get("time_range") or ""), indexes=",".join(str(value) for value in citation.get("timeline_indexes") or []), snippet=str(citation.get("snippet") or "")[:180]))
    semantic_items = evidence.get("semantic_correction_items") if isinstance(evidence.get("semantic_correction_items"), list) else []
    for item in semantic_items[:6]:
        status = str(item.get("correction_status") or "candidate")
        if status == "applied":
            corrected = str(item.get("corrected_text") or item.get("candidate_text") or item.get("suggested_text") or "")[:120]
            lines.append("已应用语义修正：{ctype} / {risk} / {time} / 修正后={corrected} / confidence={confidence}。最终总结只使用修正后表达，不要复述 ASR 错词、不要写成待复核。".format(ctype=str(item.get("correction_type") or ""), risk=str(item.get("risk_level") or ""), time=str(item.get("time_range") or ""), corrected=corrected, confidence=str(item.get("correction_confidence") or "")))
        else:
            lines.append("待复核语义候选：{ctype} / {risk} / {time} / 原文={original} / 建议={suggested} / reason={reason}".format(ctype=str(item.get("correction_type") or ""), risk=str(item.get("risk_level") or ""), time=str(item.get("time_range") or ""), original=str(item.get("original_text") or "")[:120], suggested=str(item.get("candidate_text") or item.get("suggested_text") or "")[:120], reason=str(item.get("reason") or "")[:120]))
    for key, label in (("summary_sentences", "候选摘要"), ("key_points", "关键观点"), ("actions", "动作"), ("visual_notes", "视觉/课件证据")):
        values = evidence.get(key) if isinstance(evidence.get(key), list) else []
        if values:
            lines.append(label + "：" + "；".join(str(value) for value in values[:4]))
    return "\n".join(lines)


def _todo_payload(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for section in result.get("sections") or []:
        if isinstance(section, dict) and section.get("status") != "ready":
            rows.append({
                "section_id": section.get("section_id"),
                "status": "todo",
                "title": section.get("title"),
                "time_range": f"{section.get('start_time')} - {section.get('end_time')}",
                "reasons": section.get("reasons") or [],
                "rewrite_prompt": section.get("rewrite_prompt") or "",
                "draft_markdown": "",
                "citations": section.get("citations") or [],
                "semantic_correction_items": section.get("semantic_correction_items") or [],
            })
    return {
        "schema": "video_knowledge_pipeline.smart_summary_section_todo.v1",
        "bundle_dir": str(root),
        "created_at": result.get("created_at"),
        "rows": rows,
        "operator_boundary": result.get("operator_boundary") or {},
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Smart Summary Section Workflow",
        "",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Quality: `{result.get('quality_status')}`",
        f"- Sections: `{result.get('section_count')}`",
        f"- Need rewrite: `{result.get('sections_needing_rewrite')}`",
        f"- ASR/subtitle semantic correction: `{(result.get('transcript_semantic_correction') or {}).get('final_status', 'not_started')}`",
        f"- Semantic attention candidates: `{result.get('semantic_attention_candidate_count', 0)}`",
        "",
        "This file is a local section-level planning layer. It does not call an LLM and does not overwrite `smart-summary.md`.",
        "",
        "| Section | Time | Status | Reasons | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for section in result.get("sections") or []:
        if not isinstance(section, dict):
            continue
        evidence = section.get("evidence") if isinstance(section.get("evidence"), dict) else {}
        evidence_bits = []
        for key in ("summary_sentences", "key_points", "actions", "visual_notes", "citations", "semantic_correction_items"):
            values = evidence.get(key) if isinstance(evidence.get(key), list) else []
            if values:
                evidence_bits.append(f"{key}:{len(values)}")
        lines.append(
            "| `{section}` | `{time}` | `{status}` | {reasons} | {evidence} |".format(
                section=str(section.get("section_id") or ""),
                time=f"{section.get('start_time')} - {section.get('end_time')}",
                status=str(section.get("status") or ""),
                reasons=_md(", ".join(str(value) for value in section.get("reasons") or [])),
                evidence=_md(", ".join(evidence_bits) or "-"),
            )
        )
    lines.extend(["", "## Rewrite Prompts", ""])
    for section in result.get("sections") or []:
        if not isinstance(section, dict) or section.get("status") == "ready":
            continue
        lines.extend([f"### {section.get('section_id')} {section.get('title')}", "", "```text", str(section.get("rewrite_prompt") or ""), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _section_action_items(root: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    editor_command = _section_editor_command(root)
    apply_command = _section_apply_command(root)
    for section in result.get("sections") or []:
        if not isinstance(section, dict) or section.get("status") == "ready":
            continue
        reasons = [str(value) for value in section.get("reasons") or [] if str(value).strip()]
        time_range = f"{section.get('start_time') or ''} - {section.get('end_time') or ''}".strip(" -")
        rows.append(
            {
                "id": section.get("section_id"),
                "index": section.get("section_id"),
                "chapter_index": section.get("chapter_index"),
                "title": section.get("title") or section.get("section_id"),
                "time_range": time_range,
                "reason": "section_revision_pending",
                "reasons": reasons,
                "detail": f"{section.get('title') or section.get('section_id')} / {time_range} / " + (", ".join(reasons) or "needs rewrite"),
                "citation_count": len(section.get("citations") or []),
                "rewrite_prompt_preview": _clip_text(str(section.get("rewrite_prompt") or ""), 360),
                "todo_json": "exports/smart-summary-section-todo.json",
                "workflow_markdown": "exports/smart-summary-section-workflow.md",
                "editor_html": "smart-summary-section-editor.html",
                "suggested_next_tool": "smart_summary_section_editor",
                "suggested_retry_command": editor_command,
                "suggested_apply_command": apply_command,
            }
        )
    return rows


def _section_editor_command(root: Path) -> str:
    return f".\\scripts\\video-knowledge.ps1 smart-summary-section-editor {_ps_quote(str(root))}"


def _section_apply_command(root: Path) -> str:
    return f".\\scripts\\video-knowledge.ps1 smart-summary-section-apply {_ps_quote(str(root))} --input-json {_ps_quote(str(root / 'smart-summary-section-revisions.json'))}"


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    failed_items = _section_action_items(root, result)
    status = "needs_input" if failed_items else "completed"
    retry_command = _section_editor_command(root) if failed_items else f".\\scripts\\video-knowledge.ps1 export-knowledge-note {_ps_quote(str(root))}"
    return register_bundle_run(
        root,
        run_type="smart_summary_section_workflow",
        run_id="smart-summary-section-workflow",
        status=status,
        title="Smart summary section workflow",
        summary=f"Prepared {result.get('section_count', 0)} smart-summary sections; {len(failed_items)} need rewrite.",
        inputs={"chapter_pack": str(root / "exports" / "smart-summary-chapters.json")},
        parameters={"quality_status": result.get("quality_status"), "quality_passed": result.get("quality_passed")},
        artifacts=[
            {"key": "section_workflow_json", "path": str(root / "exports" / "smart-summary-section-workflow.json")},
            {"key": "section_workflow_markdown", "path": str(root / "exports" / "smart-summary-section-workflow.md")},
            {"key": "section_todo", "path": str(root / "exports" / "smart-summary-section-todo.json")},
            {"key": "mcp_args", "path": str(root / "mcp-smart-summary-section-workflow.args.json")},
        ],
        failed_items=failed_items,
        retry_command=retry_command,
        next_actions=_next_actions(len(failed_items)),
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )


def _next_actions(failed_count: int) -> list[str]:
    if failed_count:
        return [
            "Open smart-summary-section-editor.html or exports/smart-summary-section-workflow.md and rewrite the listed sections with Codex or an approved LLM provider.",
            "Save revisions as smart-summary-section-revisions.json, then run smart-summary-section-apply.",
            "Run generate-smart-summary-with-codex and export-knowledge-note after revisions are installed.",
        ]
    return ["Run export-knowledge-note to install or refresh final smart-summary.md."]

def _bundle_path(root: Path, value: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return root / path

def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}



def _md(value: str) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")
