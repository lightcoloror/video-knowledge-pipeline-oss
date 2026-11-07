from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .transcript_semantic_correction import build_transcript_semantic_correction_pack

SCHEMA = "video_knowledge_pipeline.evidence_conflict_index.v1"
CONFLICT_REASONS = {
    "ordinary_word_conflict_between_asr_and_visual_text",
    "ordinary_word_conflict_between_asr_and_subtitle",
    "ordinary_word_conflict_between_asr_and_tagger",
    "ordinary_word_conflict_between_dual_asr",
    "visual_text_differs_from_transcript",
    "platform_subtitle_differs_from_transcript",
    "tagger_text_differs_from_transcript",
}
STRONG_SOURCE_TYPES = {
    "platform_subtitle",
    "embedded_subtitle",
    "secondary_asr",
    "subtitle",
    "ocr",
    "structured_visual",
    "visual_text",
    "visual_understanding",
    "temporal_visual",
    "tagger",
    "qinglong_tagger",
    "page_metadata",
    "web_context",
    "glossary",
    "term_dictionary",
    "human_note",
}


def build_evidence_conflict_index(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    pack_data: dict[str, Any] | None = None,
    limit: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Build a narrow arbitration queue from real evidence conflicts only.

    The semantic correction pack may contain heuristic risk markers. This index is
    stricter: it is meant to decide what is worth LLM arbitration because ASR and
    another source disagree, or because a strong non-ASR source proposes a concrete
    replacement.
    """

    root = Path(bundle_dir).expanduser().resolve()
    pack_path = _resolve_pack(root, input_json)
    if pack_data is not None:
        if not isinstance(pack_data, dict):
            raise ValueError("semantic correction pack data must be a JSON object")
        pack = pack_data
    elif pack_path.exists():
        pack = read_json(pack_path)
        if not isinstance(pack, dict):
            pack = {}
    else:
        pack = build_transcript_semantic_correction_pack(root, limit=0, write=write)
        pack_path = root / "transcript-semantic-correction-pack.json"
    candidates = [row for row in pack.get("candidates") or [] if isinstance(row, dict)]
    conflict_rows = [_conflict_row(row) for row in candidates]
    conflict_rows = [row for row in conflict_rows if row["include_in_llm_arbitration"]]
    conflict_rows.sort(key=lambda row: (-int(row.get("priority_score") or 0), float(row.get("start") or 0), str(row.get("candidate_id") or "")))
    if limit and limit > 0:
        conflict_rows = conflict_rows[: int(limit)]
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "pack_json": str(pack_path),
        "status": "conflicts_ready" if conflict_rows else "no_real_conflicts",
        "ok": True,
        "candidate_count": len(candidates),
        "conflict_count": len(conflict_rows),
        "llm_arbitration_count": len(conflict_rows),
        "limit": int(limit or 0),
        "summary": _summary(candidates, conflict_rows),
        "conflicts": conflict_rows,
        "artifacts": {
            "json": str(root / "evidence-conflict-index.json"),
            "markdown": str(root / "evidence-conflict-index.md"),
            "mcp_args": str(root / "mcp-evidence-conflict-index.args.json"),
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "does_not_apply_corrections": True,
            "llm_should_only_review_conflicts": True,
            "heuristic_risks_without_external_evidence_are_not_arbitration_items": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="evidence_conflict_index", timeout_seconds=1.0):
            write_json(root / "evidence-conflict-index.json", result)
            (root / "evidence-conflict-index.md").write_text(_render_markdown(result), encoding="utf-8")
            write_json(root / "mcp-evidence-conflict-index.args.json", {"bundle_dir": str(root), "input_json": str(pack_path), "limit": int(limit or 0), "write": True})
            manifest = _read_manifest(root)
            manifest["evidence_conflict_index_json"] = "evidence-conflict-index.json"
            manifest["evidence_conflict_index_markdown"] = "evidence-conflict-index.md"
            manifest["mcp_evidence_conflict_index_args"] = "mcp-evidence-conflict-index.args.json"
            manifest["evidence_conflict_index_summary"] = {"status": result["status"], "conflict_count": len(conflict_rows), "updated_at": result["updated_at"]}
            write_json(root / "manifest.json", manifest)
    return result


def _resolve_pack(root: Path, input_json: str | Path | None) -> Path:
    if input_json:
        path = Path(input_json).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    manifest = _read_manifest(root)
    value = str(manifest.get("transcript_semantic_correction_pack_json") or "transcript-semantic-correction-pack.json")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _conflict_row(candidate: dict[str, Any]) -> dict[str, Any]:
    summary = candidate.get("source_support_summary") if isinstance(candidate.get("source_support_summary"), dict) else {}
    source_types = [str(item) for item in candidate.get("evidence_source_types") or [] if str(item)]
    reason = str(candidate.get("reason") or "")
    original = str(candidate.get("original_text") or "").strip()
    suggested = str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").strip()
    has_delta = bool(suggested and _compact(suggested) != _compact(original))
    support_candidate = [str(item) for item in summary.get("supports_candidate") or [] if str(item)]
    support_original = [str(item) for item in summary.get("supports_original") or [] if str(item)]
    strong_sources = sorted({item for item in source_types if item in STRONG_SOURCE_TYPES})
    real_conflict = bool(candidate.get("has_conflict") or summary.get("has_source_conflict") or reason in CONFLICT_REASONS)
    strong_external_delta = bool(has_delta and strong_sources and set(strong_sources) - {"asr_or_subtitle"})
    include = bool(has_delta and (real_conflict or strong_external_delta or bool(candidate.get("llm_review_eligible") and strong_sources)))
    classification = _classification(reason, source_types, support_candidate, support_original)
    priority = _priority(candidate, real_conflict=real_conflict, strong_sources=strong_sources, classification=classification)
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "segment_index": candidate.get("segment_index"),
        "start": candidate.get("start"),
        "end": candidate.get("end"),
        "time_range": candidate.get("time_range", ""),
        "classification": classification,
        "priority_score": priority,
        "include_in_llm_arbitration": include,
        "reason": reason,
        "risk_level": candidate.get("risk_level", ""),
        "original_text": original,
        "candidate_text": suggested,
        "context_text": candidate.get("context_text", ""),
        "evidence_source_types": source_types,
        "supports_candidate": support_candidate,
        "supports_original": support_original,
        "strong_external_sources": strong_sources,
        "source_support_summary": summary,
        "evidence": candidate.get("evidence") or [],
        "llm_review_eligible": bool(candidate.get("llm_review_eligible")),
        "llm_review_priority_class": candidate.get("llm_review_priority_class", ""),
        "llm_review_defer_reason": "" if include else str(candidate.get("llm_review_defer_reason") or "no_real_external_conflict"),
    }


def _classification(reason: str, source_types: list[str], support_candidate: list[str], support_original: list[str]) -> str:
    tokens = [item for item in source_types + support_candidate + support_original if item != "asr_or_subtitle"] + [reason]
    joined = " ".join(tokens)
    if "ocr" in joined or "visual_text" in joined or "structured_visual" in joined:
        return "screen_text_conflict"
    if "visual_understanding" in joined or "temporal_visual" in joined:
        return "multimodal_conflict"
    if "tagger" in joined or "qinglong" in joined:
        return "tagger_conflict"
    if "secondary_asr" in joined:
        return "dual_asr_conflict"
    if "platform_subtitle" in joined or "embedded_subtitle" in joined or "subtitle" in joined:
        return "subtitle_conflict"
    if "page_metadata" in joined or "web_context" in joined:
        return "web_context_conflict"
    if "glossary" in joined or "term_dictionary" in joined:
        return "term_dictionary_conflict"
    return "source_conflict"


def _priority(candidate: dict[str, Any], *, real_conflict: bool, strong_sources: list[str], classification: str) -> int:
    score = 20
    if real_conflict:
        score += 40
    score += min(30, len(strong_sources) * 10)
    if candidate.get("risk_level") == "high":
        score += 15
    if classification in {"screen_text_conflict", "multimodal_conflict", "tagger_conflict"}:
        score += 10
    if candidate.get("llm_review_eligible"):
        score += 10
    return score


def _summary(candidates: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(str(row.get("reason") or "") for row in conflicts)
    class_counts = Counter(str(row.get("classification") or "") for row in conflicts)
    source_counts: Counter[str] = Counter()
    for row in conflicts:
        for source in row.get("evidence_source_types") or []:
            source_counts[str(source)] += 1
    deferred = [row for row in candidates if not _conflict_row(row)["include_in_llm_arbitration"]]
    return {
        "total_semantic_candidates": len(candidates),
        "real_conflicts": len(conflicts),
        "deferred_heuristic_or_low_evidence": len(deferred),
        "classification_counts": dict(class_counts),
        "reason_counts": dict(reason_counts),
        "source_type_counts": dict(source_counts),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Evidence Conflict Index",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Status: `{result.get('status', '')}`",
        f"- Conflicts for LLM arbitration: `{result.get('conflict_count', 0)}`",
        f"- Updated: `{result.get('updated_at', '')}`",
        "",
        "## Rule",
        "",
        "Only ASR/subtitle/OCR/tagger/visual/web/glossary disagreements with a concrete text delta are sent to LLM arbitration. Pure risk markers are deferred.",
        "",
        "## Conflicts",
        "",
        "| Time | Type | Original | Candidate | Sources | Priority |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("conflicts") or []:
        sources = ", ".join(str(item) for item in (row.get("evidence_source_types") or [])[:8])
        lines.append(f"| {row.get('time_range', '')} | `{row.get('classification', '')}` | {str(row.get('original_text', ''))[:80]} | {str(row.get('candidate_text', ''))[:80]} | {sources} | {row.get('priority_score', 0)} |")
    return "\n".join(lines).rstrip() + "\n"


def _compact(value: str) -> str:
    return "".join(str(value or "").lower().split())


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}
