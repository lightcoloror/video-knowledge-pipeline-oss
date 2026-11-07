from __future__ import annotations

from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .storage import read_json


def term_correction_status(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    glossary = _read_object(root / "term-arbitration-glossary.json")
    impact = _read_object(root / "term-correction-impact-report.json")
    closure = _read_object(root / "term-correction-closure.json")
    validation = _read_object(root / "term-arbitration-codex-validation.json")
    quality = _read_object(root / "exports" / "smart-summary-quality.json")
    terms = glossary.get("terms") if isinstance(glossary.get("terms"), list) else []
    accepted_terms = [row for row in terms if isinstance(row, dict) and not bool(row.get("review_required"))]
    glossary_exists = bool(terms)
    source_arbitrated_exists = (root / "source-arbitrated-transcript.json").exists()
    impact_exists = bool(impact)
    final_alias_total = int(impact.get("final_export_alias_total") or 0) if impact_exists else 0
    quality_passed = bool(quality.get("passed")) if quality else False
    closure_status = str(closure.get("status") or "").strip()
    validation_status = str(validation.get("status") or closure.get("term_validation_status") or "").strip()
    validation_rejected = validation.get("rejected_decisions") if isinstance(validation.get("rejected_decisions"), list) else []
    validation_failed = validation_status in {"invalid", "no_accepted_decisions"} or str(closure.get("semantic_review_status") or "") == "codex_validation_failed"
    if validation_failed:
        status = "needs_codex_term_validation"
    elif closure_status:
        status = closure_status
    elif not glossary_exists:
        status = "needs_term_arbitration"
    elif not source_arbitrated_exists:
        status = "needs_transcript_arbitration"
    elif not impact_exists:
        status = "needs_impact_check"
    elif not bool(impact.get("ok")):
        status = "needs_retry"
    elif not quality_passed:
        status = "needs_smart_summary_fix"
    else:
        status = "ready"
    next_action_by_status = {
        "needs_term_arbitration": "term_arbitration_codex",
        "needs_transcript_arbitration": "transcript_source_arbitration",
        "needs_impact_check": "term_correction_impact",
        "needs_retry": "term_correction_closure",
        "needs_smart_summary_fix": "term_correction_closure",
        "needs_term_review": "term_arbitration_codex",
        "needs_codex_term_validation": "term_arbitration_codex_validate",
    }
    codex_substitute = _codex_substitute(root, next_action_by_status.get(status, ""))
    return {
        "status": status,
        "closure_status": closure_status,
        "term_validation_status": validation_status,
        "term_validation_ok": bool(validation.get("ok")) if validation else False,
        "accepted_validation_decisions": int(validation.get("accepted_decision_count") or closure.get("accepted_validation_decisions") or 0),
        "rejected_validation_decisions": int(validation.get("rejected_decision_count") or closure.get("rejected_validation_decisions") or 0),
        "validation_rejection_reasons": _validation_rejection_reasons(validation_rejected),
        "validation_rejected_decisions": _validation_rejected_decision_rows(validation_rejected),
        "glossary_exists": glossary_exists,
        "accepted_term_count": len(accepted_terms),
        "accepted_terms": _accepted_term_rows(accepted_terms),
        "source_arbitrated_transcript_exists": source_arbitrated_exists,
        "impact_status": str(impact.get("status") or "") if impact_exists else "missing",
        "final_export_alias_total": final_alias_total,
        "smart_summary_quality_passed": quality_passed,
        "next_action_key": next_action_by_status.get(status, ""),
        "codex_substitute": codex_substitute,
        "artifacts": {
            "term_pack_json": str(root / "term-arbitration-codex-pack.json"),
            "term_prompt_markdown": str(root / "term-arbitration-codex-prompt.md"),
            "term_result_template_json": str(root / "term-arbitration-codex-result.template.json"),
            "term_result_codex_markdown": str(root / "term-arbitration-codex-result.codex.md"),
            "term_result_json": str(root / "term-arbitration-codex-result.json"),
            "glossary_json": str(root / "term-arbitration-glossary.json"),
            "source_arbitrated_transcript_json": str(root / "source-arbitrated-transcript.json"),
            "impact_report_json": str(root / "term-correction-impact-report.json"),
            "impact_report_markdown": str(root / "term-correction-impact-report.md"),
            "closure_json": str(root / "term-correction-closure.json"),
            "closure_markdown": str(root / "term-correction-closure.md"),
            "term_validation_json": str(root / "term-arbitration-codex-validation.json"),
            "term_validation_markdown": str(root / "term-arbitration-codex-validation.md"),
        },
    }



def _codex_substitute(root: Path, next_action_key: str) -> dict[str, Any]:
    bundle = _ps_quote(str(root))
    result_path = _ps_quote(str(root / "term-arbitration-codex-result.codex.md"))
    return {
        "mode": "codex_substitute_for_online_text_llm",
        "purpose": "semantic terminology and tool-name arbitration",
        "online_llm_api_required": False,
        "manual_review_required_for_ambiguous_terms": True,
        "prompt_markdown": str(root / "term-arbitration-codex-prompt.md"),
        "context_pack_json": str(root / "term-arbitration-codex-pack.json"),
        "result_template_json": str(root / "term-arbitration-codex-result.template.json"),
        "suggested_result_markdown": str(root / "term-arbitration-codex-result.codex.md"),
        "validation_required": True,
        "next_action_key": next_action_key,
        "commands": {
            "build_pack": f".\\scripts\\video-knowledge.ps1 term-arbitration-codex {bundle}",
            "validate_result": f".\\scripts\\video-knowledge.ps1 validate-term-arbitration-codex-result {bundle} --input-json {result_path}",
            "import_and_close": f".\\scripts\\video-knowledge.ps1 term-correction-closure {bundle} --input-json {result_path}",
            "impact_check": f".\\scripts\\video-knowledge.ps1 term-correction-impact-report {bundle}",
        },
        "acceptance_rule": "Only decisions with semantic rationale, candidate_id, and evidence_indexes may enter the glossary.",
    }



def _accepted_term_rows(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for term in terms[:80]:
        canonical = str(term.get("canonical_term") or term.get("canonical") or term.get("term") or "").strip()
        if not canonical:
            continue
        aliases = term.get("aliases") or term.get("raw_mentions") or []
        if not isinstance(aliases, list):
            aliases = []
        rows.append(
            {
                "canonical_term": canonical,
                "aliases": [str(value) for value in aliases if str(value).strip()][:12],
                "confidence": term.get("confidence"),
            }
        )
    return rows


def _validation_rejection_reasons(rows: list[Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        reasons = row.get("rejection_reasons") if isinstance(row.get("rejection_reasons"), list) else []
        for reason in reasons:
            key = str(reason or "").strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return [{"reason": reason, "count": count} for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _validation_rejected_decision_rows(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "canonical": str(row.get("canonical") or ""),
                "confidence": row.get("confidence"),
                "rejection_reasons": [str(value) for value in row.get("rejection_reasons") or [] if str(value).strip()],
                "rationale": str(row.get("rationale") or ""),
                "evidence_indexes": row.get("evidence_indexes") if isinstance(row.get("evidence_indexes"), list) else [],
            }
        )
    return out
def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}
