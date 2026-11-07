from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json
from .text_llm_gateway import extract_json_document

SCHEMA = "video_knowledge_pipeline.term_arbitration_codex.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.term_arbitration_codex_result.v1"
GLOSSARY_SCHEMA = "video_knowledge_pipeline.term_arbitration_glossary.v1"
VALIDATION_SCHEMA = "video_knowledge_pipeline.term_arbitration_codex_validation.v1"

CODEX_SEMANTIC_STRATEGY = {
    "strategy": "codex_substitute_for_online_text_llm",
    "purpose": "semantic_arbitration_for_terms_and_tool_names",
    "default_online_api_call": False,
    "codex_substitute_enabled": True,
    "rule_draft_is_not_semantic_confirmation": True,
    "semantic_inputs": [
        "asr_or_subtitle",
        "ocr_or_ebook",
        "structured_visual",
        "visual_understanding",
        "temporal_visual",
        "tagger",
        "metadata",
    ],
}



KNOWN_TOOL_ALIASES: dict[str, str] = {
    "a i": "AI",
    "c d p": "CDP",
    "m c p": "MCP",
    "n p c": "NPC",
    "play right": "Playwright",
    "playright": "Playwright",
    "playwright m c p": "Playwright MCP",
    "chrom": "Chrome",
    "chrome dive tooth": "Chrome DevTools",
    "chrome dev tools": "Chrome DevTools",
    "stay hand": "Stagehand",
    "stage hand": "Stagehand",
    "u i tars": "UI-TARS",
    "u i task": "UI-TARS",
    "ui tars": "UI-TARS",
    "ui task": "UI-TARS",
    "brow harness": "BrowserHarness",
    "browser honeys": "BrowserHarness",
    "browser harness": "BrowserHarness",
    "browserbase": "Browserbase",
    "codinging": "coding",
    "browser base": "Browserbase",
}


def build_term_arbitration_codex_pack(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    max_terms: int = 60,
    min_confidence: float = 0.88,
    accept_draft: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Build or import a Codex/LLM-ready terminology arbitration pack.

    This is deliberately provider-neutral. Without ``input_json`` it prepares a
    Codex review pack and a conservative local draft. With ``input_json`` it
    imports reviewed decisions and writes a glossary that can be reused by
    ``resolve-terms`` and ``transcript-source-arbitration``.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    timeline = _read_timeline(root)
    term_resolution = _read_optional_json(root / "term-resolution.json")
    context_sources = _context_sources(manifest, timeline)
    candidates = _candidate_terms(timeline, term_resolution, context_sources=context_sources, max_terms=max_terms)
    draft_decisions = _draft_decisions(candidates, min_confidence=min_confidence)
    imported = _load_import(input_json) if input_json else {}
    import_source = "codex_reviewed_import" if imported else ""
    imported_decisions = _normalise_decisions(imported.get("decisions") or imported.get("terms") or []) if imported else []
    if accept_draft and not imported_decisions:
        imported_decisions = _normalise_decisions([row for row in draft_decisions if _decision_accepted(row, min_confidence=min_confidence)])
        import_source = "codex_substitute_local_draft" if imported_decisions else ""
    accepted_decisions = [row for row in imported_decisions if _decision_accepted(row, min_confidence=min_confidence)]
    glossary = _glossary_from_decisions(accepted_decisions, source=import_source or "term_arbitration_codex")
    status = "imported" if imported_decisions else ("draft_ready" if draft_decisions else "no_candidates")
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "status": status,
        "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
        "candidate_count": len(candidates),
        "draft_decision_count": len(draft_decisions),
        "accepted_decision_count": len(accepted_decisions),
        "accept_draft": bool(accept_draft),
        "import_source": import_source,
        "max_terms": int(max_terms or 0),
        "min_confidence": float(min_confidence or 0),
        "candidates": candidates,
        "context_sources": context_sources,
        "draft_decisions": draft_decisions,
        "imported_decisions": imported_decisions,
        "glossary": glossary,
        "llm_semantic_arbitration": _llm_semantic_arbitration_summary(
            status=status,
            import_source=import_source,
            accept_draft=accept_draft,
            imported_decisions=imported_decisions,
            accepted_decisions=accepted_decisions,
        ),
        "artifacts": {
            "pack_json": "term-arbitration-codex-pack.json",
            "prompt_markdown": "term-arbitration-codex-prompt.md",
            "draft_json": "term-arbitration-codex-draft.json",
            "result_template_json": "term-arbitration-codex-result.template.json",
            "result_codex_markdown": "term-arbitration-codex-result.codex.md",
            "result_json": "term-arbitration-codex-result.json",
            "glossary_json": "term-arbitration-glossary.json",
            "report_markdown": "term-arbitration-codex.md",
            "mcp_args": "mcp-term-arbitration-codex.args.json",
            "mcp_validate_args": "mcp-term-arbitration-codex-validate.args.json",
            "mcp_closure_codex_args": "mcp-term-correction-closure-codex.args.json",
        },
        "operator_boundary": {
            "local_only_by_default": True,
            "codex_or_llm_review_required_for_ambiguous_terms": True,
            "no_cloud_call": True,
            "does_not_modify_raw_sources": True,
            "glossary_can_feed_transcript_source_arbitration": bool(accepted_decisions),
            "auto_accepts_only_high_confidence_draft": bool(accept_draft),
        },
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="term_arbitration_codex", timeout_seconds=1.0):
            write_json(root / "term-arbitration-codex-pack.json", result)
            (root / "term-arbitration-codex-prompt.md").write_text(_render_prompt(result), encoding="utf-8")
            write_json(root / "term-arbitration-codex-draft.json", {"schema": RESULT_SCHEMA, "draft_only": True, "decisions": draft_decisions})
            result_template = _result_template(result)
            write_json(root / "term-arbitration-codex-result.template.json", result_template)
            codex_response_stub = root / "term-arbitration-codex-result.codex.md"
            if not codex_response_stub.exists():
                codex_response_stub.write_text(_render_codex_response_stub(result, result_template), encoding="utf-8")
            if imported_decisions:
                write_json(root / "term-arbitration-codex-result.json", {"schema": RESULT_SCHEMA, "source": import_source or "codex_reviewed_import", "decisions": imported_decisions})
            write_json(root / "term-arbitration-glossary.json", glossary)
            (root / "term-arbitration-codex.md").write_text(_render_report(result), encoding="utf-8")
            write_json(
                root / "mcp-term-arbitration-codex.args.json",
                {
                    "bundle_dir": str(root),
                    "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
                    "max_terms": max_terms,
                    "min_confidence": min_confidence,
                    "write": True,
                },
            )

            write_json(
                root / "mcp-term-arbitration-codex-validate.args.json",
                {
                    "bundle_dir": str(root),
                    "input_json": str(codex_response_stub),
                    "min_confidence": min_confidence,
                    "write": True,
                },
            )
            write_json(
                root / "mcp-term-correction-closure-codex.args.json",
                {
                    "bundle_dir": str(root),
                    "accept_draft": False,
                    "input_json": str(codex_response_stub),
                    "max_terms": max_terms,
                    "term_min_confidence": min_confidence,
                    "transcript_min_confidence": 0.72,
                    "generate_codex_summary": True,
                    "write": True,
                },
            )
            manifest["term_arbitration_codex_pack_json"] = "term-arbitration-codex-pack.json"
            manifest["term_arbitration_codex_prompt_markdown"] = "term-arbitration-codex-prompt.md"
            manifest["term_arbitration_codex_draft_json"] = "term-arbitration-codex-draft.json"
            manifest["term_arbitration_codex_result_template_json"] = "term-arbitration-codex-result.template.json"
            manifest["term_arbitration_codex_result_codex_markdown"] = "term-arbitration-codex-result.codex.md"
            manifest["term_arbitration_glossary_json"] = "term-arbitration-glossary.json"
            if imported_decisions:
                manifest["term_arbitration_codex_result_json"] = "term-arbitration-codex-result.json"
            manifest["term_arbitration_codex_markdown"] = "term-arbitration-codex.md"
            manifest["mcp_term_arbitration_codex_args"] = "mcp-term-arbitration-codex.args.json"
            manifest["mcp_term_arbitration_codex_validate_args"] = "mcp-term-arbitration-codex-validate.args.json"
            manifest["mcp_term_correction_closure_codex_args"] = "mcp-term-correction-closure-codex.args.json"
            manifest["term_arbitration_codex_summary"] = {
                "status": status,
                "candidate_count": len(candidates),
                "draft_decision_count": len(draft_decisions),
                "accepted_decision_count": len(accepted_decisions),
                "accept_draft": bool(accept_draft),
                "import_source": import_source,
                "updated_at": result["updated_at"],
            }
            write_json(root / "manifest.json", manifest)
            result["run_registry"] = _register_run(root, result, write=True)
    else:
        result["run_registry"] = _register_run(root, result, write=False)
    return result


def validate_term_arbitration_codex_result(
    bundle_dir: str | Path,
    *,
    input_json: str | Path,
    min_confidence: float = 0.88,
    write: bool = True,
) -> dict[str, Any]:
    """Validate a Codex/LLM term arbitration response before importing it.

    This is a preview/preflight step. It parses JSON or Markdown containing JSON,
    applies the same acceptance rule used by ``build_term_arbitration_codex_pack``,
    and writes a human-readable report without changing the glossary or transcript.
    """

    root = Path(bundle_dir).expanduser().resolve()
    input_path = Path(input_json).expanduser().resolve()
    parsed_error = ""
    imported: dict[str, Any] = {}
    try:
        imported = _load_import(input_path)
    except Exception as exc:
        parsed_error = str(exc)
    decisions = _normalise_decisions(imported.get("decisions") or imported.get("terms") or []) if imported else []
    candidate_context = _candidate_validation_context(root)
    rows = [_validation_row(row, min_confidence=min_confidence, candidate_context=candidate_context) for row in decisions]
    accepted = [row for row in rows if row.get("accepted")]
    rejected = [row for row in rows if not row.get("accepted")]
    status = "ready_for_import" if accepted and not parsed_error else ("no_accepted_decisions" if rows else "invalid")
    result = {
        "schema": VALIDATION_SCHEMA,
        "bundle_dir": str(root),
        "input_json": str(input_path),
        "status": status,
        "ok": status == "ready_for_import",
        "parse_error": parsed_error,
        "min_confidence": float(min_confidence),
        "decision_count": len(rows),
        "accepted_decision_count": len(accepted),
        "rejected_decision_count": len(rejected),
        "accepted_decisions": accepted,
        "rejected_decisions": rejected,
        "next_actions": _validation_next_actions(status, root, input_path),
        "operator_boundary": {
            "preview_only": True,
            "no_glossary_write": True,
            "no_transcript_write": True,
            "no_cloud_call": True,
        },
        "updated_at": now_iso(),
        "write": bool(write),
    }
    if write:
        with bundle_write_lock(root, operation="term_arbitration_codex_validation", timeout_seconds=1.0):
            write_json(root / "term-arbitration-codex-validation.json", result)
            (root / "term-arbitration-codex-validation.md").write_text(_render_validation_report(result), encoding="utf-8")
            manifest = _read_manifest(root)
            manifest["term_arbitration_codex_validation_json"] = "term-arbitration-codex-validation.json"
            manifest["term_arbitration_codex_validation_markdown"] = "term-arbitration-codex-validation.md"
            manifest["mcp_term_arbitration_codex_validate_args"] = "mcp-term-arbitration-codex-validate.args.json"
            manifest["mcp_term_correction_closure_codex_args"] = "mcp-term-correction-closure-codex.args.json"
            manifest["term_arbitration_codex_validation_summary"] = {
                "status": status,
                "accepted_decision_count": len(accepted),
                "rejected_decision_count": len(rejected),
                "updated_at": result["updated_at"],
            }
            write_json(root / "manifest.json", manifest)
            write_json(
                root / "mcp-term-arbitration-codex-validate.args.json",
                {"bundle_dir": str(root), "input_json": str(input_path), "min_confidence": min_confidence, "write": True},
            )
    return result

def _candidate_terms(timeline: list[Any], term_resolution: dict[str, Any], *, context_sources: dict[str, Any] | None = None, max_terms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context_sources = context_sources if isinstance(context_sources, dict) else {}
    for term in term_resolution.get("terms") or []:
        if not isinstance(term, dict):
            continue
        raw_mentions = [str(value or "").strip() for value in term.get("raw_mentions") or [] if str(value or "").strip()]
        if not raw_mentions:
            continue
        confidence = _float(term.get("confidence"), 0.0)
        needs_review = bool(term.get("needs_human_review")) or confidence < 0.88 or len({_norm(value) for value in raw_mentions}) > 1
        if not needs_review and not any(_known_alias(value) for value in raw_mentions):
            continue
        rows.append(
            {
                "id": f"term-{len(rows) + 1}",
                "canonical_guess": str(term.get("canonical_term") or raw_mentions[0]),
                "raw_mentions": raw_mentions,
                "confidence": confidence,
                "risk_reasons": _risk_reasons(raw_mentions, confidence, term),
                "source_counts": term.get("source_counts") if isinstance(term.get("source_counts"), dict) else {},
                "evidence": _term_evidence_rows(timeline, context_sources, raw_mentions, term.get("evidence") or []),
            }
        )
    for alias, canonical in KNOWN_TOOL_ALIASES.items():
        contexts = _timeline_contexts(timeline, alias)
        if not contexts:
            continue
        if any(alias in [raw.lower() for raw in row.get("raw_mentions", [])] for row in rows):
            continue
        rows.append(
            {
                "id": f"term-{len(rows) + 1}",
                "canonical_guess": canonical,
                "raw_mentions": [alias],
                "confidence": 0.0,
                "risk_reasons": ["known_tool_alias_seen_in_transcript"],
                "source_counts": _context_source_counts(contexts),
                "evidence": contexts[:8],
            }
        )
    rows.sort(key=lambda row: (-len(row.get("risk_reasons") or []), -len(row.get("evidence") or []), str(row.get("canonical_guess") or "").lower()))
    if max_terms and max_terms > 0:
        return rows[: int(max_terms)]
    return rows


def _draft_decisions(candidates: list[dict[str, Any]], *, min_confidence: float) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in candidates:
        raw_mentions = row.get("raw_mentions") if isinstance(row.get("raw_mentions"), list) else []
        canonical = _canonical_from_aliases(raw_mentions) or str(row.get("canonical_guess") or "").strip()
        if not canonical:
            continue
        confidence = max(_float(row.get("confidence"), 0.0), 0.94 if _canonical_from_aliases(raw_mentions) else 0.0)
        action = "replace" if confidence >= min_confidence and any(_display_differs(raw, canonical) for raw in raw_mentions) else "review"
        decisions.append(
            {
                "candidate_id": row.get("id"),
                "canonical": canonical,
                "aliases": _unique([*raw_mentions, canonical]),
                "confidence": round(confidence, 3),
                "action": action,
                "rationale": "Known browser/AI tooling alias or source-supported canonical term; review if context contradicts this.",
                "evidence_indexes": [evidence.get("timeline_index") for evidence in row.get("evidence") or [] if isinstance(evidence, dict)][:8],
                "needs_human_review": action != "replace",
            }
        )
    return decisions


def _llm_semantic_arbitration_summary(
    *,
    status: str,
    import_source: str,
    accept_draft: bool,
    imported_decisions: list[dict[str, Any]],
    accepted_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    reviewed = bool(imported_decisions) and import_source == "codex_reviewed_import"
    draft_only = not imported_decisions and status == "draft_ready"
    accepted_rule_draft = bool(accept_draft) and import_source == "codex_substitute_local_draft"
    if reviewed:
        review_status = "codex_or_llm_reviewed_import"
    elif accepted_rule_draft:
        review_status = "rule_draft_accepted_not_llm_semantic"
    elif draft_only:
        review_status = "codex_review_pending"
    else:
        review_status = "no_candidates" if status == "no_candidates" else "not_ready"
    return {
        **CODEX_SEMANTIC_STRATEGY,
        "review_status": review_status,
        "reviewed_decision_count": len(imported_decisions),
        "accepted_decision_count": len(accepted_decisions),
        "accept_draft": bool(accept_draft),
        "import_source": import_source,
        "ready_for_transcript_arbitration": bool(accepted_decisions),
        "prompt_path": "term-arbitration-codex-prompt.md",
        "result_template_path": "term-arbitration-codex-result.template.json",
        "result_path": "term-arbitration-codex-result.json",
        "operator_guidance": "Use Codex as the temporary semantic reviewer. Import term-arbitration-codex-result.json before treating ambiguous tool names as confirmed.",
    }


def _result_template(result: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for candidate in result.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        rows.append(
            {
                "candidate_id": candidate.get("id"),
                "canonical": candidate.get("canonical_guess") or "",
                "aliases": candidate.get("raw_mentions") or [],
                "confidence": "0.00-1.00",
                "action": "replace|review",
                "rationale": "Explain the semantic reason using ASR/OCR/visual/tagger/metadata evidence.",
                "evidence_indexes": [row.get("timeline_index") for row in candidate.get("evidence") or [] if isinstance(row, dict)][:8],
                "needs_human_review": True,
            }
        )
    return {
        "schema": RESULT_SCHEMA,
        "source": "codex_reviewed_import",
        "reviewer": "codex_substitute_for_online_text_llm",
        "instructions": "Fill decisions after semantic review. Use action=replace only when the canonical term is strongly supported by context.",
        "decisions": rows,
    }

def _validation_row(row: dict[str, Any], *, min_confidence: float, candidate_context: dict[str, Any] | None = None) -> dict[str, Any]:
    confidence = _float(row.get("confidence"), 0.0)
    action = str(row.get("action") or "replace")
    needs_review = bool(row.get("needs_human_review"))
    candidate_id = str(row.get("candidate_id") or "").strip()
    evidence_indexes = row.get("evidence_indexes") if isinstance(row.get("evidence_indexes"), list) else []
    rejection_reasons: list[str] = []
    if action != "replace":
        rejection_reasons.append("action_not_replace")
    if confidence < min_confidence:
        rejection_reasons.append("confidence_below_minimum")
    if needs_review:
        rejection_reasons.append("needs_human_review")
    rejection_reasons.extend(_semantic_validation_rejection_reasons(row, candidate_context=candidate_context))
    accepted = _decision_accepted(row, min_confidence=min_confidence) and not rejection_reasons
    return {
        "candidate_id": candidate_id,
        "canonical": str(row.get("canonical") or ""),
        "aliases": [str(value or "") for value in row.get("aliases") or [] if str(value or "")],
        "confidence": confidence,
        "action": action,
        "needs_human_review": needs_review,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "rationale": str(row.get("rationale") or ""),
        "evidence_indexes": evidence_indexes,
    }


def _validation_next_actions(status: str, root: Path, input_path: Path) -> list[str]:
    if status == "ready_for_import":
        return [
            f"Run .\\scripts\\video-knowledge.ps1 term-correction-closure {_ps_quote(str(root))} --input-json {_ps_quote(str(input_path))}",
            "If any rejected_decisions remain, review their rejection_reasons before treating those aliases as confirmed.",
        ]
    if status == "no_accepted_decisions":
        return ["Review rejected_decisions, raise confidence only with stronger evidence, or keep these terms for human review."]
    return ["Fix the Codex/LLM response so it contains parseable JSON with a decisions array."]


def _render_validation_report(result: dict[str, Any]) -> str:
    lines = [
        "# Term Arbitration Codex Validation",
        "",
        f"- Status: `{result.get('status')}`",
        f"- OK: `{bool(result.get('ok'))}`",
        f"- Input: `{result.get('input_json', '')}`",
        f"- Min confidence: `{result.get('min_confidence')}`",
        f"- Decisions: `{result.get('decision_count')}`",
        f"- Accepted: `{result.get('accepted_decision_count')}`",
        f"- Rejected: `{result.get('rejected_decision_count')}`",
    ]
    if result.get("parse_error"):
        lines.extend(["", "## Parse Error", "", str(result.get("parse_error"))])
    lines.extend(["", "## Accepted Decisions", "", "| Canonical | Aliases | Confidence | Rationale |", "| --- | --- | ---: | --- |"])
    for row in result.get("accepted_decisions") or []:
        lines.append(f"| {_md(row.get('canonical'))} | {_md(', '.join(row.get('aliases') or []))} | {row.get('confidence')} | {_md(row.get('rationale'))} |")
    lines.extend(["", "## Rejected Decisions", "", "| Canonical | Confidence | Reasons |", "| --- | ---: | --- |"])
    for row in result.get("rejected_decisions") or []:
        lines.append(f"| {_md(row.get('canonical'))} | {row.get('confidence')} | {_md(', '.join(row.get('rejection_reasons') or []))} |")
    rejection_guidance = _validation_rejection_guidance(result.get("rejected_decisions") or [])
    if rejection_guidance:
        lines.extend(["", "## Rejection Guidance", "", "| Reason | How to fix |", "| --- | --- |"])
        for reason, guidance in rejection_guidance.items():
            lines.append(f"| `{reason}` | {_md(guidance)} |")
    actions = result.get("next_actions") if isinstance(result.get("next_actions"), list) else []
    if actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend([f"- {action}" for action in actions])
    return "\n".join(lines).rstrip() + "\n"


def _validation_rejection_guidance(rows: list[Any]) -> dict[str, str]:
    guidance = {
        "action_not_replace": "Keep action=review for uncertain terms; use action=replace only when the canonical term is strongly supported.",
        "confidence_below_minimum": "Raise confidence only after checking ASR/OCR/visual/tagger/metadata evidence.",
        "needs_human_review": "Set needs_human_review=false only after Codex or a human has confirmed the term.",
        "missing_semantic_rationale": "Add a rationale that names the semantic evidence, for example OCR shows the tool name and ASR discusses that tool.",
        "missing_evidence_indexes": "Add evidence_indexes copied from the candidate evidence timeline indexes in term-arbitration-codex-pack.json.",
        "missing_candidate_id": "Copy candidate_id from the candidate row in term-arbitration-codex-pack.json.",
        "unknown_candidate_id": "Use an existing candidate_id from term-arbitration-codex-pack.json; do not invent IDs.",
        "evidence_indexes_not_in_candidate_context": "Use timeline indexes that belong to the selected candidate evidence, or keep the decision as review.",
    }
    seen: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for reason in row.get("rejection_reasons") if isinstance(row.get("rejection_reasons"), list) else []:
            key = str(reason or "").strip()
            if key and key in guidance:
                seen[key] = guidance[key]
    return seen


def _glossary_from_decisions(decisions: list[dict[str, Any]], *, source: str = "term_arbitration_codex") -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    for row in decisions:
        canonical = str(row.get("canonical") or row.get("canonical_term") or "").strip()
        if not canonical:
            continue
        aliases = _unique([canonical, *[str(value or "").strip() for value in row.get("aliases") or row.get("raw_mentions") or []]])
        terms.append(
            {
                "canonical": canonical,
                "aliases": aliases,
                "confidence": _float(row.get("confidence"), 0.0),
                "source": source or "term_arbitration_codex",
                "review_required": bool(row.get("needs_human_review")) or str(row.get("action") or "") == "review",
                "rationale": str(row.get("rationale") or ""),
            }
        )
    return {"schema": GLOSSARY_SCHEMA, "terms": terms, "updated_at": now_iso()}


def _render_codex_response_stub(result: dict[str, Any], template: dict[str, Any]) -> str:
    lines = [
        "# Codex Term Arbitration Response",
        "",
        "> Fill this file by reviewing `term-arbitration-codex-prompt.md` and `term-arbitration-codex-pack.json`.",
        "> Keep parseable JSON inside the fenced block. `validate-term-arbitration-codex-result` will reject replacements without semantic rationale, candidate_id, and evidence_indexes.",
        "",
        "```json",
        json.dumps(template, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Use `action=replace` only when ASR/OCR/visual/tagger/metadata evidence strongly supports the canonical term.",
        "- Use `action=review` for ambiguous tools, homophones, brand names, or weak evidence.",
        f"- Candidate count: {int(result.get('candidate_count') or 0)}.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _render_prompt(result: dict[str, Any]) -> str:
    lines = [
        "# Codex Term Arbitration Prompt",
        "",
        "你是视频知识提取工具的术语仲裁器。当前由 Codex 临时代替在线文本大模型 API，基于 ASR、字幕、OCR/图文、视觉理解、上下文语义判断工具名和专有名词到底是什么。",
        "",
        "要求：",
        "- 不要只按拼写相似度判断，要结合上下文语义。",
        "- `term-arbitration-codex-draft.json` 只是规则草稿，不等于语义确认；真正确认必须来自本 prompt 复核后导入的 `term-arbitration-codex-result.json`。",
        "- 能高置信判断的，输出 `action=replace`。",
        "- 不能确定的，输出 `action=review`，不要强行改。",
        "- 输出 JSON，schema: `{ \"schema\": \"video_knowledge_pipeline.term_arbitration_codex_result.v1\", \"decisions\": [...] }`。",
        "- 每个 decision 字段：`candidate_id`, `canonical`, `aliases`, `confidence`, `action`, `rationale`, `evidence_indexes`, `needs_human_review`。",
        "",
        "## Video-Level Context Sources",
        "",
        "这些上下文用于语义判断，不是最终结论。重点看标题/简介、OCR/ebook、结构化视觉、多模态视觉、连续片段和打标器是否支持某个工具名。",
        "",
        "```json",
        json.dumps(result.get("context_sources") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Semantic Arbitration Strategy",
        "",
        "```json",
        json.dumps(result.get("llm_semantic_arbitration") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Candidates",
        "",
        "```json",
        json.dumps(result.get("candidates") or [], ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Term Arbitration Codex Pack",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Candidates: `{result.get('candidate_count')}`",
        f"- Draft decisions: `{result.get('draft_decision_count')}`",
        f"- Accepted imported decisions: `{result.get('accepted_decision_count')}`",
        f"- Accept draft: `{result.get('accept_draft', False)}`",
        f"- Import source: `{result.get('import_source', '')}`",
        f"- Semantic strategy: `{(result.get('llm_semantic_arbitration') or {}).get('strategy', '')}`",
        f"- Semantic review status: `{(result.get('llm_semantic_arbitration') or {}).get('review_status', '')}`",
        "",
        "## Artifacts",
        "",
        "| Artifact | Path |",
        "| --- | --- |",
    ]
    for key, path in (result.get("artifacts") or {}).items():
        lines.append(f"| `{key}` | `{path}` |")
    context_sources = result.get("context_sources") if isinstance(result.get("context_sources"), dict) else {}
    source_counts = context_sources.get("source_counts") if isinstance(context_sources.get("source_counts"), dict) else {}
    if source_counts:
        lines.extend(["", "## Evidence Source Counts", "", "| Source | Count |", "| --- | ---: |"])
        for source, count in sorted(source_counts.items()):
            lines.append(f"| `{source}` | {count} |")
    lines.extend(["", "## Candidate Terms", "", "| ID | Guess | Raw mentions | Risk | Evidence |", "| --- | --- | --- | --- | ---: |"])
    for row in result.get("candidates") or []:
        lines.append(
            f"| `{row.get('id')}` | {_md(row.get('canonical_guess'))} | {_md(', '.join(row.get('raw_mentions') or []))} | {_md(', '.join(row.get('risk_reasons') or []))} | {len(row.get('evidence') or [])} |"
        )
    lines.extend(["", "## Draft Decisions", "", "| Candidate | Canonical | Aliases | Confidence | Action |", "| --- | --- | --- | ---: | --- |"])
    for row in result.get("draft_decisions") or []:
        lines.append(
            f"| `{row.get('candidate_id')}` | {_md(row.get('canonical'))} | {_md(', '.join(row.get('aliases') or []))} | {row.get('confidence')} | `{row.get('action')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    status = "completed" if result.get("status") == "imported" else ("needs_input" if result.get("candidate_count") else "completed")
    failed_items = []
    if result.get("status") == "draft_ready":
        failed_items.append(
            {
                "id": "codex_review",
                "reason": "codex_review_required",
                "detail": "Review term-arbitration-codex-prompt.md and import term-arbitration-codex-result.json before treating ambiguous terms as final.",
            }
        )
    return register_bundle_run(
        root,
        run_type="term_arbitration_codex",
        run_id="term-arbitration-codex",
        status=status,
        title="Codex terminology arbitration",
        summary=f"{result.get('candidate_count')} candidate terms; status={result.get('status')}.",
        inputs={"bundle_dir": str(root), "input_json": result.get("input_json", "")},
        parameters={"max_terms": result.get("max_terms"), "min_confidence": result.get("min_confidence"), "accept_draft": bool(result.get("accept_draft"))},
        artifacts=[
            {"key": "pack_json", "path": root / "term-arbitration-codex-pack.json"},
            {"key": "prompt", "path": root / "term-arbitration-codex-prompt.md"},
            {"key": "draft", "path": root / "term-arbitration-codex-draft.json"},
            {"key": "glossary", "path": root / "term-arbitration-glossary.json"},
            {"key": "report", "path": root / "term-arbitration-codex.md"},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 term-arbitration-codex {_ps_quote(str(root))}",
        next_actions=_next_actions(result),
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )


def _next_actions(result: dict[str, Any]) -> list[str]:
    if result.get("status") == "draft_ready":
        return [
            "Open term-arbitration-codex-prompt.md, let Codex review ambiguous tool names, save the JSON or full Markdown response, then rerun with --input-json.",
            "Rerun transcript-source-arbitration with --glossary-json term-arbitration-glossary.json after importing reviewed decisions.",
        ]
    if result.get("status") == "imported":
        return ["Run transcript-source-arbitration with --glossary-json term-arbitration-glossary.json, then regenerate smart-summary."]
    return []


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("manifest.json must be a JSON object")
    return data


def _read_timeline(root: Path) -> list[Any]:
    path = root / "timeline.json"
    if not path.exists():
        return []
    data = read_json(path)
    return data if isinstance(data, list) else []


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _load_import(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path).expanduser().resolve()
    try:
        data = read_json(target)
    except Exception:
        raw = target.read_text(encoding="utf-8-sig")
        data = extract_json_document(raw, require_object=True)
    if not isinstance(data, dict):
        raise ValueError("input_json must contain a JSON object or a Codex/LLM response containing a JSON object")
    return data


def _normalise_decisions(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        canonical = str(row.get("canonical") or row.get("canonical_term") or row.get("term") or "").strip()
        aliases = row.get("aliases") if isinstance(row.get("aliases"), list) else row.get("raw_mentions")
        aliases = [str(value or "").strip() for value in aliases or [] if str(value or "").strip()]
        if canonical:
            out.append({**row, "canonical": canonical, "aliases": _unique([canonical, *aliases])})
    return out


def _decision_accepted(row: dict[str, Any], *, min_confidence: float) -> bool:
    return (
        str(row.get("action") or "replace") == "replace"
        and _float(row.get("confidence"), 0.0) >= min_confidence
        and not bool(row.get("needs_human_review"))
        and bool(str(row.get("rationale") or "").strip())
        and bool(row.get("evidence_indexes") if isinstance(row.get("evidence_indexes"), list) else [])
    )


def _candidate_validation_context(root: Path) -> dict[str, Any]:
    pack_path = root / "term-arbitration-codex-pack.json"
    if not pack_path.exists():
        return {"candidate_ids": set(), "evidence_indexes_by_candidate": {}}
    try:
        pack = read_json(pack_path)
    except Exception:
        return {"candidate_ids": set(), "evidence_indexes_by_candidate": {}}
    candidates = pack.get("candidates") if isinstance(pack, dict) and isinstance(pack.get("candidates"), list) else []
    candidate_ids: set[str] = set()
    evidence_indexes_by_candidate: dict[str, set[str]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id:
            continue
        candidate_ids.add(candidate_id)
        indexes: set[str] = set()
        for evidence in candidate.get("evidence") if isinstance(candidate.get("evidence"), list) else []:
            if isinstance(evidence, dict) and str(evidence.get("timeline_index") or "").strip():
                indexes.add(str(evidence.get("timeline_index")).strip())
        evidence_indexes_by_candidate[candidate_id] = indexes
    return {"candidate_ids": candidate_ids, "evidence_indexes_by_candidate": evidence_indexes_by_candidate}


def _semantic_validation_rejection_reasons(row: dict[str, Any], *, candidate_context: dict[str, Any] | None = None) -> list[str]:
    if str(row.get("action") or "replace") != "replace":
        return []
    reasons: list[str] = []
    rationale = str(row.get("rationale") or "").strip()
    if len(rationale) < 8:
        reasons.append("missing_semantic_rationale")
    evidence_indexes = row.get("evidence_indexes") if isinstance(row.get("evidence_indexes"), list) else []
    if not evidence_indexes:
        reasons.append("missing_evidence_indexes")
    context = candidate_context if isinstance(candidate_context, dict) else {}
    candidate_ids = context.get("candidate_ids") if isinstance(context.get("candidate_ids"), set) else set()
    candidate_id = str(row.get("candidate_id") or "").strip()
    if candidate_ids:
        if not candidate_id:
            reasons.append("missing_candidate_id")
        elif candidate_id not in candidate_ids:
            reasons.append("unknown_candidate_id")
    evidence_by_candidate = context.get("evidence_indexes_by_candidate") if isinstance(context.get("evidence_indexes_by_candidate"), dict) else {}
    expected_indexes = evidence_by_candidate.get(candidate_id) if candidate_id else set()
    if expected_indexes and evidence_indexes:
        actual = {str(value).strip() for value in evidence_indexes if str(value).strip()}
        if actual.isdisjoint(expected_indexes):
            reasons.append("evidence_indexes_not_in_candidate_context")
    return reasons

def _risk_reasons(raw_mentions: list[str], confidence: float, term: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if confidence < 0.88:
        reasons.append("low_confidence")
    if len({_norm(value) for value in raw_mentions}) > 1:
        reasons.append("conflicting_raw_mentions")
    if any(_known_alias(value) for value in raw_mentions):
        reasons.append("known_tool_alias")
    source_counts = term.get("source_counts") if isinstance(term.get("source_counts"), dict) else {}
    if source_counts.get("asr") and not any(source_counts.get(key) for key in ("ocr", "structured_visual", "metadata", "visual_understanding")):
        reasons.append("asr_only")
    return reasons or ["needs_semantic_confirmation"]


def _context_sources(manifest: dict[str, Any], timeline: list[Any]) -> dict[str, Any]:
    metadata_keys = (
        "title",
        "source_title",
        "description",
        "source_description",
        "source_url",
        "platform",
        "channel",
        "author",
        "uploader",
        "media_path",
        "expected_content_type",
        "notes",
    )
    metadata = {key: _compact_text(manifest.get(key), 600) for key in metadata_keys if _compact_text(manifest.get(key), 600)}
    source_counts: dict[str, int] = {}
    visual_rows: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        sources = _evidence_sources_for_item(item)
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        if any(source in sources for source in ("ocr_or_ebook", "structured_visual", "visual_understanding", "temporal_visual", "tagger")):
            visual_rows.append(
                {
                    "timeline_index": item.get("index") or position,
                    "start": item.get("start", ""),
                    "end": item.get("end", ""),
                    "sources": sources,
                    "context": _compact_text(_timeline_evidence_text(item), 900),
                }
            )
    return {
        "metadata": metadata,
        "source_counts": source_counts,
        "visual_context_rows": visual_rows[:24],
        "timeline_items": len([item for item in timeline if isinstance(item, dict)]),
    }


def _term_evidence_rows(timeline: list[Any], context_sources: dict[str, Any], raw_mentions: list[str], rows: list[Any]) -> list[dict[str, Any]]:
    out = _evidence_rows(rows)
    for mention in raw_mentions:
        out.extend(_timeline_contexts(timeline, mention)[:8])
        out.extend(_metadata_contexts(context_sources, mention))
    return _unique_evidence(out)[:12]


def _evidence_rows(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:12]:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "").strip()
        context = _compact_text(row.get("context") or row.get("text") or row.get("value"), 700)
        out.append(
            {
                "source": source,
                "timeline_index": row.get("timeline_index", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
                "mention": row.get("mention", ""),
                "context": context,
                "source_channels": [source] if source else [],
            }
        )
    return out


def _timeline_contexts(timeline: list[Any], alias: str) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    pattern = re.compile(re.escape(alias), flags=re.IGNORECASE)
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        text = _timeline_evidence_text(item)
        if not pattern.search(text):
            continue
        sources = _evidence_sources_for_item(item)
        contexts.append(
            {
                "source": "timeline_semantic_context",
                "source_channels": sources,
                "timeline_index": item.get("index") or position,
                "start": item.get("start", ""),
                "end": item.get("end", ""),
                "mention": alias,
                "context": _context(text, alias, radius=180),
            }
        )
    return contexts


def _metadata_contexts(context_sources: dict[str, Any], mention: str) -> list[dict[str, Any]]:
    metadata = context_sources.get("metadata") if isinstance(context_sources.get("metadata"), dict) else {}
    text = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    if not text or not re.search(re.escape(mention), text, flags=re.IGNORECASE):
        return []
    return [
        {
            "source": "metadata",
            "source_channels": ["metadata"],
            "timeline_index": "",
            "start": "",
            "end": "",
            "mention": mention,
            "context": _context(text, mention, radius=220),
        }
    ]


def _timeline_evidence_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "transcript",
        "corrected_transcript",
        "subtitle",
        "visual_text",
        "structured_visual",
        "visual_understanding",
        "temporal_visual_understanding",
        "tagger_visual_summary",
        "tagger_labels",
        "material_types",
        "quality_issues",
        "review_reason",
    ):
        value = _compact_text(item.get(key), 900)
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def _evidence_sources_for_item(item: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    if item.get("transcript") or item.get("corrected_transcript") or item.get("subtitle"):
        sources.append("asr_or_subtitle")
    if item.get("visual_text"):
        sources.append("ocr_or_ebook")
    if item.get("structured_visual"):
        sources.append("structured_visual")
    if item.get("visual_understanding"):
        sources.append("visual_understanding")
    if item.get("temporal_visual_understanding"):
        sources.append("temporal_visual")
    if item.get("tagger_visual_summary") or item.get("tagger_labels"):
        sources.append("tagger")
    if item.get("material_types"):
        sources.append("material_type")
    return _unique(sources)


def _context_source_counts(contexts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in contexts:
        channels = row.get("source_channels") if isinstance(row.get("source_channels"), list) else [row.get("source")]
        for channel in channels:
            key = str(channel or "").strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts


def _unique_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        context = _compact_text(row.get("context"), 700)
        key = (str(row.get("source") or ""), str(row.get("timeline_index") or ""), context[:160].lower())
        if not context or key in seen:
            continue
        seen.add(key)
        out.append({**row, "context": context})
    return out


def _compact_text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit > 0 and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _canonical_from_aliases(raw_mentions: list[Any]) -> str:
    for raw in raw_mentions:
        canonical = _known_alias(str(raw or ""))
        if canonical:
            return canonical
    return ""


def _known_alias(value: str) -> str:
    norm = _norm(value)
    for alias, canonical in KNOWN_TOOL_ALIASES.items():
        if _norm(alias) == norm:
            return canonical
    return ""


def _context(text: str, mention: str, radius: int = 90) -> str:
    lower = text.lower()
    pos = lower.find(mention.lower())
    if pos < 0:
        return text[: radius * 2].replace("\n", " ")
    return text[max(0, pos - radius) : pos + len(mention) + radius].replace("\n", " ")



def _display_differs(raw: Any, canonical: str) -> bool:
    return str(raw or "").strip().casefold() != str(canonical or "").strip().casefold()

def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _unique(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _md(value: Any) -> str:
    return str(value or "-").replace("\n", " ").replace("|", "\\|")
