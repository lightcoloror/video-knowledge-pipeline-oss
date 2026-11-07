from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage import read_json
from .term_correction_status import term_correction_status
from .transcript_semantic_correction import transcript_semantic_correction_status


REQUIRED_MATERIAL_CARD_FIELDS = [
    "material_id",
    "source_path",
    "source_type",
    "source_fact_status",
    "evidence_tier",
    "privacy_level",
    "desensitized",
    "compliance_risk",
    "fact_check_status",
    "target_layer",
    "publish_surface",
    "content_stage",
    "cta_type",
    "crm_followup_needed",
    "owner_thread",
    "next_action",
    "blocked_reason",
]


def content_asset_status(bundle_dir: str | Path, *, write: bool = False) -> dict[str, Any]:
    """Report whether a bundle has safe content-asset exports.

    The status is read-only. The write flag is accepted for MCP/CLI symmetry
    with other status tools, but this function never mutates bundle files.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    export_summary_path = root / "exports" / "export-summary.json"
    if not manifest_path.exists():
        return {
            "schema": "video_knowledge_pipeline.content_asset_status.v1",
            "ok": False,
            "status": "bundle_missing",
            "bundle_dir": str(root),
            "write": write,
            "next_actions": ["provide_existing_webui_bundle"],
        }

    manifest = _read_object(manifest_path)
    export_summary = _read_object(export_summary_path)
    content_assets = _content_assets(manifest, export_summary)
    content_asset_paths = _asset_paths(content_assets)
    material_card_path = _path_from_assets(content_assets, "content_material_card_path")
    material_card_markdown_path = _path_from_assets(content_assets, "content_material_card_markdown_path")
    candidate_pack_path = _path_from_assets(content_assets, "content_candidate_pack_path")
    candidate_pack_markdown_path = _path_from_assets(content_assets, "content_candidate_pack_markdown_path")
    material_card = _read_object(material_card_path) if material_card_path and material_card_path.exists() else {}
    term_correction = material_card.get("term_correction") if isinstance(material_card.get("term_correction"), dict) else {}
    live_term_correction = term_correction_status(root)
    semantic_correction = transcript_semantic_correction_status(root, write=False)
    semantic_gate = _semantic_asset_gate(semantic_correction)
    candidate_pack = _read_object(candidate_pack_path) if candidate_pack_path and candidate_pack_path.exists() else {}
    candidate_chapter_summary = _candidate_chapter_summary(candidate_pack)
    human_eval_paths = _human_sample_eval_paths(root, manifest)
    human_eval = _read_object(human_eval_paths["json"]) if human_eval_paths["json"].exists() else {}
    human_eval_summary = _human_sample_eval_summary(human_eval, human_eval_paths)
    missing_fields = _missing_material_card_fields(material_card)
    card_exists = bool(material_card_path and material_card_path.exists() and material_card_markdown_path and material_card_markdown_path.exists())
    candidate_pack_exists = bool(candidate_pack_path and candidate_pack_path.exists() and candidate_pack_markdown_path and candidate_pack_markdown_path.exists())
    safe_flags = _safe_flags(material_card)
    candidate_pack_safe = _candidate_pack_safe_flags(candidate_pack)
    semantic_blocked_card = _semantic_blocked_material_card_flags(material_card)
    semantic_blocked_candidate_pack = _semantic_blocked_candidate_pack_flags(candidate_pack)
    material_flags_ok = safe_flags or semantic_blocked_card
    candidate_pack_flags_ok = candidate_pack_safe or semantic_blocked_candidate_pack
    status = "ready_for_inspiration_review" if card_exists and candidate_pack_exists and not missing_fields and safe_flags and candidate_pack_safe and semantic_gate["passed"] else "export_required"
    if card_exists and (missing_fields or (not safe_flags and not semantic_blocked_card)):
        status = "material_card_needs_reexport"
    elif card_exists and not candidate_pack_exists:
        status = "content_candidate_pack_needs_reexport"
    elif candidate_pack_exists and not candidate_pack_safe and not semantic_blocked_candidate_pack:
        status = "content_candidate_pack_needs_reexport"
    elif card_exists and candidate_pack_exists and material_flags_ok and candidate_pack_flags_ok and not semantic_gate["passed"]:
        status = "semantic_correction_needs_action"
    ok = status == "ready_for_inspiration_review"

    return {
        "schema": "video_knowledge_pipeline.content_asset_status.v1",
        "ok": ok,
        "status": status,
        "bundle_dir": str(root),
        "write": write,
        "content_assets_present": bool(content_assets),
        "content_asset_paths": content_asset_paths,
        "content_material_card_path": str(material_card_path) if material_card_path else "",
        "content_material_card_markdown_path": str(material_card_markdown_path) if material_card_markdown_path else "",
        "content_candidate_pack_path": str(candidate_pack_path) if candidate_pack_path else "",
        "content_candidate_pack_markdown_path": str(candidate_pack_markdown_path) if candidate_pack_markdown_path else "",
        "material_card_exists": card_exists,
        "content_candidate_pack_exists": candidate_pack_exists,
        "content_candidate_count": int(candidate_pack.get("candidate_count") or len(candidate_pack.get("candidates") or [])) if candidate_pack else 0,
        "content_candidate_chapter_ref_count": candidate_chapter_summary["candidate_with_chapter_refs"],
        "content_candidate_linked_chapter_count": candidate_chapter_summary["linked_chapter_count"],
        "content_candidate_linked_chapters": candidate_chapter_summary["linked_chapters"],
        "content_candidate_chapter_refs_available": candidate_chapter_summary["candidate_with_chapter_refs"] > 0,
        "missing_fields": missing_fields,
        "review_required": bool(material_card.get("review_required")) if material_card else True,
        "publication_allowed": bool(material_card.get("publication_allowed")) if material_card else False,
        "allowed_as_inspiration": bool(material_card.get("allowed_as_inspiration")) if material_card else False,
        "allowed_as_fact": bool(material_card.get("allowed_as_fact")) if material_card else False,
        "circle_of_friends_status": str(material_card.get("circle_of_friends_status") or ""),
        "term_correction_status": str(live_term_correction.get("status") or term_correction.get("status") or "missing"),
        "term_validation_status": str(live_term_correction.get("term_validation_status") or term_correction.get("term_validation_status") or "missing"),
        "accepted_validation_decisions": int(live_term_correction.get("accepted_validation_decisions") or term_correction.get("accepted_validation_decisions") or 0),
        "rejected_validation_decisions": int(live_term_correction.get("rejected_validation_decisions") or term_correction.get("rejected_validation_decisions") or 0),
        "accepted_term_count": int(live_term_correction.get("accepted_term_count") or term_correction.get("accepted_term_count") or 0),
        "validation_rejection_reasons": live_term_correction.get("validation_rejection_reasons", []),
        "validation_rejected_decisions": live_term_correction.get("validation_rejected_decisions", []),
        "term_next_action_key": str(live_term_correction.get("next_action_key") or ""),
        "semantic_correction_ui_summary": semantic_correction.get("ui_summary", {}),
        "semantic_correction_source_vote_summary": semantic_correction.get("source_vote_summary", {}),
        "semantic_correction_status": str(semantic_correction.get("status") or "missing"),
        "semantic_correction_candidate_count": int(semantic_correction.get("candidate_count") or 0),
        "semantic_correction_accepted_count": int(semantic_correction.get("accepted_decision_count") or 0),
        "semantic_correction_review_count": int(semantic_correction.get("review_required_count") or 0),
        "semantic_correction_final_residual_error_total": int(semantic_correction.get("final_residual_error_total") or 0),
        "semantic_correction_readable_impact_status": str(semantic_correction.get("readable_impact_status") or "missing"),
        "semantic_correction_readable_required_residual_total": int(semantic_correction.get("readable_required_residual_total") or 0),
        "semantic_correction_summary_impact_status": str(semantic_correction.get("summary_impact_status") or "missing"),
        "semantic_correction_summary_impact_ok": bool(semantic_correction.get("summary_impact_ok")),
        "semantic_correction_summary_absorption_rate": float(semantic_correction.get("summary_absorption_rate") or 0.0),
        "semantic_correction_summary_residual_original_total": int(semantic_correction.get("summary_residual_original_total") or 0),
        "semantic_correction_asset_gate": semantic_gate,
        "semantic_correction_candidate_type_counts": semantic_correction.get("candidate_type_counts", {}),
        "semantic_correction_risk_level_counts": semantic_correction.get("risk_level_counts", {}),
        "semantic_correction_evidence_source_counts": semantic_correction.get("evidence_source_counts", {}),
        "semantic_correction_validation_rejection_reason_counts": semantic_correction.get("validation_rejection_reason_counts", {}),
        "semantic_correction_review_required_preview": semantic_correction.get("review_required_preview", []),
        "semantic_correction_review_closure_summary": semantic_correction.get("review_closure_summary", {}),
        "semantic_correction_chapter_risk_summary": semantic_correction.get("chapter_risk_summary", []),
        "semantic_correction_next_action_key": str(semantic_correction.get("next_action_key") or ""),
        "semantic_correction_artifacts": semantic_correction.get("artifacts", {}),
        "semantic_correction_commands": semantic_correction.get("commands", {}),
        "term_optional_next_actions": _term_optional_next_actions(root, live_term_correction),
        "term_optional_next_action_artifacts": _term_optional_next_action_artifacts(root, live_term_correction),
        "source_arbitrated_transcript_exists": bool(live_term_correction.get("source_arbitrated_transcript_exists") or term_correction.get("source_arbitrated_transcript_exists")),
        "final_export_alias_total": int(live_term_correction.get("final_export_alias_total") or term_correction.get("final_export_alias_total") or 0),
        "human_confirmation_required": material_card.get("human_confirmation_required") if isinstance(material_card.get("human_confirmation_required"), list) else [],
        "fact_check_required_for": material_card.get("must_fact_check_before_claiming") if isinstance(material_card.get("must_fact_check_before_claiming"), list) else [],
        "content_candidate_pack_safe": candidate_pack_safe,
        "semantic_blocked_material_card_flags": semantic_blocked_card,
        "semantic_blocked_content_candidate_pack_flags": semantic_blocked_candidate_pack,
        "content_candidate_pack_review_required": bool(candidate_pack.get("review_required")) if candidate_pack else True,
        "content_candidate_pack_publication_allowed": bool(candidate_pack.get("publication_allowed")) if candidate_pack else False,
        "content_candidate_pack_allowed_as_fact": bool(candidate_pack.get("allowed_as_fact")) if candidate_pack else False,
        "content_candidate_pack_allowed_as_inspiration": bool(candidate_pack.get("allowed_as_inspiration")) if candidate_pack else False,
        "human_sample_eval": human_eval_summary,
        "human_sample_eval_status": human_eval_summary["status"],
        "human_sample_eval_json_path": human_eval_summary["json_path"],
        "human_sample_eval_markdown_path": human_eval_summary["markdown_path"],
        "human_sample_eval_exists": human_eval_summary["exists"],
        "human_sample_eval_labeled_rows": human_eval_summary["labeled_rows"],
        "human_sample_eval_content_candidate_usable_rate": human_eval_summary["rates"].get("content_candidate_usable_rate"),
        "human_sample_eval_content_candidate_evidence_sufficient_rate": human_eval_summary["rates"].get("content_candidate_evidence_sufficient_rate"),
        "human_sample_eval_multimodal_net_help_rate": human_eval_summary["rates"].get("human_sampled_multimodal_net_help_rate"),
        "next_actions": _next_actions(status, semantic_gate=semantic_gate, semantic_correction=semantic_correction),
    }




def _term_optional_next_actions(root: Path, status: dict[str, Any]) -> list[str]:
    key = str(status.get("next_action_key") or "").strip()
    if not key:
        return []
    bundle = str(root)
    result_path = str(root / "term-arbitration-codex-result.codex.md")
    if key == "term_arbitration_codex":
        return [f".\\scripts\\video-knowledge.ps1 term-arbitration-codex {bundle}"]
    if key == "term_arbitration_codex_validate":
        return [f".\\scripts\\video-knowledge.ps1 validate-term-arbitration-codex-result {bundle} --input-json {result_path}"]
    if key == "term_correction_closure":
        return [f".\\scripts\\video-knowledge.ps1 term-correction-closure {bundle} --input-json {result_path}"]
    if key == "term_correction_impact":
        return [f".\\scripts\\video-knowledge.ps1 term-correction-impact-report {bundle}"]
    if key == "transcript_source_arbitration":
        return [f".\\scripts\\video-knowledge.ps1 transcript-source-arbitration {bundle}"]
    return []


def _term_optional_next_action_artifacts(root: Path, status: dict[str, Any]) -> dict[str, str]:
    key = str(status.get("next_action_key") or "").strip()
    if not key:
        return {}
    artifacts: dict[str, str] = {}
    if key == "term_arbitration_codex":
        artifacts["term_arbitration_codex"] = str(root / "mcp-term-arbitration-codex.args.json")
    elif key == "term_arbitration_codex_validate":
        artifacts["term_arbitration_codex_validate"] = str(root / "mcp-term-arbitration-codex-validate.args.json")
        artifacts["term_correction_closure_codex"] = str(root / "mcp-term-correction-closure-codex.args.json")
    elif key == "term_correction_closure":
        artifacts["term_correction_closure_codex"] = str(root / "mcp-term-correction-closure-codex.args.json")
        artifacts["term_correction_closure"] = str(root / "mcp-term-correction-closure.args.json")
    elif key == "term_correction_impact":
        artifacts["term_correction_impact"] = str(root / "mcp-term-correction-impact-report.args.json")
    elif key == "transcript_source_arbitration":
        artifacts["transcript_source_arbitration"] = str(root / "mcp-transcript-source-arbitration.args.json")
    return artifacts
def _human_sample_eval_paths(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    json_raw = str(manifest.get("human_sample_eval_json") or "human-sample-eval.json").strip()
    markdown_raw = str(manifest.get("human_sample_eval_report") or "human-sample-eval.md").strip()
    return {
        "json": _bundle_path(root, json_raw),
        "markdown": _bundle_path(root, markdown_raw),
    }


def _human_sample_eval_summary(human_eval: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    rates = human_eval.get("rates") if isinstance(human_eval.get("rates"), dict) else {}
    interpretation = human_eval.get("interpretation") if isinstance(human_eval.get("interpretation"), dict) else {}
    exists = bool(paths["json"].exists() and paths["markdown"].exists() and human_eval)
    return {
        "exists": exists,
        "status": str(human_eval.get("status") or ("available" if exists else "not_available")),
        "json_path": str(paths["json"]),
        "markdown_path": str(paths["markdown"]),
        "sample_count": int(human_eval.get("sample_count") or 0) if human_eval else 0,
        "labeled_rows": int(human_eval.get("labeled_rows") or 0) if human_eval else 0,
        "annotated_rows": int(human_eval.get("annotated_rows") or 0) if human_eval else 0,
        "rates": {
            "content_candidate_usable_rate": rates.get("content_candidate_usable_rate"),
            "content_candidate_evidence_sufficient_rate": rates.get("content_candidate_evidence_sufficient_rate"),
            "human_sampled_multimodal_net_help_rate": rates.get("human_sampled_multimodal_net_help_rate"),
            "final_note_acceptable_rate": rates.get("final_note_acceptable_rate"),
            "overall_correct_or_partial_rate": rates.get("overall_correct_or_partial_rate"),
        },
        "interpretation_verdict": str(interpretation.get("verdict") or "") if interpretation else "",
        "review_signal_only": True,
        "publication_allowed": False,
    }


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()

def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _content_assets(manifest: dict[str, Any], export_summary: dict[str, Any]) -> dict[str, Any]:
    for source in (export_summary, manifest):
        value = source.get("content_assets") if isinstance(source, dict) else {}
        if isinstance(value, dict) and value:
            return value
    export = manifest.get("knowledge_note_export") if isinstance(manifest.get("knowledge_note_export"), dict) else {}
    value = export.get("content_assets") if isinstance(export.get("content_assets"), dict) else {}
    return value if isinstance(value, dict) else {}



def _asset_paths(content_assets: dict[str, Any]) -> dict[str, str]:
    keys = [
        "summary_path",
        "smart_summary_path",
        "smart_summary_prompt_path",
        "timeline_path",
        "audit_path",
        "key_segments_path",
        "short_video_script_drafts_path",
        "highlight_post_drafts_path",
        "content_material_card_path",
        "content_material_card_markdown_path",
        "content_candidate_pack_path",
        "content_candidate_pack_markdown_path",
        "human_sample_eval_json_path",
        "human_sample_eval_markdown_path",
    ]
    result: dict[str, str] = {}
    for key in keys:
        raw = str(content_assets.get(key) or "").strip()
        if raw:
            result[key] = str(Path(raw).expanduser().resolve())
    return result
def _path_from_assets(content_assets: dict[str, Any], key: str) -> Path | None:
    raw = str(content_assets.get(key) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _missing_material_card_fields(material_card: dict[str, Any]) -> list[str]:
    if not material_card:
        return REQUIRED_MATERIAL_CARD_FIELDS[:]
    missing = []
    for field in REQUIRED_MATERIAL_CARD_FIELDS:
        if field not in material_card:
            missing.append(field)
    return missing


def _safe_flags(material_card: dict[str, Any]) -> bool:
    if not material_card:
        return False
    return (
        material_card.get("review_required") is True
        and material_card.get("publication_allowed") is False
        and material_card.get("allowed_as_inspiration") is True
        and material_card.get("allowed_as_fact") is False
        and material_card.get("circle_of_friends_status") == "needs_review_inspiration"
    )



def _candidate_pack_safe_flags(candidate_pack: dict[str, Any]) -> bool:
    if not candidate_pack:
        return False
    return (
        candidate_pack.get("review_required") is True
        and candidate_pack.get("publication_allowed") is False
        and candidate_pack.get("allowed_as_inspiration") is True
        and candidate_pack.get("allowed_as_fact") is False
    )


def _semantic_blocked_material_card_flags(material_card: dict[str, Any]) -> bool:
    if not material_card:
        return False
    semantic = material_card.get("transcript_semantic_correction") if isinstance(material_card.get("transcript_semantic_correction"), dict) else {}
    gate = semantic.get("asset_gate") if isinstance(semantic.get("asset_gate"), dict) else {}
    return (
        material_card.get("review_required") is True
        and material_card.get("publication_allowed") is False
        and material_card.get("allowed_as_inspiration") is False
        and material_card.get("allowed_as_fact") is False
        and material_card.get("circle_of_friends_status") == "semantic_correction_required"
        and gate
        and not bool(gate.get("passed"))
    )


def _semantic_blocked_candidate_pack_flags(candidate_pack: dict[str, Any]) -> bool:
    if not candidate_pack:
        return False
    semantic = candidate_pack.get("transcript_semantic_correction") if isinstance(candidate_pack.get("transcript_semantic_correction"), dict) else {}
    gate = semantic.get("asset_gate") if isinstance(semantic.get("asset_gate"), dict) else {}
    return (
        candidate_pack.get("review_required") is True
        and candidate_pack.get("publication_allowed") is False
        and candidate_pack.get("allowed_as_inspiration") is False
        and candidate_pack.get("allowed_as_fact") is False
        and gate
        and not bool(gate.get("passed"))
    )


def _candidate_chapter_summary(candidate_pack: dict[str, Any]) -> dict[str, Any]:
    candidates = candidate_pack.get("candidates") if isinstance(candidate_pack.get("candidates"), list) else []
    linked: dict[int, dict[str, Any]] = {}
    candidate_with_refs = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        refs = candidate.get("summary_chapter_refs") if isinstance(candidate.get("summary_chapter_refs"), list) else []
        valid_refs = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            try:
                chapter_index = int(ref.get("chapter_index") or 0)
            except (TypeError, ValueError):
                chapter_index = 0
            if chapter_index <= 0:
                continue
            valid_refs.append(ref)
            linked.setdefault(
                chapter_index,
                {
                    "chapter_index": chapter_index,
                    "chapter_title": str(ref.get("chapter_title") or ""),
                    "chapter_time_range": str(ref.get("chapter_time_range") or ""),
                    "candidate_count": 0,
                    "candidate_ids": [],
                },
            )
            linked[chapter_index]["candidate_count"] = int(linked[chapter_index].get("candidate_count") or 0) + 1
            candidate_id = str(candidate.get("id") or "")
            if candidate_id:
                ids = linked[chapter_index].setdefault("candidate_ids", [])
                if candidate_id not in ids:
                    ids.append(candidate_id)
        if valid_refs:
            candidate_with_refs += 1
    chapters = [linked[key] for key in sorted(linked)]
    for chapter in chapters:
        chapter["candidate_ids"] = chapter.get("candidate_ids", [])[:12]
    return {
        "candidate_with_chapter_refs": candidate_with_refs,
        "linked_chapter_count": len(chapters),
        "linked_chapters": chapters[:24],
    }

def _semantic_asset_gate(semantic_correction: dict[str, Any]) -> dict[str, Any]:
    status = str(semantic_correction.get("status") or "missing") if isinstance(semantic_correction, dict) else "missing"
    review_count = int(semantic_correction.get("review_required_count") or 0) if isinstance(semantic_correction, dict) else 0
    residual_total = int(semantic_correction.get("final_residual_error_total") or 0) if isinstance(semantic_correction, dict) else 0
    readable_status = str(semantic_correction.get("readable_impact_status") or "missing") if isinstance(semantic_correction, dict) else "missing"
    readable_residual = int(semantic_correction.get("readable_required_residual_total") or 0) if isinstance(semantic_correction, dict) else 0
    summary_status = str(semantic_correction.get("summary_impact_status") or "missing") if isinstance(semantic_correction, dict) else "missing"
    summary_residual = int(semantic_correction.get("summary_residual_original_total") or 0) if isinstance(semantic_correction, dict) else 0
    candidate_discovery_status = str(semantic_correction.get("candidate_discovery_status") or "not_planned") if isinstance(semantic_correction, dict) else "not_planned"
    candidate_discovery_next_action = str(semantic_correction.get("candidate_discovery_next_action") or "") if isinstance(semantic_correction, dict) else ""
    next_action = str(semantic_correction.get("next_action_key") or "") if isinstance(semantic_correction, dict) else ""
    no_candidates = status == "no_candidates"
    candidate_discovery_complete = candidate_discovery_status in {"no_segments_selected", "no_suggestions", "no_candidates_imported", "imported"}
    no_candidates_ready = bool(no_candidates and candidate_discovery_complete)
    no_candidates_needs_discovery = bool(no_candidates and not candidate_discovery_complete)
    impact_ready = bool(
        status == "impact_passed"
        and residual_total == 0
        and readable_status == "passed"
        and readable_residual == 0
        and summary_status in {"passed", "no_accepted_decisions", "no_evaluable_replacements"}
        and summary_residual == 0
    )
    passed = bool(no_candidates_ready or impact_ready)
    if passed:
        if impact_ready and review_count > 0:
            gate_status = "passed_with_open_review"
        else:
            gate_status = "passed" if impact_ready else "not_required_no_candidates"
    elif no_candidates_needs_discovery:
        gate_status = "needs_candidate_discovery"
        next_action = candidate_discovery_next_action or "run_candidate_discovery"
    elif summary_status in {"missing", ""} and status in {"needs_summary_impact_report", "impact_passed"}:
        gate_status = "needs_summary_impact_report"
        next_action = next_action or "run_summary_impact"
    elif summary_residual > 0 or summary_status not in {"passed", "no_accepted_decisions", "no_evaluable_replacements", "missing", ""}:
        gate_status = "summary_impact_needs_fix"
        next_action = next_action or "refresh_summary_or_review"
    else:
        gate_status = "semantic_correction_needs_action"
    return {
        "passed": passed,
        "status": gate_status,
        "semantic_status": status,
        "readable_impact_status": readable_status,
        "readable_required_residual_total": readable_residual,
        "summary_impact_status": summary_status,
        "summary_residual_original_total": summary_residual,
        "summary_absorption_rate": float(semantic_correction.get("summary_absorption_rate") or 0.0) if isinstance(semantic_correction, dict) else 0.0,
        "final_residual_error_total": residual_total,
        "review_required_count": review_count,
        "review_required_nonblocking": bool(passed and review_count > 0),
        "candidate_discovery_status": candidate_discovery_status,
        "candidate_discovery_complete": candidate_discovery_complete,
        "next_action_key": next_action or "none",
    }


def _next_actions(status: str, *, semantic_gate: dict[str, Any] | None = None, semantic_correction: dict[str, Any] | None = None) -> list[str]:
    if status == "ready_for_inspiration_review":
        return ["route_to_content_assets_or_circle_of_friends_as_needs_review_inspiration"]
    if status == "semantic_correction_needs_action":
        commands = semantic_correction.get("commands", {}) if isinstance(semantic_correction, dict) else {}
        key = str((semantic_gate or {}).get("next_action_key") or "")
        command = str(commands.get(key) or commands.get("status") or "").strip()
        actions = ["finish_transcript_semantic_correction_before_handoff"]
        if command:
            actions.append(command)
        actions.append("rerun_content_asset_status")
        return actions
    if status in {"material_card_needs_reexport", "content_candidate_pack_needs_reexport"}:
        return ["rerun_export_knowledge_note", "rerun_content_asset_status"]
    return ["run_export_knowledge_note", "rerun_content_asset_status"]
