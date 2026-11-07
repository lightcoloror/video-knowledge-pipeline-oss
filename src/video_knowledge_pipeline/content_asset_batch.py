from __future__ import annotations

from pathlib import Path
from typing import Any

from .content_asset_status import content_asset_status
from .models import now_iso
from .storage import read_json, write_json


def batch_content_asset_status(
    batch_input: str | Path,
    *,
    output_dir: str | Path = "",
    write: bool = True,
) -> dict[str, Any]:
    bundles = _discover_bundles(batch_input)
    rows = [_row_for_bundle(bundle) for bundle in bundles]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    semantic_correction_summary = _semantic_correction_batch_summary(rows)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(batch_input)
    json_path = out_dir / "batch-content-material-cards.json"
    markdown_path = out_dir / "batch-content-material-cards.md"
    result = {
        "schema": "video_knowledge_pipeline.batch_content_asset_status.v1",
        "created_at": now_iso(),
        "ok": bool(rows) and all(row.get("ok") for row in rows),
        "status": "ok" if rows and all(row.get("ok") for row in rows) else "needs_action",
        "batch_input": str(Path(batch_input).expanduser()),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "status_counts": counts,
        "semantic_correction_summary": semantic_correction_summary,
        "items": rows,
        "next_actions": _batch_next_actions(rows),
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_batch_markdown(result), encoding="utf-8")
    return result


def content_handoff_pack(
    batch_input: str | Path,
    *,
    output_dir: str | Path = "",
    write: bool = True,
) -> dict[str, Any]:
    rows = [_row_for_bundle(bundle) for bundle in _discover_bundles(batch_input)]
    ready = [row for row in rows if row.get("ok") and row.get("publication_allowed") is False and row.get("allowed_as_fact") is False]
    semantic_correction_summary = _semantic_correction_batch_summary(rows)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(batch_input)
    json_path = out_dir / "content-handoff-pack.json"
    markdown_path = out_dir / "content-handoff-pack.md"
    result = {
        "schema": "video_knowledge_pipeline.content_handoff_pack.v1",
        "created_at": now_iso(),
        "ok": bool(ready),
        "status": "ready" if ready else "no_ready_material_cards",
        "batch_input": str(Path(batch_input).expanduser()),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "ready_count": len(ready),
        "skipped_count": len(rows) - len(ready),
        "semantic_correction_summary": semantic_correction_summary,
        "items": ready,
        "operator_boundary": {
            "draft_only": True,
            "publication_allowed": False,
            "no_logseq_or_obsidian_writeback": True,
            "no_auto_publish": True,
        },
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_handoff_markdown(result), encoding="utf-8")
    return result


def _discover_bundles(batch_input: str | Path) -> list[Path]:
    path = Path(batch_input).expanduser()
    if path.is_dir():
        if (path / "manifest.json").exists():
            return [path.resolve()]
        bundles = [candidate.resolve() for candidate in path.rglob("manifest.json") if candidate.parent.name == "webui-bundle"]
        return sorted({bundle.parent for bundle in bundles}, key=lambda item: str(item).lower())
    if path.is_file():
        data = read_json(path)
        return _bundles_from_json(data, base_dir=path.parent)
    return []


def discover_bundles(batch_input: str | Path) -> list[Path]:
    """Public read-only Bundle discovery reused by cross-run lineage tools."""

    return _discover_bundles(batch_input)


def _bundles_from_json(data: Any, *, base_dir: Path) -> list[Path]:
    bundles: list[Path] = []
    if isinstance(data, dict):
        for key in ("bundle_dir", "webui_bundle", "bundle", "path"):
            value = data.get(key)
            if isinstance(value, str) and value:
                bundles.append(_resolve_path(value, base_dir))
        for key in ("items", "bundles", "results"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    bundles.extend(_bundles_from_json(item, base_dir=base_dir))
    elif isinstance(data, list):
        for item in data:
            bundles.extend(_bundles_from_json(item, base_dir=base_dir))
    deduped: list[Path] = []
    seen: set[str] = set()
    for bundle in bundles:
        key = str(bundle.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(bundle.resolve())
    return deduped


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _row_for_bundle(bundle: Path) -> dict[str, Any]:
    status = content_asset_status(bundle, write=False)
    return {
        "bundle_dir": str(bundle.resolve()),
        "ok": bool(status.get("ok")),
        "status": status.get("status"),
        "content_material_card_path": status.get("content_material_card_path", ""),
        "content_material_card_markdown_path": status.get("content_material_card_markdown_path", ""),
        "content_candidate_pack_path": status.get("content_candidate_pack_path", ""),
        "content_candidate_pack_markdown_path": status.get("content_candidate_pack_markdown_path", ""),
        "content_candidate_pack_exists": status.get("content_candidate_pack_exists", False),
        "content_candidate_count": status.get("content_candidate_count", 0),
        "content_candidate_chapter_ref_count": status.get("content_candidate_chapter_ref_count", 0),
        "content_candidate_linked_chapter_count": status.get("content_candidate_linked_chapter_count", 0),
        "content_candidate_linked_chapters": status.get("content_candidate_linked_chapters", []),
        "content_candidate_chapter_refs_available": status.get("content_candidate_chapter_refs_available", False),
        "content_candidate_pack_safe": status.get("content_candidate_pack_safe", False),
        "allowed_as_inspiration": status.get("allowed_as_inspiration", False),
        "allowed_as_fact": status.get("allowed_as_fact", False),
        "publication_allowed": status.get("publication_allowed", False),
        "review_required": status.get("review_required", True),
        "circle_of_friends_status": status.get("circle_of_friends_status", ""),
        "term_correction_status": status.get("term_correction_status", "missing"),
        "term_validation_status": status.get("term_validation_status", "missing"),
        "accepted_validation_decisions": status.get("accepted_validation_decisions", 0),
        "rejected_validation_decisions": status.get("rejected_validation_decisions", 0),
        "accepted_term_count": status.get("accepted_term_count", 0),
        "validation_rejection_reasons": status.get("validation_rejection_reasons", []),
        "validation_rejected_decisions": status.get("validation_rejected_decisions", []),
        "term_next_action_key": status.get("term_next_action_key", ""),
        "term_optional_next_actions": status.get("term_optional_next_actions", []),
        "term_optional_next_action_artifacts": status.get("term_optional_next_action_artifacts", {}),
        "semantic_correction_ui_summary": status.get("semantic_correction_ui_summary", {}),
        "semantic_correction_source_vote_summary": status.get("semantic_correction_source_vote_summary", {}),
        "semantic_correction_status": status.get("semantic_correction_status", "missing"),
        "semantic_correction_candidate_count": status.get("semantic_correction_candidate_count", 0),
        "semantic_correction_accepted_count": status.get("semantic_correction_accepted_count", 0),
        "semantic_correction_review_count": status.get("semantic_correction_review_count", 0),
        "semantic_correction_final_residual_error_total": status.get("semantic_correction_final_residual_error_total", 0),
        "semantic_correction_readable_impact_status": status.get("semantic_correction_readable_impact_status", "missing"),
        "semantic_correction_readable_required_residual_total": status.get("semantic_correction_readable_required_residual_total", 0),
        "semantic_correction_summary_impact_status": status.get("semantic_correction_summary_impact_status", "missing"),
        "semantic_correction_summary_impact_ok": status.get("semantic_correction_summary_impact_ok", False),
        "semantic_correction_summary_absorption_rate": status.get("semantic_correction_summary_absorption_rate", 0.0),
        "semantic_correction_summary_residual_original_total": status.get("semantic_correction_summary_residual_original_total", 0),
        "semantic_correction_asset_gate": status.get("semantic_correction_asset_gate", {}),
        "semantic_correction_candidate_type_counts": status.get("semantic_correction_candidate_type_counts", {}),
        "semantic_correction_risk_level_counts": status.get("semantic_correction_risk_level_counts", {}),
        "semantic_correction_evidence_source_counts": status.get("semantic_correction_evidence_source_counts", {}),
        "semantic_correction_validation_rejection_reason_counts": status.get("semantic_correction_validation_rejection_reason_counts", {}),
        "semantic_correction_review_closure_summary": status.get("semantic_correction_review_closure_summary", {}),
        "semantic_correction_chapter_risk_summary": status.get("semantic_correction_chapter_risk_summary", []),
        "semantic_correction_next_action_key": status.get("semantic_correction_next_action_key", ""),
        "semantic_correction_artifacts": status.get("semantic_correction_artifacts", {}),
        "semantic_correction_commands": status.get("semantic_correction_commands", {}),
        "fact_check_required_for": status.get("fact_check_required_for", []),
        "human_confirmation_required": status.get("human_confirmation_required", []),
        "human_sample_eval_status": status.get("human_sample_eval_status", "not_available"),
        "human_sample_eval_json_path": status.get("human_sample_eval_json_path", ""),
        "human_sample_eval_markdown_path": status.get("human_sample_eval_markdown_path", ""),
        "human_sample_eval_exists": status.get("human_sample_eval_exists", False),
        "human_sample_eval_labeled_rows": status.get("human_sample_eval_labeled_rows", 0),
        "human_sample_eval_content_candidate_usable_rate": status.get("human_sample_eval_content_candidate_usable_rate"),
        "human_sample_eval_content_candidate_evidence_sufficient_rate": status.get("human_sample_eval_content_candidate_evidence_sufficient_rate"),
        "human_sample_eval_multimodal_net_help_rate": status.get("human_sample_eval_multimodal_net_help_rate"),
        "next_actions": status.get("next_actions", []),
    }


def _default_output_dir(batch_input: str | Path) -> Path:
    path = Path(batch_input).expanduser()
    if path.is_dir():
        return path.resolve() if not (path / "manifest.json").exists() else path.resolve() / "exports"
    return path.parent.resolve() if path.parent else Path.cwd()


def _batch_next_actions(rows: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for row in rows:
        actions.extend(str(action) for action in row.get("next_actions") or [])
    return _dedupe(actions)



def _semantic_correction_batch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_next_action: dict[str, int] = {}
    by_summary_impact_status: dict[str, int] = {}
    by_asset_gate_status: dict[str, int] = {}
    by_candidate_type: dict[str, int] = {}
    by_risk_level: dict[str, int] = {}
    by_evidence_source: dict[str, int] = {}
    by_rejection_reason: dict[str, int] = {}
    by_review_action: dict[str, int] = {}
    by_ui_state: dict[str, int] = {}
    by_accepted_type: dict[str, int] = {}
    by_applied_type: dict[str, int] = {}
    by_source_vote_dominant_side: dict[str, int] = {}
    by_candidate_support_source: dict[str, int] = {}
    by_original_support_source: dict[str, int] = {}
    items_needing_action: list[dict[str, Any]] = []
    chapter_risk_items: list[dict[str, Any]] = []
    totals = {
        "bundle_count": len(rows),
        "candidate_count": 0,
        "accepted_count": 0,
        "review_required_count": 0,
        "final_residual_error_total": 0,
        "readable_required_residual_total": 0,
        "summary_residual_original_total": 0,
        "summary_impact_missing_or_failed_count": 0,
        "semantic_asset_gate_blocked_count": 0,
        "imported_review_decision_count": 0,
        "closed_review_decision_count": 0,
        "open_review_required_count": 0,
        "auto_candidate_count": 0,
        "human_review_candidate_count": 0,
        "applied_correction_count": 0,
        "rejected_decision_count": 0,
        "source_vote_candidate_count": 0,
        "source_conflict_count": 0,
        "needs_review_by_source_vote_count": 0,
        "source_vote_candidate_weight_total": 0,
        "source_vote_original_weight_total": 0,
        "source_vote_neutral_weight_total": 0,
    }
    for row in rows:
        status = str(row.get("semantic_correction_status") or "missing")
        next_action = str(row.get("semantic_correction_next_action_key") or "none")
        by_status[status] = by_status.get(status, 0) + 1
        by_next_action[next_action] = by_next_action.get(next_action, 0) + 1
        totals["candidate_count"] += int(row.get("semantic_correction_candidate_count") or 0)
        totals["accepted_count"] += int(row.get("semantic_correction_accepted_count") or 0)
        review_required_count = int(row.get("semantic_correction_review_count") or 0)
        residual_error_total = int(row.get("semantic_correction_final_residual_error_total") or 0)
        totals["review_required_count"] += review_required_count
        totals["final_residual_error_total"] += residual_error_total
        totals["readable_required_residual_total"] += int(row.get("semantic_correction_readable_required_residual_total") or 0)
        summary_status = str(row.get("semantic_correction_summary_impact_status") or "missing")
        asset_gate = row.get("semantic_correction_asset_gate") if isinstance(row.get("semantic_correction_asset_gate"), dict) else {}
        asset_gate_status = str(asset_gate.get("status") or "missing")
        by_summary_impact_status[summary_status] = by_summary_impact_status.get(summary_status, 0) + 1
        by_asset_gate_status[asset_gate_status] = by_asset_gate_status.get(asset_gate_status, 0) + 1
        summary_residual = int(row.get("semantic_correction_summary_residual_original_total") or 0)
        totals["summary_residual_original_total"] += summary_residual
        if summary_status not in {"passed", "no_accepted_decisions", "no_evaluable_replacements"}:
            totals["summary_impact_missing_or_failed_count"] += 1
        if asset_gate and not asset_gate.get("passed"):
            totals["semantic_asset_gate_blocked_count"] += 1
        _merge_counts(by_candidate_type, row.get("semantic_correction_candidate_type_counts"))
        _merge_counts(by_risk_level, row.get("semantic_correction_risk_level_counts"))
        _merge_counts(by_evidence_source, row.get("semantic_correction_evidence_source_counts"))
        _merge_counts(by_rejection_reason, row.get("semantic_correction_validation_rejection_reason_counts"))
        ui_summary = row.get("semantic_correction_ui_summary") if isinstance(row.get("semantic_correction_ui_summary"), dict) else {}
        ui_state = str(ui_summary.get("ui_state") or "missing")
        by_ui_state[ui_state] = by_ui_state.get(ui_state, 0) + 1
        totals["auto_candidate_count"] += int(ui_summary.get("auto_candidate_count") or 0)
        totals["human_review_candidate_count"] += int(ui_summary.get("human_review_candidate_count") or 0)
        totals["applied_correction_count"] += int(ui_summary.get("applied_correction_count") or 0)
        totals["rejected_decision_count"] += int(ui_summary.get("rejected_decision_count") or 0)
        _merge_counts(by_accepted_type, ui_summary.get("accepted_decision_type_counts"))
        _merge_counts(by_applied_type, ui_summary.get("applied_correction_type_counts"))
        source_vote_summary = row.get("semantic_correction_source_vote_summary") if isinstance(row.get("semantic_correction_source_vote_summary"), dict) else {}
        totals["source_vote_candidate_count"] += int(source_vote_summary.get("candidate_count_with_votes") or 0)
        totals["source_conflict_count"] += int(source_vote_summary.get("source_conflict_count") or 0)
        totals["needs_review_by_source_vote_count"] += int(source_vote_summary.get("needs_review_by_source_vote_count") or 0)
        totals["source_vote_candidate_weight_total"] += int(source_vote_summary.get("candidate_weight_total") or 0)
        totals["source_vote_original_weight_total"] += int(source_vote_summary.get("original_weight_total") or 0)
        totals["source_vote_neutral_weight_total"] += int(source_vote_summary.get("neutral_weight_total") or 0)
        _merge_counts(by_source_vote_dominant_side, source_vote_summary.get("by_dominant_side"))
        _merge_counts(by_candidate_support_source, source_vote_summary.get("by_candidate_support_source"))
        _merge_counts(by_original_support_source, source_vote_summary.get("by_original_support_source"))
        closure = row.get("semantic_correction_review_closure_summary") if isinstance(row.get("semantic_correction_review_closure_summary"), dict) else {}
        totals["imported_review_decision_count"] += int(closure.get("imported_review_decision_count") or 0)
        totals["closed_review_decision_count"] += int(closure.get("closed_review_decision_count") or 0)
        open_review_required_count = int(closure.get("open_review_required_count") or review_required_count)
        totals["open_review_required_count"] += open_review_required_count
        _merge_counts(by_review_action, closure.get("actions"))
        for chapter in row.get("semantic_correction_chapter_risk_summary") or []:
            if not isinstance(chapter, dict):
                continue
            if int(chapter.get("candidate_count") or 0) or int(chapter.get("review_required_count") or 0):
                chapter_risk_items.append({
                    "bundle_dir": row.get("bundle_dir", ""),
                    "chapter_index": chapter.get("chapter_index"),
                    "chapter_title": chapter.get("chapter_title", ""),
                    "chapter_time_range": chapter.get("chapter_time_range", ""),
                    "candidate_count": int(chapter.get("candidate_count") or 0),
                    "review_required_count": int(chapter.get("review_required_count") or 0),
                    "risk_level_counts": chapter.get("risk_level_counts", {}),
                })
        if next_action != "none" or review_required_count or residual_error_total or open_review_required_count or (asset_gate and not asset_gate.get("passed")):
            items_needing_action.append(
                {
                    "bundle_dir": row.get("bundle_dir", ""),
                    "status": status,
                    "next_action": next_action,
                    "review_required_count": review_required_count,
                    "open_review_required_count": open_review_required_count,
                    "final_residual_error_total": residual_error_total,
                    "summary_impact_status": summary_status,
                    "summary_residual_original_total": summary_residual,
                    "asset_gate_status": asset_gate_status,
                }
            )
    return {
        **totals,
        "by_status": dict(sorted(by_status.items())),
        "by_next_action": dict(sorted(by_next_action.items())),
        "by_summary_impact_status": dict(sorted(by_summary_impact_status.items())),
        "by_asset_gate_status": dict(sorted(by_asset_gate_status.items())),
        "by_candidate_type": dict(sorted(by_candidate_type.items())),
        "by_risk_level": dict(sorted(by_risk_level.items())),
        "by_evidence_source": dict(sorted(by_evidence_source.items())),
        "by_rejection_reason": dict(sorted(by_rejection_reason.items())),
        "by_review_action": dict(sorted(by_review_action.items())),
        "by_ui_state": dict(sorted(by_ui_state.items())),
        "by_accepted_type": dict(sorted(by_accepted_type.items())),
        "by_applied_type": dict(sorted(by_applied_type.items())),
        "by_source_vote_dominant_side": dict(sorted(by_source_vote_dominant_side.items())),
        "by_candidate_support_source": dict(sorted(by_candidate_support_source.items())),
        "by_original_support_source": dict(sorted(by_original_support_source.items())),
        "items_needing_action": items_needing_action,
        "chapter_risk_items": chapter_risk_items[:50],
    }


def _merge_counts(target: dict[str, int], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        name = str(key or "").strip()
        if not name:
            continue
        target[name] = target.get(name, 0) + int(value or 0)


def _render_batch_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Batch Content Material Cards",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- Count: `{result.get('count')}`",
        "",
    ]
    lines.extend(_semantic_correction_summary_markdown(result.get("semantic_correction_summary")))
    lines.extend(
        [
            "| Bundle | Status | Inspiration | Fact | Publish | Term correction | Term validation | Accepted/Rejected | Term blockers | Term action | Semantic correction | Semantic review | Semantic action | Card | Candidates | Chapters | Sample eval | Candidate usable | Candidate evidence | Next action |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for item in result.get("items") or []:
        lines.append(
            "| {bundle} | `{status}` | `{inspiration}` | `{fact}` | `{publish}` | `{term_correction}` | `{term_validation}` | `{accepted_rejected}` | {term_blockers} | `{term_action}` | `{semantic_correction}` | {semantic_review} | `{semantic_action}` | `{card}` | `{candidates}` | `{chapters}` | `{sample_eval}` | `{candidate_usable}` | `{candidate_evidence}` | `{next_action}` |".format(
                bundle=item.get("bundle_dir", ""),
                status=item.get("status", ""),
                inspiration=str(bool(item.get("allowed_as_inspiration"))).lower(),
                fact=str(bool(item.get("allowed_as_fact"))).lower(),
                publish=str(bool(item.get("publication_allowed"))).lower(),
                term_correction=item.get("term_correction_status") or "missing",
                term_validation=item.get("term_validation_status") or "missing",
                accepted_rejected=f"{int(item.get('accepted_validation_decisions') or 0)}/{int(item.get('rejected_validation_decisions') or 0)}",
                term_blockers=_term_blockers_text(item),
                term_action=item.get("term_next_action_key") or "none",
                semantic_correction=_semantic_correction_text(item),
                semantic_review=_semantic_review_text(item),
                semantic_action=item.get("semantic_correction_next_action_key") or "none",
                card=item.get("content_material_card_path", ""),
                candidates=item.get("content_candidate_count", 0),
                chapters=item.get("content_candidate_linked_chapter_count", 0),
                sample_eval=item.get("human_sample_eval_status") or "not_available",
                candidate_usable=_rate_text(item.get("human_sample_eval_content_candidate_usable_rate")),
                candidate_evidence=_rate_text(item.get("human_sample_eval_content_candidate_evidence_sufficient_rate")),
                next_action=", ".join(item.get("next_actions") or []),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _semantic_correction_summary_markdown(summary: Any) -> list[str]:
    if not isinstance(summary, dict):
        return []
    lines = [
        "## Semantic Correction Summary",
        "",
        f"- Candidates: `{int(summary.get('candidate_count') or 0)}`",
        f"- Accepted decisions: `{int(summary.get('accepted_count') or 0)}`",
        f"- Review required: `{int(summary.get('review_required_count') or 0)}`",
        f"- Imported review decisions: `{int(summary.get('imported_review_decision_count') or 0)}`",
        f"- Closed review decisions: `{int(summary.get('closed_review_decision_count') or 0)}`",
        f"- Open review targets: `{int(summary.get('open_review_required_count') or 0)}`",
        f"- Residual errors: `{int(summary.get('final_residual_error_total') or 0)}`",
        f"- Readable residuals: `{int(summary.get('readable_required_residual_total') or 0)}`",
        f"- Summary residuals: `{int(summary.get('summary_residual_original_total') or 0)}`",
        f"- Summary impact missing/failed: `{int(summary.get('summary_impact_missing_or_failed_count') or 0)}`",
        f"- Semantic asset gate blocked: `{int(summary.get('semantic_asset_gate_blocked_count') or 0)}`",
        f"- Auto candidates: `{int(summary.get('auto_candidate_count') or 0)}`",
        f"- Human review candidates: `{int(summary.get('human_review_candidate_count') or 0)}`",
        f"- Applied corrections: `{int(summary.get('applied_correction_count') or 0)}`",
        f"- Rejected decisions: `{int(summary.get('rejected_decision_count') or 0)}`",
        f"- Source vote candidates: `{int(summary.get('source_vote_candidate_count') or 0)}`",
        f"- Source conflicts: `{int(summary.get('source_conflict_count') or 0)}`",
        f"- Needs review by source vote: `{int(summary.get('needs_review_by_source_vote_count') or 0)}`",
        f"- Source vote weights candidate/original/neutral: `{int(summary.get('source_vote_candidate_weight_total') or 0)}` / `{int(summary.get('source_vote_original_weight_total') or 0)}` / `{int(summary.get('source_vote_neutral_weight_total') or 0)}`",
        "",
    ]
    for title, key in (
        ("Status", "by_status"),
        ("Next action", "by_next_action"),
        ("Candidate type", "by_candidate_type"),
        ("Risk level", "by_risk_level"),
        ("Evidence source", "by_evidence_source"),
        ("Validation rejection", "by_rejection_reason"),
        ("Review action", "by_review_action"),
        ("UI state", "by_ui_state"),
        ("Accepted type", "by_accepted_type"),
        ("Applied type", "by_applied_type"),
        ("Source vote dominant side", "by_source_vote_dominant_side"),
        ("Candidate support source", "by_candidate_support_source"),
        ("Original support source", "by_original_support_source"),    ):
        values = summary.get(key)
        if not isinstance(values, dict) or not values:
            continue
        lines.extend([f"### {title}", "", "| Key | Count |", "| --- | ---: |"])
        for name, count in sorted(values.items()):
            lines.append(f"| `{name}` | `{int(count or 0)}` |")
        lines.append("")
    chapter_items = summary.get("chapter_risk_items") if isinstance(summary.get("chapter_risk_items"), list) else []
    if chapter_items:
        lines.extend(["### Chapter Risk Items", "", "| Bundle | Chapter | Time | Candidates | Review | Risks |", "| --- | --- | --- | ---: | ---: | --- |"])
        for item in chapter_items[:20]:
            if not isinstance(item, dict):
                continue
            risks = item.get("risk_level_counts") if isinstance(item.get("risk_level_counts"), dict) else {}
            risk_text = ", ".join(f"{key}={value}" for key, value in sorted(risks.items())) or "none"
            lines.append(
                "| {bundle} | `{chapter}` {title} | {time_range} | `{candidates}` | `{review}` | `{risks}` |".format(
                    bundle=item.get("bundle_dir", ""),
                    chapter=item.get("chapter_index", ""),
                    title=str(item.get("chapter_title") or "").replace("|", "/"),
                    time_range=str(item.get("chapter_time_range") or "").replace("|", "/"),
                    candidates=int(item.get("candidate_count") or 0),
                    review=int(item.get("review_required_count") or 0),
                    risks=risk_text.replace("|", "/"),
                )
            )
        lines.append("")
    items = summary.get("items_needing_action") if isinstance(summary.get("items_needing_action"), list) else []
    if items:
        lines.extend(["### Items Needing Semantic Action", "", "| Bundle | Status | Next action | Review open | Residual |", "| --- | --- | --- | ---: | ---: |"])
        for item in items[:20]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| {bundle} | `{status}` | `{next_action}` | `{review}` | `{residual}` |".format(
                    bundle=item.get("bundle_dir", ""),
                    status=item.get("status", ""),
                    next_action=item.get("next_action", "none"),
                    review=int(item.get("open_review_required_count") or 0),
                    residual=int(item.get("final_residual_error_total") or 0),
                )
            )
        lines.append("")
    return lines


def _render_handoff_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Content Handoff Pack",
        "",
        f"- Created: `{result.get('created_at')}`",
        "- Review required: `true`",
        "- Publication allowed: `false`",
        "",
    ]
    lines.extend(_semantic_correction_summary_markdown(result.get("semantic_correction_summary")))
    if not result.get("items"):
        lines.append("（暂无可交接的素材卡。）")
    for index, item in enumerate(result.get("items") or [], start=1):
        lines.extend(
            [
                f"## {index}. {item.get('bundle_dir')}",
                "",
                f"- Status: `{item.get('status')}`",
                f"- Circle-of-friends status: `{item.get('circle_of_friends_status')}`",
                f"- Term correction status: `{item.get('term_correction_status') or 'missing'}`",
                f"- Codex term validation: `{item.get('term_validation_status') or 'missing'}`",
                f"- Term validation accepted/rejected: `{int(item.get('accepted_validation_decisions') or 0)}/{int(item.get('rejected_validation_decisions') or 0)}`",
                f"- Term validation blockers: `{_term_blockers_plain(item)}`",
                f"- Optional term action: `{item.get('term_next_action_key') or 'none'}`",
                f"- Optional term commands: `{_term_actions_plain(item)}`",
                f"- Optional term MCP args: `{_term_action_artifacts_plain(item)}`",
                f"- Material card: `{item.get('content_material_card_path')}`",
                f"- Content candidate pack: `{item.get('content_candidate_pack_path')}`",
                f"- Content candidates: `{item.get('content_candidate_count')}`",
                f"- Linked smart-summary chapters: `{item.get('content_candidate_linked_chapter_count', 0)}`",
                f"- Candidate pack safe: `{str(bool(item.get('content_candidate_pack_safe'))).lower()}`",
                f"- Human sample eval: `{item.get('human_sample_eval_status') or 'not_available'}`",
                f"- Human sample eval report: `{item.get('human_sample_eval_markdown_path') or ''}`",
                f"- Candidate usable rate: `{_rate_text(item.get('human_sample_eval_content_candidate_usable_rate'))}`",
                f"- Candidate evidence sufficient rate: `{_rate_text(item.get('human_sample_eval_content_candidate_evidence_sufficient_rate'))}`",
                f"- Multimodal net help rate: `{_rate_text(item.get('human_sample_eval_multimodal_net_help_rate'))}`",
                f"- Publication allowed: `{str(bool(item.get('publication_allowed'))).lower()}`",
                f"- Allowed as fact: `{str(bool(item.get('allowed_as_fact'))).lower()}`",
                "",
                "### Must Fact Check Before Claiming",
                "",
            ]
        )
        for risk in item.get("fact_check_required_for") or []:
            lines.append(f"- `{risk}`")
        lines.extend(["", "### Required Human Confirmation", ""])
        for action in item.get("human_confirmation_required") or []:
            lines.append(f"- `{action}`")
        chapters = item.get("content_candidate_linked_chapters") if isinstance(item.get("content_candidate_linked_chapters"), list) else []
        if chapters:
            lines.extend(["", "### Smart Summary Chapter Links", ""])
            for chapter in chapters[:12]:
                ids = ", ".join(str(value) for value in chapter.get("candidate_ids") or [])
                lines.append(f"- Chapter `{chapter.get('chapter_index')}` `{chapter.get('chapter_title')}` `{chapter.get('chapter_time_range')}` candidates={chapter.get('candidate_count')} ids={ids}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"





def _term_action_artifacts_plain(item: dict[str, Any]) -> str:
    artifacts = item.get("term_optional_next_action_artifacts") if isinstance(item.get("term_optional_next_action_artifacts"), dict) else {}
    parts: list[str] = []
    for key in sorted(artifacts):
        value = str(artifacts.get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    return " | ".join(parts) or "none"
def _term_actions_plain(item: dict[str, Any]) -> str:
    actions = item.get("term_optional_next_actions") if isinstance(item.get("term_optional_next_actions"), list) else []
    return " | ".join(str(action) for action in actions if str(action).strip()) or "none"

def _term_blockers_plain(item: dict[str, Any]) -> str:
    reasons = item.get("validation_rejection_reasons") if isinstance(item.get("validation_rejection_reasons"), list) else []
    parts: list[str] = []
    for row in reasons[:6]:
        if isinstance(row, dict) and row.get("reason"):
            parts.append(f"{row.get('reason')} x{int(row.get('count') or 0)}")
    return ", ".join(parts) if parts else "none"


def _term_blockers_text(item: dict[str, Any]) -> str:
    text = _term_blockers_plain(item)
    return "`none`" if text == "none" else "<br>".join(f"`{part}`" for part in text.split(", "))

def _semantic_correction_text(item: dict[str, Any]) -> str:
    status = str(item.get("semantic_correction_status") or "missing")
    candidates = int(item.get("semantic_correction_candidate_count") or 0)
    accepted = int(item.get("semantic_correction_accepted_count") or 0)
    residual = int(item.get("semantic_correction_final_residual_error_total") or 0)
    summary_status = str(item.get("semantic_correction_summary_impact_status") or "missing")
    summary_residual = int(item.get("semantic_correction_summary_residual_original_total") or 0)
    asset_gate = item.get("semantic_correction_asset_gate") if isinstance(item.get("semantic_correction_asset_gate"), dict) else {}
    gate_status = str(asset_gate.get("status") or "missing")
    return f"{status}; {accepted}/{candidates}; residual={residual}; summary={summary_status}/{summary_residual}; gate={gate_status}"

def _semantic_review_text(item: dict[str, Any]) -> str:
    summary = item.get("semantic_correction_review_closure_summary") if isinstance(item.get("semantic_correction_review_closure_summary"), dict) else {}
    review_count = int(item.get("semantic_correction_review_count") or 0)
    if not summary:
        return f"`open={review_count}`"
    imported = int(summary.get("imported_review_decision_count") or 0)
    closed = int(summary.get("closed_review_decision_count") or 0)
    open_count = int(summary.get("open_review_required_count") or review_count)
    return f"`imported={imported}`<br>`closed={closed}`<br>`open={open_count}`"
def _rate_text(value: Any) -> str:
    return "n/a" if value is None else f"{value}%"

def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
