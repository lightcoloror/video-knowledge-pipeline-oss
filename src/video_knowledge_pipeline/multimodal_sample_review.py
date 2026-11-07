from __future__ import annotations
from .review_export_naming import review_export_script

import html
import json
from pathlib import Path
from typing import Any

from .models import TranscriptCue, now_iso
from .path_utils import file_uri_or_empty as _file_url
from .powershell import quote_powershell_literal as _ps_quote
from .powershell import quote_powershell_literal as _quote_ps_path
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import parse_transcript
from .transcript_sidecar import transcript_source_kind

SCHEMA = "video_knowledge_pipeline.multimodal_sample_review.v1"
NOTES_SCHEMA = "video_knowledge_pipeline.multimodal_sample_review_notes.v1"
SUMMARY_SCHEMA = "video_knowledge_pipeline.multimodal_sample_review_summary.v1"
HUMAN_EVAL_SCHEMA = "video_knowledge_pipeline.human_sample_eval.v1"
MEDIA_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm", ".flv", ".avi", ".m4v", ".ts")



def multimodal_sample_review(
    bundle_dir: str | Path,
    *,
    comparison_json: str | Path | None = None,
    sample_size: int = 30,
    include_missing: bool = True,
    media_path: str | Path | None = None,
    potplayer_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build a static human sampling UI for checking multimodal impact.

    This is intentionally review-only. It does not call model APIs and it does
    not write corrections back to the timeline. Humans export the notes JSON and
    a later import step can decide how to use those labels.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")

    manifest = _read_object(manifest_path)
    timeline_value = read_json(timeline_path)
    if not isinstance(timeline_value, list):
        raise ValueError("timeline.json must be a JSON array")
    timeline = [item for item in timeline_value if isinstance(item, dict)]
    comparison = _read_comparison(root, comparison_json)
    content_candidate_pack = _load_content_candidate_pack(root, manifest)
    content_candidate_by_index = _content_candidates_by_timeline_index(content_candidate_pack)
    transcript_cues, transcript_source = _load_review_transcript_cues(root, manifest)
    rows = _select_samples(
        timeline,
        comparison=comparison,
        sample_size=max(1, int(sample_size or 30)),
        include_missing=include_missing,
        transcript_cues=transcript_cues,
        transcript_source=transcript_source,
        content_candidate_by_index=content_candidate_by_index,
    )
    media_info = _resolve_media_info(root, manifest, media_path)
    potplayer_info = _potplayer_info(root, media_info.get("path"), potplayer_path)
    _attach_potplayer_commands(rows, potplayer_info)
    notes = _notes_template(root, rows)

    json_path = root / "multimodal-sample-review.json"
    todo_path = root / "multimodal-sample-review.todo.json"
    md_path = root / "multimodal-sample-review.md"
    html_path = root / "multimodal-sample-review.html"
    args_path = root / "mcp-multimodal-sample-review.args.json"
    potplayer_script_path = root / "potplayer-jump.ps1"
    potplayer_playlist_path = root / "potplayer-review-playlist.m3u8"
    potplayer_chapters_path = root / "potplayer-review-chapters.txt"
    potplayer_csv_path = root / "potplayer-review-timestamps.csv"
    potplayer_md_path = root / "potplayer-review-timestamps.md"

    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "title": str(manifest.get("title") or root.name),
        "generated_at": now_iso(),
        "write": write,
        "sample_size": int(sample_size or 30),
        "include_missing": bool(include_missing),
        "comparison_json_path": str(_comparison_path(root, comparison_json)),
        "comparison_loaded": bool(comparison),
        "content_candidate_pack": _content_candidate_pack_summary(content_candidate_pack),
        "media": media_info,
        "potplayer": potplayer_info,
        "counts": _counts(timeline, rows),
        "samples": rows,
        "outputs": {
            "json": str(json_path),
            "todo_json": str(todo_path),
            "markdown": str(md_path),
            "html": str(html_path),
            "mcp_args": str(args_path),
            "potplayer_jump_script": str(potplayer_script_path),
            "potplayer_review_playlist": str(potplayer_playlist_path),
            "potplayer_review_chapters": str(potplayer_chapters_path),
            "potplayer_review_timestamps_csv": str(potplayer_csv_path),
            "potplayer_review_timestamps_markdown": str(potplayer_md_path),
        },
        "operator_boundary": {
            "default": "human_sampling_only",
            "api_calls": "none",
            "timeline_writeback": "none",
            "notes_writeback": "save exported JSON, then validate/import through a separate reviewed path",
        },
    }

    if write:
        write_json(json_path, result)
        write_json(todo_path, notes)
        md_path.write_text(_render_markdown(result), encoding="utf-8")
        html_path.write_text(_render_html(result, notes), encoding="utf-8")
        _write_potplayer_jump_script(potplayer_script_path, media_info.get("path") or "", potplayer_info.get("configured_path") or "")
        _write_potplayer_review_pack(
            playlist_path=potplayer_playlist_path,
            chapters_path=potplayer_chapters_path,
            csv_path=potplayer_csv_path,
            md_path=potplayer_md_path,
            rows=rows,
            media_path=media_info.get("path") or "",
        )
        write_json(args_path, {"bundle_dir": str(root), "comparison_json": str(_comparison_path(root, comparison_json)), "sample_size": int(sample_size or 30), "include_missing": bool(include_missing), "media_path": str(media_path or ""), "potplayer_path": str(potplayer_path or ""), "write": True})
        manifest.update(
            {
                "multimodal_sample_review_json": "multimodal-sample-review.json",
                "multimodal_sample_review_todo": "multimodal-sample-review.todo.json",
                "multimodal_sample_review_report": "multimodal-sample-review.md",
                "multimodal_sample_review_html": "multimodal-sample-review.html",
                "mcp_multimodal_sample_review_args": "mcp-multimodal-sample-review.args.json",
                "potplayer_jump_script": "potplayer-jump.ps1",
                "potplayer_review_playlist": "potplayer-review-playlist.m3u8",
                "potplayer_review_chapters": "potplayer-review-chapters.txt",
                "potplayer_review_timestamps_csv": "potplayer-review-timestamps.csv",
                "potplayer_review_timestamps_markdown": "potplayer-review-timestamps.md",
                "multimodal_sample_review_media_path": media_info.get("path") or "",
                "multimodal_sample_review_generated_at": result["generated_at"],
            }
        )
        write_json(manifest_path, manifest)
        run = register_bundle_run(
            root,
            run_type="multimodal_sample_review",
            run_id="multimodal-sample-review",
            status="needs_input" if rows else "needs_review",
            title="多模态人工抽样评估",
            summary=f"Prepared {len(rows)} sample review rows; waiting for human labels.",
            inputs={"comparison_json": str(_comparison_path(root, comparison_json)), "media_path": str(media_path or "")},
            parameters={"sample_size": int(sample_size or 30), "include_missing": bool(include_missing)},
            artifacts=[
                {"key": "sample_review_json", "path": json_path},
                {"key": "sample_review_todo", "path": todo_path},
                {"key": "sample_review_markdown", "path": md_path},
                {"key": "sample_review_html", "path": html_path},
                {"key": "potplayer_playlist", "path": potplayer_playlist_path},
                {"key": "potplayer_timestamps", "path": potplayer_md_path},
                {"key": "mcp_args", "path": args_path},
            ],
            failed_items=_sample_review_failed_items(root, rows),
            retry_command=f".\\scripts\\video-knowledge.ps1 multimodal-sample-review {_quote_ps_path(root)} --sample-size {int(sample_size or 30)}",
            next_actions=[
                "Open multimodal-sample-review.html and label the generated todo JSON.",
                "Save labels as multimodal-sample-review-notes.json, then run validate-multimodal-sample-notes.",
            ],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
        result["run_artifact"] = run
        write_json(json_path, result)
    return result



def validate_multimodal_sample_notes(
    bundle_dir: str | Path,
    *,
    notes_json: str | Path | None = None,
    min_reviewed: int = 10,
    write: bool = True,
) -> dict[str, Any]:
    """Validate and summarize human labels exported from the sample review UI."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    sample_path = root / "multimodal-sample-review.json"
    notes_path = Path(notes_json).expanduser().resolve() if notes_json else root / "multimodal-sample-review-notes.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"sample review json not found: {sample_path}")
    if not notes_path.exists():
        fallback = root / "multimodal-sample-review.todo.json"
        if fallback.exists() and not notes_json:
            notes_path = fallback
        else:
            raise FileNotFoundError(f"sample review notes not found: {notes_path}")

    manifest = _read_object(manifest_path)
    sample = _read_object(sample_path)
    notes = _read_object(notes_path)
    samples = sample.get("samples") if isinstance(sample.get("samples"), list) else []
    sample_by_index = {_int(row.get("index")): row for row in samples if isinstance(row, dict) and _int(row.get("index")) > 0}
    reviews = notes.get("reviews") if isinstance(notes.get("reviews"), list) else []
    rows, issues = _validate_note_rows(reviews, sample_by_index)
    summary = _sample_label_summary(rows, min_reviewed=max(1, int(min_reviewed or 10)), issues=issues, sample_count=len(samples))

    json_path = root / "multimodal-sample-review-summary.json"
    md_path = root / "multimodal-sample-review-summary.md"
    eval_json_path = root / "human-sample-eval.json"
    eval_md_path = root / "human-sample-eval.md"
    args_path = root / "mcp-validate-multimodal-sample-notes.args.json"
    result = {
        "schema": SUMMARY_SCHEMA,
        "bundle_dir": str(root),
        "sample_review_path": str(sample_path),
        "notes_path": str(notes_path),
        "generated_at": now_iso(),
        "write": bool(write),
        "min_reviewed": int(min_reviewed or 10),
        "status": summary["status"],
        "summary": summary,
        "issues": issues,
        "rows": rows,
        "outputs": {
            "json": str(json_path),
            "markdown": str(md_path),
            "human_sample_eval_json": str(eval_json_path),
            "human_sample_eval_markdown": str(eval_md_path),
            "mcp_args": str(args_path),
        },
        "operator_boundary": {
            "timeline_writeback": "none",
            "purpose": "accuracy sampling report only",
            "next_step": "Use the report to decide whether more OCR, ASR, or multimodal review is needed before exporting final notes.",
        },
    }
    human_eval = _human_sample_eval_result(root, result)
    result["human_sample_eval"] = human_eval
    if write:
        write_json(json_path, result)
        md_path.write_text(_render_summary_markdown(result), encoding="utf-8")
        write_json(eval_json_path, human_eval)
        eval_md_path.write_text(_render_human_sample_eval_markdown(human_eval), encoding="utf-8")
        write_json(args_path, {"bundle_dir": str(root), "notes_json": str(notes_path), "min_reviewed": int(min_reviewed or 10), "write": True})
        manifest.update(
            {
                "multimodal_sample_review_summary_json": "multimodal-sample-review-summary.json",
                "multimodal_sample_review_summary_report": "multimodal-sample-review-summary.md",
                "human_sample_eval_json": "human-sample-eval.json",
                "human_sample_eval_report": "human-sample-eval.md",
                "mcp_validate_multimodal_sample_notes_args": "mcp-validate-multimodal-sample-notes.args.json",
                "multimodal_sample_review_summary_status": summary["status"],
                "multimodal_sample_review_summary_generated_at": result["generated_at"],
            }
        )
        if _path_is_child(notes_path, root):
            manifest["multimodal_sample_review_notes"] = notes_path.name
        write_json(manifest_path, manifest)
        run = register_bundle_run(
            root,
            run_type="human_sample_eval",
            run_id="human-sample-eval",
            status=_human_sample_eval_run_status(summary["status"]),
            title="人工抽样质量评估",
            summary=f"Validated {summary.get('labeled_rows', 0)} labeled rows out of {summary.get('sample_count', 0)} samples; status={summary['status']}.",
            inputs={"notes_json": str(notes_path), "sample_review_json": str(sample_path)},
            parameters={"min_reviewed": int(min_reviewed or 10)},
            artifacts=[
                {"key": "summary_json", "path": json_path},
                {"key": "summary_markdown", "path": md_path},
                {"key": "human_sample_eval_json", "path": eval_json_path},
                {"key": "human_sample_eval_markdown", "path": eval_md_path},
                {"key": "mcp_args", "path": args_path},
            ],
            failed_items=_human_sample_eval_failed_items(root, result),
            retry_command=f".\\scripts\\video-knowledge.ps1 validate-multimodal-sample-notes {_quote_ps_path(root)} --notes-json {_quote_ps_path(notes_path)} --min-reviewed {int(min_reviewed or 10)}",
            next_actions=summary.get("next_actions") or [],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
        result["run_artifact"] = run
        write_json(json_path, result)
        write_json(eval_json_path, human_eval)
    return result


def _validate_note_rows(reviews: list[Any], sample_by_index: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    valid = {
        "asr_correct": {"", "yes", "partial", "no", "not_applicable"},
        "ocr_correct": {"", "yes", "partial", "no", "not_applicable"},
        "multimodal_added_key_info": {"", "yes", "some", "no", "not_applicable"},
        "multimodal_error_or_hallucination": {"", "no", "minor", "major", "unclear"},
        "final_note_sufficient": {"", "yes", "partial", "no"},
        "video_checked": {"", "yes", "no", "not_needed"},
        "term_accuracy": {"", "yes", "partial", "no", "not_applicable"},
        "visual_fact_accuracy": {"", "yes", "partial", "no", "not_applicable"},
        "step_completeness": {"", "yes", "partial", "no", "not_applicable"},
        "timestamp_accuracy": {"", "yes", "partial", "no", "not_applicable"},
        "keep_image_required": {"", "yes", "no", "unclear", "not_applicable"},
        "content_candidate_usable": {"", "yes", "partial", "no", "not_applicable"},
        "content_candidate_evidence_sufficient": {"", "yes", "partial", "no", "not_applicable"},
        "overall_label": {"", "correct", "partial", "wrong", "hallucination", "missing", "not_applicable"},
    }
    for pos, raw in enumerate(reviews, start=1):
        if not isinstance(raw, dict):
            issues.append({"row": pos, "key": "invalid_row", "message": "review row must be an object"})
            continue
        idx = _int(raw.get("index"))
        row_issues = []
        if idx <= 0:
            row_issues.append("missing_or_invalid_index")
        elif idx in seen:
            row_issues.append("duplicate_index")
        elif idx not in sample_by_index:
            row_issues.append("index_not_in_sample_review")
        seen.add(idx)
        normalized = {
            "index": idx,
            "sample_type": str(raw.get("sample_type") or (sample_by_index.get(idx, {}) or {}).get("sample_type") or ""),
            "time": str(raw.get("time") or ""),
            "human_notes": str(raw.get("human_notes") or raw.get("notes") or "").strip(),
        }
        for key, allowed in valid.items():
            value = str(raw.get(key) or "").strip()
            normalized[key] = value
            if value not in allowed:
                row_issues.append(f"invalid_{key}")
        normalized["annotated"] = any(str(normalized.get(key) or "").strip() for key in valid) or bool(normalized["human_notes"])
        normalized["labeled"] = bool(str(normalized.get("overall_label") or "").strip())
        normalized["issues"] = row_issues
        if row_issues:
            issues.append({"row": pos, "index": idx, "key": "invalid_review_row", "messages": row_issues})
        rows.append(normalized)
    missing_indexes = sorted(set(sample_by_index) - {row["index"] for row in rows if row.get("index")})
    for idx in missing_indexes:
        issues.append({"index": idx, "key": "missing_review_row", "message": "sample index has no review row"})
    return rows, issues


def _sample_label_summary(rows: list[dict[str, Any]], *, min_reviewed: int, issues: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    annotated = [row for row in rows if row.get("annotated")]
    labeled = [row for row in rows if row.get("labeled")]
    valid_rows = [row for row in rows if not row.get("issues")]
    counts = {
        "asr_correct": _field_counts(rows, "asr_correct"),
        "ocr_correct": _field_counts(rows, "ocr_correct"),
        "multimodal_added_key_info": _field_counts(rows, "multimodal_added_key_info"),
        "multimodal_error_or_hallucination": _field_counts(rows, "multimodal_error_or_hallucination"),
        "final_note_sufficient": _field_counts(rows, "final_note_sufficient"),
        "video_checked": _field_counts(rows, "video_checked"),
        "term_accuracy": _field_counts(rows, "term_accuracy"),
        "visual_fact_accuracy": _field_counts(rows, "visual_fact_accuracy"),
        "step_completeness": _field_counts(rows, "step_completeness"),
        "timestamp_accuracy": _field_counts(rows, "timestamp_accuracy"),
        "keep_image_required": _field_counts(rows, "keep_image_required"),
        "content_candidate_usable": _field_counts(rows, "content_candidate_usable"),
        "content_candidate_evidence_sufficient": _field_counts(rows, "content_candidate_evidence_sufficient"),
        "overall_label": _field_counts(rows, "overall_label"),
        "sample_type": _field_counts(rows, "sample_type"),
    }
    added_total = counts["multimodal_added_key_info"].get("yes", 0) + counts["multimodal_added_key_info"].get("some", 0) + counts["multimodal_added_key_info"].get("no", 0)
    hallucination_total = counts["multimodal_error_or_hallucination"].get("no", 0) + counts["multimodal_error_or_hallucination"].get("minor", 0) + counts["multimodal_error_or_hallucination"].get("major", 0)
    final_total = counts["final_note_sufficient"].get("yes", 0) + counts["final_note_sufficient"].get("partial", 0) + counts["final_note_sufficient"].get("no", 0)
    term_total = _dimension_total(counts["term_accuracy"])
    visual_fact_total = _dimension_total(counts["visual_fact_accuracy"])
    step_total = _dimension_total(counts["step_completeness"])
    timestamp_total = _dimension_total(counts["timestamp_accuracy"])
    keep_image_total = counts["keep_image_required"].get("yes", 0) + counts["keep_image_required"].get("no", 0) + counts["keep_image_required"].get("unclear", 0)
    content_candidate_usable_total = _dimension_total(counts["content_candidate_usable"])
    content_candidate_evidence_total = _dimension_total(counts["content_candidate_evidence_sufficient"])
    multimodal_added_rate = _pct(counts["multimodal_added_key_info"].get("yes", 0) + counts["multimodal_added_key_info"].get("some", 0), added_total)
    any_hallucination_rate = _pct(counts["multimodal_error_or_hallucination"].get("minor", 0) + counts["multimodal_error_or_hallucination"].get("major", 0), hallucination_total)
    net_help_rate = None if multimodal_added_rate is None or any_hallucination_rate is None else round(multimodal_added_rate - any_hallucination_rate, 1)
    status = "ready"
    if any(issue.get("key") != "missing_review_row" for issue in issues):
        status = "invalid"
    elif len(labeled) < min_reviewed:
        status = "needs_more_labels" if annotated else "not_started"
    elif counts["overall_label"].get("hallucination", 0) or counts["multimodal_error_or_hallucination"].get("major", 0):
        status = "needs_model_review"
    return {
        "status": status,
        "sample_count": int(sample_count),
        "review_rows": len(rows),
        "valid_rows": len(valid_rows),
        "annotated_rows": len(annotated),
        "labeled_rows": len(labeled),
        "min_reviewed": int(min_reviewed),
        "unlabeled_rows": max(0, len(rows) - len(labeled)),
        "issue_count": len(issues),
        "counts": counts,
        "rates": {
            "multimodal_added_key_info_rate": multimodal_added_rate,
            "major_hallucination_rate": _pct(counts["multimodal_error_or_hallucination"].get("major", 0), hallucination_total),
            "any_hallucination_rate": any_hallucination_rate,
            "human_sampled_multimodal_net_help_rate": net_help_rate,
            "final_note_acceptable_rate": _pct(counts["final_note_sufficient"].get("yes", 0) + counts["final_note_sufficient"].get("partial", 0), final_total),
            "overall_correct_or_partial_rate": _pct(counts["overall_label"].get("correct", 0) + counts["overall_label"].get("partial", 0), len(labeled)),
            "term_accuracy_accept_rate": _pct(counts["term_accuracy"].get("yes", 0) + counts["term_accuracy"].get("partial", 0), term_total),
            "visual_fact_accuracy_accept_rate": _pct(counts["visual_fact_accuracy"].get("yes", 0) + counts["visual_fact_accuracy"].get("partial", 0), visual_fact_total),
            "step_completeness_accept_rate": _pct(counts["step_completeness"].get("yes", 0) + counts["step_completeness"].get("partial", 0), step_total),
            "timestamp_accuracy_accept_rate": _pct(counts["timestamp_accuracy"].get("yes", 0) + counts["timestamp_accuracy"].get("partial", 0), timestamp_total),
            "keep_image_required_rate": _pct(counts["keep_image_required"].get("yes", 0), keep_image_total),
            "content_candidate_usable_rate": _pct(counts["content_candidate_usable"].get("yes", 0) + counts["content_candidate_usable"].get("partial", 0), content_candidate_usable_total),
            "content_candidate_evidence_sufficient_rate": _pct(counts["content_candidate_evidence_sufficient"].get("yes", 0) + counts["content_candidate_evidence_sufficient"].get("partial", 0), content_candidate_evidence_total),
        },
        "quality_dimensions": {
            "term_accuracy": "术语/工具名是否准确",
            "visual_fact_accuracy": "画面事实是否准确",
            "step_completeness": "操作步骤/流程是否完整",
            "timestamp_accuracy": "时间戳是否能定位到对应内容",
            "keep_image_required": "是否必须保留图片证据而不能降维成文字",
            "content_candidate_usable": "内容素材候选是否可继续加工",
            "content_candidate_evidence_sufficient": "内容素材候选证据是否足够",
        },
        "next_actions": _sample_summary_next_actions(status),
    }


def _dimension_total(counts: dict[str, int]) -> int:
    return counts.get("yes", 0) + counts.get("partial", 0) + counts.get("no", 0)

def _field_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            value = "unlabeled"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) * 100.0 / float(denominator), 1)


def _sample_summary_next_actions(status: str) -> list[str]:
    if status == "invalid":
        return ["Fix invalid rows in multimodal-sample-review-notes.json, then rerun validate-multimodal-sample-notes."]
    if status in {"not_started", "needs_more_labels"}:
        return ["Open multimodal-sample-review.html and label more rows before trusting the comparison rates."]
    if status == "needs_model_review":
        return ["Inspect hallucination rows and consider rerunning targeted multimodal/OCR before final export."]
    return ["Use the summary rates as the human-sampled evidence for multimodal impact."]



def _sample_review_failed_items(root: Path, rows: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for row in rows[:limit]:
        index = _int(row.get("index"))
        failed.append(
            {
                "id": f"sample_review:{index or len(failed) + 1}",
                "index": index,
                "reason": "human_sample_label_required",
                "detail": f"Sample row {index or '?'} needs human labels before multimodal impact rates can be trusted.",
                "sample_type": str(row.get("sample_type") or ""),
                "time_range": str(row.get("time") or row.get("time_range") or ""),
                "review_start": row.get("review_start"),
                "suggested_next_tool": "validate_multimodal_sample_notes",
                "suggested_next_reason": "After labeling, validate notes to produce human-sample-eval metrics.",
                "review_html": str(root / "multimodal-sample-review.html"),
                "todo_json": str(root / "multimodal-sample-review.todo.json"),
                "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 validate-multimodal-sample-notes {_quote_ps_path(root)} --notes-json {_quote_ps_path(root / 'multimodal-sample-review-notes.json')}",
                "evidence_paths": [str(value) for value in row.get("frame_paths") or [] if str(value)][:4],
            }
        )
    return failed


def _human_sample_eval_run_status(status: str) -> str:
    if status == "ready":
        return "completed"
    if status in {"not_started", "needs_more_labels"}:
        return "needs_input"
    if status == "needs_model_review":
        return "needs_review"
    if status == "invalid":
        return "needs_retry"
    return "needs_review"


def _human_sample_eval_failed_items(root: Path, result: dict[str, Any], *, limit: int = 20) -> list[dict[str, Any]]:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    status = str(result.get("status") or summary.get("status") or "unknown")
    failed: list[dict[str, Any]] = []
    if status in {"not_started", "needs_more_labels"}:
        for row in result.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if row.get("labeled"):
                continue
            index = _int(row.get("index"))
            failed.append(
                {
                    "id": f"sample_unlabeled:{index or len(failed) + 1}",
                    "index": index,
                    "reason": "human_sample_label_missing",
                    "detail": "Sample row is not labeled yet.",
                    "sample_type": str(row.get("sample_type") or ""),
                    "time_range": str(row.get("time") or ""),
                    "suggested_next_tool": "multimodal_sample_review",
                    "review_html": str(root / "multimodal-sample-review.html"),
                    "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 validate-multimodal-sample-notes {_quote_ps_path(root)} --notes-json {_quote_ps_path(result.get('notes_path') or root / 'multimodal-sample-review-notes.json')}",
                }
            )
            if len(failed) >= limit:
                return failed
    if status == "invalid":
        for issue in (result.get("issues") or [])[:limit]:
            if not isinstance(issue, dict):
                continue
            failed.append(
                {
                    "id": f"sample_issue:{issue.get('row') or issue.get('index') or len(failed) + 1}",
                    "index": _int(issue.get("index")),
                    "reason": str(issue.get("key") or "invalid_review_row"),
                    "detail": str(issue.get("message") or issue.get("messages") or "Invalid sample review row."),
                    "suggested_next_tool": "validate_multimodal_sample_notes",
                    "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 validate-multimodal-sample-notes {_quote_ps_path(root)} --notes-json {_quote_ps_path(result.get('notes_path') or root / 'multimodal-sample-review-notes.json')}",
                }
            )
    if status == "needs_model_review":
        for row in result.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if row.get("overall_label") != "hallucination" and row.get("multimodal_error_or_hallucination") != "major":
                continue
            index = _int(row.get("index"))
            failed.append(
                {
                    "id": f"sample_model_review:{index or len(failed) + 1}",
                    "index": index,
                    "reason": "human_sample_model_review_required",
                    "detail": "Human sample flagged hallucination or major multimodal error.",
                    "sample_type": str(row.get("sample_type") or ""),
                    "time_range": str(row.get("time") or ""),
                    "suggested_next_tool": "vision_review_triage",
                    "review_html": str(root / "multimodal-sample-review.html"),
                    "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 vision-review-triage {_quote_ps_path(root)} --indexes {index}",
                }
            )
            if len(failed) >= limit:
                return failed
    return failed


def _human_sample_eval_result(root: Path, summary_result: dict[str, Any]) -> dict[str, Any]:
    summary = summary_result.get("summary") if isinstance(summary_result.get("summary"), dict) else {}
    rates = summary.get("rates") if isinstance(summary.get("rates"), dict) else {}
    rows = summary_result.get("rows") if isinstance(summary_result.get("rows"), list) else []
    return {
        "schema": HUMAN_EVAL_SCHEMA,
        "bundle_dir": str(root),
        "generated_at": summary_result.get("generated_at") or now_iso(),
        "status": summary_result.get("status") or summary.get("status") or "unknown",
        "source_summary_json": summary_result.get("outputs", {}).get("json", ""),
        "notes_path": summary_result.get("notes_path", ""),
        "sample_review_path": summary_result.get("sample_review_path", ""),
        "sample_count": summary.get("sample_count", 0),
        "labeled_rows": summary.get("labeled_rows", 0),
        "annotated_rows": summary.get("annotated_rows", 0),
        "quality_dimensions": summary.get("quality_dimensions") or {},
        "rates": {
            "term_accuracy_accept_rate": rates.get("term_accuracy_accept_rate"),
            "visual_fact_accuracy_accept_rate": rates.get("visual_fact_accuracy_accept_rate"),
            "step_completeness_accept_rate": rates.get("step_completeness_accept_rate"),
            "timestamp_accuracy_accept_rate": rates.get("timestamp_accuracy_accept_rate"),
            "keep_image_required_rate": rates.get("keep_image_required_rate"),
            "multimodal_added_key_info_rate": rates.get("multimodal_added_key_info_rate"),
            "any_hallucination_rate": rates.get("any_hallucination_rate"),
            "major_hallucination_rate": rates.get("major_hallucination_rate"),
            "human_sampled_multimodal_net_help_rate": rates.get("human_sampled_multimodal_net_help_rate"),
            "final_note_acceptable_rate": rates.get("final_note_acceptable_rate"),
            "content_candidate_usable_rate": rates.get("content_candidate_usable_rate"),
            "content_candidate_evidence_sufficient_rate": rates.get("content_candidate_evidence_sufficient_rate"),
            "overall_correct_or_partial_rate": rates.get("overall_correct_or_partial_rate"),
        },
        "counts": summary.get("counts") or {},
        "rows": rows,
        "interpretation": _human_eval_interpretation(summary),
        "operator_boundary": {
            "human_sample_only": True,
            "not_randomized_causal_estimate": True,
            "timeline_writeback": "none",
            "use": "Use as human-sampled quality evidence for deciding whether targeted OCR, ASR, multimodal, or manual review is needed.",
        },
    }


def _human_eval_interpretation(summary: dict[str, Any]) -> dict[str, Any]:
    rates = summary.get("rates") if isinstance(summary.get("rates"), dict) else {}
    net = rates.get("human_sampled_multimodal_net_help_rate")
    hallucination = rates.get("any_hallucination_rate")
    if net is None:
        verdict = "insufficient_labels"
    elif net >= 50 and (hallucination is None or hallucination <= 20):
        verdict = "multimodal_helpful"
    elif hallucination is not None and hallucination >= 40:
        verdict = "multimodal_risky"
    elif net > 0:
        verdict = "multimodal_mixed_but_useful"
    else:
        verdict = "no_clear_multimodal_gain"
    return {
        "verdict": verdict,
        "short_text": _human_eval_verdict_text(verdict),
        "caveat": "This is a human sampled review, not a randomized causal accuracy estimate.",
    }


def _human_eval_verdict_text(verdict: str) -> str:
    mapping = {
        "insufficient_labels": "人工标注不足，暂时不能判断多模态收益。",
        "multimodal_helpful": "抽样显示多模态明显补充关键信息，且错误率可控。",
        "multimodal_risky": "抽样显示多模态错误/幻觉风险较高，需要定向复核。",
        "multimodal_mixed_but_useful": "抽样显示多模态有帮助，但仍需关注错误和边界。",
        "no_clear_multimodal_gain": "抽样没有显示明确多模态收益，优先检查 ASR/OCR/人工复核。",
    }
    return mapping.get(verdict, verdict)


def _render_human_sample_eval_markdown(result: dict[str, Any]) -> str:
    rates = result.get("rates") if isinstance(result.get("rates"), dict) else {}
    interpretation = result.get("interpretation") if isinstance(result.get("interpretation"), dict) else {}
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    lines = [
        "# 人工抽样质量评估",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Status: `{result.get('status', '')}`",
        f"- Reviewed: `{result.get('labeled_rows', 0)}/{result.get('sample_count', 0)}` labeled, `{result.get('annotated_rows', 0)}` annotated",
        f"- Verdict: `{interpretation.get('verdict', '')}` - {interpretation.get('short_text', '')}",
        "- Boundary: human-sampled quality evidence only; not a randomized causal estimate.",
        "",
        "## 质量维度",
        "",
        "| Dimension | Rate |",
        "| --- | ---: |",
        f"| 术语/工具名准确率 | `{_rate_text(rates.get('term_accuracy_accept_rate'))}` |",
        f"| 画面事实准确率 | `{_rate_text(rates.get('visual_fact_accuracy_accept_rate'))}` |",
        f"| 步骤完整率 | `{_rate_text(rates.get('step_completeness_accept_rate'))}` |",
        f"| 时间戳准确率 | `{_rate_text(rates.get('timestamp_accuracy_accept_rate'))}` |",
        f"| 必须保留图片比例 | `{_rate_text(rates.get('keep_image_required_rate'))}` |",
        f"| 多模态补充关键信息率 | `{_rate_text(rates.get('multimodal_added_key_info_rate'))}` |",
        f"| 多模态任意错误/幻觉率 | `{_rate_text(rates.get('any_hallucination_rate'))}` |",
        f"| 多模态净帮助率 proxy | `{_rate_text(rates.get('human_sampled_multimodal_net_help_rate'))}` |",
        f"| 最终笔记可接受率 | `{_rate_text(rates.get('final_note_acceptable_rate'))}` |",
        f"| 内容素材候选可用率 | `{_rate_text(rates.get('content_candidate_usable_rate'))}` |",
        f"| 内容素材证据充分率 | `{_rate_text(rates.get('content_candidate_evidence_sufficient_rate'))}` |",
        "",
        "## 样本明细",
        "",
        "| Index | Type | Overall | Term | Visual fact | Step | Timestamp | Keep image | Candidate usable | Candidate evidence | Notes |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows[:200]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("index", "")),
                    f"`{row.get('sample_type', '')}`",
                    f"`{row.get('overall_label', '') or '-'}`",
                    f"`{row.get('term_accuracy', '') or '-'}`",
                    f"`{row.get('visual_fact_accuracy', '') or '-'}`",
                    f"`{row.get('step_completeness', '') or '-'}`",
                    f"`{row.get('timestamp_accuracy', '') or '-'}`",
                    f"`{row.get('keep_image_required', '') or '-'}`",
                    f"`{row.get('content_candidate_usable', '') or '-'}`",
                    f"`{row.get('content_candidate_evidence_sufficient', '') or '-'}`",
                    _md_cell(row.get("human_notes") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"

def _render_summary_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    rates = summary.get("rates") if isinstance(summary.get("rates"), dict) else {}
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    lines = [
        "# 多模态抽样标注汇总",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Notes: `{result.get('notes_path', '')}`",
        f"- Status: `{result.get('status', '')}`",
        f"- Reviewed: `{summary.get('labeled_rows', 0)}/{summary.get('sample_count', 0)}` labeled, `{summary.get('annotated_rows', 0)}` annotated",
        f"- Issues: `{summary.get('issue_count', 0)}`",
        "",
        "## 关键比例",
        "",
        f"- 多模态补充关键信息率：`{_rate_text(rates.get('multimodal_added_key_info_rate'))}`",
        f"- 多模态任意幻觉/错误率：`{_rate_text(rates.get('any_hallucination_rate'))}`",
        f"- 多模态严重幻觉/错误率：`{_rate_text(rates.get('major_hallucination_rate'))}`",
        f"- 最终笔记可接受率：`{_rate_text(rates.get('final_note_acceptable_rate'))}`",
        f"- 总体 correct/partial 比例：`{_rate_text(rates.get('overall_correct_or_partial_rate'))}`",
        f"- 术语/工具名准确率：`{_rate_text(rates.get('term_accuracy_accept_rate'))}`",
        f"- 画面事实准确率：`{_rate_text(rates.get('visual_fact_accuracy_accept_rate'))}`",
        f"- 步骤完整率：`{_rate_text(rates.get('step_completeness_accept_rate'))}`",
        f"- 时间戳准确率：`{_rate_text(rates.get('timestamp_accuracy_accept_rate'))}`",
        f"- 必须保留图片比例：`{_rate_text(rates.get('keep_image_required_rate'))}`",
        f"- 多模态净帮助率 proxy：`{_rate_text(rates.get('human_sampled_multimodal_net_help_rate'))}`",
        "",
        "## 字段分布",
        "",
    ]
    for field in ("overall_label", "video_checked", "term_accuracy", "visual_fact_accuracy", "step_completeness", "timestamp_accuracy", "keep_image_required", "multimodal_added_key_info", "multimodal_error_or_hallucination", "final_note_sufficient", "asr_correct", "ocr_correct", "sample_type"):
        lines.extend([f"### {field}", "", "| Value | Count |", "| --- | ---: |"])
        for key, value in sorted((counts.get(field) or {}).items()):
            lines.append(f"| `{key}` | {value} |")
        lines.append("")
    issues = result.get("issues") if isinstance(result.get("issues"), list) else []
    if issues:
        lines.extend(["## 问题", "", "| Index | Key | Message |", "| ---: | --- | --- |"])
        for issue in issues[:100]:
            lines.append(f"| {issue.get('index', issue.get('row', ''))} | `{issue.get('key', '')}` | {_md_cell(issue.get('message') or ', '.join(issue.get('messages') or []))} |")
        lines.append("")
    lines.extend(["## 下一步", ""])
    for action in summary.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def _rate_text(value: Any) -> str:
    return "n/a" if value is None else f"{value}%"


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _path_is_child(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

def _select_samples(
    timeline: list[dict[str, Any]],
    *,
    comparison: dict[str, Any],
    sample_size: int,
    include_missing: bool,
    transcript_cues: list[TranscriptCue] | None = None,
    transcript_source: str = "",
    content_candidate_by_index: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_index = {_index(item, pos): item for pos, item in enumerate(timeline, start=1)}
    preferred = _comparison_indexes(comparison)
    content_candidate_by_index = content_candidate_by_index or {}
    buckets: dict[str, list[dict[str, Any]]] = {
        "content_candidate": [],
        "comparison_example": [],
        "temporal": [],
        "visual_with_ocr": [],
        "visual_without_ocr": [],
        "missing_visual": [],
    }
    for idx, candidate in sorted(content_candidate_by_index.items()):
        item = by_index.get(idx)
        if item:
            buckets["content_candidate"].append(
                _sample_row(item, idx, "content_candidate", transcript_cues=transcript_cues, transcript_source=transcript_source, content_candidate=candidate)
            )
    for idx in preferred:
        item = by_index.get(idx)
        if item:
            buckets["comparison_example"].append(
                _sample_row(item, idx, "comparison_example", transcript_cues=transcript_cues, transcript_source=transcript_source, content_candidate=content_candidate_by_index.get(idx))
            )
    preferred_set = set(preferred)
    for pos, item in enumerate(timeline, start=1):
        idx = _index(item, pos)
        if idx in preferred_set:
            continue
        has_visual = _non_empty(item.get("visual_understanding"))
        has_temporal = _non_empty(item.get("temporal_visual_understanding"))
        has_text = bool(str(item.get("visual_text") or "").strip()) or _non_empty(item.get("structured_visual"))
        route = str(item.get("visual_route") or "unknown")
        candidate = content_candidate_by_index.get(idx)
        if has_temporal:
            buckets["temporal"].append(_sample_row(item, idx, "temporal", transcript_cues=transcript_cues, transcript_source=transcript_source, content_candidate=candidate))
        elif has_visual and has_text:
            buckets["visual_with_ocr"].append(_sample_row(item, idx, "visual_with_ocr", transcript_cues=transcript_cues, transcript_source=transcript_source, content_candidate=candidate))
        elif has_visual:
            buckets["visual_without_ocr"].append(_sample_row(item, idx, "visual_without_ocr", transcript_cues=transcript_cues, transcript_source=transcript_source, content_candidate=candidate))
        elif include_missing and route in {"semantic_frame", "temporal_sequence", "mixed"}:
            buckets["missing_visual"].append(_sample_row(item, idx, "missing_visual", transcript_cues=transcript_cues, transcript_source=transcript_source, content_candidate=candidate))

    quotas = _quotas(sample_size)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name in ("content_candidate", "comparison_example", "temporal", "visual_with_ocr", "visual_without_ocr", "missing_visual"):
        limit = quotas.get(name, sample_size)
        for row in buckets[name][:limit]:
            if row["index"] in seen or len(selected) >= sample_size:
                continue
            selected.append(row)
            seen.add(row["index"])
    if len(selected) < sample_size:
        for name in ("content_candidate", "visual_with_ocr", "visual_without_ocr", "temporal", "missing_visual"):
            for row in buckets[name]:
                if row["index"] in seen or len(selected) >= sample_size:
                    continue
                selected.append(row)
                seen.add(row["index"])
    selected.sort(key=lambda row: int(row.get("index") or 0))
    return selected


def _quotas(sample_size: int) -> dict[str, int]:
    if sample_size <= 8:
        return {"content_candidate": 2, "comparison_example": 2, "temporal": 2, "visual_with_ocr": 2, "visual_without_ocr": 2, "missing_visual": 2}
    return {
        "content_candidate": max(3, sample_size // 5),
        "comparison_example": min(6, sample_size),
        "temporal": max(3, sample_size // 6),
        "visual_with_ocr": max(6, sample_size // 3),
        "visual_without_ocr": max(6, sample_size // 3),
        "missing_visual": max(4, sample_size // 5),
    }


def _sample_row(
    item: dict[str, Any],
    index: int,
    sample_type: str,
    *,
    transcript_cues: list[TranscriptCue] | None = None,
    transcript_source: str = "",
    content_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = content_candidate if isinstance(content_candidate, dict) else {}
    candidate_citations = _content_candidate_citation_rows(candidate)
    candidate_evidence = _dedupe([str(value) for value in candidate.get("evidence_paths") or [] if str(value)] + _content_candidate_citation_evidence_paths(candidate_citations))
    return {
        "index": index,
        "sample_type": sample_type,
        "start": item.get("start"),
        "end": item.get("end"),
        "frame_time": item.get("midpoint"),
        **_review_start_fields(item, transcript_cues=transcript_cues, transcript_source=transcript_source),
        "visual_route": str(item.get("visual_route") or "unknown"),
        "tagger_tags": _list_text(item.get("tagger_tags") or item.get("tagger_annotations")),
        "quality_issues": _list_text(item.get("quality_issues")),
        "transcript_excerpt": _excerpt(str(item.get("corrected_transcript") or item.get("transcript") or item.get("text") or ""), 420),
        "visual_text_excerpt": _excerpt(str(item.get("visual_text") or ""), 420),
        "structured_visual_excerpt": _excerpt(_compact_text(item.get("structured_visual")), 520),
        "visual_understanding_excerpt": _excerpt(_compact_text(item.get("visual_understanding")), 700),
        "temporal_visual_understanding_excerpt": _excerpt(_compact_text(item.get("temporal_visual_understanding")), 700),
        "content_candidate_id": str(candidate.get("id") or ""),
        "content_candidate_types": [str(value) for value in candidate.get("candidate_types") or [] if str(value)],
        "content_candidate_viewpoint": _excerpt(str(candidate.get("viewpoint") or ""), 520),
        "content_candidate_case_or_example": _excerpt(str(candidate.get("case_or_example") or ""), 420),
        "content_candidate_reusable_quote": _excerpt(str(candidate.get("reusable_quote") or ""), 240),
        "content_candidate_evidence_paths": candidate_evidence[:8],
        "content_candidate_evidence_citations": candidate_citations,
        "content_candidate_citation_summary": _content_candidate_citation_summary(candidate_citations),
        "content_candidate_fact_status": str(candidate.get("fact_check_status") or candidate.get("source_fact_status") or ""),
        "frame_paths": _dedupe(_evidence_paths(item) + candidate_evidence),
        "review_fields": _blank_review_fields(),
    }


def _content_candidate_citation_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows = candidate.get("evidence_citations") if isinstance(candidate.get("evidence_citations"), list) else []
    for row in raw_rows[:8]:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "source_type": str(row.get("source_type") or "unknown"),
                "time": str(row.get("time") or ""),
                "timeline_indexes": _citation_timeline_indexes(row),
                "text": _excerpt(str(row.get("text") or ""), 220),
                "evidence_paths": [str(value) for value in row.get("evidence_paths") or [] if str(value)][:4],
            }
        )
    return rows


def _citation_timeline_indexes(row: dict[str, Any]) -> list[int]:
    values = row.get("timeline_indexes") if isinstance(row.get("timeline_indexes"), list) else []
    indexes: list[int] = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in indexes:
            indexes.append(index)
    return indexes


def _content_candidate_citation_evidence_paths(citations: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for citation in citations:
        for value in citation.get("evidence_paths") or []:
            path = str(value)
            if path and path not in paths:
                paths.append(path)
    return paths


def _content_candidate_citation_summary(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    first = citations[0]
    suffix = f" +{len(citations) - 1}" if len(citations) > 1 else ""
    time = str(first.get("time") or "unknown_time")
    source_type = str(first.get("source_type") or "unknown")
    text = _excerpt(str(first.get("text") or ""), 120)
    return f"{time} {source_type}: {text}{suffix}"

def _review_start_fields(item: dict[str, Any], *, transcript_cues: list[TranscriptCue] | None = None, transcript_source: str = "") -> dict[str, Any]:
    asr = _asr_review_time(item, transcript_cues or [], source=transcript_source)
    if asr is not None:
        return {"review_start": asr[0], "review_start_source": asr[1]}
    for key in ("review_start", "transcript_start", "subtitle_start", "asr_start"):
        if key in item and item.get(key) is not None:
            return {"review_start": _seconds_value(item.get(key)), "review_start_source": key}
    tagger = _tagger_review_time(item)
    if tagger is not None:
        return {"review_start": tagger[0], "review_start_source": tagger[1]}
    return {"review_start": _seconds_value(item.get("start")), "review_start_source": "segment_start_fallback"}


def _asr_review_time(item: dict[str, Any], cues: list[TranscriptCue], *, source: str = "") -> tuple[float, str] | None:
    if not cues:
        return None
    start = _seconds_value(item.get("start"))
    end = _seconds_value(item.get("end"))
    if end <= start:
        end = start
    overlapping = [cue for cue in cues if cue.end >= start and cue.start <= end and str(cue.text or "").strip()]
    if not overlapping:
        return None
    excerpt = _normalise_match_text(item.get("corrected_transcript") or item.get("transcript") or item.get("text") or "")
    if excerpt:
        for cue in sorted(overlapping, key=lambda row: row.start):
            cue_text = _normalise_match_text(cue.text)
            if _text_matches_excerpt(cue_text, excerpt):
                return (max(0.0, float(cue.start)), _transcript_start_source(source, exact=True))
    first = sorted(overlapping, key=lambda row: row.start)[0]
    return (max(0.0, float(first.start)), _transcript_start_source(source, exact=False))


def _transcript_start_source(source: str, *, exact: bool) -> str:
    if source == "corrected":
        return "corrected_transcript_start" if exact else "corrected_transcript_overlap_start"
    if source == "timeline_fallback":
        return "timeline_transcript_start" if exact else "timeline_transcript_overlap_start"
    return "asr_segment_start" if exact else "asr_overlap_start"


def _text_matches_excerpt(cue_text: str, excerpt: str) -> bool:
    if not cue_text or not excerpt:
        return False
    if len(cue_text) >= 4 and cue_text in excerpt:
        return True
    if len(excerpt) >= 4 and excerpt in cue_text:
        return True
    return False


def _normalise_match_text(value: Any) -> str:
    text = str(value or "").lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

def _tagger_review_time(item: dict[str, Any]) -> tuple[float, str] | None:
    rows: list[tuple[str, dict[str, Any]]] = []
    for row in _dict_rows(item.get("tagger_time_axis")):
        rows.append(("tagger_time_axis", row))
    for row in _dict_rows(item.get("tagger_annotations")):
        rows.append(("tagger_annotations", row))
    integrated = item.get("integrated_visual") if isinstance(item.get("integrated_visual"), dict) else {}
    for row in _dict_rows(integrated.get("tagger_time_axis")):
        rows.append(("integrated_visual.tagger_time_axis", row))
    for row in _dict_rows(integrated.get("tagger_annotations")):
        rows.append(("integrated_visual.tagger_annotations", row))

    candidates: list[tuple[float, str]] = []
    for source, row in rows:
        if row.get("start") is not None:
            candidates.append((_seconds_value(row.get("start")), f"{source}.start"))
        elif row.get("time") is not None:
            candidates.append((_seconds_value(row.get("time")), f"{source}.time"))
        elif row.get("timestamp") is not None:
            candidates.append((_seconds_value(row.get("timestamp")), f"{source}.timestamp"))
        elif row.get("midpoint") is not None:
            candidates.append((_seconds_value(row.get("midpoint")), f"{source}.midpoint"))
    if not candidates:
        return None
    segment_start = _seconds_value(item.get("start"))
    segment_end = _seconds_value(item.get("end"))
    in_segment = [(value, source) for value, source in candidates if segment_start <= value <= segment_end]
    if in_segment:
        return sorted(in_segment, key=lambda pair: (abs(pair[0] - segment_start), pair[0]))[0]
    return sorted(candidates, key=lambda pair: (abs(pair[0] - segment_start), pair[0]))[0]


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _review_start(row: dict[str, Any]) -> float:
    explicit = row.get("review_start")
    if explicit is not None:
        return _seconds_value(explicit)
    return _seconds_value(row.get("start"))

def _blank_review_fields() -> dict[str, str]:
    return {
        "asr_correct": "",
        "ocr_correct": "",
        "multimodal_added_key_info": "",
        "multimodal_error_or_hallucination": "",
        "final_note_sufficient": "",
        "overall_label": "",
        "video_checked": "",
        "term_accuracy": "",
        "visual_fact_accuracy": "",
        "step_completeness": "",
        "timestamp_accuracy": "",
        "keep_image_required": "",
        "content_candidate_usable": "",
        "content_candidate_evidence_sufficient": "",
        "human_notes": "",
    }


def _notes_template(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": NOTES_SCHEMA,
        "bundle_dir": str(root),
        "generated_at": now_iso(),
        "instructions": [
            "在 multimodal-sample-review.html 中完成抽样标注。",
            "overall_label 可用 correct / partial / wrong / hallucination / missing / not_applicable。",
            "保存为 multimodal-sample-review-notes.json 后，再通过后续 reviewed import 工具写回。",
        ],
        "reviews": [
            {
                "index": row["index"],
                "sample_type": row["sample_type"],
                "time": _time_label(row),
                **_blank_review_fields(),
            }
            for row in rows
        ],
    }


def _counts(timeline: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    content_candidate_rows = [row for row in rows if str(row.get("sample_type") or "") == "content_candidate" or row.get("content_candidate_id")]
    return {
        "timeline_items": len(timeline),
        "sample_count": len(rows),
        "content_candidate_sample_count": len(content_candidate_rows),
        "items_with_visual_understanding": sum(1 for item in timeline if _non_empty(item.get("visual_understanding"))),
        "items_with_temporal_understanding": sum(1 for item in timeline if _non_empty(item.get("temporal_visual_understanding"))),
        "items_with_visual_text_or_structure": sum(1 for item in timeline if str(item.get("visual_text") or "").strip() or _non_empty(item.get("structured_visual"))),
        "sample_types": _type_counts(rows),
    }


def _type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("sample_type") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out



def _resolve_media_info(root: Path, manifest: dict[str, Any], explicit: str | Path | None) -> dict[str, Any]:
    if explicit:
        path = _normalise_candidate_path(root, str(explicit))
        return {"path": path, "source": "explicit", "exists": Path(path).exists()}
    for key in ("media_path", "local_media_path", "source_media_path", "video_path", "local_video_path", "source_path", "path"):
        value = manifest.get(key)
        if isinstance(value, str) and _looks_like_media_path(value):
            path = _normalise_candidate_path(root, value)
            return {"path": path, "source": f"manifest.{key}", "exists": Path(path).exists()}
    candidate = _first_media_path(manifest, root)
    if candidate:
        return candidate
    for name in ("source-artifacts.json", "source_manifest.json", "source-package.json"):
        path = root / name
        if path.exists():
            candidate = _first_media_path(_read_object(path), root, source_prefix=name)
            if candidate:
                return candidate
    return {"path": "", "source": "not_found", "exists": False}


def _first_media_path(value: Any, root: Path, *, source_prefix: str = "manifest") -> dict[str, Any]:
    stack: list[tuple[Any, str]] = [(value, source_prefix)]
    while stack:
        current, source = stack.pop(0)
        if isinstance(current, dict):
            for key, child in current.items():
                child_source = f"{source}.{key}"
                if isinstance(child, str) and _looks_like_media_path(child):
                    path = _normalise_candidate_path(root, child)
                    return {"path": path, "source": child_source, "exists": Path(path).exists()}
                if isinstance(child, (dict, list)):
                    stack.append((child, child_source))
        elif isinstance(current, list):
            for idx, child in enumerate(current[:200]):
                child_source = f"{source}[{idx}]"
                if isinstance(child, str) and _looks_like_media_path(child):
                    path = _normalise_candidate_path(root, child)
                    return {"path": path, "source": child_source, "exists": Path(path).exists()}
                if isinstance(child, (dict, list)):
                    stack.append((child, child_source))
    return {}


def _looks_like_media_path(value: str) -> bool:
    lowered = value.strip().lower()
    return any(lowered.endswith(ext) for ext in MEDIA_EXTENSIONS)


def _normalise_candidate_path(root: Path, value: str) -> str:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _potplayer_info(root: Path, media_path: Any, explicit: str | Path | None) -> dict[str, Any]:
    script_path = root / "potplayer-jump.ps1"
    configured = str(explicit or "").strip()
    return {
        "script_path": str(script_path),
        "configured_path": configured,
        "media_path": str(media_path or ""),
        "launch_mode": "copy_powershell_command",
        "note": "Static HTML cannot directly launch PotPlayer; copy/run the generated PowerShell command.",
    }


def _attach_potplayer_commands(rows: list[dict[str, Any]], potplayer: dict[str, Any]) -> None:
    script_path = str(potplayer.get("script_path") or "")
    media_path = str(potplayer.get("media_path") or "")
    configured_player = str(potplayer.get("configured_path") or "")
    for row in rows:
        seconds = _review_start(row)
        command = ""
        if media_path:
            command_parts = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                _ps_quote(script_path),
                "-Seconds",
                _seconds_text(seconds),
                "-MediaPath",
                _ps_quote(media_path),
            ]
            if configured_player:
                command_parts.extend(["-PotPlayerPath", _ps_quote(configured_player)])
            command = " ".join(command_parts)
        row["video_review"] = {
            "media_path": media_path,
            "start_seconds": seconds,
            "segment_start_seconds": _seconds_value(row.get("start")),
            "frame_time_seconds": _frame_time(row),
            "end_seconds": _seconds_value(row.get("end")),
            "potplayer_command": command,
            "potplayer_jump_script": script_path,
        }
        row["potplayer_command"] = command


def _frame_time(row: dict[str, Any]) -> float:
    explicit = row.get("frame_time")
    if explicit is not None:
        return _seconds_value(explicit)
    start = _seconds_value(row.get("start"))
    end = _seconds_value(row.get("end"))
    if end > start:
        return (start + end) / 2
    return start


def _seconds_value(value: Any) -> float:
    try:
        return max(0.0, float(value or 0))
    except Exception:
        return 0.0


def _seconds_text(value: float) -> str:
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".") or "0"



def _write_potplayer_jump_script(path: Path, default_media_path: str, default_potplayer_path: str) -> None:
    content = f'''param(
    [double]$Seconds = 0,
    [string]$MediaPath = {_ps_quote(default_media_path)},
    [string]$PotPlayerPath = {_ps_quote(default_potplayer_path)}
)
$ErrorActionPreference = "Stop"

function Format-SeekTime([double]$Value) {{
    if ($Value -lt 0) {{ $Value = 0 }}
    $ts = [TimeSpan]::FromSeconds($Value)
    return "{{0:00}}:{{1:00}}:{{2:00}}.{{3:000}}" -f [int]$ts.TotalHours, $ts.Minutes, $ts.Seconds, $ts.Milliseconds
}}

function Resolve-PotPlayer([string]$ConfiguredPath) {{
    $candidates = @()
    if ($ConfiguredPath) {{ $candidates += $ConfiguredPath }}
    if ($env:VKP_POTPLAYER_PATH) {{ $candidates += $env:VKP_POTPLAYER_PATH }}
    $candidates += @(
        "C:\\Program Files\\DAUM\\PotPlayer\\PotPlayerMini64.exe",
        "C:\\Program Files\\PotPlayer\\PotPlayerMini64.exe",
        "C:\\Program Files (x86)\\DAUM\\PotPlayer\\PotPlayerMini.exe",
        "C:\\Program Files (x86)\\PotPlayer\\PotPlayerMini.exe"
    )
    foreach ($candidate in $candidates) {{
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {{ return $candidate }}
    }}
    $cmd = Get-Command PotPlayerMini64.exe -ErrorAction SilentlyContinue
    if ($cmd) {{ return $cmd.Source }}
    $cmd = Get-Command PotPlayerMini.exe -ErrorAction SilentlyContinue
    if ($cmd) {{ return $cmd.Source }}
    throw "PotPlayer executable not found. Pass -PotPlayerPath or set VKP_POTPLAYER_PATH."
}}

if (-not $MediaPath) {{
    throw "MediaPath is required. Regenerate the review page with --media-path or pass -MediaPath."
}}
if (-not (Test-Path -LiteralPath $MediaPath)) {{
    throw "MediaPath not found: $MediaPath"
}}
$player = Resolve-PotPlayer $PotPlayerPath
$seek = Format-SeekTime $Seconds
Start-Process -FilePath $player -ArgumentList @($MediaPath, "/seek=$seek")
'''
    path.write_text(content, encoding="utf-8")


def _write_potplayer_review_pack(*, playlist_path: Path, chapters_path: Path, csv_path: Path, md_path: Path, rows: list[dict[str, Any]], media_path: str) -> None:
    _write_review_playlist(playlist_path, rows, media_path)
    _write_review_chapters(chapters_path, rows)
    _write_review_csv(csv_path, rows, media_path)
    _write_review_markdown(md_path, rows, media_path, playlist_path, chapters_path, csv_path)


def _write_review_playlist(path: Path, rows: list[dict[str, Any]], media_path: str) -> None:
    lines = ["#EXTM3U", "# VKP review playlist. If your player ignores start-time, use potplayer-review-chapters.txt or potplayer-review-timestamps.md."]
    for row in rows:
        start = _review_start(row)
        end = _seconds_value(row.get("end"))
        duration = max(1, int(round(end - start))) if end > start else 8
        title = _review_title(row)
        lines.append(f"#EXTINF:{duration},{title}")
        lines.append(f"#EXTVLCOPT:start-time={_seconds_text(start)}")
        if end > start:
            lines.append(f"#EXTVLCOPT:stop-time={_seconds_text(end)}")
        if media_path:
            lines.append(media_path)
        else:
            lines.append("# MEDIA_PATH_NOT_FOUND")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8-sig")


def _write_review_chapters(path: Path, rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for pos, row in enumerate(rows, start=1):
        number = f"{pos:02d}"
        lines.append(f"CHAPTER{number}={_chapter_time(_review_start(row))}")
        lines.append(f"CHAPTER{number}NAME={_review_title(row)}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8-sig")


def _write_review_csv(path: Path, rows: list[dict[str, Any]], media_path: str) -> None:
    import csv

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "review_start", "segment_start", "frame_time", "end", "review_start_hms", "segment_start_hms", "frame_time_hms", "end_hms", "sample_type", "visual_route", "issues", "tags", "transcript_excerpt", "media_path", "potplayer_command"])
        for row in rows:
            writer.writerow([
                row.get("index"),
                _seconds_text(_review_start(row)),
                _seconds_text(_seconds_value(row.get("start"))),
                _seconds_text(_frame_time(row)),
                _seconds_text(_seconds_value(row.get("end"))),
                _chapter_time(_review_start(row)),
                _chapter_time(_seconds_value(row.get("start"))),
                _chapter_time(_frame_time(row)),
                _chapter_time(_seconds_value(row.get("end"))),
                row.get("sample_type") or "",
                row.get("visual_route") or "",
                "; ".join(row.get("quality_issues") or []),
                "; ".join(row.get("tagger_tags") or []),
                row.get("transcript_excerpt") or "",
                media_path,
                row.get("potplayer_command") or "",
            ])


def _write_review_markdown(md_path: Path, rows: list[dict[str, Any]], media_path: str, playlist_path: Path, chapters_path: Path, csv_path: Path) -> None:
    lines = [
        "# PotPlayer 待审核时间戳清单",
        "",
        f"- 原视频: `{media_path or '未找到'}`",
        f"- 播放列表: `{playlist_path.name}`",
        f"- 章节文件: `{chapters_path.name}`",
        f"- CSV: `{csv_path.name}`",
        "",
        "## 使用方式",
        "",
        "1. 优先尝试用 PotPlayer 打开 `potplayer-review-playlist.m3u8`。",
        "2. 如果播放器不按时间跳转，就打开本 Markdown 或 CSV，按时间戳在 PotPlayer 中跳转。",
        "3. `potplayer-review-chapters.txt` 使用 OGM chapter 格式，可作为章节/书签导入候选。",
        "",
        "## 时间戳",
        "",
        "| # | Time | Type | Route | Issues | Transcript |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('index')} | `{_chapter_time(_review_start(row))}` jump / `{_chapter_time(_seconds_value(row.get('start')))}`-`{_chapter_time(_seconds_value(row.get('end')))}` segment | "
            f"`{_md_cell(row.get('sample_type'))}` | `{_md_cell(row.get('visual_route'))}` | "
            f"{_md_cell('; '.join(row.get('quality_issues') or [])) or '-'} | {_md_cell(row.get('transcript_excerpt'))} |"
        )
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _review_title(row: dict[str, Any]) -> str:
    index = row.get("index") or ""
    sample_type = str(row.get("sample_type") or "unknown")
    route = str(row.get("visual_route") or "unknown")
    transcript = str(row.get("transcript_excerpt") or "").replace("\n", " ").strip()
    if len(transcript) > 42:
        transcript = transcript[:42] + "..."
    return f"#{index} {_chapter_time(_review_start(row))} {sample_type}/{route} {transcript}".strip()


def _chapter_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"

def _render_markdown(result: dict[str, Any]) -> str:
    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    lines = [
        "# 多模态抽样标注",
        "",
        f"- Bundle: `{result['bundle_dir']}`",
        f"- Timeline: `{counts.get('timeline_items', 0)}`",
        f"- Sample count: `{counts.get('sample_count', 0)}`",
        f"- Comparison loaded: `{result.get('comparison_loaded')}`",
        f"- Media path: `{(result.get('media') or {}).get('path', '')}`",
        f"- PotPlayer jump script: `{Path((result.get('outputs') or {}).get('potplayer_jump_script', '')).name}`",
        f"- PotPlayer review playlist: `{Path((result.get('outputs') or {}).get('potplayer_review_playlist', '')).name}`",
        "",
        "## 用途",
        "",
        "这个页面用于人工抽样判断：多模态是否补充了 OCR/ASR 没有捕捉到的信息，是否产生幻觉，以及最终人类可读文件是否足够可靠。",
        "",
        "## 标注文件",
        "",
        f"- HTML: `{Path(result['outputs']['html']).name}`",
        f"- Todo JSON: `{Path(result['outputs']['todo_json']).name}`",
        f"- PotPlayer playlist: `{Path(result['outputs'].get('potplayer_review_playlist', '')).name}`",
        f"- Timestamp list: `{Path(result['outputs'].get('potplayer_review_timestamps_markdown', '')).name}`",
        "",
        "## 抽样分布",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((counts.get("sample_types") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## 样本", "", "| Index | Time | Type | Route | Tags |", "| ---: | --- | --- | --- | --- |"])
    for row in result.get("samples") or []:
        tags = ", ".join(str(tag) for tag in row.get("tagger_tags") or [])
        lines.append(f"| {row.get('index')} | {_time_label(row)} | `{row.get('sample_type')}` | `{row.get('visual_route')}` | {tags or '-'} |")
    return "\n".join(lines).rstrip() + "\n"



def _render_html(result: dict[str, Any], notes: dict[str, Any]) -> str:
    title = html.escape(str(result.get("title") or "多模态抽样标注"))
    data_json = _script_json({"result": result, "notes": notes})
    media = result.get("media") if isinstance(result.get("media"), dict) else {}
    media_src = html.escape(_file_url(str(media.get("path") or "")))
    potplayer = result.get("potplayer") if isinstance(result.get("potplayer"), dict) else {}
    rows = result.get("samples") if isinstance(result.get("samples"), list) else []
    review_queue = _review_queue_html(rows)
    cards = "\n".join(_sample_card(row) for row in rows)
    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    sample_types = counts.get("sample_types") if isinstance(counts.get("sample_types"), dict) else {}
    type_bits = " ".join(f"<span class=\"badge\">{html.escape(str(k))}: {html.escape(str(v))}</span>" for k, v in sorted(sample_types.items()))
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title} - 多模态抽样标注</title>
  <style>
    :root {{ color-scheme: light; --bg:#f7f8fa; --panel:#fff; --ink:#172026; --muted:#667085; --line:#d8dee8; --accent:#2557a7; --warn:#995c00; --bad:#b42318; --ok:#0f6b4f; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ background:var(--panel); border-bottom:1px solid var(--line); padding:22px 30px; position:sticky; top:0; z-index:3; }} main {{ max-width:1440px; margin:0 auto; padding:20px 24px 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }} h2 {{ margin:0 0 8px; font-size:18px; }} h3 {{ margin:0; font-size:16px; }}
    .muted {{ color:var(--muted); }} .summary {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }} .badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:3px 9px; font-size:12px; color:var(--muted); background:#fff; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }} button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:8px 11px; cursor:pointer; }} button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
    .sample {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; margin:14px 0; padding:14px; }} .sample.missing_visual {{ border-left:5px solid var(--warn); }} .sample.temporal {{ border-left:5px solid var(--accent); }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }} .block {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; min-height:82px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; margin:6px 0 0; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; }}
    label {{ display:block; font-size:13px; color:var(--muted); margin:8px 0 4px; }} select, textarea {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:7px; font:inherit; background:#fff; }} textarea {{ min-height:72px; resize:vertical; }}
    .review-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:8px; }} .frames a {{ display:block; word-break:break-all; font-size:12px; }} #exportBox {{ width:100%; min-height:180px; margin-top:10px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .workbench {{ display:grid; grid-template-columns:minmax(360px, 42%) minmax(0, 1fr); gap:18px; align-items:start; }} .review-main {{ min-width:0; }}
    .video-panel {{ position:sticky; top:116px; z-index:1; max-height:calc(100vh - 136px); overflow:auto; }} .video-player {{ width:100%; max-height:36vh; background:#000; border-radius:8px; display:block; }} .sample.active {{ outline:3px solid rgba(37,87,167,.35); box-shadow:0 0 0 4px rgba(37,87,167,.08); }} .time-button {{ margin-left:8px; padding:5px 8px; font-size:12px; }} .review-queue {{ margin-top:10px; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; max-height:220px; overflow:auto; }} .queue-row {{ width:100%; display:grid; grid-template-columns:72px 72px 1fr; gap:8px; align-items:center; border:0; border-bottom:1px solid var(--line); border-radius:0; text-align:left; padding:7px 9px; background:#fff; }} .queue-row:hover {{ background:#eef4ff; }} .queue-row.active {{ background:#e7f0ff; color:var(--accent); font-weight:600; }} .queue-text {{ overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }}
    .current-panel {{ margin-top:10px; border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; }} .current-panel h3 {{ margin-bottom:6px; }} .current-meta {{ display:flex; flex-wrap:wrap; gap:6px; margin:6px 0; }} .current-excerpt {{ display:grid; gap:6px; margin-top:8px; }} .current-excerpt div {{ border:1px solid var(--line); border-radius:6px; padding:8px; background:#fbfcfe; }} .current-excerpt strong {{ display:block; margin-bottom:4px; }}
    @media (max-width:980px) {{ header {{ position:static; padding:18px; }} main {{ padding:14px; }} .workbench {{ grid-template-columns:1fr; }} .video-panel {{ position:static; max-height:none; }} .grid {{ grid-template-columns:1fr; }} }}
  </style>
  {review_export_script(result)}
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class=\"muted\">多模态抽样标注：检查 ASR、OCR/ebook、多模态理解和最终笔记是否真的变准。这个页面只导出 JSON，不直接写回 timeline。</div>
  <div class=\"summary\"><span class=\"badge\">样本 {html.escape(str(counts.get('sample_count', 0)))}</span><span class=\"badge\">单帧理解 {html.escape(str(counts.get('items_with_visual_understanding', 0)))}</span><span class=\"badge\">连续理解 {html.escape(str(counts.get('items_with_temporal_understanding', 0)))}</span>{type_bits}</div>
  <div class=\"toolbar\"><button class=\"primary\" onclick=\"exportNotes()\">生成标注 JSON</button><button onclick=\"downloadNotes()\">下载 JSON</button><button onclick=\"copyNotes()\">复制 JSON</button><a href=\"review.html\"><button>打开 review.html</button></a><a href=\"task-console.html\"><button>回任务控制台</button></a></div>
</header>
<main>
  <div class="workbench">
    <aside class="sample video-panel"><h2>视频回看窗口</h2><div class="muted">原视频：<code>{html.escape(str(media.get('path') or '未找到，请用 --media-path 重新生成'))}</code></div><video id="reviewVideo" class="video-player" controls preload="metadata" src="{media_src}"></video>{review_queue}<div id="currentSamplePanel" class="current-panel"><h3>当前审核条目</h3><div class="muted">点击上方时间戳队列，或右侧样本卡片里的“在页面播放器打开”。</div></div><div class="toolbar"><label><button type="button" onclick="q('localVideoPicker').click()">选择本地视频文件</button><input id="localVideoPicker" type="file" accept="video/*" style="display:none" onchange="loadLocalVideo(this)"></label><a href="potplayer-review-playlist.m3u8"><button>打开待审核播放列表</button></a><a href="potplayer-review-timestamps.md"><button>打开时间戳清单</button></a><a href="potplayer-review-timestamps.csv"><button>打开 CSV</button></a></div><div id="videoStatus" class="muted">点击时间戳队列会跳到对应时间戳、展示当前条目并高亮右侧详情。若浏览器不允许直接读取 file:// 视频，请先点“选择本地视频文件”。</div></aside>
    <section class="review-main">{cards}
  <section class=\"sample\"><h2>导出的标注 JSON</h2><div class=\"muted\">建议保存为 <code>multimodal-sample-review-notes.json</code>。后续再通过 reviewed import 写回，不要手改 timeline。</div><textarea id=\"exportBox\"></textarea></section>
    </section>
  </div>
</main>
<script id=\"sample-data\" type=\"application/json\">{data_json}</script>
<script>
const payload = JSON.parse(document.getElementById('sample-data').textContent);
function q(id) {{ return document.getElementById(id); }}
function value(name, index) {{ const el = q(`${{name}}-${{index}}`); return el ? el.value : ''; }}
function collectNotes() {{
  const notes = structuredClone(payload.notes);
  notes.updated_at = new Date().toISOString();
  notes.reviews = payload.result.samples.map(row => ({{
    index: row.index,
    sample_type: row.sample_type,
    time: timeLabel(row),
    asr_correct: value('asr', row.index),
    ocr_correct: value('ocr', row.index),
    multimodal_added_key_info: value('added', row.index),
    multimodal_error_or_hallucination: value('hallucination', row.index),
    final_note_sufficient: value('final', row.index),
    video_checked: value('video', row.index),
    term_accuracy: value('term', row.index),
    visual_fact_accuracy: value('visualfact', row.index),
    step_completeness: value('step', row.index),
    timestamp_accuracy: value('timestamp', row.index),
    keep_image_required: value('keepimage', row.index),
    content_candidate_usable: value('candidateusable', row.index),
    content_candidate_evidence_sufficient: value('candidateevidence', row.index),
    overall_label: value('overall', row.index),
    human_notes: value('notes', row.index)
  }}));
  return notes;
}}
function timeLabel(row) {{ return `${{row.start ?? ''}}-${{row.end ?? ''}}`; }}
function exportNotes() {{ q('exportBox').value = JSON.stringify(collectNotes(), null, 2); }}
async function copyNotes() {{ exportNotes(); await navigator.clipboard.writeText(q('exportBox').value); }}
async function copyText(id) {{ const el = q(id); if (!el) return; await navigator.clipboard.writeText(el.textContent); }}
function escapeHTML(value) {{ return String(value ?? '').replace(/[&<>"']/g, ch => {{ if (ch === '&') return '&amp;'; if (ch === '<') return '&lt;'; if (ch === '>') return '&gt;'; if (ch === '"') return '&quot;'; return '&#39;'; }}); }}
function setVideoStatus(text) {{ const el = q('videoStatus'); if (el) el.textContent = text; }}
function loadLocalVideo(input) {{ const file = input.files && input.files[0]; if (!file) return; const video = q('reviewVideo'); video.src = URL.createObjectURL(file); video.load(); setVideoStatus(`已加载本地视频：${{file.name}}。现在可以点击样本时间戳跳转。`); }}
function updateCurrentSample(index) {{ const panel = q('currentSamplePanel'); if (!panel) return; const row = payload.result.samples.find(item => Number(item.index) === Number(index)); if (!row) return; const chips = [row.sample_type, row.visual_route, timeLabel(row)].filter(Boolean).map(value => `<span class="badge">${{escapeHTML(value)}}</span>`).join(''); const transcript = row.transcript_excerpt || '未提取'; const visual = row.visual_understanding_excerpt || row.temporal_visual_understanding_excerpt || '未提取'; const ocr = row.visual_text_excerpt || row.structured_visual_excerpt || '未提取'; panel.innerHTML = `<h3>当前审核条目 #${{escapeHTML(row.index)}}</h3><div class="current-meta">${{chips}}</div><div class="current-excerpt"><div><strong>ASR/纠正版逐字稿</strong>${{escapeHTML(transcript)}}</div><div><strong>OCR/图文结构化</strong>${{escapeHTML(ocr)}}</div><div><strong>多模态/连续片段理解</strong>${{escapeHTML(visual)}}</div></div>`; }}
function seekSample(index, seconds) {{ const video = q('reviewVideo'); const start = Number(seconds || 0); updateCurrentSample(index); document.querySelectorAll('.sample.active,.queue-row.active').forEach(el => el.classList.remove('active')); const card = q(`sample-${{index}}`); if (card) {{ card.classList.add('active'); card.scrollIntoView({{behavior:'smooth', block:'nearest'}}); }} const queue = q(`queue-${{index}}`); if (queue) queue.classList.add('active'); const checked = q(`video-${{index}}`); if (checked && !checked.value) checked.value = 'yes'; if (!video || !video.src) {{ setVideoStatus('已选中审核条目，但还没有加载视频。请先选择本地视频文件，或确认浏览器允许读取原 file:// 视频。'); return; }} video.currentTime = Math.max(0, start); video.play().catch(() => {{}}); setVideoStatus(`已跳转到样本 #${{index}}，时间戳 ${{timeFromSeconds(start)}}。`); }}
function timeFromSeconds(value) {{ const total = Math.max(0, Math.floor(Number(value) || 0)); const h = String(Math.floor(total / 3600)).padStart(2, '0'); const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0'); const s = String(total % 60).padStart(2, '0'); return `${{h}}:${{m}}:${{s}}`; }}
function downloadNotes() {{ exportNotes(); const blob = new Blob([q('exportBox').value], {{type:'application/json'}}); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = vkpReviewExportName('多模态审核', JSON.parse(q('exportBox').value)); a.click(); URL.revokeObjectURL(a.href); }}
</script>
</body>
</html>"""



def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace('</', '<\\/')

def _review_queue_html(rows: list[Any]) -> str:
    buttons = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("index") or 0)
        start = _review_start(row)
        label = _chapter_time(start).split(".", 1)[0]
        sample_type = html.escape(str(row.get("sample_type") or "unknown"))
        text = html.escape(_excerpt(str(row.get("transcript_excerpt") or row.get("visual_understanding_excerpt") or ""), 86))
        buttons.append(
            f'<button type="button" id="queue-{idx}" class="queue-row" onclick="seekSample({idx}, {html.escape(_seconds_text(start))})">'
            f'<span>#{idx}</span><span>{html.escape(label)}</span><span class="queue-text"><strong>{sample_type}</strong> {text}</span></button>'
        )
    body = "".join(buttons) or '<div class="muted" style="padding:8px">暂无待审核样本</div>'
    return f'<div class="review-queue" aria-label="待审核时间戳队列">{body}</div>'

def _sample_card(row: dict[str, Any]) -> str:
    idx = int(row.get("index") or 0)
    frame_links = "\n".join(_frame_link(path) for path in row.get("frame_paths") or []) or "<span class=\"muted\">无帧路径</span>"
    potplayer_command = str(row.get("potplayer_command") or "")
    potplayer_id = f"potplayer-command-{idx}"
    potplayer_block = _potplayer_command_block(potplayer_id, potplayer_command)
    select = _select_html
    review_start_seconds = html.escape(_seconds_text(_review_start(row)))
    start_seconds = html.escape(_seconds_text(_seconds_value(row.get("start"))))
    end_seconds = html.escape(_seconds_text(_seconds_value(row.get("end"))))
    return f"""<article id=\"sample-{idx}\" class=\"sample {html.escape(str(row.get('sample_type') or ''))}\" data-start=\"{start_seconds}\" data-end=\"{end_seconds}\">
  <h2>#{idx} <span class=\"badge\">{html.escape(str(row.get('sample_type') or ''))}</span><span class=\"badge\">{html.escape(str(row.get('visual_route') or ''))}</span><span class=\"badge\">{html.escape(_time_label(row))}</span><button class=\"time-button\" onclick=\"seekSample({idx}, {review_start_seconds})\">按时间轴打开</button></h2>
  <div class=\"muted\">Tags: {html.escape(', '.join(row.get('tagger_tags') or []) or '-')} | Issues: {html.escape(', '.join(row.get('quality_issues') or []) or '-')}</div>
  <div class=\"grid\">
    {_block('ASR/纠正版逐字稿', row.get('transcript_excerpt'))}
    {_block('OCR/ebook 屏幕文字', row.get('visual_text_excerpt'))}
    {_block('图文结构化', row.get('structured_visual_excerpt'))}
    {_block('多模态单帧理解', row.get('visual_understanding_excerpt'))}
    {_block('连续片段理解', row.get('temporal_visual_understanding_excerpt'))}
    {_block('内容素材候选', _content_candidate_block_text(row))}
    <div class=\"block frames\"><strong>证据帧</strong>{frame_links}</div>
    {potplayer_block}
  </div>
  <div class=\"review-grid\">
    <div><label>ASR 是否正确</label>{select(f'asr-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>OCR/ebook 是否正确</label>{select(f'ocr-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>多模态是否补了关键信息</label>{select(f'added-{idx}', ['','yes','some','no','not_applicable'])}</div>
    <div><label>多模态是否有错/幻觉</label>{select(f'hallucination-{idx}', ['','no','minor','major','unclear'])}</div>
    <div><label>最终笔记是否足够</label>{select(f'final-{idx}', ['','yes','partial','no'])}</div>
    <div><label>是否已看视频片段</label>{select(f'video-{idx}', ['','yes','no','not_needed'])}</div>
    <div><label>术语/工具名准确</label>{select(f'term-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>画面事实准确</label>{select(f'visualfact-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>步骤/流程完整</label>{select(f'step-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>时间戳准确</label>{select(f'timestamp-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>必须保留图片</label>{select(f'keepimage-{idx}', ['','yes','no','unclear','not_applicable'])}</div>
    <div><label>素材候选可继续加工</label>{select(f'candidateusable-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>素材候选证据足够</label>{select(f'candidateevidence-{idx}', ['','yes','partial','no','not_applicable'])}</div>
    <div><label>总体标签</label>{select(f'overall-{idx}', ['','correct','partial','wrong','hallucination','missing','not_applicable'])}</div>
  </div>
  <label>人工备注</label><textarea id=\"notes-{idx}\" placeholder=\"记录错词、漏掉的画面信息、幻觉、是否需要重跑 OCR/多模态...\"></textarea>
</article>"""




def _content_candidate_block_text(row: dict[str, Any]) -> str:
    parts = []
    if row.get("content_candidate_id"):
        parts.append(f"ID: {row.get('content_candidate_id')}")
    if row.get("content_candidate_types"):
        parts.append("Types: " + ", ".join(str(value) for value in row.get("content_candidate_types") or []))
    for label, key in (
        ("Viewpoint", "content_candidate_viewpoint"),
        ("Case", "content_candidate_case_or_example"),
        ("Quote", "content_candidate_reusable_quote"),
        ("Fact status", "content_candidate_fact_status"),
    ):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    citation_summary = str(row.get("content_candidate_citation_summary") or "").strip()
    if citation_summary:
        parts.append("Citation summary: " + citation_summary)
    citations = row.get("content_candidate_evidence_citations") if isinstance(row.get("content_candidate_evidence_citations"), list) else []
    for citation in citations[:4]:
        if not isinstance(citation, dict):
            continue
        source = str(citation.get("source_type") or "unknown")
        time = str(citation.get("time") or "") or "unknown_time"
        text = str(citation.get("text") or "").strip()
        indexes = ",".join(str(value) for value in citation.get("timeline_indexes") or []) or "unknown"
        paths = [str(value) for value in citation.get("evidence_paths") or [] if str(value)]
        path_note = " | evidence=" + "; ".join(paths[:3]) if paths else ""
        parts.append(f"Citation: {time} {source} timeline={indexes} | {text}{path_note}")
    evidence = [str(value) for value in row.get("content_candidate_evidence_paths") or [] if str(value)]
    if evidence:
        parts.append("Evidence: " + "; ".join(evidence[:6]))
    return "\n".join(parts)
def _potplayer_command_block(element_id: str, command: str) -> str:
    if not command:
        return '<div class="block"><strong>PotPlayer 跳转</strong><div class="muted">未找到原视频路径，请用 --media-path 重新生成。</div></div>'
    return (
        f'<div class="block"><strong>PotPlayer 跳转</strong>'
        f'<div class="muted">复制到 PowerShell 执行，打开原视频到该时间点。</div>'
        f'<pre id="{html.escape(element_id)}">{html.escape(command)}</pre>'
        f'<button onclick="copyText(\'{html.escape(element_id)}\')">复制 PotPlayer 命令</button></div>'
    )

def _block(title: str, value: Any) -> str:
    text = html.escape(str(value or "").strip() or "未提取")
    return f"<div class=\"block\"><strong>{html.escape(title)}</strong><pre>{text}</pre></div>"


def _select_html(element_id: str, options: list[str]) -> str:
    items = []
    for option in options:
        label = option or "未标注"
        items.append(f"<option value=\"{html.escape(option)}\">{html.escape(label)}</option>")
    return f"<select id=\"{html.escape(element_id)}\">{''.join(items)}</select>"


def _frame_link(path: str) -> str:
    text = html.escape(str(path))
    href = html.escape(str(path).replace("\\", "/"))
    return f"<a href=\"{href}\">{text}</a>"




def _load_content_candidate_pack(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    assets = manifest.get("content_assets") if isinstance(manifest.get("content_assets"), dict) else {}
    export = manifest.get("knowledge_note_export") if isinstance(manifest.get("knowledge_note_export"), dict) else {}
    if not assets and isinstance(export.get("content_assets"), dict):
        assets = export.get("content_assets")
    raw = assets.get("content_candidate_pack_path") or manifest.get("content_candidate_pack_json") or "exports/content-candidate-pack.json"
    path = _bundle_path(root, str(raw))
    payload = _read_object(path)
    if not payload:
        return {"exists": False, "path": str(path), "candidates": []}
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    return {**payload, "exists": path.exists(), "path": str(path), "candidates": [row for row in candidates if isinstance(row, dict)]}


def _content_candidate_pack_summary(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    return {
        "exists": bool(payload.get("exists")),
        "path": str(payload.get("path") or ""),
        "candidate_count": int(payload.get("candidate_count") or len(candidates)),
        "review_required": bool(payload.get("review_required")) if payload else True,
        "publication_allowed": bool(payload.get("publication_allowed")) if payload else False,
        "allowed_as_fact": bool(payload.get("allowed_as_fact")) if payload else False,
        "allowed_as_inspiration": bool(payload.get("allowed_as_inspiration")) if payload else False,
    }


def _content_candidates_by_timeline_index(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = _int(row.get("timeline_index") or row.get("index"))
        if idx > 0 and idx not in result:
            result[idx] = row
    return result
def _load_review_transcript_cues(root: Path, manifest: dict[str, Any]) -> tuple[list[TranscriptCue], str]:
    seen: set[Path] = set()
    for path in _review_transcript_candidates(root, manifest):
        if path in seen:
            continue
        seen.add(path)
        if not path.exists() or not path.is_file():
            continue
        try:
            cues = parse_transcript(path)
        except Exception:
            continue
        timed = [cue for cue in cues if float(cue.end) >= float(cue.start) and str(cue.text or "").strip()]
        if timed:
            return sorted(timed, key=lambda cue: cue.start), transcript_source_kind(root, path)
    return []


def _review_transcript_candidates(root: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for key in (
        "corrected_transcript_json",
        "corrected_transcript_srt",
        "normalized_transcript_json",
        "normalized_transcript_srt",
        "transcript_json",
        "transcript_srt",
        "source_transcript",
        "transcript_path",
    ):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(_bundle_path(root, value))
    candidates.extend(
        [
            root / "corrected-transcript.json",
            root / "corrected-transcript.srt",
            root / "normalized-transcript.json",
            root / "normalized-transcript.srt",
            root / "transcript.json",
            root / "transcript.srt",
            root / "exports" / "corrected-transcript.json",
            root / "exports" / "corrected-transcript.srt",
            root / "exports" / "normalized-transcript.json",
            root / "exports" / "normalized-transcript.srt",
        ]
    )
    return candidates


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path

def _read_comparison(root: Path, comparison_json: str | Path | None) -> dict[str, Any]:
    path = _comparison_path(root, comparison_json)
    return _read_object(path)


def _comparison_path(root: Path, comparison_json: str | Path | None) -> Path:
    if comparison_json:
        return Path(comparison_json).expanduser().resolve()
    return root / "exports" / "multimodal-effect-comparison-report.json"


def _comparison_indexes(comparison: dict[str, Any]) -> list[int]:
    raw = comparison.get("example_indexes") or comparison.get("examples") or []
    out: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                value = item.get("index") or item.get("timeline_index")
            else:
                value = item
            idx = _int(value)
            if idx > 0 and idx not in out:
                out.append(idx)
    return out


def _evidence_paths(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    containers: list[dict[str, Any]] = [item]
    for nested_key in ("visual_understanding", "temporal_visual_understanding", "structured_visual"):
        nested = item.get(nested_key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in ("frame_paths", "temporal_frame_paths", "evidence_paths", "evidence_frame_paths"):
            value = container.get(key)
            if isinstance(value, list):
                for path in value:
                    if isinstance(path, str) and path and path not in out:
                        out.append(path)
        for key in ("frame_path", "image_path"):
            value = container.get(key)
            if isinstance(value, str) and value and value not in out:
                out.append(value)
    return out[:12]



def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
def _compact_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _excerpt(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _list_text(value: Any) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                text = item.get("label") or item.get("tag") or item.get("name") or item.get("type")
                if text:
                    out.append(str(text))
        return out
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _time_label(row: dict[str, Any]) -> str:
    start = row.get("start")
    end = row.get("end")
    return f"{start if start is not None else ''}-{end if end is not None else ''}"


def _index(item: dict[str, Any], fallback: int) -> int:
    return _int(item.get("index")) or fallback


def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}
