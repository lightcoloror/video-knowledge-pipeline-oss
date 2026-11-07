from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .content_clip_candidate_pack import PACK_SCHEMA, REVIEW_SCHEMA, _validate_schema
from .file_hash import sha256_file
from .run_artifact_registry import register_bundle_run
from .script_clip_alignment import (
    _artifact_ref,
    _find_clip_plan,
    _has_fragment_boundary,
    _joined_text,
    _keep_ranges,
    _optional_bound_path,
    _payload_sha,
    _segments_for_ranges,
    _semantic_expansions,
    _text_match_score,
)
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic
from .transcript import format_timestamp, parse_transcript


SCHEMA = "video_knowledge_pipeline.content_clip_alignment_check.v1"
OUTPUT_PATH = "exports/content-clip-alignment-check.json"
MARKDOWN_PATH = "exports/content-clip-alignment-check.md"
REPAIR_TODO_PATH = "content-clip-repair.todo.json"
MCP_ARGS_PATH = "mcp-content-clip-alignment-check.args.json"

ALLOWED_STATUSES = {
    "ready_for_human_final_review",
    "needs_candidate_selection",
    "needs_boundary_review",
    "needs_speaker_review",
    "needs_transcript_review",
    "needs_visual_review",
    "needs_recut",
    "needs_subtitle_revision",
}

ISSUE_STATUS = {
    "missing_required_clip": "needs_candidate_selection",
    "candidate_not_searched": "needs_candidate_selection",
    "invalid_candidate_selection": "needs_candidate_selection",
    "duplicate_or_conflicting_binding": "needs_candidate_selection",
    "boundary_evidence_missing": "needs_boundary_review",
    "boundary_not_human_confirmed": "needs_boundary_review",
    "speaker_role_unresolved": "needs_speaker_review",
    "excluded_speaker_present": "needs_speaker_review",
    "clip_asr_missing": "needs_transcript_review",
    "required_term_missing_after_cut": "needs_transcript_review",
    "clip_contains_unreviewed_claim": "needs_transcript_review",
    "required_multimodal_evidence_missing": "needs_visual_review",
    "multimodal_review_missing": "needs_visual_review",
    "required_ocr_text_missing_after_cut": "needs_visual_review",
    "excluded_content_present_after_cut": "needs_visual_review",
    "cut_outside_safe_extension": "needs_recut",
    "duration_outside_request": "needs_recut",
    "sentence_fragment_at_boundary": "needs_recut",
    "subtitle_semantic_expansion": "needs_subtitle_revision",
}

STATUS_PRIORITY = {
    "ready_for_human_final_review": 0,
    "needs_subtitle_revision": 1,
    "needs_recut": 2,
    "needs_visual_review": 3,
    "needs_transcript_review": 4,
    "needs_boundary_review": 5,
    "needs_speaker_review": 6,
    "needs_candidate_selection": 7,
}


def check_content_clip_alignment(
    bundle_dir: str | Path,
    review_notes_json: str | Path,
    fine_cut_plan_json: str | Path,
    *,
    candidate_pack_json: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Verify selected generic clips against clip-only multimodal evidence.

    Intent: prove the requested content survived cutting, regardless of whether
    it is speech, screen text, a visual/audio event, a whole shot, or a story
    beat. Decision: reuse the established script alignment helpers for source
    binding, range parsing, sentence checks, speaker checks, and semantic
    expansion, adding only modality-specific gates. Reason: source presence is
    not clip presence, and ASR cannot prove a visual event. Evidence:
    script_clip_alignment.v1 real interview run; WhisperX word/speaker
    provenance at 5f2f9d4320dd93a7d12f5ba2495eef7e0a5af963; VKP OCR,
    temporal, and technical-shot contracts. Effective scope: derived validation
    and repair artifacts only; never publication approval or media mutation.
    """

    root = Path(bundle_dir).expanduser().resolve()
    review_path = Path(review_notes_json).expanduser().resolve()
    fine_cut_path = Path(fine_cut_plan_json).expanduser().resolve()
    review = _read_object(review_path, "content clip review notes")
    fine_cut = _read_object(fine_cut_path, "fine cut plan")
    _validate_schema(review, "content-clip-review-notes.v1.schema.json")
    pack_ref = review.get("candidate_pack") if isinstance(review.get("candidate_pack"), dict) else {}
    pack_path = Path(candidate_pack_json).expanduser().resolve() if candidate_pack_json else _resolve_path(root, review_path.parent, str(pack_ref.get("path") or "exports/content-clip-candidate-pack.json"))
    pack = _read_object(pack_path, "content clip candidate pack")
    _validate_schema(pack, "content-clip-candidate-pack.v1.schema.json")
    if pack.get("schema") != PACK_SCHEMA or review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("content clip pack/review schema mismatch")
    if str(pack_ref.get("pack_sha256") or "") != _payload_sha(pack, field="pack_sha256"):
        raise ValueError("content clip candidate pack logical hash changed")
    expected_artifact_hash = str(pack_ref.get("artifact_sha256") or "")
    if not expected_artifact_hash or sha256_file(pack_path).lower() != expected_artifact_hash.lower():
        raise ValueError("content clip candidate pack artifact hash changed")

    source_artifacts = [_artifact_ref(pack_path, role="content_clip_candidate_pack"), _artifact_ref(review_path, role="content_clip_review_notes"), _artifact_ref(fine_cut_path, role="fine_cut_plan")]
    transcript_path = _pack_transcript_path(pack)
    source_artifacts.append(_artifact_ref(transcript_path, role="source_transcript"))
    source_cues = parse_transcript(transcript_path)
    clip_plan_rows = [row for row in (fine_cut.get("clips") or fine_cut.get("items") or []) if isinstance(row, dict)]
    review_rows = [row for row in review.get("clips") or [] if isinstance(row, dict)]
    review_by_clip = {str(row.get("clip_id") or ""): row for row in review_rows}
    if len(review_by_clip) != len(review_rows):
        raise ValueError("content clip review contains duplicate clip_id")
    candidates = {
        str(candidate.get("candidate_id") or ""): (clip, candidate)
        for clip in pack.get("clips") or []
        if isinstance(clip, dict)
        for candidate in clip.get("candidates") or []
        if isinstance(candidate, dict)
    }
    duplicate_orders = _duplicates(review_rows, "fine_cut_order", skip_empty=True)
    duplicate_outputs = _duplicates(review_rows, "fine_cut_output", skip_empty=True)
    seen_evidence_paths: set[Path] = set()
    clip_results: list[dict[str, Any]] = []

    for pack_clip in pack.get("clips") or []:
        if not isinstance(pack_clip, dict) or bool(pack_clip.get("excluded")):
            continue
        clip_id = str(pack_clip.get("clip_id") or "")
        issues: list[dict[str, Any]] = []
        review_row = review_by_clip.get(clip_id)
        if review_row is None:
            issues.append(_issue("missing_required_clip", clip_id, "No review row exists for this content request."))
            clip_results.append(_clip_result(clip_id, pack_clip, None, None, issues))
            continue
        if str(review.get("review_status") or "") != "human_confirmed" or not bool(review_row.get("human_confirmed")):
            issues.append(_issue("missing_required_clip", clip_id, "Candidate selection is still a draft."))
        if review_row.get("fine_cut_order") in duplicate_orders or str(review_row.get("fine_cut_output") or "") in duplicate_outputs:
            issues.append(_issue("duplicate_or_conflicting_binding", clip_id, "Fine-cut order or output is bound to more than one request."))

        selected_id = str(review_row.get("selected_candidate_id") or "")
        selected_entry = candidates.get(selected_id)
        selected = selected_entry[1] if selected_entry else None
        if selected is None or str(selected_entry[0].get("clip_id") or "") != clip_id:
            issues.append(_issue("invalid_candidate_selection", clip_id, "No candidate from this request was selected."))
            clip_results.append(_clip_result(clip_id, pack_clip, review_row, selected, issues))
            continue

        clip_plan = _find_clip_plan(clip_plan_rows, review_row)
        keep_ranges = _keep_ranges(clip_plan or {}) if clip_plan else []
        if not keep_ranges:
            issues.append(_issue("cut_outside_safe_extension", clip_id, "No valid fine-cut keep range is bound."))
        safe = selected.get("boundary", {}).get("safe_extension_range") or {}
        safe_start, safe_end = float(safe.get("start") or 0.0), float(safe.get("end") or 0.0)
        if keep_ranges and any(start < safe_start - 0.001 or end > safe_end + 0.001 for start, end in keep_ranges):
            issues.append(_issue("cut_outside_safe_extension", clip_id, "A keep range extends outside the candidate safe-extension range.", evidence={"safe_extension": safe, "keep_ranges": keep_ranges}))
        duration_seconds = sum(end - start for start, end in keep_ranges)
        duration = pack_clip.get("duration") or {}
        minimum, maximum = float(duration.get("minimum_seconds") or 0.0), float(duration.get("maximum_seconds") or 0.0)
        if keep_ranges and ((minimum and duration_seconds < minimum - 0.001) or (maximum and duration_seconds > maximum + 0.001)):
            issues.append(_issue("duration_outside_request", clip_id, "Fine-cut duration is outside the requested minimum/maximum.", evidence={"duration_seconds": round(duration_seconds, 6), "minimum_seconds": minimum, "maximum_seconds": maximum}))
        source_segments = _segments_for_ranges(source_cues, keep_ranges)
        if str(pack_clip.get("boundary_policy") or "") == "complete_sentence" and _has_fragment_boundary(source_cues, keep_ranges):
            issues.append(_issue("sentence_fragment_at_boundary", clip_id, "A keep-range boundary cuts through a transcript segment."))
        boundary = selected.get("boundary") if isinstance(selected.get("boundary"), dict) else {}
        if boundary.get("status") == "unavailable":
            issues.append(_issue("boundary_evidence_missing", clip_id, "The requested boundary type lacks direct source evidence."))
        approved_window = review_row.get("approved_window") if isinstance(review_row.get("approved_window"), dict) else None
        if boundary.get("human_boundary_review_required") and not approved_window:
            issues.append(_issue("boundary_not_human_confirmed", clip_id, "The candidate requires an explicit human-approved window."))

        _check_speakers(clip_id, pack_clip, review_row, source_segments, issues)
        required = set(str(value) for value in pack_clip.get("required_modalities") or [])
        modality_reviews = review_row.get("modality_reviews") if isinstance(review_row.get("modality_reviews"), dict) else {}
        clip_evidence = review_row.get("clip_evidence") if isinstance(review_row.get("clip_evidence"), dict) else {}
        evidence_paths: dict[str, Path] = {}
        for modality in ("asr", "ocr", "visual", "audio"):
            ref = clip_evidence.get(modality) if isinstance(clip_evidence.get(modality), dict) else {}
            path = _optional_bound_path(root, review_path.parent, str(ref.get("path") or ""), str(ref.get("sha256") or ""))
            if path:
                evidence_paths[modality] = path
                if path not in seen_evidence_paths:
                    source_artifacts.append(_artifact_ref(path, role=f"clip_{modality}_evidence"))
                    seen_evidence_paths.add(path)

        source_text = _joined_text(source_segments)
        clip_asr_text = ""
        clip_asr_segments: list[dict[str, Any]] = []
        if "asr" in evidence_paths:
            clip_asr_segments = _cue_rows(parse_transcript(evidence_paths["asr"]))
            clip_asr_text = _joined_text(clip_asr_segments)
        if "asr" in required and not clip_asr_text:
            issues.append(_issue("clip_asr_missing", clip_id, "Required clip-only ASR evidence is not bound."))
        if clip_asr_text:
            for term in pack_clip.get("must_include") or []:
                if _normalise(term) and _normalise(term) not in _normalise(clip_asr_text):
                    issues.append(_issue("required_term_missing_after_cut", clip_id, f"Required term is missing from clip-only ASR: {term}"))
            approved = str(review_row.get("approved_clip_text") or pack_clip.get("query") or "")
            if approved and _text_match_score(approved, clip_asr_text) < 60.0:
                issues.append(_issue("required_term_missing_after_cut", clip_id, "Clip-only ASR does not sufficiently match the approved content.", evidence={"match_score": round(_text_match_score(approved, clip_asr_text), 2)}))
            additions = _semantic_expansions(approved, clip_asr_text)
            if additions:
                issues.append(_issue("clip_contains_unreviewed_claim", clip_id, "Clip-only ASR contains protected claims absent from the approved text.", evidence={"additional_claim_tokens": additions}))

        ocr_text = _read_evidence_text(evidence_paths.get("ocr"))
        if "ocr" in required:
            if not ocr_text:
                issues.append(_issue("required_multimodal_evidence_missing", clip_id, "Required clip-only OCR evidence is not bound."))
            else:
                for term in pack_clip.get("must_include") or []:
                    if _normalise(term) and _normalise(term) not in _normalise(ocr_text):
                        issues.append(_issue("required_ocr_text_missing_after_cut", clip_id, f"Required text is missing from clip-only OCR: {term}"))
        for term in pack_clip.get("must_exclude") or []:
            if _normalise(term) and (_normalise(term) in _normalise(clip_asr_text) or _normalise(term) in _normalise(ocr_text)):
                issues.append(_issue("excluded_content_present_after_cut", clip_id, f"Explicitly excluded content is present after cutting: {term}"))

        for modality in ("visual", "audio"):
            if modality in required:
                if modality not in evidence_paths:
                    issues.append(_issue("required_multimodal_evidence_missing", clip_id, f"Required clip-only {modality} evidence is not bound."))
                elif str(modality_reviews.get(modality) or "") != "human_confirmed":
                    issues.append(_issue("multimodal_review_missing", clip_id, f"Required {modality} evidence has not been human-confirmed."))
        if "shot" in required and not selected.get("boundary", {}).get("source_shot_ids"):
            issues.append(_issue("boundary_evidence_missing", clip_id, "A complete technical shot is required but no verified shot ID is bound."))

        subtitle_text = str(review_row.get("subtitle_text") or "").strip()
        approved_text = str(review_row.get("approved_clip_text") or pack_clip.get("query") or "")
        subtitle_additions = _semantic_expansions(approved_text, subtitle_text) if approved_text and subtitle_text else []
        if subtitle_additions:
            issues.append(_issue("subtitle_semantic_expansion", clip_id, "Subtitle adds protected meaning absent from the approved clip text.", evidence={"additional_claim_tokens": subtitle_additions}))
        clip_results.append(_clip_result(clip_id, pack_clip, review_row, selected, issues, clip_plan=clip_plan, keep_ranges=keep_ranges, source_segments=source_segments, clip_asr_segments=clip_asr_segments, checks={"source_text": source_text, "clip_asr_text": clip_asr_text, "clip_ocr_text": ocr_text, "duration_seconds": round(duration_seconds, 6)}))

    status = _combined_status([row["status"] for row in clip_results])
    issue_counts: dict[str, int] = {}
    for row in clip_results:
        for issue in row["issues"]:
            code = str(issue["code"])
            issue_counts[code] = issue_counts.get(code, 0) + 1
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "bundle_dir": str(root),
        "review_id": str(review.get("review_id") or ""),
        "source_artifacts": source_artifacts,
        "summary": {"clip_count": len(clip_results), "ready_clip_count": sum(1 for row in clip_results if row["status"] == "ready_for_human_final_review"), "issue_count": sum(issue_counts.values()), "issue_counts": dict(sorted(issue_counts.items())), "human_final_review_still_required": True},
        "clips": clip_results,
        "publication_allowed": False,
        "operator_boundary": {"local_only": True, "external_provider_called": False, "media_uploaded": False, "canonical_transcript_mutated": False, "timeline_mutated": False, "source_media_mutated": False, "automatic_recut": False, "automatic_publication": False, "human_final_review_required": True},
        "artifacts": {"json": str(root / OUTPUT_PATH), "markdown": str(root / MARKDOWN_PATH), "repair_todo": str(root / REPAIR_TODO_PATH), "mcp_args": str(root / MCP_ARGS_PATH)},
    }
    result["check_sha256"] = _payload_sha(result, field="check_sha256")
    _validate_schema(result, "content-clip-alignment-check.v1.schema.json")
    todo = _repair_todo(result)
    if write:
        manifest_path = root / "manifest.json"
        with bundle_write_lock(root, operation="content_clip_alignment_check", timeout_seconds=1.0):
            write_json(root / OUTPUT_PATH, result)
            write_text_atomic(root / MARKDOWN_PATH, _render_markdown(result))
            write_json(root / REPAIR_TODO_PATH, todo)
            write_json(root / MCP_ARGS_PATH, {"bundle_dir": str(root), "review_notes_json": str(review_path), "fine_cut_plan_json": str(fine_cut_path), "candidate_pack_json": str(pack_path), "write": True})
            manifest = _read_object(manifest_path, "bundle manifest")
            manifest.update({"content_clip_alignment_check_json": OUTPUT_PATH, "content_clip_alignment_check_markdown": MARKDOWN_PATH, "content_clip_repair_todo": REPAIR_TODO_PATH, "mcp_content_clip_alignment_check_args": MCP_ARGS_PATH})
            write_json(manifest_path, manifest)
        register_bundle_run(
            root,
            run_type="content_clip_alignment_check",
            run_id="content-clip-alignment-check",
            status="completed" if status == "ready_for_human_final_review" else "needs_review",
            title="通用内容片段剪后多模态验真",
            summary=f"{len(clip_results)} clips; {result['summary']['issue_count']} issues; never publication approval.",
            inputs={"candidate_pack": source_artifacts[0], "review_notes": source_artifacts[1], "fine_cut_plan": source_artifacts[2]},
            artifacts=[{"key": "alignment_check", "path": root / OUTPUT_PATH}, {"key": "alignment_markdown", "path": root / MARKDOWN_PATH}, {"key": "repair_todo", "path": root / REPAIR_TODO_PATH}, {"key": "mcp_args", "path": root / MCP_ARGS_PATH}],
            failed_items=[{"id": row["clip_id"], "reason": issue["code"], "detail": issue["detail"]} for row in clip_results for issue in row["issues"]],
            retry_command=f'.\\scripts\\video-knowledge.ps1 content-clip-alignment-check "{root}" "{review_path}" "{fine_cut_path}"',
            next_actions=["Resolve only listed selection/boundary/speaker/transcript/visual/recut/subtitle issues, then rerun."],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
    return result


def _check_speakers(clip_id: str, pack_clip: dict[str, Any], review: dict[str, Any], segments: list[dict[str, Any]], issues: list[dict[str, Any]]) -> None:
    speakers = {str(row.get("speaker") or "") for row in segments if str(row.get("speaker") or "")}
    roles = {str(row.get("speaker_role") or "") for row in segments if str(row.get("speaker_role") or "")}
    expected_ids = {str(value) for value in review.get("expected_speaker_ids") or [] if str(value)}
    expected_role = str(review.get("expected_speaker_role") or "")
    constraints = pack_clip.get("speaker_constraints") if isinstance(pack_clip.get("speaker_constraints"), dict) else {}
    allowed_roles = {str(value) for value in constraints.get("allowed_roles") or [] if str(value)}
    excluded_roles = {str(value) for value in review.get("excluded_speaker_roles") or [] if str(value)}
    excluded_ids = {str(value) for value in review.get("excluded_speaker_ids") or [] if str(value)}
    if (allowed_roles or expected_role) and not roles and not expected_ids:
        issues.append(_issue("speaker_role_unresolved", clip_id, "Speaker constraints exist but no human-confirmed role or speaker ID is available."))
    if expected_ids and speakers and not speakers.issubset(expected_ids):
        issues.append(_issue("speaker_role_unresolved", clip_id, "Source range contains a speaker outside the confirmed set.", evidence={"source_speakers": sorted(speakers), "expected_speakers": sorted(expected_ids)}))
    if expected_role and roles and expected_role not in roles:
        issues.append(_issue("speaker_role_unresolved", clip_id, "Source role differs from the confirmed role."))
    if excluded_roles.intersection(roles) or excluded_ids.intersection(speakers):
        issues.append(_issue("excluded_speaker_present", clip_id, "The selected range contains an explicitly excluded speaker."))


def _clip_result(clip_id: str, pack_clip: dict[str, Any], review: dict[str, Any] | None, candidate: dict[str, Any] | None, issues: list[dict[str, Any]], *, clip_plan: dict[str, Any] | None = None, keep_ranges: list[tuple[float, float]] | None = None, source_segments: list[dict[str, Any]] | None = None, clip_asr_segments: list[dict[str, Any]] | None = None, checks: dict[str, Any] | None = None) -> dict[str, Any]:
    status = _combined_status([ISSUE_STATUS.get(str(issue.get("code") or ""), "needs_transcript_review") for issue in issues])
    return {
        "clip_id": clip_id,
        "purpose": str(pack_clip.get("purpose") or ""),
        "status": status,
        "selected_candidate_id": str((review or {}).get("selected_candidate_id") or ""),
        "selected_candidate_time_range": (candidate or {}).get("source_time_range") or {},
        "fine_cut": {"order": (clip_plan or {}).get("order"), "output": str((clip_plan or {}).get("output") or ""), "keep_ranges": [{"start": start, "end": end, "start_time": format_timestamp(start), "end_time": format_timestamp(end)} for start, end in keep_ranges or []]},
        "source_clip_segment_ids": [str(row.get("segment_id") or "") for row in source_segments or []],
        "clip_asr_segment_ids": [str(row.get("segment_id") or "") for row in clip_asr_segments or []],
        "checks": checks or {},
        "issues": issues,
        "human_final_review_required": True,
        "publication_allowed": False,
    }


def _read_evidence_text(path: Path | None) -> str:
    if path is None:
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() != ".json":
        return text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"clip evidence JSON is invalid: {path}") from exc
    return _flatten_text(payload)


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def _cue_rows(cues: list[Any]) -> list[dict[str, Any]]:
    return [{"segment_id": str(cue.segment_id or f"segment-{index:06d}"), "start": float(cue.start), "end": float(cue.end), "text": str(cue.text or ""), "speaker": str(cue.speaker or ""), "speaker_role": str(cue.speaker_role or "")} for index, cue in enumerate(cues, start=1)]


def _pack_transcript_path(pack: dict[str, Any]) -> Path:
    for ref in pack.get("source_artifacts") or []:
        if isinstance(ref, dict) and str(ref.get("role") or "") == "canonical_or_corrected_transcript":
            path = Path(str(ref.get("path") or "")).expanduser().resolve()
            if not path.is_file() or sha256_file(path).lower() != str(ref.get("sha256") or "").lower():
                raise ValueError("content clip source transcript changed")
            return path
    raise ValueError("content clip candidate pack has no bound source transcript")


def _resolve_path(root: Path, base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        primary = (base / path).resolve()
        path = primary if primary.is_file() else (root / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"bound artifact not found: {path}")
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _duplicates(rows: list[dict[str, Any]], key: str, *, skip_empty: bool) -> set[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for row in rows:
        value = row.get(key)
        if skip_empty and value in {None, ""}:
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _issue(code: str, clip_id: str, detail: str, *, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "clip_id": clip_id, "detail": detail, "evidence": evidence or {}}


def _combined_status(statuses: list[str]) -> str:
    if not statuses:
        return "ready_for_human_final_review"
    return max((status if status in ALLOWED_STATUSES else "needs_transcript_review" for status in statuses), key=lambda status: STATUS_PRIORITY[status])


def _normalise(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def _repair_todo(result: dict[str, Any]) -> dict[str, Any]:
    return {"schema": "video_knowledge_pipeline.content_clip_repair_todo.v1", "status": "completed" if result["status"] == "ready_for_human_final_review" else "needs_review", "alignment_check_sha256": result["check_sha256"], "items": [{"clip_id": row["clip_id"], "status": row["status"], "issues": row["issues"]} for row in result["clips"] if row["issues"]], "publication_allowed": False}


def _render_markdown(result: dict[str, Any]) -> str:
    lines = ["# 通用内容片段剪后验真", "", f"- Status: `{result.get('status')}`", f"- Clips: `{result.get('summary', {}).get('clip_count', 0)}`", f"- Issues: `{result.get('summary', {}).get('issue_count', 0)}`", "- 即使全部机器门通过，仍只进入最终人工复核，不构成发布批准。", ""]
    for row in result.get("clips") or []:
        lines.extend([f"## {row.get('clip_id')} · {row.get('purpose')}", "", f"- Status: `{row.get('status')}`", f"- Keep ranges: `{len(row.get('fine_cut', {}).get('keep_ranges') or [])}`", ""])
        for issue in row.get("issues") or []:
            lines.append(f"- `{issue.get('code')}`：{issue.get('detail')}")
        if not row.get("issues"):
            lines.append("- 机器检查通过；等待最终人工复核。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
