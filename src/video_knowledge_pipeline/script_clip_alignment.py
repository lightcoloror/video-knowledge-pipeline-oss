from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jsonschema
from rapidfuzz import fuzz

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .run_artifact_registry import register_bundle_run
from .script_clip_candidate_pack import PACK_SCHEMA, REVIEW_SCHEMA
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic
from .transcript import format_timestamp, parse_timestamp, parse_transcript


SCHEMA = "video_knowledge_pipeline.script_clip_alignment_check.v1"
OUTPUT_PATH = "exports/script-clip-alignment-check.json"
MARKDOWN_PATH = "exports/script-clip-alignment-check.md"
REPAIR_TODO_PATH = "script-clip-repair.todo.json"
MCP_ARGS_PATH = "mcp-script-clip-alignment-check.args.json"

ALLOWED_STATUSES = {
    "ready_for_human_final_review",
    "needs_candidate_selection",
    "needs_speaker_review",
    "needs_transcript_review",
    "needs_recut",
    "needs_subtitle_revision",
}

ISSUE_STATUS = {
    "missing_required_slot": "needs_candidate_selection",
    "candidate_not_searched": "needs_candidate_selection",
    "duplicate_or_conflicting_episode_binding": "needs_candidate_selection",
    "speaker_role_unresolved": "needs_speaker_review",
    "excluded_speaker_present": "needs_speaker_review",
    "multiple_speakers_mislabeled_as_customer_quote": "needs_speaker_review",
    "approved_quote_missing_after_cut": "needs_transcript_review",
    "clip_contains_unreviewed_claim": "needs_transcript_review",
    "source_only_not_clip_present": "needs_transcript_review",
    "cut_outside_approved_window": "needs_recut",
    "sentence_fragment_at_boundary": "needs_recut",
    "subtitle_semantic_expansion": "needs_subtitle_revision",
}

STATUS_PRIORITY = {
    "ready_for_human_final_review": 0,
    "needs_subtitle_revision": 1,
    "needs_recut": 2,
    "needs_transcript_review": 3,
    "needs_speaker_review": 4,
    "needs_candidate_selection": 5,
}

_PROTECTED_WORDS = {
    "医生",
    "客户",
    "家属",
    "患者",
    "采访者",
    "保险公司",
    "医院",
    "提醒",
    "赔付",
    "理赔",
    "报销",
    "借款",
    "支付",
    "选择",
    "决定",
    "治疗",
    "检查",
    "复查",
    "沟通",
    "梳理",
    "因为",
    "所以",
    "导致",
    "因此",
    "才",
    "一定",
    "肯定",
    "唯一",
    "全部",
    "完全",
    "最好",
    "最大",
    "及时",
    "之前",
    "以后",
    "当天",
    "第二天",
}


def check_script_clip_alignment(
    bundle_dir: str | Path,
    review_notes_json: str | Path,
    fine_cut_plan_json: str | Path,
    *,
    candidate_pack_json: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Check script, approved quote, selected cut, clip ASR, and subtitle.

    Intent: prevent a valid source phrase from being mistaken for a valid
    exported clip and prevent edited subtitles from expanding what was said.
    Decision: compare only human-selected, hash-bound candidates against the
    existing fine-cut plan and an independent clip transcript sidecar.
    Reason: source transcript presence alone cannot prove that the final cut
    contains the phrase, the intended speaker, or an unexpanded subtitle.
    Evidence: the self-media workflow exposed missing clip-only validation and
    multi-speaker customer-quote mislabelling in real interview cuts.
    Effective scope: derived review artifacts. This check cannot publish,
    approve, recut, rewrite subtitles, or mutate canonical evidence.
    """

    root = Path(bundle_dir).expanduser().resolve()
    review_path = Path(review_notes_json).expanduser().resolve()
    fine_cut_path = Path(fine_cut_plan_json).expanduser().resolve()
    review = _read_object(review_path, "script clip review notes")
    fine_cut = _read_object(fine_cut_path, "fine cut plan")
    _validate_schema(review, "script-clip-review-notes.v1.schema.json")

    pack_ref = review.get("candidate_pack") if isinstance(review.get("candidate_pack"), dict) else {}
    pack_path = (
        Path(candidate_pack_json).expanduser().resolve()
        if candidate_pack_json
        else _resolve_path(root, review_path.parent, str(pack_ref.get("path") or "exports/script-clip-candidate-pack.json"))
    )
    pack = _read_object(pack_path, "script clip candidate pack")
    _validate_schema(pack, "script-clip-candidate-pack.v1.schema.json")
    if pack.get("schema") != PACK_SCHEMA:
        raise ValueError(f"candidate pack schema must be {PACK_SCHEMA}")
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"review notes schema must be {REVIEW_SCHEMA}")
    expected_pack_hash = str(pack_ref.get("pack_sha256") or "")
    actual_pack_hash = _payload_sha(pack, field="pack_sha256")
    if expected_pack_hash != actual_pack_hash or str(pack.get("pack_sha256") or "") != actual_pack_hash:
        raise ValueError("candidate pack semantic hash changed")
    expected_artifact_hash = str(pack_ref.get("artifact_sha256") or "").lower()
    if str(review.get("review_status") or "") == "human_confirmed":
        if not expected_artifact_hash:
            raise ValueError("human-confirmed review must bind candidate pack artifact_sha256")
        if sha256_file(pack_path).lower() != expected_artifact_hash:
            raise ValueError("candidate pack artifact SHA-256 changed")
        if not str(review.get("review_id") or "").strip():
            raise ValueError("human-confirmed review requires review_id")

    transcript_path = _pack_transcript_path(pack)
    source_cues = parse_transcript(transcript_path)
    clip_plan_rows = [row for row in fine_cut.get("clips") or [] if isinstance(row, dict)]
    review_rows = [row for row in review.get("slots") or [] if isinstance(row, dict)]
    review_by_slot, duplicate_review_slots = _unique_by(review_rows, "slot_id")
    pack_slot_ids = {
        str(row.get("slot_id") or "") for row in pack.get("slots") or [] if isinstance(row, dict)
    }
    unknown_review_slots = sorted(set(review_by_slot) - pack_slot_ids)
    if unknown_review_slots:
        raise ValueError(f"review notes contain slot IDs outside candidate pack: {unknown_review_slots}")
    duplicate_episodes = _duplicate_values(review_rows, "episode_binding", skip_empty=True)
    duplicate_orders = _duplicate_values(review_rows, "fine_cut_order", skip_empty=True)
    duplicate_outputs = _duplicate_values(review_rows, "fine_cut_output", skip_empty=True)
    pack_candidates = {
        str(candidate.get("candidate_id") or ""): (slot, candidate)
        for slot in pack.get("slots") or []
        if isinstance(slot, dict)
        for candidate in slot.get("candidates") or []
        if isinstance(candidate, dict)
    }

    slot_results: list[dict[str, Any]] = []
    source_artifacts = [
        _artifact_ref(pack_path, role="script_clip_candidate_pack"),
        _artifact_ref(review_path, role="script_clip_review_notes"),
        _artifact_ref(fine_cut_path, role="fine_cut_plan"),
        _artifact_ref(transcript_path, role="source_transcript"),
    ]
    seen_clip_transcripts: set[Path] = set()
    for pack_slot in pack.get("slots") or []:
        if not isinstance(pack_slot, dict):
            continue
        slot_id = str(pack_slot.get("slot_id") or "")
        if bool(pack_slot.get("excluded")):
            continue
        issues: list[dict[str, Any]] = []
        review_row = review_by_slot.get(slot_id)
        if bool(pack_slot.get("required")) and review_row is None:
            issues.append(_issue("missing_required_slot", slot_id, "No review row exists for this required script slot."))
        if str(pack_slot.get("search_status") or "") == "candidate_not_searched":
            issues.append(_issue("candidate_not_searched", slot_id, "The script slot has no explicit search query or preferred time window."))
        if slot_id in duplicate_review_slots:
            issues.append(_issue("duplicate_or_conflicting_episode_binding", slot_id, "The review file contains the slot more than once."))
        if review_row is None:
            slot_results.append(_slot_result(slot_id, pack_slot, None, None, issues))
            continue
        if str(review.get("review_status") or "") != "human_confirmed" or not bool(review_row.get("human_confirmed")):
            issues.append(_issue("missing_required_slot", slot_id, "Candidate selection is still a draft and has not been human-confirmed."))
        episode = str(review_row.get("episode_binding") or "")
        if episode in duplicate_episodes or review_row.get("fine_cut_order") in duplicate_orders or str(review_row.get("fine_cut_output") or "") in duplicate_outputs:
            issues.append(_issue("duplicate_or_conflicting_episode_binding", slot_id, "Episode or fine-cut binding is reused by multiple script slots."))

        selected_id = str(review_row.get("selected_candidate_id") or "")
        selected_entry = pack_candidates.get(selected_id)
        selected = selected_entry[1] if selected_entry else None
        if selected is None:
            issues.append(_issue("missing_required_slot", slot_id, "No valid candidate from this pack was selected."))
        elif str(selected_entry[0].get("slot_id") or "") != slot_id:
            issues.append(_issue("duplicate_or_conflicting_episode_binding", slot_id, "The selected candidate belongs to another script slot."))

        clip_plan = _find_clip_plan(clip_plan_rows, review_row)
        if clip_plan is None:
            issues.append(_issue("cut_outside_approved_window", slot_id, "No fine-cut entry is bound to the selected slot."))
            slot_results.append(_slot_result(slot_id, pack_slot, review_row, selected, issues))
            continue
        keep_ranges = _keep_ranges(clip_plan)
        if not keep_ranges:
            issues.append(_issue("cut_outside_approved_window", slot_id, "The bound fine-cut entry has no valid keep ranges."))
        approved_window = _approved_window(review_row, selected)
        if approved_window and any(start < approved_window[0] - 0.001 or end > approved_window[1] + 0.001 for start, end in keep_ranges):
            issues.append(
                _issue(
                    "cut_outside_approved_window",
                    slot_id,
                    "At least one keep range extends outside the human-approved source window.",
                    evidence={"approved_window": approved_window, "keep_ranges": keep_ranges},
                )
            )
        source_clip_segments = _segments_for_ranges(source_cues, keep_ranges)
        if _has_fragment_boundary(source_cues, keep_ranges):
            issues.append(_issue("sentence_fragment_at_boundary", slot_id, "A keep-range boundary cuts through a source transcript segment."))

        explicit_expected_role = str(review_row.get("expected_speaker_role") or "").strip()
        explicit_expected_ids = {
            str(value).strip() for value in review_row.get("expected_speaker_ids") or [] if str(value).strip()
        }
        candidate_role_status = str((selected or {}).get("speaker_evidence", {}).get("role_status") or "unresolved")
        source_speakers = sorted({str(row.get("speaker") or "") for row in source_clip_segments if str(row.get("speaker") or "")})
        source_roles = sorted({str(row.get("speaker_role") or "") for row in source_clip_segments if str(row.get("speaker_role") or "")})
        if not explicit_expected_role and candidate_role_status != "resolved" and not source_roles:
            issues.append(_issue("speaker_role_unresolved", slot_id, "No human-confirmed role or explicit source role is bound; VKP will not infer identity."))
        if explicit_expected_ids and source_speakers and not set(source_speakers).issubset(explicit_expected_ids):
            issues.append(
                _issue(
                    "speaker_role_unresolved",
                    slot_id,
                    "The source range contains an anonymous speaker outside the human-confirmed speaker set.",
                    evidence={"expected_speaker_ids": sorted(explicit_expected_ids), "source_speaker_ids": source_speakers},
                )
            )
        if explicit_expected_role and source_roles and explicit_expected_role not in source_roles:
            issues.append(
                _issue(
                    "speaker_role_unresolved",
                    slot_id,
                    "The source role does not match the human-confirmed expected role.",
                    evidence={"expected_speaker_role": explicit_expected_role, "source_speaker_roles": source_roles},
                )
            )
        excluded_roles = {str(value) for value in review_row.get("excluded_speaker_roles") or [] if str(value)}
        excluded_ids = {str(value) for value in review_row.get("excluded_speaker_ids") or [] if str(value)}
        if excluded_roles.intersection(source_roles) or excluded_ids.intersection(source_speakers):
            issues.append(
                _issue(
                    "excluded_speaker_present",
                    slot_id,
                    "The selected source ranges include an explicitly excluded speaker.",
                    evidence={"speaker_ids": source_speakers, "speaker_roles": source_roles},
                )
            )
        if str(review_row.get("label") or "") == "customer_quote" and len(source_speakers) > 1:
            issues.append(
                _issue(
                    "multiple_speakers_mislabeled_as_customer_quote",
                    slot_id,
                    "The selected ranges contain multiple anonymous speakers but are labelled as one customer quote.",
                    evidence={"speaker_ids": source_speakers},
                )
            )

        source_clip_text = _joined_text(source_clip_segments)
        clip_transcript_path = _optional_bound_path(
            root,
            review_path.parent,
            str(review_row.get("clip_transcript_path") or ""),
            str(review_row.get("clip_transcript_sha256") or ""),
        )
        clip_segments: list[dict[str, Any]] = []
        clip_text = ""
        if clip_transcript_path:
            clip_segments = _cue_rows(parse_transcript(clip_transcript_path))
            clip_text = _joined_text(clip_segments)
            if clip_transcript_path not in seen_clip_transcripts:
                source_artifacts.append(_artifact_ref(clip_transcript_path, role="clip_transcript"))
                seen_clip_transcripts.add(clip_transcript_path)
        clip_speakers = sorted({str(row.get("speaker") or "") for row in clip_segments if str(row.get("speaker") or "")})
        clip_roles = sorted({str(row.get("speaker_role") or "") for row in clip_segments if str(row.get("speaker_role") or "")})
        if explicit_expected_ids and clip_speakers and not set(clip_speakers).issubset(explicit_expected_ids):
            issues.append(
                _issue(
                    "speaker_role_unresolved",
                    slot_id,
                    "The clip-only transcript contains an anonymous speaker outside the human-confirmed speaker set.",
                    evidence={"expected_speaker_ids": sorted(explicit_expected_ids), "clip_speaker_ids": clip_speakers},
                )
            )
        if explicit_expected_role and clip_roles and explicit_expected_role not in clip_roles:
            issues.append(
                _issue(
                    "speaker_role_unresolved",
                    slot_id,
                    "The clip-only transcript role does not match the human-confirmed expected role.",
                    evidence={"expected_speaker_role": explicit_expected_role, "clip_speaker_roles": clip_roles},
                )
            )
        if excluded_roles.intersection(clip_roles) or excluded_ids.intersection(clip_speakers):
            issues.append(
                _issue(
                    "excluded_speaker_present",
                    slot_id,
                    "The clip-only transcript contains an explicitly excluded speaker.",
                    evidence={"speaker_ids": clip_speakers, "speaker_roles": clip_roles},
                )
            )
        if str(review_row.get("label") or "") == "customer_quote" and len(clip_speakers) > 1:
            issues.append(
                _issue(
                    "multiple_speakers_mislabeled_as_customer_quote",
                    slot_id,
                    "The clip-only transcript contains multiple anonymous speakers but is labelled as one customer quote.",
                    evidence={"speaker_ids": clip_speakers},
                )
            )
        approved_quote = str(review_row.get("approved_quote") or pack_slot.get("expected_quote") or "").strip()
        source_quote_score = _text_match_score(approved_quote, source_clip_text)
        clip_quote_score = _text_match_score(approved_quote, clip_text)
        if not clip_text:
            issues.append(_issue("source_only_not_clip_present", slot_id, "No independently generated clip transcript is bound; source presence cannot prove clip presence."))
        else:
            if approved_quote and clip_quote_score < 68.0:
                issues.append(
                    _issue(
                        "approved_quote_missing_after_cut",
                        slot_id,
                        "The approved quote is not sufficiently present in the clip-only transcript.",
                        evidence={"clip_match_score": round(clip_quote_score, 2)},
                    )
                )
            if approved_quote and source_quote_score >= 75.0 and clip_quote_score < 68.0:
                issues.append(
                    _issue(
                        "source_only_not_clip_present",
                        slot_id,
                        "The phrase is supported by the source window but not by the clip-only transcript.",
                        evidence={"source_match_score": round(source_quote_score, 2), "clip_match_score": round(clip_quote_score, 2)},
                    )
                )
            approved_clip_text = str(review_row.get("approved_clip_text") or approved_quote)
            additions = _semantic_expansions(approved_clip_text, clip_text)
            if additions:
                issues.append(
                    _issue(
                        "clip_contains_unreviewed_claim",
                        slot_id,
                        "The clip transcript contains protected claim tokens absent from the reviewed clip text.",
                        evidence={"additional_claim_tokens": additions},
                    )
                )
        subtitle_text = str(review_row.get("subtitle_text") or "").strip()
        subtitle_additions = _semantic_expansions(approved_quote, subtitle_text) if approved_quote and subtitle_text else []
        if subtitle_additions:
            issues.append(
                _issue(
                    "subtitle_semantic_expansion",
                    slot_id,
                    "The subtitle adds protected subject/action/number/causal/evaluation/time meaning not found in the approved quote.",
                    evidence={"additional_claim_tokens": subtitle_additions},
                )
            )
        slot_results.append(
            _slot_result(
                slot_id,
                pack_slot,
                review_row,
                selected,
                issues,
                clip_plan=clip_plan,
                keep_ranges=keep_ranges,
                source_clip_segments=source_clip_segments,
                clip_segments=clip_segments,
                match_scores={"source_quote": source_quote_score, "clip_quote": clip_quote_score},
            )
        )

    status = _combined_status([row["status"] for row in slot_results])
    issue_counts: dict[str, int] = {}
    for row in slot_results:
        for issue in row["issues"]:
            code = str(issue.get("code") or "")
            issue_counts[code] = issue_counts.get(code, 0) + 1
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "bundle_dir": str(root),
        "review_id": str(review.get("review_id") or ""),
        "source_artifacts": source_artifacts,
        "summary": {
            "slot_count": len(slot_results),
            "ready_slot_count": sum(1 for row in slot_results if row["status"] == "ready_for_human_final_review"),
            "issue_count": sum(issue_counts.values()),
            "issue_counts": dict(sorted(issue_counts.items())),
            "human_final_review_still_required": True,
        },
        "slots": slot_results,
        "publication_allowed": False,
        "operator_boundary": _operator_boundary(),
        "artifacts": {
            "json": str(root / OUTPUT_PATH),
            "markdown": str(root / MARKDOWN_PATH),
            "repair_todo": str(root / REPAIR_TODO_PATH),
            "mcp_args": str(root / MCP_ARGS_PATH),
        },
    }
    result["check_sha256"] = _payload_sha(result, field="check_sha256")
    _validate_schema(result, "script-clip-alignment-check.v1.schema.json")
    repair_todo = _repair_todo(result)
    if write:
        manifest_path = root / "manifest.json"
        with bundle_write_lock(root, operation="script_clip_alignment_check", timeout_seconds=1.0):
            write_json(root / OUTPUT_PATH, result)
            write_text_atomic(root / MARKDOWN_PATH, _render_markdown(result))
            write_json(root / REPAIR_TODO_PATH, repair_todo)
            write_json(
                root / MCP_ARGS_PATH,
                {
                    "bundle_dir": str(root),
                    "review_notes_json": str(review_path),
                    "fine_cut_plan_json": str(fine_cut_path),
                    "candidate_pack_json": str(pack_path),
                    "write": True,
                },
            )
            manifest = _read_object(manifest_path, "bundle manifest")
            manifest.update(
                {
                    "script_clip_alignment_check_json": OUTPUT_PATH,
                    "script_clip_alignment_check_markdown": MARKDOWN_PATH,
                    "script_clip_repair_todo": REPAIR_TODO_PATH,
                    "mcp_script_clip_alignment_check_args": MCP_ARGS_PATH,
                }
            )
            write_json(manifest_path, manifest)
        register_bundle_run(
            root,
            run_type="script_clip_alignment_check",
            run_id="script-clip-alignment-check",
            status="needs_review" if status != "ready_for_human_final_review" else "completed",
            title="脚本／客户原声／剪后片段一致性检查",
            summary=f"{len(slot_results)} slots; {result['summary']['issue_count']} review issues; never publication approval.",
            inputs={"candidate_pack": source_artifacts[0], "review_notes": source_artifacts[1], "fine_cut_plan": source_artifacts[2]},
            artifacts=[
                {"key": "alignment_check", "path": root / OUTPUT_PATH},
                {"key": "alignment_markdown", "path": root / MARKDOWN_PATH},
                {"key": "repair_todo", "path": root / REPAIR_TODO_PATH},
                {"key": "mcp_args", "path": root / MCP_ARGS_PATH},
            ],
            failed_items=[
                {"id": row["slot_id"], "reason": issue["code"], "detail": issue["detail"]}
                for row in slot_results
                for issue in row["issues"]
            ],
            retry_command=f'.\\scripts\\video-knowledge.ps1 script-clip-alignment-check "{root}" "{review_path}" "{fine_cut_path}"',
            next_actions=["Resolve only the listed review/recut/subtitle gaps, regenerate clip-only ASR when needed, and rerun this check."],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
    return result


def _slot_result(
    slot_id: str,
    pack_slot: dict[str, Any],
    review: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    issues: list[dict[str, Any]],
    *,
    clip_plan: dict[str, Any] | None = None,
    keep_ranges: list[tuple[float, float]] | None = None,
    source_clip_segments: list[dict[str, Any]] | None = None,
    clip_segments: list[dict[str, Any]] | None = None,
    match_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    status = _combined_status([ISSUE_STATUS.get(str(issue.get("code") or ""), "needs_transcript_review") for issue in issues])
    return {
        "slot_id": slot_id,
        "episode_binding": str((review or {}).get("episode_binding") or pack_slot.get("episode_binding") or ""),
        "status": status,
        "selected_candidate_id": str((review or {}).get("selected_candidate_id") or ""),
        "selected_candidate_time_range": (candidate or {}).get("source_time_range") or {},
        "fine_cut": {
            "order": (clip_plan or {}).get("order"),
            "output": str((clip_plan or {}).get("output") or ""),
            "keep_ranges": [
                {"start": start, "end": end, "start_time": format_timestamp(start), "end_time": format_timestamp(end)}
                for start, end in keep_ranges or []
            ],
        },
        "source_clip_segment_ids": [row["segment_id"] for row in source_clip_segments or []],
        "clip_transcript_segment_ids": [row["segment_id"] for row in clip_segments or []],
        "match_scores": {key: round(float(value), 2) for key, value in (match_scores or {}).items()},
        "issues": issues,
        "human_final_review_required": True,
        "publication_allowed": False,
    }


def _find_clip_plan(rows: list[dict[str, Any]], review: dict[str, Any]) -> dict[str, Any] | None:
    order = review.get("fine_cut_order")
    output = str(review.get("fine_cut_output") or "").replace("\\", "/").lower()
    matches = []
    for row in rows:
        row_output = str(row.get("output") or "").replace("\\", "/").lower()
        if order is not None and int(row.get("order") or 0) == int(order):
            matches.append(row)
        elif output and row_output == output:
            matches.append(row)
    return matches[0] if len(matches) == 1 else None


def _keep_ranges(row: dict[str, Any]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for value in row.get("keep_ranges") or []:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        start = _seconds(value[0])
        end = _seconds(value[1])
        if start >= 0 and end > start:
            result.append((start, end))
    result.sort()
    return result


def _approved_window(review: dict[str, Any], candidate: dict[str, Any] | None) -> tuple[float, float] | None:
    value = review.get("approved_window")
    if isinstance(value, dict):
        start, end = float(value.get("start") or 0.0), float(value.get("end") or 0.0)
        return (start, end) if end > start else None
    value = (candidate or {}).get("source_time_range")
    if isinstance(value, dict):
        start, end = float(value.get("start") or 0.0), float(value.get("end") or 0.0)
        return (start, end) if end > start else None
    return None


def _segments_for_ranges(cues: list[Any], ranges: list[tuple[float, float]]) -> list[dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    for cue_index, cue in enumerate(cues, start=1):
        start, end = float(cue.start), max(float(cue.start), float(cue.end))
        # Treat transcript and keep ranges as half-open intervals.  A cue that
        # starts exactly where a cut ends belongs to the following clip; using
        # inclusive comparisons here would falsely introduce that speaker and
        # text into the preceding clip.
        if not any(end > range_start and start < range_end for range_start, range_end in ranges):
            continue
        segment_id = str(cue.segment_id or f"segment-{cue_index:06d}")
        if segment_id in seen:
            continue
        seen.add(segment_id)
        selected.append(_cue_row(cue, segment_id))
    return selected


def _cue_rows(cues: list[Any]) -> list[dict[str, Any]]:
    return [_cue_row(cue, str(cue.segment_id or f"segment-{index:06d}")) for index, cue in enumerate(cues, start=1)]


def _cue_row(cue: Any, segment_id: str) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "source_segment_ids": [str(value) for value in (cue.source_segment_ids or [segment_id]) if str(value)],
        "start": round(float(cue.start), 6),
        "end": round(float(cue.end), 6),
        "text": str(cue.text or ""),
        "speaker": str(cue.speaker or ""),
        "speaker_role": str(cue.speaker_role or ""),
    }


def _has_fragment_boundary(cues: list[Any], ranges: list[tuple[float, float]], *, tolerance: float = 0.18) -> bool:
    for start, end in ranges:
        for cue in cues:
            cue_start, cue_end = float(cue.start), max(float(cue.start), float(cue.end))
            if cue_start + tolerance < start < cue_end - tolerance:
                return True
            if cue_start + tolerance < end < cue_end - tolerance:
                return True
    return False


def _semantic_expansions(reference: str, candidate: str) -> list[str]:
    reference_norm = _normalise_text(reference)
    candidate_norm = _normalise_text(candidate)
    additions: list[str] = []
    for token in sorted(_PROTECTED_WORDS):
        if token in candidate_norm and token not in reference_norm:
            additions.append(token)
    reference_numbers = set(_number_tokens(reference))
    for token in _number_tokens(candidate):
        if token not in reference_numbers and token not in additions:
            additions.append(token)
    return additions


def _number_tokens(value: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?(?:万|千|百|元|次|天|月|年|%|％)?", str(value or ""))


def _text_match_score(expected: str, actual: str) -> float:
    left, right = _normalise_text(expected), _normalise_text(actual)
    if not left or not right:
        return 0.0
    return float(fuzz.partial_ratio(left, right))


def _joined_text(rows: list[dict[str, Any]]) -> str:
    return " ".join(str(row.get("text") or "").strip() for row in rows if str(row.get("text") or "").strip())


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "").lower())


def _pack_transcript_path(pack: dict[str, Any]) -> Path:
    for row in pack.get("source_artifacts") or []:
        if isinstance(row, dict) and str(row.get("role") or "") == "canonical_or_corrected_transcript":
            path = Path(str(row.get("path") or "")).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"candidate pack transcript is missing: {path}")
            if sha256_file(path).lower() != str(row.get("sha256") or "").lower():
                raise ValueError("candidate pack transcript changed")
            return path
    raise ValueError("candidate pack does not bind a source transcript")


def _optional_bound_path(root: Path, base: Path, raw: str, expected_sha: str) -> Path | None:
    if not raw.strip():
        return None
    path = _resolve_path(root, base, raw)
    if not path.is_file():
        return None
    if not expected_sha:
        raise ValueError(f"clip transcript must be SHA-256 bound: {path}")
    if sha256_file(path).lower() != expected_sha.lower():
        raise ValueError(f"clip transcript SHA-256 changed: {path}")
    return path


def _resolve_path(root: Path, base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    bundle_path = (root / path).resolve()
    return bundle_path if bundle_path.exists() else (base / path).resolve()


def _artifact_ref(path: Path, *, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_schema(payload: dict[str, Any], filename: str) -> None:
    schema = read_json(Path(__file__).with_name("schemas") / filename)
    jsonschema.validate(payload, schema)


def _payload_sha(payload: dict[str, Any], *, field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _unique_by(rows: list[dict[str, Any]], key: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        value = str(row.get(key) or "")
        if value in result:
            duplicates.add(value)
        else:
            result[value] = row
    return result, duplicates


def _duplicate_values(rows: list[dict[str, Any]], key: str, *, skip_empty: bool = False) -> set[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for row in rows:
        value = row.get(key)
        if skip_empty and value in (None, ""):
            continue
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return duplicates


def _issue(code: str, slot_id: str, detail: str, *, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "slot_id": slot_id, "detail": detail, "evidence": evidence or {}}


def _combined_status(statuses: list[str]) -> str:
    values = [status for status in statuses if status in ALLOWED_STATUSES]
    return max(values, key=lambda value: STATUS_PRIORITY[value]) if values else "ready_for_human_final_review"


def _seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return parse_timestamp(str(value))


def _repair_todo(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.script_clip_repair_todo.v1",
        "check_sha256": result["check_sha256"],
        "status": result["status"],
        "rows": [
            {
                "slot_id": row["slot_id"],
                "status": row["status"],
                "issue_codes": [issue["code"] for issue in row["issues"]],
                "human_action_required": True,
                "automatic_mutation_allowed": False,
            }
            for row in result["slots"]
            if row["issues"]
        ],
        "publication_allowed": False,
    }


def _operator_boundary() -> dict[str, Any]:
    return {
        "local_only": True,
        "review_only": True,
        "publication_allowed": False,
        "human_final_review_required": True,
        "external_provider_called": False,
        "media_uploaded": False,
        "identity_inference_performed": False,
        "automatic_recut_performed": False,
        "automatic_subtitle_rewrite_performed": False,
        "canonical_transcript_mutated": False,
        "timeline_mutated": False,
        "source_media_mutated": False,
        "upstream_script_mutated": False,
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 脚本／采访原声／剪后片段一致性检查",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Slots: `{result.get('summary', {}).get('slot_count', 0)}`",
        f"- Issues: `{result.get('summary', {}).get('issue_count', 0)}`",
        "- 结论边界：即使全部通过，也只进入人工终审，不构成发布批准。",
        "",
    ]
    for row in result.get("slots") or []:
        lines.extend([f"## {row.get('slot_id')}", "", f"- Status: `{row.get('status')}`", f"- Candidate: `{row.get('selected_candidate_id') or '-'}`", ""])
        if row.get("issues"):
            for issue in row["issues"]:
                lines.append(f"- `{issue.get('code')}`：{issue.get('detail')}")
        else:
            lines.append("- 当前机器检查无阻断，仍须人工终审画面、语气和上下文。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
