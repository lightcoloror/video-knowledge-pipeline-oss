from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .file_hash import sha256_file as _sha256
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .text_normalization import compact_ascii_cjk as _compact
from .transcript_source_arbitration import _render_corrected_markdown, _render_srt


SCHEMA = "video_knowledge_pipeline.asr_evidence_autoadjudication.v1"
_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff]|[A-Za-z]+(?:[-_.][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[^\s]",
    re.UNICODE,
)
_FACT_RE = re.compile(
    r"(?:\d+(?:\.\d+)?|[零一二三四五六七八九十百千万两]+)"
    r"\s*(?:%|％|万|万元|千|元|岁|年|月|日|次|家|款|种)?"
)


def adjudicate_asr_with_independent_evidence(
    bundle_dir: str | Path,
    *,
    secondary_transcript: str | Path,
    corroborating_transcripts: Sequence[str | Path],
    refresh_exports: bool = False,
    write: bool = False,
) -> dict[str, Any]:
    """Patch canonical ASR only where an independent transcript corroborates a delta.

    The secondary transcript is never promoted wholesale.  A non-empty secondary
    diff may replace the corresponding canonical span only when the same compact
    text occurs in an overlapping independent transcript and the canonical
    alternative does not. Numeric/fact-bearing changes always remain unresolved.
    """

    root = Path(bundle_dir).expanduser().resolve()
    canonical_path = root / "source-arbitrated-transcript.json"
    secondary_path = Path(secondary_transcript).expanduser().resolve()
    corroborator_paths = [Path(value).expanduser().resolve() for value in corroborating_transcripts]
    if not corroborator_paths:
        raise ValueError("at least one independent corroborating transcript is required")
    if refresh_exports and not write:
        raise ValueError("refresh_exports requires write=True")

    canonical = _object(read_json(canonical_path), "canonical transcript")
    secondary = _object(read_json(secondary_path), "secondary transcript")
    corroborators = [_object(read_json(path), f"corroborating transcript {path}") for path in corroborator_paths]
    canonical_segments = _segments(canonical, "canonical transcript")
    secondary_segments = _segments(secondary, "secondary transcript")
    corroborator_segments = [
        _segments(payload, f"corroborating transcript {path}")
        for payload, path in zip(corroborators, corroborator_paths, strict=True)
    ]

    updated = json.loads(json.dumps(canonical, ensure_ascii=False))
    updated_segments = _segments(updated, "updated canonical transcript")
    applied: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for position, segment in enumerate(canonical_segments):
        secondary_segment = _secondary_for_primary(segment, position, secondary_segments)
        if secondary_segment is None:
            continue
        primary_text = str(segment.get("text") or "")
        secondary_text = str(secondary_segment.get("text") or "")
        if _compact(primary_text) == _compact(secondary_text):
            continue
        evidence_rows = [
            row
            for rows in corroborator_segments
            for row in _overlapping(rows, segment)
        ]
        evidence_text = "".join(str(row.get("text") or "") for row in evidence_rows)
        patches, rejected = _supported_patches(
            primary_text,
            secondary_text,
            evidence_text=evidence_text,
        )
        for row in rejected:
            unresolved.append(
                {
                    **row,
                    "canonical_segment_position": position,
                    "start": float(segment.get("start") or 0.0),
                    "end": float(segment.get("end") or segment.get("start") or 0.0),
                }
            )
        if not patches:
            continue
        patched_text = primary_text
        segment_applied: list[dict[str, Any]] = []
        for patch in sorted(patches, key=lambda row: int(row["start"]), reverse=True):
            start = int(patch["start"])
            end = int(patch["end"])
            if patched_text[start:end] != str(patch["primary_text"]):
                raise ValueError("canonical span changed while applying ASR evidence patch")
            patched_text = patched_text[:start] + str(patch["secondary_text"]) + patched_text[end:]
            segment_applied.append(patch)
        if patched_text == primary_text:
            continue
        transformation = {
            "schema": SCHEMA,
            "source": "independent_asr_exact_corroboration",
            "secondary_transcript": str(secondary_path),
            "secondary_sha256": _sha256(secondary_path),
            "corroborating_transcripts": [str(path) for path in corroborator_paths],
            "corroborating_sha256": [_sha256(path) for path in corroborator_paths],
            "original_text": primary_text,
            "replacement_text": patched_text,
            "patches": list(reversed(segment_applied)),
            "timestamp_boundary_preserved": True,
            "human_confirmed": False,
            "updated_at": now_iso(),
        }
        target = updated_segments[position]
        target["text"] = patched_text
        target["corrected_text"] = patched_text
        target["changed"] = True
        history = [dict(row) for row in target.get("asr_evidence_transformations") or [] if isinstance(row, dict)]
        history.append(transformation)
        target["asr_evidence_transformations"] = history
        applied.append(
            {
                "canonical_segment_position": position,
                "start": float(segment.get("start") or 0.0),
                "end": float(segment.get("end") or segment.get("start") or 0.0),
                "original_text": primary_text,
                "replacement_text": patched_text,
                "patches": list(reversed(segment_applied)),
            }
        )

    updated["segments"] = updated_segments
    updated["source"] = "transcript_semantic_correction+independent_asr_adjudication"
    updated["updated_at"] = now_iso()
    status = "completed" if applied and not unresolved else ("degraded" if applied else "review_required")
    if write and applied and not refresh_exports:
        status = "degraded"
    result = {
        "schema": SCHEMA,
        "status": status,
        "ok": bool(applied),
        "quality_gate_passed": status == "completed",
        "write": bool(write),
        "refresh_exports_requested": bool(refresh_exports),
        "export_integrity_status": "preview" if not write else ("pending_refresh" if refresh_exports else "stale"),
        "bundle_dir": str(root),
        "canonical_path": str(canonical_path),
        "canonical_before_sha256": _sha256(canonical_path),
        "secondary_transcript": str(secondary_path),
        "secondary_sha256": _sha256(secondary_path),
        "corroborating_transcripts": [
            {"path": str(path), "sha256": _sha256(path)} for path in corroborator_paths
        ],
        "applied_segment_count": len(applied),
        "applied_patch_count": sum(len(row["patches"]) for row in applied),
        "unresolved_patch_count": len(unresolved),
        "applied": applied,
        "unresolved": unresolved,
        "operator_boundary": {
            "secondary_never_wholesale_promoted": True,
            "independent_exact_corroboration_required": True,
            "aligned_context_required": True,
            "single_character_insertions_and_all_deletions_never_auto_applied": True,
            "short_insertions_require_exact_independent_aligned_asr": True,
            "boundary_repetition_never_auto_applied": True,
            "long_gap_recovery_requires_independent_aligned_asr": True,
            "long_gap_recovery_requires_within_cue_bilateral_anchors": True,
            "numeric_or_fact_bearing_changes_never_auto_applied": True,
            "timestamp_and_segment_identity_preserved": True,
            "semantic_closure_base_preserved": True,
            "raw_asr_modified": False,
            "evaluation_reference_used": False,
            "human_confirmation_claimed": False,
            "no_network_call": True,
            "no_provider_or_location_fallback": True,
        },
        "updated_at": now_iso(),
    }
    if write and applied:
        report_path = root / "asr-evidence-autoadjudication.json"
        base_path = root / "asr-evidence-adjudicated-base.json"
        manifest_path = root / "manifest.json"
        manifest = _object(read_json(manifest_path), "bundle manifest") if manifest_path.exists() else {}
        with bundle_write_lock(root, operation="asr_evidence_autoadjudication", timeout_seconds=5):
            write_json(canonical_path, updated)
            write_json(base_path, updated)
            (root / "source-arbitrated-transcript.srt").write_text(
                _render_srt(updated_segments), encoding="utf-8"
            )
            (root / "source-arbitrated-transcript.md").write_text(
                _render_corrected_markdown(updated), encoding="utf-8"
            )
            manifest["source_arbitrated_transcript_json"] = canonical_path.name
            manifest["source_arbitrated_transcript_srt"] = "source-arbitrated-transcript.srt"
            manifest["source_arbitrated_transcript_markdown"] = "source-arbitrated-transcript.md"
            manifest["transcript_semantic_correction_base_json"] = base_path.name
            manifest["asr_evidence_autoadjudication"] = report_path.name
            manifest["asr_evidence_autoadjudication_status"] = status
            write_json(manifest_path, manifest)
            result["canonical_after_sha256"] = _sha256(canonical_path)
            result["semantic_correction_base_path"] = str(base_path)
            result["semantic_correction_base_sha256"] = _sha256(base_path)
            result["report_path"] = str(report_path)
            write_json(report_path, result)
        if refresh_exports:
            from .transcript_semantic_correction import _refresh_semantic_correction_outputs

            refresh_result = _refresh_semantic_correction_outputs(root)
            integrity = refresh_result.get("canonical_transcript_integrity") or {}
            integrity_status = str(integrity.get("status") or "unknown")
            result["refresh_exports"] = refresh_result
            result["export_integrity_status"] = integrity_status
            if str(refresh_result.get("status") or "") != "refreshed" or integrity_status not in {"passed", "no_export_hashes"}:
                result["status"] = "degraded"
                result["quality_gate_passed"] = False
            result["updated_at"] = now_iso()
            with bundle_write_lock(root, operation="asr_evidence_autoadjudication_report", timeout_seconds=5):
                manifest = _object(read_json(manifest_path), "bundle manifest") if manifest_path.exists() else {}
                manifest["asr_evidence_autoadjudication_status"] = result["status"]
                write_json(manifest_path, manifest)
                write_json(report_path, result)
    return result


def _supported_patches(
    primary_text: str,
    secondary_text: str,
    *,
    evidence_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_tokens = _tokens(primary_text)
    secondary_tokens = _tokens(secondary_text)
    matcher = difflib.SequenceMatcher(
        a=[row["normalized"] for row in primary_tokens],
        b=[row["normalized"] for row in secondary_tokens],
        autojunk=False,
    )
    raw_evidence_compact = _compact(evidence_text)
    evidence_compact = _compact(_canonicalize_domain_variants(evidence_text))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for operation, a0, a1, b0, b1 in matcher.get_opcodes():
        if operation == "equal":
            continue
        start, end = _char_span(primary_tokens, a0, a1, len(primary_text))
        secondary_start, secondary_end = _char_span(
            secondary_tokens, b0, b1, len(secondary_text)
        )
        primary_piece = primary_text[start:end]
        secondary_piece = secondary_text[secondary_start:secondary_end]
        canonical_secondary_piece = _canonicalize_domain_variants(secondary_piece)
        if operation == "replace":
            canonical_secondary_piece = _preserve_replaced_boundary_punctuation(
                primary_piece, canonical_secondary_piece
            )
        secondary_compact = _compact(canonical_secondary_piece)
        primary_compact = _compact(primary_piece)
        row = {
            "operation": operation,
            "start": start,
            "end": end,
            "primary_text": primary_piece,
            "secondary_text": canonical_secondary_piece,
            "raw_secondary_text": secondary_piece,
            "support_rule": "aligned_context_in_independent_overlapping_asr",
        }
        if _FACT_RE.search(primary_piece + canonical_secondary_piece):
            rejected.append({**row, "reason": "fact_or_number_change_requires_review"})
            continue
        left_context = _token_context(secondary_tokens, max(0, b0 - 5), b0)
        right_context = _token_context(secondary_tokens, b1, min(len(secondary_tokens), b1 + 5))
        if len(left_context) + len(right_context) < 4:
            rejected.append({**row, "reason": "insufficient_aligned_context_for_auto_adjudication"})
            continue
        row["aligned_context"] = {"left": left_context, "right": right_context}
        if operation == "insert" and not primary_compact:
            if not 2 <= len(secondary_compact) <= 80:
                rejected.append({**row, "reason": "structural_delta_outside_targeted_recovery_bounds"})
                continue
            secondary_window = left_context + secondary_compact + right_context
            if len(secondary_compact) < 8:
                primary_window = left_context + right_context
                if _has_low_information_boundary_repetition(
                    secondary_compact, left_context, right_context
                ):
                    rejected.append({**row, "reason": "low_information_boundary_repetition_requires_review"})
                    continue
                if secondary_window not in evidence_compact:
                    rejected.append({**row, "reason": "short_insertion_not_exactly_supported_by_independent_aligned_asr"})
                    continue
                if primary_window and primary_window in raw_evidence_compact:
                    rejected.append({**row, "reason": "independent_evidence_also_supports_primary"})
                    continue
                row["support_rule"] = "short_gap_exact_in_independent_aligned_asr"
                accepted.append(row)
                continue
            # The existing four-character context floor must not be supplied
            # entirely by one neighbouring cue. Bind both anchors to this cue
            # before accepting a long near-subsequence insertion.
            if (
                len(left_context) < 2
                or len(right_context) < 2
                or not _compact(primary_text[:start]).endswith(left_context)
                or not _compact(primary_text[end:]).startswith(right_context)
            ):
                rejected.append({**row, "reason": "long_insertion_requires_within_cue_bilateral_anchors"})
                continue
            if not _near_subsequence_supported(
                secondary_window,
                evidence_compact,
                max_internal_gap=max(4, len(secondary_compact) // 5),
            ):
                rejected.append({**row, "reason": "long_insertion_not_supported_by_independent_aligned_asr"})
                continue
            row["support_rule"] = "long_gap_near_subsequence_in_independent_aligned_asr"
            accepted.append(row)
            continue
        if operation != "replace" or not primary_compact or not secondary_compact:
            rejected.append({**row, "reason": "insertion_or_deletion_not_auto_applied"})
            continue
        if len(secondary_compact) < 2:
            rejected.append({**row, "reason": "secondary_delta_too_short_for_auto_adjudication"})
            continue
        primary_supported = primary_compact in raw_evidence_compact
        expanded_primary_window = left_context + primary_compact + right_context
        expanded_primary_supported = bool(
            expanded_primary_window and expanded_primary_window in raw_evidence_compact
        )
        if (
            len(secondary_compact) >= 4
            and len(primary_compact) <= 3
            and len(secondary_compact) - len(primary_compact) >= 3
            and not expanded_primary_supported
            and _near_subsequence_supported(
                secondary_compact,
                evidence_compact,
                max_internal_gap=max(4, len(secondary_compact) // 5),
            )
        ):
            row["support_rule"] = "expanded_gap_near_subsequence_in_independent_aligned_asr"
            accepted.append(row)
            continue
        secondary_window = left_context + secondary_compact + right_context
        primary_window = left_context + primary_compact + right_context
        secondary_supported = secondary_window in evidence_compact
        primary_supported = primary_window in raw_evidence_compact
        if not secondary_supported:
            rejected.append({**row, "reason": "secondary_text_not_independently_supported"})
            continue
        if primary_supported and primary_compact != secondary_compact:
            rejected.append({**row, "reason": "independent_evidence_also_supports_primary"})
            continue
        accepted.append(row)
    return accepted, rejected


def _preserve_replaced_boundary_punctuation(primary: str, replacement: str) -> str:
    value = str(replacement or "")
    trailing = re.search(r"[，。！？；：、,.!?;:]+$", str(primary or ""))
    if trailing and not re.search(r"[，。！？；：、,.!?;:]+$", value):
        value += trailing.group(0)
    return value


def _canonicalize_domain_variants(value: str) -> str:
    text = str(value or "")
    try:
        from .transcript_semantic_correction import DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS
    except ImportError:
        return text
    for original in sorted(DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS, key=len, reverse=True):
        corrected = str(DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS[original][0])
        text = text.replace(original, corrected)
    return text


def _has_low_information_boundary_repetition(
    piece: str, left_context: str, right_context: str
) -> bool:
    for context, compare_suffix in (
        (left_context, True),
        (right_context, False),
    ):
        maximum = min(len(piece), len(context), 8)
        for width in range(maximum, 1, -1):
            boundary = context[-width:] if compare_suffix else context[:width]
            if piece.startswith(boundary) or piece.endswith(boundary):
                return True
    return False


def _near_subsequence_supported(needle: str, haystack: str, *, max_internal_gap: int) -> bool:
    if not needle or not haystack:
        return False
    first = haystack.find(needle[0])
    while first >= 0:
        position = first + 1
        skipped = 0
        matched = True
        for char in needle[1:]:
            next_position = haystack.find(char, position)
            if next_position < 0:
                matched = False
                break
            skipped += next_position - position
            if skipped > max_internal_gap:
                matched = False
                break
            position = next_position + 1
        if matched:
            return True
        first = haystack.find(needle[0], first + 1)
    return False


def _token_context(tokens: list[dict[str, Any]], start: int, end: int) -> str:
    return "".join(
        compact
        for row in tokens[start:end]
        if (compact := _compact(str(row.get("text") or "")))
    )


def _tokens(text: str) -> list[dict[str, Any]]:
    return [
        {
            "text": match.group(0),
            "normalized": _compact(match.group(0)).lower() or match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for match in _TOKEN_RE.finditer(text or "")
    ]


def _char_span(tokens: list[dict[str, Any]], start: int, end: int, text_length: int) -> tuple[int, int]:
    if start < end:
        return int(tokens[start]["start"]), int(tokens[end - 1]["end"])
    point = int(tokens[start]["start"]) if start < len(tokens) else text_length
    return point, point


def _secondary_for_primary(
    primary: dict[str, Any],
    position: int,
    secondary_segments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    identities = {
        f"segment-{position + 1:06d}",
        str(primary.get("id") or ""),
        str(primary.get("segment_id") or ""),
    }
    identities.discard("")
    for row in secondary_segments:
        primary_segment_id = str(row.get("primary_segment_id") or "").strip()
        if primary_segment_id and primary_segment_id in identities:
            return row
    overlaps = _overlapping(secondary_segments, primary)
    if not overlaps:
        return None
    return max(overlaps, key=lambda row: _overlap_seconds(row, primary))


def _overlapping(rows: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in rows if _overlap_seconds(row, target) > 0]


def _overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start = float(left.get("start") or 0.0)
    left_end = float(left.get("end") or left_start)
    right_start = float(right.get("start") or 0.0)
    right_end = float(right.get("end") or right_start)
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _segments(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in payload.get("segments") or [] if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"{label} contains no segments")
    return rows


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value
