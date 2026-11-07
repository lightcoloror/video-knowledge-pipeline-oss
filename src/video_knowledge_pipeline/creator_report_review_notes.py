"""Read-only, cue-local projection of the existing ASR review receipt.

The adjudication module remains the authority for review closure. This adapter
restricts text evidence to the Bundle and locates unresolved IDs on raw cues.
Original media retain the existing adjudication workflow's hash-bound source
boundary; they may legitimately live outside the Bundle.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .asr_diff_adjudication import _checked_text_binding, asr_diff_review_status
from .courseware_material import safe_path
from .smart_summary_input_pack import select_canonical_transcript_path
from .storage import read_json

_RECEIPT = "asr-consensus-adjudication-review.json"
_TEXT_LIMIT = 64 * 1024 * 1024


def _mapping(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _TEXT_LIMIT:
        raise ValueError("review metadata exceeds text budget")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("review metadata must be an object")
    return value


def _review_inputs(root: Path, receipt: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Preflight pointers followed by the shared status reader before calling it."""
    pack_path = _checked_text_binding(root, receipt["pack_binding"])
    decisions_path = _checked_text_binding(root, receipt["decisions_binding"])
    pack, decisions = _mapping(pack_path), _mapping(decisions_path)
    sources = {role: _checked_text_binding(root, binding)
               for role, binding in pack["source_bindings"].items()}
    for binding in decisions["evidence_bindings"].values():
        _checked_text_binding(root, binding)
    comparison = _mapping(sources["consensus"])
    for key in ("primary_transcript", "secondary_transcript"):
        if comparison.get(key):
            # The existing comparison builder reads these as process-relative
            # paths, rather than Bundle-relative paths. Check that exact target.
            path = Path(str(comparison[key])).absolute()
            safe_path(root, str(path))
    # Do not impose the text-artifact boundary on original source media. The
    # native reader validates comparison.media_path against source_media_binding
    # using current bytes/hash/size; this projection opens no media itself.
    manifest_path = safe_path(root, "manifest.json", exists=False)
    manifest = _mapping(manifest_path) if manifest_path.is_file() else {}
    selected = select_canonical_transcript_path(root, manifest)
    if selected is not None:
        safe_path(root, str(selected))
    # use_canonical decisions consult these local, optional semantic ledgers.
    for name in ("transcript-semantic-correction-decision-ledger.json",
                 "transcript-semantic-correction-validation.json"):
        path = safe_path(root, name, exists=False)
        if path.is_file() and path.stat().st_size > _TEXT_LIMIT:
            raise ValueError("review metadata exceeds text budget")
    return pack_path, pack


def _span(row: dict[str, Any]) -> tuple[float, float] | None:
    start, end = row.get("start"), row.get("end")
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        return None
    return start, end


def _attach(cue: dict[str, Any], issue: dict[str, Any]) -> None:
    cue["issues"].append(copy.deepcopy(issue))
    cue["status"] = "review-required"
    for ref in issue["evidence_refs"]:
        if ref not in cue["evidence_refs"]:
            cue["evidence_refs"].append(ref)


def collect_review_notes(root: Path, transcript: list[dict]) -> dict[str, Any]:
    """Preserve input cues and annotate current, unresolved adjudication spans.

    Missing review evidence does not establish an ASR disagreement. Invalid or
    stale evidence produces one global note and is never used for positioning.
    No source text, review decision, ledger, or input dictionary is written.
    """
    root = Path(root).resolve()
    cues = copy.deepcopy(transcript)
    issues: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        cue["issues"] = list(cue.get("issues") or [])
        cue["evidence_refs"] = list(cue.get("evidence_refs") or [])
        cue.setdefault("status", "transcribed")
        if not str(cue.get("text") or "").strip():
            issue: dict[str, Any] = {"code": "transcript_empty_cue",
                "message": "此处原转写为空，待核", "evidence_refs": [], "cue_index": index}
            if (span := _span(cue)) is not None:
                issue.update(start=span[0], end=span[1])
            issues.append(issue)
            _attach(cue, issue)

    try:
        receipt_path = safe_path(root, _RECEIPT, exists=False)
        if not receipt_path.is_file():
            return {"transcript": cues, "issues": issues}
        receipt = _mapping(receipt_path)
        pack_path, pack = _review_inputs(root, receipt)
        current = asr_diff_review_status(root)
        if current.get("status") not in {"completed", "review-required"}:
            raise ValueError("review receipt is stale or invalid")
        if (current.get("pack_binding") != receipt["pack_binding"]
                or current.get("decisions_binding") != receipt["decisions_binding"]):
            raise ValueError("review bindings changed")
        _checked_text_binding(root, receipt["pack_binding"])
        _checked_text_binding(root, receipt["decisions_binding"])
        resolved = {row["diff_id"] for row in current.get("resolved", [])}
        pack_ref = pack_path.relative_to(root).as_posix()
        for diff in pack["differences"]:
            identity = diff["diff_id"]
            if identity in resolved:
                continue
            issue = {"code": "asr_difference_pending", "message": "两路识别不同，待核",
                     "diff_id": identity, "evidence_refs": [
                         _RECEIPT, pack_ref + "#diff_id=" + quote(str(identity), safe="")]}
            if diff.get("primary_text") == diff.get("secondary_text"):
                issue["message"] = "识别结果复核尚未完成，待核"
            span = (_span(diff.get("estimated_time") or {})
                    if diff.get("time_basis") != "aggregate_bounds_unresolved" else None)
            if span is not None:
                issue.update(start=span[0], end=span[1])
                for cue in cues:
                    cue_span = _span(cue)
                    if cue_span and (min(span[1], cue_span[1]) > max(span[0], cue_span[0])
                                     or span[0] == span[1] and cue_span[0] <= span[0] < cue_span[1]):
                        _attach(cue, issue)
            issues.append(issue)
        if current.get("remaining_count", 0) and not any(
                issue["code"] == "asr_difference_pending" for issue in issues):
            issues.append({"code": "asr_review_incomplete", "message": "待核记录尚未闭合",
                           "evidence_refs": [_RECEIPT]})
    except (OSError, ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        issues.append({"code": "asr_review_stale", "message": "待核记录已过期",
                       "evidence_refs": [_RECEIPT]})
    return {"transcript": cues, "issues": issues}
