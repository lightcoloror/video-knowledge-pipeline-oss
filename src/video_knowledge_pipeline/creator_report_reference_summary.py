"""Read-only summary projection for the practical creator report.

An explicit strict summary is already validated by the caller's existing gate.
Otherwise only the registered, watermarked handoff copy can be a reference.
This adapter never prepares, registers, installs, or repairs a summary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .smart_summary_handoff import SCHEMA, STATE, WATERMARK, _inside, handoff_status, summary_input_path

_TEXT_BUDGET = 4 * 1024 * 1024
_METADATA_BUDGET = 4 * 1024 * 1024


def _bytes(path: Path, *, budget: int) -> bytes:
    if path.stat().st_size > budget:
        raise ValueError("summary_reference_size_budget")
    raw = path.read_bytes()
    if len(raw) > budget:
        raise ValueError("summary_reference_size_budget")
    return raw


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(_bytes(path, budget=_METADATA_BUDGET).decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("summary_reference_metadata_not_object")
    return value


def _source(artifact_id: str, role: str, path: Path, raw: bytes) -> dict[str, str]:
    return {"artifact_id": artifact_id, "role": role, "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _unavailable(key: str, detail: str = "") -> dict[str, Any]:
    return {"text": "", "status": "unavailable", "sources": [],
            "issues": [{"key": key, "detail": detail}]}


def _error_detail(exc: Exception) -> str:
    # OS errors can embed source absolute paths; retain a stable public reason.
    return type(exc).__name__ if isinstance(exc, OSError) else str(exc)


def _preflight_inputs(root: Path, state: dict[str, Any]) -> None:
    # The shared freshness check reads these exact metadata/dependency files.
    # Check their resolved paths before it can follow a link outside the Bundle.
    for name in ("manifest.json", "sidecar.json", "timeline.json",
                 "exports/smart-summary-input-pack.json",
                 "exports/smart-summary-chapter-fact-pack.json",
                 "exports/smart-summary-chapters.json",
                 "exports/smart-summary-llm-rewrite-pack.md",
                 "exports/smart-summary-section-llm-revisions.json"):
        _inside(root, name)
    for row in state.get("dependency_snapshot", {}).get("inputs", []):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"].strip():
            raise ValueError("summary_reference_invalid_dependency")
        _inside(root, row["path"])
    selected = summary_input_path(root)
    if selected is None:
        raise ValueError("summary_reference_missing_transcript")
    _inside(root, str(selected))
    # A selected corrected transcript can coexist with an older input pack.
    # Do not let the existing checker read an external pack-source pointer.
    pack = _inside(root, "exports/smart-summary-input-pack.json")
    if pack.is_file():
        packed = _mapping(pack)
        value = packed.get("transcript_source")
        if not value or not packed.get("transcript_source_sha256"):
            raise ValueError("summary_reference_input_pack_binding_missing")
        # The shared transcript-source checker resolves this field from cwd
        # rather than from the Bundle; guard its actual read target.
        pack_source = _inside(root, str(Path(str(value)).expanduser().resolve()))
        if pack_source != selected or packed["transcript_source_sha256"] != sha256_file(selected):
            raise ValueError("summary_reference_input_pack_stale")


def load_reference_summary(root: Path, *, strict_summary: str = "",
                           strict_summary_path: Path | None = None) -> dict[str, Any]:
    """Return validated text, a current registered reference, or its rejection.

    ``strict_summary`` is an explicit result of the caller's strict validation;
    a path by itself cannot confer that status. Paths are private provenance,
    and all returned fields are deterministic for the same stored evidence.
    """
    root = Path(root).resolve()
    if strict_summary.strip():
        sources = []
        if strict_summary_path is not None:
            try:
                path = _inside(root, str(strict_summary_path))
                raw = _bytes(path, budget=_TEXT_BUDGET)
                stored_text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
                expected_text = strict_summary.replace("\r\n", "\n").replace("\r", "\n")
                if stored_text != expected_text:
                    return _unavailable("strict_summary_content_changed")
                sources.append(_source("validated-summary-001", "validated_smart_summary", path, raw))
            except (OSError, ValueError, UnicodeError) as exc:
                return _unavailable("strict_summary_source_unavailable", _error_detail(exc))
        return {"text": strict_summary, "status": "validated", "sources": sources, "issues": []}

    try:
        state_path = _inside(root, STATE)
        if not state_path.is_file():
            return _unavailable("reference_summary_not_registered")
        state = _mapping(state_path)
        if state.get("schema") != SCHEMA or state.get("status") not in {"review-required", "completed"}:
            return _unavailable("reference_summary_not_registered")
        snapshot = state.get("dependency_snapshot")
        if not isinstance(snapshot, dict) or not snapshot.get("inputs") or (
                state.get("handoff_id") != snapshot.get("snapshot_sha256")):
            return _unavailable("reference_summary_binding_missing")
        for path_key, digest_key in (("raw_candidate_path", "candidate_sha256"),
                                     ("review_candidate_path", "review_candidate_sha256")):
            value, digest = state.get(path_key), state.get(digest_key)
            if not isinstance(value, str) or not value.strip() or not isinstance(digest, str) or len(digest) != 64:
                return _unavailable("reference_summary_not_registered")
            _inside(root, value)
        _preflight_inputs(root, state)
        current = handoff_status(root)
        if current.get("freshness") != "fresh":
            return _unavailable("reference_summary_binding_stale", str(current.get("detail") or current.get("reason") or ""))
        # A concurrently replaced registration cannot select a different file
        # after the checks above. The next call can assess the newer state.
        for key in ("handoff_id", "candidate_sha256", "raw_candidate_path",
                    "review_candidate_path", "review_candidate_sha256", "dependency_snapshot"):
            if current.get(key) != state.get(key):
                return _unavailable("reference_summary_registration_changed")
        path = _inside(root, state["review_candidate_path"])
        raw = _bytes(path, budget=_TEXT_BUDGET)
        text = raw.decode("utf-8-sig")
        if hashlib.sha256(raw).hexdigest() != state["review_candidate_sha256"]:
            return _unavailable("reference_summary_content_changed")
        if not text.startswith(WATERMARK + "\n\n") or not text[len(WATERMARK):].strip():
            return _unavailable("reference_summary_watermark_missing")
        return {"text": text, "status": "reference",
                "sources": [_source("reference-summary-001", "reference_smart_summary", path, raw)],
                "issues": [{"key": "reference_summary_review_required",
                            "detail": "参考总结，待核；尚未通过正式总结质量与来源审核。",
                            "remaining_review_items": state.get("remaining_review_items") or []}]}
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        return _unavailable("reference_summary_source_unavailable", _error_detail(exc))
