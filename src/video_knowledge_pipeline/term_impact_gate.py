from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage import read_json
from .storage import read_json_object_or_empty as _read_optional_object

PASSING_STATUSES = {"passed", "no_source_aliases"}


def load_term_correction_impact_gate(bundle_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the smart-summary gate state for terminology correction impact.

    The gate is intentionally read-only. It only becomes blocking when the bundle
    has evidence that term correction/arbitration happened or a report exists.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_obj = manifest if isinstance(manifest, dict) else _read_manifest(root)
    candidates: list[Path] = []
    for key in ("term_correction_impact_report_json",):
        value = str(manifest_obj.get(key) or "").strip()
        if value:
            candidates.append(_bundle_path(root, value))
    candidates.append(root / "term-correction-impact-report.json")
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve()).lower()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = read_json(path)
        except Exception as exc:
            return _missing_or_invalid(root, required=True, status="invalid_report", detail=f"failed to read term impact report: {exc}")
        if not isinstance(payload, dict):
            return _missing_or_invalid(root, required=True, status="invalid_report", detail="term impact report is not a JSON object")
        return _gate_from_report(path, payload)
    required = _term_impact_required(root, manifest_obj)
    return _missing_or_invalid(
        root,
        required=required,
        status="missing_report" if required else "not_required",
        detail="term impact report missing after high-confidence term correction" if required else "no term impact report required for this bundle",
    )


def _gate_from_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status") or "unknown").strip()
    final_total = _safe_int(payload.get("final_export_alias_total"))
    output_total = _safe_int(payload.get("output_alias_total"))
    source_total = _safe_int(payload.get("source_alias_total"))
    replacement_count = _safe_int(payload.get("replacement_count"))
    passed = status in PASSING_STATUSES and final_total == 0
    required = replacement_count > 0 or source_total > 0 or output_total > 0 or final_total > 0
    if passed:
        detail = f"status={status}; final_export_alias_total={final_total}; final outputs clean"
    else:
        detail = f"status={status}; final_export_alias_total={final_total}; output_alias_total={output_total}; run term correction before final summary"
    return {
        "exists": True,
        "required": required,
        "passed": passed,
        "status": status,
        "path": str(path.resolve()),
        "replacement_count": replacement_count,
        "source_alias_total": source_total,
        "output_alias_total": output_total,
        "final_export_alias_total": final_total,
        "reduction_rate": payload.get("reduction_rate"),
        "final_clean_rate": payload.get("final_clean_rate"),
        "detail": detail,
        "next_actions": [str(value) for value in (payload.get("next_actions") or []) if str(value)],
    }


def _missing_or_invalid(root: Path, *, required: bool, status: str, detail: str) -> dict[str, Any]:
    return {
        "exists": False,
        "required": bool(required),
        "passed": not required and status == "not_required",
        "status": status,
        "path": str(root / "term-correction-impact-report.json"),
        "replacement_count": 0,
        "source_alias_total": 0,
        "output_alias_total": 0,
        "final_export_alias_total": 0,
        "reduction_rate": None,
        "final_clean_rate": None,
        "detail": detail,
        "next_actions": [f"run .\\scripts\\video-knowledge.ps1 term-correction-impact-report '{root}'"] if required else [],
    }


def _term_impact_required(root: Path, manifest: dict[str, Any]) -> bool:
    summary = manifest.get("term_correction_impact_summary") if isinstance(manifest.get("term_correction_impact_summary"), dict) else {}
    if summary:
        return True
    arbitration = _read_optional_object(_bundle_path(root, str(manifest.get("transcript_source_arbitration_json") or "transcript-source-arbitration.json")))
    quality = arbitration.get("quality_summary") if isinstance(arbitration.get("quality_summary"), dict) else {}
    arb_summary = arbitration.get("summary") if isinstance(arbitration.get("summary"), dict) else {}
    high_conf = _safe_int(quality.get("high_confidence_term_replacements", arb_summary.get("high_confidence_term_replacements")))
    changed = _safe_int(quality.get("changed_segments", arb_summary.get("changed_segments")))
    if high_conf > 0 or changed > 0:
        return True
    for fallback in (
        "term-arbitration-codex-result.json",
        "term-arbitration-codex-pack.json",
        "term-arbitration-glossary.json",
        "term-resolution-report.json",
        "term-resolution.json",
        "transcript-correction-pack.json",
    ):
        if (root / fallback).exists():
            return True
    return False


def _read_manifest(root: Path) -> dict[str, Any]:
    return _read_optional_object(root / "manifest.json")



def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0
