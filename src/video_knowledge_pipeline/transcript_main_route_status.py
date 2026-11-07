from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .smart_summary_codex import smart_summary_quality_check
from .storage import bundle_write_lock, write_json
from .storage import read_json_object_or_empty as _read_json_object

SCHEMA = "video_knowledge_pipeline.transcript_main_route_status.v1"


def transcript_main_route_status(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Report whether the optimized transcript -> smart-summary route is closed.

    This is a read-oriented acceptance surface. It does not run ASR, LLMs,
    OCR, vision, or exports; it only inspects current artifacts and explains the
    next action needed to make the route true for a bundle.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_json_object(root / "manifest.json")
    checks = [
        _asr_full_mode_check(root, manifest),
        _postprocessed_check(root, manifest),
        _evidence_conflict_check(root, manifest),
        _corrected_transcript_check(root, manifest),
        _full_transcript_check(root, manifest),
        _smart_summary_check(root),
        _whisperx_alignment_check(root, manifest),
        _asr_ab_check(root, manifest),
    ]
    hard_checks = [row for row in checks if row.get("required")]
    failed = [row for row in hard_checks if not row.get("passed")]
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": "passed" if not failed else "needs_action",
        "ok": not failed,
        "hard_check_count": len(hard_checks),
        "hard_failed_count": len(failed),
        "checks": checks,
        "next_actions": _next_actions(checks),
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_asr_execution": True,
            "no_media_processing": True,
            "does_not_modify_raw_asr": True,
        },
        "artifacts": {
            "json": str(root / "transcript-main-route-status.json"),
            "markdown": str(root / "transcript-main-route-status.md"),
        },
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_main_route_status", timeout_seconds=1.0):
            write_json(root / "transcript-main-route-status.json", result)
            (root / "transcript-main-route-status.md").write_text(_render_markdown(result), encoding="utf-8")
            if isinstance(manifest, dict):
                manifest["transcript_main_route_status_json"] = "transcript-main-route-status.json"
                manifest["transcript_main_route_status_markdown"] = "transcript-main-route-status.md"
                manifest["transcript_main_route_status_summary"] = {
                    "status": result["status"],
                    "hard_failed_count": result["hard_failed_count"],
                    "updated_at": result["updated_at"],
                }
                write_json(root / "manifest.json", manifest)
    return result


def _asr_full_mode_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    plan = _find_asr_plan(root, manifest)
    if not plan:
        return _check(
            "asr_full_mode_plan",
            False,
            "advisory",
            "not_verified",
            "No asr-run-plan.json found in bundle; final transcript may have been imported or generated elsewhere.",
            required=False,
            next_action="Run plan-asr with preset=sensevoice to verify VAD+ITN+punc full mode for this media.",
        )
    full = plan.get("full_mode") if isinstance(plan.get("full_mode"), dict) else {}
    passed = (
        str(plan.get("preset") or "") in {"sensevoice", "funasr"}
        and str(full.get("vad_model") or "") == "fsmn-vad"
        and str(full.get("punc_model") or "") == "ct-punc"
        and bool(full.get("use_itn"))
        and bool(full.get("merge_vad"))
    )
    detail = (
        f"preset={plan.get('preset')}; vad={full.get('vad_model')}; punc={full.get('punc_model')}; "
        f"use_itn={full.get('use_itn')}; merge_vad={full.get('merge_vad')}; spk={full.get('spk_model') or ''}"
    )
    return _check("asr_full_mode_plan", passed, "advisory", "passed" if passed else "incomplete", detail, required=False)


def _postprocessed_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _first_existing(root, manifest, ["postprocessed_transcript_json"], ["postprocessed-transcript.json"])
    return _check(
        "postprocessed_transcript",
        bool(path),
        "hard",
        "present" if path else "missing",
        f"path={path}" if path else "postprocessed-transcript.json is missing",
        next_action=f"Run .\\scripts\\video-knowledge.ps1 postprocess-asr-transcript '{root}'" if not path else "",
    )


def _evidence_conflict_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    index_path = _first_existing(root, manifest, ["evidence_conflict_index_json"], ["evidence-conflict-index.json"])
    llm_pack_path = _first_existing(root, manifest, ["evidence_conflict_llm_pack_json"], ["evidence-conflict-llm-pack.json"])
    if not index_path:
        return _check(
            "evidence_conflict_index",
            False,
            "hard",
            "missing",
            "evidence-conflict-index.json is missing",
            next_action=f"Run .\\scripts\\video-knowledge.ps1 evidence-conflict-index '{root}' or transcript-evidence-correction-pipeline.",
        )
    index = _read_json_object(index_path)
    boundary = index.get("operator_boundary") if isinstance(index.get("operator_boundary"), dict) else {}
    filtered = bool(boundary.get("heuristic_risks_without_external_evidence_are_not_arbitration_items"))
    llm_pack_ok = bool(llm_pack_path) or int(index.get("llm_arbitration_count") or 0) == 0
    passed = filtered and llm_pack_ok
    detail = f"conflicts={index.get('conflict_count')}; llm_items={index.get('llm_arbitration_count')}; llm_pack={llm_pack_path or ''}"
    return _check("evidence_conflict_index", passed, "hard", "present" if passed else "incomplete", detail)


def _corrected_transcript_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _first_existing(
        root,
        manifest,
        ["source_arbitrated_transcript_json", "human_corrected_transcript_json", "llm_corrected_transcript_json", "corrected_transcript_json"],
        ["source-arbitrated-transcript.json", "llm-corrected-transcript.json", "corrected-transcript.json"],
    )
    source_kind = _source_kind(path)
    passed = bool(path and source_kind in {"source_arbitrated", "human_corrected", "llm_corrected", "corrected"})
    return _check(
        "corrected_transcript",
        passed,
        "hard",
        source_kind or "missing",
        f"path={path}" if path else "corrected/source-arbitrated transcript is missing",
        next_action="Run transcript-evidence-correction-pipeline, transcript-source-arbitration, or postprocess-asr-transcript." if not passed else "",
    )


def _full_transcript_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _first_existing(root, manifest, ["knowledge_note_transcript_markdown"], ["exports/full-transcript.md"])
    if not path:
        return _check("full_transcript", False, "hard", "missing", "exports/full-transcript.md is missing", next_action=f"Run .\\scripts\\video-knowledge.ps1 export-knowledge-note '{root}'")
    text = path.read_text(encoding="utf-8-sig")
    source_ok = "Source: `source_arbitrated_transcript`" in text or "Source: `corrected_transcript`" in text or "Source: `llm_corrected_transcript`" in text or "Source: `human_corrected_transcript`" in text
    segment_status_ok = "来源状态：`" in text and "仲裁状态：`" in text
    raw_fallback = "raw_asr_fallback_not_arbitrated" in text
    passed = source_ok and segment_status_ok and not raw_fallback
    detail = f"source_ok={source_ok}; segment_status_ok={segment_status_ok}; raw_fallback={raw_fallback}; path={path}"
    return _check("full_transcript", passed, "hard", "passed" if passed else "not_from_corrected_or_missing_segment_status", detail, next_action=f"Run export-knowledge-note after corrected transcript exists: {root}" if not passed else "")


def _smart_summary_check(root: Path) -> dict[str, Any]:
    try:
        quality = smart_summary_quality_check(root, write=False)
    except Exception as exc:
        return _check("smart_summary_quality", False, "hard", "error", str(exc), next_action="Run export-knowledge-note and smart-summary quality check after transcript correction.")
    passed = bool(quality.get("passed"))
    failed_keys = [str(row.get("key")) for row in quality.get("checks") or [] if isinstance(row, dict) and not row.get("passed")]
    detail = f"quality_status={quality.get('status')}; failed={','.join(failed_keys)}; summary={quality.get('summary_path')}"
    return _check("smart_summary_quality", passed, "hard", "passed" if passed else "failed", detail, next_action="Run chapter-level LLM smart-summary rewrite from corrected transcript, then export-knowledge-note." if not passed else "")


def _whisperx_alignment_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _first_existing(root, manifest, ["whisperx_alignment_run_json", "whisperx_alignment_transcript_json"], ["transcripts/whisperx-alignment-run.json"])
    corrected_value = " ".join(str(manifest.get(key) or "") for key in ("corrected_transcript_json", "source_arbitrated_transcript_json", "transcript_json"))
    replaces_asr = "whisperx" in corrected_value.lower()
    if not path:
        return _check("whisperx_alignment", not replaces_asr, "advisory", "optional_not_run", "WhisperX alignment not found; this is OK unless precise word timestamps/speakers are needed.", required=False)
    return _check("whisperx_alignment", not replaces_asr, "advisory", "present" if not replaces_asr else "misused_as_primary_asr", f"path={path}; replaces_asr={replaces_asr}", required=False)


def _asr_ab_check(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _first_existing(root, manifest, ["asr_ab_comparison_json"], ["asr-ab-comparison.json", "transcripts/asr-ab-sample/asr-ab-comparison.json"])
    if not path:
        matches = list(root.glob("**/asr-ab-comparison.json"))
        path = matches[0] if matches else None
    if not path:
        return _check("asr_ab_sample", False, "advisory", "optional_not_run", "No 5-minute ASR A/B comparison found.", required=False, next_action="Run asr-ab-sample-plan/run/compare for one 5-minute sample before choosing a second ASR.")
    data = _read_json_object(path)
    return _check("asr_ab_sample", True, "advisory", str(data.get("status") or "present"), f"path={path}", required=False)


def _find_asr_plan(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[Path] = []
    for key in ("asr_plan_path", "asr_run_plan", "asr_plan"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(_bundle_path(root, value))
        elif isinstance(value, dict):
            raw = value.get("plan_path") or value.get("path")
            if raw:
                candidates.append(_bundle_path(root, str(raw)))
    candidates.extend(root.glob("**/asr-run-plan.json"))
    for path in candidates:
        if path.exists():
            data = _read_json_object(path)
            if data:
                return data
    return None


def _first_existing(root: Path, manifest: dict[str, Any], keys: list[str], fallbacks: list[str]) -> Path | None:
    candidates: list[Path] = []
    for key in keys:
        value = str(manifest.get(key) or "").strip()
        if value:
            candidates.append(_bundle_path(root, value))
    candidates.extend(_bundle_path(root, value) for value in fallbacks)
    for path in candidates:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _source_kind(path: Path | None) -> str:
    if not path:
        return ""
    name = path.name.lower()
    if "source-arbitrated" in name:
        return "source_arbitrated"
    if "human-corrected" in name:
        return "human_corrected"
    if "llm-corrected" in name:
        return "llm_corrected"
    if "corrected" in name:
        return "corrected"
    if "normalized" in name:
        return "normalized_asr"
    return "unknown"


def _check(key: str, passed: bool, severity: str, status: str, detail: str, *, required: bool = True, next_action: str = "") -> dict[str, Any]:
    return {
        "key": key,
        "passed": bool(passed),
        "severity": severity,
        "required": bool(required),
        "status": status,
        "detail": detail,
        "next_action": next_action,
    }


def _next_actions(checks: list[dict[str, Any]]) -> list[str]:
    actions = []
    for row in checks:
        action = str(row.get("next_action") or "").strip()
        if action and action not in actions:
            actions.append(action)
    return actions


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Transcript Main Route Status",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Status: `{result.get('status', '')}`",
        f"- Hard failed: `{result.get('hard_failed_count', 0)}` / `{result.get('hard_check_count', 0)}`",
        f"- Updated: `{result.get('updated_at', '')}`",
        "",
        "## Checks",
        "",
        "| Check | Severity | Required | Passed | Status | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("checks") or []:
        if not isinstance(row, dict):
            continue
        detail = str(row.get("detail") or "").replace("|", "/")[:240]
        lines.append(f"| `{row.get('key', '')}` | `{row.get('severity', '')}` | `{row.get('required', '')}` | `{row.get('passed', '')}` | `{row.get('status', '')}` | {detail} |")
    actions = [str(item) for item in result.get("next_actions") or [] if str(item)]
    if actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend([f"- {action}" for action in actions])
    lines.extend(["", "## Boundary", ""])
    for key, value in (result.get("operator_boundary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path
