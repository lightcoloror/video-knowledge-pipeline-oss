"""Read-only stage completion derived from native evidence, not exit codes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .smart_summary_handoff import handoff_status, mapping, summary_input_path
from .storage import read_json


def workflow_completion_status(bundle_dir: str | Path, *, include_readable_export: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    from .knowledge_note_export import canonical_export_integrity_status
    from .production_artifact_gate import evaluate_production_artifact_gate, transcript_completeness_status
    from .smart_summary_codex import _existing_codex_summary, smart_summary_quality_check
    from .transcript import parse_transcript
    stages: list[dict[str, Any]] = []

    def add(key: str, status: str, detail: Any, action: str = "") -> None:
        stages.append({"key": key, "status": status, "evidence": detail, "next_action": action})

    manifest = mapping(root / "manifest.json")
    source = summary_input_path(root)
    timeline = []
    try:
        timeline = read_json(root / "timeline.json")
        input_ok = bool(manifest) and isinstance(timeline, list) and source is not None
        cues = parse_transcript(source) if source else []
    except (OSError, ValueError, TypeError):
        input_ok, cues = False, []
    add("input_validation", "completed" if input_ok else "blocked", str(source or ""),
        "Validate Bundle manifest, timeline and transcript input.")
    transcript_quality = mapping(root / "transcript-quality-gate.json")
    quality_bad = transcript_quality.get("status") == "failed" or transcript_quality.get("ok") is False
    add("primary_transcript", "review-required" if cues and quality_bad else "completed" if cues else "blocked",
        {"path": str(source or ""), "segments": len(cues), "quality_status": transcript_quality.get("status", "missing")},
        "Inspect transcript quality; process exit 0 alone is not acceptance.")

    dual = mapping(root / "dual-asr-review.json")
    consensus = dual.get("consensus") if isinstance(dual.get("consensus"), dict) else {}
    dual_ok = dual.get("ok") is True and dual.get("status") in {
        "dual_model_completed", "dual_inputs_compared", "targeted_review_completed"}
    dual_stages = dual.get("stages") if isinstance(dual.get("stages"), dict) else {}
    for role in ("primary", "secondary"):
        row = dual_stages.get(role) if isinstance(dual_stages.get(role), dict) else {}
        value = str(row.get("path") or dual.get(f"{role}_transcript") or "")
        path = root / value
        if not value or not row.get("sha256") or not path.is_file() or sha256_file(path) != row["sha256"]:
            dual_ok = False
    unresolved = consensus.get("review_required_count")
    from .asr_diff_adjudication import asr_diff_review_status
    fine_review = asr_diff_review_status(root)
    bindings = fine_review.get('source_bindings') or {}
    review_matches_dual = fine_review.get('ok') is True and bool(bindings.get('dual_workflow'))
    for role in ('primary','secondary'):
        review_matches_dual = review_matches_dual and (
            bindings.get(role,{}).get('sha256') == dual_stages.get(role,{}).get('sha256'))
    if review_matches_dual:
        unresolved = fine_review['remaining_count']
    add("secondary_asr_and_differences", "completed" if dual_ok and unresolved == 0 else
        "review-required" if dual_ok else "blocked", {"status": dual.get("status", "missing"),
        "review_required_count": unresolved,"raw_aggregate_review_count":consensus.get('review_required_count'),
        "fine_review_status":fine_review.get('status'),"fine_review_resolved_count":fine_review.get('resolved_count',0),
        "fine_review_remaining_count":fine_review.get('remaining_count')},
        "Complete or bind the second ASR and resolve its actual difference evidence.")
    completeness = transcript_completeness_status(root)
    add("speech_completeness", "completed" if completeness.get("passed") is True and
        completeness.get("speech_completeness_verified") is True else "review-required",
        completeness, "Review independent speech coverage and unresolved segment boundaries.")

    gate = evaluate_production_artifact_gate(root, discover_prior_reviews=False, write=False)
    applicability = gate.get("media_applicability") or {}
    from .knowledge_coverage import build_knowledge_coverage
    # Reuse current channel evidence instead of a stale manifest summary.
    coverage = build_knowledge_coverage(manifest, timeline if isinstance(timeline, list) else [], bundle_dir=root)
    visual_na = applicability.get("visual_status") == "not_applicable"
    add("visual_evidence", "not-applicable" if visual_na else
        "completed" if coverage.get("status") in {"passed", "ok"} else "blocked", applicability,
        "Verify video frame/OCR/visual coverage; only source-bound audio-only input is exempt.")
    handoff = handoff_status(root)
    installed = _existing_codex_summary(root)
    add("smart_summary", "completed" if installed else
        handoff["status"] if handoff.get("status") in {"awaiting-agent", "review-required", "blocked"} else "awaiting-agent",
        {"path": str(installed or ""), "candidate_path": handoff.get("review_candidate_path", "")},
        "Use generate-smart-summary-with-codex, then submit a bound candidate through --input-md.")
    quality = smart_summary_quality_check(root, summary_path=installed, require_codex=True, write=False) if installed else {}
    quality_ok = quality.get("passed") is True and gate.get("formal_generation_allowed") is True
    add("summary_validation", "completed" if quality_ok else "review-required" if installed else "blocked",
        {"quality_status": quality.get("status", "missing"), "production_blockers": gate.get("blocking_reasons", [])},
        "Resolve summary quality and production gates; candidate existence is not formal acceptance.")
    readers = [root / "exports" / name for name in ("full-transcript.md", "smart-summary.md")]
    readable = all(p.is_file() and p.stat().st_size > 0 for p in readers)
    try:
        integrity = canonical_export_integrity_status(root) if readable else {}
    except (OSError, ValueError, TypeError):
        integrity = {"status": "invalid_reader_receipt", "passed": False}
    # Require exact summary projection as well as the existing transcript receipt.
    summary_matches = bool(installed and readable and readers[1].read_text(encoding="utf-8-sig").strip()
                           == installed.read_text(encoding="utf-8-sig").strip())
    if include_readable_export:
        creator = _creator_readable_export_status(root, manifest, installed)
        creator_mode = (root / 'exports/creator-report/latest.json').is_file()
        if creator_mode:
            add('readable_export', 'completed' if quality_ok and creator['passed'] else 'blocked', creator,
                f".\\scripts\\video-knowledge.ps1 export-creator-report '{root}'")
        else:
            add("readable_export", "completed" if readable and summary_matches and quality_ok and
                integrity.get("status") == "passed" else "review-required" if readable else "blocked",
                {"files": [str(p) for p in readers], "integrity": integrity, "summary_matches": summary_matches},
                f".\\scripts\\video-knowledge.ps1 export-knowledge-note '{root}' --no-transcript-evidence-check")
    remaining = [s for s in stages if s["status"] not in {"completed", "not-applicable"}]
    status = "completed" if not remaining else "awaiting-agent" if any(
        s["status"] == "awaiting-agent" for s in remaining) else "review-required" if any(
        s["status"] == "review-required" for s in remaining) else "blocked"
    return {"schema": "video_knowledge_pipeline.workflow_completion.v1", "status": status,
            "completed": not remaining, "stages": stages, "remaining_stages": [s["key"] for s in remaining],
            "next_actions": [s["next_action"] for s in remaining], "model_called": False}


def _creator_readable_export_status(root: Path, manifest: dict[str, Any], installed: Path | None) -> dict[str, Any]:
    """Check native report files and freshness with the export stage excluded."""
    from .creator_report_renderers import bundle_file
    from .creator_report import creator_report_status
    try:
        from .smart_summary_input_pack import select_canonical_transcript_path
        from .transcript import parse_transcript
        transcript = select_canonical_transcript_path(root, manifest)
        if not transcript or not installed:
            return {'passed': False, 'reason': 'missing_or_stale_transcript_or_summary'}
        receipt = read_json(root / 'exports/creator-report/latest.json')
        artifacts = {row['format']: row for row in receipt['artifacts']}
        for kind in ('html','pdf','md','projection'):
            row = artifacts[kind]
            if sha256_file(bundle_file(root,row['path'])) != row['sha256']:
                return {'passed': False, 'reason': 'artifact_hash_mismatch'}
        projection = read_json(bundle_file(root,artifacts['projection']['path']))
        expected = [(cue.start,cue.end,cue.text,cue.speaker) for cue in parse_transcript(transcript)]
        actual = [(row['start'],row['end'],row['text'],row.get('speaker','')) for row in projection['transcript']]
        if not expected or actual != expected:
            return {'passed': False, 'reason': 'canonical_transcript_projection_mismatch'}
        if projection.get('summary','').strip() != installed.read_text(encoding='utf-8-sig').strip():
            return {'passed': False, 'reason': 'summary_projection_mismatch'}
        installed_hash = sha256_file(installed)
        if not any(row.get('role') in {'smart_summary','validated_smart_summary'}
                   and Path(row['path']).resolve() == installed.resolve()
                   and row['sha256'] == installed_hash for row in projection['sources']):
            return {'passed': False, 'reason': 'summary_source_mismatch'}
        # The report builder excludes this readable-export stage from its own
        # inputs, so reuse its native freshness check without a recursive loop.
        freshness = creator_report_status(root)
        if freshness.get('status') != 'fresh' or freshness.get('export_status') != 'exported_candidate':
            return {'passed': False, 'reason': 'creator_export_not_fresh', 'freshness': freshness}
        return {'passed':True,'scope':'hash_bound_native_html_pdf_md_exports','files':[r['path'] for r in artifacts.values()]}
    except (OSError,ValueError,TypeError,KeyError,AttributeError,StopIteration):
        return {'passed':False,'reason':'invalid_or_incomplete_creator_export'}


def finalize_readable_delivery(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Resume the local reader projection only after the native summary gates pass."""
    completion = workflow_completion_status(bundle_dir)
    stages = {row["key"]: row for row in completion["stages"]}
    if write and stages["summary_validation"]["status"] == "completed" and stages["readable_export"]["status"] != "completed":
        from .knowledge_note_export import export_knowledge_note
        try:
            exported = export_knowledge_note(bundle_dir, run_transcript_evidence_check=False, write=True)
        except (OSError, ValueError, RuntimeError) as exc:
            completion["export_error"] = f"{type(exc).__name__}: {exc}"
        else:
            completion = workflow_completion_status(bundle_dir)
            completion["readable_export_result"] = exported
    return completion


_VOLATILE = {"updated_at", "generated_at", "checked_at", "created_at", "heartbeat_at",
             "state_revision", "transition_count", "elapsed_seconds", "duration_ms"}


def semantic_progress_signature(root: Path) -> str:
    """Keep evidence/count/state changes; exclude audit timestamp-only rewrites."""
    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in _VOLATILE}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value
    rows: dict[str, Any] = {}
    for name in ("manifest.json", "timeline.json", "transcript-quality-gate.json", "dual-asr-review.json",
                 "exports/smart-summary-handoff.json", "exports/smart-summary-quality.json",
                 "exports/reader-export-receipt.json", "transcript-semantic-correction-status.json"):
        path = root / name
        if path.is_file():
            try:
                rows[name] = stable(read_json(path))
            except (OSError, ValueError):
                rows[name] = "invalid"
    for path in (summary_input_path(root), root / "exports/smart-summary.codex.md", root / "exports/smart-summary.md"):
        if path and path.is_file():
            rows[str(path)] = sha256_file(path)
    return canonical_json_sha256(rows)
