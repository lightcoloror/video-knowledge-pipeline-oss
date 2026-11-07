from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .file_hash import sha256_file
from .models import now_iso
from .storage import read_json, write_json, write_text_atomic


SCHEMA = "video_knowledge_pipeline.transcript_downstream_refresh.v1"
INVALIDATION_SCHEMA = "video_knowledge_pipeline.smart_summary_invalidation.v1"

_SUMMARY_CANDIDATES = (
    "exports/smart-summary.md",
    "exports/smart-summary.codex.md",
    "exports/smart-summary.llm.md",
    "smart-summary.codex.md",
    "codex-smart-summary.md",
)


def invalidate_smart_summary_for_transcript_change(
    bundle_dir: str | Path,
    *,
    canonical_before_sha256: str,
    canonical_after_sha256: str,
    reason: str = "canonical_transcript_updated",
    write: bool = True,
) -> dict[str, Any]:
    """Record and archive summaries that were built from an older transcript."""

    root = Path(bundle_dir).expanduser().resolve()
    exports = root / "exports"
    marker_path = exports / "smart-summary-invalidation.json"
    report_path = exports / "smart-summary-invalidation.md"
    rows: list[dict[str, Any]] = []
    for relative in _SUMMARY_CANDIDATES:
        path = root / relative
        if not path.is_file():
            continue
        digest = sha256_file(path)
        archive_relative = (
            f"exports/invalidated-summaries/{digest[:12]}-{path.name}"
        )
        rows.append(
            {
                "path": str(path),
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "archive_path": str(root / archive_relative),
                "archive_relative_path": archive_relative,
            }
        )

    result = {
        "schema": INVALIDATION_SCHEMA,
        "status": "invalidated",
        "ok": True,
        "write": bool(write),
        "bundle_dir": str(root),
        "reason": str(reason or "canonical_transcript_updated"),
        "canonical_before_sha256": canonical_before_sha256,
        "canonical_after_sha256": canonical_after_sha256,
        "invalidated_summary_count": len(rows),
        "invalidated_summaries": rows,
        "requires_summary_regeneration": True,
        "marker_path": str(marker_path),
        "report_path": str(report_path),
        "updated_at": now_iso(),
    }
    if not write:
        return result

    for row in rows:
        source = Path(row["path"])
        archive = Path(row["archive_path"])
        if not archive.exists():
            write_text_atomic(
                archive,
                source.read_text(encoding="utf-8-sig"),
                encoding="utf-8",
            )
    write_json(marker_path, result)
    write_text_atomic(report_path, _render_invalidation_markdown(result))
    _update_manifest(
        root,
        {
            "smart_summary_invalidation_json": str(
                marker_path.relative_to(root)
            ).replace("\\", "/"),
            "smart_summary_invalidation_markdown": str(
                report_path.relative_to(root)
            ).replace("\\", "/"),
            "smart_summary_status": "invalidated_after_transcript_update",
            "smart_summary_requires_regeneration": True,
        },
    )
    return result


def refresh_transcript_downstream_outputs(
    bundle_dir: str | Path,
    *,
    canonical_before_sha256: str,
    canonical_after_sha256: str,
    reason: str = "targeted_asr_merge",
    write: bool = True,
) -> dict[str, Any]:
    """Rebuild local downstream artifacts and rerun the final quality gates.

    This function intentionally never calls a model provider. It prepares the
    existing summary workflow and leaves the bundle in
    ``needs_summary_regeneration`` until an approved model route installs a new
    summary.
    """

    root = Path(bundle_dir).expanduser().resolve()
    canonical_path = root / "source-arbitrated-transcript.json"
    current_canonical_hash = sha256_file(canonical_path) if canonical_path.is_file() else ""
    canonical_before_sha256 = canonical_before_sha256 or current_canonical_hash
    canonical_after_sha256 = canonical_after_sha256 or current_canonical_hash
    report_path = root / "transcript-downstream-refresh.json"
    report_markdown_path = root / "transcript-downstream-refresh.md"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "write": bool(write),
        "reason": str(reason or "targeted_asr_merge"),
        "canonical_before_sha256": canonical_before_sha256,
        "canonical_after_sha256": canonical_after_sha256,
        "status": "running",
        "ok": False,
        "steps": [],
        "artifacts": {
            "report_json": str(report_path),
            "report_markdown": str(report_markdown_path),
            "summary_invalidation": str(
                root / "exports" / "smart-summary-invalidation.json"
            ),
            "full_transcript": str(root / "exports" / "full-transcript.md"),
            "smart_summary_input_pack": str(
                root / "exports" / "smart-summary-input-pack.json"
            ),
            "transcript_quality_gate": str(
                root / "transcript-quality-gate.json"
            ),
            "smart_summary_quality_gate": str(
                root / "exports" / "smart-summary-quality.json"
            ),
        },
        "operator_boundary": {
            "local_deterministic_refresh_only": True,
            "external_model_called": False,
            "summary_model_execution_requires_existing_consent_route": True,
            "timeline_truth_source_replaced": False,
        },
        "updated_at": now_iso(),
    }

    invalidation = _run_step(
        "invalidate_smart_summary",
        lambda: invalidate_smart_summary_for_transcript_change(
            root,
            canonical_before_sha256=canonical_before_sha256,
            canonical_after_sha256=canonical_after_sha256,
            reason=reason,
            write=write,
        ),
    )
    result["steps"].append(invalidation)

    # Import lazily: this is glue over established VKP front doors and keeps
    # the targeted-ASR merge module free of import cycles.
    from .knowledge_note_export import export_knowledge_note
    from .quality_finalize import finalize_quality_outputs
    from .knowledge_coverage import audit_knowledge_coverage
    from .webui_bridge import refresh_bundle_review_html
    from .transcript_quality_gate import run_transcript_quality_gate
    from .smart_summary_codex import smart_summary_quality_check
    from .quality_console import export_quality_console
    from .task_console import export_task_console
    from .video_workbench import export_video_workbench

    rebuild_steps: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        (
            "export_knowledge_note",
            lambda: export_knowledge_note(
                root,
                run_transcript_evidence_check=False,
                write=write,
            ),
        ),
        (
            "prepare_quality_outputs",
            lambda: finalize_quality_outputs(
                root,
                execute_llm=False,
                auto_from_profile=False,
                write=write,
            ),
        ),
        (
            "audit_knowledge_coverage",
            lambda: audit_knowledge_coverage(root, write=write),
        ),
        (
            "refresh_review_html",
            lambda: refresh_bundle_review_html(root, write=write),
        ),
    )
    for name, action in rebuild_steps:
        result["steps"].append(_run_step(name, action))

    transcript_gate = _run_step(
        "final_transcript_quality_gate",
        lambda: run_transcript_quality_gate(
            root,
            input_path=canonical_path,
            write=write,
        ),
    )
    summary_gate = _run_step(
        "final_smart_summary_quality_gate",
        lambda: smart_summary_quality_check(
            root,
            require_codex=True,
            write=write,
        ),
    )
    result["steps"].extend((transcript_gate, summary_gate))

    console_steps: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        (
            "export_quality_console",
            lambda: export_quality_console(root, write=write),
        ),
        (
            "export_task_console",
            lambda: export_task_console(root, write=write, refresh=write),
        ),
        (
            "export_video_workbench",
            lambda: export_video_workbench(root, write=write),
        ),
    )
    for name, action in console_steps:
        result["steps"].append(_run_step(name, action))

    local_failures = [
        row["name"] for row in result["steps"] if not row["ok"]
    ]
    transcript_payload = transcript_gate.get("result") or {}
    summary_payload = summary_gate.get("result") or {}
    transcript_passed = bool(
        transcript_gate["ok"]
        and transcript_payload.get(
            "ok",
            int(transcript_payload.get("fail_count") or 0) == 0,
        )
    )
    summary_passed = bool(
        summary_gate["ok"] and summary_payload.get("passed")
    )
    result["local_refresh_completed"] = not local_failures
    result["transcript_quality_passed"] = transcript_passed
    result["smart_summary_quality_passed"] = summary_passed
    result["requires_summary_regeneration"] = not summary_passed
    result["full_pipeline_production_qualified"] = bool(
        not local_failures and transcript_passed and summary_passed
    )
    if local_failures or not transcript_passed:
        result["status"] = "degraded"
    elif not summary_passed:
        result["status"] = "needs_summary_regeneration"
    else:
        result["status"] = "completed"
    result["ok"] = result["status"] == "completed"
    result["failed_steps"] = local_failures
    result["next_actions"] = _next_actions(result)

    if write:
        write_json(report_path, result)
        write_text_atomic(
            report_markdown_path,
            _render_refresh_markdown(result),
        )
        _update_manifest(
            root,
            {
                "transcript_downstream_refresh_json": report_path.name,
                "transcript_downstream_refresh_markdown": (
                    report_markdown_path.name
                ),
                "transcript_downstream_refresh_status": result["status"],
                "smart_summary_requires_regeneration": (
                    result["requires_summary_regeneration"]
                ),
                "full_pipeline_production_qualified": (
                    result["full_pipeline_production_qualified"]
                ),
            },
        )
    return result


def _run_step(
    name: str,
    action: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        value = action()
        if not isinstance(value, dict):
            raise TypeError(f"{name} must return a JSON object")
        return {
            "name": name,
            "status": str(value.get("status") or "completed"),
            "ok": True,
            "result": value,
        }
    except Exception as exc:  # noqa: BLE001 - one failed derivative must not roll back canonical ASR.
        return {
            "name": name,
            "status": "failed",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _update_manifest(root: Path, values: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest = dict(manifest) if isinstance(manifest, dict) else {}
    manifest.update(values)
    write_json(manifest_path, manifest)


def _next_actions(result: dict[str, Any]) -> list[str]:
    if result["full_pipeline_production_qualified"]:
        return []
    actions: list[str] = []
    if result["failed_steps"]:
        actions.append(
            "Inspect transcript-downstream-refresh.md and repair failed local stages."
        )
    if not result["transcript_quality_passed"]:
        actions.append(
            "Resolve transcript-quality-gate failures before regenerating the summary."
        )
    if result["requires_summary_regeneration"]:
        actions.append(
            "Run the existing consented summary route, install a fresh summary, then rerun refresh-transcript-downstream."
        )
    return actions


def _render_invalidation_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Smart Summary Invalidation",
        "",
        f"- Status: `{result['status']}`",
        f"- Reason: `{result['reason']}`",
        f"- Canonical before: `{result['canonical_before_sha256']}`",
        f"- Canonical after: `{result['canonical_after_sha256']}`",
        f"- Invalidated summaries: {result['invalidated_summary_count']}",
        "",
        "Old summaries are preserved only as audit snapshots and must not pass the production quality gate.",
    ]
    for row in result["invalidated_summaries"]:
        lines.append(
            f"- `{row['relative_path']}` → `{row['archive_relative_path']}` (`{row['sha256']}`)"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_refresh_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Transcript Downstream Refresh",
        "",
        f"- Status: `{result['status']}`",
        f"- Local refresh completed: `{result['local_refresh_completed']}`",
        f"- Transcript quality passed: `{result['transcript_quality_passed']}`",
        f"- Smart Summary quality passed: `{result['smart_summary_quality_passed']}`",
        f"- Full pipeline production qualified: `{result['full_pipeline_production_qualified']}`",
        "",
        "## Steps",
        "",
    ]
    for row in result["steps"]:
        suffix = f" — {row['error']}" if row.get("error") else ""
        lines.append(f"- `{row['name']}`: `{row['status']}`{suffix}")
    if result["next_actions"]:
        lines.extend(["", "## Next actions", ""])
        lines.extend(f"- {action}" for action in result["next_actions"])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Invalidate stale summaries, rebuild local downstream artifacts, "
            "and rerun transcript/summary quality gates"
        )
    )
    parser.add_argument("bundle_dir")
    parser.add_argument("--canonical-before-sha256", default="")
    parser.add_argument("--canonical-after-sha256", default="")
    parser.add_argument("--reason", default="operator_refresh")
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.bundle_dir).expanduser().resolve()
    canonical = root / "source-arbitrated-transcript.json"
    current_hash = sha256_file(canonical) if canonical.is_file() else ""
    result = refresh_transcript_downstream_outputs(
        root,
        canonical_before_sha256=args.canonical_before_sha256 or current_hash,
        canonical_after_sha256=args.canonical_after_sha256 or current_hash,
        reason=args.reason,
        write=args.write,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"completed", "needs_summary_regeneration"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
