from __future__ import annotations

from pathlib import Path
from typing import Any

from .acceptance_check import acceptance_check
from .acceptance_run import run_acceptance_bundle, run_acceptance_run
from .bundle_next import bundle_next_action
from .bundle_status import bundle_status_report
from .knowledge_coverage import audit_knowledge_coverage
from .knowledge_note_export import export_knowledge_note
from .models import now_iso
from .storage import read_json, write_json
from .storage import read_json_object_or_empty as _read_bundle_json


BATCH_SCHEMA = "video_knowledge_batch_run.v1"
MANIFEST_SCHEMA = "video_knowledge_batch.v1"
COMPLETED_STATUSES = {"accepted", "accepted_with_known_gaps", "ready"}


def batch_video_knowledge_run(
    batch_manifest: str | Path,
    *,
    resume: bool = False,
    force_reexport: bool = False,
    execute_asr: bool = False,
    execute_temporal_groups: bool = False,
    execute_vision: bool = False,
    execute_ebook_pipeline: bool = False,
    semantic_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    timeout_seconds: int = 1800,
    write: bool = True,
) -> dict[str, Any]:
    """Run a preview-safe batch over local knowledge-video items.

    By default this only creates/reuses workspaces and runs the existing
    acceptance preview chain. Cloud vision, ebook parsing, and ASR execution stay
    off unless the caller explicitly enables those branches.
    """
    manifest_path = Path(batch_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("batch manifest must be a JSON object")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("batch manifest must contain an items array")
    workspace = Path(str(manifest.get("workspace") or manifest_path.parent / "batch-workspace")).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(items, start=1):
        rows.append(
            _run_item(
                item if isinstance(item, dict) else {},
                workspace=workspace,
                position=position,
                resume=resume,
                force_reexport=force_reexport,
                execute_asr=execute_asr,
                execute_temporal_groups=execute_temporal_groups,
                execute_vision=execute_vision,
                execute_ebook_pipeline=execute_ebook_pipeline,
                semantic_limit=semantic_limit,
                temporal_limit=temporal_limit,
                frame_count=frame_count,
                timeout_seconds=timeout_seconds,
            )
        )
    result = {
        "schema": BATCH_SCHEMA,
        "manifest_schema": str(manifest.get("schema") or ""),
        "manifest_path": str(manifest_path),
        "workspace": str(workspace),
        "started_at": started_at,
        "finished_at": now_iso(),
        "mode": {
            "resume": bool(resume),
            "force_reexport": bool(force_reexport),
            "execute_asr": bool(execute_asr),
            "execute_temporal_groups": bool(execute_temporal_groups),
            "execute_vision": bool(execute_vision),
            "execute_ebook_pipeline": bool(execute_ebook_pipeline),
        },
        "summary": _summary(rows),
        "items": rows,
    }
    acceptance_summary = build_batch_acceptance_summary(result)
    json_path = workspace / "batch-run.json"
    markdown_path = workspace / "batch-run.md"
    acceptance_json_path = workspace / "batch-acceptance-summary.json"
    acceptance_markdown_path = workspace / "batch-acceptance-summary.md"
    result["json_path"] = str(json_path)
    result["report_path"] = str(markdown_path)
    result["acceptance_summary_path"] = str(acceptance_json_path)
    result["acceptance_summary_markdown_path"] = str(acceptance_markdown_path)
    result["acceptance_summary"] = acceptance_summary
    if write:
        write_json(json_path, result)
        markdown_path.write_text(render_batch_run_markdown(result), encoding="utf-8")
        write_json(acceptance_json_path, acceptance_summary)
        acceptance_markdown_path.write_text(render_batch_acceptance_summary_markdown(acceptance_summary), encoding="utf-8")
    return result


def render_batch_run_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Video Knowledge Batch Run",
        "",
        f"- Manifest: `{result.get('manifest_path', '')}`",
        f"- Workspace: `{result.get('workspace', '')}`",
        f"- Total: `{summary.get('total', 0)}`",
        f"- Completed/skipped: `{summary.get('completed_or_skipped', 0)}`",
        f"- Needs action: `{summary.get('needs_action', 0)}`",
        f"- Errors: `{summary.get('errors', 0)}`",
        "",
        "## Mode",
        "",
    ]
    mode = result.get("mode") if isinstance(result.get("mode"), dict) else {}
    for key in ("resume", "force_reexport", "execute_asr", "execute_temporal_groups", "execute_vision", "execute_ebook_pipeline"):
        lines.append(f"- `{key}`: `{mode.get(key, False)}`")
    lines.extend(
        [
            "",
            "## Items",
            "",
            "| # | ID | Status | Action | Bundle | Next |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        next_action = item.get("next_action") if isinstance(item.get("next_action"), dict) else {}
        lines.append(
            "| {position} | `{item_id}` | `{status}` | `{action}` | `{bundle}` | `{next_key}` |".format(
                position=item.get("position", ""),
                item_id=_md_cell(str(item.get("id") or "")),
                status=_md_cell(str(item.get("status") or "")),
                action=_md_cell(str(item.get("action") or "")),
                bundle=_md_cell(str(item.get("bundle_dir") or "")),
                next_key=_md_cell(str(next_action.get("key") or "")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _run_item(
    item: dict[str, Any],
    *,
    workspace: Path,
    position: int,
    resume: bool,
    force_reexport: bool,
    execute_asr: bool,
    execute_temporal_groups: bool,
    execute_vision: bool,
    execute_ebook_pipeline: bool,
    semantic_limit: int | None,
    temporal_limit: int | None,
    frame_count: int | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    item_id = str(item.get("id") or f"item-{position:03d}").strip()
    title = str(item.get("title") or item_id).strip()
    item_workspace = Path(str(item.get("workspace") or workspace / item_id)).expanduser().resolve()
    item_workspace.mkdir(parents=True, exist_ok=True)
    media_path = str(item.get("media_path") or item.get("source") or "").strip()
    bundle_dir = _bundle_dir(item, item_workspace)
    row: dict[str, Any] = {
        "position": position,
        "id": item_id,
        "title": title,
        "media_path": media_path,
        "expected_content_type": str(item.get("expected_content_type") or ""),
        "priority": str(item.get("priority") or ""),
        "notes": str(item.get("notes") or ""),
        "workspace_dir": str(item_workspace),
        "bundle_dir": str(bundle_dir) if bundle_dir else "",
        "status": "unknown",
        "action": "",
        "next_action": {},
        "reports": {},
    }
    try:
        completed = _completed_status(bundle_dir)
        if completed and not force_reexport:
            is_complete = completed.get("completed") is True
            row.update(
                {
                    "status": "skipped_completed" if is_complete else "needs_resume",
                    "action": "skip" if is_complete else "inspect_incomplete_completion",
                    "completion_status": completed.get("status", ""),
                    "completion": completed.get("completion", {}),
                    "next_action": completed.get("next_action", {}),
                    "reports": _bundle_reports(bundle_dir),
                }
            )
            return row
        if bundle_dir and bundle_dir.exists() and force_reexport:
            row.update(_force_reexport(bundle_dir))
            return row
        if bundle_dir and bundle_dir.exists() and resume:
            run = run_acceptance_bundle(
                bundle_dir,
                output_dir=item_workspace,
                title=title,
                execute_temporal_groups=execute_temporal_groups,
                execute_vision=execute_vision,
                execute_ebook_pipeline=execute_ebook_pipeline,
                semantic_limit=semantic_limit,
                temporal_limit=temporal_limit,
                frame_count=frame_count,
            )
            row.update(_run_summary("resumed", "acceptance_bundle_run", run))
            return row
        if bundle_dir and bundle_dir.exists():
            next_result = bundle_next_action(bundle_dir, refresh=True)
            row.update(
                {
                    "status": "needs_resume",
                    "action": "inspect_existing_bundle",
                    "next_action": next_result.get("next_action", {}),
                    "reports": _bundle_reports(bundle_dir),
                }
            )
            return row
        if not media_path:
            row.update({"status": "error", "action": "missing_media_path", "error": "media_path is required when bundle does not exist"})
            return row
        run = run_acceptance_run(
            media_path,
            item_workspace,
            title=title,
            execute_asr=execute_asr,
            execute_temporal_groups=execute_temporal_groups,
            execute_vision=execute_vision,
            execute_ebook_pipeline=execute_ebook_pipeline,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            timeout_seconds=timeout_seconds,
        )
        row["bundle_dir"] = str(run.get("bundle_dir") or row.get("bundle_dir") or "")
        row.update(_run_summary("created", "acceptance_run", run))
        return row
    except Exception as exc:  # pragma: no cover - exercised through public report.
        row.update({"status": "error", "action": "exception", "error": f"{type(exc).__name__}: {exc}"})
        return row


def _force_reexport(bundle_dir: Path) -> dict[str, Any]:
    coverage = audit_knowledge_coverage(bundle_dir, write=True)
    export = export_knowledge_note(bundle_dir, write=True)
    status = bundle_status_report(bundle_dir, refresh=True)
    acceptance = acceptance_check(bundle_dir, refresh=True, write=True)
    next_result = bundle_next_action(bundle_dir, refresh=False)
    from .workflow_completion import workflow_completion_status
    completion = workflow_completion_status(bundle_dir)
    return {
        "status": "reexported",
        "completion": completion,
        "completion_status": completion["status"],
        "action": "force_reexport",
        "next_action": next_result.get("next_action", {}),
        "reports": {
            "knowledge_coverage": coverage.get("coverage_markdown_path", ""),
            "knowledge_note": export.get("note_path", ""),
            "bundle_status": status.get("report_path", ""),
            "acceptance_check": acceptance.get("report_path", ""),
        },
    }


def _completed_status(bundle_dir: Path | None) -> dict[str, Any]:
    if not bundle_dir or not bundle_dir.exists() or not (bundle_dir / "manifest.json").exists():
        return {}
    from .workflow_completion import workflow_completion_status
    completion = workflow_completion_status(bundle_dir)
    acceptance_path = bundle_dir / "acceptance-check.json"
    acceptance = read_json(acceptance_path) if acceptance_path.exists() else {}
    if isinstance(acceptance, dict) and str(acceptance.get("status") or "") in COMPLETED_STATUSES:
        if not completion["completed"]:
            # Preserve prior reports and media work. A cached acceptance label
            # lacking stage evidence is a closeout task, not a full rerun.
            return {"status": completion["status"], "completed": False, "completion": completion,
                    "next_action": {"key": "complete_missing_stages", "actions": completion["next_actions"]}}
        return {"status": str(acceptance.get("status")), "completed": True,
                "completion": completion, "next_action": acceptance.get("next_action", {})}
    if not completion["completed"]:
        return {}
    next_result = bundle_next_action(bundle_dir, refresh=True)
    action = next_result.get("next_action") if isinstance(next_result.get("next_action"), dict) else {}
    if str(next_result.get("status") or action.get("status") or "") in COMPLETED_STATUSES:
        return {"status": str(next_result.get("status") or action.get("status")),
                "completed": True, "completion": completion, "next_action": action}
    return {}


def _run_summary(status: str, action: str, run: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = str(run.get("bundle_dir") or "")
    next_action: dict[str, Any] = {}
    if bundle_dir:
        try:
            next_action = bundle_next_action(bundle_dir, refresh=True).get("next_action", {})
        except Exception:
            next_action = {}
    from .workflow_completion import workflow_completion_status
    completion = workflow_completion_status(bundle_dir) if bundle_dir else {"completed": False, "status": "blocked"}
    return {
        "status": status,
        "completion": completion,
        "completion_status": completion["status"],
        "action": action,
        "bundle_dir": bundle_dir,
        "next_action": next_action,
        "reports": {
            "acceptance_run": run.get("report_path", ""),
            "acceptance_run_json": run.get("json_path", ""),
            **(_bundle_reports(Path(bundle_dir)) if bundle_dir else {}),
        },
        "run_summary": run.get("summary", {}),
    }


def _bundle_dir(item: dict[str, Any], item_workspace: Path) -> Path | None:
    explicit = str(item.get("bundle_dir") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return item_workspace / "webui-bundle"


def _bundle_reports(bundle_dir: Path | None) -> dict[str, str]:
    if not bundle_dir:
        return {}
    return {
        "bundle_status": str(bundle_dir / "bundle-status.md"),
        "knowledge_coverage": str(bundle_dir / "knowledge-coverage.md"),
        "knowledge_note": str(bundle_dir / "exports" / "knowledge-note.md"),
        "acceptance_check": str(bundle_dir / "acceptance-check.md"),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "completed_or_skipped": sum(1 for row in rows if (row.get("completion") or {}).get("completed") is True),
        "needs_action": sum(1 for row in rows if row.get("status") != "error" and (row.get("completion") or {}).get("completed") is not True),
        "errors": sum(1 for row in rows if str(row.get("status") or "") == "error"),
    }


def build_batch_acceptance_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        rows.append(_acceptance_summary_row(item))
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("acceptance_status") or row.get("batch_status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "schema": "video_knowledge_batch_acceptance_summary.v1",
        "batch_manifest": result.get("manifest_path", ""),
        "workspace": result.get("workspace", ""),
        "created_at": result.get("finished_at") or now_iso(),
        "total": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "needs_action": sum(1 for row in rows if bool(row.get("needs_action"))),
        "review_pending_total": sum(int(row.get("review_pending_count") or 0) for row in rows),
        "screen_text_weak_or_blocked": sum(1 for row in rows if str(row.get("screen_text_status") or "") in {"weak", "blocked"}),
        "items": rows,
    }


def render_batch_acceptance_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Batch Acceptance Summary",
        "",
        f"- Workspace: `{summary.get('workspace', '')}`",
        f"- Manifest: `{summary.get('batch_manifest', '')}`",
        f"- Total: `{summary.get('total', 0)}`",
        f"- Needs action: `{summary.get('needs_action', 0)}`",
        f"- Pending review rows: `{summary.get('review_pending_total', 0)}`",
        f"- Screen text weak/blocked videos: `{summary.get('screen_text_weak_or_blocked', 0)}`",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in (summary.get("status_counts") or {}).items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "## Items",
            "",
            "| # | ID | Priority | Content | Historical acceptance | Workflow completion | Screen text | Semantic gap | Temporal gap | Review pending | Export | Next | Bundle |",
            "|---:|---|---|---|---|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for item in summary.get("items") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "| {position} | `{item_id}` | {priority} | {content_type} | `{acceptance}` | `{completion}` | `{screen}` | {semantic} | {temporal} | {review} | `{export}` | `{next_key}` | `{bundle}` |".format(
                position=item.get("position", ""),
                item_id=_md_cell(str(item.get("id") or "")),
                priority=_md_cell(str(item.get("priority") or "")),
                content_type=_md_cell(str(item.get("expected_content_type") or "")),
                acceptance=_md_cell(str(item.get("acceptance_status") or "")),
                completion=_md_cell(str(item.get("completion_status") or "unverified")),
                screen=_md_cell(str(item.get("screen_text_status") or "")),
                semantic=item.get("semantic_missing", 0),
                temporal=item.get("temporal_missing", 0),
                review=item.get("review_pending_count", 0),
                export=_md_cell(str(item.get("export_freshness") or "")),
                next_key=_md_cell(str(item.get("next_action_key") or "")),
                bundle=_md_cell(str(item.get("bundle_dir") or "")),
            )
        )
    lines.extend(["", "## Notes", ""])
    note_count = 0
    for item in summary.get("items") or []:
        if not isinstance(item, dict) or not str(item.get("notes") or "").strip():
            continue
        note_count += 1
        lines.append(f"- `{item.get('id', '')}`: {_md_cell(str(item.get('notes') or ''))}")
    if note_count == 0:
        lines.append("- 无")
    return "\n".join(lines).rstrip() + "\n"


def _acceptance_summary_row(item: dict[str, Any]) -> dict[str, Any]:
    bundle_text = str(item.get("bundle_dir") or "").strip()
    bundle_dir = Path(bundle_text).expanduser() if bundle_text else None
    acceptance = _read_bundle_json(bundle_dir / "acceptance-check.json") if bundle_dir else {}
    coverage = _read_bundle_json(bundle_dir / "knowledge-coverage.json") if bundle_dir else {}
    manifest = _read_bundle_json(bundle_dir / "manifest.json") if bundle_dir else {}
    acceptance_summary = acceptance.get("summary") if isinstance(acceptance.get("summary"), dict) else {}
    next_action = acceptance.get("next_action") if isinstance(acceptance.get("next_action"), dict) else item.get("next_action", {})
    coverage_channels = coverage.get("channels") if isinstance(coverage.get("channels"), list) else []
    screen_channel = next((channel for channel in coverage_channels if isinstance(channel, dict) and channel.get("key") == "screen_text"), {})
    review_lifecycle = acceptance.get("review_lifecycle") if isinstance(acceptance.get("review_lifecycle"), dict) else {}
    open_review = review_lifecycle.get("open_review_target_count")
    if open_review is None:
        open_review = _review_pending_from_manifest(manifest)
    export_freshness = acceptance_summary.get("export_freshness") or _export_freshness_from_manifest(manifest)
    semantic_missing = int(acceptance_summary.get("semantic_missing") or coverage.get("semantic_frame_without_analysis") or 0)
    temporal_missing = int(acceptance_summary.get("temporal_missing") or coverage.get("temporal_sequence_without_analysis") or 0)
    screen_status = str(screen_channel.get("status") or acceptance_summary.get("screen_text") or "")
    acceptance_status = str(acceptance.get("status") or item.get("completion_status") or item.get("status") or "unknown")
    completion = item.get("completion") or {}
    if completion.get("completed") is not True and item.get("next_action"):
        next_action = item["next_action"]
    next_key = str((next_action or {}).get("key") or "")
    return {
        "position": item.get("position", 0),
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "priority": item.get("priority", ""),
        "expected_content_type": item.get("expected_content_type", ""),
        "notes": item.get("notes", ""),
        "batch_status": item.get("status", ""),
        "acceptance_status": acceptance_status,
        "completion_status": completion.get("status", "unverified"),
        "completion": completion,
        "bundle_dir": str(bundle_dir) if bundle_dir else "",
        "screen_text_status": screen_status or "unknown",
        "screen_text_covered": screen_channel.get("covered_count", 0) if isinstance(screen_channel, dict) else 0,
        "screen_text_blockers": screen_channel.get("blocker_count", 0) if isinstance(screen_channel, dict) else 0,
        "semantic_missing": semantic_missing,
        "temporal_missing": temporal_missing,
        "review_pending_count": int(open_review or 0),
        "export_freshness": str(export_freshness or "unknown"),
        "next_action_key": next_key,
        "needs_action": completion.get("completed") is not True or acceptance_status not in COMPLETED_STATUSES or next_key not in {"", "done"},
        "reports": item.get("reports", {}),
    }



def _review_pending_from_manifest(manifest: dict[str, Any]) -> int:
    readiness = manifest.get("review_readiness") if isinstance(manifest.get("review_readiness"), dict) else {}
    counts = readiness.get("counts") if isinstance(readiness.get("counts"), dict) else {}
    for key in ("open_review_targets", "needs_review", "unreviewed_items"):
        if key in counts:
            return int(counts.get(key) or 0)
    return 0


def _export_freshness_from_manifest(manifest: dict[str, Any]) -> str:
    export = manifest.get("knowledge_note_export") if isinstance(manifest.get("knowledge_note_export"), dict) else {}
    return "present" if export.get("exported_at") else "missing"


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
