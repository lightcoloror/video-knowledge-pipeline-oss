from __future__ import annotations

from pathlib import Path
from typing import Any

from .acceptance_check import acceptance_check
from .asr_runner import plan_asr_run
from .bundle_next import bundle_advance_queue, bundle_next_action
from .bundle_status import bundle_status_report
from .knowledge_coverage import audit_knowledge_coverage
from .knowledge_note_export import export_knowledge_note
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .review_session import prepare_review_session, review_closure_status
from .screen_text_recovery import run_screen_text_recovery
from .storage import read_json, write_json
from .storage import read_json_object_or_empty as _read_json
from .webui_bridge import refresh_bundle_review_html

BATCH_REPAIR_SCHEMA = "video_knowledge_batch_repair_run.v1"


def batch_repair_run(
    batch_manifest_or_summary: str | Path,
    *,
    execute: bool = False,
    limit: int = 0,
    max_rounds: int = 1,
    allow_asr: bool = False,
    allow_vision: bool = False,
    allow_ocr: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Plan or run the next safe repair action across a batch of bundles."""
    input_path = Path(batch_manifest_or_summary).expanduser().resolve()
    payload = read_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError("batch repair input must be a JSON object")
    source = _source_from_payload(input_path, payload)
    workspace = source["workspace"]
    workspace.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    entries = source["items"][: int(limit)] if limit and limit > 0 else source["items"]
    for position, item in enumerate(entries, start=1):
        rows.append(
            _repair_item(
                item,
                position=position,
                execute=execute,
                max_rounds=max_rounds,
                allow_asr=allow_asr,
                allow_vision=allow_vision,
                allow_ocr=allow_ocr,
            )
        )
    human_review = _human_review_summary(rows)
    result = {
        "schema": BATCH_REPAIR_SCHEMA,
        "input_path": str(input_path),
        "input_schema": str(payload.get("schema") or ""),
        "workspace": str(workspace),
        "created_at": now_iso(),
        "mode": {
            "execute": bool(execute),
            "limit": int(limit or 0),
            "max_rounds": max(int(max_rounds or 1), 1),
            "allow_asr": bool(allow_asr),
            "allow_vision": bool(allow_vision),
            "allow_ocr": bool(allow_ocr),
        },
        "summary": _summary(rows, human_review),
        "items": rows,
        "human_review": human_review,
    }
    json_path = workspace / "batch-repair-run.json"
    markdown_path = workspace / "batch-repair-run.md"
    human_path = workspace / "batch-human-review.md"
    result["json_path"] = str(json_path)
    result["report_path"] = str(markdown_path)
    result["human_review_path"] = str(human_path)
    if write:
        write_json(json_path, result)
        markdown_path.write_text(render_batch_repair_markdown(result), encoding="utf-8")
        human_path.write_text(render_batch_human_review_markdown(result), encoding="utf-8")
    return result


def render_batch_repair_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    mode = result.get("mode") if isinstance(result.get("mode"), dict) else {}
    lines = [
        "# Batch Repair Run",
        "",
        f"- Input: `{result.get('input_path', '')}`",
        f"- Workspace: `{result.get('workspace', '')}`",
        f"- Execute: `{mode.get('execute', False)}`",
        f"- Allow ASR: `{mode.get('allow_asr', False)}`",
        f"- Allow vision: `{mode.get('allow_vision', False)}`",
        f"- Allow OCR: `{mode.get('allow_ocr', False)}`",
        f"- Total: `{summary.get('total', 0)}`",
        f"- Advanced: `{summary.get('advanced', 0)}`",
        f"- Planned/blocked: `{summary.get('planned_or_blocked', 0)}`",
        f"- Human review rows: `{summary.get('human_review_count', 0)}`",
        "",
        "## Items",
        "",
        "| # | ID | Status | Action | Kind | Before | After | Human | Bundle |",
        "|---:|---|---|---|---|---|---|---:|---|",
    ]
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "| {position} | `{item_id}` | `{status}` | `{action}` | `{kind}` | `{before}` | `{after}` | {human} | `{bundle}` |".format(
                position=item.get("position", ""),
                item_id=_md_cell(str(item.get("id") or "")),
                status=_md_cell(str(item.get("status") or "")),
                action=_md_cell(str(item.get("action") or "")),
                kind=_md_cell(str(item.get("action_kind") or "")),
                before=_md_cell(str(item.get("before_key") or "")),
                after=_md_cell(str(item.get("after_key") or "")),
                human=len(item.get("human_review_rows") or []),
                bundle=_md_cell(str(item.get("bundle_dir") or "")),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Human review: `{result.get('human_review_path', '')}`",
            "- Per-bundle review packs and closure reports are listed in `reports.review_pack` and `reports.review_closure_status`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_batch_human_review_markdown(result: dict[str, Any]) -> str:
    rows = (result.get("human_review") or {}).get("items") if isinstance(result.get("human_review"), dict) else []
    lines = [
        "# Batch Human Review",
        "",
        f"- Batch repair: `{result.get('report_path', '')}`",
        f"- Total review rows: `{len(rows or [])}`",
        "",
    ]
    if not rows:
        lines.append("当前批次没有需要集中人工复核的条目。")
        return "\n".join(lines).rstrip() + "\n"
    lines.extend(["| Bundle | Index/Scope | Reason | Suggested action | Evidence |", "|---|---|---|---|---|"])
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{bundle}` | `{scope}` | {reason} | {action} | {evidence} |".format(
                bundle=_md_cell(str(row.get("bundle_id") or row.get("bundle_dir") or "")),
                scope=_md_cell(str(row.get("scope") or "")),
                reason=_md_cell(str(row.get("reason") or "")),
                action=_md_cell(str(row.get("suggested_action") or "")),
                evidence=_md_cell(str(row.get("evidence") or "")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _repair_item(
    item: dict[str, Any],
    *,
    position: int,
    execute: bool,
    max_rounds: int,
    allow_asr: bool,
    allow_vision: bool,
    allow_ocr: bool,
) -> dict[str, Any]:
    bundle_dir = Path(str(item.get("bundle_dir") or "")).expanduser() if str(item.get("bundle_dir") or "").strip() else None
    row: dict[str, Any] = {
        "position": position,
        "id": str(item.get("id") or item.get("title") or f"item-{position:03d}"),
        "title": str(item.get("title") or item.get("id") or ""),
        "priority": str(item.get("priority") or ""),
        "bundle_dir": str(bundle_dir.resolve()) if bundle_dir else "",
        "status": "unknown",
        "action": "",
        "action_kind": "",
        "before_key": "",
        "after_key": "",
        "reports": {},
        "human_review_rows": [],
    }
    if not bundle_dir or not bundle_dir.exists():
        row.update({"status": "error", "action": "missing_bundle", "error": "bundle_dir is missing or does not exist"})
        return row
    before = _next(bundle_dir, refresh=True)
    action = before.get("next_action") if isinstance(before.get("next_action"), dict) else {}
    key = str(action.get("key") or "")
    kind = _action_kind(action)
    row.update({"action_kind": kind, "before_key": key, "before": _compact_next(before)})
    if _should_retry_weak_screen_text(item, execute=execute, allow_ocr=allow_ocr):
        action = {
            "key": "screen_text",
            "label": "补跑屏幕文字裁剪/OCR",
            "reason": f"batch summary screen_text_status={item.get('screen_text_status')}",
            "mcp_tool": "run_screen_text_recovery",
        }
        key = "screen_text"
        kind = "ocr"
        row.update({"action_kind": kind, "before_key": key})
    elif _is_done(before, action):
        _refresh_bundle(bundle_dir)
        row.update({"status": "skipped_completed", "action": "skip", "after_key": key, "reports": _reports(bundle_dir)})
        return row
    if kind == "asr":
        row.update(_handle_asr(bundle_dir, item, action, execute=execute, allow_asr=allow_asr))
    elif kind == "ocr":
        row.update(_handle_ocr(bundle_dir, action, execute=execute, allow_ocr=allow_ocr))
    elif kind == "vision":
        row.update(_handle_vision(bundle_dir, action, execute=execute, allow_vision=allow_vision, max_rounds=max_rounds))
    elif kind == "export":
        row.update(_handle_export(bundle_dir, execute=execute))
    elif kind == "human_review":
        row.update(_handle_human_review(bundle_dir, action, execute=execute))
    else:
        row.update({"status": "planned", "action": "manual_or_unknown", "command": action.get("command", "")})
    if execute and row.get("status") not in {"blocked_not_allowed", "planned"}:
        _refresh_bundle(bundle_dir)
    after = _next(bundle_dir, refresh=True)
    after_action = after.get("next_action") if isinstance(after.get("next_action"), dict) else {}
    row["after_key"] = str(after_action.get("key") or "")
    row["after"] = _compact_next(after)
    row["reports"] = _reports(bundle_dir)
    row["human_review_rows"] = _review_rows_for_bundle(row, action, after_action)
    return row


def _handle_asr(bundle_dir: Path, item: dict[str, Any], action: dict[str, Any], *, execute: bool, allow_asr: bool) -> dict[str, Any]:
    media_path = str(item.get("media_path") or _source_media_path(bundle_dir) or "").strip()
    if not allow_asr:
        return {"status": "blocked_not_allowed", "action": "plan_asr", "command": _asr_hint(bundle_dir, media_path)}
    if not execute:
        return {"status": "planned", "action": "plan_asr", "command": _asr_hint(bundle_dir, media_path)}
    if not media_path or not Path(media_path).exists():
        return {"status": "human_review_required", "action": "asr_media_missing", "error": "source media not found for ASR planning"}
    plan = plan_asr_run(bundle_dir.parent, media_path, preset="sensevoice", model="iic/SenseVoiceSmall")
    return {"status": "advanced", "action": "asr_plan_created", "action_result": _compact_result(plan), "command": plan.get("powershell", "")}


def _handle_ocr(bundle_dir: Path, action: dict[str, Any], *, execute: bool, allow_ocr: bool) -> dict[str, Any]:
    if not allow_ocr:
        return {"status": "blocked_not_allowed", "action": "screen_text_recovery", "command": action.get("command", "")}
    result = run_screen_text_recovery(bundle_dir, execute_crops=execute, execute_ocr=execute, limit=1 if execute else 0)
    ocr_summary = result.get("ocr_summary") if isinstance(result.get("ocr_summary"), dict) else {}
    status = "advanced" if execute else "planned"
    if execute and int(ocr_summary.get("updated") or 0) == 0:
        status = "human_review_required"
    return {"status": status, "action": "screen_text_recovery", "action_result": _compact_result(result), "command": result.get("mcp_args_path", "")}


def _handle_vision(bundle_dir: Path, action: dict[str, Any], *, execute: bool, allow_vision: bool, max_rounds: int) -> dict[str, Any]:
    if not allow_vision:
        return {"status": "blocked_not_allowed", "action": "vision_repair", "command": action.get("command", "")}
    if not execute:
        return {"status": "planned", "action": "vision_repair", "command": action.get("command", "")}
    result = bundle_advance_queue(bundle_dir, max_steps=max(int(max_rounds or 1), 1), execute=True, refresh_outputs=True)
    return {
        "status": "advanced" if result.get("status") == "advanced" else "blocked",
        "action": "bundle_advance_queue",
        "action_result": _compact_result(result),
        "command": result.get("mcp_args_path", ""),
    }


def _handle_export(bundle_dir: Path, *, execute: bool) -> dict[str, Any]:
    if not execute:
        return {"status": "planned", "action": "refresh_exports", "command": f"export_knowledge_note {bundle_dir}"}
    reports = _refresh_bundle(bundle_dir)
    return {"status": "advanced", "action": "refresh_exports", "action_result": reports}


def _handle_human_review(bundle_dir: Path, action: dict[str, Any], *, execute: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    closure: dict[str, Any] = {}
    if execute:
        result = prepare_review_session(bundle_dir, refresh=True, limit=30)
        closure = review_closure_status(bundle_dir, write=True)
    return {
        "status": "human_review_required",
        "action": "prepare_review_session" if execute else "human_review",
        "action_result": _compact_result(result),
        "closure_status": _compact_result(closure),
        "command": action.get("command", ""),
    }


def _refresh_bundle(bundle_dir: Path) -> dict[str, str]:
    coverage = audit_knowledge_coverage(bundle_dir, write=True)
    export = export_knowledge_note(bundle_dir, write=True)
    acceptance = acceptance_check(bundle_dir, refresh=True, write=True)
    status = bundle_status_report(bundle_dir, refresh=True)
    try:
        review = refresh_bundle_review_html(bundle_dir, write=True)
    except Exception:
        review = {}
    return {
        "knowledge_coverage": str(coverage.get("coverage_markdown_path", "")),
        "acceptance_check": str(acceptance.get("report_markdown_path") or acceptance.get("report_path") or ""),
        "bundle_status": str(status.get("report_markdown_path") or status.get("report_path") or ""),
        "knowledge_note": str(export.get("note_path", "")),
        "review_html": str(review.get("review_html_path", "")) if isinstance(review, dict) else "",
    }


def _source_from_payload(input_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    schema = str(payload.get("schema") or "")
    workspace = Path(str(payload.get("workspace") or input_path.parent)).expanduser().resolve()
    if schema == "video_knowledge_batch_acceptance_summary.v1":
        return {"workspace": workspace, "items": [item for item in payload.get("items") or [] if isinstance(item, dict)]}
    if schema == "video_knowledge_batch_run.v1":
        return {"workspace": workspace, "items": [item for item in payload.get("items") or [] if isinstance(item, dict)]}
    if schema == "video_knowledge_batch.v1":
        items = []
        for position, item in enumerate(payload.get("items") or [], start=1):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or f"item-{position:03d}")
            bundle_dir = str(item.get("bundle_dir") or "")
            if not bundle_dir:
                bundle_dir = str((workspace / item_id / "webui-bundle").resolve())
            items.append({**item, "id": item_id, "bundle_dir": bundle_dir})
        return {"workspace": workspace, "items": items}
    raise ValueError(f"unsupported batch repair input schema: {schema}")


def _next(bundle_dir: Path, *, refresh: bool) -> dict[str, Any]:
    try:
        return bundle_next_action(bundle_dir, refresh=refresh)
    except Exception as exc:
        return {"status": "error", "next_action": {"key": "error", "label": str(exc), "human_required": True}}


def _action_kind(action: dict[str, Any]) -> str:
    key = str(action.get("key") or "")
    tool = str(action.get("mcp_tool") or "")
    if key in {"none", "done", ""} and not tool:
        return "done"
    if key == "speech" or tool in {"normalize_asr_output", "run_asr_plan"}:
        return "asr"
    if key in {"screen_text", "ocr_text_empty_review"} or tool in {"run_screen_text_recovery", "run_ocr_backfill", "run_visual_structure_plan"}:
        return "ocr"
    if key in {"semantic_frame_understanding", "temporal_visual_understanding", "temporal_frame_groups", "provider_matrix_repair"} or tool in {
        "run_multimodal_frame_analysis",
        "run_temporal_visual_analysis",
        "run_temporal_frame_groups",
        "vision_provider_matrix",
        "vision_provider_smoke",
    }:
        return "vision"
    if bool(action.get("human_required")) or str(action.get("status") or "") in {"human_review_required", "review_blocked"}:
        return "human_review"
    if str(action.get("status") or "") == "ready":
        return "export"
    return "other"


def _is_done(next_result: dict[str, Any], action: dict[str, Any]) -> bool:
    status = str(next_result.get("status") or "")
    key = str(action.get("key") or "")
    return status in {"accepted", "accepted_with_known_gaps"} or key in {"none", "done"}


def _should_retry_weak_screen_text(item: dict[str, Any], *, execute: bool, allow_ocr: bool) -> bool:
    if not execute or not allow_ocr:
        return False
    status = str(item.get("screen_text_status") or "").lower()
    next_key = str(item.get("next_action_key") or "").lower()
    return status in {"weak", "blocked"} and next_key in {"", "none", "done", "screen_text", "ocr_text_empty_review"}


def _review_rows_for_bundle(row: dict[str, Any], before_action: dict[str, Any], after_action: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    status = str(row.get("status") or "")
    if status in {"human_review_required", "blocked", "blocked_not_allowed"} or bool(after_action.get("human_required")):
        rows.extend(_review_pack_rows(row))
    if not rows and (status in {"human_review_required", "blocked", "blocked_not_allowed"} or bool(after_action.get("human_required"))):
        rows.append(
            {
                "bundle_id": row.get("id", ""),
                "bundle_dir": row.get("bundle_dir", ""),
                "scope": str(after_action.get("key") or before_action.get("key") or row.get("action") or ""),
                "reason": str(after_action.get("reason") or before_action.get("reason") or status),
                "suggested_action": str(after_action.get("label") or before_action.get("label") or row.get("action") or ""),
                "evidence": str(after_action.get("mcp_args_path") or before_action.get("mcp_args_path") or ""),
            }
        )
    return rows


def _review_pack_rows(row: dict[str, Any], *, limit: int = 50) -> list[dict[str, Any]]:
    bundle_dir = Path(str(row.get("bundle_dir") or "")).expanduser()
    pack_path = bundle_dir / "review-pack.json"
    closure_path = bundle_dir / "review-closure-status.md"
    if not pack_path.exists():
        return []
    pack = _read_json(pack_path)
    rows: list[dict[str, Any]] = []
    for group in pack.get("groups") or []:
        if not isinstance(group, dict):
            continue
        for item in group.get("items") or []:
            if not isinstance(item, dict):
                continue
            evidence_paths = item.get("evidence_paths") or item.get("crop_paths") or item.get("asset_paths") or []
            rows.append(
                {
                    "bundle_id": row.get("id", ""),
                    "bundle_dir": row.get("bundle_dir", ""),
                    "scope": f"timeline:{item.get('index', '')}",
                    "reason": ", ".join(str(value) for value in item.get("reasons") or []) or str(group.get("key") or ""),
                    "suggested_action": str(item.get("suggested_action") or item.get("suggested_status") or group.get("label") or ""),
                    "evidence": ", ".join(str(path) for path in evidence_paths[:2]) or str(pack_path),
                    "review_pack": str(pack_path),
                    "review_closure_status": str(closure_path),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _human_review_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    review_rows = []
    for row in rows:
        review_rows.extend(item for item in row.get("human_review_rows") or [] if isinstance(item, dict))
    return {"count": len(review_rows), "items": review_rows}


def _summary(rows: list[dict[str, Any]], human_review: dict[str, Any]) -> dict[str, int]:
    return {
        "total": len(rows),
        "advanced": sum(1 for row in rows if str(row.get("status") or "") == "advanced"),
        "skipped": sum(1 for row in rows if str(row.get("status") or "").startswith("skipped")),
        "planned_or_blocked": sum(1 for row in rows if str(row.get("status") or "") in {"planned", "blocked", "blocked_not_allowed", "human_review_required"}),
        "errors": sum(1 for row in rows if str(row.get("status") or "") == "error"),
        "human_review_count": int(human_review.get("count") or 0),
    }


def _source_media_path(bundle_dir: Path) -> str:
    manifest = _read_json(bundle_dir / "manifest.json")
    for source in manifest.get("sources") or manifest.get("source_records") or []:
        if isinstance(source, dict) and str(source.get("path") or "").strip():
            return str(source.get("path"))
    return ""


def _asr_hint(bundle_dir: Path, media_path: str) -> str:
    media = media_path or "<media_path>"
    return f".\\scripts\\video-knowledge.ps1 plan-asr {bundle_dir.parent} {media} --preset sensevoice --model iic/SenseVoiceSmall"


def _reports(bundle_dir: Path) -> dict[str, str]:
    return {
        "knowledge_coverage": str(bundle_dir / "knowledge-coverage.md"),
        "acceptance_check": str(bundle_dir / "acceptance-check.md"),
        "bundle_status": str(bundle_dir / "bundle-status.md"),
        "knowledge_note": str(bundle_dir / "exports" / "knowledge-note.md"),
        "review_html": str(bundle_dir / "review.html"),
        "review_pack": str(bundle_dir / "review-pack.md"),
        "review_closure_status": str(bundle_dir / "review-closure-status.md"),
    }


def _compact_next(value: dict[str, Any]) -> dict[str, Any]:
    action = value.get("next_action") if isinstance(value.get("next_action"), dict) else {}
    return {
        "status": value.get("status", ""),
        "key": action.get("key", ""),
        "mcp_tool": action.get("mcp_tool", ""),
        "human_required": bool(action.get("human_required", False)),
    }


def _compact_result(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = {}
    for key in (
        "schema",
        "status",
        "summary",
        "report_path",
        "report_markdown_path",
        "json_path",
        "mcp_args_path",
        "human_review_path",
        "review_pack_path",
        "review_pack_json_path",
        "review_closure_status_path",
        "plan_path",
        "powershell",
    ):
        if key in value:
            keep[key] = value[key]
    return keep
