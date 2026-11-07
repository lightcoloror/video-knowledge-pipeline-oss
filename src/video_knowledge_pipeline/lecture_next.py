from __future__ import annotations

from pathlib import Path
from typing import Any

from .lecture_package import build_lecture_package
from .lecture_pipeline import run_ready_lecture_pipeline
from .lecture_status import lecture_project_health
from .lecture_workflow import refresh_lecture_review_outputs
from .models import now_iso
from .storage import append_jsonl, ensure_project_dirs, read_jsonl
from .webui_bridge import export_webui_bundle


def lecture_next_step(
    root: str | Path,
    *,
    plan_json: str | Path | None = None,
    transcript: str | Path | None = None,
    webui_output_dir: str | Path | None = None,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    target: str = "bilinote",
    title: str | None = None,
    merge_window: float = 1.0,
    force_reimport: bool = False,
    bilinote_root: str | Path | None = None,
    bilinote_patch_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the next safe local glue step for a lecture project."""
    before = lecture_project_health(
        root,
        plan_json=plan_json,
        transcript=transcript,
        webui_output_dir=webui_output_dir,
        bilinote_root=bilinote_root,
        bilinote_patch_root=bilinote_patch_root,
    )
    selected = _select_safe_action(before)
    result: dict[str, Any] = {
        "project": str(root),
        "dry_run": dry_run,
        "selected_action": selected,
        "before": before,
    }
    if selected["mode"] != "execute" or dry_run:
        result["executed"] = False
        result["after"] = before
        return _write_action_log(root, result)

    action = selected["action"]
    if action == "run_ready_pipeline":
        if not plan_json:
            raise ValueError("plan_json is required for run_ready_pipeline")
        result["operation"] = run_ready_lecture_pipeline(
            plan_json,
            transcript=transcript,
            webui_output_dir=webui_output_dir,
            vault=vault,
            folder=folder,
            merge_window=merge_window,
            target=target,
            force_reimport=force_reimport,
        )
    elif action == "build_package":
        result["operation"] = build_lecture_package(root, title=title, merge_window=merge_window)
    elif action == "export_webui":
        output_dir = webui_output_dir or _planned_webui_output_dir(before)
        result["operation"] = export_webui_bundle(root, output_dir=output_dir, target=target)
    elif action == "refresh_review":
        review_json = _review_notes_path(before)
        if not review_json:
            raise ValueError("review-notes.json not found in lecture project health")
        output_dir = webui_output_dir or str(Path(review_json).parent)
        result["operation"] = refresh_lecture_review_outputs(root, review_json, webui_output_dir=output_dir, vault=vault, folder=folder, target=target)
    else:
        result["executed"] = False
        result["after"] = before
        return _write_action_log(root, result)

    result["executed"] = True
    result["after"] = lecture_project_health(
        root,
        plan_json=plan_json,
        transcript=transcript,
        webui_output_dir=webui_output_dir,
        bilinote_root=bilinote_root,
        bilinote_patch_root=bilinote_patch_root,
    )
    return _write_action_log(root, result)


def lecture_action_log(root: str | Path) -> dict[str, Any]:
    """Return and render the persisted lecture-next action log."""
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "lecture-action-log.jsonl"
    markdown_path = paths["notes"] / "lecture-action-log.md"
    rows = read_jsonl(log_path)
    markdown_path.write_text(_render_action_log_markdown(root, rows), encoding="utf-8")
    return {
        "project": str(root),
        "log_path": str(log_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "actions": rows,
    }


def _write_action_log(root: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "lecture-action-log.jsonl"
    record = _action_record(result)
    append_jsonl(log_path, [record])
    log = lecture_action_log(root)
    result["action_log"] = {
        "record": record,
        "log_path": log["log_path"],
        "markdown_path": log["markdown_path"],
        "count": log["count"],
    }
    return result


def _action_record(result: dict[str, Any]) -> dict[str, Any]:
    selected = result.get("selected_action") if isinstance(result.get("selected_action"), dict) else {}
    before = result.get("before") if isinstance(result.get("before"), dict) else {}
    after = result.get("after") if isinstance(result.get("after"), dict) else {}
    operation = result.get("operation") if isinstance(result.get("operation"), dict) else {}
    return {
        "created_at": now_iso(),
        "action": selected.get("action", ""),
        "mode": selected.get("mode", ""),
        "reason": selected.get("reason", ""),
        "command": selected.get("command", ""),
        "dry_run": bool(result.get("dry_run")),
        "executed": bool(result.get("executed")),
        "before_level": before.get("level", ""),
        "after_level": after.get("level", ""),
        "operation_keys": sorted(operation.keys()),
    }


def _render_action_log_markdown(root: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Lecture Action Log",
        "",
        f"- Project: `{root}`",
        f"- Count: {len(rows)}",
        "",
        "| Time | Action | Mode | Executed | Level | Command | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        level = f"{row.get('before_level', '')} -> {row.get('after_level', '')}"
        executed = "yes" if row.get("executed") else "no"
        command = _markdown_cell(str(row.get("command") or ""))
        reason = _markdown_cell(str(row.get("reason") or ""))
        lines.append(
            f"| {row.get('created_at', '')} | {row.get('action', '')} | {row.get('mode', '')} | {executed} | {level} | {command} | {reason} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _select_safe_action(health: dict[str, Any]) -> dict[str, str]:
    for action in health.get("next_actions") or []:
        name = str(action.get("action") or "")
        command = str(action.get("command") or "")
        reason = str(action.get("reason") or "")
        if name in {"run_ready_pipeline", "build_package", "export_webui", "refresh_review"}:
            return {"action": name, "mode": "execute", "reason": reason, "command": command}
        if name in {"run_extractors", "human_review", "init_or_prepare", "setup_bilinote_patch"}:
            return {"action": name, "mode": "manual_required", "reason": reason, "command": command}
    return {"action": "none", "mode": "noop", "reason": "no safe next action detected", "command": ""}


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _planned_webui_output_dir(health: dict[str, Any]) -> str | None:
    files = health.get("files") if isinstance(health.get("files"), dict) else {}
    manifest = files.get("webui_manifest") if isinstance(files.get("webui_manifest"), dict) else {}
    manifest_path = str(manifest.get("path") or "")
    return str(Path(manifest_path).parent) if manifest_path else None


def _review_notes_path(health: dict[str, Any]) -> str:
    files = health.get("files") if isinstance(health.get("files"), dict) else {}
    review = files.get("webui_review_notes") if isinstance(files.get("webui_review_notes"), dict) else {}
    return str(review.get("path") or "")
