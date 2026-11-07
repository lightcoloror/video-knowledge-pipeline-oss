from __future__ import annotations

from pathlib import Path
from typing import Any

from .lecture_package import export_lecture_obsidian, import_lecture_review
from .storage import read_json, write_json
from .webui_bridge import export_webui_bundle, refresh_bundle_review_html


def refresh_lecture_review_outputs(
    root: str | Path,
    review_json: str | Path,
    *,
    webui_output_dir: str | Path | None = None,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    target: str = "bilinote",
    allow_blocked_export: bool = False,
) -> dict[str, Any]:
    """Import a BiliNote/WebUI review JSON and refresh downstream handoff outputs."""
    review_path = Path(review_json).expanduser()
    default_review_notes_created = False
    if not review_path.exists():
        write_json(
            review_path,
            {
                "schema": "lecture_review_notes.v1",
                "reviews": [],
                "notes": "Auto-created empty review notes; no timeline changes were requested.",
            },
        )
        default_review_notes_created = True
    output_dir = Path(webui_output_dir) if webui_output_dir else _default_webui_output_dir(review_path)
    imported = import_lecture_review(root, review_path)
    bundle = export_webui_bundle(root, output_dir=output_dir, target=target)
    review_html_refresh = refresh_bundle_review_html(output_dir)
    result: dict[str, Any] = {
        "review_json": str(review_path),
        "default_review_notes_created": default_review_notes_created,
        "imported": imported,
        "webui_bundle": bundle,
        "review_html_refresh": review_html_refresh,
    }
    if vault:
        readiness = _bundle_readiness(bundle)
        result["review_readiness"] = readiness
        if not allow_blocked_export and readiness and not readiness.get("ready"):
            result["obsidian_export_blocked"] = {
                "reason": "review_readiness_not_ready",
                "blockers": readiness.get("blockers", []),
                "next_action": readiness.get("next_action", {}),
                "hint": "Pass allow_blocked_export=true only after a human accepts the remaining blockers.",
            }
            return result
        result["obsidian_export"] = export_lecture_obsidian(root, vault, folder)
    return result


def _bundle_readiness(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(bundle.get("manifest_path") or ""))
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        return {}
    readiness = manifest.get("review_readiness")
    return readiness if isinstance(readiness, dict) else {}


def _default_webui_output_dir(review_path: Path) -> Path | None:
    parent = review_path.parent
    if review_path.name == "review-notes.json" and (parent / "manifest.json").exists():
        return parent
    return None
