from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping


def build_personal_vlog_workbench(
    project: Mapping[str, Any],
    evidence: Mapping[str, Any],
    event_plan: Mapping[str, Any],
    recipes: Mapping[str, Mapping[str, Any]],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Build the local-only workbench payload; it does not create a service."""

    evidence_by_id = {
        str(row.get("asset_id") or ""): row
        for row in evidence.get("assets") or []
        if isinstance(row, Mapping)
    }
    event_by_id = {}
    for event in event_plan.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        for asset_id in event.get("member_asset_ids") or []:
            event_by_id[str(asset_id)] = str(event.get("event_id") or "")
    selected: dict[str, list[dict[str, Any]]] = {}
    for preset, recipe in recipes.items():
        for clip in recipe.get("clips") or []:
            if not isinstance(clip, Mapping):
                continue
            selected.setdefault(str(clip.get("source_asset_id") or ""), []).append(
                {
                    "preset": preset,
                    "clip_id": clip.get("clip_id"),
                    "story_role": clip.get("story_role"),
                    "score": (clip.get("score_breakdown") or {}).get("total"),
                    "timeline_start_seconds": clip.get("timeline_start_seconds"),
                    "duration_seconds": clip.get("timeline_duration_seconds"),
                }
            )
    assets = []
    for source in project.get("assets") or []:
        if not isinstance(source, Mapping):
            continue
        asset_id = str(source.get("asset_id") or "")
        fact = evidence_by_id.get(asset_id) or {}
        assets.append(
            {
                "asset_id": asset_id,
                "media_type": source.get("media_type"),
                "capture_time": source.get("capture_time"),
                "event_id": event_by_id.get(asset_id),
                "source_path": source.get("source_path"),
                "source_sha256": source.get("source_sha256"),
                "probe_status": (fact.get("probe_receipt") or {}).get("status", "ok"),
                "evidence_level": (fact.get("semantic") or {}).get("evidence_level"),
                "tags": (fact.get("semantic") or {}).get("tags") or [],
                "limitations": fact.get("limitations") or [],
                "appearance_cluster_ids": (fact.get("appearance") or {}).get(
                    "cluster_ids"
                )
                or [],
                "location_cluster_id": (fact.get("location") or {}).get("cluster_id"),
                "selected": selected.get(asset_id, []),
                "not_selected_is_not_deleted": True,
            }
        )
    payload = {
        "schema": "video_knowledge_pipeline.personal_vlog_workbench.v1",
        "surface": "video-workbench:personal-vlog-v1",
        "project_id": project.get("project_id"),
        "vlog_run_id": run_id,
        "status": "complete_draft",
        "review_required": True,
        "local_private_surface": True,
        "source_jump_available": True,
        "assets": assets,
        "events": list(event_plan.get("events") or []),
        "recipes": {
            preset: {
                "recipe_sha256": recipe.get("recipe_sha256"),
                "target": recipe.get("target"),
                "clip_count": len(recipe.get("clips") or []),
                "review_gap_count": len(recipe.get("review_gaps") or []),
            }
            for preset, recipe in recipes.items()
        },
        "actions": {
            "accept": "human_only",
            "render": "not_available_in_vkp",
            "publish": "not_available_in_vkp",
            "delete": "not_available",
        },
    }
    return payload


def render_personal_vlog_workbench(payload: Mapping[str, Any]) -> str:
    cards = []
    for row in payload.get("assets") or []:
        if not isinstance(row, Mapping):
            continue
        source_path = str(row.get("source_path") or "")
        selected = ", ".join(
            f"{item.get('preset')} / {item.get('story_role')}"
            for item in row.get("selected") or []
            if isinstance(item, Mapping)
        ) or "未入选当前配方（不代表删除）"
        cards.append(
            "<article class='asset-card'>"
            f"<h3>{html.escape(str(row.get('asset_id') or ''))}</h3>"
            f"<p>{html.escape(str(row.get('media_type') or ''))} · "
            f"{html.escape(str(row.get('event_id') or '未分段'))}</p>"
            f"<p>证据：{html.escape(str(row.get('evidence_level') or 'unknown'))} · "
            f"探针：{html.escape(str(row.get('probe_status') or 'unknown'))}</p>"
            f"<p>入选：{html.escape(selected)}</p>"
            f"<a href='{html.escape(Path(source_path).as_uri() if source_path else '#')}'>源文件</a>"
            "</article>"
        )
    event_rows = []
    for event in payload.get("events") or []:
        if isinstance(event, Mapping):
            event_rows.append(
                "<li><strong>"
                + html.escape(str(event.get("event_id") or ""))
                + "</strong>："
                + html.escape(", ".join(str(v) for v in event.get("member_asset_ids") or []))
                + "</li>"
            )
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VKP 个人 Vlog 审核工作台</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; background:#101419; color:#eef2f5; }}
body {{ margin:0; padding:24px; }} header,.panel {{ max-width:1440px; margin:0 auto 20px; }}
.badge {{ display:inline-block; padding:4px 9px; border-radius:999px; background:#493b12; color:#ffd86b; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
.asset-card {{ border:1px solid #34414d; border-radius:10px; padding:14px; background:#18212a; overflow-wrap:anywhere; }}
a {{ color:#7dccff; }} .warning {{ border-left:4px solid #e3ad28; padding-left:12px; }}
</style></head><body data-vkp-surface="personal-vlog-v1">
<header><span class="badge">complete_draft · review_required</span><h1>个人 Vlog 候选剪辑工作台</h1>
<p class="warning">未入选不代表删除；接受、渲染和发布均需人工操作。</p></header>
<section class="panel"><h2>活动段落</h2><ol>{''.join(event_rows)}</ol></section>
<section class="panel"><h2>素材与入选理由</h2><div class="grid">{''.join(cards)}</div></section>
<script type="application/json" id="vkp-personal-vlog-data">{data}</script>
</body></html>"""
