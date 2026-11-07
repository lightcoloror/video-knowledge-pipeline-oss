from __future__ import annotations

from typing import Any

from .models import now_iso


def integrated_visual(item: dict[str, Any]) -> dict[str, Any]:
    """Build the merged visual evidence record for one timeline item."""
    evidence_frame_paths = _evidence_frame_paths(item)
    return {
        "schema": "lecture_integrated_visual.v1",
        "visual_route": item.get("visual_route") or "",
        "secondary_visual_routes": item.get("secondary_visual_routes") if isinstance(item.get("secondary_visual_routes"), list) else [],
        "visual_text": item.get("visual_text") or "",
        "structured_visual": item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else [],
        "visual_understanding": item.get("visual_understanding") if isinstance(item.get("visual_understanding"), dict) else {},
        "temporal_visual_understanding": item.get("temporal_visual_understanding") if isinstance(item.get("temporal_visual_understanding"), dict) else {},
        "tagger_annotations": item.get("tagger_annotations") if isinstance(item.get("tagger_annotations"), list) else [],
        "tagger_tags": item.get("tagger_tags") if isinstance(item.get("tagger_tags"), list) else [],
        "tagger_visual_summary": item.get("tagger_visual_summary") or "",
        "tagger_time_axis": item.get("tagger_time_axis") if isinstance(item.get("tagger_time_axis"), list) else [],
        "term_candidates": item.get("term_candidates") if isinstance(item.get("term_candidates"), list) else [],
        "evidence_frame_paths": evidence_frame_paths,
        "updated_at": now_iso(),
    }


def _evidence_frame_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in paths:
            paths.append(text)

    for path in item.get("frame_paths") or []:
        add(path)
    for path in item.get("temporal_frame_paths") or []:
        add(path)
    group = item.get("temporal_frame_group") if isinstance(item.get("temporal_frame_group"), dict) else {}
    for path in group.get("frame_paths") or []:
        add(path)
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            add(asset.get("source") or asset.get("resolved_path") or asset.get("path"))
    return paths
