from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .frame_recapture import _coverage_audit, _quality_audit, _quality_issues, _quality_score
from .models import now_iso
from .repair_status import build_repair_status
from .storage import read_json, write_json
from .visual_integration import integrated_visual


ROUTES = {"document_visual", "semantic_frame", "temporal_sequence", "mixed", "unknown"}
DOCUMENT_VISUAL_TYPES = {
    "formula",
    "table",
    "code",
    "board",
    "slide",
    "document",
    "diagram",
    "text",
}


def run_video_frame_router(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    content_profile: str = "auto",
    write: bool = True,
) -> dict[str, Any]:
    """Classify timeline frames into document, semantic-frame, or temporal visual routes."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    timeline = _read_timeline(root)
    resolved_profile = _resolve_content_profile(
        manifest, timeline, requested=content_profile
    )

    imported = _read_route_input(input_json) if input_json else []
    imported_by_index = {int(row["index"]): row for row in imported if _int_value(row.get("index")) > 0}
    items: list[dict[str, Any]] = []
    updated_indexes: list[int] = []
    for index, item in enumerate(timeline, start=1):
        route = (
            _normalise_imported_route(imported_by_index.get(index))
            if index in imported_by_index
            else _route_item(item, content_profile=resolved_profile)
        )
        route["index"] = index
        route["start"] = item.get("start", 0)
        route["end"] = item.get("end", 0)
        route["frame_paths"] = _frame_paths(item)
        route["transcript"] = str(item.get("transcript") or "")
        route["visual_text"] = str(item.get("visual_text") or "")
        items.append(route)
        if write:
            _apply_route(item, route)
            updated_indexes.append(index)

    summary = {
        "schema": "lecture_video_frame_router_summary.v1",
        "total": len(items),
        "write": write,
        "input_json": str(Path(input_json).expanduser()) if input_json else "",
        "content_profile": resolved_profile,
        "updated": len(updated_indexes) if write else 0,
        "routes": {route: sum(1 for item in items if item.get("visual_route") == route) for route in sorted(ROUTES)},
        "low_confidence": sum(1 for item in items if float(item.get("confidence") or 0) < 0.6),
        "updated_at": now_iso(),
    }
    template_path = write_video_frame_router_input_template(root, items)
    manifest["video_frame_router"] = {
        "schema": "lecture_video_frame_router.v1",
        "count": len(items),
        "items": items,
        "input_template_json": str(template_path),
        "last_run": summary,
    }
    if write:
        write_json(timeline_path, timeline)
        _sync_source_package(root, manifest, timeline)
    manifest["coverage"] = _coverage_audit(timeline)
    manifest["quality_audit"] = _quality_audit(timeline)
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    manifest["video_frame_router"]["last_run"]["updated_indexes"] = updated_indexes
    write_json(manifest_path, manifest)

    report_path = root / "video-frame-router-report.md"
    report_path.write_text(_render_report(root, items, summary, template_path), encoding="utf-8")
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "report_path": str(report_path),
        "input_template_json": str(template_path),
        "summary": summary,
        "items": items,
    }


def write_video_frame_router_input_template(root: str | Path, items: list[dict[str, Any]]) -> Path:
    path = Path(root) / "video-frame-router-input-template.json"
    payload = {
        "schema": "lecture_video_frame_route_input.v1",
        "items": [
            {
                "index": item.get("index"),
                "visual_route": item.get("visual_route"),
                "secondary_visual_routes": item.get("secondary_visual_routes", []),
                "confidence": item.get("confidence"),
                "reasons": item.get("reasons", []),
                "frame_paths": item.get("frame_paths", []),
                "human_note": "",
            }
            for item in items
        ],
    }
    write_json(path, payload)
    return path


def _route_item(
    item: dict[str, Any], *, content_profile: str = "general"
) -> dict[str, Any]:
    text = "\n".join(
        [
            str(item.get("transcript") or ""),
            str(item.get("visual_text") or ""),
            " ".join(str(value) for value in item.get("signals") or []),
            " ".join(str(value) for value in item.get("material_types") or []),
        ]
    ).lower()
    material_types = {str(value).lower() for value in item.get("material_types") or []}
    frame_count = len(_frame_paths(item))
    document_score, document_reasons = _score_document(item, text, material_types)
    temporal_score, temporal_reasons = _score_temporal(item, text, frame_count)
    semantic_score, semantic_reasons = _score_semantic(item, text, material_types, frame_count)

    route = "unknown"
    confidence = 0.35
    reasons: list[str] = []
    if document_score >= 2 and temporal_score >= 2:
        route = "mixed"
        confidence = min(0.85, 0.45 + 0.1 * (document_score + temporal_score))
        reasons = document_reasons + temporal_reasons
    elif temporal_score >= 2:
        route = "temporal_sequence"
        confidence = min(0.9, 0.45 + 0.12 * temporal_score)
        reasons = temporal_reasons
    elif document_score >= 2:
        route = "document_visual"
        confidence = min(0.9, 0.45 + 0.12 * document_score)
        reasons = document_reasons
    elif (
        content_profile == "lecture-slides-v1"
        and _lecture_document_signal(item, material_types)
        and not _strong_semantic_signal(item, text, material_types)
    ):
        route = "document_visual"
        confidence = min(0.82, 0.62 + 0.1 * document_score)
        reasons = [*document_reasons, "lecture_slides_ocr_first"]
    elif semantic_score >= 1:
        route = "semantic_frame"
        confidence = min(0.8, 0.45 + 0.1 * semantic_score)
        reasons = semantic_reasons
    secondary_routes = []
    if route == "semantic_frame" and document_score > 0:
        secondary_routes.append("document_visual")

    return {
        "visual_route": route,
        "secondary_visual_routes": secondary_routes,
        "confidence": round(confidence, 2),
        "reasons": _dedupe(reasons or ["insufficient_visual_signals"]),
        "needs_human_review": confidence < 0.6 or route in {"unknown", "mixed"},
    }


def _score_document(item: dict[str, Any], text: str, material_types: set[str]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if material_types & {"table", "formula", "code", "board", "slide", "document", "diagram"}:
        score += 2
        reasons.append("document_like_material_type")
    visual_text = str(item.get("visual_text") or "")
    if len(visual_text.strip()) >= 40:
        score += 1
        reasons.append("dense_visual_text")
    if re.search(r"(\|.+\|)|(```|def |class |function |SELECT )|(\$[^$]+\$)|([=+\-*/^]{2,})", visual_text):
        score += 2
        reasons.append("table_formula_or_code_pattern")
    if any(word in text for word in ["ppt", "slide", "blackboard", "whiteboard", "board", "table", "formula", "code", "document", "板书", "公式", "表格", "代码", "文档"]):
        score += 1
        reasons.append("document_keyword")
    image_score, image_reason = _image_document_score(item)
    if image_score:
        score += image_score
        reasons.append(image_reason)
    return score, reasons


def _score_temporal(item: dict[str, Any], text: str, frame_count: int) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if frame_count >= 2:
        score += 1
        reasons.append("multiple_frames_available")
    if _safe_float(item.get("end")) - _safe_float(item.get("start")) >= 2.0 and frame_count >= 2:
        score += 1
        reasons.append("multi_second_frame_span")
    if any(word in text for word in ["scene_change", "mouse", "cursor", "click", "drag", "scroll", "operation", "workflow", "demo", "step", "操作", "鼠标", "点击", "拖动", "滚动", "演示", "流程", "变化", "步骤"]):
        score += 2
        reasons.append("operation_or_change_signal")
    return score, reasons


def _score_semantic(item: dict[str, Any], text: str, material_types: set[str], frame_count: int) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    # A frame existing is a system fact, not evidence that a multimodal model is
    # needed. The former baseline promoted every otherwise-unclassified slide to
    # semantic_frame and created exhaustive remote-review blockers.
    if material_types & {"image", "screen", "object", "demo", "experiment", "interface"}:
        score += 1
        reasons.append("visual_semantic_material_type")
    if any(word in text for word in ["object", "person", "gesture", "point", "interface", "screen", "software", "demo", "experiment", "实物", "动作", "指向", "界面", "软件", "实验", "空间"]):
        score += 1
        reasons.append("semantic_visual_keyword")
    return score, reasons


def _strong_semantic_signal(
    item: dict[str, Any], text: str, material_types: set[str]
) -> bool:
    if material_types & {"image", "object", "demo", "experiment"}:
        return True
    return any(
        word in text
        for word in (
            "gesture",
            "point",
            "spatial",
            "demo",
            "experiment",
            "动作",
            "指向",
            "空间",
            "实验",
            "实物",
        )
    ) or bool(item.get("non_text_visual_information"))


def _lecture_document_signal(
    item: dict[str, Any], material_types: set[str]
) -> bool:
    return bool(
        material_types & (DOCUMENT_VISUAL_TYPES | {"screen", "interface"})
        or str(item.get("visual_text") or item.get("ocr_text") or "").strip()
    )


def _resolve_content_profile(
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    *,
    requested: str,
) -> str:
    """Resolve the OCR-first lecture profile without guessing from a title."""
    value = str(requested or "auto").strip().lower()
    aliases = {
        "lecture": "lecture-slides-v1",
        "slides": "lecture-slides-v1",
        "ppt": "lecture-slides-v1",
        "lecture-slides": "lecture-slides-v1",
        "general": "general",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "general", "lecture-slides-v1"}:
        raise ValueError("content_profile must be auto, general, or lecture-slides-v1")
    if value != "auto":
        return value
    explicit = str(
        manifest.get("content_profile")
        or manifest.get("video_content_profile")
        or manifest.get("processing_preset")
        or ""
    ).strip().lower()
    explicit = aliases.get(explicit, explicit)
    if explicit in {"general", "lecture-slides-v1"}:
        return explicit
    if not timeline:
        return "general"
    document_items = 0
    for item in timeline:
        material_types = {
            str(entry).strip().lower()
            for entry in item.get("material_types") or []
        }
        visual_text = str(item.get("visual_text") or "").strip()
        if material_types & DOCUMENT_VISUAL_TYPES or len(visual_text) >= 20:
            document_items += 1
    threshold = max(2, (len(timeline) + 1) // 2)
    return "lecture-slides-v1" if document_items >= threshold else "general"


def _normalise_imported_route(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {"visual_route": "unknown", "confidence": 0.0, "reasons": ["empty_import"], "needs_human_review": True}
    route = str(row.get("visual_route") or row.get("route") or "").strip()
    if route not in ROUTES:
        route = "unknown"
    confidence = _safe_float(row.get("confidence") if row.get("confidence") is not None else row.get("visual_route_confidence"))
    if confidence <= 0:
        confidence = 0.6 if route != "unknown" else 0.3
    reasons = row.get("reasons") or row.get("visual_route_reasons") or ["external_import"]
    secondary = row.get("secondary_visual_routes") if isinstance(row.get("secondary_visual_routes"), list) else []
    return {
        "visual_route": route,
        "secondary_visual_routes": [str(value) for value in secondary if str(value) in ROUTES],
        "confidence": round(confidence, 2),
        "reasons": [str(value) for value in reasons] if isinstance(reasons, list) else [str(reasons)],
        "needs_human_review": bool(row.get("needs_human_review", confidence < 0.6 or route in {"unknown", "mixed"})),
    }


def _apply_route(item: dict[str, Any], route: dict[str, Any]) -> None:
    item["visual_route"] = route.get("visual_route") or "unknown"
    item["secondary_visual_routes"] = route.get("secondary_visual_routes") or []
    item["visual_route_confidence"] = route.get("confidence", 0)
    item["visual_route_reasons"] = route.get("reasons", [])
    item["visual_route_updated_at"] = now_iso()
    if route.get("needs_human_review"):
        item["needs_human_review"] = True
    issues = _quality_issues(item)
    item["quality_issues"] = issues
    item["quality_score"] = _quality_score(issues)
    item["integrated_visual"] = integrated_visual(item)


def _read_route_input(input_json: str | Path) -> list[dict[str, Any]]:
    path = Path(input_json).expanduser()
    data = read_json(path)
    rows = data.get("items") or data.get("results") or data.get("routes") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("video frame route input JSON must be a list or an object with items/results/routes")
    return [row for row in rows if isinstance(row, dict)]


def _read_timeline(root: Path) -> list[dict[str, Any]]:
    path = root / "timeline.json"
    if not path.exists():
        return []
    data = read_json(path)
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _sync_source_package(root: Path, manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> None:
    source = Path(str(manifest.get("source_package") or "")).expanduser()
    if not source.exists() or not source.is_file():
        return
    package = read_json(source)
    if not isinstance(package, dict) or not isinstance(package.get("timeline"), list):
        return
    for index, item in enumerate(package["timeline"], start=1):
        if index <= len(timeline) and isinstance(item, dict):
            for key in ("visual_route", "secondary_visual_routes", "visual_route_confidence", "visual_route_reasons", "visual_route_updated_at", "needs_human_review", "integrated_visual"):
                if key in timeline[index - 1]:
                    item[key] = timeline[index - 1][key]
    package["coverage"] = _coverage_audit(package["timeline"])
    package["quality_audit"] = _quality_audit(package["timeline"])
    package["video_frame_router_backfilled_at"] = now_iso()
    write_json(source, package)


def _render_report(root: Path, items: list[dict[str, Any]], summary: dict[str, Any], template_path: Path) -> str:
    lines = [
        "# Video Frame Router Report",
        "",
        f"- Bundle: `{root}`",
        f"- Total: {summary.get('total', 0)}",
        f"- Template: `{template_path}`",
        "",
    ]
    for route, count in sorted((summary.get("routes") or {}).items()):
        lines.append(f"- {route}: {count}")
    lines.append("")
    for item in items:
        lines.extend(
            [
                f"## Timeline {item.get('index')}",
                "",
                f"- Route: `{item.get('visual_route')}`",
                f"- Confidence: `{item.get('confidence')}`",
                f"- Reasons: {', '.join(item.get('reasons') or [])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _frame_paths(item: dict[str, Any]) -> list[str]:
    values = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
    if not values and isinstance(item.get("assets"), list):
        values = [asset.get("path") or asset.get("source") for asset in item["assets"] if isinstance(asset, dict)]
    return [str(value) for value in values if str(value)]


def _absolute_image_paths(item: dict[str, Any]) -> list[Path]:
    values: list[str] = []
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        raw = str(asset.get("source") or asset.get("resolved_path") or asset.get("path") or "")
        if raw:
            values.append(raw)
    values.extend(str(path) for path in item.get("frame_paths") or [] if path)
    result = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_absolute() and path.exists() and path.is_file():
            result.append(path)
    return result


def _image_document_score(item: dict[str, Any]) -> tuple[int, str]:
    paths = _absolute_image_paths(item)
    if not paths:
        return 0, ""
    try:
        from PIL import Image, ImageStat
    except Exception:
        return 0, ""
    try:
        with Image.open(paths[0]) as image:
            sample = image.convert("RGB").resize((96, 54))
            gray = sample.convert("L")
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0])
            stddev = float(stat.stddev[0])
            colors = sample.quantize(colors=32).getcolors(maxcolors=4096) or []
    except Exception:
        return 0, ""
    color_count = len(colors)
    dark_title_card = mean < 80 and stddev > 35 and color_count <= 24
    light_slide_card = mean > 170 and stddev > 25 and color_count <= 28
    if dark_title_card or light_slide_card:
        return 2, "slide_or_title_card_image_heuristic"
    return 0, ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in result:
            result.append(text)
    return result
