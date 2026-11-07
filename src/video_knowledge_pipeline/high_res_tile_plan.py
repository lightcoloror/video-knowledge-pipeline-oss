from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json

HIGH_RES_TILE_PLAN_SCHEMA = "video_knowledge_pipeline.high_res_tile_plan.v1"

DOCUMENT_ROUTES = {"document_visual", "mixed"}
DOCUMENT_MATERIALS = {"slide", "ppt", "document", "table", "formula", "code", "diagram", "board", "whiteboard", "screen"}
BLOCKER_KEYWORDS = {
    "ocr_text_empty",
    "ocr_wrapper_only",
    "screen_text_low_confidence",
    "missing_visual_text",
    "structured_visual_without_structure",
    "ebook_pipeline_failed",
    "low_information",
    "visual_text_empty",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def run_high_res_tile_plan(
    bundle_dir: str | Path,
    *,
    execute_tiles: bool = False,
    indexes: list[int] | None = None,
    limit: int = 0,
    tile_size: int = 768,
    overlap: float = 0.12,
    max_tiles_per_image: int = 12,
    include_routes: list[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Plan or write high-resolution local tiles for detail-heavy frame review.

    This is a local-only evidence preparation step inspired by InternVL dynamic
    tiling. It does not run OCR, VLMs, cloud APIs, or overwrite visual text.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"bundle missing timeline.json: {root}")

    manifest = _read_object(manifest_path)
    timeline = read_json(timeline_path)
    if not isinstance(timeline, list):
        timeline = []

    requested_indexes = _normalise_indexes(indexes)
    candidates = _select_candidates(root, timeline, requested_indexes=requested_indexes, include_routes=include_routes)
    available_candidates = len(candidates)
    if limit:
        candidates = candidates[: max(0, int(limit))]

    tile_size = max(128, int(tile_size or 768))
    max_tiles_per_image = max(1, int(max_tiles_per_image or 12))
    overlap = max(0.0, min(float(overlap or 0.0), 0.45))
    items = [_build_item(root, candidate, tile_size=tile_size, overlap=overlap, max_tiles_per_image=max_tiles_per_image) for candidate in candidates]
    if execute_tiles:
        _execute_tiles(root, items)

    summary = _tile_summary(items)
    result = {
        "schema": HIGH_RES_TILE_PLAN_SCHEMA,
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "execute_tiles": bool(execute_tiles),
        "limit": int(limit or 0),
        "requested_indexes": requested_indexes,
        "include_routes": include_routes or [],
        "tile_size": tile_size,
        "overlap": overlap,
        "max_tiles_per_image": max_tiles_per_image,
        "available_candidates": available_candidates,
        "selected_indexes": [item.get("index") for item in items],
        "summary": summary,
        "items": items,
    }

    json_path = root / "high-res-tile-plan.json"
    report_path = root / "high-res-tile-plan.md"
    args_path = root / "mcp-high-res-tile-plan.args.json"
    result["json_path"] = str(json_path)
    result["report_path"] = str(report_path)
    result["mcp_args_path"] = str(args_path)

    if write:
        with bundle_write_lock(root, operation="high_res_tile_plan"):
            write_json(json_path, result)
            report_path.write_text(render_high_res_tile_plan_markdown(result), encoding="utf-8")
            write_json(
                args_path,
                {
                    "bundle_dir": str(root),
                    "execute_tiles": False,
                    "indexes": requested_indexes,
                    "limit": int(limit or 0),
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "max_tiles_per_image": max_tiles_per_image,
                    "include_routes": include_routes or [],
                    "write": True,
                },
            )
            manifest["high_res_tile_plan"] = {
                "schema": HIGH_RES_TILE_PLAN_SCHEMA,
                "last_run_at": result["created_at"],
                "json_path": str(json_path),
                "report_path": str(report_path),
                "mcp_args_path": str(args_path),
                "summary": summary,
            }
            manifest["mcp_high_res_tile_plan_args"] = "mcp-high-res-tile-plan.args.json"
            write_json(manifest_path, manifest)
            _register_run(root, result)
    return result


def render_high_res_tile_plan_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# High-res Tile Plan",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Execute tiles: `{result.get('execute_tiles')}`",
        f"- Available candidates: `{result.get('available_candidates', 0)}`",
        f"- Selected items: `{summary.get('items', 0)}`",
        f"- Tiles planned: `{summary.get('tiles_planned', 0)}`",
        f"- Tiles written: `{summary.get('tiles_written', 0)}`",
        f"- Tile failures: `{summary.get('tiles_failed', 0)}`",
        f"- Tile size: `{result.get('tile_size', '')}`",
        f"- Overlap: `{result.get('overlap', '')}`",
        "",
        "## Why this exists",
        "",
        "This is a local-only evidence preparation step for frames where whole-frame OCR/ebook extraction is empty, wrapper-only, low-information, or too coarse for small UI text, tables, slides, code, formulas, or software screens.",
        "",
        "It does not mark OCR as successful, does not call a cloud model, and does not overwrite `visual_text`. Tile outputs are evidence for local VLM, targeted cloud VLM review, or human review.",
        "",
        "## Items",
        "",
        "| Index | Route | Reasons | Tiles | Written | Image |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        reasons = ", ".join(str(reason) for reason in item.get("reasons") or [])
        lines.append(
            "| {index} | `{route}` | {reasons} | {tiles} | {written} | `{image}` |".format(
                index=item.get("index", ""),
                route=_md_cell(str(item.get("visual_route") or "")),
                reasons=_md_cell(reasons),
                tiles=len(item.get("tiles") or []),
                written=sum(1 for tile in item.get("tiles") or [] if isinstance(tile, dict) and tile.get("status") == "written"),
                image=_md_cell(str(item.get("image_path") or "")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _select_candidates(
    root: Path,
    timeline: list[Any],
    *,
    requested_indexes: list[int],
    include_routes: list[str] | None,
) -> list[dict[str, Any]]:
    route_filter = {str(route).strip() for route in include_routes or [] if str(route).strip()}
    requested = set(requested_indexes)
    candidates: list[dict[str, Any]] = []
    for row in timeline:
        if not isinstance(row, dict):
            continue
        index = _int(row.get("index"))
        if requested and index not in requested:
            continue
        image_path = _first_image_path(root, row)
        if not image_path:
            continue
        reasons = _candidate_reasons(row)
        route = str(row.get("visual_route") or "")
        if route_filter and route not in route_filter:
            if not (requested or reasons):
                continue
        elif not requested and not reasons and route not in DOCUMENT_ROUTES:
            continue
        candidates.append(
            {
                "index": index,
                "timeline_item": row,
                "image_path": str(image_path),
                "visual_route": route,
                "reasons": reasons or [f"visual_route:{route or 'unknown'}"],
                "ebook_pipeline_status": _ebook_status(row),
            }
        )
    return candidates


def _candidate_reasons(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    route = str(item.get("visual_route") or "")
    if route in DOCUMENT_ROUTES:
        reasons.append(f"visual_route:{route}")
    material_types = _as_list(item.get("material_types"))
    if any(str(kind).lower() in DOCUMENT_MATERIALS for kind in material_types):
        reasons.append("document_or_screen_material")
    quality = _as_list(item.get("quality_issues")) + _as_list(item.get("knowledge_gaps"))
    recovery = item.get("screen_text_recovery") if isinstance(item.get("screen_text_recovery"), dict) else {}
    quality += _as_list(recovery.get("issues"))
    ebook_status = _ebook_status(item)
    if ebook_status and ebook_status.get("ok") is False:
        blocker = str(ebook_status.get("blocker") or "ebook_pipeline_failed")
        reasons.append(blocker)
        if blocker in {"ocr_wrapper_only", "ocr_text_empty", "ocr_text_low_information"}:
            reasons.append("needs_high_res_tile_recovery")
    if not str(item.get("visual_text") or "").strip() and route in DOCUMENT_ROUTES:
        reasons.append("visual_text_empty")
    for issue in quality:
        text = str(issue)
        if any(keyword in text for keyword in BLOCKER_KEYWORDS):
            reasons.append(text)
    return _dedupe(reasons)


def _ebook_status(item: dict[str, Any]) -> dict[str, Any]:
    status = item.get("ebook_pipeline_status") if isinstance(item.get("ebook_pipeline_status"), dict) else {}
    if status:
        return status
    structured = item.get("structured_visual")
    if isinstance(structured, dict):
        nested = structured.get("ebook_pipeline_status") if isinstance(structured.get("ebook_pipeline_status"), dict) else {}
        if nested:
            return nested
    if isinstance(structured, list):
        for value in structured:
            if isinstance(value, dict) and isinstance(value.get("ebook_pipeline_status"), dict):
                return value.get("ebook_pipeline_status") or {}
    return {}


def _compact_ebook_status(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}
    return {
        "ok": bool(value.get("ok")),
        "blocker": str(value.get("blocker") or ""),
        "quality": str(value.get("quality") or ""),
        "reason": str(value.get("reason") or value.get("error") or ""),
        "next_action": str(value.get("next_action") or ""),
        "meaningful_text_char_count": int(value.get("meaningful_text_char_count") or 0),
        "meaningful_line_count": int(value.get("meaningful_line_count") or 0),
    }

def _build_item(root: Path, candidate: dict[str, Any], *, tile_size: int, overlap: float, max_tiles_per_image: int) -> dict[str, Any]:
    image_path = Path(str(candidate.get("image_path") or "")).expanduser()
    if not image_path.is_absolute():
        image_path = root / image_path
    item = {
        "index": candidate.get("index"),
        "visual_route": candidate.get("visual_route", ""),
        "reasons": candidate.get("reasons") or [],
        "ebook_pipeline_status": _compact_ebook_status(candidate.get("ebook_pipeline_status")),
        "image_path": str(image_path),
        "image_exists": image_path.exists(),
        "image_width": 0,
        "image_height": 0,
        "tiles": [],
        "status": "planned",
        "error": "",
    }
    if not image_path.exists():
        item.update({"status": "failed", "error": f"source image not found: {image_path}"})
        return item
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        item.update({"status": "failed", "error": f"cannot read image size: {exc}"})
        return item
    item["image_width"] = width
    item["image_height"] = height
    boxes = _dynamic_tile_boxes(width, height, tile_size=tile_size, overlap=overlap, max_tiles=max_tiles_per_image)
    out_dir = root / "high-res-tiles" / f"timeline-{_int(candidate.get('index')):04d}"
    tiles = []
    for seq, box in enumerate(boxes, start=1):
        output = out_dir / f"tile-{seq:02d}.jpg"
        tiles.append(
            {
                "tile_id": f"{_int(candidate.get('index')):04d}-{seq:02d}",
                "sequence": seq,
                "box": list(box),
                "relative_box": _relative_box(box, width, height),
                "source_image": str(image_path),
                "planned_output": str(output),
                "status": "planned",
                "output_exists": output.exists(),
                "purpose": "high_resolution_detail_review",
                "prompt_hint": "Inspect small text, table cells, code, UI labels, formulas, and layout details in this crop. Preserve uncertainty.",
                "error": "",
            }
        )
    item["tiles"] = tiles
    return item


def _execute_tiles(root: Path, items: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        for item in items:
            item["status"] = "failed"
            item["error"] = f"Pillow unavailable: {exc}"
            for tile in item.get("tiles") or []:
                if isinstance(tile, dict):
                    tile.update({"status": "failed", "error": item["error"]})
        return
    for item in items:
        image_path = Path(str(item.get("image_path") or "")).expanduser()
        if not image_path.exists():
            item.update({"status": "failed", "error": f"source image not found: {image_path}"})
            continue
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                for tile in item.get("tiles") or []:
                    if not isinstance(tile, dict):
                        continue
                    output = Path(str(tile.get("planned_output") or "")).expanduser()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    box = tuple(int(part) for part in tile.get("box") or [0, 0, image.width, image.height])
                    image.crop(box).save(output, quality=92)
                    tile.update({"status": "written", "output_exists": output.exists(), "output_path": str(output), "error": ""})
            item["status"] = "completed"
            item["error"] = ""
        except Exception as exc:  # pragma: no cover - robust report path
            item.update({"status": "failed", "error": str(exc)})
            for tile in item.get("tiles") or []:
                if isinstance(tile, dict) and tile.get("status") != "written":
                    tile.update({"status": "failed", "error": str(exc)})


def _dynamic_tile_boxes(width: int, height: int, *, tile_size: int, overlap: float, max_tiles: int) -> list[tuple[int, int, int, int]]:
    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]
    aspect = max(0.1, width / max(1, height))
    cols = max(1, int(round(math.sqrt(max_tiles * aspect))))
    rows = max(1, int(math.ceil(max_tiles / cols)))
    while cols * rows > max_tiles and rows > 1:
        rows -= 1
    while cols * rows > max_tiles and cols > 1:
        cols -= 1
    cols = min(cols, max(1, math.ceil(width / max(1, tile_size))))
    rows = min(rows, max(1, math.ceil(height / max(1, tile_size))))
    if cols * rows < 2 and (width > tile_size or height > tile_size):
        if width >= height:
            cols = min(2, max_tiles)
        elif max_tiles >= 2:
            rows = 2
    tile_w = min(width, max(tile_size, math.ceil(width / cols * (1.0 + overlap))))
    tile_h = min(height, max(tile_size, math.ceil(height / rows * (1.0 + overlap))))
    xs = _axis_positions(width, tile_w, cols)
    ys = _axis_positions(height, tile_h, rows)
    boxes: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(width, x + tile_w), min(height, y + tile_h)))
            if len(boxes) >= max_tiles:
                return boxes
    return boxes


def _axis_positions(total: int, span: int, count: int) -> list[int]:
    if count <= 1 or total <= span:
        return [0]
    step = (total - span) / max(1, count - 1)
    return [max(0, min(total - span, int(round(step * idx)))) for idx in range(count)]


def _register_run(root: Path, result: dict[str, Any]) -> None:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    failed_items = _failed_items(result)
    if not result.get("items"):
        status = "not_needed"
    elif not result.get("execute_tiles"):
        status = "needs_execution"
    elif failed_items:
        status = "needs_retry"
    else:
        status = "completed"
    retry = ""
    if status in {"needs_execution", "needs_retry"}:
        retry = (
            f".\\scripts\\video-knowledge.ps1 high-res-tile-plan '{root}' "
            f"--execute-tiles --limit {int(result.get('limit') or 0)} "
            f"--tile-size {int(result.get('tile_size') or 768)} "
            f"--overlap {result.get('overlap')} "
            f"--max-tiles-per-image {int(result.get('max_tiles_per_image') or 12)}"
        )
    register_bundle_run(
        root,
        run_type="high_res_tile_plan",
        run_id="high-res-tile-plan",
        status=status,
        title="High-res tile plan",
        summary=f"Planned {summary.get('tiles_planned', 0)} tiles for {summary.get('items', 0)} detail-heavy frames.",
        inputs={"selected_indexes": result.get("selected_indexes", []), "available_candidates": result.get("available_candidates", 0)},
        parameters={
            "execute_tiles": result.get("execute_tiles", False),
            "tile_size": result.get("tile_size"),
            "overlap": result.get("overlap"),
            "max_tiles_per_image": result.get("max_tiles_per_image"),
        },
        artifacts=[
            {"key": "high_res_tile_plan_json", "path": result.get("json_path", "")},
            {"key": "high_res_tile_plan_report", "path": result.get("report_path", "")},
            {"key": "mcp_args", "path": result.get("mcp_args_path", "")},
        ],
        failed_items=failed_items,
        retry_command=retry,
        next_actions=_next_actions(status),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "no_vlm_execution": True,
            "no_ocr_success_claim": True,
            "purpose": "Prepare high-resolution tile evidence for targeted VLM or human review.",
        },
        write=True,
    )


def _tile_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    tiles = [tile for item in items for tile in (item.get("tiles") or []) if isinstance(tile, dict)]
    return {
        "items": len(items),
        "items_failed": sum(1 for item in items if item.get("status") == "failed"),
        "tiles_planned": len(tiles),
        "tiles_written": sum(1 for tile in tiles if tile.get("status") == "written"),
        "tiles_failed": sum(1 for tile in tiles if tile.get("status") == "failed"),
    }


def _failed_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    root = Path(str(result.get("bundle_dir") or ".")).expanduser().resolve()
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        index_arg = str(index or "")
        item_evidence = _item_evidence_paths(item)
        if item.get("status") == "failed":
            failed.append(
                {
                    "index": index,
                    "reason": "tile_plan_failed",
                    "detail": item.get("error", ""),
                    "suggested_next_tool": "high_res_tile_plan",
                    "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 high-res-tile-plan '{root}' --indexes {index_arg} --execute-tiles",
                    "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session '{root}' --limit 0 --group-by reason",
                    "evidence_paths": item_evidence,
                }
            )
        for tile in item.get("tiles") or []:
            if isinstance(tile, dict) and tile.get("status") == "failed":
                failed.append(
                    {
                        "index": index,
                        "reason": "tile_write_failed",
                        "detail": tile.get("error", ""),
                        "tile_id": tile.get("tile_id", ""),
                        "suggested_next_tool": "high_res_tile_plan",
                        "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 high-res-tile-plan '{root}' --indexes {index_arg} --execute-tiles",
                        "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session '{root}' --limit 0 --group-by reason",
                        "evidence_paths": _dedupe(item_evidence + [str(tile.get("planned_output") or tile.get("output_path") or "")]),
                    }
                )
    return failed


def _item_evidence_paths(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("image_path", "frame_path", "path"):
        value = str(item.get(key) or "").strip()
        if value:
            values.append(value)
    for key in ("frame_paths", "image_paths", "evidence_paths"):
        for value in _as_list(item.get(key)):
            text = str(value or "").strip()
            if text:
                values.append(text)
    return _dedupe(values)

def _next_actions(status: str) -> list[str]:
    if status == "needs_execution":
        return ["Run again with --execute-tiles to write local tile images, then review with local/targeted VLM or human audit."]
    if status == "needs_retry":
        return ["Inspect failed source images or Pillow availability, then retry --execute-tiles."]
    if status == "completed":
        return ["Use tile evidence for targeted multimodal review or human audit; do not treat tiles as OCR success by themselves."]
    return ["No high-resolution tile candidates were found."]


def _first_image_path(root: Path, item: dict[str, Any]) -> Path | None:
    candidates: list[Any] = []
    for key in ("image_path", "frame_path", "path"):
        candidates.append(item.get(key))
    for key in ("frame_paths", "image_paths", "evidence_paths"):
        candidates.extend(_as_list(item.get(key)))
    for value in candidates:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path.is_absolute():
            path = root / path
        return path
    return None


def _relative_box(box: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [round(x1 / width, 6), round(y1 / height, 6), round(x2 / width, 6), round(y2 / height, 6)]


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalise_indexes(values: list[int] | None) -> list[int]:
    if not values:
        return []
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        index = _int(value)
        if index and index not in seen:
            seen.add(index)
            result.append(index)
    return result


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
