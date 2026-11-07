from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any


OCR_HIGH_CONFIDENCE = 0.85
STATIC_MAIN_DELTA = 0.012
DYNAMIC_MAIN_DELTA = 0.04
NEAR_DUPLICATE_TEXT_SIMILARITY = 0.97
NEAR_DUPLICATE_DHASH_DISTANCE = 4
MOTION_GRID_COLUMNS = 8
MOTION_GRID_ROWS = 6
ACTIVE_TILE_DELTA = 0.018
LOCALIZED_TILE_FRACTION_MAX = 0.25
LOCALIZED_BBOX_FRACTION_MAX = 0.4


def evaluate_escalation_signals(
    root: Path,
    item: dict[str, Any],
    *,
    transcript: str,
    visual_text: str,
    tags: list[str],
    frame_paths: list[str],
    scene_boundaries: list[dict[str, Any]],
) -> dict[str, Any]:
    temporal_paths = [
        str(value) for value in item.get("temporal_frame_paths") or [] if value
    ]
    ocr = _ocr_evidence(item, visual_text)
    layout = _complex_layout_signals(item, transcript, tags)
    conflicts = _cross_source_conflicts(transcript, visual_text, _issue_keys(item))
    non_text = _non_text_signals(item, tags)
    frame_change = _frame_change_evidence(
        root, temporal_paths, overlay_regions=_overlay_regions(item)
    )
    scene_boundary = _scene_boundary_evidence(item, scene_boundaries)
    ocr_sufficient = bool(
        visual_text
        and not ocr["weak_text"]
        and item.get("structured_visual")
        and ocr["status"] == "high_confidence"
        and not layout
        and not conflicts
        and not non_text
    )
    return {
        "ocr_evidence": ocr,
        "complex_layout_signals": layout,
        "cross_source_conflicts": conflicts,
        "non_text_signals": non_text,
        "frame_change_evidence": frame_change,
        "scene_boundary_evidence": scene_boundary,
        "ocr_sufficient_simple_layout": ocr_sufficient,
        "representative_fingerprint": _representative_fingerprint(root, frame_paths),
    }


def apply_duplicate_suppression(rows: list[dict[str, Any]]) -> None:
    _suppress_semantic_duplicates_of_completed_analysis(rows)
    _suppress_visual_structure_duplicates_of_resolved_ocr(rows)
    previous: dict[str, Any] | None = None
    for row in sorted(rows, key=lambda value: int(value.get("index") or 0)):
        if row.get("recommended_action") == "none":
            continue
        if (row.get("frame_change_evidence") or {}).get("status") == "dynamic":
            previous = row
            continue
        if previous is None:
            previous = row
            continue
        if previous.get("recommended_action") != row.get(
            "recommended_action"
        ) or not _rows_are_near_duplicates(previous, row):
            previous = row
            continue
        if _duplicate_rank(row) > _duplicate_rank(previous):
            _suppress_duplicate(previous, preferred_index=row.get("index"))
            previous = row
        else:
            _suppress_duplicate(row, preferred_index=previous.get("index"))


def _suppress_semantic_duplicates_of_completed_analysis(
    rows: list[dict[str, Any]],
) -> None:
    """Reuse a completed analysis for an identical static slide at any timestamp."""

    anchors = [
        row
        for row in rows
        if row.get("has_valid_visual_understanding")
        and isinstance(row.get("representative_fingerprint"), int)
        and (row.get("frame_change_evidence") or {}).get("status") != "dynamic"
    ]
    for row in rows:
        if row.get("recommended_action") != "semantic_multimodal" or row.get(
            "has_valid_visual_understanding"
        ):
            continue
        matching = [
            anchor
            for anchor in anchors
            if _rows_have_identical_static_visual(anchor, row)
        ]
        if not matching:
            continue
        preferred = min(matching, key=lambda anchor: _row_time_distance(anchor, row))
        _suppress_duplicate(row, preferred_index=preferred.get("index"))
        row["suppression_reasons"] = _unique(
            [
                *(row.get("suppression_reasons") or []),
                f"completed_semantic_analysis_duplicate_of_index_{preferred.get('index')}",
            ]
        )


def _rows_have_identical_static_visual(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    if left.get("representative_fingerprint") != right.get(
        "representative_fingerprint"
    ):
        return False
    left_text = re.sub(
        r"\s+", "", str(left.get("visual_text_excerpt") or "")
    ).casefold()
    right_text = re.sub(
        r"\s+", "", str(right.get("visual_text_excerpt") or "")
    ).casefold()
    return bool(len(left_text) >= 8 and left_text == right_text)


def _suppress_visual_structure_duplicates_of_resolved_ocr(
    rows: list[dict[str, Any]],
) -> None:
    """Avoid progressively reselecting frames already represented by OCR evidence.

    OCR can change a representative frame's next action from visual-structure
    recovery to semantic review (or no action). It must still remain an anchor
    for adjacent, visually equivalent frames whose OCR has not run yet.
    """

    ordered = sorted(rows, key=lambda value: int(value.get("index") or 0))
    anchors = [
        row
        for row in ordered
        if (row.get("ocr_evidence") or {}).get("has_text")
        and row.get("has_structured_visual_evidence")
        and (row.get("frame_change_evidence") or {}).get("status") != "dynamic"
    ]
    for row in ordered:
        if row.get("recommended_action") != "visual_structure_first":
            continue
        matching = [
            anchor
            for anchor in anchors
            if anchor is not row
            and _row_time_distance(anchor, row) <= 30.0
            and _rows_are_near_duplicates(anchor, row)
        ]
        if not matching:
            continue
        preferred = min(matching, key=lambda anchor: _row_time_distance(anchor, row))
        _suppress_duplicate(row, preferred_index=preferred.get("index"))
        row["suppression_reasons"] = _unique(
            [
                *(row.get("suppression_reasons") or []),
                f"resolved_ocr_near_duplicate_of_index_{preferred.get('index')}",
            ]
        )


def _row_time_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_midpoint = _row_midpoint(left)
    right_midpoint = _row_midpoint(right)
    if left_midpoint is None or right_midpoint is None:
        return float("inf")
    return abs(left_midpoint - right_midpoint)


def _row_midpoint(row: dict[str, Any]) -> float | None:
    start = _seconds(row.get("start"))
    end = _seconds(row.get("end"))
    if start is None and end is None:
        return None
    if start is None:
        return end
    if end is None:
        return start
    return (start + end) / 2.0


def _duplicate_rank(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    ocr = row.get("ocr_evidence") if isinstance(row.get("ocr_evidence"), dict) else {}
    return (
        int(bool(row.get("has_temporal_frame_evidence"))),
        len(row.get("cross_source_conflicts") or []),
        int(bool(row.get("visual_text_excerpt"))),
        int(ocr.get("status") == "high_confidence"),
        int(row.get("score") or 0),
        len(row.get("benefit_reasons") or []),
        -int(row.get("index") or 0),
    )


def _suppress_duplicate(row: dict[str, Any], *, preferred_index: Any) -> None:
    row["duplicate_of_index"] = preferred_index
    row["suppression_reasons"] = _unique(
        [
            *(row.get("suppression_reasons") or []),
            f"near_duplicate_of_index_{preferred_index}",
        ]
    )
    row["recommended_action"] = "none"
    row["selected_action"] = "none"
    row["score"] = 0
    row["priority"] = "none"
    row["estimated_model_calls"] = 0
    row["estimated_images"] = 0
    row["recommended_execution_location"] = "none"
    row["local_prerequisite_action"] = "none"


def execution_estimate(action: str, temporal_paths: list[str]) -> tuple[int, int, str]:
    if action == "temporal_multimodal":
        return (
            1,
            max(2, min(12, len(temporal_paths))),
            "local_preferred_remote_approved",
        )
    if action == "semantic_multimodal":
        return 1, 1, "local_preferred_remote_approved"
    if action == "visual_structure_first":
        return 0, 1, "local"
    return 0, 0, "none"


def _ocr_evidence(item: dict[str, Any], visual_text: str) -> dict[str, Any]:
    confidence = _first_confidence(item)
    if not visual_text:
        status = "empty"
    elif confidence is None:
        status = "unknown"
    elif confidence >= OCR_HIGH_CONFIDENCE:
        status = "high_confidence"
    else:
        status = "low_confidence"
    return {
        "status": status,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "high_confidence_threshold": OCR_HIGH_CONFIDENCE,
        "has_text": bool(visual_text),
        "weak_text": _weak_visual_text(visual_text) if visual_text else True,
        "has_structure": bool(item.get("structured_visual")),
    }


def _first_confidence(item: dict[str, Any]) -> float | None:
    for key in ("ocr_confidence", "visual_text_confidence", "screen_text_confidence"):
        value = _normalise_confidence(item.get(key))
        if value is not None:
            return value
    for key in (
        "ocr",
        "ocr_result",
        "screen_text",
        "screen_text_evidence",
        "integrated_visual",
    ):
        payload = item.get(key)
        if not isinstance(payload, dict):
            continue
        for field in ("confidence", "ocr_confidence", "visual_text_confidence"):
            value = _normalise_confidence(payload.get(field))
            if value is not None:
                return value
    return None


def _normalise_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if 1 < numeric <= 100:
        numeric /= 100
    return numeric if 0 <= numeric <= 1 else None


def _complex_layout_signals(
    item: dict[str, Any], transcript: str, tags: list[str]
) -> list[str]:
    haystack = " ".join(
        [transcript, " ".join(tags), str(item.get("structured_visual") or "")]
    ).lower()
    mapping = {
        "diagram": ["diagram", "flowchart", "mindmap", "架构图", "流程图", "关系图"],
        "multi_column": ["multi_column", "two_column", "多栏", "双栏", "左边", "右边"],
        "spatial_relation": ["arrow", "connector", "箭头", "连线", "对应", "关系"],
        "chat_layout": ["chat", "message_bubble", "聊天", "对话框"],
        "chart": ["chart", "plot", "graph", "图表", "曲线", "柱状", "饼图"],
        "formula_or_code": ["formula", "equation", "公式", "代码", "code_block"],
    }
    return [
        name
        for name, terms in mapping.items()
        if any(term in haystack for term in terms)
    ]


def _non_text_signals(item: dict[str, Any], tags: list[str]) -> list[str]:
    material = (
        item.get("material_types")
        if isinstance(item.get("material_types"), list)
        else []
    )
    haystack = " ".join([*tags, *[str(value) for value in material]]).lower()
    mapping = {
        "object_or_photo": ["photo", "object", "人物", "实物", "照片"],
        "gesture_or_focus": ["gesture", "pointer", "highlight", "手势", "指向", "高亮"],
        "map_or_ui_state": ["map", "interface_state", "地图", "界面状态", "弹窗"],
    }
    return [
        name
        for name, terms in mapping.items()
        if any(term in haystack for term in terms)
    ]


def _cross_source_conflicts(
    transcript: str, visual_text: str, issues: set[str]
) -> list[str]:
    conflicts: list[str] = []
    if transcript and visual_text:
        left_numbers = set(re.findall(r"\d+(?:\.\d+)?", transcript))
        right_numbers = set(re.findall(r"\d+(?:\.\d+)?", visual_text))
        if (
            left_numbers
            and right_numbers
            and not left_numbers.intersection(right_numbers)
        ):
            conflicts.append("number_mismatch_between_asr_and_ocr")
        left_terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", transcript)
        }
        right_terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", visual_text)
        }
        if left_terms and right_terms and left_terms.symmetric_difference(right_terms):
            conflicts.append("term_mismatch_between_asr_and_ocr")
    if issues.intersection(
        {"asr_ocr_conflict", "cross_source_conflict", "term_resolution_needs_review"}
    ):
        conflicts.append("explicit_cross_source_conflict")
    return conflicts


def _scene_boundary_evidence(
    item: dict[str, Any], boundaries: list[dict[str, Any]]
) -> dict[str, Any]:
    start = _seconds(item.get("start"))
    end = _seconds(item.get("end"))
    matched: list[dict[str, Any]] = []
    if start is not None and end is not None:
        for row in boundaries:
            point = _seconds(
                row.get("seconds")
                if row.get("seconds") is not None
                else row.get("time")
            )
            if point is not None and start <= point <= end:
                matched.append(
                    {
                        "seconds": point,
                        "reason": str(row.get("reason") or "scene_boundary"),
                    }
                )
    return {
        "matched": bool(matched),
        "boundaries": matched[:8],
        "source": "exports/scene-detection.json",
    }


def _frame_change_evidence(
    root: Path,
    values: list[str],
    *,
    overlay_regions: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    paths = [
        path for value in values if (path := _safe_frame_path(root, value)) is not None
    ]
    overlay_regions = overlay_regions or []
    base = {
        "method": "adaptive_grid_localized_motion_v1",
        "presenter_region_masked": bool(overlay_regions),
        "overlay_region_source": "explicit_metadata" if overlay_regions else "none",
        "mask_regions_normalized": overlay_regions,
        "static_threshold": STATIC_MAIN_DELTA,
        "dynamic_threshold": DYNAMIC_MAIN_DELTA,
        "frame_count": len(paths),
    }
    if len(paths) < 2:
        return {
            **base,
            "status": "not_available",
            "max_main_region_delta": None,
            "max_full_frame_delta": None,
        }
    try:
        from PIL import Image, ImageChops, ImageStat

        frames = []
        for path in paths[:12]:
            with Image.open(path) as image:
                frames.append(image.convert("L").resize((96, 54)).copy())
        main_deltas: list[float] = []
        full_deltas: list[float] = []
        active_tile_fractions: list[float] = []
        background_deltas: list[float] = []
        motion_boxes: list[dict[str, float]] = []
        localized_flags: list[bool] = []
        for left, right in zip(frames, frames[1:]):
            difference = ImageChops.difference(left, right)
            full_delta = float(ImageStat.Stat(difference).mean[0]) / 255
            full_deltas.append(full_delta)
            if overlay_regions:
                main_difference = difference.copy()
                for region in overlay_regions:
                    main_difference.paste(0, _region_box(region, width=96, height=54))
                main_delta = float(ImageStat.Stat(main_difference).mean[0]) / 255
                background_deltas.append(main_delta)
                main_deltas.append(main_delta)
                continue

            grid = _grid_motion(difference, ImageStat)
            active_tile_fractions.append(grid["active_tile_fraction"])
            background_deltas.append(grid["background_delta"])
            if grid["motion_bbox_normalized"]:
                motion_boxes.append(grid["motion_bbox_normalized"])
            localized_flags.append(bool(grid["localized"]))
            main_deltas.append(full_delta)
        main_max = max(main_deltas, default=0.0)
        full_max = max(full_deltas, default=0.0)
        localized_motion = (
            not overlay_regions
            and bool(localized_flags)
            and any(localized_flags)
            and all(
                localized or delta <= STATIC_MAIN_DELTA
                for localized, delta in zip(localized_flags, full_deltas)
            )
            and full_max > STATIC_MAIN_DELTA
        )
        if (
            overlay_regions
            and main_max <= STATIC_MAIN_DELTA
            and full_max > main_max + 0.01
        ):
            status = "explicit_overlay_only"
        elif localized_motion:
            status = "localized_motion"
        elif main_max <= STATIC_MAIN_DELTA:
            status = "static"
        elif main_max >= DYNAMIC_MAIN_DELTA:
            status = "dynamic"
        else:
            status = "low_change"
        return {
            **base,
            "status": status,
            "max_main_region_delta": round(main_max, 6),
            "max_full_frame_delta": round(full_max, 6),
            "max_active_tile_fraction": round(max(active_tile_fractions), 6)
            if active_tile_fractions
            else None,
            "max_background_delta": round(max(background_deltas), 6)
            if background_deltas
            else None,
            "motion_bbox_normalized": _union_regions(motion_boxes),
        }
    except ImportError:
        return {
            **base,
            "status": "not_available",
            "reason": "pillow_not_installed",
            "max_main_region_delta": None,
            "max_full_frame_delta": None,
        }
    except Exception as exc:
        return {
            **base,
            "status": "not_available",
            "reason": f"image_read_failed:{type(exc).__name__}",
            "max_main_region_delta": None,
            "max_full_frame_delta": None,
        }


def _overlay_regions(item: dict[str, Any]) -> list[dict[str, float]]:
    payloads: list[Any] = []
    for key in (
        "presenter_regions",
        "picture_in_picture_regions",
        "overlay_regions",
        "presenter_region",
        "picture_in_picture_region",
        "pip_region",
    ):
        if item.get(key) is not None:
            payloads.append(item[key])
    for parent_key in ("layout", "layout_evidence", "structured_visual"):
        parent = item.get(parent_key)
        if not isinstance(parent, dict):
            continue
        for key in (
            "presenter_regions",
            "picture_in_picture_regions",
            "overlay_regions",
            "presenter_region",
            "pip_region",
        ):
            if parent.get(key) is not None:
                payloads.append(parent[key])
    regions: list[dict[str, float]] = []
    for payload in payloads:
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            region = _normalise_region(value)
            if region and region not in regions:
                regions.append(region)
    return regions


def _normalise_region(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        if all(key in value for key in ("left", "top", "right", "bottom")):
            left, top, right, bottom = (
                float(value[key]) for key in ("left", "top", "right", "bottom")
            )
        elif all(key in value for key in ("x", "y", "width", "height")):
            left, top = float(value["x"]), float(value["y"])
            right, bottom = left + float(value["width"]), top + float(value["height"])
        else:
            return None
    except (TypeError, ValueError):
        return None
    if max(abs(left), abs(top), abs(right), abs(bottom)) > 1:
        return None
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(1.0, right), min(1.0, bottom)
    if right <= left or bottom <= top:
        return None
    return {
        "left": round(left, 6),
        "top": round(top, 6),
        "right": round(right, 6),
        "bottom": round(bottom, 6),
    }


def _region_box(
    region: dict[str, float], *, width: int, height: int
) -> tuple[int, int, int, int]:
    return (
        max(0, min(width, int(region["left"] * width))),
        max(0, min(height, int(region["top"] * height))),
        max(0, min(width, int(round(region["right"] * width)))),
        max(0, min(height, int(round(region["bottom"] * height)))),
    )


def _grid_motion(difference: Any, image_stat: Any) -> dict[str, Any]:
    width, height = difference.size
    active: list[tuple[int, int]] = []
    inactive_values: list[float] = []
    for row in range(MOTION_GRID_ROWS):
        top = row * height // MOTION_GRID_ROWS
        bottom = (row + 1) * height // MOTION_GRID_ROWS
        for column in range(MOTION_GRID_COLUMNS):
            left = column * width // MOTION_GRID_COLUMNS
            right = (column + 1) * width // MOTION_GRID_COLUMNS
            value = (
                float(
                    image_stat.Stat(difference.crop((left, top, right, bottom))).mean[0]
                )
                / 255
            )
            if value >= ACTIVE_TILE_DELTA:
                active.append((column, row))
            else:
                inactive_values.append(value)
    total_tiles = MOTION_GRID_COLUMNS * MOTION_GRID_ROWS
    active_fraction = len(active) / total_tiles
    if not active:
        return {
            "localized": False,
            "active_tile_fraction": 0.0,
            "background_delta": max(inactive_values, default=0.0),
            "motion_bbox_normalized": None,
        }
    columns = [column for column, _ in active]
    rows = [row for _, row in active]
    left, right = min(columns), max(columns) + 1
    top, bottom = min(rows), max(rows) + 1
    bbox_fraction = ((right - left) * (bottom - top)) / total_tiles
    bbox = {
        "left": round(left / MOTION_GRID_COLUMNS, 6),
        "top": round(top / MOTION_GRID_ROWS, 6),
        "right": round(right / MOTION_GRID_COLUMNS, 6),
        "bottom": round(bottom / MOTION_GRID_ROWS, 6),
    }
    return {
        "localized": active_fraction <= LOCALIZED_TILE_FRACTION_MAX
        and bbox_fraction <= LOCALIZED_BBOX_FRACTION_MAX,
        "active_tile_fraction": active_fraction,
        "background_delta": max(inactive_values, default=0.0),
        "motion_bbox_normalized": bbox,
    }


def _union_regions(regions: list[dict[str, float]]) -> dict[str, float] | None:
    if not regions:
        return None
    return {
        "left": min(region["left"] for region in regions),
        "top": min(region["top"] for region in regions),
        "right": max(region["right"] for region in regions),
        "bottom": max(region["bottom"] for region in regions),
    }


def _safe_frame_path(root: Path, value: str) -> Path | None:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _representative_fingerprint(root: Path, frame_paths: list[str]) -> int | None:
    for value in frame_paths:
        path = _safe_frame_path(root, value)
        if path is None:
            continue
        try:
            from PIL import Image

            with Image.open(path) as image:
                gray = image.convert("L")
                pixels = list(gray.resize((9, 8)).getdata())
            bits = 0
            for row in range(8):
                offset = row * 9
                for column in range(8):
                    bits = (bits << 1) | int(
                        pixels[offset + column] > pixels[offset + column + 1]
                    )
            return bits
        except Exception:
            continue
    return None


def _rows_are_near_duplicates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_end = _seconds(left.get("end"))
    right_start = _seconds(right.get("start"))
    if left_end is not None and right_start is not None and right_start - left_end > 30:
        return False
    left_fp, right_fp = (
        left.get("representative_fingerprint"),
        right.get("representative_fingerprint"),
    )
    if (
        isinstance(left_fp, int)
        and isinstance(right_fp, int)
        and (left_fp ^ right_fp).bit_count() <= NEAR_DUPLICATE_DHASH_DISTANCE
    ):
        return True
    left_text = re.sub(r"\s+", "", str(left.get("visual_text_excerpt") or ""))
    right_text = re.sub(r"\s+", "", str(right.get("visual_text_excerpt") or ""))
    return bool(
        len(left_text) >= 8
        and len(right_text) >= 8
        and difflib.SequenceMatcher(None, left_text, right_text).ratio()
        >= NEAR_DUPLICATE_TEXT_SIMILARITY
    )


def _issue_keys(item: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    raw = (
        item.get("quality_issues")
        or item.get("coverage_issues")
        or item.get("issues")
        or []
    )
    if isinstance(raw, dict):
        raw = raw.get("issues") or raw.get("gaps") or raw.get("reasons") or []
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, str):
            values.add(value)
        elif isinstance(value, dict):
            key = value.get("key") or value.get("reason") or value.get("type")
            if key:
                values.add(str(key))
    return values


def _weak_visual_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 8:
        return True
    useful = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact)
    return len(useful) < max(4, len(compact) // 3)


def _seconds(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    parts = str(value).strip().split(":")
    if len(parts) <= 1:
        return None
    try:
        total = 0.0
        for part in parts:
            total = total * 60 + float(part.replace(",", "."))
        return total
    except ValueError:
        return None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
