"""Image-bound OCR block candidates for reference views, separate from strict text.

Reuses the existing OCR result blocks and frame resolver. It neither recognizes
text nor promotes a suspicious whole-frame result into accepted visual_text.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Any

from .file_hash import sha256_file

SCHEMA = "video_knowledge_pipeline.ocr_block_candidates.v1"
ROLES = {"title", "subtitle", "tablet", "background", "unclassified"}
MIN_SCORE = .8


def _bbox(value: Any, width: int, height: int) -> list[float] | None:
    # ebook's axis-aligned bbox is navigation, not a recovered native polygon.
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        return None
    left, top, right, bottom = [float(v) for v in value]
    return [left, top, right, bottom] if 0 <= left < right <= width and 0 <= top < bottom <= height else None


def _role(block: dict, box: list[float] | None, width: int, height: int, row: dict) -> tuple[str, str]:
    explicit = block.get("role") or block.get("text_role")
    if explicit in ROLES:
        return explicit, "declared_ocr_region_candidate_not_human_verified"
    if not box:
        return "unclassified", "missing_navigation_bbox"
    left, top, right, bottom = box
    center_y, box_height = (top + bottom) / (2 * height), (bottom - top) / height
    if box_height >= .055 and (right-left) / width >= .15 and .08 <= center_y <= .72:
        return "title", "large_text_position_heuristic_not_verified_role"
    if center_y >= .78:
        return "subtitle", "bottom_band_heuristic_not_verified_role"
    context = str(row.get("visual_understanding") or "")
    if ("平板" in context or "tablet" in context.lower()) and .3 <= center_y <= .85:
        return "tablet", "existing_visual_context_and_position_heuristic"
    if re.fullmatch(r"[A-Z]{5,}", str(block.get("text") or "").strip()):
        return "background", "latin_mark_candidate_not_verified_role"
    return "unclassified", "no_reliable_role_hint"


def project_ocr_block_candidates(root: str | Path, timeline: list[dict], results: list[dict]) -> list[dict]:
    """Read pixels/hash and project already saved OCR, with no writes or calls.

    A supplied source hash must match the current frame. Without one, the saved
    result's source image bytes must still be available and identical. Engine
    scores remain uncalibrated and cannot establish factual correctness.
    """
    from PIL import Image
    from .ocr_backfill import _first_existing_image

    root = Path(root).resolve()
    rows = deepcopy(timeline)
    by_index = {int(row.get("index") or i): row for i, row in enumerate(rows, 1)}
    if not isinstance(results, list) or any(not isinstance(result, dict) for result in results):
        raise ValueError("OCR results must be objects")
    seen = set()
    for result in results:
        raw_blocks = result.get("ocr_blocks")
        if raw_blocks is None:
            continue
        if (not isinstance(raw_blocks, list) or len(raw_blocks) > 512
                or any(not isinstance(block, dict) for block in raw_blocks)):
            raise ValueError("OCR blocks must be at most 512 objects")
        index = result.get("index")
        if type(index) is not int or index not in by_index or index in seen:
            raise ValueError("OCR block result requires a unique current timeline index")
        seen.add(index)
        row = by_index[index]
        image = _first_existing_image(root, row)
        digest = sha256_file(image) if image else ""
        source = str(result.get("image_path") or result.get("source") or "")
        supplied = str(result.get("source_sha256") or "")
        source_path = Path(source).expanduser() if source else None
        if source_path and not source_path.is_absolute():
            source_path = root / source_path
        observed_digest = supplied or (sha256_file(source_path) if source_path and source_path.is_file() else "")
        bound = bool(digest and observed_digest == digest)
        try:
            if image is None:
                raise ValueError("current frame missing")
            with Image.open(image) as picture:
                width, height = picture.size
        except (OSError, TypeError, ValueError):
            width, height, bound = 0, 0, False
        blocks = []
        for number, block in enumerate(raw_blocks, 1):
            text = str(block.get("text") or "").strip()
            if len(text) > 4000:
                raise ValueError("OCR block text exceeds reference budget")
            score = block.get("score")
            valid_score = type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
            box = _bbox(block.get("bbox"), width, height)
            role, basis = _role(block, box, width, height, row)
            usable = bool(bound and text and valid_score and score >= MIN_SCORE and box)
            blocks.append({"block_index": number, "text": text, "score": float(score) if valid_score else None,
                "bbox": box, "coordinate_system": "source_frame_pixels_axis_aligned_navigation",
                "role": role, "role_basis": basis, "reference_usable": usable,
                "status": "reference_candidate" if usable else "review-required",
                "source_image_sha256": digest, "provider": str(block.get("provider") or ""),
                "candidate_only": True, "human_confirmed": False})
        row["ocr_block_candidates"] = {"schema": SCHEMA, "candidate_only": True,
            "source_verified": bound, "source_image_sha256": digest, "observed_source_sha256": observed_digest,
            "image_width": width, "image_height": height, "source_path": source,
            "strict_frame_status": str(result.get("recognition_status") or "unknown"),
            "strict_result_ok": result.get("ok") is True,
            "minimum_reference_score": MIN_SCORE,
            "score_semantics": "engine_score_not_calibrated_factual_accuracy",
            "formal_visual_text_modified": False, "blocks": blocks}
    return rows


def bind_ocr_candidates_for_display(root: str | Path, timeline: list[dict]) -> list[dict]:
    """Recheck stored candidate image identity before pure report projection."""
    from .ocr_backfill import _first_existing_image

    rows = deepcopy(timeline)
    for row in rows:
        payload = row.get("ocr_block_candidates")
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            continue
        image = _first_existing_image(Path(root).resolve(), row)
        current = sha256_file(image) if image else ""
        valid = bool(current and current == payload.get("source_image_sha256")
                     and current == payload.get("observed_source_sha256"))
        payload["source_verified"] = valid
        if not valid:
            for block in payload.get("blocks", []):
                block["reference_usable"] = False
                block["status"] = "stale_image_binding"
    return rows
