"""Budgeted oral-video OCR region planning over existing saved observations.

No engine or VLM calls occur here. Crop rows deliberately use the existing
screen_text_recovery executor contract, preserving strict unresolved states.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .ocr_block_candidates import bind_ocr_candidates_for_display, project_ocr_block_candidates
from .storage import read_json, write_json

SCHEMA = "video_knowledge_pipeline.oral_ocr_recovery_queue.v1"
ROLE_BOXES = {"title": [.08, .10, .92, .70], "subtitle": [.02, .76, .98, .98],
              "tablet": [.08, .28, .94, .90]}


def build_oral_ocr_recovery_queue(bundle_dir: str | Path, *, results: list[dict] | None = None,
                                  input_json: str | Path | None = None, max_regions: int = 12,
                                  max_attempts: int = 2, indexes: list[int] | None = None,
                                  write: bool = False) -> dict[str, Any]:
    """Select important uncertain regions; write=False performs no mutations."""
    from .ocr_backfill import _first_existing_image

    if type(max_regions) is not int or not 0 <= max_regions <= 100:
        raise ValueError("max_regions must be an integer in 0..100")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be an integer in 1..5")
    if results is not None and input_json is not None:
        raise ValueError("provide saved results or input_json, not both")
    if indexes is not None and (any(type(index) is not int or index < 1 for index in indexes)
                                or len(set(indexes)) != len(indexes)):
        raise ValueError("indexes must be distinct positive integers")
    root = Path(bundle_dir).resolve()
    timeline = read_json(root / "timeline.json")
    if not isinstance(timeline, list) or any(not isinstance(row, dict) for row in timeline):
        raise ValueError("timeline must contain objects")
    if input_json is not None:
        payload = read_json(Path(input_json))
        results = payload.get("items", payload.get("results", [])) if isinstance(payload, dict) else payload
    rows = project_ocr_block_candidates(root, timeline, results) if results is not None else timeline
    rows = bind_ocr_candidates_for_display(root, rows)
    prior_path = root / "oral-ocr-recovery-state.json"
    prior = read_json(prior_path) if prior_path.is_file() else {}
    if not isinstance(prior, dict):
        raise ValueError("oral OCR recovery state must be an object")
    prior_regions = prior.get("regions", {})
    if not isinstance(prior_regions, dict):
        raise ValueError("oral OCR regions must be an object")
    regions, blocked, seen = [], [], set()
    for position, row in enumerate(rows, 1):
        index = int(row.get("index") or position)
        if indexes and index not in indexes:
            continue
        image = _first_existing_image(root, row)
        if not image:
            blocked.append({"index": index, "reason": "image_missing"})
            continue
        digest = sha256_file(image)
        block_payload = row.get("ocr_block_candidates", {})
        if block_payload and not block_payload.get("source_verified"):
            blocked.append({"index": index, "reason": "stale_or_unbound_OCR_source"})
            continue
        blocks = block_payload.get("blocks", [])
        proposed = []
        for block in blocks:
            role = block.get("role")
            if role not in ROLE_BOXES or block.get("reference_usable"):
                continue
            box = block.get("bbox")
            width, height = block_payload.get("image_width", 0), block_payload.get("image_height", 0)
            if not box or not width or not height:
                continue
            # Slight margin around the existing navigation bbox; clipping is
            # bounded to the source image, not a geometric re-identification.
            relative = [max(0, box[0]/width-.015), max(0, box[1]/height-.015),
                        min(1, box[2]/width+.015), min(1, box[3]/height+.015)]
            proposed.append((role, relative, "uncertain_existing_ocr_block"))
        context = str(row.get("visual_understanding") or "").lower()
        for role, words in {"title": ("标题",), "subtitle": ("字幕",), "tablet": ("平板", "tablet")}.items():
            if any(word in context for word in words) and not any(b.get("role") == role for b in blocks):
                proposed.append((role, ROLE_BOXES[role], "visible_region_without_ocr_blocks"))
        for role, box, reason in proposed:
            identity = hashlib.sha256(json.dumps([digest, role, box], sort_keys=True).encode()).hexdigest()[:24]
            if identity in seen:
                continue
            seen.add(identity)
            state = prior_regions.get(identity, {})
            if not isinstance(state, dict) or type(state.get("attempts", 0)) is not int or state.get("attempts", 0) < 0:
                raise ValueError("invalid oral OCR attempt state")
            attempts = state.get("attempts", 0)
            if attempts >= max_attempts or state.get("status") == "resolved":
                blocked.append({"index": index, "region_id": identity,
                    "reason": "attempt_budget_exhausted" if attempts >= max_attempts else "already_resolved"})
                continue
            name = f"{role}-{identity}"
            crop = {"name": name, "source_image": str(image), "box": box,
                "coordinate_system": "relative_xyxy", "planned_output": str(root / "ocr-crops" / "oral" / f"{name}.jpg"),
                "purpose": "重要口播文字区域的限定补救；结果保持机器候选，原严格待核不解除。"}
            regions.append({"index": index, "region_id": identity, "role": role, "reason": reason,
                "image_path": str(image), "image_sha256": digest, "attempts": attempts,
                "max_attempts": max_attempts, "candidate_only": True, "crop": crop})
    regions.sort(key=lambda row: ({"title": 0, "tablet": 1, "subtitle": 2}[row["role"]], row["index"], row["region_id"]))
    selected = regions[:max_regions]
    candidates = []
    for region in selected:
        candidates.append({"index": region["index"], "image_path": region["image_path"], "image_exists": True,
            "region_id": region["region_id"], "source_sha256": region["image_sha256"],
            "screen_text_recovery": {"strategy": "oral_region_crop", "crop_candidates": [region["crop"]]}})
    result = {"schema": SCHEMA, "bundle_dir": str(root), "profile": "talking_head", "plan_only": True,
        "candidate_only": True, "max_regions": max_regions, "max_attempts": max_attempts,
        "eligible_regions": len(regions), "deferred_regions": max(0, len(regions)-len(selected)),
        "selected_regions": selected, "candidates": candidates, "blocked": blocked,
        "operator_boundary": {"model_calls_made": 0, "network_calls_made": 0,
            "strict_visual_text_modified": False, "attempts_incremented": False}}
    if write:
        write_json(root / "oral-ocr-recovery-queue.json", result)
    return result
