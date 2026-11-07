from __future__ import annotations

import hashlib
import csv
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .captiocr_resolver import (
    captiocr_candidate_roots,
    resolve_captiocr_root,
    resolve_tesseract_runtime,
)
from .file_hash import sha256_file
from .frame_recapture import _coverage_audit, _quality_audit
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .repair_status import build_repair_status
from .storage import read_json, write_json, write_text_atomic
from .ocr_inventory import persist_ocr_inventory, record_ocr_result

OCR_GAP_ISSUES = {
    "missing_visual_text",
    "structured_visual_without_ocr",
    "missing_ocr",
    "low_ocr_confidence",
    "ocr_text_empty",
    "screen_text_low_confidence",
}


def run_ocr_backfill(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    execute: bool = False,
    language: str = "eng",
    captiocr_root: str | Path | None = None,
    limit: int = 0,
    apply_mode: str = "merge",
    engine: str = "auto",
) -> dict[str, Any]:
    """Preview, import, or execute OCR backfill for a WebUI lecture bundle."""
    if apply_mode not in {"merge", "replace_snapshot"}:
        raise ValueError("apply_mode must be merge or replace_snapshot")
    if engine not in {"auto", "rapidocr"}:
        raise ValueError("engine must be auto or rapidocr")
    if apply_mode == "replace_snapshot" and int(limit or 0) > 0:
        raise ValueError("replace_snapshot requires full candidate coverage; limit must be 0")
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    timeline = _read_timeline(root)
    inventory = persist_ocr_inventory(root, timeline)
    candidates = _ocr_candidates(root, timeline, include_existing=apply_mode == "replace_snapshot" or input_json is not None)
    if execute and apply_mode == "merge":
        # Finish never-run frames before retries; successful empty output is a
        # recovery task, not an indefinitely repeated whole-frame OCR task.
        states = {row["index"]: row for row in inventory["items"]}
        known = {c["index"] for c in candidates}
        candidates.extend(c for c in _ocr_candidates(root, timeline, include_existing=True)
                          if c["index"] not in known and states[c["index"]]["execution_status"] in {"not_run", "failed", "blocked", "running"})
        candidates = [c for c in candidates if states[c["index"]]["execution_status"] != "completed"
                      or (states[c["index"]]["recognition_status"] == "empty" and not states[c["index"]].get("empty_recovery_attempted"))
                      or (engine == "rapidocr" and states[c["index"]].get("engine") != "ebook_rapidocr"
                          and states[c["index"]]["recognition_status"] == "suspicious")]
        candidates.sort(key=lambda c: (states[c["index"]]["attempts"] > 0, c["index"]))
    all_candidates = list(candidates)
    if limit and int(limit) > 0:
        candidates = candidates[: int(limit)]
    template_path = _write_ocr_input_template(root, candidates)
    screen_text_recovery = _screen_text_recovery_plan(root, candidates)
    imported = input_json is not None
    imported_entries = _read_ocr_input(input_json) if imported else []
    results: list[dict[str, Any]]
    runner = {"available": False, "name": "captiocr", "error": ""}
    captiocr = resolve_captiocr_root(captiocr_root)

    if imported:
        results = _results_from_imported_entries(imported_entries, candidates)
    elif execute:
        # Checkpoint each frame. A killed process loses at most its active frame,
        # and every receipt precedes progression to the next frame.
        results = []
        for candidate in candidates:
            if engine == "rapidocr":
                runner, batch = _run_rapidocr_candidates([candidate])
            else:
                runner, batch = _run_captiocr_candidates([candidate], language=language, captiocr_root=captiocr_root)
            for result in batch:
                if not runner.get("available"):
                    result["executed"] = False
                if apply_mode == "merge":
                    _backfill_ocr_results(root, manifest, timeline, [result], candidates=[candidate], apply_mode="merge", input_json=None)
                record_ocr_result(root, timeline, result, engine=str(runner.get("name") or ""), language=language)
                results.append(result)
            if not runner.get("available"):
                break
    else:
        results = [_candidate_result(candidate, executed=False, ok=False) for candidate in candidates]

    backfill = _backfill_ocr_results(
        root,
        manifest,
        timeline,
        results,
        candidates=candidates,
        apply_mode=apply_mode,
        input_json=input_json,
    ) if imported or execute else {
        "updated": 0,
        "updated_indexes": [],
        "cleared": 0,
        "cleared_indexes": [],
        "preserved_indexes": [],
        "applied": False,
        "source_package_updated": False,
    }
    timeline = _read_timeline(root)
    if imported:
        for result in results:
            if result.get("authoritative_observation") or result.get("ok"):
                record_ocr_result(root, timeline, result, engine="imported_evidence", language=language)
    inventory = persist_ocr_inventory(root, timeline)
    tesseract = (
        dict(runner.get("tesseract"))
        if isinstance(runner.get("tesseract"), dict)
        else resolve_tesseract_runtime(required_languages=language)
    )
    capabilities = _ocr_capabilities(captiocr, runner, tesseract, language=language)
    status, ok = _ocr_result_status(
        results,
        execute=execute,
        imported=imported,
        capabilities=capabilities,
    )
    if apply_mode == "replace_snapshot":
        if backfill.get("applied"):
            status, ok = "replace_snapshot_applied", True
        else:
            status, ok = "replace_snapshot_incomplete", False
    elif execute and inventory["summary"]["remaining"] and not inventory["summary"]["failed"] and not inventory["summary"]["blocked"]:
        status, ok = "batch_completed_pending", False
    elif execute and not inventory["summary"]["remaining"] and inventory["empty_indexes"]:
        status, ok = "empty_results_need_recovery", False
    elif execute and inventory["summary"]["suspicious"]:
        status, ok = "suspicious_results_need_review", False
    recovery_commands = _ocr_recovery_commands(capabilities)
    summary = {
        "total": len(results),
        "execute": execute,
        "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
        "language": language,
        "limit": int(limit or 0),
        "apply_mode": apply_mode,
        "engine_requested": engine,
        "succeeded": sum(1 for item in results if item.get("ok") and str(item.get("text") or "").strip()),
        "failed": sum(1 for item in results if item.get("executed") and not item.get("ok") and not item.get("authoritative_observation")),
        "blocked": sum(1 for item in results if not item.get("executed") and item.get("stderr")),
        "empty": sum(1 for item in results if item.get("authoritative_observation") and not str(item.get("text") or "").strip()),
        "whole_bundle": inventory["summary"],
        "pending_indexes": inventory["pending_indexes"],
        "resume_argv": ["run-ocr-backfill", str(root), "--execute", "--language", language,
                        "--limit", str(int(limit or 0)), "--engine", engine, "--apply-mode", apply_mode]
                       + (["--captiocr-root", str(captiocr_root)] if captiocr_root else []),
        "planned": sum(1 for item in results if not item.get("executed")),
        "cleared": int(backfill.get("cleared") or 0),
        "updated_at": now_iso(),
        "input_template_json": str(template_path),
        "overwrite_receipt_path": str(backfill.get("overwrite_receipt_path") or ""),
        "status": status,
        "ok": ok,
    }
    manifest["coverage"] = _coverage_audit(timeline)
    manifest["ocr_backfill"] = {
        "schema": "lecture_ocr_backfill.v1",
        "count": len(all_candidates),
        "language": language,
        "captiocr": captiocr,
        "runner": runner,
        "capabilities": capabilities,
        "recovery_commands": recovery_commands,
        "items": all_candidates,
        "inventory_path": "ocr-execution-inventory.json",
        "whole_bundle": inventory["summary"],
        "last_run": summary,
        "last_backfill": {**backfill, "updated_at": now_iso()},
        "overwrite_receipt_path": str(backfill.get("overwrite_receipt_path") or ""),
        "input_template_json": str(template_path),
        "screen_text_recovery": screen_text_recovery,
    }
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    write_json(manifest_path, manifest)

    report_path = root / "ocr-backfill-report.md"
    write_text_atomic(report_path,
        _render_ocr_backfill_report(
            root,
            results,
            summary,
            runner,
            template_path,
            screen_text_recovery,
            capabilities,
            recovery_commands,
        ),
        encoding="utf-8",
    )
    handoff_path = root / "ocr-backfill-handoff.md"
    handoff_json_path = root / "ocr-backfill-handoff.json"
    handoff = _build_ocr_backfill_handoff(
        root,
        manifest,
        results,
        summary,
        runner,
        captiocr,
        screen_text_recovery,
        template_path,
        report_path,
        handoff_path,
        handoff_json_path,
        capabilities,
        recovery_commands,
    )
    write_json(handoff_json_path, handoff)
    write_text_atomic(handoff_path, _render_ocr_backfill_handoff_markdown(handoff), encoding="utf-8")
    manifest["ocr_backfill"]["handoff_markdown"] = str(handoff_path)
    manifest["ocr_backfill"]["handoff_json"] = str(handoff_json_path)
    write_json(manifest_path, manifest)
    return {
        "ok": ok,
        "status": status,
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "handoff_path": str(handoff_path),
        "handoff_json_path": str(handoff_json_path),
        "input_template_json": str(template_path),
        "summary": summary,
        "runner": runner,
        "captiocr": captiocr,
        "capabilities": capabilities,
        "recovery_commands": recovery_commands,
        "backfill": backfill,
        "apply_mode": apply_mode,
        "overwrite_receipt_path": str(backfill.get("overwrite_receipt_path") or ""),
        "items": results,
        "screen_text_recovery": screen_text_recovery,
    }


def _read_timeline(root: Path) -> list[dict[str, Any]]:
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        return []
    timeline = read_json(timeline_path)
    return [item for item in timeline if isinstance(item, dict)] if isinstance(timeline, list) else []


def _ocr_candidates(
    root: Path,
    timeline: list[dict[str, Any]],
    *,
    include_existing: bool = False,
) -> list[dict[str, Any]]:
    candidates = []
    for index, item in enumerate(timeline, start=1):
        issues = _quality_issues(item)
        inferred = _inferred_screen_text_issues(item)
        issues = _dedupe([*issues, *inferred])
        has_visual_text = bool(str(item.get("visual_text") or "").strip())
        image_path = _first_existing_image(root, item)
        if include_existing and not image_path:
            continue
        if not image_path and not (set(issues) & OCR_GAP_ISSUES):
            continue
        if has_visual_text and not (set(issues) & OCR_GAP_ISSUES) and not include_existing:
            continue
        candidates.append(
            {
                "index": int(item.get("index") or index),
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "material_types": item.get("material_types", []),
                "quality_issues": issues,
                "image_path": str(image_path) if image_path else "",
                "image_exists": bool(image_path and image_path.exists()),
                "visual_text": item.get("visual_text", ""),
                "visual_route": item.get("visual_route", ""),
                "signals": item.get("signals", []),
                "recovery": _screen_text_recovery_for_item(root, item, issues, image_path),
            }
        )
    return candidates


def _first_existing_image(root: Path, item: dict[str, Any]) -> Path | None:
    # A successful recapture is the current representative frame. Retain old
    # assets for audit, but do not repeatedly OCR the old blurry image first.
    paths = [str(item["recaptured_frame_path"])] if item.get("recaptured_frame_path") else []
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        candidate = str(asset.get("source") or asset.get("resolved_path") or asset.get("path") or "")
        if candidate:
            paths.append(candidate)
    paths.extend(str(path) for path in item.get("frame_paths") or [] if path)
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            return path.resolve()
    return None


def _screen_text_recovery_plan(root: Path, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    counts: dict[str, int] = {}
    for candidate in candidates:
        recovery = candidate.get("recovery") if isinstance(candidate.get("recovery"), dict) else {}
        strategy = str(recovery.get("strategy") or "unknown")
        counts[strategy] = counts.get(strategy, 0) + 1
        items.append(
            {
                "index": candidate.get("index"),
                "strategy": strategy,
                "recommended_tool": recovery.get("recommended_tool", ""),
                "reason": recovery.get("reason", ""),
                "image_path": candidate.get("image_path", ""),
                "crop_candidates": recovery.get("crop_candidates", []),
            }
        )
    return {
        "schema": "lecture_screen_text_recovery_plan.v1",
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "strategy_counts": counts,
        "items": items,
        "notes": [
            "This is a planning layer only; run_ocr_backfill preview does not crop images or execute OCR.",
            "document_visual and mixed screenshots should first use ebook_markdown_pipeline through run_visual_structure_plan.",
            "small UI text and software/browser/editor screenshots should use crop-and-OCR only as a targeted recovery step, then keep the original frame evidence.",
            "When OCR cannot be trusted, use multimodal visual description or human review/keep-image instead of pretending screen text is covered.",
        ],
    }


def _screen_text_recovery_for_item(root: Path, item: dict[str, Any], issues: list[str], image_path: Path | None) -> dict[str, Any]:
    route = str(item.get("visual_route") or "")
    material_types = {str(value) for value in item.get("material_types") or []}
    signals = {str(value) for value in item.get("signals") or []}
    low_confidence = "screen_text_low_confidence" in issues
    has_image = bool(image_path and image_path.exists())
    if route == "document_visual":
        strategy = "ebook_pipeline"
        tool = "ebook_markdown_pipeline"
        reason = "图文型截图应优先用 ebook_markdown_pipeline 做版面、表格、公式、代码和文字结构化。"
    elif route == "mixed":
        strategy = "ebook_pipeline_plus_multimodal"
        tool = "ebook_markdown_pipeline + multimodal_frame_analyzer"
        reason = "混合画面既可能有文档/表格/代码，也可能有界面状态或动作；图文解析和多模态理解都应保留。"
    elif low_confidence or _is_ui_like_item(route, material_types, signals):
        strategy = "crop_and_ocr"
        tool = "crop plan + CaptiOCR/Tesseract or ebook_markdown_pipeline on crops"
        reason = "软件界面、浏览器、编辑器或小 UI 文本容易被全帧 OCR 漏掉，应先裁剪主体区域再识别。"
    elif route in {"semantic_frame", "temporal_sequence"}:
        strategy = "multimodal_text_description"
        tool = "multimodal_frame_analyzer"
        reason = "非图文主导画面更需要多模态描述屏幕状态；只在存在可读文字区域时补 OCR。"
    else:
        strategy = "human_review_keep_image"
        tool = "human review"
        reason = "无法稳定判断文字区域；保留原图证据并人工决定是否转写。"
    return {
        "strategy": strategy,
        "recommended_tool": tool,
        "reason": reason,
        "has_image": has_image,
        "crop_candidates": _crop_candidates_for_item(root, item, image_path) if has_image else [],
        "issues": issues,
    }


def _inferred_screen_text_issues(item: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if _is_wrapper_only_visual_text(item):
        issues.append("ocr_text_empty")
    route = str(item.get("visual_route") or "")
    material_types = {str(value) for value in item.get("material_types") or []}
    signals = {str(value) for value in item.get("signals") or []}
    if _is_ui_like_item(route, material_types, signals) and not _has_human_screen_text_acceptance(item):
        issues.append("screen_text_low_confidence")
    return issues


def _is_ui_like_item(route: str, material_types: set[str], signals: set[str]) -> bool:
    ui_tokens = {
        "ui",
        "software",
        "browser",
        "editor",
        "code_editor",
        "terminal",
        "screen",
        "interface",
        "app",
        "operation",
        "mouse",
        "workflow",
    }
    return route in {"semantic_frame", "temporal_sequence", "mixed"} and bool((material_types | signals) & ui_tokens)


def _is_wrapper_only_visual_text(item: dict[str, Any]) -> bool:
    visual_text = str(item.get("visual_text") or "").strip()
    if not visual_text:
        return False
    meaningful = []
    stems = _frame_stems(item)
    for line in visual_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->") and "source:" in stripped.lower():
            continue
        if stripped.startswith("# ") and stripped[2:].strip() in stems:
            continue
        meaningful.append(stripped)
    return not meaningful


def _has_human_screen_text_acceptance(item: dict[str, Any]) -> bool:
    review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    status = str(item.get("review_status") or review.get("status") or "").lower()
    return status in {"accepted", "reviewed", "keep_image", "corrected_visual_text"} or bool(item.get("human_corrected_visual_text") or review.get("corrected_visual_text"))


def _crop_candidates_for_item(root: Path, item: dict[str, Any], image_path: Path | None) -> list[dict[str, Any]]:
    source = str(image_path or "")
    index = int(item.get("index") or 0)
    crop_dir = root / "ocr-crops" / f"timeline-{index:04d}"
    base = [
        ("full_frame", [0.0, 0.0, 1.0, 1.0], "保留全帧，作为裁剪和人工复核的基准。"),
        ("central_content", [0.08, 0.08, 0.92, 0.86], "优先覆盖课件、浏览器主体、编辑器主体，避开边缘 UI。"),
        ("without_subtitle_band", [0.0, 0.0, 1.0, 0.82], "排除底部字幕/播放器控制栏，减少字幕污染。"),
        ("browser_or_editor_body", [0.05, 0.12, 0.95, 0.88], "针对浏览器、IDE、文档编辑器主体区域的小字。"),
        ("exclude_instructor_pip", [0.0, 0.0, 0.78, 0.88], "避开右下/侧边讲师画中画，保留屏幕主体。"),
    ]
    return [
        {
            "name": name,
            "source_image": source,
            "box": box,
            "coordinate_system": "relative_xyxy",
            "planned_output": str(crop_dir / f"{name}.jpg"),
            "purpose": purpose,
        }
        for name, box, purpose in base
    ]


def _frame_stems(item: dict[str, Any]) -> set[str]:
    stems: set[str] = set()
    for key in ("frame_paths", "temporal_frame_paths"):
        values = item.get(key)
        if isinstance(values, list):
            for value in values:
                text = str(value or "").strip()
                if text:
                    stems.add(Path(text).stem)
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            for key in ("path", "source"):
                text = str(asset.get(key) or "").strip()
                if text:
                    stems.add(Path(text).stem)
    return stems


def _recovery_strategy_label(strategy: str) -> str:
    labels = {
        "ebook_pipeline": "图文型截图，优先复用 ebook_markdown_pipeline。",
        "ebook_pipeline_plus_multimodal": "混合画面，同时需要图文解析和多模态画面理解。",
        "crop_and_ocr": "小 UI 文本或界面文字，先规划裁剪再 OCR。",
        "multimodal_text_description": "非纯文字画面，优先用多模态描述屏幕状态。",
        "human_review_keep_image": "不确定或不可降维内容，保留图片并人工复核。",
        "unknown": "未分类。",
    }
    return labels.get(strategy, "")


def _read_ocr_input(input_json: str | Path) -> list[dict[str, Any]]:
    path = Path(input_json).expanduser().resolve()
    data = read_json(path)
    if isinstance(data, dict):
        rows = data.get("items") or data.get("results") or data.get("ocr") or []
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError("OCR input JSON must be a list or an object with items/results/ocr")
    return [row for row in rows if isinstance(row, dict)]


def _write_ocr_input_template(root: Path, candidates: list[dict[str, Any]]) -> Path:
    path = root / "ocr-backfill-input-template.json"
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "index": candidate.get("index"),
                "start": candidate.get("start", 0),
                "end": candidate.get("end", 0),
                "text": "",
                "source": candidate.get("image_path", ""),
                "image_path": candidate.get("image_path", ""),
                "image_exists": bool(candidate.get("image_exists")),
                "material_types": candidate.get("material_types", []),
                "quality_issues": candidate.get("quality_issues", []),
                "screen_text_recovery": candidate.get("recovery", {}),
                "crop_candidates": (candidate.get("recovery") or {}).get("crop_candidates", []),
                "current_visual_text": candidate.get("visual_text", ""),
                "notes": "",
            }
        )
    write_json(
        path,
        {
            "schema": "lecture_ocr_backfill_input.v1",
            "generated_at": now_iso(),
            "bundle_dir": str(root),
            "instructions": "Fill text with corrected OCR/manual visual text, then pass this JSON to run-ocr-backfill --input-json.",
            "items": rows,
        },
    )
    return path


def _results_from_imported_entries(entries: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_indexes = {_int_value(candidate.get("index")) for candidate in candidates}
    candidate_by_index = {_int_value(row.get("index")): row for row in candidates}
    results = []
    for entry in entries:
        index = _int_value(entry.get("index") or entry.get("timeline_index"))
        text = str(entry.get("text") or entry.get("visual_text") or entry.get("ocr_text") or "").strip()
        source = str(entry.get("source") or "imported_json")
        notes = str(entry.get("notes") or "")
        requested_sha = str(entry.get("source_sha256") or "")
        expected_image = Path(str(candidate_by_index.get(index, {}).get("image_path") or ""))
        image_identity_ok = not requested_sha or (expected_image.is_file() and sha256_file(expected_image) == requested_sha)
        wrapper_only = _imported_text_is_wrapper_only(text, source)
        needs_review = _imported_text_requires_human_review(source, notes)
        results.append(
            {
                "index": index,
                "text": "" if wrapper_only else text,
                "source": source,
                "text_role": entry.get("text_role") if entry.get("text_role") in ("incidental_background", "transcript_support") else "unclassified",
                "executed": False,
                "ok": bool(index and text and not wrapper_only and not needs_review and image_identity_ok),
                "candidate": index in candidate_indexes,
                "stderr": (
                    "OCR source image hash mismatch" if not image_identity_ok else "wrapper-only OCR text"
                    if wrapper_only
                    else (
                        "crop OCR requires human review before clearing screen text gap"
                        if needs_review
                        else ("" if index else "missing timeline index")
                    )
                ),
                "authoritative_observation": bool(
                    index and index in candidate_indexes and not wrapper_only and not needs_review and image_identity_ok
                ),
            }
        )
        if isinstance(entry.get("ocr_blocks"), list):
            # Reusing an engine receipt must not reinterpret suspicious raw
            # output as a human-corrected import or a newly executed OCR call.
            cached = results[-1]
            source_image = Path(str(entry.get("image_path") or source))
            observed = requested_sha or (sha256_file(source_image) if source_image.is_file() else "")
            bound = bool(observed and expected_image.is_file() and sha256_file(expected_image) == observed)
            cached.update(ocr_blocks=entry["ocr_blocks"], image_path=str(source_image),
                source_sha256=observed, cached_observation=True,
                recognition_status=str(entry.get("recognition_status") or "unknown"),
                ocr_confidence=entry.get("ocr_confidence", {}),
                ok=bool(cached["ok"] and entry.get("ok") is True and bound),
                authoritative_observation=bool(bound and not needs_review and not wrapper_only),
                stderr="OCR source image hash mismatch" if not bound else cached["stderr"])
    return results


def _imported_text_requires_human_review(source: str, notes: str) -> bool:
    source_text = str(source or "").replace("\\", "/").lower()
    notes_text = str(notes or "").lower()
    return "ocr-crops/" in source_text or "screen_text_recovery crop ocr" in notes_text


def _imported_text_is_wrapper_only(text: str, source: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    source_stems = {Path(part.strip()).stem for part in str(source or "").split(";") if part.strip()}
    meaningful = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("<!--") and lower.endswith("-->"):
            continue
        if line.startswith("![") and "](" in line and line.endswith(")"):
            continue
        if line.startswith("# ") and (not source_stems or line[2:].strip() in source_stems):
            continue
        meaningful.append(line)
    return not meaningful


def _run_rapidocr_candidates(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse the existing ebook OCR owner; never install or download models."""
    from .path_defaults import workspace_root
    owner = workspace_root() / "ebook_markdown_pipeline"
    runner: dict[str, Any] = {"name": "ebook_rapidocr", "available": False,
                              "owner_root": str(owner), "device": "cpu", "offline": True}
    try:
        if not owner.is_dir():
            raise FileNotFoundError(f"ebook_markdown_pipeline not found: {owner}")
        parent = str(owner.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from ebook_markdown_pipeline.ocr_providers import (  # type: ignore
            create_rapidocr_engine, rapidocr_engine_context, normalize_rapidocr_blocks,
        )
        output = []
        with rapidocr_engine_context(create_rapidocr_engine(device="cpu", offline=True)) as reader:
            runner["available"] = True
            for candidate in candidates:
                blocks = normalize_rapidocr_blocks(reader(str(candidate["image_path"])))
                low = [block for block in blocks if float(block.get("score") or 0) < .8]
                output.append({**_candidate_result(candidate, executed=True, ok=bool(blocks) and not low),
                    "text": "\n".join(str(block.get("text") or "") for block in blocks),
                    "source": str(candidate["image_path"]), "ocr_blocks": blocks,
                    "recognition_status": "suspicious" if low else ("text_available" if blocks else "empty"),
                    "authoritative_observation": True,
                    "ocr_confidence": {"minimum_block_score": .8, "low_confidence_block_count": len(low),
                        "block_count": len(blocks), "semantics": "engine_score_not_calibrated_factual_accuracy"}})
        return runner, output
    except Exception as exc:
        runner["error"] = f"rapidocr_offline_preflight_or_execution_failed: {exc}"
        return runner, [{**_candidate_result(row, executed=runner["available"], ok=False),
                         "stderr": runner["error"]} for row in candidates]


def _run_captiocr_candidates(
    candidates: list[dict[str, Any]],
    *,
    language: str,
    captiocr_root: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(_requested_languages(language)) > 1:
        tesseract = resolve_tesseract_runtime(required_languages=language)
        route_reason = "multilingual_request_requires_tesseract_cli"
        if not tesseract.get("language_ready") or not tesseract.get("cmd"):
            status = str(tesseract.get("status") or "runtime_unavailable")
            missing = ",".join(str(item) for item in tesseract.get("missing_languages") or [])
            detail = f"{status}: {missing}" if missing else status
            runner = {
                "available": False,
                "name": "tesseract_cli",
                "tesseract": tesseract,
                "route_reason": route_reason,
                "fallback_from": "",
                "error": detail,
            }
            return runner, [
                {**_candidate_result(candidate, executed=True, ok=False), "stderr": detail}
                for candidate in candidates
            ]
        return _run_tesseract_cli_candidates(
            candidates,
            language=language,
            tesseract=tesseract,
            previous_error="",
            route_reason=route_reason,
        )
    runner, processor, image_cls = _load_captiocr_runner(captiocr_root, language=language)
    if not runner.get("available"):
        tesseract = (
            runner.get("tesseract")
            if isinstance(runner.get("tesseract"), dict)
            else resolve_tesseract_runtime(required_languages=language)
        )
        if not tesseract.get("language_ready"):
            status = str(tesseract.get("status") or "runtime_unavailable")
            missing = ",".join(str(item) for item in tesseract.get("missing_languages") or [])
            detail = f"{status}: {missing}" if missing else status
            runner = {**runner, "tesseract": tesseract, "error": detail}
            return runner, [
                {
                    **_candidate_result(candidate, executed=True, ok=False),
                    "stderr": detail,
                }
                for candidate in candidates
            ]
        if tesseract.get("cmd"):
            return _run_tesseract_cli_candidates(candidates, language=language, tesseract=tesseract, previous_error=str(runner.get("error") or ""))
        return runner, [
            {
                **_candidate_result(candidate, executed=True, ok=False),
                "stderr": runner.get("error", "CaptiOCR runner unavailable"),
            }
            for candidate in candidates
        ]

    results = []
    for candidate in candidates:
        result = _candidate_result(candidate, executed=True, ok=False)
        image_path = Path(str(candidate.get("image_path") or ""))
        if not image_path.is_file():
            result["stderr"] = "image path does not exist"
            result["executed"] = False
            results.append(result)
            continue
        try:
            image = image_cls.open(image_path)
            image = processor.optimize_image_for_ocr(image)
            text = processor.process_image(image, lang_code=language, caption_mode=False)
            result["text"] = str(text or "").strip()
            result["ok"] = bool(result["text"])
            result["authoritative_observation"] = True
        except Exception as exc:  # pragma: no cover - depends on optional local OCR stack
            result["stderr"] = str(exc)
        results.append(result)
    return runner, results


def _run_tesseract_cli_candidates(
    candidates: list[dict[str, Any]],
    *,
    language: str,
    tesseract: dict[str, Any],
    previous_error: str,
    route_reason: str = "captiocr_unavailable_fallback",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cmd = str(tesseract.get("cmd") or "")
    runner = {
        "available": bool(cmd),
        "name": "tesseract_cli",
        "tesseract": tesseract,
        "fallback_from": "captiocr" if previous_error else "",
        "fallback_error": previous_error,
        "route_reason": route_reason,
        "error": "" if cmd else "Tesseract command not found",
    }
    results = []
    for candidate in candidates:
        result = _candidate_result(candidate, executed=True, ok=False)
        image_path = Path(str(candidate.get("image_path") or ""))
        if not image_path.is_file():
            result["stderr"] = "image path does not exist"
            result["executed"] = False
            results.append(result)
            continue
        command = [cmd, str(image_path), "stdout", "-l", language, "tsv"]
        process_env = dict(os.environ)
        tessdata_prefix = str(tesseract.get("tessdata_prefix") or "").strip()
        if tessdata_prefix:
            process_env["TESSDATA_PREFIX"] = tessdata_prefix
        try:
            completed = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                env=process_env,
                timeout=120,
            )
        except Exception as exc:  # pragma: no cover - depends on optional local OCR stack
            result["stderr"] = str(exc)
            results.append(result)
            continue
        text, confidence = _tesseract_tsv_observation(str(completed.stdout or ""))
        result["ocr_attempts"] = [{"page_segmentation_mode": 3, "returncode": completed.returncode, "text_chars": len(text)}]
        if completed.returncode == 0 and confidence["valid_tsv"] and not text:
            # Tesseract's automatic page segmentation can miss large text on
            # mixed video/PPT layouts. Reuse its block and sparse-text modes.
            result["empty_recovery_attempted"] = True
            for mode in (6, 11):
                try:
                    retry = subprocess.run(command[:-1] + ["--psm", str(mode), "tsv"], text=True, encoding="utf-8", errors="replace",
                                           capture_output=True, check=False, env=process_env, timeout=120)
                    recovered, retry_confidence = _tesseract_tsv_observation(str(retry.stdout or ""))
                    result["ocr_attempts"].append({"page_segmentation_mode": mode, "returncode": retry.returncode, "text_chars": len(recovered)})
                    if retry.returncode == 0 and recovered:
                        completed, text, confidence = retry, recovered, retry_confidence
                        # A different segmentation mode can hallucinate text
                        # from texture. Keep the observation, not an automatic
                        # replacement of verified screen text.
                        result["recognition_status"] = "suspicious"
                        break
                except (OSError, subprocess.TimeoutExpired) as exc:
                    result["ocr_attempts"].append({"page_segmentation_mode": mode, "error": str(exc)})
        result["text"] = text
        result["ocr_confidence"] = confidence
        if not confidence["valid_tsv"] or (text and not confidence["quality_threshold_passed"]):
            result["recognition_status"] = "suspicious"
        result["ok"] = completed.returncode == 0 and bool(text) and result.get("recognition_status") != "suspicious"
        result["authoritative_observation"] = completed.returncode == 0 and confidence["valid_tsv"]
        result["stderr"] = str(completed.stderr or "").strip() if completed.returncode != 0 else ""
        if completed.returncode == 0 and not confidence["valid_tsv"]:
            result["stderr"] = "tesseract_invalid_tsv_output"
        results.append(result)
    return runner, results


def _tesseract_tsv_observation(output: str) -> tuple[str, dict[str, Any]]:
    """Read native Tesseract word confidence; process success is not text quality."""
    reader = csv.DictReader(io.StringIO(output), delimiter="\t")
    valid = {"level", "conf", "text"}.issubset(reader.fieldnames or [])
    lines: dict[tuple[str, ...], list[str]] = {}
    weighted, weight, invalid_confidence = 0.0, 0, False
    words = 0
    for row in reader if valid else []:
        text = str(row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        words += 1
        key = tuple(str(row.get(k) or "") for k in ("page_num", "block_num", "par_num", "line_num"))
        lines.setdefault(key, []).append(text)
        try:
            score = float(row.get("conf") or "")
        except (TypeError, ValueError):
            score = -1.0
        if not 0 <= score <= 100:
            invalid_confidence = True
            continue
        weighted += score * len(text)
        weight += len(text)
    mean = weighted / weight if weight else None
    return "\n".join(" ".join(values) for values in lines.values()), {
        "valid_tsv": valid, "word_count": words, "mean_confidence": mean,
        "minimum_mean_confidence": 70.0,
        "quality_threshold_passed": bool(valid and not invalid_confidence and mean is not None and mean >= 70.0),
        "semantics": "engine_score_not_calibrated_factual_accuracy",
    }


def _load_captiocr_runner(
    captiocr_root: str | Path | None,
    *,
    language: str = "eng",
) -> tuple[dict[str, Any], Any, Any]:
    resolved = resolve_captiocr_root(captiocr_root)
    tesseract = resolve_tesseract_runtime(required_languages=language)
    roots = [Path(root) for root in captiocr_candidate_roots(captiocr_root)]
    for root in roots:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        import captiocr.core.ocr as ocr_module
        from captiocr.core.ocr import OCRProcessor
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional local OCR stack
        return {
            "available": False,
            "name": "captiocr",
            "root": str(resolved.get("root") or ""),
            "tesseract": tesseract,
            "checked_paths": list(resolved.get("checked") or []),
            "error": str(exc),
        }, None, None
    if tesseract.get("cmd"):
        ocr_module.TESSERACT_CMD = str(tesseract.get("cmd"))
    if tesseract.get("tessdata_prefix"):
        ocr_module.TESSDATA_PREFIX = str(tesseract.get("tessdata_prefix"))
    try:
        processor = OCRProcessor()
    except Exception as exc:  # pragma: no cover - depends on optional local OCR stack
        return {
            "available": False,
            "name": "captiocr",
            "root": str(resolved.get("root") or ""),
            "tesseract": tesseract,
            "checked_paths": list(resolved.get("checked") or []),
            "error": str(exc),
        }, None, None
    available = bool(processor.is_tesseract_available())
    return {
        "available": available,
        "name": "captiocr",
        "root": str(resolved.get("root") or ""),
        "tesseract": tesseract,
        "evidence": list(resolved.get("evidence") or []),
        "checked_paths": list(resolved.get("checked") or []),
        "error": "" if available else "Tesseract is not initialized",
    }, processor, Image


def _candidate_result(candidate: dict[str, Any], *, executed: bool, ok: bool) -> dict[str, Any]:
    return {
        "index": candidate.get("index"),
        "image_path": candidate.get("image_path", ""),
        "executed": executed,
        "ok": ok,
        "image_exists": bool(candidate.get("image_exists")),
        "text": "",
        "stderr": "",
        "screen_text_recovery": candidate.get("recovery", {}),
    }


def _backfill_ocr_results(
    root: Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    candidates: list[dict[str, Any]],
    apply_mode: str,
    input_json: str | Path | None,
) -> dict[str, Any]:
    candidate_indexes = sorted(
        {_int_value(item.get("index")) for item in candidates if _int_value(item.get("index"))}
    )
    authoritative_indexes = sorted(
        {
            _int_value(item.get("index"))
            for item in results
            if item.get("authoritative_observation")
            and _int_value(item.get("index")) in candidate_indexes
        }
    )
    missing_indexes = sorted(set(candidate_indexes) - set(authoritative_indexes))
    coverage_complete = not missing_indexes
    apply_allowed = apply_mode == "merge" or coverage_complete
    changes: dict[str, list[Any]] = {
        "updated_indexes": [],
        "cleared_indexes": [],
        "preserved_indexes": sorted(
            {
                _int_value(item.get("index"))
                for item in results
                if not item.get("authoritative_observation") and _int_value(item.get("index"))
            }
        ),
        "changes": [],
    }
    if apply_allowed:
        if any(isinstance(result.get("ocr_blocks"), list) for result in results):
            from .ocr_block_candidates import project_ocr_block_candidates
            timeline[:] = project_ocr_block_candidates(root, timeline, results)
        changes = _apply_ocr_results_to_timeline(timeline, results, apply_mode=apply_mode)
        write_json(root / "timeline.json", timeline)

    source_updated = False
    source_package_text = str(manifest.get("source_package") or "").strip()
    source_package = Path(source_package_text).expanduser() if source_package_text else None
    if apply_allowed and source_package and source_package.exists() and source_package.is_file():
        package = read_json(source_package)
        if isinstance(package, dict) and isinstance(package.get("timeline"), list):
            _apply_ocr_results_to_timeline(package["timeline"], results, apply_mode=apply_mode)
            package["coverage"] = _coverage_audit(package["timeline"])
            package["quality_audit"] = _quality_audit(package["timeline"])
            package["ocr_backfilled_at"] = now_iso()
            write_json(source_package, package)
            source_updated = True
    receipt_path = root / "ocr-backfill-overwrite-receipt.json" if apply_mode == "replace_snapshot" else None
    if receipt_path is not None:
        input_path = Path(input_json).expanduser().resolve() if input_json else None
        write_json(
            receipt_path,
            {
                "schema": "video_knowledge_pipeline.ocr_overwrite_receipt.v1",
                "created_at": now_iso(),
                "status": "applied" if apply_allowed else "rejected_incomplete_snapshot",
                "mode": apply_mode,
                "bundle_dir": str(root),
                "source_input": str(input_path or ""),
                "source_input_sha256": _sha256_file_if_available(input_path),
                "coverage": {
                    "complete": coverage_complete,
                    "candidate_indexes": candidate_indexes,
                    "authoritative_indexes": authoritative_indexes,
                    "missing_indexes": missing_indexes,
                },
                "updated_indexes": changes["updated_indexes"],
                "cleared_indexes": changes["cleared_indexes"],
                "preserved_indexes": changes["preserved_indexes"],
                "changes": changes["changes"],
                "scope": "timeline visual_text and OCR backfill fields for image-backed candidates only",
                "rollback": {
                    "source": "changes[].before_text",
                    "command": "Review changes[].before_text and restore only accepted values; no automatic rollback is performed.",
                },
            },
        )
    return {
        "updated": len(changes["updated_indexes"]),
        "updated_indexes": changes["updated_indexes"],
        "cleared": len(changes["cleared_indexes"]),
        "cleared_indexes": changes["cleared_indexes"],
        "preserved_indexes": changes["preserved_indexes"],
        "coverage_complete": coverage_complete,
        "missing_indexes": missing_indexes,
        "applied": apply_allowed,
        "overwrite_receipt_path": str(receipt_path or ""),
        "source_package_updated": source_updated,
    }


def _apply_ocr_results_to_timeline(
    timeline: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    apply_mode: str,
) -> dict[str, list[Any]]:
    updated: list[int] = []
    cleared: list[int] = []
    preserved: list[int] = []
    changes: list[dict[str, Any]] = []
    for result in results:
        text = str(result.get("text") or "").strip()
        index = _int_value(result.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        item = timeline[index - 1]
        current = str(item.get("visual_text") or "").strip()
        authoritative = bool(result.get("authoritative_observation"))
        if apply_mode == "replace_snapshot" and authoritative and not text:
            if current:
                item.setdefault("original_visual_text", current)
            item["ocr_snapshot_previous_visual_text"] = current
            item.pop("visual_text", None)
            item.pop("ocr_backfilled_text", None)
            item.pop("ocr_backfilled_at", None)
            item["ocr_snapshot_cleared_at"] = now_iso()
            item["quality_issues"] = _dedupe([*_quality_issues(item), "missing_visual_text"])
            cleared.append(index)
            changes.append(
                {
                    "index": index,
                    "action": "cleared",
                    "before_sha256": _text_sha256(current),
                    "after_sha256": _text_sha256(""),
                    "before_text": current,
                    "after_text": "",
                }
            )
            continue
        if not (result.get("ok") and text):
            preserved.append(index)
            continue
        if current and current != text:
            item.setdefault("original_visual_text", current)
        if apply_mode == "replace_snapshot":
            item["ocr_snapshot_previous_visual_text"] = current
        item["visual_text"] = text
        item["visual_text_role"] = str(result.get("text_role") or "unclassified")
        item["ocr_backfilled_text"] = text
        item["ocr_backfilled_at"] = now_iso()
        item["quality_issues"] = [issue for issue in _quality_issues(item) if issue not in OCR_GAP_ISSUES]
        updated.append(index)
        changes.append(
            {
                "index": index,
                "action": "updated",
                "before_sha256": _text_sha256(current),
                "after_sha256": _text_sha256(text),
                "before_text": current,
                "after_text": text,
            }
        )
    return {
        "updated_indexes": sorted(set(updated)),
        "cleared_indexes": sorted(set(cleared)),
        "preserved_indexes": sorted(set(preserved)),
        "changes": changes,
    }


def _quality_issues(item: dict[str, Any]) -> list[str]:
    audit = _quality_audit([item])
    priority_items = audit.get("priority_items") if isinstance(audit, dict) else []
    if isinstance(priority_items, list) and priority_items:
        issues = priority_items[0].get("issues", [])
        return [str(issue) for issue in issues if issue]
    return []


def _render_ocr_backfill_report(
    root: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    runner: dict[str, Any],
    template_path: Path,
    screen_text_recovery: dict[str, Any],
    capabilities: dict[str, Any],
    recovery_commands: list[dict[str, Any]],
) -> str:
    tesseract = capabilities.get("tesseract") if isinstance(capabilities.get("tesseract"), dict) else {}
    captiocr = capabilities.get("captiocr") if isinstance(capabilities.get("captiocr"), dict) else {}
    lines = [
        "# OCR Backfill Report",
        "",
        f"- Status: `{summary.get('status', '')}`",
        f"- OK: `{summary.get('ok', False)}`",
        f"- Bundle: `{root}`",
        f"- Execute: `{summary.get('execute')}`",
        f"- Apply mode: `{summary.get('apply_mode', 'merge')}`",
        f"- Input JSON: `{summary.get('input_json') or ''}`",
        f"- Input template JSON: `{template_path}`",
        f"- Overwrite receipt: `{summary.get('overwrite_receipt_path') or ''}`",
        f"- Runner: `{runner.get('name')}` available=`{runner.get('available')}`",
        f"- Total: {summary.get('total', 0)}",
        f"- Whole bundle (sampled frames only): `{summary.get('whole_bundle', {})}`",
        f"- Remaining indexes: `{summary.get('pending_indexes', [])}`",
        f"- Resume argv: `{summary.get('resume_argv', [])}`",
        "- Empty OCR output does not prove that the frame contains no text.",
        f"- Succeeded: {summary.get('succeeded', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Planned: {summary.get('planned', 0)}",
        f"- Cleared: {summary.get('cleared', 0)}",
        f"- CaptiOCR capability: `{captiocr.get('status', '')}` / available=`{captiocr.get('available', False)}`",
        f"- Tesseract capability: `{tesseract.get('status', '')}` / runtime=`{tesseract.get('runtime_available', False)}` / language_ready=`{tesseract.get('language_ready', False)}`",
        f"- Requested languages: `{', '.join(tesseract.get('requested_languages') or [])}`",
        f"- Missing languages: `{', '.join(tesseract.get('missing_languages') or [])}`",
        "",
        "## Screen Text Recovery Plan",
        "",
        "这个计划只给出恢复策略和裁剪候选，不默认执行新的 OCR。图文型画面优先走 ebook_markdown_pipeline；小 UI 字、浏览器/编辑器界面、字幕干扰和画中画遮挡会被标成低置信度并进入人工或 crop OCR 计划。",
        "",
        "| Strategy | Count | Meaning |",
        "|---|---:|---|",
    ]
    for key, count in sorted((screen_text_recovery.get("strategy_counts") or {}).items()):
        lines.append(f"| `{key}` | {count} | {_recovery_strategy_label(key)} |")
    lines.extend([""])
    if runner.get("error"):
        lines.extend(["```text", str(runner.get("error") or "").strip(), "```", ""])
    if recovery_commands:
        lines.extend(["## Recovery Commands", ""])
        for row in recovery_commands:
            lines.append(f"- `{row.get('key', '')}`: `{row.get('command', '')}`")
            if row.get("reason"):
                lines.append(f"  - {row.get('reason')}")
        lines.append("")
    for item in results:
        lines.extend(
            [
                f"## Timeline {item.get('index')}",
                "",
                f"- Image: `{item.get('image_path', '')}`",
                f"- OK: `{item.get('ok')}`",
                f"- Image exists: `{item.get('image_exists')}`",
                "",
            ]
        )
        recovery = item.get("screen_text_recovery") if isinstance(item.get("screen_text_recovery"), dict) else {}
        if recovery:
            lines.extend(
                [
                    f"- Recovery strategy: `{recovery.get('strategy', '')}`",
                    f"- Recommended tool: `{recovery.get('recommended_tool', '')}`",
                    f"- Reason: {recovery.get('reason', '')}",
                    "",
                ]
            )
            crop_candidates = recovery.get("crop_candidates") if isinstance(recovery.get("crop_candidates"), list) else []
            if crop_candidates:
                lines.extend(["### Crop candidates", "", "| Name | Box | Purpose |", "|---|---|---|"])
                for crop in crop_candidates:
                    if not isinstance(crop, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                _md_cell(str(crop.get("name") or "")),
                                _md_cell(str(crop.get("box") or "")),
                                _md_cell(str(crop.get("purpose") or "")),
                            ]
                        )
                        + " |"
                    )
                lines.append("")
        text = str(item.get("text") or "").strip()
        if text:
            lines.extend(["```text", text, "```", ""])
        if item.get("stderr"):
            lines.extend(["```text", str(item.get("stderr") or "").strip(), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _build_ocr_backfill_handoff(
    root: Path,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    runner: dict[str, Any],
    captiocr: dict[str, Any],
    screen_text_recovery: dict[str, Any],
    template_path: Path,
    report_path: Path,
    handoff_path: Path,
    handoff_json_path: Path,
    capabilities: dict[str, Any],
    recovery_commands: list[dict[str, Any]],
) -> dict[str, Any]:
    mcp_args_path = _resolve_manifest_path(root, manifest.get("mcp_ocr_backfill_args"))
    mcp_command = _mcp_command("run_ocr_backfill", mcp_args_path) if mcp_args_path else ""
    unresolved = [
        item
        for item in results
        if not (item.get("ok") and str(item.get("text") or "").strip())
    ]
    return {
        "schema": "lecture_ocr_backfill_handoff.v1",
        "created_at": now_iso(),
        "bundle_dir": str(root),
        "status": _ocr_handoff_status(summary),
        "objective": "Fill missing screen text, board text, formulas, code, tables, and diagram labels without summarizing or dropping visual information.",
        "paths": {
            "handoff_markdown": str(handoff_path),
            "handoff_json": str(handoff_json_path),
            "report_markdown": str(report_path),
            "input_template_json": str(template_path),
            "overwrite_receipt_json": str(summary.get("overwrite_receipt_path") or ""),
            "manifest_json": str(root / "manifest.json"),
            "timeline_json": str(root / "timeline.json"),
        },
        "captiocr": {
            **captiocr,
            "runner_available": bool(runner.get("available")) and runner.get("name") == "captiocr",
            "runner_error": str(runner.get("error") or ""),
        },
        "capabilities": capabilities,
        "recovery_commands": recovery_commands,
        "mcp": {
            "tool": "run_ocr_backfill",
            "args_path": str(mcp_args_path) if mcp_args_path else "",
            "args_exists": bool(mcp_args_path and mcp_args_path.exists()),
            "command": mcp_command,
            "import_argument": "input_json",
        },
        "summary": summary,
        "screen_text_recovery": screen_text_recovery,
        "next_steps": _ocr_handoff_next_steps(summary, captiocr, mcp_command),
        "import_schema": {
            "schema": "lecture_ocr_backfill_input.v1",
            "items": [
                {
                    "index": 1,
                    "text": "Exact OCR/manual transcription for the frame.",
                    "source": "absolute/or bundle-relative image path",
                    "notes": "Optional uncertainty or human review notes.",
                }
            ],
        },
        "items": [
            {
                "index": item.get("index"),
                "image_path": item.get("image_path", ""),
                "image_exists": bool(item.get("image_exists")),
                "needs_text": not bool(str(item.get("text") or "").strip()),
                "ok": bool(item.get("ok")),
                "stderr": str(item.get("stderr") or ""),
                "text": str(item.get("text") or ""),
                "screen_text_recovery": item.get("screen_text_recovery", {}),
            }
            for item in results
        ],
        "unresolved_indexes": [item.get("index") for item in unresolved],
    }


def _ocr_handoff_status(summary: dict[str, Any]) -> str:
    total = int(summary.get("total") or 0)
    succeeded = int(summary.get("succeeded") or 0)
    if total == 0:
        return "not_needed"
    if succeeded >= total:
        return "filled"
    if succeeded:
        return "partially_filled"
    return "needs_ocr"


def _ocr_handoff_next_steps(summary: dict[str, Any], captiocr: dict[str, Any], mcp_command: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if int(summary.get("total") or 0) == 0:
        return [{"actor": "human_or_agent", "action": "No OCR gap candidates were detected for this bundle."}]
    if captiocr.get("available"):
        steps.append(
            {
                "actor": "agent",
                "action": "Run CaptiOCR/Tesseract through the existing OCR backfill tool for available frames.",
                "command": mcp_command,
            }
        )
    else:
        steps.append(
            {
                "actor": "human",
                "action": "Configure LECTURE_CAPTIOCR_ROOT or open another OCR/Tk tool, then fill ocr-backfill-input-template.json.",
                "configure_hint": captiocr.get("configure_hint", ""),
                "command_hint": captiocr.get("command_hint", ""),
            }
        )
    steps.append(
        {
            "actor": "human_or_agent",
            "action": "Import corrected OCR JSON through run_ocr_backfill input_json; preserve formulas, code indentation, tables, and diagram labels verbatim where possible.",
        }
    )
    return steps


def _resolve_manifest_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else root / path


def _mcp_command(tool: str, args_path: Path) -> str:
    escaped = str(args_path).replace("'", "''")
    return f".\\scripts\\video-knowledge.ps1 mcp-call {tool} '{escaped}'"


def _render_ocr_backfill_handoff_markdown(handoff: dict[str, Any]) -> str:
    paths = handoff.get("paths") if isinstance(handoff.get("paths"), dict) else {}
    captiocr = handoff.get("captiocr") if isinstance(handoff.get("captiocr"), dict) else {}
    mcp = handoff.get("mcp") if isinstance(handoff.get("mcp"), dict) else {}
    capabilities = handoff.get("capabilities") if isinstance(handoff.get("capabilities"), dict) else {}
    tesseract = capabilities.get("tesseract") if isinstance(capabilities.get("tesseract"), dict) else {}
    lines = [
        "# OCR Backfill Handoff",
        "",
        f"- Status: `{handoff.get('status')}`",
        f"- Bundle: `{handoff.get('bundle_dir')}`",
        f"- Input template: `{paths.get('input_template_json', '')}`",
        f"- Report: `{paths.get('report_markdown', '')}`",
        f"- CaptiOCR root: `{captiocr.get('root', '')}`",
        f"- CaptiOCR checkout available: `{captiocr.get('available')}`",
        f"- CaptiOCR runner available: `{captiocr.get('runner_available')}`",
        f"- Tesseract status: `{tesseract.get('status', '')}`",
        f"- Missing languages: `{', '.join(tesseract.get('missing_languages') or [])}`",
        f"- MCP args: `{mcp.get('args_path', '')}`",
        "",
        "## MCP",
        "",
        "```powershell",
        str(mcp.get("command") or ""),
        "```",
        "",
        "## Next Steps",
        "",
    ]
    for step in handoff.get("next_steps") or []:
        if not isinstance(step, dict):
            continue
        lines.append(f"- `{step.get('actor', '')}`: {step.get('action', '')}")
        if step.get("command_hint"):
            lines.append(f"  Command hint: `{step.get('command_hint')}`")
        if step.get("configure_hint"):
            lines.append(f"  Configure: `{step.get('configure_hint')}`")
    recovery_commands = [row for row in handoff.get("recovery_commands") or [] if isinstance(row, dict)]
    if recovery_commands:
        lines.extend(["", "## Recovery Commands", ""])
        for row in recovery_commands:
            lines.append(f"- `{row.get('key', '')}`: `{row.get('command', '')}`")
    lines.extend(
        [
            "",
            "## Screen Text Recovery Plan",
            "",
            "| Strategy | Count | Meaning |",
            "|---|---:|---|",
        ]
    )
    recovery = handoff.get("screen_text_recovery") if isinstance(handoff.get("screen_text_recovery"), dict) else {}
    for key, count in sorted((recovery.get("strategy_counts") or {}).items()):
        lines.append(f"| `{key}` | {count} | {_recovery_strategy_label(key)} |")
    lines.extend(
        [
            "",
            "## Import Schema",
            "",
            "```json",
            '{ "items": [ { "index": 1, "text": "exact screen text/formula/code/table text", "source": "frame path", "notes": "" } ] }',
            "```",
            "",
            "## Items",
            "",
            "| Index | OK | Needs text | Image | Error |",
            "|---:|---|---|---|---|",
        ]
    )
    for item in handoff.get("items") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("index") or ""),
                    str(bool(item.get("ok"))),
                    str(bool(item.get("needs_text"))),
                    _md_cell(str(item.get("image_path") or "")),
                    _md_cell(str(item.get("stderr") or "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _ocr_capabilities(
    captiocr: dict[str, Any],
    runner: dict[str, Any],
    tesseract: dict[str, Any],
    *,
    language: str,
) -> dict[str, Any]:
    runner_available = bool(runner.get("available")) and runner.get("name") == "captiocr"
    checkout_available = bool(captiocr.get("available"))
    captiocr_status = "ready" if runner_available else "checkout_only" if checkout_available else "unavailable"
    return {
        "requested_language": language,
        "captiocr": {
            "available": runner_available,
            "checkout_available": checkout_available,
            "status": captiocr_status,
            "root": str(captiocr.get("root") or ""),
            "error": str(runner.get("error") or ""),
            "configure_hint": str(captiocr.get("configure_hint") or ""),
        },
        "tesseract": dict(tesseract),
    }


def _ocr_result_status(
    results: list[dict[str, Any]],
    *,
    execute: bool,
    imported: bool,
    capabilities: dict[str, Any],
) -> tuple[str, bool]:
    total = len(results)
    succeeded = sum(1 for item in results if item.get("ok") and str(item.get("text") or "").strip())
    if total == 0:
        return "not_needed", True
    if not execute and not imported:
        return "planned", True
    if succeeded >= total:
        return "filled", True
    if succeeded:
        return "partially_filled", False
    tesseract = capabilities.get("tesseract") if isinstance(capabilities.get("tesseract"), dict) else {}
    captiocr = capabilities.get("captiocr") if isinstance(capabilities.get("captiocr"), dict) else {}
    if bool(tesseract.get("runtime_available")) and not bool(tesseract.get("language_ready")):
        return "missing_language_packs", False
    if not bool(captiocr.get("available")) and not bool(tesseract.get("runtime_available")):
        return "ocr_backend_unavailable", False
    return "ocr_failed", False


def _ocr_recovery_commands(capabilities: dict[str, Any]) -> list[dict[str, str]]:
    captiocr = capabilities.get("captiocr") if isinstance(capabilities.get("captiocr"), dict) else {}
    tesseract = capabilities.get("tesseract") if isinstance(capabilities.get("tesseract"), dict) else {}
    commands: list[dict[str, str]] = []
    if not tesseract.get("runtime_available"):
        commands.append(
            {
                "key": "configure_tesseract_runtime",
                "command": "Set LECTURE_TESSERACT_CMD=<tesseract-executable> and LECTURE_TESSDATA_PREFIX=<tessdata-directory>, then rerun run-ocr-backfill.",
                "reason": "The preflight does not install Tesseract or language data.",
            }
        )
    elif tesseract.get("missing_languages"):
        commands.append(
            {
                "key": "configure_tesseract_languages",
                "command": "Provide the missing *.traineddata files in <tessdata-directory>, then rerun run-ocr-backfill.",
                "reason": "Missing languages: " + ", ".join(str(item) for item in tesseract.get("missing_languages") or []),
            }
        )
    if not captiocr.get("checkout_available"):
        commands.append(
            {
                "key": "configure_captiocr",
                "command": "Set LECTURE_CAPTIOCR_ROOT=<captiocr-root>, or import reviewed OCR JSON through run-ocr-backfill --input-json.",
                "reason": "CaptiOCR is optional and remains a local fallback/import path.",
            }
        )
    return commands




def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _requested_languages(language: str) -> list[str]:
    normalized = str(language or "").replace(",", "+")
    return _dedupe([part.strip() for part in normalized.split("+") if part.strip()])


def _text_sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _sha256_file_if_available(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return sha256_file(path)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
