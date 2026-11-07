from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_argument as _quote_command_part
from .models import now_iso
from .repair_status import build_repair_status
from .storage import read_json, write_json


def run_frame_recapture_plan(
    bundle_dir: str | Path,
    *,
    execute: bool = False,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Preview or execute ffmpeg frame recapture commands from a WebUI bundle manifest."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    plan = manifest.get("frame_recapture") if isinstance(manifest.get("frame_recapture"), dict) else {}
    items = plan.get("items") if isinstance(plan.get("items"), list) else []
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result = _run_frame_recapture_item(item, execute=execute, timeout_seconds=timeout_seconds)
        results.append(result)

    summary = {
        "total": len(results),
        "execute": execute,
        "succeeded": sum(1 for item in results if item.get("ok")),
        "failed": sum(1 for item in results if item.get("executed") and not item.get("ok")),
        "planned": sum(1 for item in results if not item.get("executed")),
        "updated_at": now_iso(),
    }
    plan["last_run"] = summary
    plan["items"] = [_merge_item_result(item, result) for item, result in zip(items, results) if isinstance(item, dict)]
    manifest["frame_recapture"] = plan
    backfill = _backfill_successful_frames(root, manifest, results) if execute else {"updated": 0, "source_package_updated": False}
    timeline = _read_timeline(root)
    manifest["coverage"] = _coverage_audit(timeline)
    manifest["frame_recapture"]["last_backfill"] = {**backfill, "updated_at": now_iso()}
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    write_json(manifest_path, manifest)

    report_path = root / "frame-recapture-report.md"
    report_path.write_text(_render_frame_recapture_report(root, results, summary), encoding="utf-8")
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "summary": summary,
        "backfill": backfill,
        "items": results,
    }


def _backfill_successful_frames(root: Path, manifest: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    timeline_path = root / "timeline.json"
    timeline = _read_timeline(root)
    updated_indexes = _backfill_timeline(timeline, results, root=root)
    write_json(timeline_path, timeline)

    source_updated = False
    source_package_text = str(manifest.get("source_package") or "").strip()
    source_package = Path(source_package_text).expanduser() if source_package_text else None
    if source_package and source_package.exists() and source_package.is_file():
        package = read_json(source_package)
        if isinstance(package, dict) and isinstance(package.get("timeline"), list):
            _backfill_package_timeline(package["timeline"], results)
            package["coverage"] = _coverage_audit(package["timeline"])
            package["quality_audit"] = _quality_audit(package["timeline"])
            package["frame_recapture_backfilled_at"] = now_iso()
            write_json(source_package, package)
            source_updated = True
    return {
        "updated": len(updated_indexes),
        "updated_indexes": updated_indexes,
        "source_package_updated": source_updated,
    }


def _read_timeline(root: Path) -> list[dict[str, Any]]:
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        return []
    timeline = read_json(timeline_path)
    return [item for item in timeline if isinstance(item, dict)] if isinstance(timeline, list) else []


def _backfill_timeline(timeline: list[dict[str, Any]], results: list[dict[str, Any]], *, root: Path) -> list[int]:
    updated: list[int] = []
    for result in results:
        if not (result.get("ok") and result.get("exists")):
            continue
        index = _int_value(result.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        item = timeline[index - 1]
        output = Path(str(result.get("output_path") or "")).expanduser()
        relative = _relative_to(output, root)
        assets = item.setdefault("assets", [])
        if not isinstance(assets, list):
            assets = []
            item["assets"] = assets
        if not any(str(asset.get("source") or asset.get("path") or "") == str(output) for asset in assets if isinstance(asset, dict)):
            assets.append({"source": str(output), "path": relative, "copied": "true", "recaptured": True})
        frame_paths = item.setdefault("frame_paths", [])
        if not isinstance(frame_paths, list):
            frame_paths = []
            item["frame_paths"] = frame_paths
        if str(output) not in frame_paths:
            frame_paths.append(str(output))
        material_types = item.setdefault("material_types", [])
        if isinstance(material_types, list) and "image" not in material_types:
            material_types.append("image")
        item["recaptured_frame_path"] = str(output)
        item["recaptured_at"] = now_iso()
        item["quality_issues"] = [issue for issue in item.get("quality_issues", []) if issue not in FRAME_GAP_ISSUES]
        updated.append(index)
    return updated


def _backfill_package_timeline(timeline: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    for result in results:
        if not (result.get("ok") and result.get("exists")):
            continue
        index = _int_value(result.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        item = timeline[index - 1]
        output = str(Path(str(result.get("output_path") or "")).expanduser())
        frame_paths = item.setdefault("frame_paths", [])
        if not isinstance(frame_paths, list):
            frame_paths = []
            item["frame_paths"] = frame_paths
        if output not in frame_paths:
            frame_paths.append(output)
        material_types = item.setdefault("material_types", [])
        if isinstance(material_types, list) and "image" not in material_types:
            material_types.append("image")
        item["recaptured_frame_path"] = output
        item["recaptured_at"] = now_iso()


FRAME_GAP_ISSUES = {"missing_frame", "structured_visual_without_frame", "keep_image_without_frame"}


def _run_frame_recapture_item(item: dict[str, Any], *, execute: bool, timeout_seconds: int) -> dict[str, Any]:
    source = str(item.get("video_key") or "").strip()
    output = Path(str(item.get("output_path") or "")).expanduser()
    midpoint = _float_value(item.get("midpoint"))
    command = ["ffmpeg", "-y", "-ss", f"{max(midpoint, 0):.3f}", "-i", source, "-frames:v", "1", str(output)]
    result = {
        "index": item.get("index"),
        "midpoint": midpoint,
        "video_key": source,
        "output_path": str(output),
        "command": " ".join(_quote_command_part(part) for part in command),
        "executed": execute,
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "exists": output.exists(),
    }
    if not execute:
        result["ok"] = bool(source and item.get("ffmpeg_command"))
        return result
    if not source:
        result["stderr"] = "missing source video path"
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=int(timeout_seconds or 0) or None,
            check=False,
        )
    except FileNotFoundError as exc:
        result["stderr"] = str(exc)
        return result
    except subprocess.TimeoutExpired as exc:
        result["stderr"] = f"timeout after {timeout_seconds}s: {exc}"
        return result
    result["returncode"] = completed.returncode
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    result["exists"] = output.exists()
    result["ok"] = completed.returncode == 0 and output.exists()
    return result


def _merge_item_result(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    merged = dict(item)
    merged["last_run"] = {
        "executed": result.get("executed"),
        "ok": result.get("ok"),
        "returncode": result.get("returncode"),
        "exists": result.get("exists"),
        "updated_at": now_iso(),
    }
    return merged


def _render_frame_recapture_report(root: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Frame Recapture Report",
        "",
        f"- Bundle: `{root}`",
        f"- Execute: `{summary.get('execute')}`",
        f"- Total: {summary.get('total', 0)}",
        f"- Succeeded: {summary.get('succeeded', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Planned: {summary.get('planned', 0)}",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## Timeline {item.get('index')}",
                "",
                f"- Time: `{item.get('midpoint')}`",
                f"- Output: `{item.get('output_path')}`",
                f"- OK: `{item.get('ok')}`",
                "",
                "```powershell",
                str(item.get("command") or ""),
                "```",
                "",
            ]
        )
        if item.get("stderr"):
            lines.extend(["```text", str(item.get("stderr") or "").strip(), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _coverage_audit(timeline: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "timeline_items": len(timeline),
        "items_with_transcript": sum(1 for item in timeline if item.get("transcript")),
        "items_with_visual_text": sum(1 for item in timeline if item.get("visual_text")),
        "items_with_frames": sum(1 for item in timeline if item.get("frame_paths") or item.get("assets")),
        "items_needing_review": sum(1 for item in timeline if item.get("needs_human_review")),
        "items_reviewed": sum(1 for item in timeline if item.get("review_status") == "reviewed"),
        "items_needing_revision": sum(1 for item in timeline if item.get("review_status") == "needs_revision"),
        "items_with_human_notes": sum(1 for item in timeline if (item.get("human_review") or {}).get("notes")),
        "items_with_corrected_transcript": sum(1 for item in timeline if "original_transcript" in item),
        "items_with_corrected_visual_text": sum(1 for item in timeline if "original_visual_text" in item),
        "items_with_structured_visual": sum(1 for item in timeline if item.get("structured_visual")),
        "items_with_visual_route": sum(1 for item in timeline if item.get("visual_route")),
        "items_with_visual_understanding": sum(1 for item in timeline if item.get("visual_understanding")),
        "items_with_temporal_understanding": sum(1 for item in timeline if item.get("temporal_visual_understanding")),
        "structured_visual_entries": sum(
            len(item.get("structured_visual") or [])
            for item in timeline
            if isinstance(item.get("structured_visual"), list)
        ),
        "possible_code_items": sum(1 for item in timeline if "code" in item.get("material_types", [])),
        "possible_formula_items": sum(1 for item in timeline if "formula" in item.get("material_types", [])),
        "possible_table_items": sum(1 for item in timeline if "table" in item.get("material_types", [])),
    }


def _quality_audit(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    priority_items = []
    issue_counts: dict[str, int] = {}
    for index, item in enumerate(timeline, start=1):
        issues = _quality_issues(item)
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        score = _quality_score(issues)
        if score:
            priority_items.append(
                {
                    "index": index,
                    "start": item.get("start", 0),
                    "end": item.get("end", 0),
                    "score": score,
                    "issues": issues,
                    "material_types": item.get("material_types", []),
                    "source_segment_ids": item.get("source_segment_ids", []),
                    "transcript": item.get("transcript", ""),
                    "visual_text": item.get("visual_text", ""),
                    "frame_paths": item.get("frame_paths", []),
                    "review_status": item.get("review_status", "pending"),
                }
            )
    priority_items.sort(key=lambda row: (-int(row["score"]), float(row.get("start", 0)), int(row.get("index", 0))))
    return {
        "summary": {
            "timeline_items": len(timeline),
            "items_with_issues": len(priority_items),
            "max_score": max([int(item["score"]) for item in priority_items], default=0),
            **{f"issue_{key}": value for key, value in sorted(issue_counts.items())},
        },
        "priority_items": priority_items,
    }


def _quality_issues(item: dict[str, Any]) -> list[str]:
    issues = []
    transcript = str(item.get("transcript") or "").strip()
    visual_text = str(item.get("visual_text") or "").strip()
    frames = item.get("frame_paths") or []
    if not frames and isinstance(item.get("assets"), list):
        frames = [asset.get("path") or asset.get("source") for asset in item["assets"] if isinstance(asset, dict)]
    material_types = set(item.get("material_types") or [])
    structured_visual = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
    visual_route = str(item.get("visual_route") or "")
    review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    ebook_status = item.get("ebook_pipeline_status") if isinstance(item.get("ebook_pipeline_status"), dict) else {}
    if item.get("needs_human_review", True):
        issues.append("needs_human_review")
    if ebook_status and ebook_status.get("ok") is False:
        blocker = str(ebook_status.get("blocker") or "ebook_pipeline_failed")
        if blocker:
            issues.append(blocker)
        if blocker in {"ocr_wrapper_only", "ocr_text_empty", "ocr_text_low_information"}:
            issues.append("needs_high_res_tile_recovery")
            issues.append("needs_multimodal_or_human_review")
    if item.get("review_status") == "needs_revision":
        issues.append("needs_revision")
    if not transcript:
        issues.append("missing_transcript")
    if not visual_text:
        issues.append("missing_visual_text")
    if not frames:
        issues.append("missing_frame")
    if material_types & {"formula", "table", "code"} and not frames:
        issues.append("structured_visual_without_frame")
    if material_types & {"formula", "table", "code"} and not visual_text:
        issues.append("structured_visual_without_ocr")
    if material_types & {"formula", "table", "code"} and not structured_visual:
        issues.append("structured_visual_without_structure")
    if frames and not visual_route:
        issues.append("missing_visual_route")
    if visual_route in {"semantic_frame", "mixed"} and not item.get("visual_understanding"):
        issues.append("semantic_frame_without_analysis")
        issues.append("missing_visual_understanding")
    if visual_route in {"semantic_frame", "mixed"} and _understanding_incomplete(item.get("visual_understanding")):
        issues.append("visual_understanding_incomplete")
        issues.append("missing_visual_understanding")
    if visual_route in {"temporal_sequence", "mixed"} and not item.get("temporal_visual_understanding"):
        issues.append("temporal_sequence_without_analysis")
        issues.append("missing_visual_understanding")
    if visual_route in {"temporal_sequence", "mixed"} and _understanding_incomplete(item.get("temporal_visual_understanding")):
        issues.append("temporal_understanding_incomplete")
        issues.append("missing_visual_understanding")
    if review.get("asr_ocr_error"):
        issues.append("reported_asr_ocr_error")
    if review.get("missing_info"):
        issues.append("reported_missing_info")
    if review.get("keep_images") and not frames:
        issues.append("keep_image_without_frame")
    return _dedupe(issues)


def _quality_score(issues: list[str]) -> int:
    weights = {
        "reported_missing_info": 8,
        "keep_image_without_frame": 8,
        "reported_asr_ocr_error": 7,
        "structured_visual_without_frame": 6,
        "structured_visual_without_ocr": 6,
        "structured_visual_without_structure": 5,
        "temporal_sequence_without_analysis": 6,
        "temporal_understanding_incomplete": 6,
        "semantic_frame_without_analysis": 5,
        "visual_understanding_incomplete": 5,
        "missing_visual_understanding": 4,
        "missing_visual_route": 3,
        "needs_revision": 5,
        "missing_frame": 3,
        "missing_visual_text": 3,
        "missing_transcript": 3,
        "ocr_wrapper_only": 4,
        "ocr_text_empty": 4,
        "ocr_text_low_information": 3,
        "needs_multimodal_or_human_review": 3,
        "needs_human_review": 1,
    }
    return sum(weights.get(issue, 1) for issue in issues)


def _understanding_incomplete(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and (value.get("parse_failed") or value.get("validation_status") == "incomplete")


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
