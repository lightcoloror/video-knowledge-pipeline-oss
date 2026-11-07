"""Bounded orchestration over the existing FFmpeg and ebook adapters."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .content_scope import load_content_scope
from .courseware_material import (PAGE_SCHEMA, SCAN_SCHEMA, build_courseware_material,
                                 import_courseware_page, read_scan, safe_path, store_scan)
from .file_hash import sha256_file
from .media_tools import resolve_media_tool
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .video import parse_showinfo_scene_times, probe_video


def _ranges(start: float, end: float, exclusions: list) -> list[tuple[float, float]]:
    if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end <= start:
        raise ValueError("invalid courseware time range")
    ranges = [(start, end)]
    for row in exclusions:
        a, b = float(row["start"]), float(row["end"])
        if not all(math.isfinite(x) for x in (a, b)) or a < 0 or b <= a:
            raise ValueError("invalid courseware excluded range")
        remaining = []
        for left, right in ranges:
            if b <= left or a >= right:
                remaining.append((left, right))
            else:
                if left < a:
                    remaining.append((left, a))
                if b < right:
                    remaining.append((b, right))
        ranges = remaining
    if not ranges:
        raise ValueError("entire courseware scope excluded")
    return ranges


def prepare_courseware_scan(bundle_dir, source, *, start, end, excluded_intervals=None,
                            window_seconds=30.0, threshold=0.0001) -> dict:
    root = Path(bundle_dir).absolute()
    safe_path(root, "manifest.json")
    safe_path(root, "timeline.json")
    if not 1 <= window_seconds <= 60 or not 0 <= threshold <= 0.01:
        raise ValueError("courseware window/threshold outside safe bounds")
    source = Path(source).absolute()
    safe_path(source.parent, str(source))
    metadata = probe_video(source)
    if end > metadata.duration_seconds + 0.001 or not metadata.width or not metadata.height or not metadata.fps:
        raise ValueError("invalid source duration/dimensions/fps")
    scope = load_content_scope(root)
    inherited = [row for row in scope.get("excluded_intervals", [])
                 if "smart_summary" in row.get("projections", ["smart_summary"]) and row.get("action", "exclude") == "exclude"]
    exclusions = [*(excluded_intervals or []), *inherited]
    ranges = _ranges(float(start), float(end), exclusions)
    windows = []
    for left, right in ranges:
        while left < right:
            following = min(right, left + window_seconds)
            windows.append({"start": left, "end": following})
            left = following
    if len(windows) > 10000:
        raise ValueError("courseware window count budget exceeded")
    identity = {"path": str(source), "sha256": metadata.sha256,
                "bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns,
                "width": metadata.width, "height": metadata.height, "fps": metadata.fps}
    state = {"schema": SCAN_SCHEMA, "run_id": uuid4().hex, "source": identity,
             "scope": {"start": float(start), "end": float(end), "excluded_intervals": exclusions},
             "windows": windows, "completed_windows": [], "candidates": [], "failure": None,
             "content_scope_sha256": scope.get("content_scope_sha256"),
             "detector": {"backend": "ffmpeg-select-scene-showinfo", "threshold": threshold,
                          "frame_sampling": "all_decoded_frames", "independent_review_required": True}}
    with bundle_write_lock(root, operation="prepare_courseware_scan"):
        existing = safe_path(root, "courseware/scan.json", exists=False)
        if existing.exists():
            prior = read_scan(root)
            for key in ("source", "scope", "windows", "detector", "content_scope_sha256"):
                if prior.get(key) != state.get(key):
                    raise ValueError("existing courseware scan identity differs; use a new derived Bundle")
            return prior
        store_scan(root, state)
    return read_scan(root)


def _check_source(state: dict) -> Path:
    path = Path(state["source"]["path"])
    safe_path(path.parent, str(path))
    if path.stat().st_size != state["source"]["bytes"] or sha256_file(path) != state["source"]["sha256"]:
        raise ValueError("courseware source hash drift")
    return path


def _scan_window(source: Path, window: dict, threshold: float, timeout_seconds: int) -> list[float]:
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg unavailable")
    command = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-ss", str(window["start"]),
               "-i", str(source), "-t", str(window["end"] - window["start"]),
               "-vf", f"select='eq(n,0)+gt(scene,{threshold})',showinfo", "-an", "-sn", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            check=False, timeout=timeout_seconds)
    if result.returncode:
        raise ValueError(f"courseware decoder failed: {result.stderr[-1000:]}")
    points = parse_showinfo_scene_times(result.stderr)
    if not points:
        raise ValueError("courseware decoder produced no observations")
    return [round(window["start"] + time, 6) for time in points if 0 <= time < window["end"] - window["start"]]


def run_courseware_scan(bundle_dir, *, max_windows=2, timeout_seconds=60, max_candidates_per_window=2000,
                        disk_budget_bytes=512 * 1024 * 1024, resume=False) -> dict:
    root = Path(bundle_dir).absolute()
    if not 1 <= max_windows <= 16 or not 1 <= timeout_seconds <= 600 or not 1 <= max_candidates_per_window <= 10000:
        raise ValueError("courseware scan batch bounds invalid")
    with bundle_write_lock(root, operation="courseware_scan"):
        state = read_scan(root)
        source = _check_source(state)
        if state.get("failure") and not resume:
            raise ValueError("explicit --resume required after courseware failure")
        state["failure"] = None
        selected = [i for i in range(len(state["windows"])) if i not in state["completed_windows"]][:max_windows]
        for i in selected:
            try:
                window = state["windows"][i]
                # A bounded log/plan budget; frame allocation is checked separately at capture.
                estimated = math.ceil((window["end"] - window["start"]) * state["source"]["fps"]) * 4096
                if estimated > disk_budget_bytes or shutil.disk_usage(root).free < estimated + 512 * 1024 * 1024:
                    raise ValueError("courseware scan disk budget blocked")
                points = _scan_window(source, window, state["detector"]["threshold"], timeout_seconds)
                if len(points) > max_candidates_per_window:
                    raise ValueError("courseware candidate budget exceeded; nothing silently truncated")
                if len(points) + len(state["candidates"]) > 10000:
                    raise ValueError("courseware total candidate budget exceeded; use a smaller explicit scope")
                if not points or points != sorted(set(points)) or any(not window["start"] <= p < window["end"] for p in points):
                    raise ValueError("invalid detector timestamps")
                for n, timestamp in enumerate(points):
                    page_id = "page-" + canonical_json_sha256({"source": state["source"]["sha256"], "time": timestamp})[:20]
                    state["candidates"].append({"page_id": page_id, "time": timestamp,
                                               "end": points[n + 1] if n + 1 < len(points) else window["end"],
                                               "window_index": i, "reason": "first_frame_or_pixel_change", "frame": "", "frame_sha256": ""})
                state["completed_windows"].append(i)
                state["last_checkpoint_at"] = now_iso()
                store_scan(root, state)
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                state["failure"] = {"window_index": i, "error": str(exc), "recovery": "courseware-scan <bundle> --execute --resume"}
                store_scan(root, state)
                return {"ok": False, "status": "blocked", "failure": state["failure"]}
    return {"ok": True, "status": "scan_complete" if len(state["completed_windows"]) == len(state["windows"]) else "partial_scan",
            "completed_windows": len(state["completed_windows"]), "total_windows": len(state["windows"]),
            "candidate_count": len(state["candidates"]), "independent_review_complete": False,
            "next_action": "courseware-extract <bundle> --execute (or continue courseware-scan)"}


def extract_courseware_candidates(bundle_dir, *, limit=16, indexes=None, timeout_seconds=60,
                                   disk_budget_bytes=512 * 1024 * 1024, execute_ocr=False) -> dict:
    from .temporal_frame_groups import _run_candidate
    from .visual_structure_batch import run_visual_structure_ebook_batches

    root = Path(bundle_dir).absolute()
    if not 1 <= limit <= 32 or not 1 <= timeout_seconds <= 600:
        raise ValueError("courseware extraction bounds invalid")
    with bundle_write_lock(root, operation="courseware_capture"):
        state = read_scan(root)
        if indexes is not None:
            raw = indexes.split(",") if isinstance(indexes, str) else indexes
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError("courseware indexes must be a nonempty exact selection")
            if any(isinstance(n, bool) or not str(n).strip().isdigit() for n in raw):
                raise ValueError("courseware indexes must be positive integers")
            selected_indexes = [int(n) for n in raw]
            if (len(selected_indexes) > limit or len(set(selected_indexes)) != len(selected_indexes)
                    or any(n < 1 or n > len(state["candidates"]) for n in selected_indexes)):
                raise ValueError("courseware indexes duplicate, out of range or exceed limit")
        else:
            selected_indexes = [n for n, p in enumerate(state["candidates"], 1)
                                if not p.get("frame")][:limit]
        requested = [state["candidates"][n - 1] for n in selected_indexes]
        source = _check_source(state)
        selected = [p for p in requested if not p.get("frame")]
        estimated = len(selected) * 3 * state["source"]["width"] * state["source"]["height"] * 4
        if estimated > disk_budget_bytes or shutil.disk_usage(root).free < estimated + 512 * 1024 * 1024:
            return {"ok": False, "status": "blocked_disk_budget"}
        for page in selected:
            # Unique attempt destination: even orphaned prior output is never overwritten.
            from .temporal_frame_preprocess import _temporal_image_metrics
            times = [page["time"]]
            span = page["end"] - page["time"]
            if span >= max(1.0, 4 / state["source"]["fps"]):
                times += [page["time"] + span / 3, page["time"] + span * 2 / 3]
            outputs = [safe_path(root, f"courseware/frames/{page['page_id']}-{uuid4().hex[:8]}.png", exists=False) for _ in times]
            result = _run_candidate({"video_key": str(source), "times": times, "output_paths": [str(frame) for frame in outputs]}, execute=True, timeout_seconds=timeout_seconds)
            if not result.get("ok") or any(code != 0 for code in result.get("returncodes", [])):
                return {"ok": False, "status": "capture_failed", "page_id": page["page_id"], "recovery": "courseware-extract <bundle> --execute", "detail": result.get("stderr")}
            from PIL import Image
            for frame in outputs:
                with Image.open(frame) as image:
                    if image.size != (state["source"]["width"], state["source"]["height"]):
                        raise ValueError("courseware screenshot is not native resolution")
            metrics, _ = _temporal_image_metrics([str(frame) for frame in outputs])
            best = max(range(len(outputs)), key=lambda n: metrics[n].get("sharpness_score", 0))
            frame = outputs[best]
            page.update(frame=frame.relative_to(root).as_posix(), frame_sha256=sha256_file(frame), frame_time=times[best],
                        selection={"method": "existing_temporal_sharpness", "occlusion_independently_verified": False,
                                   "alternatives": [{"time": t, "path": p.relative_to(root).as_posix(), "sha256": sha256_file(p), "metrics": m} for t, p, m in zip(times, outputs, metrics)]})
            store_scan(root, state)
        child = safe_path(root, "courseware/ocr-bundle/manifest.json", exists=False).parent
        if not (child / "manifest.json").exists():
            write_json(child / "manifest.json", {"title": "Video-displayed courseware extraction", "source_kind": "video_displayed_courseware"})
        existing = read_json(child / "timeline.json") if (child / "timeline.json").exists() else []
        by_id = {p.get("courseware_page_id"): p for p in existing}
        if len(by_id) != len(existing) or len({r["index"] for r in existing}) != len(existing):
            raise ValueError("duplicate existing courseware OCR identity/index")
        candidates_by_id = {p["page_id"]: p for p in state["candidates"]}
        for row in existing:
            page = candidates_by_id.get(row.get("courseware_page_id"))
            if (not page or row.get("courseware_frame_sha256") != page.get("frame_sha256")
                    or row.get("frame_paths") != [str(root / page["frame"])]):
                raise ValueError("existing courseware OCR screenshot identity drift")
            if sha256_file(safe_path(root, page["frame"])) != page["frame_sha256"]:
                raise ValueError("existing courseware screenshot content drift")
        # Index is an immutable OCR checkpoint identity, not chronological rank.
        rows = list(existing)
        next_index = max((r["index"] for r in rows), default=0) + 1
        for page in state["candidates"]:
            if not page.get("frame") or page["page_id"] in by_id:
                continue
            row = {"index": next_index, "courseware_page_id": page["page_id"],
                        "start": page["time"], "end": page["end"], "visual_route": "document_visual", "material_types": ["text"],
                        "frame_paths": [str(root / page["frame"])], "courseware_frame_sha256": page["frame_sha256"], "transcript": ""}
            rows.append(row)
            next_index += 1
        write_json(child / "timeline.json", rows)
        ocr_result = None
        if execute_ocr:
            allowed = {p["page_id"] for p in requested} if indexes is not None else set(candidates_by_id)
            pending = [r["index"] for r in rows if r["courseware_page_id"] in allowed
                       and not (root / f"courseware/pages/{r['courseware_page_id']}.json").exists()][:limit]
            if pending:
                ocr_result = run_visual_structure_ebook_batches(child, execute=True, indexes=pending, batch_size=1, timeout_seconds=timeout_seconds)
                if not ocr_result.get("ok"):
                    return {"ok": False, "status": "ocr_partial_or_failed", "captured_this_batch": len(selected),
                            "ocr_bundle": str(child), "ocr_result": ocr_result,
                            "next_action": "Collect successful page results, inspect OCR receipt, then explicitly resume the bounded OCR batch."}
        return {"ok": True, "status": "captured", "selected_indexes": selected_indexes,
                "captured_this_batch": len(selected), "ocr_bundle": str(child),
                "ocr_result": ocr_result,
                "next_actions": ["Use existing run-visual-structure / high-res-tile-plan / tile-result-merge on the child Bundle.",
                                 "courseware-collect <bundle> --executor <actual backend> --model-actual <actual model>",
                                 "Import reviewed page blocks with courseware-import; do not alter the original lecture timeline."]}


def collect_courseware_extractions(bundle_dir, *, executor: str, model_actual: str,
                                   indexes=None, refresh_material=True) -> dict:
    root = Path(bundle_dir).absolute()
    state = read_scan(root)
    timeline = read_json(safe_path(root, "courseware/ocr-bundle/timeline.json"))
    if indexes is not None:
        if (not isinstance(indexes, (list, tuple)) or not 1 <= len(indexes) <= 32
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in indexes)
                or len(set(indexes)) != len(indexes)):
            raise ValueError("bounded unique OCR collection indexes required")
        available = {row['index'] for row in timeline}
        if not set(indexes).issubset(available) or len(available) != len(timeline):
            raise ValueError("OCR collection index missing or duplicated")
        timeline = [row for row in timeline if row['index'] in indexes]
    by_id = {r["page_id"]: r for r in state["candidates"]}
    results = []
    for row in timeline:
        page_id = row.get("courseware_page_id")
        if page_id not in by_id:
            raise ValueError("unknown courseware OCR page identity")
        page = by_id[page_id]
        if row.get("courseware_frame_sha256") != page["frame_sha256"] or row.get("frame_paths") != [str(root / page["frame"])]:
            raise ValueError("courseware OCR screenshot identity drift")
        existing = safe_path(root, f"courseware/pages/{page_id}.json", exists=False)
        if existing.exists():
            from .courseware_material import _validate_page
            prior = read_json(existing)
            if prior.get("page_id") != page_id:
                raise ValueError("courseware existing page filename identity drift")
            _validate_page(root, state, prior)
            results.append({"ok": True, "status": "no_op", "page_id": page_id,
                            "reason": "validated existing result retained; revisions require explicit import"})
            continue
        blocks = [{"kind": "table" if v.get("type") == "table" else "body", "text": v["markdown"],
                   "structure": {k: item for k, item in v.items() if k != "markdown"}}
                  for v in row.get("structured_visual", []) if isinstance(v, dict) and isinstance(v.get("markdown"), str)]
        if row.get("visual_text") and all(b["text"] != row["visual_text"] for b in blocks):
            blocks.append({"kind": "body", "text": row["visual_text"]})
        if not blocks:
            continue  # Remains an explicit not_extracted gap, not a successful empty OCR.
        payload = {"schema": PAGE_SCHEMA, "scan_identity": state["identity"], "page_id": page_id,
                   "time": page["time"], "frame_sha256": page["frame_sha256"], "executor": executor,
                   "frame_time": page.get("frame_time", page["time"]),
                   "model_actual": model_actual, "status": "extracted", "blocks": blocks, "gaps": []}
        if executor == "ebook/RapidOCR":
            from .courseware_dedupe import frame_visual_identity
            payload["frame_visual_identity"] = frame_visual_identity(root, page)
        results.append(import_courseware_page(root, payload))
    return {"ok": True, "items": results, "material": build_courseware_material(root) if refresh_material else None}


def add_courseware_candidate(bundle_dir, value: dict) -> dict:
    """Independent reviewers can append a missed flash/animation/annotation."""
    root = Path(bundle_dir).absolute()
    with bundle_write_lock(root, operation="courseware_review_candidate"):
        state = read_scan(root)
        if value.get("scan_identity") != state["identity"] or not value.get("reviewer") or not value.get("reason"):
            raise ValueError("review candidate needs scan identity, reviewer and reason")
        if set(state["completed_windows"]) != set(range(len(state["windows"]))):
            raise ValueError("complete detector scan before independent candidate additions")
        start, end = float(value["time"]), float(value["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("invalid reviewed candidate time")
        if not any(w["start"] <= start < end <= w["end"] for w in state["windows"]):
            raise ValueError("review candidate outside included scope/window")
        if any(abs(p["time"] - start) < 0.001 for p in state["candidates"]):
            raise ValueError("candidate timestamp already registered")
        if len(state["candidates"]) >= 10000:
            raise ValueError("courseware total candidate budget exceeded")
        page_id = "page-" + canonical_json_sha256({"source": state["source"]["sha256"], "time": start})[:20]
        state["candidates"].append({"page_id": page_id, "time": start, "end": end, "frame": "", "frame_sha256": "",
                                    "reason": value["reason"], "reviewer": value["reviewer"], "origin": "independent_review"})
        store_scan(root, state)
    return {"ok": True, "page_id": page_id, "status": "candidate_added", "next_action": "courseware-extract <bundle> --execute"}
