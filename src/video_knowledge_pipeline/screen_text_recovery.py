from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .ocr_backfill import _run_captiocr_candidates, _run_rapidocr_candidates, run_ocr_backfill
from .powershell import quote_powershell_literal as _quote_ps_path
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json

SCREEN_TEXT_RECOVERY_SCHEMA = "lecture_screen_text_recovery_run.v1"


def screen_text_runtime_preflight() -> dict[str, Any]:
    """Check the actual interpreter before planning can write/import anything."""
    try:
        from PIL import Image
        available = callable(Image.open)
        error = "" if available else "PIL.Image.open unavailable"
    except Exception as exc:
        available, error = False, f"{type(exc).__name__}: {exc}"
    return {
        "status": "ready" if available else "blocked_dependency",
        "pillow_available": available,
        "python_executable": sys.executable,
        "error": error,
        "install_command": f"& {_quote_ps_path(sys.executable)} -m pip install 'pillow>=10,<13'",
    }


def run_screen_text_recovery(
    bundle_dir: str | Path,
    *,
    execute_crops: bool = False,
    execute_ocr: bool = False,
    input_json: str | Path | None = None,
    language: str = "chi_sim+eng",
    captiocr_root: str | Path | None = None,
    engine: str = "auto",
    limit: int = 0,
    indexes: list[int] | None = None,
    write: bool = True,
    profile: str = "default",
    max_regions: int = 12,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Plan or execute targeted screen-text recovery without replacing the main OCR branch."""
    if engine not in {"auto", "rapidocr"}:
        raise ValueError("engine must be auto or rapidocr")
    if profile not in {"default", "talking_head"}:
        raise ValueError("profile must be default or talking_head")
    if profile == "talking_head" and (execute_crops or execute_ocr) and not write:
        raise ValueError("oral OCR execution requires write=True")
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"bundle missing timeline.json: {root}")

    runtime_preflight = screen_text_runtime_preflight()
    if (execute_crops or execute_ocr) and runtime_preflight["status"] != "ready":
        raise RuntimeError(
            "screen_text_recovery_blocked_dependency: Pillow unavailable in "
            f"{sys.executable}; {runtime_preflight['error']}; "
            f"install in this interpreter: {runtime_preflight['install_command']}"
        )

    requested_indexes = _normalise_indexes(indexes)
    oral_queue = None
    if profile == "talking_head":
        from .oral_ocr_recovery import build_oral_ocr_recovery_queue
        oral_queue = build_oral_ocr_recovery_queue(root, input_json=input_json,
            max_regions=max_regions, max_attempts=max_attempts, indexes=requested_indexes, write=False)
        plan = {"items": oral_queue["candidates"], "report_path": ""}
    else:
        plan_limit = 0 if indexes else limit
        plan = run_ocr_backfill(root, input_json=input_json, execute=False, language=language, captiocr_root=captiocr_root, limit=plan_limit)
    candidates = [item for item in plan.get("items") or [] if isinstance(item, dict)]
    available_candidates = len(candidates)
    if requested_indexes:
        requested = set(requested_indexes)
        candidates = [item for item in candidates if _int(item.get("index")) in requested]
    if limit and requested_indexes:
        candidates = candidates[:limit]
    if oral_queue and (execute_crops or execute_ocr):
        from .file_hash import sha256_file
        for candidate in candidates:
            if sha256_file(Path(candidate["image_path"])) != candidate["source_sha256"]:
                raise ValueError("oral OCR source changed before crop execution")
    crop_results = _execute_crop_plan(root, candidates) if (execute_crops or execute_ocr) else _preview_crop_plan(candidates)
    ocr_results: list[dict[str, Any]] = []
    ocr_import: dict[str, Any] = {"updated": 0, "updated_indexes": []}
    runner: dict[str, Any] = {"available": False, "name": "", "error": ""}
    import_path = root / "screen-text-recovery-ocr-import.json"

    if execute_ocr:
        crop_candidates = _ocr_candidates_from_crops(crop_results)
        if crop_candidates:
            if engine == "rapidocr":
                runner, ocr_results = _run_rapidocr_candidates(crop_candidates)
            else:
                runner, ocr_results = _run_captiocr_candidates(crop_candidates, language=language, captiocr_root=captiocr_root)
            imported = _ocr_import_payload(root, ocr_results)
            write_json(import_path, imported)
            if imported["items"]:
                ocr_import = run_ocr_backfill(root, input_json=import_path, execute=False, language=language, captiocr_root=captiocr_root, limit=limit)
        else:
            runner = {"available": False, "name": "screen_text_recovery", "error": "no crop images available for OCR"}

    if write and (execute_crops or execute_ocr):
        _write_crop_paths_to_timeline(root, crop_results)
    if write and oral_queue and execute_ocr:
        _record_oral_region_attempts(root, candidates, ocr_results)

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        manifest = {}
    result = {
        "schema": SCREEN_TEXT_RECOVERY_SCHEMA,
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "execute_crops": bool(execute_crops),
        "execute_ocr": bool(execute_ocr),
        "runtime_preflight": runtime_preflight,
        "language": language,
        "engine_requested": engine,
        "profile": profile,
        "oral_recovery_queue": oral_queue,
        "limit": int(limit or 0),
        "requested_indexes": requested_indexes,
        "available_candidates": available_candidates,
        "selected_indexes": [_int(item.get("index")) for item in candidates],
        "ocr_backfill_report_path": plan.get("report_path", ""),
        "crop_summary": _crop_summary(crop_results),
        "ocr_summary": {
            "runner": runner,
            "total": len(ocr_results),
            "succeeded": sum(1 for item in ocr_results if item.get("ok") and str(item.get("text") or "").strip()),
            "updated": int((ocr_import.get("backfill") or ocr_import).get("updated") or 0) if isinstance(ocr_import, dict) else 0,
            "updated_indexes": (ocr_import.get("backfill") or ocr_import).get("updated_indexes", []) if isinstance(ocr_import, dict) else [],
            "import_path": str(import_path) if execute_ocr else "",
        },
        "items": _items_from_plan_and_crops(candidates, crop_results, ocr_results),
    }
    report_path = root / "screen-text-recovery.md"
    json_path = root / "screen-text-recovery.json"
    args_path = root / "mcp-run-screen-text-recovery.args.json"
    result["report_path"] = str(report_path)
    result["json_path"] = str(json_path)
    result["mcp_args_path"] = str(args_path)
    if write:
        with bundle_write_lock(root, operation="screen_text_recovery"):
            write_json(json_path, result)
            report_path.write_text(render_screen_text_recovery_markdown(result), encoding="utf-8")
            write_json(
                args_path,
                {
                    "bundle_dir": str(root),
                    "execute_crops": False,
                    "execute_ocr": False,
                    "language": language,
                    "limit": 0,
                    "indexes": [],
                    "engine": engine,
                    "profile": profile,
                    "max_regions": max_regions,
                    "max_attempts": max_attempts,
                    "write": True,
                },
            )
            manifest["screen_text_recovery"] = {
                "schema": SCREEN_TEXT_RECOVERY_SCHEMA,
                "last_run_at": result["created_at"],
                "report_path": str(report_path),
                "json_path": str(json_path),
                "mcp_args_path": str(args_path),
                "crop_summary": result["crop_summary"],
                "ocr_summary": result["ocr_summary"],
            }
            manifest["mcp_screen_text_recovery_args"] = "mcp-run-screen-text-recovery.args.json"
            write_json(manifest_path, manifest)
        result["run_registry"] = _register_run(root, result)
    return result


def _record_oral_region_attempts(root: Path, candidates: list[dict], results: list[dict]) -> None:
    """Keep attempt budgets durable; crop text remains machine-review evidence."""
    from .file_hash import sha256_file
    path = root / "oral-ocr-recovery-state.json"
    state = read_json(path) if path.is_file() else {"schema": "video_knowledge_pipeline.oral_ocr_recovery_state.v1", "regions": {}}
    by_source = {str(row.get("image_path")): row for row in results}
    for candidate in candidates:
        region = candidate["region_id"]
        previous = state["regions"].get(region, {})
        crop = candidate["screen_text_recovery"]["crop_candidates"][0]
        result = by_source.get(str(crop["planned_output"]))
        if result is None:
            continue  # No engine observation is not an attempt.
        state["regions"][region] = {"attempts": previous.get("attempts", 0) + 1,
            "status": "review-required", "source_image_sha256": candidate["source_sha256"],
            "crop_path": crop["planned_output"],
            "crop_sha256": sha256_file(Path(crop["planned_output"])) if Path(crop["planned_output"]).is_file() else "",
            "last_observation": result, "strict_visual_text_promoted": False}
    write_json(path, state)



def _register_run(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    status = _run_status(result)
    failed_items = _failed_items(result)
    selected = result.get("selected_indexes") if isinstance(result.get("selected_indexes"), list) else []
    retry_indexes = [str(row.get("index")) for row in failed_items if row.get("index")]
    retry_suffix = f" --indexes {','.join(_dedupe(retry_indexes))}" if retry_indexes else (f" --indexes {','.join(str(x) for x in selected)}" if selected else "")
    retry_command = f".\\scripts\\video-knowledge.ps1 run-screen-text-recovery {_quote_ps_path(root)}{retry_suffix} --execute-crops --execute-ocr"
    if result.get('engine_requested') == 'rapidocr':
        retry_command += ' --engine rapidocr'
    crop_summary = result.get("crop_summary") if isinstance(result.get("crop_summary"), dict) else {}
    ocr_summary = result.get("ocr_summary") if isinstance(result.get("ocr_summary"), dict) else {}
    return register_bundle_run(
        root,
        run_type="screen_text_recovery",
        run_id="screen-text-recovery",
        status=status,
        title="Screen text recovery",
        summary=(
            f"Selected {len(selected)} items; crops written={int(crop_summary.get('written') or 0)}; "
            f"OCR updated={int(ocr_summary.get('updated') or 0)}; status={status}."
        ),
        inputs={
            "selected_indexes": selected,
            "requested_indexes": result.get("requested_indexes") or [],
            "available_candidates": result.get("available_candidates"),
        },
        parameters={
            "execute_crops": bool(result.get("execute_crops")),
            "execute_ocr": bool(result.get("execute_ocr")),
            "language": result.get("language", ""),
            "limit": int(result.get("limit") or 0),
        },
        artifacts=[
            {"key": "report", "path": result.get("report_path", "")},
            {"key": "json", "path": result.get("json_path", "")},
            {"key": "mcp_args", "path": result.get("mcp_args_path", "")},
            {"key": "ocr_backfill_report", "path": result.get("ocr_backfill_report_path", "")},
            {"key": "ocr_import", "path": ocr_summary.get("import_path", "")},
        ],
        failed_items=failed_items,
        retry_command=retry_command,
        next_actions=_next_actions(status),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "preview_first": True,
            "does_not_clear_empty_ocr": True,
            "human_review_for_low_confidence": True,
            "purpose": "Expose screen-text crop/OCR recovery status, failures, and retry commands to VKP workbench/task console.",
        },
        write=True,
    )


def _run_status(result: dict[str, Any]) -> str:
    selected = result.get("selected_indexes") if isinstance(result.get("selected_indexes"), list) else []
    if not selected:
        return "not_needed"
    crop_summary = result.get("crop_summary") if isinstance(result.get("crop_summary"), dict) else {}
    ocr_summary = result.get("ocr_summary") if isinstance(result.get("ocr_summary"), dict) else {}
    runner = ocr_summary.get("runner") if isinstance(ocr_summary.get("runner"), dict) else {}
    if not result.get("execute_crops") and not result.get("execute_ocr"):
        return "needs_execution"
    if int(crop_summary.get("failed") or 0) > 0:
        return "needs_retry"
    if result.get("execute_crops") and not result.get("execute_ocr"):
        return "needs_execution"
    if result.get("execute_ocr"):
        if runner.get("error"):
            return "needs_retry"
        updated = int(ocr_summary.get("updated") or 0)
        if updated >= len(selected):
            return "completed"
        return "needs_review"
    return "completed"


def _failed_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    execute_ocr = bool(result.get("execute_ocr"))
    root = Path(str(result.get("bundle_dir") or ".")).expanduser().resolve()
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        index_arg = str(index or "")
        evidence_paths = _screen_text_evidence_paths(item)
        for crop in item.get("crops") or []:
            if isinstance(crop, dict) and crop.get("status") == "failed":
                failed.append(
                    {
                        "index": index,
                        "reason": "crop_failed",
                        "detail": crop.get("error", ""),
                        "suggested_next_tool": "run_screen_text_recovery",
                        "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 run-screen-text-recovery {_quote_ps_path(root)} --indexes {index_arg} --execute-crops",
                        "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session {_quote_ps_path(root)} --limit 0 --group-by reason",
                        "evidence_paths": _dedupe(evidence_paths + [str(crop.get("source_image") or ""), str(crop.get("planned_output") or crop.get("output_path") or "")]),
                    }
                )
        if execute_ocr and not str(item.get("ocr_text") or "").strip():
            failed.append(
                {
                    "index": index,
                    "reason": "ocr_text_empty",
                    "detail": "Crop OCR returned no usable text; keep blocker and route to high-res tile, multimodal, or human review.",
                    "suggested_next_tool": "high_res_tile_plan",
                    "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 high-res-tile-plan {_quote_ps_path(root)} --indexes {index_arg} --execute-tiles",
                    "tile_recovery_command": f".\\scripts\\video-knowledge.ps1 high-res-tile-plan {_quote_ps_path(root)} --indexes {index_arg} --execute-tiles",
                    "multimodal_triage_command": f".\\scripts\\video-knowledge.ps1 vision-review-triage {_quote_ps_path(root)} --indexes {index_arg}",
                    "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session {_quote_ps_path(root)} --limit 0 --group-by reason",
                    "evidence_paths": evidence_paths,
                }
            )
    ocr_summary = result.get("ocr_summary") if isinstance(result.get("ocr_summary"), dict) else {}
    runner = ocr_summary.get("runner") if isinstance(ocr_summary.get("runner"), dict) else {}
    if runner.get("error"):
        failed.append(
            {
                "item": "ocr_runner",
                "reason": "ocr_runner_error",
                "detail": runner.get("error", ""),
                "suggested_next_tool": "run_screen_text_recovery",
                "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 run-screen-text-recovery {_quote_ps_path(root)} --execute-crops --execute-ocr",
                "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session {_quote_ps_path(root)} --limit 0 --group-by reason",
            }
        )
    return failed


def _screen_text_evidence_paths(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if item.get("image_path"):
        values.append(str(item.get("image_path")))
    for crop in item.get("crops") or []:
        if not isinstance(crop, dict):
            continue
        for key in ("output_path", "planned_output", "source_image"):
            value = str(crop.get(key) or "").strip()
            if value:
                values.append(value)
    return _dedupe(values)

def _next_actions(status: str) -> list[str]:
    if status == "needs_execution":
        return ["Run screen text recovery with --execute-crops and, when crops exist, --execute-ocr."]
    if status == "needs_retry":
        return ["Retry failed crop/OCR indexes, then route remaining empty OCR to multimodal or human review."]
    if status == "needs_review":
        return ["Review empty or low-information crop OCR results; do not clear screen-text blockers automatically."]
    if status == "not_needed":
        return ["No screen-text recovery candidates were selected."]
    return ["Refresh coverage/export after accepted screen-text updates."]


def render_screen_text_recovery_markdown(result: dict[str, Any]) -> str:
    crop_summary = result.get("crop_summary") if isinstance(result.get("crop_summary"), dict) else {}
    ocr_summary = result.get("ocr_summary") if isinstance(result.get("ocr_summary"), dict) else {}
    runner = ocr_summary.get("runner") if isinstance(ocr_summary.get("runner"), dict) else {}
    lines = [
        "# Screen Text Recovery",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Execute crops: `{result.get('execute_crops')}`",
        f"- Execute OCR: `{result.get('execute_ocr')}`",
        f"- Runtime preflight: `{(result.get('runtime_preflight') or {}).get('status', 'unknown')}`",
        f"- Dependency recovery: `{(result.get('runtime_preflight') or {}).get('install_command', '')}`",
        f"- Language: `{result.get('language', '')}`",
        f"- Crop planned: `{crop_summary.get('planned', 0)}`",
        f"- Crop written: `{crop_summary.get('written', 0)}`",
        f"- Crop failed: `{crop_summary.get('failed', 0)}`",
        f"- OCR runner: `{runner.get('name', '')}` available=`{runner.get('available', False)}`",
        f"- OCR succeeded: `{ocr_summary.get('succeeded', 0)}`",
        f"- Timeline updated: `{ocr_summary.get('updated', 0)}`",
        f"- OCR import: `{ocr_summary.get('import_path', '')}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Count |",
        "|---|---:|",
    ]
    for key, count in sorted((crop_summary.get("strategy_counts") or {}).items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "## Items",
            "",
            "| Index | Strategy | Crops | OCR text | Image |",
            "|---:|---|---:|---|---|",
        ]
    )
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "| {index} | `{strategy}` | {crop_count} | {ocr_text} | `{image}` |".format(
                index=item.get("index", ""),
                strategy=_md_cell(str(item.get("strategy") or "")),
                crop_count=len(item.get("crops") or []),
                ocr_text=_md_cell(str(item.get("ocr_text") or ""))[:160],
                image=_md_cell(str(item.get("image_path") or "")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _preview_crop_plan(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        recovery = candidate.get("screen_text_recovery") if isinstance(candidate.get("screen_text_recovery"), dict) else {}
        for crop in recovery.get("crop_candidates") or []:
            if isinstance(crop, dict):
                rows.append(_crop_row(candidate, crop, status="planned"))
    return rows


def _execute_crop_plan(root: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional local package
        for candidate in candidates:
            recovery = candidate.get("screen_text_recovery") if isinstance(candidate.get("screen_text_recovery"), dict) else {}
            for crop in recovery.get("crop_candidates") or []:
                if isinstance(crop, dict):
                    rows.append({**_crop_row(candidate, crop, status="failed"), "error": f"Pillow unavailable: {exc}"})
        return rows
    for candidate in candidates:
        recovery = candidate.get("screen_text_recovery") if isinstance(candidate.get("screen_text_recovery"), dict) else {}
        for crop in recovery.get("crop_candidates") or []:
            if not isinstance(crop, dict):
                continue
            row = _crop_row(candidate, crop, status="failed")
            source = Path(str(crop.get("source_image") or candidate.get("image_path") or "")).expanduser()
            target = Path(str(crop.get("planned_output") or "")).expanduser()
            try:
                if not source.is_absolute():
                    source = root / source
                if not target.is_absolute():
                    target = root / target
                if not source.exists():
                    raise FileNotFoundError(f"source image not found: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    width, height = image.size
                    box = _absolute_box(crop.get("box"), width, height)
                    image.crop(box).save(target)
                row.update({"status": "written", "output_path": str(target), "output_exists": True, "error": ""})
            except Exception as exc:  # pragma: no cover - robust report path
                row.update({"status": "failed", "output_path": str(target), "output_exists": False, "error": str(exc)})
            rows.append(row)
    return rows


def _crop_row(candidate: dict[str, Any], crop: dict[str, Any], *, status: str) -> dict[str, Any]:
    recovery = candidate.get("screen_text_recovery") if isinstance(candidate.get("screen_text_recovery"), dict) else {}
    output = str(crop.get("planned_output") or "")
    return {
        "index": candidate.get("index"),
        "strategy": recovery.get("strategy", ""),
        "name": crop.get("name", ""),
        "source_image": crop.get("source_image") or candidate.get("image_path", ""),
        "box": crop.get("box", []),
        "coordinate_system": crop.get("coordinate_system", "relative_xyxy"),
        "purpose": crop.get("purpose", ""),
        "status": status,
        "output_path": output,
        "output_exists": Path(output).expanduser().exists() if output else False,
        "error": "",
    }


def _absolute_box(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        return (0, 0, width, height)
    left, top, right, bottom = [float(part) for part in value]
    if max(left, top, right, bottom) <= 1.0:
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    x1 = max(0, min(int(round(left)), width - 1))
    y1 = max(0, min(int(round(top)), height - 1))
    x2 = max(x1 + 1, min(int(round(right)), width))
    y2 = max(y1 + 1, min(int(round(bottom)), height))
    return (x1, y1, x2, y2)


def _ocr_candidates_from_crops(crop_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in crop_results:
        if row.get("status") != "written" or not row.get("output_exists"):
            continue
        candidates.append(
            {
                "index": row.get("index"),
                "image_path": row.get("output_path", ""),
                "image_exists": True,
                "screen_text_recovery": {"strategy": row.get("strategy", ""), "crop_name": row.get("name", "")},
            }
        )
    return candidates


def _ocr_import_payload(root: Path, ocr_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[str]] = {}
    sources: dict[int, list[str]] = {}
    for row in ocr_results:
        text = str(row.get("text") or "").strip()
        index = _int(row.get("index"))
        if not (row.get("ok") and index and text):
            continue
        if _looks_wrapper_only(text, row.get("image_path")):
            continue
        grouped.setdefault(index, []).append(text)
        sources.setdefault(index, []).append(str(row.get("image_path") or ""))
    items = []
    for index, texts in sorted(grouped.items()):
        items.append(
            {
                "index": index,
                "text": "\n\n".join(_dedupe(texts)),
                "source": "; ".join(_dedupe(sources.get(index, []))),
                "notes": "Imported from screen_text_recovery crop OCR.",
            }
        )
    return {"schema": "lecture_ocr_backfill_input.v1", "generated_at": now_iso(), "bundle_dir": str(root), "items": items}


def _write_crop_paths_to_timeline(root: Path, crop_results: list[dict[str, Any]]) -> None:
    timeline_path = root / "timeline.json"
    timeline = read_json(timeline_path)
    if not isinstance(timeline, list):
        return
    by_index = {_int(item.get("index")): item for item in timeline if isinstance(item, dict)}
    for row in crop_results:
        if row.get("status") != "written":
            continue
        item = by_index.get(_int(row.get("index")))
        if not isinstance(item, dict):
            continue
        recovery = item.setdefault("screen_text_recovery", {})
        if not isinstance(recovery, dict):
            recovery = {}
            item["screen_text_recovery"] = recovery
        crop_paths = recovery.setdefault("crop_paths", [])
        if isinstance(crop_paths, list) and row.get("output_path") not in crop_paths:
            crop_paths.append(row.get("output_path"))
    write_json(timeline_path, timeline)


def _items_from_plan_and_crops(candidates: list[dict[str, Any]], crop_results: list[dict[str, Any]], ocr_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    crops_by_index: dict[int, list[dict[str, Any]]] = {}
    text_by_index: dict[int, list[str]] = {}
    for crop in crop_results:
        crops_by_index.setdefault(_int(crop.get("index")), []).append(crop)
    for ocr in ocr_results:
        if ocr.get("ok") and str(ocr.get("text") or "").strip() and not _looks_wrapper_only(str(ocr.get("text") or ""), ocr.get("image_path")):
            text_by_index.setdefault(_int(ocr.get("index")), []).append(str(ocr.get("text") or "").strip())
    items = []
    for candidate in candidates:
        recovery = candidate.get("screen_text_recovery") if isinstance(candidate.get("screen_text_recovery"), dict) else {}
        index = _int(candidate.get("index"))
        items.append(
            {
                "index": index,
                "strategy": recovery.get("strategy", ""),
                "recommended_tool": recovery.get("recommended_tool", ""),
                "image_path": candidate.get("image_path", ""),
                "quality_issues": recovery.get("issues", []),
                "crops": crops_by_index.get(index, []),
                "ocr_text": "\n\n".join(_dedupe(text_by_index.get(index, []))),
            }
        )
    return items


def _crop_summary(crop_results: list[dict[str, Any]]) -> dict[str, Any]:
    strategies: dict[str, int] = {}
    for row in crop_results:
        strategy = str(row.get("strategy") or "unknown")
        strategies[strategy] = strategies.get(strategy, 0) + 1
    return {
        "planned": len(crop_results),
        "written": sum(1 for row in crop_results if row.get("status") == "written"),
        "failed": sum(1 for row in crop_results if row.get("status") == "failed"),
        "strategy_counts": dict(sorted(strategies.items())),
    }


def _looks_wrapper_only(text: str, image_path: Any) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return True
    stem = Path(str(image_path or "")).stem
    meaningful = []
    for line in lines:
        if line.startswith("<!--") and line.endswith("-->") and "source:" in line.lower():
            continue
        if line.startswith("# ") and line[2:].strip() == stem:
            continue
        meaningful.append(line)
    return not meaningful


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
