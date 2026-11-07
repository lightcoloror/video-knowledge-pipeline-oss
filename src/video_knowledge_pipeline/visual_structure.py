from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .config import ebook_pipeline_profile, runtime_config_manifest, service_url
from .frame_recapture import (
    _coverage_audit,
    _quality_audit,
    _quality_issues,
    _quality_score,
)
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .path_defaults import tool_source_review_root, workspace_root
from .powershell import quote_powershell_literal as _ps_quote
from .repair_status import build_repair_status
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .visual_integration import integrated_visual
from .canonical_json import canonical_json_sha256
from .courseware_material import safe_path
from .file_hash import sha256_file

STRUCTURED_MATERIAL_TYPES = {"formula", "table", "code"}
DOCUMENT_VISUAL_TYPES = {"formula", "table", "code", "board", "slide", "document", "diagram", "text"}
DEFAULT_EBOOK_ROUTES = {"document_visual", "mixed", "semantic_frame", "temporal_sequence"}
EBOOK_CHECKPOINT_SCHEMA = "video_knowledge_pipeline.ebook_ocr_item_checkpoint.v1"
EBOOK_BOUND_CHECKPOINT_SCHEMA = "video_knowledge_pipeline.ebook_ocr_item_checkpoint.v2"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
NON_GENERIC_DOCUMENT_VISUAL_TYPES = DOCUMENT_VISUAL_TYPES - {"text"}
EXPLICIT_VISUAL_STRUCTURE_ISSUES = {
    "ocr_text_empty",
    "screen_text_low_confidence",
    "structured_visual_without_structure",
}
TILE_RECOVERABLE_EBOOK_BLOCKERS = {"ocr_wrapper_only", "ocr_text_empty", "ocr_text_low_information"}

ACCEPTED_REVIEW_STATUSES = {
    "accepted",
    "reviewed",
    "keep_image",
    "accepted_known_gap",
    "accepted_no_visual_content",
    "accepted_provider_blocked",
    "corrected_visual_text",
    "corrected_visual_understanding",
    "corrected_temporal_visual_understanding",
}


def run_visual_structure_plan(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    execute_ebook_pipeline: bool = False,
    include_routes: list[str] | None = None,
    timeout_seconds: int = 120,
    indexes: list[int] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Preview or import structured visual extraction for a WebUI lecture bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    # Intent: make completed per-frame OCR survive a later process interruption.
    # Decision: reconcile atomic item checkpoints and legacy on-disk Markdown
    # before selecting new work. Reason: an OCR artifact can be complete while
    # the former batch-level Timeline write has not happened yet. Evidence: the
    # long-video run preserved 12 artifacts on disk but its next pass only saw
    # the 11 rows written after restart. Effective scope: derived OCR/structured
    # visual evidence only; original frames, ASR, and Timeline provenance remain
    # authoritative and no OCR call is performed here.
    checkpoint_reconciliation = reconcile_ebook_pipeline_checkpoints(root, write=True)
    manifest = read_json(manifest_path)

    timeline = _read_timeline(root)
    routes = {str(route) for route in (include_routes or sorted(DEFAULT_EBOOK_ROUTES)) if str(route)}
    all_candidates = _visual_structure_candidates(root, timeline, include_routes=routes)
    candidates = _select_candidates(all_candidates, indexes=indexes, limit=limit)
    template_path = write_visual_structure_input_template(root, candidates)
    tools = _tool_statuses()
    ebook_results = _run_ebook_pipeline_candidates(root, candidates, timeout_seconds=timeout_seconds) if execute_ebook_pipeline else []
    imported_entries = _read_visual_structure_input(input_json, default_index=_single_candidate_index(candidates)) if input_json else []
    imported_entries.extend(_ebook_results_to_import_rows(ebook_results))
    backfill = (
        _backfill_visual_structure_results(root, manifest, timeline, imported_entries)
        if imported_entries
        else {"updated": 0, "updated_indexes": [], "source_package_updated": False}
    )
    ebook_status_backfill = _backfill_ebook_pipeline_status(root, manifest, timeline, ebook_results) if ebook_results else {"updated": 0, "updated_indexes": []}
    timeline = _read_timeline(root)
    # Project the same selected batch from persisted evidence after import.
    # Never select a new batch or use the number of writes as proof of coverage.
    refreshed = {item["index"]: item for item in _visual_structure_candidates(root, timeline, include_routes=routes)}
    candidates = [refreshed.get(item["index"], {**item, "visual_text": "", "structured_visual": []}) for item in candidates]
    unresolved_indexes = [item["index"] for item in candidates if not _candidate_has_meaningful_visual_structure(item)]
    summary = {
        "total_candidates": len(candidates),
        "available_candidates": len(all_candidates),
        "completion_scope": "selected_candidates_only",
        "bundle_completeness": "not_assessed",
        "unselected_candidate_count": len(all_candidates) - len(candidates),
        "structured_count": len(candidates) - len(unresolved_indexes),
        "unresolved_indexes": unresolved_indexes,
        "selected_indexes": [item.get("index") for item in candidates],
        "requested_indexes": [int(value) for value in (indexes or []) if _int_value(value)],
        "limit": int(limit or 0),
        "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
        "input_template_json": str(template_path),
        "imported": len(imported_entries),
        "execute_ebook_pipeline": execute_ebook_pipeline,
        "ebook_pipeline_total": len(ebook_results),
        "ebook_pipeline_succeeded": sum(1 for item in ebook_results if item.get("ok")),
        "ebook_pipeline_blockers": _count_ebook_blockers(ebook_results),
        "checkpoint_reconciliation": checkpoint_reconciliation,
        "include_routes": sorted(routes),
        "updated": backfill.get("updated", 0),
        "ebook_status_updated": ebook_status_backfill.get("updated", 0),
        "updated_at": now_iso(),
        "runtime_config": runtime_config_manifest(),
    }
    manifest["coverage"] = _coverage_audit(timeline)
    manifest["visual_structure"] = {
        "schema": "lecture_visual_structure_plan.v1",
        "count": len(candidates),
        "tools": tools,
        "items": candidates,
        "input_template_json": str(template_path),
        "last_import": {**backfill, "ebook_status_backfill": ebook_status_backfill, "updated_at": now_iso()},
        "last_run": summary,
        "ebook_pipeline_results": ebook_results,
        "notes": [
            "This command extracts text/layout evidence from routed visual frames with ebook_markdown_pipeline.",
            "document_visual frames use this as the primary visual text/layout branch; semantic_frame and temporal_sequence frames also use it for any text that can be lowered from screenshots.",
            "Prefer ebook_markdown_pipeline process_material -> get_job_status -> read_artifact over direct MinerU/Marker/PaddleOCR commands.",
        ],
    }
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    write_json(manifest_path, manifest)
    report_path = root / "visual-structure-report.md"
    report_path.write_text(_render_visual_structure_report(root, candidates, tools, summary, template_path, ebook_results), encoding="utf-8")
    handoff_path = root / "visual-structure-handoff.md"
    handoff_json_path = root / "visual-structure-handoff.json"
    handoff = _build_visual_structure_handoff(
        root,
        manifest,
        candidates,
        tools,
        summary,
        template_path,
        report_path,
        handoff_path,
        handoff_json_path,
    )
    write_json(handoff_json_path, handoff)
    handoff_path.write_text(_render_visual_structure_handoff_markdown(handoff), encoding="utf-8")
    manifest["visual_structure"]["handoff_markdown"] = str(handoff_path)
    manifest["visual_structure"]["handoff_json"] = str(handoff_json_path)
    write_json(manifest_path, manifest)
    run_artifact = _register_visual_structure_run(
        root,
        summary=summary,
        candidates=candidates,
        ebook_results=ebook_results,
        report_path=report_path,
        handoff_path=handoff_path,
        handoff_json_path=handoff_json_path,
        template_path=template_path,
        execute_ebook_pipeline=execute_ebook_pipeline,
        include_routes=sorted(routes),
        indexes=indexes,
        limit=limit,
        timeout_seconds=timeout_seconds,
    )
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "handoff_path": str(handoff_path),
        "handoff_json_path": str(handoff_json_path),
        "input_template_json": str(template_path),
        "run_artifact": run_artifact,
        "summary": summary,
        "tools": tools,
        "backfill": backfill,
        "items": candidates,
    }


def _register_visual_structure_run(
    root: Path,
    *,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    ebook_results: list[dict[str, Any]],
    report_path: Path,
    handoff_path: Path,
    handoff_json_path: Path,
    template_path: Path,
    execute_ebook_pipeline: bool,
    include_routes: list[str],
    indexes: list[int] | None,
    limit: int | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    blockers = summary.get("ebook_pipeline_blockers") if isinstance(summary.get("ebook_pipeline_blockers"), dict) else {}
    failed_items = _visual_structure_failed_items(root, ebook_results) if execute_ebook_pipeline else []
    total_candidates = int(summary.get("total_candidates") or 0)
    succeeded = int(summary.get("ebook_pipeline_succeeded") or 0)
    unresolved_indexes = [item["index"] for item in candidates if not _candidate_has_meaningful_visual_structure(item)]
    if total_candidates == 0 and not candidates:
        status = "not_needed"
    elif failed_items or blockers:
        status = "needs_retry"
    elif len(candidates) != total_candidates or unresolved_indexes:
        status = "needs_retry" if execute_ebook_pipeline else "needs_execution"
    else:
        status = "completed"
    retry_command = _visual_structure_retry_command(
        root,
        execute_ebook_pipeline=execute_ebook_pipeline or status == "needs_execution",
        include_routes=include_routes,
        indexes=unresolved_indexes or indexes,
        limit=None if unresolved_indexes else limit,
        timeout_seconds=timeout_seconds,
    )
    if status in {"completed", "not_needed"}:
        retry_command = ""
    summary_text = (
        f"{total_candidates} candidates; ebook executed={execute_ebook_pipeline}; "
        f"succeeded={succeeded}; unresolved={len(unresolved_indexes)}; "
        f"scope=selected_candidates_only; blockers={dict(blockers)}."
    )
    next_actions: list[str] = []
    if status == "needs_execution":
        next_actions.append("Run visual structure with --execute-ebook-pipeline before treating screen text as recovered.")
    if status == "needs_retry":
        next_actions.append("Retry ebook batch after fixing external parser blockers, or route failed frames to high-res tile recovery / multimodal review / human review.")
    if failed_items:
        next_actions.append("Failed timeline indexes are recorded in run.json failed_items for task-console retry panels.")
    return register_bundle_run(
        root,
        run_type="visual_structure_ebook",
        run_id="visual-structure-ebook",
        status=status,
        title="Visual structure ebook batch",
        summary=summary_text,
        inputs={
            "selected_indexes": summary.get("selected_indexes") or [],
            "requested_indexes": summary.get("requested_indexes") or [],
            "include_routes": include_routes,
            "completion_scope": "selected_candidates_only",
            "bundle_completeness": "not_assessed",
            "unresolved_indexes": unresolved_indexes,
        },
        parameters={
            "execute_ebook_pipeline": execute_ebook_pipeline,
            "limit": int(limit or 0),
            "timeout_seconds": int(timeout_seconds or 0),
        },
        artifacts=[
            {"key": "report", "path": str(report_path), "description": "Human-readable visual structure and ebook batch report."},
            {"key": "handoff_markdown", "path": str(handoff_path), "description": "MCP/handoff markdown for external or manual structured visual imports."},
            {"key": "handoff_json", "path": str(handoff_json_path), "description": "Machine-readable visual structure handoff."},
            {"key": "input_template", "path": str(template_path), "description": "JSON template for manual/external visual structure imports."},
        ],
        failed_items=failed_items,
        retry_command=retry_command,
        next_actions=next_actions,
        operator_boundary={
            "local_only": True,
            "uses_ebook_markdown_pipeline": bool(execute_ebook_pipeline),
            "no_cloud_vision_call": True,
            "does_not_download_media": True,
            "empty_or_wrapper_only_results_do_not_clear_blockers": True,
        },
        write=True,
    )


def _visual_structure_failed_items(root: Path, ebook_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for result in ebook_results:
        if not isinstance(result, dict) or result.get("ok"):
            continue
        quality = result.get("ebook_quality") if isinstance(result.get("ebook_quality"), dict) else {}
        blocker = str(result.get("blocker") or quality.get("quality") or "ebook_pipeline_failed")
        index = _int_value(result.get("index"))
        item = {
            "index": index,
            "reason": blocker,
            "detail": str(result.get("next_action") or quality.get("reason") or result.get("error") or ""),
            "image_path": str(result.get("image_path") or ""),
            "output_dir": str(result.get("output_dir") or ""),
            "artifact_path": str((result.get("artifact") if isinstance(result.get("artifact"), dict) else {}).get("path") or ""),
            "evidence_paths": _visual_structure_failed_evidence_paths(result),
            "ebook_retry_command": _visual_structure_index_retry_command(root, index),
            "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session {_ps_quote(str(root))} --group-by reason",
        }
        if blocker in TILE_RECOVERABLE_EBOOK_BLOCKERS:
            item.update(
                {
                    "suggested_next_tool": "high_res_tile_plan",
                    "suggested_next_reason": "Whole-frame ebook/OCR returned empty, wrapper-only, or low-information text; prepare local high-resolution tile evidence before cloud VLM or human review.",
                    "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 high-res-tile-plan {_ps_quote(str(root))} --indexes {_ps_quote(str(index))} --execute-tiles",
                    "tile_recovery_command": f".\\scripts\\video-knowledge.ps1 high-res-tile-plan {_ps_quote(str(root))} --indexes {_ps_quote(str(index))} --execute-tiles",
                    "multimodal_triage_command": f".\\scripts\\video-knowledge.ps1 vision-review-triage {_ps_quote(str(root))} --indexes {_ps_quote(str(index))}",
                }
            )
        else:
            item.update(
                {
                    "suggested_next_tool": "run_visual_structure_plan",
                    "suggested_next_reason": "Repair the local ebook_markdown_pipeline blocker, then retry the same timeline index before routing to tile/multimodal review.",
                    "suggested_retry_command": _visual_structure_index_retry_command(root, index),
                }
            )
        failed.append(item)
    return failed


def _visual_structure_failed_evidence_paths(result: dict[str, Any]) -> list[str]:
    values = [
        str(result.get("image_path") or ""),
        str(result.get("output_dir") or ""),
        str((result.get("artifact") if isinstance(result.get("artifact"), dict) else {}).get("path") or ""),
    ]
    return [value for value in values if value]


def _visual_structure_index_retry_command(root: Path, index: int) -> str:
    parts = [
        ".\\scripts\\video-knowledge.ps1",
        "run-visual-structure",
        _ps_quote(str(root)),
        "--execute-ebook-pipeline",
    ]
    if index:
        parts.extend(["--indexes", _ps_quote(str(index))])
    return " ".join(parts)

def _visual_structure_retry_command(
    root: Path,
    *,
    execute_ebook_pipeline: bool,
    include_routes: list[str],
    indexes: list[int] | None,
    limit: int | None,
    timeout_seconds: int,
) -> str:
    parts = [
        ".\\scripts\\video-knowledge.ps1",
        "run-visual-structure",
        _ps_quote(str(root)),
    ]
    if execute_ebook_pipeline:
        parts.append("--execute-ebook-pipeline")
    if include_routes:
        parts.extend(["--include-routes", _ps_quote(",".join(include_routes))])
    if indexes:
        values = [str(int(value)) for value in indexes if _int_value(value)]
        if values:
            parts.extend(["--indexes", _ps_quote(",".join(values))])
    if limit and int(limit) > 0:
        parts.extend(["--limit", str(int(limit))])
    if timeout_seconds:
        parts.extend(["--timeout-seconds", str(int(timeout_seconds))])
    return " ".join(parts)
def _read_timeline(root: Path) -> list[dict[str, Any]]:
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        return []
    timeline = read_json(timeline_path)
    return [item for item in timeline if isinstance(item, dict)] if isinstance(timeline, list) else []


def _visual_structure_candidates(root: Path, timeline: list[dict[str, Any]], *, include_routes: set[str] | None = None) -> list[dict[str, Any]]:
    candidates = []
    routes = include_routes or DEFAULT_EBOOK_ROUTES
    for index, item in enumerate(timeline, start=1):
        if _is_review_closed(item):
            continue
        material_types = {str(value) for value in item.get("material_types") or []}
        structured_types = sorted(material_types & STRUCTURED_MATERIAL_TYPES)
        visual_route = str(item.get("visual_route") or "")
        secondary_routes = {str(value) for value in item.get("secondary_visual_routes") or []}
        route_set = {visual_route, *secondary_routes}
        quality_issues = {str(value) for value in item.get("quality_issues") or []}
        route_matches = (visual_route in routes) or bool(secondary_routes & routes)
        if visual_route and not route_matches:
            continue
        should_try_document_parser = bool(
            route_set & {"document_visual", "mixed"}
            or structured_types
            or material_types & NON_GENERIC_DOCUMENT_VISUAL_TYPES
            or quality_issues & EXPLICIT_VISUAL_STRUCTURE_ISSUES
        )
        if not should_try_document_parser:
            continue
        if (
            not route_matches
            and not structured_types
            and "document_visual" not in ({visual_route} | secondary_routes)
            and "mixed" not in ({visual_route} | secondary_routes)
        ):
            continue
        if not structured_types:
            structured_types = sorted(material_types & DOCUMENT_VISUAL_TYPES) or ["text"]
        image_path = _first_existing_image(root, item)
        commands = _candidate_commands(image_path, root, index)
        candidates.append(
            {
                "index": int(item.get("index") or index),
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "material_types": structured_types,
                "visual_route": visual_route,
                "secondary_visual_routes": sorted(secondary_routes),
                "quality_issues": item.get("quality_issues", []),
                "image_path": str(image_path) if image_path else "",
                "image_exists": bool(image_path and image_path.exists()),
                "visual_text": item.get("visual_text", ""),
                "structured_visual": item.get("structured_visual", []),
                "routing_decision": _visual_structure_routing_decision(visual_route, material_types, secondary_routes),
                "commands": commands,
            }
        )
    return candidates


def _visual_structure_routing_decision(visual_route: str, material_types: set[str], secondary_routes: set[str]) -> dict[str, Any]:
    route_set = {visual_route, *secondary_routes}
    if "document_visual" in route_set:
        return {
            "primary_branch": "document_visual_parser",
            "primary_tool": "ebook_markdown_pipeline",
            "also_requires_multimodal": False,
            "reason": "PPT/板书/表格/公式/代码/文档页优先用 ebook_markdown_pipeline 做结构化图文解析。",
        }
    if "mixed" in route_set:
        return {
            "primary_branch": "document_visual_parser",
            "primary_tool": "ebook_markdown_pipeline",
            "also_requires_multimodal": True,
            "reason": "混合画面既要用 ebook_markdown_pipeline 降维图文，也要保留多模态理解非文字信息。",
        }
    if material_types & DOCUMENT_VISUAL_TYPES:
        return {
            "primary_branch": "document_visual_parser",
            "primary_tool": "ebook_markdown_pipeline",
            "also_requires_multimodal": visual_route in {"semantic_frame", "temporal_sequence"},
            "reason": "检测到文本/图表/代码/公式材料；先尝试结构化图文解析，非图文语义另走多模态。",
        }
    return {
        "primary_branch": "ocr_auxiliary",
        "primary_tool": "ebook_markdown_pipeline",
        "also_requires_multimodal": visual_route in {"semantic_frame", "temporal_sequence"},
        "reason": "此分支仅补可降维的屏幕文字，不替代视频画面理解。",
    }


def _is_review_closed(item: dict[str, Any]) -> bool:
    human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    status = str(item.get("review_status") or human_review.get("status") or "").strip().lower()
    return status in ACCEPTED_REVIEW_STATUSES


def _select_candidates(candidates: list[dict[str, Any]], *, indexes: list[int] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    selected = list(candidates)
    wanted = {_int_value(value) for value in (indexes or []) if _int_value(value)}
    if wanted:
        selected = [item for item in selected if _int_value(item.get("index")) in wanted]
    count = int(limit or 0)
    if count > 0:
        selected = selected[:count]
    return selected


def _first_existing_image(root: Path, item: dict[str, Any]) -> Path | None:
    paths = []
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        raw = str(asset.get("source") or asset.get("resolved_path") or asset.get("path") or "")
        if raw:
            paths.append(raw)
    paths.extend(str(path) for path in item.get("frame_paths") or [] if path)
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path.resolve()
    return None


def _tool_statuses() -> list[dict[str, Any]]:
    local_roots = {
        "docling": tool_source_review_root() / "docling",
        "marker": tool_source_review_root() / "marker",
        "mineru": workspace_root() / "mineru-full-output",
        "paddleocr": tool_source_review_root() / "PaddleOCR",
    }
    command_names = {
        "docling": ("docling",),
        "marker": ("marker_single", "marker"),
        "mineru": ("mineru",),
        "paddleocr": ("paddleocr",),
    }
    rows = []
    for name in ("docling", "marker", "mineru", "paddleocr"):
        commands = [path for command in command_names[name] if (path := shutil.which(command))]
        root = local_roots[name]
        rows.append(
            {
                "name": name,
                # Compatibility field means command discovery only. Neither a
                # source checkout nor PATH discovery proves executable readiness.
                "available": bool(commands),
                "availability_scope": "command_discovery_only",
                "command_available": bool(commands),
                "local_root_exists": root.is_dir(),
                "runtime_status": "not_probed" if commands else "command_missing",
                "model_status": "not_assessed",
                "ready_to_execute": False,
                "probe_executed": False,
                "blockers": ["runtime_not_probed" if commands else "command_missing", "models_not_assessed"],
                "commands": commands,
                "local_root": str(root) if root.is_dir() else "",
                "role": _tool_role(name),
            }
        )
    return rows


def _tool_role(name: str) -> str:
    roles = {
        "docling": "general image/document layout, table, code, formula to Markdown/JSON",
        "marker": "image/PDF to Markdown/JSON with layout blocks, equations, tables",
        "mineru": "complex document/image parsing with formula/table recognition",
        "paddleocr": "OCR, PP-Structure table recovery, formula recognition",
    }
    return roles.get(name, "")


def _candidate_commands(image_path: Path | None, root: Path, index: int) -> dict[str, str]:
    if not image_path:
        return {}
    out_dir = root / "visual-structure" / f"timeline-{index:04d}"
    ebook_http_url = service_url("ebook_markdown_pipeline_http")
    return {
        "ebook_pipeline_mcp": (
            "process_material -> get_job_status -> read_artifact "
            f"(material_path={_ps_quote(str(image_path))}, output_dir={_ps_quote(str(out_dir / 'ebook_pipeline'))})"
        ),
        "ebook_pipeline_http": (
            f"POST {ebook_http_url} "
            f"name=process_material material_path={_ps_quote(str(image_path))}"
        ),
        "docling": f"reference only; prefer ebook_pipeline_mcp: docling {_ps_quote(str(image_path))} --output {_ps_quote(str(out_dir / 'docling'))}",
        "marker": f"reference only; prefer ebook_pipeline_mcp: marker_single {_ps_quote(str(image_path))} --output_dir {_ps_quote(str(out_dir / 'marker'))}",
        "mineru": f"reference only; prefer ebook_pipeline_mcp: mineru -p {_ps_quote(str(image_path))} -o {_ps_quote(str(out_dir / 'mineru'))}",
        "paddleocr": (
            "reference only; prefer ebook_pipeline_mcp: paddleocr "
            f"--image_dir={_ps_quote(str(image_path))} "
            "--type=structure --recovery=true --formula=true --recovery_to_markdown=true --lang=ch"
        ),
    }


def _ebook_input_identity(root: Path, *, index: int, image_path: str) -> dict[str, Any]:
    """Selected interval + existing local image, not whole-media truth."""
    timeline = _read_timeline(root)
    if not 1 <= index <= len(timeline):
        raise ValueError('ebook input identity index out of range')
    item = timeline[index - 1]
    path = safe_path(root, image_path)
    if path.suffix.lower() not in IMAGE_SUFFIXES or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('ebook input identity image budget/type')
    if _first_existing_image(root, item) != path:
        raise ValueError('ebook input identity frame not selected in timeline')
    return dict(index=index, start=item.get('start', 0), end=item.get('end', 0),
                frame_path=path.relative_to(root).as_posix(), frame_sha256=sha256_file(path))


def _ebook_result_binding(root: Path, result: dict, before: dict) -> dict:
    from .evidence_locator import TextEvidenceSnapshot
    # The output's declared image is also part of the identity. Rechecking only
    # the saved input would allow a checkpoint to point at a different frame.
    image_path = result.get('image_path')
    if (not isinstance(image_path, str) or not image_path
            or safe_path(root, image_path) != safe_path(root, before['frame_path'])):
        raise ValueError('ebook result image identity mismatch')
    if _ebook_input_identity(root, index=before['index'], image_path=before['frame_path']) != before:
        raise ValueError('ebook input identity drift during OCR')
    artifact = result.get('artifact') or {}
    path = safe_path(root, str(artifact.get('path') or ''))
    output = root / 'visual-structure' / f"timeline-{before['index']:04d}" / 'ebook_pipeline'
    if not path.is_relative_to(output) or path.suffix.lower() not in {'.md', '.txt', '.json', '.jsonl'}:
        raise ValueError('ebook artifact identity outside output or unsupported type')
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('ebook artifact identity budget exceeded')
    relative = path.relative_to(root).as_posix()
    raw_text = _ebook_markdown(result)
    if path.suffix.lower() in {'.md', '.txt'}:
        snapshot = TextEvidenceSnapshot.read(root, relative)
        # Match the existing read_text newline projection, retain raw-byte Hash.
        if snapshot.data.replace('\r\n', '\n').replace('\r', '\n').strip() != raw_text:
            raise ValueError('ebook artifact identity text mismatch')
        artifact_hash = snapshot.artifact_sha256
    else:
        artifact_hash = sha256_file(path)
    body = dict(schema='video_knowledge_pipeline.ebook_recognition_binding.v1', input=before,
                artifact_path=relative, artifact_sha256=artifact_hash,
                recognized_text_sha256=canonical_json_sha256(raw_text),
                projected_text_sha256=canonical_json_sha256(_meaningful_ebook_markdown(raw_text, image_path=result['image_path'])))
    return {**body, 'binding_sha256': canonical_json_sha256(body)}


def _validate_ebook_binding(root: Path, binding: Any, result: dict | None = None) -> None:
    expected = {'schema', 'input', 'artifact_path', 'artifact_sha256', 'recognized_text_sha256', 'projected_text_sha256', 'binding_sha256'}
    if (not isinstance(binding, dict) or set(binding) != expected
            or binding.get('schema') != 'video_knowledge_pipeline.ebook_recognition_binding.v1'
            or canonical_json_sha256({k: v for k, v in binding.items() if k != 'binding_sha256'}) != binding['binding_sha256']):
        raise ValueError('ebook checkpoint identity malformed or drifted')
    source = binding['input']
    if not isinstance(source, dict) or type(source.get('index')) is not int:
        raise ValueError('ebook checkpoint identity input invalid')
    if _ebook_input_identity(root, index=source['index'], image_path=str(source.get('frame_path') or '')) != source:
        raise ValueError('ebook checkpoint input identity drift')
    path = safe_path(root, binding['artifact_path'])
    if path.stat().st_size > 64 * 1024 * 1024 or sha256_file(path) != binding['artifact_sha256']:
        raise ValueError('ebook checkpoint artifact identity drift')
    if result is not None:
        if result.get('index') != source['index'] or _ebook_result_binding(root, result, source) != binding:
            raise ValueError('ebook checkpoint result identity drift')


def _ebook_recognition_provenance(binding: dict) -> dict:
    return dict(schema='video_knowledge_pipeline.recognition_provenance.v1',
                executor='ebook_markdown_pipeline', model_actual='', model_requested='',
                input_identity_basis='ebook_pre_post_input_and_checkpoint_binding',
                frame_sha256s=[binding['input']['frame_sha256']], context_text_sha256s=[],
                observation_text_sha256=binding['projected_text_sha256'], independence_is_verified=False)


def _run_ebook_pipeline_candidates(root: Path, candidates: list[dict[str, Any]], *, timeout_seconds: int) -> list[dict[str, Any]]:
    results = []
    call_tool = _ebook_call_tool()
    with _ebook_pipeline_environment(ebook_pipeline_profile()):
        for candidate in candidates:
            image_path = str(candidate.get("image_path") or "")
            output_dir = root / "visual-structure" / f"timeline-{int(candidate.get('index') or 0):04d}" / "ebook_pipeline"
            result = {
                "index": candidate.get("index"),
                "image_path": image_path,
                "output_dir": str(output_dir),
                "ok": False,
                "error": "",
                "blocker": "",
                "next_action": "",
                "routed": {},
                "job": {},
                "artifact": {},
            }
            if not image_path:
                result["error"] = "missing image_path"
                result.update(_classify_ebook_blocker(result["error"]))
                _write_ebook_result_checkpoint(output_dir, result)
                results.append(result)
                continue
            try:
                before = _ebook_input_identity(root, index=int(candidate.get('index') or 0), image_path=image_path)
                result.update(_run_ebook_material_flow(call_tool, image_path, output_dir, timeout_seconds=timeout_seconds))
                if result.get('ok'):
                    result['recognition_binding'] = _ebook_result_binding(root, result, before)
            except Exception as exc:  # noqa: BLE001 - persisted as repair evidence for agents.
                result['ok'] = False
                result["error"] = str(exc)
            if not result.get("ok"):
                result.update(_classify_ebook_blocker(str(result.get("error") or ""), result=result))
            _write_ebook_result_checkpoint(output_dir, result)
            results.append(result)
    return results


def _write_ebook_result_checkpoint(output_dir: Path, result: dict[str, Any]) -> Path:
    """Atomically persist one OCR result before the batch advances."""
    checkpointed_at = str(result.get("checkpointed_at") or now_iso())
    result["checkpointed_at"] = checkpointed_at
    path = output_dir / "result-checkpoint.json"
    write_json(
        path,
        {
            "schema": EBOOK_BOUND_CHECKPOINT_SCHEMA if result.get('recognition_binding') else EBOOK_CHECKPOINT_SCHEMA,
            "checkpointed_at": checkpointed_at,
            "result": result,
        },
    )
    return path


def reconcile_ebook_pipeline_checkpoints(
    bundle_dir: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Recover completed OCR artifacts that were not registered in Timeline.

    The scan is deliberately local and fail-closed: only timeline-namespaced
    directories below the Bundle are considered, artifact paths must stay below
    the Bundle, and wrapper-only/empty Markdown is never promoted to evidence.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    timeline = _read_timeline(root)
    results: dict[int, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    checkpoint_indexes: set[int] = set()
    checkpoint_paths = sorted(
        (root / "visual-structure").glob(
            "timeline-*/ebook_pipeline/result-checkpoint.json"
        )
    )
    for checkpoint_path in checkpoint_paths:
        directory_index = _int_value(checkpoint_path.parent.parent.name.removeprefix('timeline-'))
        checkpoint_indexes.add(directory_index)
        try:
            payload = read_json(safe_path(root, str(checkpoint_path)))
        except (OSError, ValueError) as exc:
            invalid.append({"path": str(checkpoint_path), "reason": str(exc)})
            continue
        if not isinstance(payload, dict) or payload.get("schema") not in {EBOOK_CHECKPOINT_SCHEMA, EBOOK_BOUND_CHECKPOINT_SCHEMA}:
            invalid.append({"path": str(checkpoint_path), "reason": "checkpoint_schema_mismatch"})
            continue
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        index = _int_value(result.get("index"))
        if not (1 <= index <= len(timeline)) or index != directory_index:
            invalid.append({"path": str(checkpoint_path), "reason": "timeline_index_out_of_range"})
            continue
        result = dict(result)
        if payload['schema'] == EBOOK_BOUND_CHECKPOINT_SCHEMA and 'recognition_binding' not in result:
            invalid.append({'path': str(checkpoint_path), 'reason': 'bound_checkpoint_missing_identity'})
            continue
        if 'recognition_binding' in result:
            try:
                _validate_ebook_binding(root, result['recognition_binding'], result)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                invalid.append({'path': str(checkpoint_path), 'reason': str(exc)})
                continue
        result["checkpointed_at"] = str(
            result.get("checkpointed_at")
            or payload.get("checkpointed_at")
            or now_iso()
        )
        results[index] = result

    # Backward compatibility for artifacts produced before item checkpoints
    # existed. Prefer a usable Markdown artifact and bind it to the directory's
    # explicit timeline index; never infer an index from a title or timestamp.
    visual_root = root / "visual-structure"
    if visual_root.is_dir():
        for output_dir in sorted(visual_root.glob("timeline-*/ebook_pipeline")):
            try:
                index = int(output_dir.parent.name.removeprefix("timeline-"))
            except ValueError:
                continue
            if index in checkpoint_indexes or index in results or not (1 <= index <= len(timeline)):
                continue
            image_path = _candidate_image_path(timeline[index - 1])
            usable: tuple[Path, str, dict[str, Any]] | None = None
            for artifact_path in sorted(output_dir.rglob("*.md")):
                if not artifact_path.is_file() or not _path_is_within(artifact_path, root):
                    continue
                markdown = artifact_path.read_text(
                    encoding="utf-8-sig", errors="replace"
                )
                quality = _ebook_markdown_quality(markdown, image_path=image_path)
                if quality.get("quality") == "usable":
                    usable = (artifact_path.resolve(), markdown, quality)
                    break
            if usable is None:
                continue
            artifact_path, markdown, quality = usable
            completed_at = now_iso()
            results[index] = {
                "index": index,
                "image_path": image_path,
                "output_dir": str(output_dir.resolve()),
                "ok": True,
                "error": "",
                "artifact": {
                    "path": str(artifact_path),
                    "artifact_type": "markdown",
                    "text": markdown,
                    "text_encoding_source": "direct_utf8_artifact",
                },
                "ebook_quality": quality,
                "meaningful_text_char_count": int(quality.get("text_char_count") or 0),
                "meaningful_line_count": int(quality.get("line_count") or 0),
                "checkpointed_at": completed_at,
                "recovered_from_legacy_artifact": True,
            }

    recovered_results = [results[index] for index in sorted(results)]
    imported_entries = _ebook_results_to_import_rows(recovered_results)
    before = {
        index
        for index, item in enumerate(timeline, start=1)
        if _candidate_has_meaningful_visual_structure(item)
    }
    pending_entries = [
        entry
        for entry in imported_entries
        if not _ebook_entry_registered(timeline, entry)
    ]
    pending_status_results = [
        result
        for result in recovered_results
        if not _ebook_status_registered(timeline, result)
    ]
    backfill = {"updated": 0, "updated_indexes": [], "source_package_updated": False}
    status_backfill = {"updated": 0, "updated_indexes": [], "source_package_updated": False}
    if write and pending_entries:
        backfill = _backfill_visual_structure_results(root, manifest, timeline, pending_entries)
        timeline = _read_timeline(root)
    if write and pending_status_results:
        status_backfill = _backfill_ebook_pipeline_status(
            root, manifest, timeline, pending_status_results
        )
        timeline = _read_timeline(root)
    if write and (pending_entries or pending_status_results or invalid):
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            manifest["ebook_checkpoint_reconciliation"] = {
                "schema": "video_knowledge_pipeline.ebook_checkpoint_reconciliation.v1",
                "checkpoint_count": len(checkpoint_paths),
                "result_count": len(recovered_results),
                "recovered_indexes": sorted(
                    index
                    for index, item in enumerate(timeline, start=1)
                    if index not in before
                    and _candidate_has_meaningful_visual_structure(item)
                ),
                "invalid_count": len(invalid),
                "updated_at": now_iso(),
            }
            manifest["coverage"] = _coverage_audit(timeline)
            manifest["quality_audit"] = _quality_audit(timeline)
            manifest["repair_status"] = build_repair_status(manifest, timeline)
            write_json(manifest_path, manifest)
    recovered_indexes = sorted(
        index
        for index in results
        if index not in before
        and any(_int_value(row.get("index")) == index for row in imported_entries)
    )
    return {
        "schema": "video_knowledge_pipeline.ebook_checkpoint_reconciliation.v1",
        "status": "degraded" if invalid else "completed",
        "write": bool(write),
        "checkpoint_count": len(checkpoint_paths),
        "result_count": len(recovered_results),
        "recovered_count": len(recovered_indexes),
        "recovered_indexes": recovered_indexes,
        "invalid_count": len(invalid),
        "invalid": invalid,
        "backfill": backfill,
        "status_backfill": status_backfill,
    }


def _ebook_entry_registered(
    timeline: list[dict[str, Any]], entry: dict[str, Any]
) -> bool:
    index = _int_value(entry.get("index"))
    if not (1 <= index <= len(timeline)):
        return False
    markdown = str(entry.get("markdown") or "").strip()
    for value in timeline[index - 1].get("structured_visual") or []:
        if (
            isinstance(value, dict)
            and str(value.get("source") or "") == "ebook_markdown_pipeline"
            and str(value.get("markdown") or "").strip() == markdown
        ):
            return not entry.get('recognition_provenance') or (
                value.get('recognition_provenance') == entry['recognition_provenance']
                and timeline[index - 1].get('visual_text_recognition_provenance') == entry['recognition_provenance'])
    return False


def _ebook_status_registered(
    timeline: list[dict[str, Any]], result: dict[str, Any]
) -> bool:
    index = _int_value(result.get("index"))
    if not (1 <= index <= len(timeline)):
        return False
    current = timeline[index - 1].get("ebook_pipeline_status")
    if not isinstance(current, dict):
        return False
    return current == _ebook_pipeline_status_from_result(result)


def _count_ebook_blockers(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        if not isinstance(result, dict) or result.get("ok"):
            continue
        key = str(result.get("blocker") or "ebook_pipeline_failed")
        counts[key] = counts.get(key, 0) + 1
    return counts



@contextlib.contextmanager
def _ebook_pipeline_environment(profile: dict[str, Any]):
    updates: dict[str, str | None] = {}
    device = str(profile.get("rapidocr_device") or "auto").strip().lower()
    if device and device != "auto":
        updates["EBOOK_CONVERTER_RAPIDOCR_DEVICE"] = device
    if "rapidocr_cuda_device_id" in profile:
        updates["EBOOK_CONVERTER_RAPIDOCR_CUDA_DEVICE_ID"] = str(max(0, int(profile.get("rapidocr_cuda_device_id") or 0)))
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def _ebook_call_tool():
    project_dir = workspace_root() / "ebook_markdown_pipeline"
    if not project_dir.exists():
        raise FileNotFoundError(f"ebook_markdown_pipeline not found: {project_dir}")
    parent = str(project_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from ebook_markdown_pipeline.ebook_converter_mcp import call_tool  # type: ignore

    return call_tool


def _run_ebook_material_flow(call_tool, image_path: str, output_dir: Path, *, timeout_seconds: int) -> dict[str, Any]:
    routed = call_tool(
        "process_material",
        {
            "input": image_path,
            "output": str(output_dir),
            "recursive": True,
            "ocr": "always",
        },
    )
    job_id = routed.get("job_id") if isinstance(routed, dict) else ""
    if not job_id:
        return {"ok": False, "error": "process_material did not return job_id", "routed": routed}
    job = _poll_ebook_job(call_tool, str(job_id), timeout_seconds=timeout_seconds)
    artifact = _read_best_ebook_artifact(call_tool, job, output_dir=output_dir)
    raw_markdown = _ebook_markdown({"artifact": artifact})
    quality = _ebook_markdown_quality(raw_markdown, image_path=image_path)
    ok = str(job.get("status") or "") == "done" and quality["quality"] == "usable"
    error = "" if ok else str(job.get("error") or "")
    if str(job.get("status") or "") == "done" and raw_markdown and quality["quality"] != "usable":
        error = f"OCR artifact quality={quality['quality']}: {quality['reason']}"
    result: dict[str, Any] = {
        "ok": ok,
        "error": error,
        "routed": routed,
        "job": job,
        "artifact": artifact,
        "ebook_quality": quality,
        "meaningful_text_char_count": quality["text_char_count"],
        "meaningful_line_count": quality["line_count"],
    }
    if not ok:
        result.update(_classify_ebook_blocker(error, result=result))
    return result


def _classify_ebook_blocker(error: str, *, result: dict[str, Any] | None = None) -> dict[str, str]:
    text = error or ""
    if result:
        job = result.get("job") if isinstance(result.get("job"), dict) else {}
        artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
        text = "\n".join(
            part
            for part in (
                text,
                str(job.get("error") or ""),
                str(job.get("traceback") or ""),
                str(artifact.get("error") or ""),
            )
            if part
        )
    lower = text.lower()
    if "umi-ocr module not found" in lower or "ppocr_api.py" in lower:
        return {
            "blocker": "umi_ocr_missing",
            "next_action": "Repair ebook_markdown_pipeline Umi-OCR/PPOCR_api.py image OCR dependency, then rerun run-visual-structure with --execute-ebook-pipeline for the same indexes.",
        }
    if "ebook_markdown_pipeline not found" in lower:
        return {
            "blocker": "ebook_pipeline_unavailable",
            "next_action": "Set VKP_WORKSPACE_ROOT or install ebook_markdown_pipeline beside VKP, then rerun visual structure extraction.",
        }
    if "did not return job_id" in lower:
        return {
            "blocker": "ebook_pipeline_unavailable",
            "next_action": "Check ebook_markdown_pipeline MCP process_material response; it must return a job_id.",
        }
    if "timed out" in lower:
        return {
            "blocker": "ebook_pipeline_timeout",
            "next_action": "Increase --timeout-seconds or inspect ebook_markdown_pipeline job health before rerunning.",
        }
    if "missing image_path" in lower:
        return {
            "blocker": "missing_frame_image",
            "next_action": "Run frame recapture or repair bundle assets before document-visual parsing.",
        }
    if "quality=wrapper_only" in lower or "wrapper/source" in lower or "synthetic wrapper" in lower:
        return {
            "blocker": "ocr_wrapper_only",
            "next_action": "ebook_markdown_pipeline returned only wrapper/source-image metadata. Route this frame to high-res-tile-plan first; if tile OCR/VLM is still low confidence, keep it for multimodal or human review.",
        }
    if "quality=low_information" in lower or "low information" in lower:
        return {
            "blocker": "ocr_text_low_information",
            "next_action": "OCR extracted too little reliable text to clear this visual gap. Route this frame to high-res-tile-plan first; then merge trusted tile OCR/VLM/human results or keep it for review.",
        }
    if "quality=empty" in lower or "no meaningful text" in lower:
        return {
            "blocker": "ocr_text_empty",
            "next_action": "OCR ran but extracted no meaningful screen text. Route this frame to high-res-tile-plan first; if no tile result is meaningful, keep the screenshot evidence for multimodal or human review.",
        }
    if result is not None and not result.get("artifact"):
        return {
            "blocker": "artifact_missing",
            "next_action": "Inspect ebook job artifacts; read_artifact returned no usable Markdown/text artifact.",
        }
    return {
        "blocker": "ebook_pipeline_failed",
        "next_action": "Inspect ebook_pipeline_results job/error/traceback and rerun after repairing the external parser.",
    }


def _poll_ebook_job(call_tool, job_id: str, *, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + max(1, int(timeout_seconds or 120))
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = call_tool("get_job_status", {"job_id": job_id})
        if str(last.get("status") or "") != "running":
            return last
        time.sleep(0.5)
    raise TimeoutError(f"ebook_markdown_pipeline job timed out: {job_id}")


def _read_best_ebook_artifact(call_tool, job: dict[str, Any], *, output_dir: Path | None = None) -> dict[str, Any]:
    preferred = ["markdown", "location_index_jsonl", "text", "summary_report", "review_report"]
    artifacts = [item for item in job.get("artifacts") or [] if isinstance(item, dict)]
    for artifact_type in preferred:
        for artifact in artifacts:
            artifact_path = artifact.get("path")
            if artifact.get("type") == artifact_type and artifact_path:
                if output_dir and not _path_is_within(Path(str(artifact_path)), output_dir):
                    continue
                result = call_tool(
                    "read_artifact",
                    {
                        "path": artifact_path,
                        "artifact_type": artifact_type,
                        "max_chars": 12000,
                        "max_lines": 240,
                    },
                )
                if isinstance(result, dict) and not result.get("path"):
                    result["path"] = str(artifact_path)
                if isinstance(result, dict) and artifact_type in {"markdown", "text", "summary_report", "review_report"}:
                    local_path = Path(str(artifact_path)).expanduser()
                    if output_dir is not None and local_path.is_file() and _path_is_within(local_path, output_dir):
                        # The MCP transport may decode otherwise-valid UTF-8 artifact text
                        # with the host code page. Prefer the verified local artifact bytes.
                        result["text"] = local_path.read_text(encoding="utf-8-sig", errors="replace")
                        result["text_encoding_source"] = "direct_utf8_artifact"
                return result
    return {}


def repair_ebook_artifact_text(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Repair ebook OCR text from existing UTF-8 artifacts without rerunning OCR."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    timeline = _read_timeline(root)
    entries: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in timeline:
        status = item.get("ebook_pipeline_status") if isinstance(item.get("ebook_pipeline_status"), dict) else {}
        if status.get("ok") is not True:
            continue
        index = _int_value(item.get("index"))
        artifact_path = Path(str(status.get("artifact_path") or "")).expanduser()
        if not artifact_path.is_file() or not _path_is_within(artifact_path, root):
            failed.append({"index": index, "artifact_path": str(artifact_path), "reason": "artifact_missing_or_outside_bundle"})
            continue
        if 'recognition_binding' in status:
            try:
                _validate_ebook_binding(root, status['recognition_binding'])
                binding = status['recognition_binding']
                if binding['input']['index'] != index or safe_path(root, binding['artifact_path']) != artifact_path:
                    raise ValueError('ebook repair identity mismatch')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failed.append({'index': index, 'artifact_path': str(artifact_path), 'reason': str(exc)})
                continue
        markdown = artifact_path.read_text(encoding="utf-8-sig", errors="replace")
        image_path = str(status.get("image_path") or _candidate_image_path(item))
        markdown = _meaningful_ebook_markdown(markdown, image_path=image_path)
        if not markdown:
            failed.append({"index": index, "artifact_path": str(artifact_path), "reason": "artifact_has_no_meaningful_text"})
            continue
        entries.append(
            {
                "index": index,
                "type": _type_from_markdown(markdown) or "document_visual",
                "source": "ebook_markdown_pipeline",
                "markdown": markdown,
                "artifact_path": str(artifact_path.resolve()),
                "artifact_type": str(status.get("artifact_type") or "markdown"),
                "image_path": image_path,
                "output_dir": str(status.get("output_dir") or artifact_path.parent),
                "evidence_paths": [path for path in (image_path, str(artifact_path.resolve())) if path],
                **({'recognition_provenance': _ebook_recognition_provenance(status['recognition_binding'])}
                   if status.get('recognition_binding') else {}),
            }
        )
    repaired_indexes = _replace_ebook_visual_entries(timeline, entries)
    source_updated = False
    if write and repaired_indexes:
        write_json(root / "timeline.json", timeline)
        source_package_text = str(manifest.get("source_package") or "").strip()
        source_package = Path(source_package_text).expanduser() if source_package_text else None
        if source_package and source_package.is_file():
            package = read_json(source_package)
            if isinstance(package, dict) and isinstance(package.get("timeline"), list):
                _replace_ebook_visual_entries(package["timeline"], entries)
                package["coverage"] = _coverage_audit(package["timeline"])
                package["quality_audit"] = _quality_audit(package["timeline"])
                package["ebook_artifact_text_repaired_at"] = now_iso()
                write_json(source_package, package)
                source_updated = True
    result = {
        "ok": not failed,
        "status": "completed" if not failed else "degraded",
        "bundle_dir": str(root),
        "write": bool(write),
        "candidate_count": len(entries) + len(failed),
        "repaired_count": len(repaired_indexes),
        "repaired_indexes": repaired_indexes,
        "failed_count": len(failed),
        "failed": failed,
        "source_package_updated": source_updated,
    }
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        result["report_path"] = str(exports / "ebook-artifact-text-repair.json")
        write_json(exports / "ebook-artifact-text-repair.json", result)
    return result


def _replace_ebook_visual_entries(timeline: list[dict[str, Any]], entries: list[dict[str, Any]]) -> list[int]:
    by_index = {_int_value(entry.get("index")): entry for entry in entries}
    repaired: list[int] = []
    for position, item in enumerate(timeline, start=1):
        index = _int_value(item.get("index")) or position
        entry = by_index.get(index)
        if not entry:
            continue
        markdown = str(entry.get("markdown") or "").strip()
        values = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
        values = [
            value
            for value in values
            if not (isinstance(value, dict) and str(value.get("source") or "") == "ebook_markdown_pipeline")
        ]
        row = {
            "source": "ebook_markdown_pipeline",
            "type": str(entry.get("type") or "document_visual"),
            "markdown": markdown,
            "imported_at": now_iso(),
            "text_encoding_source": "direct_utf8_artifact",
        }
        for key in ("artifact_path", "artifact_type", "image_path", "output_dir", "evidence_paths", "recognition_provenance"):
            if entry.get(key):
                row[key] = entry[key]
        values.append(row)
        item["structured_visual"] = _dedupe_structured_visual(values)
        item["visual_text"] = markdown
        if row.get('recognition_provenance'):
            item['visual_text_recognition_provenance'] = row['recognition_provenance']
        else:
            item.pop('visual_text_recognition_provenance', None)
        legacy = item.get("legacy_visual_text")
        if isinstance(legacy, list):
            item["legacy_visual_text"] = [value for value in legacy if "\ufffd" not in str(value)]
        item["integrated_visual"] = integrated_visual(item)
        issues = _quality_issues(item)
        item["quality_issues"] = issues
        item["quality_score"] = _quality_score(issues)
        repaired.append(index)
    return repaired

def _ebook_results_to_import_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if not result.get("ok"):
            continue
        markdown = _meaningful_ebook_markdown(_ebook_markdown(result), image_path=str(result.get("image_path") or ""))
        if not markdown:
            continue
        row_type = _type_from_markdown(markdown) or "document_visual"
        artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
        rows.append(
            {
                "index": result.get("index"),
                "type": row_type,
                "source": "ebook_markdown_pipeline",
                "visual_text": markdown,
                "markdown": markdown,
                "artifact": artifact,
                "artifact_path": str(artifact.get("path") or ""),
                "artifact_type": str(artifact.get("artifact_type") or ""),
                "image_path": str(result.get("image_path") or ""),
                "output_dir": str(result.get("output_dir") or ""),
                "evidence_paths": [path for path in (str(result.get("image_path") or ""), str(artifact.get("path") or "")) if path],
                **({'recognition_provenance': _ebook_recognition_provenance(result['recognition_binding'])}
                   if result.get('recognition_binding') else {}),
            }
        )
    return rows


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ebook_markdown(result: dict[str, Any]) -> str:
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    text = _artifact_text(artifact)
    records = _artifact_records(artifact)
    if records:
        values = [str(record.get("text") or "").strip() for record in records if isinstance(record, dict) and str(record.get("text") or "").strip()]
        if values:
            return "\n\n".join(values)
    return text.strip()


def _meaningful_ebook_markdown(markdown: str, *, image_path: str | None = None) -> str:
    """Remove ebook wrapper text that is not extracted screen content."""
    return str(_ebook_markdown_quality(markdown, image_path=image_path)["meaningful_text"] or "")


def _ebook_markdown_quality(markdown: str, *, image_path: str | None = None) -> dict[str, Any]:
    stem = Path(str(image_path)).stem if image_path else ""
    stem_lower = stem.lower()
    kept: list[str] = []
    wrapper_lines = 0
    image_only_lines = 0
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("<!--") and lower.endswith("-->"):
            wrapper_lines += 1
            continue
        if _is_markdown_image_line(stripped):
            image_only_lines += 1
            continue
        if stem and lower in {f"# {stem_lower}", stem_lower}:
            wrapper_lines += 1
            continue
        if lower in {"# source image", "source image", "![source image]"}:
            wrapper_lines += 1
            continue
        kept.append(line.rstrip())
    meaningful_text = "\n".join(kept).strip()
    line_count = len([line for line in meaningful_text.splitlines() if line.strip()])
    text_char_count = len("".join(ch for ch in meaningful_text if not ch.isspace()))
    has_structure = any(token in meaningful_text for token in ("|---", "```", "$$", "\\begin", "<table"))
    if not str(markdown or "").strip():
        quality = "empty"
        reason = "empty artifact"
    elif not meaningful_text:
        quality = "wrapper_only" if wrapper_lines or image_only_lines else "empty"
        reason = "only wrapper comments or source-image links" if quality == "wrapper_only" else "no extracted text"
    elif not has_structure and (text_char_count < 12 or line_count < 1):
        quality = "low_information"
        reason = f"only {text_char_count} non-space characters after wrapper filtering"
    else:
        quality = "usable"
        reason = "contains meaningful screen text or structured visual content"
    return {
        "quality": quality,
        "reason": reason,
        "meaningful_text": meaningful_text,
        "text_char_count": text_char_count,
        "line_count": line_count,
        "wrapper_line_count": wrapper_lines,
        "image_only_line_count": image_only_lines,
    }


def _is_markdown_image_line(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("![") and "](" in stripped and stripped.endswith(")")


def _artifact_text(artifact: dict[str, Any]) -> str:
    for key in ("markdown", "output", "report", "artifact", "text", "content"):
        value = artifact.get(key)
        text = _artifact_value_text(value)
        if text:
            return text
    return ""


def _artifact_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("markdown", "output", "report", "text", "content", "value", "data"):
            text = _artifact_value_text(value.get(key))
            if text:
                return text
        return ""
    if isinstance(value, list):
        parts = [_artifact_value_text(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    return str(value).strip()


def _artifact_records(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    records = artifact.get("records") if isinstance(artifact.get("records"), list) else []
    return [record for record in records if isinstance(record, dict)]


def _single_candidate_index(candidates: list[dict[str, Any]]) -> int | None:
    if len(candidates) != 1:
        return None
    return _int_value(candidates[0].get("index")) or None


def write_visual_structure_input_template(root: str | Path, candidates: list[dict[str, Any]]) -> Path:
    bundle_dir = Path(root).expanduser().resolve()
    path = bundle_dir / "visual-structure-input-template.json"
    rows = []
    for candidate in candidates:
        material_types = candidate.get("material_types") or []
        rows.append(
            {
                "index": candidate.get("index"),
                "start": candidate.get("start", 0),
                "end": candidate.get("end", 0),
                "type": material_types[0] if material_types else "structured_visual",
                "markdown": "",
                "source": candidate.get("image_path", ""),
                "image_path": candidate.get("image_path", ""),
                "image_exists": bool(candidate.get("image_exists")),
                "material_types": material_types,
                "quality_issues": candidate.get("quality_issues", []),
                "current_visual_text": candidate.get("visual_text", ""),
                "existing_structured_visual": candidate.get("structured_visual", []),
                "notes": "",
            }
        )
    first_index = rows[0]["index"] if rows else 1
    write_json(
        path,
        {
            "schema": "lecture_visual_structure_input.v1",
            "generated_at": now_iso(),
            "bundle_dir": str(bundle_dir),
            "instructions": "Fill markdown with corrected table/formula/code/board structure, then pass this JSON to run-visual-structure --input-json.",
            "items": rows,
            "tool_output_examples": {
                "docling_or_generic_table": {
                    "tables": [
                        {
                            "index": first_index,
                            "data": {
                                "table_cells": [
                                    {"row": 0, "col": 0, "text": "A"},
                                    {"row": 0, "col": 1, "text": "B"},
                                    {"row": 1, "col": 0, "text": "1"},
                                    {"row": 1, "col": 1, "text": "2"},
                                ]
                            },
                        }
                    ]
                },
                "marker_or_mineru_blocks": {
                    "blocks": [
                        {"index": first_index, "type": "equation", "latex": "E = mc^2"},
                        {"index": first_index, "type": "text", "markdown": "关键板书内容"},
                    ]
                },
                "paddleocr_pp_structure": {
                    "results": [
                        {"index": first_index, "type": "table", "res": {"html": "<table><tr><td>A</td><td>B</td></tr></table>"}},
                        {"index": first_index, "type": "equation", "res": {"text": "\\frac{a}{b}"}},
                    ]
                },
            },
            "note": "When there is only one structured-visual candidate, Docling/Marker/MinerU/PaddleOCR output may omit index; run-visual-structure assigns the only candidate index.",
        },
    )
    return path


def _read_visual_structure_input(input_json: str | Path, *, default_index: int | None = None) -> list[dict[str, Any]]:
    path = Path(input_json).expanduser().resolve()
    if path.suffix.lower() in {".md", ".markdown"}:
        return _markdown_file_rows(path, default_index=default_index, source=path.stem)
    data = read_json(path)
    rows = _normalise_visual_structure_input(data, default_index=default_index, source=path.stem)
    if rows:
        return rows
    rows = _marker_markdown_rows(path, default_index=default_index)
    if rows:
        return rows
    if isinstance(data, dict):
        rows = data.get("items") or data.get("results") or data.get("visual_structure") or []
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError("visual structure input JSON must be a list or an object with items/results/visual_structure")
    return [row for row in rows if isinstance(row, dict)]


def _normalise_visual_structure_input(data: Any, *, default_index: int | None, source: str) -> list[dict[str, Any]]:
    rows = _explicit_visual_rows(data)
    if not rows:
        rows = _extract_tool_rows(data)
    normalised = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        markdown = _markdown_from_tool_row(row)
        if not markdown:
            continue
        index = _int_value(row.get("index") or row.get("timeline_index") or row.get("timelineIndex") or default_index)
        normalised.append(
            {
                "index": index,
                "type": _type_from_tool_row(row),
                "source": str(row.get("source") or row.get("tool") or source or "visual_structure_tool"),
                "markdown": markdown,
            }
        )
    return normalised


def _marker_markdown_rows(path: Path, *, default_index: int | None) -> list[dict[str, Any]]:
    if path.name.lower() != "blocks.json":
        return []
    markdown_candidates = [
        candidate
        for candidate in sorted(path.parent.glob("*.md"))
        if candidate.is_file() and not candidate.name.lower().endswith("_meta.md")
    ]
    if not markdown_candidates:
        return []
    return _markdown_file_rows(markdown_candidates[0], default_index=default_index, source="marker")


def _markdown_file_rows(path: Path, *, default_index: int | None, source: str) -> list[dict[str, Any]]:
    markdown = path.read_text(encoding="utf-8").strip()
    if not markdown:
        return []
    return [
        {
            "index": default_index or 1,
            "type": _type_from_markdown(markdown),
            "source": source,
            "markdown": markdown,
        }
    ]


def _explicit_visual_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("items", "results", "visual_structure"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if _markdown_from_tool_row(data):
        return [data]
    return []


def _extract_tool_rows(data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any, inherited_type: str = "") -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, inherited_type=inherited_type)
            return
        if not isinstance(value, dict):
            return
        row_type = _type_from_tool_row(value) or inherited_type
        if row_type and _markdown_from_tool_row(value):
            rows.append(value)
            return
        for key in ("blocks", "children", "para_blocks", "content_list", "tables", "equations", "formulas", "texts"):
            child = value.get(key)
            if child is not None:
                visit(child, inherited_type=_type_hint_from_key(key) or row_type)
        res = value.get("res")
        if isinstance(res, (dict, list)):
            visit(res, inherited_type=row_type)

    visit(data)
    return rows


def _markdown_from_tool_row(row: dict[str, Any]) -> str:
    for key in ("markdown", "md", "text", "content", "latex", "table_markdown", "structured_text", "html"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    res = row.get("res")
    if isinstance(res, dict):
        for key in ("markdown", "html", "text", "latex", "table_markdown", "structured_text"):
            value = res.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    cells = row.get("cells") or row.get("table_cells")
    if isinstance(cells, list):
        table = _table_cells_to_markdown(cells)
        if table:
            return table
    data = row.get("data")
    if isinstance(data, dict):
        cells = data.get("table_cells") or data.get("cells")
        if isinstance(cells, list):
            table = _table_cells_to_markdown(cells)
            if table:
                return table
    return ""


def _type_from_tool_row(row: dict[str, Any]) -> str:
    raw = str(
        row.get("type")
        or row.get("material_type")
        or row.get("block_type")
        or row.get("category")
        or row.get("label")
        or row.get("kind")
        or ""
    ).strip().lower()
    if not raw and isinstance(row.get("res"), dict):
        raw = str((row.get("res") or {}).get("type") or "").strip().lower()
    if any(token in raw for token in ("table", "tabular")):
        return "table"
    if any(token in raw for token in ("formula", "equation", "latex")):
        return "formula"
    if any(token in raw for token in ("code", "program")):
        return "code"
    if raw:
        return raw
    markdown = _markdown_from_tool_row(row)
    return _type_from_markdown(markdown)


def _type_from_markdown(markdown: str) -> str:
    if "|" in markdown and "---" in markdown:
        return "table"
    if any(token in markdown for token in ("\\frac", "\\sum", "\\int")):
        return "formula"
    if "```" in markdown:
        return "code"
    return "structured_visual"


def _type_hint_from_key(key: str) -> str:
    hints = {
        "tables": "table",
        "equations": "formula",
        "formulas": "formula",
        "texts": "text",
    }
    return hints.get(key, "")


def _table_cells_to_markdown(cells: list[Any]) -> str:
    grid: dict[tuple[int, int], str] = {}
    max_row = 0
    max_col = 0
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = _int_value(cell.get("row") or cell.get("row_index") or cell.get("start_row_offset_idx")) + 1
        col = _int_value(cell.get("col") or cell.get("col_index") or cell.get("start_col_offset_idx")) + 1
        text = str(cell.get("text") or cell.get("content") or cell.get("cell_value") or "").strip()
        if not text:
            continue
        row = max(row, 1)
        col = max(col, 1)
        grid[(row, col)] = text
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    if not grid or max_col <= 0:
        return ""
    rows = []
    for row in range(1, max_row + 1):
        rows.append([grid.get((row, col), "") for col in range(1, max_col + 1)])
    header = rows[0]
    body = rows[1:] or [["" for _ in header]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _backfill_ebook_pipeline_status(
    root: Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    updated_indexes = _apply_ebook_pipeline_statuses(timeline, results)
    if updated_indexes:
        write_json(root / "timeline.json", timeline)
    source_updated = False
    source_package_text = str(manifest.get("source_package") or "").strip()
    source_package = Path(source_package_text).expanduser() if source_package_text else None
    if source_package and source_package.exists() and source_package.is_file():
        package = read_json(source_package)
        if isinstance(package, dict) and isinstance(package.get("timeline"), list):
            package_updated = _apply_ebook_pipeline_statuses(package["timeline"], results)
            if package_updated:
                package["coverage"] = _coverage_audit(package["timeline"])
                package["quality_audit"] = _quality_audit(package["timeline"])
                package["ebook_pipeline_status_imported_at"] = now_iso()
                write_json(source_package, package)
                source_updated = True
    return {"updated": len(updated_indexes), "updated_indexes": updated_indexes, "source_package_updated": source_updated}


def _apply_ebook_pipeline_statuses(timeline: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[int]:
    updated: list[int] = []
    for result in results:
        index = _int_value(result.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        item = timeline[index - 1]
        status = _ebook_pipeline_status_from_result(result)
        previous = item.get("ebook_pipeline_status") if isinstance(item.get("ebook_pipeline_status"), dict) else {}
        if previous == status:
            continue
        item["ebook_pipeline_status"] = status
        if status.get("ok") is False:
            _remove_wrapper_only_ebook_visuals(item)
            item["needs_human_review"] = True
            review = item.setdefault("human_review", {})
            if isinstance(review, dict):
                review["missing_info"] = True
                review["ebook_blocker"] = status.get("blocker")
                review["ebook_next_action"] = status.get("next_action")
        item["integrated_visual"] = integrated_visual(item)
        issues = _quality_issues(item)
        item["quality_issues"] = issues
        item["quality_score"] = _quality_score(issues)
        updated.append(index)
    return updated


def _candidate_image_path(item: dict[str, Any]) -> str:
    if item.get("image_path"):
        return str(item.get("image_path") or "")
    frame_paths = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
    for path in frame_paths:
        if path:
            return str(path)
    assets = item.get("assets") if isinstance(item.get("assets"), list) else []
    for asset in assets:
        if isinstance(asset, dict) and (asset.get("path") or asset.get("source")):
            return str(asset.get("path") or asset.get("source") or "")
    return ""

def _remove_wrapper_only_ebook_visuals(item: dict[str, Any]) -> None:
    current = str(item.get("visual_text") or "").strip()
    if current and _ebook_markdown_quality(current, image_path=_candidate_image_path(item)).get("quality") != "usable":
        legacy_values = item.setdefault("legacy_visual_text", [])
        if not isinstance(legacy_values, list):
            legacy_values = [str(legacy_values)]
            item["legacy_visual_text"] = legacy_values
        if current not in legacy_values:
            legacy_values.append(current)
        item.pop("visual_text", None)
    structured = item.get("structured_visual")
    if isinstance(structured, list):
        kept = []
        for value in structured:
            if not isinstance(value, dict):
                continue
            source = str(value.get("source") or "")
            markdown = str(value.get("markdown") or "")
            if source == "ebook_markdown_pipeline" and _ebook_markdown_quality(markdown, image_path=_candidate_image_path(item)).get("quality") != "usable":
                continue
            kept.append(value)
        item["structured_visual"] = kept

def _ebook_pipeline_status_from_result(result: dict[str, Any]) -> dict[str, Any]:
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    quality = result.get("ebook_quality") if isinstance(result.get("ebook_quality"), dict) else {}
    status = {
        "ok": bool(result.get("ok")),
        "blocker": str(result.get("blocker") or ""),
        "quality": str(quality.get("quality") or ("usable" if result.get("ok") else "unknown")),
        "reason": str(quality.get("reason") or result.get("error") or ""),
        "next_action": str(result.get("next_action") or ""),
        "artifact_path": str(artifact.get("path") or ""),
        "artifact_type": str(artifact.get("artifact_type") or ""),
        "image_path": str(result.get("image_path") or ""),
        "output_dir": str(result.get("output_dir") or ""),
        "meaningful_text_char_count": int(result.get("meaningful_text_char_count") or quality.get("text_char_count") or 0),
        "meaningful_line_count": int(result.get("meaningful_line_count") or quality.get("line_count") or 0),
        "updated_at": str(result.get("checkpointed_at") or now_iso()),
    }
    if result.get("error"):
        status["error"] = str(result.get("error") or "")
    if 'recognition_binding' in result:
        status['recognition_binding'] = result['recognition_binding']
    status['recognition_identity_status'] = 'checkpoint_bound' if result.get('recognition_binding') else 'legacy_inputs_unrecorded'
    return status

def _backfill_visual_structure_results(
    root: Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    updated_indexes = _apply_visual_structure_entries(timeline, entries)
    write_json(root / "timeline.json", timeline)

    source_updated = False
    source_package_text = str(manifest.get("source_package") or "").strip()
    source_package = Path(source_package_text).expanduser() if source_package_text else None
    if source_package and source_package.exists() and source_package.is_file():
        package = read_json(source_package)
        if isinstance(package, dict) and isinstance(package.get("timeline"), list):
            _apply_visual_structure_entries(package["timeline"], entries)
            package["coverage"] = _coverage_audit(package["timeline"])
            package["quality_audit"] = _quality_audit(package["timeline"])
            package["visual_structure_imported_at"] = now_iso()
            write_json(source_package, package)
            source_updated = True
    return {
        "updated": len(updated_indexes),
        "updated_indexes": updated_indexes,
        "source_package_updated": source_updated,
    }


def _apply_visual_structure_entries(timeline: list[dict[str, Any]], entries: list[dict[str, Any]]) -> list[int]:
    updated = []
    for entry in entries:
        index = _int_value(entry.get("index") or entry.get("timeline_index"))
        if not (1 <= index <= len(timeline)):
            continue
        markdown = str(
            entry.get("markdown")
            or entry.get("text")
            or entry.get("latex")
            or entry.get("table_markdown")
            or entry.get("structured_text")
            or ""
        ).strip()
        if not markdown:
            continue
        item = timeline[index - 1]
        row = {
            "source": str(entry.get("source") or entry.get("tool") or "imported_json"),
            "type": str(entry.get("type") or entry.get("material_type") or "structured_visual"),
            "markdown": markdown,
            "imported_at": now_iso(),
        }
        for evidence_key in ("artifact_path", "artifact_type", "image_path", "output_dir", "evidence_paths", "recognition_provenance"):
            if entry.get(evidence_key):
                row[evidence_key] = entry[evidence_key]
        values = item.setdefault("structured_visual", [])
        if not isinstance(values, list):
            values = []
            item["structured_visual"] = values
        values = _dedupe_structured_visual(values)
        if row.get('recognition_provenance'):
            for value in values:
                if (value.get('source') == row['source'] and value.get('type') == row['type']
                        and str(value.get('markdown') or '').strip() == markdown):
                    value.update({k: v for k, v in row.items() if k != 'imported_at'})
        if not any(
            isinstance(value, dict)
            and str(value.get("source") or "") == row["source"]
            and str(value.get("type") or "") == row["type"]
            and str(value.get("markdown") or "").strip() == markdown
            for value in values
        ):
            values.append(row)
        item["structured_visual"] = values
        material_types = item.setdefault("material_types", [])
        if isinstance(material_types, list):
            row_type = str(row.get("type") or "")
            if row_type and row_type not in material_types:
                material_types.append(row_type)
            if "text" not in material_types:
                material_types.append("text")
        current = str(item.get("visual_text") or "").strip()
        if row["source"] == "ebook_markdown_pipeline":
            if current and current != markdown:
                legacy_values = item.setdefault("legacy_visual_text", [])
                if not isinstance(legacy_values, list):
                    legacy_values = [str(legacy_values)]
                    item["legacy_visual_text"] = legacy_values
                if current not in legacy_values:
                    legacy_values.append(current)
            item["visual_text"] = markdown
        elif not current:
            item["visual_text"] = markdown
        elif markdown not in current:
            item["visual_text"] = f"{current}\n\n{markdown}"
        if item.get('visual_text') == markdown and row.get('recognition_provenance'):
            item['visual_text_recognition_provenance'] = row['recognition_provenance']
        elif item.get('visual_text') != current:
            item.pop('visual_text_recognition_provenance', None)
        item["integrated_visual"] = integrated_visual(item)
        issues = _quality_issues(item)
        item["quality_issues"] = issues
        item["quality_score"] = _quality_score(issues)
        if index not in updated:
            updated.append(index)
    return updated


def _dedupe_structured_visual(values: list[Any]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        key = (
            str(value.get("source") or ""),
            str(value.get("type") or ""),
            str(value.get("markdown") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _render_visual_structure_report(
    root: Path,
    candidates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    summary: dict[str, Any],
    template_path: Path,
    ebook_results: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# Visual Structure Plan",
        "",
        f"- Bundle: `{root}`",
        f"- Input template JSON: `{template_path}`",
        f"- Candidates: {len(candidates)}",
        f"- Imported: {summary.get('imported', 0)}",
        f"- Updated: {summary.get('updated', 0)}",
        f"- Completion scope: `{summary.get('completion_scope', 'selected_candidates_only')}`",
        f"- Recovered selected pages: {summary.get('structured_count', 0)}",
        f"- Unresolved selected indexes: {summary.get('unresolved_indexes', [])}",
        f"- Unselected candidates: {summary.get('unselected_candidate_count', 0)}",
        "- Whole Bundle completeness: `not_assessed`",
        f"- Ebook execution: `{summary.get('execute_ebook_pipeline', False)}`",
        f"- ebook_pipeline_total: `{summary.get('ebook_pipeline_total', 0)}`",
        f"- ebook_pipeline_succeeded: `{summary.get('ebook_pipeline_succeeded', 0)}`",
        f"- ebook_pipeline_blockers: `{summary.get('ebook_pipeline_blockers', {})}`",
        "",
        "## Runtime Config",
        "",
    ]
    runtime_config = summary.get("runtime_config") if isinstance(summary.get("runtime_config"), dict) else {}
    validation = runtime_config.get("validation") if isinstance(runtime_config.get("validation"), dict) else {}
    service_urls = runtime_config.get("service_urls") if isinstance(runtime_config.get("service_urls"), dict) else {}
    lines.extend(
        [
            f"- Config: `{runtime_config.get('config_path', '')}`",
            f"- Config ok: `{runtime_config.get('ok', False)}`",
            f"- Validation issues: `{validation.get('issue_count', 0)}`",
            f"- ebook HTTP bridge: `{service_urls.get('ebook_markdown_pipeline_http', '')}`",
            "",
        ]
    )
    lines.extend(
        [
        "## Tools",
        "",
        ]
    )
    for tool in tools:
        lines.append(f"- {tool.get('name')}: command_available=`{tool.get('command_available', False)}` "
                     f"runtime_status=`{tool.get('runtime_status', 'not_probed')}` "
                     f"ready_to_execute=`{tool.get('ready_to_execute', False)}` "
                     f"model_status=`{tool.get('model_status', 'not_assessed')}` role={tool.get('role')}")
    lines.extend(["", "## Candidates", ""])
    ebook_result_by_index = {
        _int_value(result.get("index")): result
        for result in (ebook_results or [])
        if isinstance(result, dict) and _int_value(result.get("index"))
    }
    for item in candidates:
        index = _int_value(item.get("index"))
        ebook_result = ebook_result_by_index.get(index)
        status = _visual_structure_candidate_status(summary, item, ebook_result)
        lines.extend(
            [
                f"### Timeline {item.get('index')}",
                "",
                f"- Status: `{status}`",
                f"- Types: {', '.join(item.get('material_types') or [])}",
                f"- Route: `{item.get('visual_route') or ''}`",
                f"- Image: `{item.get('image_path') or 'missing'}`",
                "",
            ]
        )
        routing = item.get("routing_decision") if isinstance(item.get("routing_decision"), dict) else {}
        if routing:
            lines.extend(
                [
                    f"- Primary branch: `{routing.get('primary_branch', '')}`",
                    f"- Primary tool: `{routing.get('primary_tool', '')}`",
                    f"- Also requires multimodal: `{routing.get('also_requires_multimodal', False)}`",
                    f"- Routing reason: {routing.get('reason', '')}",
                    "",
                ]
            )
        if ebook_result and ebook_result.get("error"):
            lines.extend([f"- ebook error: `{ebook_result.get('error')}`", ""])
        if ebook_result and ebook_result.get("blocker"):
            lines.extend([f"- ebook blocker: `{ebook_result.get('blocker')}`", ""])
        if ebook_result and ebook_result.get("next_action"):
            lines.extend([f"- next action: {ebook_result.get('next_action')}", ""])
        for name, command in (item.get("commands") or {}).items():
            lines.extend([f"#### {name}", "", "```powershell", command, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _visual_structure_candidate_status(summary: dict[str, Any], item: dict[str, Any], ebook_result: dict[str, Any] | None) -> str:
    if ebook_result and ebook_result.get("ok") and _candidate_has_meaningful_visual_structure(item):
        return "imported"
    if _candidate_has_meaningful_visual_structure(item):
        return "already_structured"
    if ebook_result is None:
        return "preview" if not summary.get("execute_ebook_pipeline") else "not_run"
    if ebook_result.get("ok"):
        return "imported"
    if ebook_result.get("error"):
        return "failed"
    return "needs_human_review"


def _candidate_has_meaningful_visual_structure(item: dict[str, Any]) -> bool:
    image_path = str(item.get("image_path") or "")
    if _meaningful_ebook_markdown(str(item.get("visual_text") or ""), image_path=image_path):
        return True
    values = item.get("structured_visual")
    if not isinstance(values, list):
        return False
    for value in values:
        if not isinstance(value, dict):
            continue
        markdown = str(value.get("markdown") or value.get("text") or "")
        candidate_image = str(value.get("image_path") or image_path)
        if _meaningful_ebook_markdown(markdown, image_path=candidate_image):
            return True
    return False


def _build_visual_structure_handoff(
    root: Path,
    manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    summary: dict[str, Any],
    template_path: Path,
    report_path: Path,
    handoff_path: Path,
    handoff_json_path: Path,
) -> dict[str, Any]:
    mcp_args_path = _resolve_manifest_path(root, manifest.get("mcp_visual_structure_args"))
    mcp_command = _mcp_command("run_visual_structure_plan", mcp_args_path) if mcp_args_path else ""
    unresolved = [
        item
        for item in candidates
        if not _candidate_has_meaningful_visual_structure(item)
    ]
    return {
        "schema": "lecture_visual_structure_handoff.v1",
        "created_at": now_iso(),
        "bundle_dir": str(root),
        "status": _visual_structure_handoff_status(summary, candidates),
        "completion_scope": "selected_candidates_only",
        "bundle_completeness": "not_assessed",
        "objective": "Recover formulas, tables, code blocks, diagrams, and board structures as Markdown/LaTeX/code/table content when plain OCR text is not enough.",
        "paths": {
            "handoff_markdown": str(handoff_path),
            "handoff_json": str(handoff_json_path),
            "report_markdown": str(report_path),
            "input_template_json": str(template_path),
            "manifest_json": str(root / "manifest.json"),
            "timeline_json": str(root / "timeline.json"),
        },
        "tools": tools,
        "mcp": {
            "tool": "run_visual_structure_plan",
            "args_path": str(mcp_args_path) if mcp_args_path else "",
            "args_exists": bool(mcp_args_path and mcp_args_path.exists()),
            "command": mcp_command,
            "import_argument": "input_json",
        },
        "summary": summary,
        "next_steps": _visual_structure_next_steps(tools, mcp_command) if unresolved else [
            {"actor": "human", "action_code": "review_selected_evidence", "execution_allowed": False,
             "action": "Selected pages have text/layout evidence or no selected work. Review fidelity and the remaining Bundle scope; do not rerun OCR solely because this batch made no new writes."}
        ],
        "import_schema": {
            "schema": "lecture_visual_structure_input.v1",
            "items": [
                {
                    "index": 1,
                    "type": "table|formula|code|diagram|structured_visual",
                    "markdown": "Markdown table, LaTeX, code block, or exact structured text.",
                    "source": "absolute/or bundle-relative image path",
                    "notes": "Optional uncertainty or manual review notes.",
                }
            ],
        },
        "items": [
            {
                "index": item.get("index"),
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "material_types": item.get("material_types", []),
                "image_path": item.get("image_path", ""),
                "image_exists": bool(item.get("image_exists")),
                "has_structured_visual": _candidate_has_meaningful_visual_structure(item),
                "commands": item.get("commands") or {},
                "current_visual_text": item.get("visual_text", ""),
            }
            for item in candidates
        ],
        "unresolved_indexes": [item.get("index") for item in unresolved],
    }


def _visual_structure_handoff_status(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    total = int(summary.get("total_candidates") or 0)
    already_structured = sum(1 for item in candidates if _candidate_has_meaningful_visual_structure(item))
    if len(candidates) != total:
        return "needs_structure"
    if total == 0:
        return "not_needed"
    if already_structured == total:
        return "structured"
    if already_structured:
        return "partially_structured"
    return "needs_structure"


def _visual_structure_next_steps(tools: list[dict[str, Any]], mcp_command: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    steps.append(
        {
            "actor": "agent_or_human",
            "action_code": "verify_local_parser_runtime",
            "execution_allowed": False,
            "action": "Check the existing ebook pipeline's configured interpreter, image backend and local model availability before explicit local execution. Source folders and command discovery are not runtime validation; per-item backend commands are reference only. Do not install/download models, start services or upload screenshots implicitly. If unavailable, keep a review gap or transcribe locally by explicit human review.",
            "discovered_commands": [tool.get("name") for tool in tools if tool.get("command_available")],
        }
    )
    steps.append(
        {
            "actor": "agent",
            "action": "Import corrected structured visual JSON through run_visual_structure_plan input_json; keep formulas, code indentation, table columns, and diagram labels intact.",
            "command": mcp_command,
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


def _render_visual_structure_handoff_markdown(handoff: dict[str, Any]) -> str:
    paths = handoff.get("paths") if isinstance(handoff.get("paths"), dict) else {}
    mcp = handoff.get("mcp") if isinstance(handoff.get("mcp"), dict) else {}
    lines = [
        "# Visual Structure Handoff",
        "",
        f"- Status: `{handoff.get('status')}`",
        f"- Completion scope: `{handoff.get('completion_scope', 'selected_candidates_only')}`",
        "- Whole Bundle completeness: `not_assessed`",
        f"- Bundle: `{handoff.get('bundle_dir')}`",
        f"- Input template: `{paths.get('input_template_json', '')}`",
        f"- Report: `{paths.get('report_markdown', '')}`",
        f"- MCP args: `{mcp.get('args_path', '')}`",
        "",
        "## MCP",
        "",
        "```powershell",
        str(mcp.get("command") or ""),
        "```",
        "",
        "## Tools",
        "",
        "| Tool | Command found | Runtime status | Ready to execute | Models | Command/local root | Role |",
        "|---|---|---|---|---|---|---|",
    ]
    for tool in handoff.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        command_or_root = ", ".join(str(value) for value in tool.get("commands") or []) or str(tool.get("local_root") or "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(str(tool.get("name") or "")),
                    str(bool(tool.get("command_available"))),
                    _md_cell(str(tool.get("runtime_status") or "not_probed")),
                    str(bool(tool.get("ready_to_execute"))),
                    _md_cell(str(tool.get("model_status") or "not_assessed")),
                    _md_cell(command_or_root),
                    _md_cell(str(tool.get("role") or "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Steps", ""])
    for step in handoff.get("next_steps") or []:
        if isinstance(step, dict):
            lines.append(f"- `{step.get('actor', '')}`: {step.get('action', '')}")
    lines.extend(
        [
            "",
            "## Import Schema",
            "",
            "```json",
            '{ "items": [ { "index": 1, "type": "table", "markdown": "| A | B |\\n|---|---|", "source": "frame path", "notes": "" } ] }',
            "```",
            "",
            "## Items",
            "",
            "| Index | Types | Structured | Image |",
            "|---:|---|---|---|",
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
                    _md_cell(", ".join(str(value) for value in item.get("material_types") or [])),
                    str(bool(item.get("has_structured_visual"))),
                    _md_cell(str(item.get("image_path") or "")),
                ]
            )
            + " |"
        )
        commands = item.get("commands") if isinstance(item.get("commands"), dict) else {}
        for name, command in commands.items():
            lines.extend(["", f"### Timeline {item.get('index')} {name}", "", "```powershell", str(command), "```"])
    return "\n".join(lines).rstrip() + "\n"




def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
