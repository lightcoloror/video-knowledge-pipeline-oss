from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .frame_recapture import _coverage_audit, _quality_audit
from .models import now_iso
from .multimodal_frame_analyzer import (
    _normalise_visual_understanding,
    _refresh_post_vision_outputs,
    _sync_source_package,
    _apply_single as _apply_visual_single,
    build_vision_restore_hint,
    snapshot_timeline_for_vision_diff,
    timeline_vision_diff,
    write_vision_analysis_run_audit,
)
from .powershell import quote_powershell_literal
from .repair_status import build_repair_status
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, replace_file_with_retry, write_json
from .temporal_frame_preprocess import build_temporal_frame_manifest
from .temporal_visual_analyzer import (
    _apply_single as _apply_temporal_single,
    _normalise_temporal_understanding,
)


PACK_SCHEMA = "video_knowledge_pipeline.codex_vision_session.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.codex_vision_result.v1"
VALIDATION_SCHEMA = "video_knowledge_pipeline.codex_vision_import_validation.v1"
SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9._-]{1,96}$")
EVIDENCE_LEVELS = {"confirmed", "inferred", "unavailable"}
MAX_FRAMES_PER_SESSION = 20
DEFAULT_SEMANTIC_LIMIT = 19
MIN_TEMPORAL_FRAMES = 5
MAX_TEMPORAL_FRAMES = 12
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_STRING_CHARS = 2000
MAX_COLLECTION_ITEMS = 32
MAX_NESTING_DEPTH = 7
TIME_TOLERANCE_SECONDS = 0.001


def prepare_codex_vision_session(
    bundle_dir: str | Path,
    *,
    kind: str = "semantic",
    indexes: list[int] | None = None,
    limit: int | None = None,
    max_frames: int = DEFAULT_SEMANTIC_LIMIT,
    model_requested: str = "luna",
    quality: str = "low",
    executor: str = "codex_session",
    viewport_pack_json: str | Path | None = None,
    viewport_timeline_index: int | None = None,
    viewport_sample_index: int = 1,
    panorama_analysis_plan_json: str | Path | None = None,
    panorama_task_id: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Prepare one immutable local frame batch; preparation never calls Codex."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    manifest = _read_mapping(manifest_path, "manifest.json")
    timeline = _read_timeline(timeline_path)
    session_kind = _session_kind(kind)
    executor_name = _executor_name(executor)
    requested_model = _required_text(model_requested, "model_requested")
    quality_name = _required_text(quality, "quality")
    frame_budget = int(max_frames or DEFAULT_SEMANTIC_LIMIT)
    if not 1 <= frame_budget <= MAX_FRAMES_PER_SESSION:
        raise ValueError(f"max_frames must be between 1 and {MAX_FRAMES_PER_SESSION}")
    requested_indexes = _indexes(indexes)
    effective_limit = int(limit) if limit is not None else (DEFAULT_SEMANTIC_LIMIT if session_kind == "semantic" else 1)
    if effective_limit < 0:
        raise ValueError("limit must be non-negative")

    viewport_context: dict[str, Any] = {}
    if viewport_pack_json is not None and panorama_analysis_plan_json is not None:
        raise ValueError("prepare exactly one panorama semantic or temporal session")
    if viewport_pack_json is not None:
        if session_kind != "semantic":
            raise ValueError("panorama viewport sessions must use kind=semantic")
        from .panorama_session import build_panorama_session_candidate

        selected_index = int(viewport_timeline_index or (requested_indexes[0] if requested_indexes else 0))
        candidate, viewport_context = build_panorama_session_candidate(
            root,
            timeline,
            viewport_pack_json=viewport_pack_json,
            timeline_index=selected_index,
            sample_index=int(viewport_sample_index),
        )
        candidates, blockers = [candidate], []
        requested_indexes = [selected_index]
    elif panorama_analysis_plan_json is not None:
        if session_kind != "temporal":
            raise ValueError("panorama direction sessions must use kind=temporal")
        from .panorama_session import build_panorama_temporal_session_candidate

        candidate, viewport_context = build_panorama_temporal_session_candidate(
            root,
            timeline,
            analysis_plan_json=panorama_analysis_plan_json,
            task_id=panorama_task_id,
        )
        candidates, blockers = [candidate], []
        requested_indexes = [int(candidate["index"])]
    else:
        candidates, blockers = _session_candidates(
            root,
            timeline,
            kind=session_kind,
            requested_indexes=requested_indexes,
            limit=effective_limit,
            max_frames=frame_budget,
        )
    session_id = uuid4().hex
    session_dir = root / "codex-vision-sessions" / session_id
    task_pack_path = session_dir / "task-pack.json"
    prompt_path = session_dir / "prompt.md"
    result_template_path = session_dir / "result.template.json"
    validation_path = session_dir / "import-validation.json"
    raw_result_path = session_dir / "raw-result.json"
    status_path = root / "codex-vision-session-status.json"

    if write:
        manifest["codex_vision_session_status_json"] = status_path.name
        manifest["codex_vision_session_task_pack_json"] = _relative_path(root, task_pack_path)
        manifest["codex_vision_session_prompt_markdown"] = _relative_path(root, prompt_path)
        manifest["codex_vision_session_result_template_json"] = _relative_path(root, result_template_path)
        manifest["codex_vision_session_import_validation_json"] = _relative_path(root, validation_path)
        manifest["codex_vision_session_raw_result_json"] = _relative_path(root, raw_result_path)
        manifest["mcp_prepare_codex_vision_session_args"] = "mcp-prepare-codex-vision-session.args.json"
        manifest["mcp_import_codex_vision_results_args"] = "mcp-import-codex-vision-results.args.json"
        write_json(manifest_path, manifest)

    manifest_sha256 = sha256_file(manifest_path)
    timeline_sha256 = sha256_file(timeline_path)
    bundle_identity = _bundle_identity_value(
        root,
        manifest_sha256=manifest_sha256,
        timeline_sha256=timeline_sha256,
        kind=session_kind,
        items=candidates,
        viewport_context=viewport_context,
    )
    pack: dict[str, Any] = {
        "schema": PACK_SCHEMA,
        "session_id": session_id,
        "bundle_dir": str(root),
        "kind": session_kind,
        "question": _question(session_kind),
        "executor": executor_name,
        "model_requested": requested_model,
        "quality": quality_name,
        "execution_mode": _execution_mode({"executor": executor_name}),
        "provider_health_check": "not_applicable",
        "manifest_sha256": manifest_sha256,
        "timeline_sha256": timeline_sha256,
        "bundle_identity_sha256": canonical_json_sha256(bundle_identity),
        "task_pack_sha256": "",
        "requested_indexes": requested_indexes,
        "selected_indexes": [int(row["index"]) for row in candidates],
        "items": candidates,
        "viewport_context": viewport_context,
        "blockers": blockers,
        "status": _ready_status({"executor": executor_name}) if candidates and not blockers else "blocked",
        "unattended_execution": executor_name == "codex_cli",
        "api_key_required": False,
        "operator_boundary": {
            "local_bundle_frames_only": True,
            "does_not_read_source_video": True,
            "does_not_call_provider": True,
            "does_not_upload_media": True,
            "requires_interactive_session": executor_name != "codex_cli",
            "execution_requires_explicit_authorization": executor_name == "codex_cli",
            "actual_model_must_be_reported_on_import": True,
        },
        "artifacts": {
            "task_pack_json": str(task_pack_path),
            "prompt_markdown": str(prompt_path),
            "result_template_json": str(result_template_path),
            "import_validation_json": str(validation_path),
            "raw_result_json": str(raw_result_path),
        },
        "created_at": now_iso(),
    }
    pack["task_pack_path"] = str(task_pack_path)
    pack["prompt_path"] = str(prompt_path)
    pack["result_template_path"] = str(result_template_path)
    pack["status_path"] = str(status_path)
    pack["task_pack_sha256"] = _task_pack_sha256(pack)
    result_template = _result_template(pack)

    if write:
        write_json(task_pack_path, pack)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(_render_prompt(pack), encoding="utf-8")
        write_json(result_template_path, result_template)
        write_json(status_path, _public_status(pack, phase="prepared"))
        write_json(
            root / "mcp-prepare-codex-vision-session.args.json",
            {
                "bundle_dir": str(root),
                "kind": session_kind,
                "indexes": requested_indexes,
                "limit": effective_limit,
                "max_frames": frame_budget,
                "model_requested": requested_model,
                "quality": quality_name,
                "executor": executor_name,
                **(
                    {
                        "viewport_pack_json": str(viewport_pack_json),
                        "viewport_timeline_index": int(viewport_timeline_index or requested_indexes[0]),
                        "viewport_sample_index": int(viewport_sample_index),
                    }
                    if viewport_pack_json is not None
                    else {}
                ),
                **(
                    {
                        "panorama_analysis_plan_json": str(panorama_analysis_plan_json),
                        "panorama_task_id": str(panorama_task_id),
                    }
                    if panorama_analysis_plan_json is not None
                    else {}
                ),
                "write": True,
            },
        )
        write_json(
            root / "mcp-import-codex-vision-results.args.json",
            {"bundle_dir": str(root), "input_json": str(result_template_path), "write": True},
        )
        pack["run_registry"] = _register_prepare_run(root, pack)
        # Run registration may add its own manifest pointers. Freeze the immutable
        # session identity only after every prepare-time Bundle write has finished.
        pack["manifest_sha256"] = sha256_file(manifest_path)
        pack["timeline_sha256"] = sha256_file(timeline_path)
        pack["bundle_identity_sha256"] = _bundle_identity_sha256(root, pack)
        pack["task_pack_sha256"] = _task_pack_sha256(pack)
        write_json(task_pack_path, pack)
        write_json(result_template_path, _result_template(pack))
        write_json(status_path, _public_status(pack, phase="prepared"))
    return pack


def import_codex_vision_results(
    bundle_dir: str | Path,
    *,
    input_json: str | Path,
    write: bool = True,
) -> dict[str, Any]:
    """Validate and atomically import one Codex vision result."""

    root = Path(bundle_dir).expanduser().resolve()
    input_path = Path(input_json).expanduser().resolve()
    raw_bytes, parsed, parse_issues = _read_result_input(input_path)
    session_id = str(parsed.get("session_id") or "") if isinstance(parsed, dict) else ""
    safe_session_id = session_id if SESSION_ID_RE.fullmatch(session_id) else f"rejected-{uuid4().hex}"
    session_dir = root / "codex-vision-sessions" / safe_session_id
    raw_result_path = session_dir / "raw-result.json"
    validation_path = session_dir / "import-validation.json"
    validation_markdown_path = session_dir / "import-validation.md"
    report_path = session_dir / "import-report.md"
    task_pack_path = session_dir / "task-pack.json"
    status_path = root / "codex-vision-session-status.json"

    with bundle_write_lock(root, operation="import_codex_vision_results", timeout_seconds=1.0):
        timeline_path = root / "timeline.json"
        manifest_path = root / "manifest.json"
        timeline_before_bytes = timeline_path.read_bytes()
        timeline = _read_timeline(timeline_path)
        before_timeline = snapshot_timeline_for_vision_diff(timeline)
        manifest = _read_mapping(manifest_path, "manifest.json")
        pack = _read_optional_mapping(task_pack_path)
        issues = [*parse_issues]
        if not SESSION_ID_RE.fullmatch(session_id):
            issues.append(_issue("session_id_invalid", "session_id is missing or unsafe"))
        if not pack:
            issues.append(_issue("task_pack_missing", f"task pack not found: {task_pack_path}"))
        if isinstance(parsed, dict) and pack:
            issues.extend(_validate_result(root, parsed, pack, timeline, manifest_path=manifest_path, timeline_path=timeline_path))

        if write:
            _write_bytes_atomic(raw_result_path, raw_bytes)

        if issues:
            validation = _validation_receipt(
                root,
                input_path=input_path,
                task_pack_path=task_pack_path,
                raw_result_path=raw_result_path,
                status="rejected",
                issues=issues,
                parsed=parsed,
                pack=pack,
                updated_count=0,
            )
            summary = _import_summary(parsed, pack, status="rejected", updated_count=0, raw_result_path=raw_result_path)
            results = _audit_results(parsed, ok=False)
            if write:
                write_json(validation_path, validation)
                validation_markdown_path.write_text(_render_validation(validation), encoding="utf-8")
                report_path.write_text(_render_import_report(summary, validation), encoding="utf-8")
                run_audit = write_vision_analysis_run_audit(
                    root,
                    kind=str(pack.get("executor") or "codex_session"),
                    summary=summary,
                    results=results,
                    report_path=report_path,
                    template_path=task_pack_path,
                    timeline_diff=[],
                )
                write_json(status_path, _public_import_status(summary, validation, run_audit=run_audit))
            else:
                run_audit = {}
            if timeline_path.read_bytes() != timeline_before_bytes:
                raise RuntimeError("fail_closed_violation: rejected import changed timeline.json")
            return {
                "schema": VALIDATION_SCHEMA,
                "bundle_dir": str(root),
                "status": "rejected",
                "updated_count": 0,
                "validation_issues": issues,
                "validation_path": str(validation_path),
                "validation_markdown_path": str(validation_markdown_path),
                "raw_result_path": str(raw_result_path),
                "run_audit": run_audit,
            }

        assert isinstance(parsed, dict)
        assert isinstance(pack, dict)
        imported_results, applied_indexes = _apply_validated_results(timeline, parsed, pack)
        timeline_diff = timeline_vision_diff(before_timeline, timeline, applied_indexes)
        validation = _validation_receipt(
            root,
            input_path=input_path,
            task_pack_path=task_pack_path,
            raw_result_path=raw_result_path,
            status="ready_to_import" if not write else "imported",
            issues=[],
            parsed=parsed,
            pack=pack,
            updated_count=len(applied_indexes),
        )
        summary = _import_summary(
            parsed,
            pack,
            status="validated" if not write else "imported",
            updated_count=len(applied_indexes),
            raw_result_path=raw_result_path,
        )
        if write:
            write_json(timeline_path, timeline)
            if pack.get("viewport_context"):
                field = (
                    "panorama_temporal_understanding"
                    if str(pack.get("kind") or "") == "temporal"
                    else "panorama_visual_understanding"
                )
            else:
                field = "visual_understanding" if str(pack.get("kind")) == "semantic" else "temporal_visual_understanding"
            _sync_source_package(manifest, timeline, field)
            manifest["coverage"] = _coverage_audit(timeline)
            manifest["quality_audit"] = _quality_audit(timeline)
            manifest["repair_status"] = build_repair_status(manifest, timeline)
            manifest["vision_analysis_runs"] = "vision-analysis-runs.md"
            manifest["vision_analysis_runs_jsonl"] = "vision-analysis-runs.jsonl"
            manifest["codex_vision_session_status_json"] = status_path.name
            manifest["codex_vision_session_import_validation_json"] = _relative_path(root, validation_path)
            manifest["codex_vision_session_raw_result_json"] = _relative_path(root, raw_result_path)
            write_json(manifest_path, manifest)
            write_json(validation_path, validation)
            validation_markdown_path.write_text(_render_validation(validation), encoding="utf-8")
            report_path.write_text(_render_import_report(summary, validation), encoding="utf-8")
            run_audit = write_vision_analysis_run_audit(
                root,
                kind=f"{pack.get('executor') or 'codex_session'}_{pack.get('kind')}",
                summary=summary,
                results=imported_results,
                report_path=report_path,
                template_path=task_pack_path,
                timeline_diff=timeline_diff,
            )
        else:
            run_audit = {}

    post_run_refresh = _refresh_post_vision_outputs(root, updated_count=len(applied_indexes)) if write else {"status": "skipped", "reason": "write_false"}
    restore_hint = build_vision_restore_hint(root, run_audit) if write else {"status": "not_written"}
    if write:
        run_registry = _register_import_run(root, pack, summary, run_audit, report_path, raw_result_path, validation_path)
        write_json(status_path, _public_import_status(summary, validation, run_audit=run_audit, restore_hint=restore_hint))
    else:
        run_registry = {}
    return {
        "schema": VALIDATION_SCHEMA,
        "bundle_dir": str(root),
        "status": "imported" if write else "ready_to_import",
        "updated_count": len(applied_indexes),
        "updated_indexes": applied_indexes,
        "validation_issues": [],
        "validation_path": str(validation_path),
        "validation_markdown_path": str(validation_markdown_path),
        "raw_result_path": str(raw_result_path),
        "report_path": str(report_path),
        "run_audit": run_audit,
        "run_registry": run_registry,
        "post_run_refresh": post_run_refresh,
        "vision_restore_hint": restore_hint,
        "summary": summary,
    }


def codex_session_preflight_summary(
    bundle_dir: str | Path,
    timeline: list[dict[str, Any]],
    *,
    semantic_selected: list[int],
    temporal_selected: list[int],
    temporal_missing_groups: list[int],
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    semantic_ready = _count_local_ready(root, timeline, semantic_selected, kind="semantic")
    temporal_ready = _count_local_ready(root, timeline, temporal_selected, kind="temporal")
    return {
        "executor": "codex_session",
        "execution_mode": "interactive_multimodal",
        "api_key_required": False,
        "provider_health_check": "not_applicable",
        "unattended_execution": False,
        "ready_to_prepare": bool(semantic_ready or temporal_ready),
        "semantic_ready": semantic_ready,
        "temporal_ready": temporal_ready,
        "semantic_indexes": list(semantic_selected),
        "temporal_indexes": list(temporal_selected[:1]),
        "temporal_missing_frame_groups": len(temporal_missing_groups),
        "commands": {
            "prepare_semantic": _prepare_command(root, "semantic", semantic_selected),
            "prepare_temporal": _prepare_command(root, "temporal", temporal_selected[:1]),
            "import_results": f".\\scripts\\video-knowledge.ps1 import-codex-vision-results {quote_powershell_literal(root)} --input-json <result.json>",
        },
        "mcp_args": {
            "prepare_semantic": {
                "bundle_dir": str(root),
                "kind": "semantic",
                "indexes": list(semantic_selected),
                "model_requested": "luna",
                "quality": "low",
                "executor": "codex_session",
                "write": True,
            },
            "prepare_temporal": {
                "bundle_dir": str(root),
                "kind": "temporal",
                "indexes": list(temporal_selected[:1]),
                "model_requested": "luna",
                "quality": "low",
                "executor": "codex_session",
                "write": True,
            },
            "import_results": {"bundle_dir": str(root), "input_json": "<result.json>", "write": True},
        },
    }


def _session_candidates(
    root: Path,
    timeline: list[dict[str, Any]],
    *,
    kind: str,
    requested_indexes: list[int],
    limit: int,
    max_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if kind == "temporal" and len(requested_indexes) > 1:
        return [], [_issue("temporal_single_group_required", "prepare one temporal timeline index per session")]
    requested = set(requested_indexes)
    candidates: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    temporal_missing: list[int] = []
    seen_requested: set[int] = set()
    for position, item in enumerate(timeline, start=1):
        index = _positive_int(item.get("index")) or position
        if requested and index not in requested:
            continue
        if requested:
            seen_requested.add(index)
        route = str(item.get("visual_route") or "")
        field = "visual_understanding" if kind == "semantic" else "temporal_visual_understanding"
        if _has_formal_result(item, kind=kind):
            if requested:
                blockers.append(_issue("existing_result", f"timeline {index} already has {field}"))
            continue
        eligible = route in ({"semantic_frame", "mixed"} if kind == "semantic" else {"temporal_sequence", "mixed"})
        if not eligible and not requested:
            continue
        paths = _semantic_local_paths(root, item) if kind == "semantic" else _temporal_local_paths(root, item)
        if kind == "semantic":
            if not paths:
                if requested:
                    blockers.append(_issue("bundle_frame_missing", f"timeline {index} has no local Bundle frame"))
                continue
            paths = paths[:1]
        else:
            if not MIN_TEMPORAL_FRAMES <= len(paths) <= MAX_TEMPORAL_FRAMES:
                if requested:
                    blockers.append(
                        _issue(
                            "temporal_frame_group_invalid",
                            f"timeline {index} requires {MIN_TEMPORAL_FRAMES}-{MAX_TEMPORAL_FRAMES} Bundle frames; found {len(paths)}",
                        )
                    )
                elif eligible:
                    temporal_missing.append(index)
                continue
        frame_manifest = build_temporal_frame_manifest([str(path) for path in paths])
        frames = [
            {
                "frame_id": str(frame.get("frame_id") or ""),
                "timestamp_ms": frame.get("timestamp_ms"),
                "timestamp": str(frame.get("timestamp") or ""),
                "filename": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, frame in zip(paths, frame_manifest, strict=True)
        ]
        candidates.append(
            {
                "task_id": f"{kind}-{index:06d}",
                "kind": kind,
                "index": index,
                "start": _float(item.get("start")),
                "end": _float(item.get("end")),
                "visual_route": route,
                "transcript": str(item.get("transcript") or "")[:MAX_STRING_CHARS],
                "visual_text": str(item.get("visual_text") or "")[:MAX_STRING_CHARS],
                "frame_paths": [str(path) for path in paths],
                "frames": frames,
                "expected_frame_ids": [str(frame["frame_id"]) for frame in frames],
            }
        )
        if kind == "temporal":
            break
    missing_requested = requested - seen_requested
    for index in sorted(missing_requested):
        blockers.append(_issue("timeline_index_missing", f"timeline index {index} does not exist"))
    if kind == "temporal" and not candidates and temporal_missing:
        blockers.append(
            _issue(
                "temporal_frame_group_missing",
                "no ready temporal group; missing/invalid Bundle frame groups: "
                + ",".join(str(value) for value in temporal_missing),
            )
        )
    if kind == "semantic":
        original_candidates = list(candidates)
        selected: list[dict[str, Any]] = []
        frame_count = 0
        item_limit = limit if limit > 0 else len(original_candidates)
        for row in original_candidates:
            if len(selected) >= item_limit:
                break
            row_frames = len(row["frames"])
            if frame_count + row_frames > max_frames:
                break
            selected.append(row)
            frame_count += row_frames
        candidates = selected
        omitted_requested = requested.intersection(
            {int(row["index"]) for row in original_candidates}
        ) - {int(row["index"]) for row in candidates}
        if omitted_requested:
            blockers.append(
                _issue(
                    "semantic_batch_capacity_exceeded",
                    "explicit indexes exceed this immutable batch capacity; split them across sessions: "
                    + ",".join(str(value) for value in sorted(omitted_requested)),
                )
            )
    return candidates, _unique_issues(blockers)


def _semantic_local_paths(root: Path, item: dict[str, Any]) -> list[Path]:
    values: list[Any] = []
    assets = item.get("assets") if isinstance(item.get("assets"), list) else []
    for asset in assets:
        if isinstance(asset, dict) and asset.get("path"):
            values.append(asset.get("path"))
        elif isinstance(asset, str):
            values.append(asset)
    values.extend(item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else [])
    return _local_bundle_files(root, values)


def _temporal_local_paths(root: Path, item: dict[str, Any]) -> list[Path]:
    values = item.get("temporal_frame_paths") if isinstance(item.get("temporal_frame_paths"), list) else []
    return _local_bundle_files(root, values)


def _local_bundle_files(root: Path, values: list[Any]) -> list[Path]:
    files: list[Path] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        if path not in files:
            files.append(path)
    return files


def _result_template(pack: dict[str, Any]) -> dict[str, Any]:
    items = []
    for task in pack.get("items") or []:
        row: dict[str, Any] = {
            "task_id": task["task_id"],
            "kind": task["kind"],
            "index": task["index"],
            "start": task["start"],
            "end": task["end"],
            "frame_paths": list(task["frame_paths"]),
            "expected_frame_ids": list(task["expected_frame_ids"]),
            "observed_frame_ids": [],
            "evidence_level": "confirmed|inferred|unavailable",
            "inference_basis": "",
            "unavailable_reason": "",
        }
        if task["kind"] == "semantic":
            row["visual_understanding"] = {
                "objects": [],
                "actions": [],
                "interface_state": "",
                "spatial_relations": [],
                "instructor_focus": "",
                "non_text_information": [],
                "confidence": 0.0,
                "keep_image_reason": "",
                "evidence_frame_paths": list(task["frame_paths"]),
            }
            if str(task.get("task_type") or "") == "panorama_six_view_semantic":
                from .panorama_session import panorama_result_template

                row["panorama_viewport_observations"] = panorama_result_template(task)
        else:
            row["temporal_visual_understanding"] = {
                "event_sequence": [],
                "state_changes": [],
                "operation_steps": [],
                "causal_links": [],
                "possible_missing_points": [],
                "confidence": 0.0,
                "observed_frame_count": 0,
                "observed_frame_ids": [],
                "per_frame_observations": [],
                "evidence_frame_paths": list(task["frame_paths"]),
            }
        items.append(row)
    return {
        "schema": RESULT_SCHEMA,
        "session_id": pack["session_id"],
        "task_pack_sha256": pack["task_pack_sha256"],
        "bundle_identity_sha256": pack["bundle_identity_sha256"],
        "executor": str(pack.get("executor") or "codex_session"),
        "model_requested": pack["model_requested"],
        "model_actual": "",
        "quality": pack["quality"],
        "execution_mode": _execution_mode(pack),
        "provider_health_check": "not_applicable",
        "items": items,
    }


def _validate_result(
    root: Path,
    result: dict[str, Any],
    pack: dict[str, Any],
    timeline: list[dict[str, Any]],
    *,
    manifest_path: Path,
    timeline_path: Path,
) -> list[dict[str, str]]:
    issues = _bounded_value_issues(result)
    if pack.get("viewport_context"):
        from .panorama_session import validate_panorama_pack

        issues.extend(validate_panorama_pack(root, pack))
    if result.get("schema") != RESULT_SCHEMA:
        issues.append(_issue("result_schema_invalid", f"expected {RESULT_SCHEMA}"))
    if pack.get("schema") != PACK_SCHEMA:
        issues.append(_issue("task_pack_schema_invalid", f"expected {PACK_SCHEMA}"))
    if pack.get("status") != _ready_status(pack) or pack.get("blockers"):
        issues.append(_issue("task_pack_not_ready", "blocked or incomplete task packs cannot be imported"))
    for key in ("session_id", "task_pack_sha256", "bundle_identity_sha256", "executor", "model_requested", "quality", "execution_mode", "provider_health_check"):
        if str(result.get(key) or "") != str(pack.get(key) or ""):
            issues.append(_issue(f"{key}_mismatch", f"result {key} does not match task pack"))
    if not str(result.get("model_actual") or "").strip():
        issues.append(_issue("model_actual_missing", "model_actual must report the execution model"))
    if str(pack.get("task_pack_sha256") or "") != _task_pack_sha256(pack):
        issues.append(_issue("task_pack_integrity_failed", "task pack content changed after preparation"))
    if str(pack.get("manifest_sha256") or "") != sha256_file(manifest_path):
        issues.append(_issue("manifest_sha256_mismatch", "manifest.json changed after preparation"))
    if str(pack.get("timeline_sha256") or "") != sha256_file(timeline_path):
        issues.append(_issue("timeline_sha256_mismatch", "timeline.json changed after preparation"))
    identity = _bundle_identity_value(
        root,
        manifest_sha256=pack.get("manifest_sha256"),
        timeline_sha256=pack.get("timeline_sha256"),
        kind=pack.get("kind"),
        items=[row for row in pack.get("items") or [] if isinstance(row, dict)],
        viewport_context=pack.get("viewport_context") if isinstance(pack.get("viewport_context"), dict) else {},
    )
    if str(pack.get("bundle_identity_sha256") or "") != canonical_json_sha256(identity):
        issues.append(_issue("bundle_identity_mismatch", "bundle identity no longer matches task pack"))
    pack_items = [row for row in pack.get("items") or [] if isinstance(row, dict)]
    result_items = result.get("items") if isinstance(result.get("items"), list) else []
    pack_by_id = {str(row.get("task_id") or ""): row for row in pack_items}
    result_ids = [str(row.get("task_id") or "") for row in result_items if isinstance(row, dict)]
    if len(result_ids) != len(set(result_ids)):
        issues.append(_issue("duplicate_task_id", "result contains duplicate task_id values"))
    if set(result_ids) != set(pack_by_id):
        issues.append(_issue("result_task_set_mismatch", "result items must match the complete task pack exactly"))
    keys: list[tuple[str, int]] = []
    for row in result_items:
        if not isinstance(row, dict):
            issues.append(_issue("result_item_invalid", "every result item must be an object"))
            continue
        kind = str(row.get("kind") or "")
        index = _positive_int(row.get("index"))
        keys.append((kind, index))
        task = pack_by_id.get(str(row.get("task_id") or ""))
        if not task:
            continue
        issues.extend(_validate_result_item(root, row, task, timeline))
    if len(keys) != len(set(keys)):
        issues.append(_issue("duplicate_kind_index", "result contains duplicate (kind, index) values"))
    if pack.get("executor") == "codex_cli":
        from .codex_vision_executor import validate_execution_receipt

        issues.extend(validate_execution_receipt(root, result, pack))
    return _unique_issues(issues)


def _validate_result_item(root: Path, row: dict[str, Any], task: dict[str, Any], timeline: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for key in ("kind", "index"):
        if row.get(key) != task.get(key):
            issues.append(_issue(f"item_{key}_mismatch", f"{task.get('task_id')} {key} does not match"))
    for key in ("start", "end"):
        if not _finite_number(row.get(key)):
            issues.append(_issue("item_time_invalid", f"{task.get('task_id')} {key} must be a finite number"))
        elif abs(_float(row.get(key)) - _float(task.get(key))) > TIME_TOLERANCE_SECONDS:
            issues.append(_issue("item_time_mismatch", f"{task.get('task_id')} {key} exceeds 1ms tolerance"))
    index = _positive_int(task.get("index"))
    if not 1 <= index <= len(timeline):
        issues.append(_issue("timeline_index_out_of_range", f"timeline index {index} is unavailable"))
    else:
        current = timeline[index - 1]
        field = "visual_understanding" if task.get("kind") == "semantic" else "temporal_visual_understanding"
        if str(task.get("task_type") or "") not in {
            "panorama_six_view_semantic",
            "panorama_temporal_direction",
        } and _has_formal_result(
            current, kind=str(task.get("kind") or "semantic")
        ):
            issues.append(_issue("existing_result", f"timeline {index} already has {field}"))
        if abs(_float(current.get("start")) - _float(task.get("start"))) > TIME_TOLERANCE_SECONDS or abs(_float(current.get("end")) - _float(task.get("end"))) > TIME_TOLERANCE_SECONDS:
            issues.append(_issue("timeline_time_changed", f"timeline {index} time range changed"))
    expected_paths = [str(frame.get("path") or "") for frame in task.get("frames") or [] if isinstance(frame, dict)]
    expected_ids = [str(frame.get("frame_id") or "") for frame in task.get("frames") or [] if isinstance(frame, dict)]
    for frame in task.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        path = Path(str(frame.get("path") or "")).expanduser().resolve()
        if not path.is_relative_to(root) or not path.is_file():
            issues.append(_issue("frame_missing_or_external", f"frame is unavailable inside Bundle: {path}"))
        elif sha256_file(path) != str(frame.get("sha256") or ""):
            issues.append(_issue("frame_sha256_mismatch", f"frame changed after preparation: {path}"))
    if [str(value) for value in row.get("frame_paths") or []] != expected_paths:
        issues.append(_issue("frame_paths_mismatch", f"{task.get('task_id')} frame_paths changed"))
    if [str(value) for value in row.get("expected_frame_ids") or []] != expected_ids:
        issues.append(_issue("expected_frame_ids_mismatch", f"{task.get('task_id')} expected_frame_ids changed"))
    observed_ids = [str(value) for value in row.get("observed_frame_ids") or []]
    if observed_ids != expected_ids:
        issues.append(_issue("observed_frame_ids_incomplete", f"{task.get('task_id')} must observe every expected frame in order"))
    evidence_level = str(row.get("evidence_level") or "")
    if evidence_level not in EVIDENCE_LEVELS:
        issues.append(_issue("evidence_level_invalid", f"{task.get('task_id')} evidence_level must be confirmed, inferred, or unavailable"))
    if evidence_level == "inferred" and not str(row.get("inference_basis") or "").strip():
        issues.append(_issue("inference_basis_missing", f"{task.get('task_id')} inferred evidence requires inference_basis"))
    if str(task.get("task_type") or "") == "panorama_six_view_semantic":
        from .panorama_session import validate_panorama_result_item

        issues.extend(validate_panorama_result_item(row, task))
    if evidence_level == "unavailable":
        if not str(row.get("unavailable_reason") or "").strip():
            issues.append(_issue("unavailable_reason_missing", f"{task.get('task_id')} unavailable evidence requires a reason"))
        field = "visual_understanding" if task.get("kind") == "semantic" else "temporal_visual_understanding"
        unavailable_understanding = row.get(field) if isinstance(row.get(field), dict) else {}
        evidence_paths = [str(value) for value in unavailable_understanding.get("evidence_frame_paths") or []]
        if evidence_paths and not set(evidence_paths).issubset(set(expected_paths)):
            issues.append(_issue("evidence_frame_paths_invalid", f"{task.get('task_id')} evidence paths must come from the task pack"))
        if task.get("kind") == "temporal":
            understanding = unavailable_understanding
            nested_ids = [str(value) for value in understanding.get("observed_frame_ids") or []]
            observations = understanding.get("per_frame_observations") if isinstance(understanding.get("per_frame_observations"), list) else []
            observation_ids = [str(value.get("frame_id") or "") for value in observations if isinstance(value, dict)]
            if nested_ids != expected_ids or _positive_int(understanding.get("observed_frame_count")) != len(expected_ids):
                issues.append(_issue("temporal_observation_incomplete", f"{task.get('task_id')} unavailable response must still cover every frame"))
            if observation_ids != expected_ids or any(not str(value.get("observation") or "").strip() for value in observations if isinstance(value, dict)):
                issues.append(_issue("per_frame_observations_incomplete", f"{task.get('task_id')} unavailable response still requires one ordered observation per frame"))
        return issues
    field = "visual_understanding" if task.get("kind") == "semantic" else "temporal_visual_understanding"
    understanding = row.get(field) if isinstance(row.get(field), dict) else {}
    if not understanding:
        issues.append(_issue("understanding_missing", f"{task.get('task_id')} requires {field}"))
        return issues
    evidence_paths = [str(value) for value in understanding.get("evidence_frame_paths") or []]
    if not evidence_paths or not set(evidence_paths).issubset(set(expected_paths)):
        issues.append(_issue("evidence_frame_paths_invalid", f"{task.get('task_id')} evidence paths must come from the task pack"))
    if task.get("kind") == "temporal":
        nested_ids = [str(value) for value in understanding.get("observed_frame_ids") or []]
        if nested_ids != expected_ids or _positive_int(understanding.get("observed_frame_count")) != len(expected_ids):
            issues.append(_issue("temporal_observation_incomplete", f"{task.get('task_id')} temporal observation count/ids must match"))
        observations = understanding.get("per_frame_observations") if isinstance(understanding.get("per_frame_observations"), list) else []
        observation_ids = [str(value.get("frame_id") or "") for value in observations if isinstance(value, dict)]
        if observation_ids != expected_ids or any(not str(value.get("observation") or "").strip() for value in observations if isinstance(value, dict)):
            issues.append(_issue("per_frame_observations_incomplete", f"{task.get('task_id')} requires one ordered observation per frame"))
    return issues


def _apply_validated_results(
    timeline: list[dict[str, Any]],
    result: dict[str, Any],
    pack: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    tasks = {str(row.get("task_id") or ""): row for row in pack.get("items") or [] if isinstance(row, dict)}
    applied: list[int] = []
    audit_results: list[dict[str, Any]] = []
    for row in result.get("items") or []:
        task = tasks[str(row.get("task_id") or "")]
        index = int(task["index"])
        level = str(row.get("evidence_level") or "")
        common = {
            "evidence_level": level,
            "observation_status": "seen_unavailable" if level == "unavailable" else "seen",
            "inference_basis": str(row.get("inference_basis") or ""),
            "unavailable_reason": str(row.get("unavailable_reason") or ""),
            "executor": str(pack.get("executor") or "codex_session"),
            "model_requested": str(result.get("model_requested") or ""),
            "model_actual": str(result.get("model_actual") or ""),
            "quality": str(result.get("quality") or ""),
            "execution_mode": _execution_mode(pack),
            "provider_health_check": "not_applicable",
            "task_pack_sha256": str(result.get("task_pack_sha256") or ""),
            "observed_frame_ids": [str(value) for value in row.get("observed_frame_ids") or []],
            # Derive only from the already hash-validated pack, never from a
            # model-supplied provenance object. Text context means this is not
            # an independent blind reading of the frame.
            "recognition_provenance": {
                "schema": "video_knowledge_pipeline.recognition_provenance.v1",
                "executor": str(pack.get("executor") or "codex_session"),
                "model_actual": str(result.get("model_actual") or ""),
                "model_requested": str(result.get("model_requested") or ""),
                "task_pack_sha256": str(pack.get("task_pack_sha256") or ""),
                "frame_sha256s": sorted({str(frame["sha256"]) for frame in task["frames"]}),
                "context_text_sha256s": sorted({canonical_json_sha256(task[key])
                    for key in ("transcript", "visual_text") if task.get(key)}),
                "input_identity_basis": "validated_session_task_pack",
                "independence_is_verified": False,
            },
        }
        if pack.get("executor") == "codex_cli":
            common["execution_receipt"] = dict(result["execution_receipt"])
            common["model_actual_basis"] = "explicit_cli_selection_not_server_attestation"
            common["server_model_attested"] = False
        if task["kind"] == "semantic":
            if level == "unavailable":
                payload = _unavailable_payload(task, common, response=row, temporal=False)
            else:
                payload = _normalise_visual_understanding(
                    dict(row.get("visual_understanding") or {}),
                    {"frame_paths": list(task["frame_paths"])},
                )
                payload.update(common)
                payload["validation_status"] = "ok" if level == "confirmed" else "inferred"
                payload["validation_issues"] = [] if level == "confirmed" else ["inferred_evidence_requires_qualification"]
            if str(task.get("task_type") or "") == "panorama_six_view_semantic":
                from .panorama_session import apply_panorama_result

                payload = apply_panorama_result(
                    timeline[index - 1],
                    row=row,
                    task=task,
                    result=result,
                    normalized_visual=payload,
                )
                result_key = "panorama_visual_understanding"
            else:
                _apply_visual_single(timeline, index, "visual_understanding", payload)
                _update_quality_issue(timeline[index - 1], level, temporal=False)
                result_key = "visual_understanding"
        else:
            candidate = {
                "frame_paths": list(task["frame_paths"]),
                "frame_manifest": build_temporal_frame_manifest(list(task["frame_paths"])),
                "strict_frame_contract": True,
            }
            if level == "unavailable":
                payload = _unavailable_payload(task, common, response=row, temporal=True)
            else:
                payload = _normalise_temporal_understanding(dict(row.get("temporal_visual_understanding") or {}), candidate)
                payload.update(common)
                payload["validation_status"] = "ok" if level == "confirmed" else "inferred"
                payload["validation_issues"] = [] if level == "confirmed" else ["inferred_evidence_requires_qualification"]
            if str(task.get("task_type") or "") == "panorama_temporal_direction":
                from .panorama_session import apply_panorama_temporal_result

                payload = apply_panorama_temporal_result(
                    timeline[index - 1],
                    row=row,
                    task=task,
                    result=result,
                    normalized_temporal=payload,
                )
                result_key = "panorama_temporal_understanding"
            else:
                _apply_temporal_single(timeline, index, payload)
                _update_quality_issue(timeline[index - 1], level, temporal=True)
                result_key = "temporal_visual_understanding"
        applied.append(index)
        audit_results.append(
            {
                "task_id": task["task_id"],
                "index": index,
                "executed": True,
                "ok": True,
                "complete": True,
                "truncated": False,
                "status": level,
                "evidence_level": level,
                result_key: payload,
            }
        )
    return audit_results, applied


def _unavailable_payload(
    task: dict[str, Any],
    common: dict[str, Any],
    *,
    response: dict[str, Any],
    temporal: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lecture_temporal_visual_understanding.v2" if temporal else "lecture_visual_understanding.v1",
        "confidence": 0.0,
        "evidence_frame_paths": list(task["frame_paths"]),
        "source": str(common.get("executor") or "codex_session"),
        "parse_failed": False,
        "validation_status": "unavailable",
        "validation_issues": ["seen_unavailable"],
        **common,
    }
    if temporal:
        understanding = response.get("temporal_visual_understanding") if isinstance(response.get("temporal_visual_understanding"), dict) else {}
        payload.update(
            {
                "event_sequence": [],
                "state_changes": [],
                "operation_steps": [],
                "causal_links": [],
                "possible_missing_points": [str(common.get("unavailable_reason") or "")],
                "expected_frame_count": len(task["expected_frame_ids"]),
                "expected_frame_ids": list(task["expected_frame_ids"]),
                "observed_frame_count": len(task["expected_frame_ids"]),
                "per_frame_observations": [
                    dict(value)
                    for value in understanding.get("per_frame_observations") or []
                    if isinstance(value, dict)
                ],
            }
        )
    else:
        payload.update(
            {
                "objects": [],
                "actions": [],
                "interface_state": "",
                "spatial_relations": [],
                "instructor_focus": "",
                "non_text_information": [],
                "keep_image_reason": "",
            }
        )
    return payload


def _update_quality_issue(item: dict[str, Any], level: str, *, temporal: bool) -> None:
    issues = [
        str(value)
        for value in item.get("quality_issues") or []
        if str(value)
        not in {
            "semantic_frame_without_analysis",
            "temporal_sequence_without_analysis",
            "missing_visual_understanding",
            "visual_evidence_inferred",
            "visual_evidence_unavailable",
            "temporal_evidence_inferred",
            "temporal_evidence_unavailable",
        }
    ]
    if level == "inferred":
        issues.append("temporal_evidence_inferred" if temporal else "visual_evidence_inferred")
    elif level == "unavailable":
        issues.append("temporal_evidence_unavailable" if temporal else "visual_evidence_unavailable")
        issues.append("temporal_sequence_without_analysis" if temporal else "semantic_frame_without_analysis")
    item["quality_issues"] = list(dict.fromkeys(issues))


def _read_result_input(path: Path) -> tuple[bytes, dict[str, Any], list[dict[str, str]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return b"", {}, [_issue("input_json_unreadable", str(exc))]
    if len(raw) > MAX_RESULT_BYTES:
        return raw, {}, [_issue("input_json_too_large", f"result exceeds {MAX_RESULT_BYTES} bytes")]
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:
        return raw, {}, [_issue("input_json_invalid", str(exc))]
    if not isinstance(value, dict):
        return raw, {}, [_issue("input_json_not_object", "result JSON must contain an object")]
    return raw, value, []


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_bytes(payload)
        replace_file_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bounded_value_issues(value: Any, *, path: str = "$", depth: int = 0) -> list[dict[str, str]]:
    if depth > MAX_NESTING_DEPTH:
        return [_issue("result_nesting_too_deep", f"{path} exceeds depth {MAX_NESTING_DEPTH}")]
    issues: list[dict[str, str]] = []
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        issues.append(_issue("result_string_too_long", f"{path} exceeds {MAX_STRING_CHARS} characters"))
    elif isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            issues.append(_issue("result_list_too_long", f"{path} exceeds {MAX_COLLECTION_ITEMS} items"))
        for index, child in enumerate(value[: MAX_COLLECTION_ITEMS + 1]):
            issues.extend(_bounded_value_issues(child, path=f"{path}[{index}]", depth=depth + 1))
    elif isinstance(value, dict):
        if len(value) > MAX_COLLECTION_ITEMS:
            issues.append(_issue("result_object_too_large", f"{path} exceeds {MAX_COLLECTION_ITEMS} fields"))
        for key, child in list(value.items())[: MAX_COLLECTION_ITEMS + 1]:
            issues.extend(_bounded_value_issues(child, path=f"{path}.{key}", depth=depth + 1))
    return issues


def _validation_receipt(
    root: Path,
    *,
    input_path: Path,
    task_pack_path: Path,
    raw_result_path: Path,
    status: str,
    issues: list[dict[str, str]],
    parsed: dict[str, Any],
    pack: dict[str, Any],
    updated_count: int,
) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "bundle_dir": str(root),
        "session_id": str(parsed.get("session_id") or pack.get("session_id") or ""),
        "status": status,
        "ok": not issues,
        "input_json": str(input_path),
        "raw_result_path": str(raw_result_path),
        "task_pack_path": str(task_pack_path),
        "task_pack_sha256": str(pack.get("task_pack_sha256") or ""),
        "model_requested": str(pack.get("model_requested") or parsed.get("model_requested") or ""),
        "model_actual": str(parsed.get("model_actual") or ""),
        "updated_count": int(updated_count),
        "issues": issues,
        "checked_at": now_iso(),
    }


def _import_summary(parsed: dict[str, Any], pack: dict[str, Any], *, status: str, updated_count: int, raw_result_path: Path) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.codex_vision_import_summary.v1",
        "execute": True,
        "status": status,
        "session_id": str(pack.get("session_id") or parsed.get("session_id") or ""),
        "kind": str(pack.get("kind") or ""),
        "artifacts": dict(pack.get("artifacts") or {}),
        "error": "" if status in {"imported", "validated"} else "codex_vision_import_rejected",
        "executor": str(pack.get("executor") or "codex_session"),
        "model_requested": str(pack.get("model_requested") or parsed.get("model_requested") or ""),
        "model_actual": str(parsed.get("model_actual") or ""),
        "quality": str(pack.get("quality") or parsed.get("quality") or ""),
        "execution_mode": _execution_mode(pack),
        "provider_health_check": "not_applicable",
        "task_pack_sha256": str(pack.get("task_pack_sha256") or parsed.get("task_pack_sha256") or ""),
        "task_pack_path": str(pack.get("task_pack_path") or (pack.get("artifacts") or {}).get("task_pack_json") or ""),
        "raw_result_sha256": sha256_file(raw_result_path) if raw_result_path.is_file() else "",
        "raw_result_path": str(raw_result_path),
        "selected": len(pack.get("items") or []),
        "total": len(pack.get("items") or []),
        "limit": len(pack.get("items") or []),
        "frame_count": sum(len(row.get("frames") or []) for row in pack.get("items") or [] if isinstance(row, dict)),
        "updated": int(updated_count),
        "imported": len(parsed.get("items") or []) if isinstance(parsed.get("items"), list) else 0,
        "indexes": [int(row.get("index")) for row in pack.get("items") or [] if isinstance(row, dict) and _positive_int(row.get("index"))],
        "provider": {
            "provider": str(pack.get("executor") or "codex_session"),
            "base_url": "",
            "model": str(parsed.get("model_actual") or ""),
            "api_key_configured": False,
        },
        "execution_control": {
            "schema": "lecture_vision_execution_control.v1",
            "execute": True,
            "preflight_required": False,
            "status": status,
            "confirmed": status in {"imported", "validated"},
            "execution_actor": "codex_cli" if pack.get("executor") == "codex_cli" else "interactive_session",
        },
        "updated_at": now_iso(),
    }


def _audit_results(parsed: dict[str, Any], *, ok: bool) -> list[dict[str, Any]]:
    return [
        {
            "task_id": str(row.get("task_id") or ""),
            "index": _positive_int(row.get("index")),
            "executed": True,
            "ok": bool(ok),
            "complete": bool(ok),
            "truncated": False,
            "error": "" if ok else "codex_vision_import_rejected",
        }
        for row in parsed.get("items") or []
        if isinstance(row, dict)
    ]


def _task_pack_sha256(pack: dict[str, Any]) -> str:
    value = dict(pack)
    value.pop("task_pack_sha256", None)
    return canonical_json_sha256(value)


def _bundle_identity_sha256(root: Path, pack: dict[str, Any]) -> str:
    identity = _bundle_identity_value(
        root,
        manifest_sha256=pack.get("manifest_sha256"),
        timeline_sha256=pack.get("timeline_sha256"),
        kind=pack.get("kind"),
        items=[row for row in pack.get("items") or [] if isinstance(row, dict)],
        viewport_context=pack.get("viewport_context") if isinstance(pack.get("viewport_context"), dict) else {},
    )
    return canonical_json_sha256(identity)


def _bundle_identity_value(
    root: Path,
    *,
    manifest_sha256: Any,
    timeline_sha256: Any,
    kind: Any,
    items: list[dict[str, Any]],
    viewport_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bundle_dir": str(root),
        "manifest_sha256": manifest_sha256,
        "timeline_sha256": timeline_sha256,
        "kind": kind,
        "items": [
            {
                "task_id": row.get("task_id"),
                "index": row.get("index"),
                "start": row.get("start"),
                "end": row.get("end"),
                "frames": [
                    {
                        "frame_id": frame.get("frame_id"),
                        "path": frame.get("path"),
                        "sha256": frame.get("sha256"),
                    }
                    for frame in row.get("frames") or []
                    if isinstance(frame, dict)
                ],
            }
            for row in items
        ],
        **({"viewport_context": viewport_context} if viewport_context else {}),
    }


def _register_prepare_run(root: Path, pack: dict[str, Any]) -> dict[str, Any]:
    return register_bundle_run(
        root,
        run_type="codex_vision_session_prepare",
        run_id=str(pack["session_id"]),
        status="needs_input" if pack.get("status") == _ready_status(pack) else "needs_retry",
        title="Codex vision task preparation",
        summary=f"kind={pack.get('kind')}; tasks={len(pack.get('items') or [])}; model_requested={pack.get('model_requested')}",
        inputs={"bundle_dir": str(root), "selected_indexes": pack.get("selected_indexes") or []},
        parameters={"executor": str(pack.get("executor") or "codex_session"), "quality": pack.get("quality"), "unattended_execution": pack.get("executor") == "codex_cli"},
        artifacts=[
            {"key": "task_pack", "path": pack["task_pack_path"]},
            {"key": "prompt", "path": pack["prompt_path"]},
            {"key": "result_template", "path": pack["result_template_path"]},
        ],
        failed_items=list(pack.get("blockers") or []),
        retry_command=_prepare_command(root, str(pack.get("kind") or "semantic"), list(pack.get("selected_indexes") or [])),
        next_actions=(
            ["Continue the owning run-codex-vision queue to execute and import this pack."]
            if pack.get("executor") == "codex_cli" else
            [f"Open {pack['prompt_path']} and write the exact JSON result to {pack['result_template_path']}."]
        ),
        operator_boundary=dict(pack.get("operator_boundary") or {}),
        write=True,
    )


def _register_import_run(
    root: Path,
    pack: dict[str, Any],
    summary: dict[str, Any],
    run_audit: dict[str, Any],
    report_path: Path,
    raw_result_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    record = run_audit.get("record") if isinstance(run_audit.get("record"), dict) else {}
    return register_bundle_run(
        root,
        run_type="codex_vision_session_import",
        run_id=str(record.get("run_id") or pack.get("session_id") or "codex-session-import"),
        status="completed",
        title="Codex vision import",
        summary=f"kind={pack.get('kind')}; updated={summary.get('updated')}; model_actual={summary.get('model_actual')}",
        inputs={"task_pack_sha256": pack.get("task_pack_sha256"), "raw_result": str(raw_result_path)},
        parameters={
            "executor": str(pack.get("executor") or "codex_session"),
            "model_requested": summary.get("model_requested"),
            "model_actual": summary.get("model_actual"),
            "quality": summary.get("quality"),
        },
        artifacts=[
            {"key": "report", "path": str(report_path)},
            {"key": "raw_result", "path": str(raw_result_path)},
            {"key": "validation", "path": str(validation_path)},
            {"key": "vision_runs", "path": str(root / "vision-analysis-runs.jsonl")},
        ],
        failed_items=[],
        retry_command=_prepare_command(root, str(pack.get("kind") or "semantic"), []),
        next_actions=["Refresh coverage, then prepare the next bounded Codex vision batch."],
        operator_boundary={"interactive_only": pack.get("executor") != "codex_cli", "no_provider_call": True, "raw_sources_unchanged": True},
        write=True,
    )


def _public_status(pack: dict[str, Any], *, phase: str) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.codex_vision_session_status.v1",
        "phase": phase,
        "status": pack.get("status"),
        "session_id": pack.get("session_id"),
        "kind": pack.get("kind"),
        "executor": str(pack.get("executor") or "codex_session"),
        "model_requested": pack.get("model_requested"),
        "model_actual": "",
        "quality": pack.get("quality"),
        "evidence_levels": {"confirmed": 0, "inferred": 0, "unavailable": 0},
        "selected_indexes": pack.get("selected_indexes") or [],
        "artifacts": pack.get("artifacts") or {},
        "blockers": pack.get("blockers") or [],
        "updated_at": now_iso(),
    }


def _public_import_status(
    summary: dict[str, Any],
    validation: dict[str, Any],
    *,
    run_audit: dict[str, Any],
    restore_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = run_audit.get("record") if isinstance(run_audit.get("record"), dict) else {}
    return {
        "schema": "video_knowledge_pipeline.codex_vision_session_status.v1",
        "phase": "imported" if validation.get("ok") else "rejected",
        "status": summary.get("status"),
        "session_id": summary.get("session_id"),
        "kind": summary.get("kind"),
        "executor": str(summary.get("executor") or "codex_session"),
        "model_requested": summary.get("model_requested"),
        "model_actual": summary.get("model_actual"),
        "quality": summary.get("quality"),
        "updated_count": summary.get("updated"),
        "evidence_levels": record.get("evidence_levels") or {},
        "artifacts": summary.get("artifacts") or {},
        "validation": validation,
        "run_id": record.get("run_id") or "",
        "restore": restore_hint or {},
        "updated_at": now_iso(),
    }


def _render_prompt(pack: dict[str, Any]) -> str:
    lines = [
        "# VKP Codex Vision Task",
        "",
        f"- Session: `{pack.get('session_id')}`",
        f"- Kind: `{pack.get('kind')}`",
        f"- Requested model: `{pack.get('model_requested')}` / quality `{pack.get('quality')}`",
        f"- Task-pack SHA-256: `{pack.get('task_pack_sha256')}`",
        "",
        "## Question",
        "",
        str(pack.get("question") or ""),
        "",
        "## Contract",
        "",
        "Inspect every listed local Bundle frame. Do not inspect the source video or unrelated files.",
        "Return only the JSON object described by result.template.json.",
        "Use confirmed for directly visible facts, inferred only with inference_basis, and unavailable when every frame was inspected but the answer cannot be determined.",
        "Set model_actual truthfully; it may differ from model_requested.",
        "",
        "## Tasks",
        "",
    ]
    for task in pack.get("items") or []:
        lines.extend(
            [
                f"### {task.get('task_id')} — timeline {task.get('index')} ({task.get('start')}–{task.get('end')})",
                "",
                f"- Transcript context: {task.get('transcript')}",
                f"- OCR context: {task.get('visual_text')}",
            ]
        )
        for frame in task.get("frames") or []:
            lines.append(f"- `{frame.get('frame_id')}` `{frame.get('timestamp')}` `{frame.get('path')}` SHA-256 `{frame.get('sha256')}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_validation(value: dict[str, Any]) -> str:
    lines = [
        "# Codex Vision Import Validation",
        "",
        f"- Status: `{value.get('status')}`",
        f"- Session: `{value.get('session_id')}`",
        f"- Model requested / actual: `{value.get('model_requested')}` / `{value.get('model_actual')}`",
        f"- Updated: `{value.get('updated_count')}`",
        "",
        "## Issues",
        "",
    ]
    issues = value.get("issues") or []
    lines.extend([f"- `{row.get('key')}`: {row.get('message')}" for row in issues] or ["- None"])
    return "\n".join(lines).rstrip() + "\n"


def _render_import_report(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    return (
        "# Codex Vision Import Report\n\n"
        f"- Status: `{summary.get('status')}`\n"
        f"- Executor: `{summary.get('executor')}`\n"
        f"- Model requested: `{summary.get('model_requested')}`\n"
        f"- Model actual: `{summary.get('model_actual')}`\n"
        f"- Quality: `{summary.get('quality')}`\n"
        f"- Updated: `{summary.get('updated')}`\n"
        f"- Validation: `{validation.get('status')}`\n"
    )


def _question(kind: str) -> str:
    if kind == "temporal":
        return "逐帧观察同一时间片，描述可见事件顺序、状态变化、操作步骤和因果关系；不要用 ASR/OCR 推测未观察到的动作。"
    return "逐张观察画面，描述可见对象、动作、界面状态、布局、讲师关注点和需要注意的隐私线索；OCR 文字只作上下文，不替代视觉观察。"


def _count_local_ready(root: Path, timeline: list[dict[str, Any]], indexes: list[int], *, kind: str) -> int:
    count = 0
    wanted = set(indexes)
    for position, item in enumerate(timeline, start=1):
        index = _positive_int(item.get("index")) or position
        if wanted and index not in wanted:
            continue
        if _has_formal_result(item, kind=kind):
            continue
        paths = _semantic_local_paths(root, item) if kind == "semantic" else _temporal_local_paths(root, item)
        if (kind == "semantic" and paths) or (kind == "temporal" and MIN_TEMPORAL_FRAMES <= len(paths) <= MAX_TEMPORAL_FRAMES):
            count += 1
    return count


def _prepare_command(root: Path, kind: str, indexes: list[int]) -> str:
    suffix = f" --indexes {','.join(str(value) for value in indexes)}" if indexes else ""
    return f".\\scripts\\video-knowledge.ps1 prepare-codex-vision-session {quote_powershell_literal(root)} --kind {kind}{suffix}"


def _session_kind(value: str) -> str:
    lowered = str(value or "semantic").strip().lower()
    aliases = {"semantic_frame": "semantic", "single": "semantic", "temporal_sequence": "temporal", "sequence": "temporal"}
    lowered = aliases.get(lowered, lowered)
    if lowered not in {"semantic", "temporal"}:
        raise ValueError("kind must be semantic or temporal")
    return lowered


def _execution_mode(pack: dict[str, Any]) -> str:
    return "automated_multimodal" if pack.get("executor") == "codex_cli" else "interactive_multimodal"


def _ready_status(pack: dict[str, Any]) -> str:
    return "ready_for_automated_analysis" if pack.get("executor") == "codex_cli" else "ready_for_interactive_analysis"


def _executor_name(value: str) -> str:
    lowered = str(value or "codex_session").strip().lower()
    if lowered not in {"codex_session", "codex_native", "codex_cli"}:
        raise ValueError("executor must be codex_session, codex_native or codex_cli")
    return "codex_cli" if lowered == "codex_cli" else "codex_session"


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _read_optional_mapping(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_timeline(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"timeline.json not found: {path}")
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError("timeline.json must contain a JSON array")
    return [row for row in value if isinstance(row, dict)]


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    if len(text) > 160:
        raise ValueError(f"{label} is too long")
    return text


def _indexes(values: list[int] | None) -> list[int]:
    return sorted({int(value) for value in values or [] if int(value) > 0})


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _non_empty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and any(child not in (None, "", [], {}) for child in value.values())


def _has_formal_result(item: dict[str, Any], *, kind: str) -> bool:
    if kind == "temporal":
        keys = ("temporal_visual_understanding", "human_corrected_temporal_visual_understanding")
        review_key = "corrected_temporal_visual_understanding"
    else:
        keys = ("visual_understanding", "human_corrected_visual_understanding")
        review_key = "corrected_visual_understanding"
    if any(_non_empty_mapping(item.get(key)) for key in keys):
        return True
    review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    if _non_empty_mapping(review.get(review_key)):
        return True
    status = str(item.get("review_status") or review.get("status") or "").strip().lower()
    return status in {
        "accepted",
        "reviewed",
        "accepted_known_gap",
        "accepted_no_visual_content",
        "accepted_provider_blocked",
        "corrected_visual_understanding",
        "corrected_temporal_visual_understanding",
    }


def _relative_path(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root)).replace("\\", "/")


def _issue(key: str, message: str) -> dict[str, str]:
    return {"key": str(key), "message": str(message)}


def _unique_issues(values: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in values:
        key = (str(row.get("key") or ""), str(row.get("message") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append({"key": key[0], "message": key[1]})
    return out
