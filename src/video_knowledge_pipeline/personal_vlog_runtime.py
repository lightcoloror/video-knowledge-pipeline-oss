from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .file_hash import sha256_file
from .local_video_run import prepare_local_video_run
from .personal_vlog import (
    RUN_SCHEMA,
    PersonalVlogBoundaryError,
    PersonalVlogDriftError,
    build_asset_evidence,
    build_edit_recipes,
    build_event_plan,
    load_personal_vlog_project,
    prepare_personal_vlog_project,
    validate_project_sources,
)
from .personal_vlog_handoff import build_mvp_v4_candidate_handoff
from .personal_vlog_otio import export_recipe_otio, opentimelineio_status
from .personal_vlog_workbench import (
    build_personal_vlog_workbench,
    render_personal_vlog_workbench,
)
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic


# Analysis is shared by every output preset.  Presets may vary the number and
# duration of proposed clips, but they must not silently reduce source
# analysis coverage.
STAGES = ("inventory", "shared_analysis", "evidence", "event_plan", "recipes", "handoff_review")


def personal_vlog_doctor() -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    magick = shutil.which("magick")
    optional = {
        name: importlib.util.find_spec(module) is not None
        for name, module in (
            ("opentimelineio", "opentimelineio"),
            ("openclip", "open_clip"),
            ("clap", "msclap"),
            ("mediapipe", "mediapipe"),
            ("ruptures", "ruptures"),
        )
    }
    blockers = []
    if not ffprobe:
        blockers.append("ffprobe_missing_for_video_and_audio_probe")
    full_evidence_gaps = [
        name
        for name in ("openclip", "clap", "mediapipe")
        if not optional[name]
    ]
    return {
        "schema": "video_knowledge_pipeline.personal_vlog_doctor.v1",
        "profile": "personal-vlog-v1",
        "ready_to_prepare": True,
        "ready_for_shared_analysis": not blockers,
        "ready_for_light_run": True,
        "ready_for_candidate_recipe": not blockers,
        "ready_for_full_local_evidence": not blockers and not full_evidence_gaps,
        "commands": {
            "ffprobe": {"available": bool(ffprobe), "path": ffprobe or ""},
            "imagemagick": {"available": bool(magick), "path": magick or ""},
        },
        "optional_workers": optional,
        "otio": opentimelineio_status(),
        "blockers": blockers,
        "full_local_evidence_gaps": full_evidence_gaps,
        "degraded_capabilities": [name for name, available in optional.items() if not available],
        "policy": {
            "local_first": True,
            "remote_fallback": False,
            "automatic_model_download": False,
            "source_mutation": False,
            "automatic_render": False,
            "automatic_upload": False,
            "automatic_publish": False,
        },
    }


def personal_vlog_prepare(
    event_root: str | Path,
    project_dir: str | Path,
    *,
    max_depth: int = 2,
    max_items: int = 2_000,
) -> dict[str, Any]:
    return prepare_personal_vlog_project(
        event_root,
        project_dir,
        max_depth=max_depth,
        max_items=max_items,
        write=True,
    )


def personal_vlog_run(
    project_dir: str | Path,
    *,
    presets: str = "all",
    stop_after: str = "",
    fail_after_stage: str = "",
) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    project_path = root / "personal-vlog-project.json"
    project = load_personal_vlog_project(root)
    validate_project_sources(project)
    run_id = f"vlog-{uuid4().hex}"
    run_dir = root / "runs" / run_id
    state = {
        "schema": RUN_SCHEMA,
        "vlog_run_id": run_id,
        "project_id": project.get("project_id"),
        "project_identity_sha256": project.get("project_identity_sha256"),
        "project_manifest_sha256": sha256_file(project_path),
        "state_revision": 0,
        "state": "executing",
        "status": "running",
        "output_mode": "candidate_only",
        "review_required": True,
        "stage_order": list(STAGES),
        "completed_stages": [],
        "stage_receipts": [],
        "review_gaps": [],
        "failure_point": None,
        "automatic_retry_count": 0,
        "created_at": _now(),
        "updated_at": _now(),
    }
    _write_state(root, run_dir, state)
    return _advance(
        root,
        run_dir,
        project,
        state,
        presets=presets,
        stop_after=stop_after,
        fail_after_stage=fail_after_stage,
    )


def personal_vlog_resume(
    project_dir: str | Path,
    run_id: str,
    *,
    stop_after: str = "",
) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    if not run_id.startswith("vlog-") or Path(run_id).name != run_id:
        raise PersonalVlogBoundaryError("invalid personal Vlog run ID")
    run_dir = root / "runs" / run_id
    state = _read_state(run_dir)
    project_path = root / "personal-vlog-project.json"
    project = load_personal_vlog_project(root)
    if str(state.get("project_identity_sha256") or "") != str(
        project.get("project_identity_sha256") or ""
    ):
        raise PersonalVlogDriftError("resume_project_identity_drift")
    if str(state.get("project_manifest_sha256") or "") != sha256_file(project_path):
        raise PersonalVlogDriftError("resume_project_manifest_hash_drift")
    validate_project_sources(project)
    _validate_completed_stage_receipts(root, run_dir, state)
    if state.get("state") == "complete_draft":
        return _status_payload(root, run_dir, state)
    state["status"] = "running"
    state["state"] = "executing"
    state["failure_point"] = None
    state["updated_at"] = _now()
    _write_state(root, run_dir, state)
    presets = str(state.get("presets") or "all")
    return _advance(
        root,
        run_dir,
        project,
        state,
        presets=presets,
        stop_after=stop_after,
        fail_after_stage="",
    )


def personal_vlog_status(
    project_dir: str | Path,
    run_id: str = "",
) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    runs_root = root / "runs"
    if run_id:
        if Path(run_id).name != run_id:
            raise PersonalVlogBoundaryError("invalid personal Vlog run ID")
        run_dir = runs_root / run_id
    else:
        candidates = sorted(
            (path for path in runs_root.glob("vlog-*") if path.is_dir()),
            key=lambda path: path.name,
        ) if runs_root.exists() else []
        if not candidates:
            return {
                "schema": RUN_SCHEMA,
                "status": "not_started",
                "project_dir": str(root),
                "review_required": True,
            }
        run_dir = candidates[-1]
    return _status_payload(root, run_dir, _read_state(run_dir))


def _advance(
    root: Path,
    run_dir: Path,
    project: Mapping[str, Any],
    state: dict[str, Any],
    *,
    presets: str,
    stop_after: str,
    fail_after_stage: str,
) -> dict[str, Any]:
    state["presets"] = presets
    if stop_after and stop_after not in STAGES:
        raise ValueError(f"unknown stop_after stage: {stop_after}")
    try:
        for stage in STAGES:
            if stage in state["completed_stages"]:
                continue
            receipt = _run_stage(root, run_dir, project, state, stage, presets=presets)
            state["completed_stages"].append(stage)
            state["stage_receipts"].append(receipt)
            state["state_revision"] = int(state["state_revision"]) + 1
            state["review_gaps"] = _merge_gaps(state.get("review_gaps"), receipt.get("review_gaps"))
            state["updated_at"] = _now()
            _write_state(root, run_dir, state)
            if fail_after_stage == stage:
                raise RuntimeError(f"injected_failure_after_stage:{stage}")
            if stop_after == stage and stage != STAGES[-1]:
                state["state"] = "paused"
                state["status"] = "paused"
                state["recovery_command"] = (
                    f"personal-vlog-resume {root} --run-id {state['vlog_run_id']}"
                )
                state["updated_at"] = _now()
                _write_state(root, run_dir, state)
                return _status_payload(root, run_dir, state)
        state["state"] = "complete_draft"
        state["status"] = "complete_draft"
        state["review_required"] = True
        state["accepted"] = False
        state["completed_at"] = _now()
        state["updated_at"] = _now()
        _write_state(root, run_dir, state)
        return _status_payload(root, run_dir, state)
    except Exception as exc:
        state["state"] = "failed"
        state["status"] = "failed"
        state["failure_point"] = next(
            (stage for stage in STAGES if stage not in state["completed_stages"]),
            "after_completed_stage",
        )
        state["error_class"] = type(exc).__name__
        state["error"] = str(exc)[:1_000]
        state["automatic_retry_count"] = 0
        state["recovery_command"] = (
            f"personal-vlog-resume {root} --run-id {state['vlog_run_id']}"
        )
        state["updated_at"] = _now()
        _write_state(root, run_dir, state)
        return _status_payload(root, run_dir, state)


def _run_stage(
    root: Path,
    run_dir: Path,
    project: Mapping[str, Any],
    state: Mapping[str, Any],
    stage: str,
    *,
    presets: str,
) -> dict[str, Any]:
    before = _now()
    artifacts: list[Path] = []
    gaps: list[dict[str, Any]] = []
    if stage == "inventory":
        result = validate_project_sources(project)
        target = run_dir / "inventory-validation.json"
        _write_run_json(root, target, result, operation="personal_vlog_inventory")
        artifacts.append(target)
    elif stage == "shared_analysis":
        result = _run_shared_analysis(project, run_dir)
        target = run_dir / "shared-analysis.json"
        _write_run_json(root, target, result, operation="personal_vlog_shared_analysis")
        artifacts.append(target)
        gaps.extend(result.get("review_gaps") or [])
    elif stage == "evidence":
        imported = _probe_assets(project)
        imported = _merge_shared_analysis(imported, _read_object(run_dir / "shared-analysis.json"))
        result = build_asset_evidence(project, imported=imported)
        target = run_dir / "asset-evidence.json"
        _write_run_json(root, target, result, operation="personal_vlog_evidence")
        artifacts.append(target)
        gaps.extend(result.get("review_gaps") or [])
    elif stage == "event_plan":
        evidence = _read_object(run_dir / "asset-evidence.json")
        result = build_event_plan(project, evidence)
        target = run_dir / "event-plan.json"
        _write_run_json(root, target, result, operation="personal_vlog_event_plan")
        artifacts.append(target)
        gaps.extend(result.get("review_gaps") or [])
    elif stage == "recipes":
        evidence = _read_object(run_dir / "asset-evidence.json")
        events = _read_object(run_dir / "event-plan.json")
        recipes = build_edit_recipes(project, evidence, events, presets=presets)
        for name, recipe in recipes.items():
            target = run_dir / f"edit-recipe-{name}.json"
            _write_run_json(root, target, recipe, operation=f"personal_vlog_recipe_{name}")
            artifacts.append(target)
            gaps.extend(recipe.get("review_gaps") or [])
        result = {"preset_count": len(recipes), "presets": list(recipes)}
    elif stage == "handoff_review":
        recipe_paths = sorted(run_dir.glob("edit-recipe-*.json"))
        if not recipe_paths:
            raise RuntimeError("no personal Vlog recipes are available")
        recipes = {
            str(_read_object(path).get("preset") or path.stem): _read_object(path)
            for path in recipe_paths
        }
        handoffs = []
        otio_receipts = []
        for recipe_path in recipe_paths:
            recipe = _read_object(recipe_path)
            preset = str(recipe.get("preset") or "unknown")
            handoff = build_mvp_v4_candidate_handoff(project, recipe)
            handoff_path = run_dir / f"mvp-v4-handoff-{preset}.json"
            _write_run_json(root, handoff_path, handoff, operation=f"personal_vlog_handoff_{preset}")
            artifacts.append(handoff_path)
            handoffs.append(handoff)
            otio = export_recipe_otio(recipe, run_dir / f"edit-recipe-{preset}.otio")
            otio_receipts.append({key: value for key, value in otio.items() if key != "timeline"})
            if otio.get("changed") and otio.get("output_path"):
                artifacts.append(Path(str(otio["output_path"])))
            if otio.get("review_gap"):
                gaps.append({"reason": otio["review_gap"], "blocking": False})
        packet = _review_packet(project, state, handoffs, otio_receipts, gaps)
        packet_path = run_dir / "review-packet.json"
        markdown_path = run_dir / "review-packet.md"
        evidence = _read_object(run_dir / "asset-evidence.json")
        events = _read_object(run_dir / "event-plan.json")
        workbench = build_personal_vlog_workbench(
            project,
            evidence,
            events,
            recipes,
            run_id=str(state.get("vlog_run_id") or ""),
        )
        workbench_json = run_dir / "personal-vlog-workbench.json"
        workbench_html = run_dir / "personal-vlog-workbench.html"
        _write_run_json(root, packet_path, packet, operation="personal_vlog_review_packet")
        _write_run_json(root, workbench_json, workbench, operation="personal_vlog_workbench")
        with bundle_write_lock(root, operation="personal_vlog_review_markdown"):
            write_text_atomic(markdown_path, _render_review_markdown(packet))
            write_text_atomic(workbench_html, render_personal_vlog_workbench(workbench))
        artifacts.extend([packet_path, markdown_path, workbench_json, workbench_html])
        result = {
            "handoff_count": len(handoffs),
            "otio": otio_receipts,
            "review_packet": packet,
        }
    else:  # pragma: no cover - guarded by STAGES
        raise ValueError(f"unsupported personal Vlog stage: {stage}")
    output_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in artifacts
        if path.is_file()
    }
    receipt = {
        "schema": "video_knowledge_pipeline.personal_vlog_stage_receipt.v1",
        "vlog_run_id": state["vlog_run_id"],
        "state_revision": int(state["state_revision"]) + 1,
        "stage": stage,
        "status": "succeeded",
        "changed": True,
        "input_hashes": {
            "project_identity_sha256": project.get("project_identity_sha256"),
        },
        "output_hashes": output_hashes,
        "artifacts": list(output_hashes),
        "executor": "deterministic_personal_vlog_controller",
        "model_actual": "local_vkp_analysis" if stage == "shared_analysis" else ("none" if stage != "evidence" else "local_probes_plus_shared_analysis"),
        "evidence_level": "shared_local_analysis" if stage in {"shared_analysis", "evidence", "recipes"} else "contract",
        "review_gaps": gaps,
        "failure_point": None,
        "recovery_command": (
            f"personal-vlog-resume {root} --run-id {state['vlog_run_id']}"
        ),
        "automatic_retries": 0,
        "started_at": before,
        "completed_at": _now(),
        "result_summary": result,
    }
    receipt_path = run_dir / "receipts" / f"{int(state['state_revision']) + 1:02d}-{stage}.json"
    _write_run_json(root, receipt_path, receipt, operation=f"personal_vlog_receipt_{stage}")
    receipt["receipt_path"] = receipt_path.relative_to(root).as_posix()
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    return receipt


def _probe_assets(project: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    magick = shutil.which("magick")
    result: dict[str, dict[str, Any]] = {}
    for asset in project.get("assets") or []:
        if not isinstance(asset, Mapping):
            continue
        asset_id = str(asset.get("asset_id") or "")
        source = Path(str(asset.get("source_path") or ""))
        kind = str(asset.get("media_type") or "")
        if kind == "image":
            result[asset_id] = _probe_image(source, magick)
        elif kind in {"video", "audio"}:
            result[asset_id] = _probe_av(source, ffprobe, kind)
        else:
            result[asset_id] = {"probe_status": "unsupported"}
    return result


def _run_shared_analysis(project: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run the same local VKP evidence baseline for every valid video asset.

    This stage deliberately runs before recipe selection.  A highlight preset
    therefore receives the same ASR, timeline, sampled-frame and initial
    Bundle evidence as a long preset; only its proposed edit changes.
    """

    ffprobe = shutil.which("ffprobe")
    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for asset in project.get("assets") or []:
        if not isinstance(asset, Mapping) or str(asset.get("media_type") or "") != "video":
            continue
        asset_id = str(asset.get("asset_id") or "")
        source = Path(str(asset.get("source_path") or ""))
        probe = _probe_av(source, ffprobe, "video")
        row: dict[str, Any] = {
            "asset_id": asset_id,
            "source_sha256": str(asset.get("source_sha256") or ""),
            "source_path": str(source),
            "analysis_profile": "shared_local_vkp_v1",
            "source_mutation": False,
        }
        if probe.get("probe_status") != "ok":
            row.update({
                "status": "skipped_invalid_media",
                "child_bundle_status": "not_completed",
                "reason": str(probe.get("probe_error_class") or "ffprobe_failed"),
            })
            gaps.append({"asset_id": asset_id, "reason": "shared_vkp_analysis_skipped_invalid_media", "blocking": False})
            rows.append(row)
            continue
        analysis_dir = run_dir / "shared-analysis" / asset_id
        report = prepare_local_video_run(
            source,
            analysis_dir,
            title=str(asset.get("relative_path") or asset_id),
            copy_media=False,
            plan_asr=True,
            execute_asr=True,
            build_initial_bundle=True,
            max_frames=80,
        )
        initial_bundle = report.get("initial_bundle") if isinstance(report.get("initial_bundle"), Mapping) else {}
        completed = bool(report.get("ok")) and str(initial_bundle.get("status") or "") == "ok"
        transcript_path = str(report.get("transcript_path") or "")
        row.update({
            "status": "completed" if completed else "partial_failure",
            "child_bundle_status": "completed" if completed else "not_completed",
            "analysis_dir": str(analysis_dir),
            "run_report": str(report.get("json_path") or ""),
            "bundle_dir": str(initial_bundle.get("bundle_dir") or ""),
            "transcript_path": transcript_path,
            "duration_seconds": probe.get("duration_seconds"),
            "width": probe.get("width"),
            "height": probe.get("height"),
            "technical_quality": probe.get("technical_quality", 0.6),
            "asr_excerpt": _transcript_excerpt(transcript_path),
            "evidence_level": "local_shared_analysis" if completed else "review_gap",
        })
        if not completed:
            gaps.append({"asset_id": asset_id, "reason": "shared_vkp_analysis_incomplete", "blocking": False})
        rows.append(row)
    return {
        "schema": "video_knowledge_pipeline.personal_vlog_shared_analysis.v1",
        "analysis_profile": "shared_local_vkp_v1",
        "preset_independent": True,
        "assets": rows,
        "review_gaps": gaps,
        "policy": {
            "local_only": True,
            "source_mutation": False,
            "automatic_render": False,
            "automatic_publish": False,
        },
    }


def _merge_shared_analysis(
    imported: Mapping[str, Mapping[str, Any]], shared: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    merged = {str(asset_id): dict(value) for asset_id, value in imported.items()}
    for row in shared.get("assets") or []:
        if not isinstance(row, Mapping):
            continue
        asset_id = str(row.get("asset_id") or "")
        if asset_id:
            merged.setdefault(asset_id, {}).update({key: value for key, value in row.items() if key != "asset_id"})
    return merged


def _transcript_excerpt(transcript_path: str) -> str:
    if not transcript_path:
        return ""
    try:
        payload = read_json(Path(transcript_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return ""
    segments = payload.get("segments") if isinstance(payload, Mapping) else []
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if isinstance(segment, Mapping) and str(segment.get("text") or "").strip()
    )[:2_000]


def _probe_image(path: Path, executable: str | None) -> dict[str, Any]:
    if not executable:
        return {
            "probe_status": "unavailable",
            "probe_error_class": "imagemagick_missing",
            "technical_quality": 0.5,
        }
    completed = subprocess.run(
        [executable, "identify", "-ping", "-format", "%w %h", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "probe_status": "corrupt_or_unsupported",
            "probe_error_class": "imagemagick_identify_failed",
            "technical_quality": 0.0,
            "evidence_level": "review_gap",
        }
    try:
        width, height = [int(value) for value in completed.stdout.strip().split()[:2]]
    except (TypeError, ValueError):
        return {
            "probe_status": "invalid_output",
            "probe_error_class": "imagemagick_output_invalid",
            "technical_quality": 0.0,
            "evidence_level": "review_gap",
        }
    return {
        "probe_status": "ok",
        "width": width,
        "height": height,
        "technical_quality": 0.6,
    }


def _probe_av(path: Path, executable: str | None, kind: str) -> dict[str, Any]:
    if not executable:
        return {
            "probe_status": "unavailable",
            "probe_error_class": "ffprobe_missing",
            "technical_quality": 0.5,
            "child_bundle_status": "blocked_ffprobe_missing" if kind == "video" else None,
        }
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "probe_status": "corrupt_or_unsupported",
            "probe_error_class": "ffprobe_failed",
            "technical_quality": 0.0,
            "evidence_level": "review_gap",
            "child_bundle_status": "blocked_invalid_media" if kind == "video" else None,
        }
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
        video_stream = next(
            (row for row in streams if isinstance(row, Mapping) and row.get("codec_type") == "video"),
            {},
        )
        duration = float((payload.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {
            "probe_status": "invalid_output",
            "probe_error_class": "ffprobe_output_invalid",
            "technical_quality": 0.0,
            "evidence_level": "review_gap",
        }
    return {
        "probe_status": "ok",
        "duration_seconds": max(0.0, duration),
        "width": video_stream.get("width") if kind == "video" else None,
        "height": video_stream.get("height") if kind == "video" else None,
        "technical_quality": 0.6,
        "audio_value": 0.5 if kind in {"video", "audio"} else 0.0,
        "child_bundle_status": "planned_filmed_v1" if kind == "video" else None,
    }


def _review_packet(
    project: Mapping[str, Any],
    state: Mapping[str, Any],
    handoffs: list[dict[str, Any]],
    otio_receipts: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.personal_vlog_review_packet.v1",
        "project_id": project.get("project_id"),
        "project_identity_sha256": project.get("project_identity_sha256"),
        "vlog_run_id": state.get("vlog_run_id"),
        "status": "complete_draft",
        "output_mode": "candidate_only",
        "review_required": True,
        "accepted": False,
        "presets": [handoff.get("recipe") for handoff in handoffs],
        "handoff_hashes": [handoff.get("handoff_sha256") for handoff in handoffs],
        "otio": otio_receipts,
        "review_gaps": _merge_gaps([], gaps),
        "privacy": {
            "source_paths_exported": False,
            "precise_gps_exported": False,
            "appearance_vectors_exported": False,
        },
        "next_action": "human_review_before_any_render_or_publish",
    }


def _render_review_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# VKP 个人 Vlog 候选剪辑审核包",
        "",
        f"- 项目：`{packet.get('project_id')}`",
        f"- 运行：`{packet.get('vlog_run_id')}`",
        "- 状态：`complete_draft / review_required`",
        "- 自动渲染、上传、发布：否",
        "- 源路径、精确 GPS、外观向量导出：否",
        "",
        "## 三档候选",
        "",
        "| 预设 | 目标时长 | 画幅 | 配方 Hash |",
        "|---|---:|---:|---|",
    ]
    for row in packet.get("presets") or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| {row.get('preset')} | {row.get('target_seconds')} 秒 | "
                f"{row.get('aspect_ratio')} | `{row.get('recipe_sha256')}` |"
            )
    lines.extend(["", "## 待复核", ""])
    gaps = packet.get("review_gaps") or []
    if not gaps:
        lines.append("- 无机器缺口；仍需人工选择与接受。")
    else:
        for gap in gaps:
            if isinstance(gap, Mapping):
                lines.append(f"- {gap.get('reason') or '未分类缺口'}")
    lines.extend(["", "> 本审核包只包含候选配方，不授权剪辑执行或发布。", ""])
    return "\n".join(lines)


def _write_state(root: Path, run_dir: Path, state: Mapping[str, Any]) -> None:
    _write_run_json(root, run_dir / "run.json", state, operation="personal_vlog_run_state")


def _write_run_json(root: Path, target: Path, payload: Mapping[str, Any], *, operation: str) -> None:
    _require_within(root, target)
    with bundle_write_lock(root, operation=operation):
        write_json(target, dict(payload))


def _read_state(run_dir: Path) -> dict[str, Any]:
    state = _read_object(run_dir / "run.json")
    if state.get("schema") != RUN_SCHEMA:
        raise PersonalVlogBoundaryError("invalid personal Vlog run state")
    return state


def _read_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _validate_completed_stage_receipts(
    root: Path,
    run_dir: Path,
    state: Mapping[str, Any],
) -> None:
    """Reject resume when a completed stage, receipt, or artifact drifted."""

    completed = [str(value) for value in state.get("completed_stages") or []]
    if completed != list(STAGES[: len(completed)]):
        raise PersonalVlogDriftError("completed_stage_order_drift")
    receipts = [row for row in state.get("stage_receipts") or [] if isinstance(row, Mapping)]
    if len(receipts) != len(completed):
        raise PersonalVlogDriftError("completed_stage_receipt_count_drift")
    for expected_stage, receipt in zip(completed, receipts, strict=True):
        if (
            str(receipt.get("vlog_run_id") or "")
            != str(state.get("vlog_run_id") or "")
            or str(receipt.get("stage") or "") != expected_stage
        ):
            raise PersonalVlogDriftError(
                f"completed_stage_receipt_identity_drift:{expected_stage}"
            )
        receipt_relative = str(receipt.get("receipt_path") or "")
        if not receipt_relative:
            raise PersonalVlogDriftError(
                f"completed_stage_receipt_path_missing:{expected_stage}"
            )
        receipt_path = (root / receipt_relative).resolve()
        _require_within(root, receipt_path)
        if not receipt_path.is_file() or sha256_file(receipt_path) != str(
            receipt.get("receipt_sha256") or ""
        ):
            raise PersonalVlogDriftError(
                f"completed_stage_receipt_hash_drift:{expected_stage}"
            )
        output_hashes = receipt.get("output_hashes")
        if not isinstance(output_hashes, Mapping) or not output_hashes:
            raise PersonalVlogDriftError(
                f"completed_stage_artifact_index_missing:{expected_stage}"
            )
        for relative, expected_hash in output_hashes.items():
            artifact = (root / str(relative)).resolve()
            _require_within(root, artifact)
            if not artifact.is_file() or sha256_file(artifact) != str(expected_hash or ""):
                raise PersonalVlogDriftError(
                    f"completed_stage_artifact_hash_drift:{expected_stage}:{relative}"
                )


def _status_payload(root: Path, run_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "vlog_run_id": state.get("vlog_run_id"),
        "status": state.get("status"),
        "state": state.get("state"),
        "state_revision": state.get("state_revision"),
        "completed_stages": list(state.get("completed_stages") or []),
        "next_stage": next(
            (stage for stage in STAGES if stage not in (state.get("completed_stages") or [])),
            None,
        ),
        "review_required": True,
        "accepted": bool(state.get("accepted", False)),
        "review_gaps": list(state.get("review_gaps") or []),
        "failure_point": state.get("failure_point"),
        "error_class": state.get("error_class"),
        "error": state.get("error"),
        "recovery_command": state.get("recovery_command"),
        "run_dir": str(run_dir),
        "run_state_sha256": sha256_file(run_dir / "run.json"),
        "automatic_retry_count": int(state.get("automatic_retry_count") or 0),
    }


def _merge_gaps(left: Any, right: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in (left, right):
        if not isinstance(value, list):
            continue
        for row in value:
            if not isinstance(row, Mapping):
                continue
            item = dict(row)
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                rows.append(item)
    return rows


def _require_within(root: Path, target: Path) -> None:
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PersonalVlogBoundaryError(f"run artifact outside project:{target}") from exc


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
