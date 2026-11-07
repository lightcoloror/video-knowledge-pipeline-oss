from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .codex_vision_session import import_codex_vision_results
from .knowledge_coverage import audit_knowledge_coverage
from .multimodal_frame_analyzer import vision_analysis_apply_restore, vision_analysis_restore_plan
from .panorama_analysis import plan_panorama_analysis
from .panorama_contract import SUPPORTED_PROJECTIONS, build_panorama_profile
from .panorama_preflight import panorama_doctor
from .panorama_projection import execute_panorama_projection, plan_panorama_projection
from .storage import bundle_write_lock, read_json, read_jsonl, write_json


def panorama_prepare(
    bundle_dir: str | Path,
    *,
    projection: str,
    indexes: list[int] | None = None,
    frame_count: int = 8,
    limit: int = 16,
    face_size: int = 512,
    input_fov: float | None = None,
    lens_order: str | None = None,
    run_id: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    source_path = _timeline_source_path(root)
    profile = build_panorama_profile(
        projection=projection,
        source_path=source_path,
        input_fov=input_fov,
        lens_order=lens_order,
    )
    plan = plan_panorama_projection(
        root,
        profile=profile,
        indexes=indexes,
        frame_count=frame_count,
        limit=limit,
        face_size=face_size,
        run_id=run_id,
    )
    plan_path = root / "runs" / f"panorama-{plan['run_id']}" / "projection-plan.json"
    result = {**plan, "plan_path": str(plan_path), "write_performed": bool(write)}
    if write:
        with bundle_write_lock(root, operation=f"panorama_prepare:{plan['run_id']}", timeout_seconds=1.0):
            write_json(plan_path, plan)
            manifest = read_json(root / "manifest.json")
            if not isinstance(manifest, dict):
                raise ValueError("manifest changed during panorama prepare")
            manifest["panorama_profile"] = profile
            manifest["panorama_projection_plan"] = {
                "run_id": plan["run_id"],
                "plan_sha256": plan["plan_sha256"],
                "path": plan_path.relative_to(root).as_posix(),
            }
            manifest["mcp_panorama_status_args"] = "mcp-panorama-status.args.json"
            write_json(root / "manifest.json", manifest)
            write_json(root / "mcp-panorama-status.args.json", {"bundle_dir": str(root), "run_id": plan["run_id"]})
    return result


def panorama_run(
    bundle_dir: str | Path,
    *,
    run_id: str,
    execute_projection: bool = False,
    interactive_session_available: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    plan_path = root / "runs" / f"panorama-{run_id}" / "projection-plan.json"
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise FileNotFoundError(f"panorama projection plan not found: {plan_path}")
    projection = (
        execute_panorama_projection(root, plan)
        if execute_projection
        else {
            "status": "planned",
            "run_id": run_id,
            "pack_path": str(root / "runs" / f"panorama-{run_id}" / "viewport-pack.json"),
            "remote_calls_made": 0,
        }
    )
    pack_path = Path(str(projection.get("pack_path") or ""))
    analysis: dict[str, Any] = {}
    if pack_path.is_file():
        analysis = plan_panorama_analysis(
            root,
            viewport_pack_json=pack_path,
            interactive_session_available=interactive_session_available,
            write=True,
        )
    return {
        "schema": "video_knowledge_pipeline.panorama_run.v1",
        "status": "ready_for_analysis" if analysis else str(projection.get("status") or "planned"),
        "run_id": run_id,
        "projection": projection,
        "analysis": analysis,
        "remote_calls_made": 0,
        "model_calls_made": 0,
        "next_action": _run_next_action(root, run_id, analysis, execute_projection=execute_projection),
    }


def panorama_import(
    bundle_dir: str | Path,
    *,
    run_id: str,
    input_json: str | Path,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    projection_root = root / "runs" / f"panorama-{run_id}"
    if not projection_root.is_dir():
        raise FileNotFoundError(f"panorama run not found: {run_id}")
    result = import_codex_vision_results(root, input_json=input_json, write=write)
    return {**result, "panorama_run_id": run_id}


def panorama_status(bundle_dir: str | Path, *, run_id: str = "") -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    selected_run = run_id or str(((manifest or {}).get("panorama_projection_plan") or {}).get("run_id") or "")
    run_root = root / "runs" / f"panorama-{selected_run}" if selected_run else root / "runs" / "panorama-unselected"
    plan = _read_mapping(run_root / "projection-plan.json")
    pack = _read_mapping(run_root / "viewport-pack.json")
    analysis = _read_mapping(root / "runs" / f"panorama-analysis-{selected_run}" / "analysis-plan.json")
    coverage = audit_knowledge_coverage(root, write=False).get("coverage") or {}
    panorama_coverage = coverage.get("panorama_coverage") if isinstance(coverage, dict) else {}
    return {
        "schema": "video_knowledge_pipeline.panorama_status.v1",
        "status": (
            "not_prepared"
            if not plan
            else "projection_pending"
            if not pack
            else "complete_draft"
            if (panorama_coverage or {}).get("status") == "complete"
            else "review_required"
        ),
        "run_id": selected_run,
        "projection": {
            "prepared": bool(plan),
            "executed": bool(pack),
            "selected_indexes": plan.get("selected_indexes") or [],
            "viewport_count": pack.get("viewport_count") or plan.get("viewport_count") or 0,
        },
        "analysis": {
            "prepared": bool(analysis),
            "semantic_task_count": analysis.get("semantic_task_count") or 0,
            "temporal_task_count": analysis.get("temporal_task_count") or 0,
        },
        "coverage": panorama_coverage or {},
        "review_gaps": (panorama_coverage or {}).get("review_gaps") or analysis.get("review_gaps") or [],
        "recovery_command": _status_recovery(root, selected_run, plan=plan, pack=pack, analysis=analysis),
        "artifacts": {
            "projection_plan": str(run_root / "projection-plan.json"),
            "viewport_pack": str(run_root / "viewport-pack.json"),
            "analysis_plan": str(root / "runs" / f"panorama-analysis-{selected_run}" / "analysis-plan.json"),
        },
    }


def panorama_restore(
    bundle_dir: str | Path,
    *,
    run_id: str,
    execute: bool = False,
    confirm_run_id: str = "",
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    resolved_run_id = _resolve_panorama_restore_run_id(root, run_id)
    plan = vision_analysis_restore_plan(root, run_id=resolved_run_id, write=True)
    if not execute:
        return {
            "status": "restore_plan_ready" if plan.get("status") in {"ready", "partial"} else "restore_not_ready",
            "run_id": run_id,
            "resolved_vision_run_id": resolved_run_id,
            "restore_plan": plan,
            "executed": False,
        }
    expected_confirmation = resolved_run_id
    applied = vision_analysis_apply_restore(
        root,
        plan_json=plan["json_path"],
        execute=True,
        confirm_run_id=confirm_run_id,
    )
    summary = applied.get("summary") if isinstance(applied.get("summary"), dict) else {}
    return {
        "status": str(summary.get("status") or "unknown"),
        "run_id": run_id,
        "resolved_vision_run_id": expected_confirmation,
        "confirmation_required": expected_confirmation,
        "restore_plan": plan,
        "apply": applied,
    }


def doctor_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panorama-doctor")
    parser.add_argument("media_or_bundle", nargs="?", default=None)
    _add_projection_options(parser, required=False)
    args = vars(parser.parse_args(argv))
    _print(panorama_doctor(**args))


def prepare_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panorama-prepare")
    parser.add_argument("bundle_dir")
    _add_projection_options(parser, required=True)
    parser.add_argument("--indexes", default="")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--face-size", type=int, default=512)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = vars(parser.parse_args(argv))
    args["indexes"] = _indexes(args.pop("indexes"))
    args["write"] = not bool(args.pop("no_write"))
    _print(panorama_prepare(**args))


def run_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panorama-run")
    parser.add_argument("bundle_dir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute-projection", action="store_true")
    _print(panorama_run(**vars(parser.parse_args(argv))))


def import_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panorama-import")
    parser.add_argument("bundle_dir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--no-write", action="store_true")
    args = vars(parser.parse_args(argv))
    args["write"] = not bool(args.pop("no_write"))
    _print(panorama_import(**args))


def status_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panorama-status")
    parser.add_argument("bundle_dir")
    parser.add_argument("--run-id", default="")
    _print(panorama_status(**vars(parser.parse_args(argv))))


def restore_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panorama-restore")
    parser.add_argument("bundle_dir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-run-id", default="")
    _print(panorama_restore(**vars(parser.parse_args(argv))))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="video-knowledge-panorama")
    parser.add_argument("command", choices=("doctor", "prepare", "run", "import", "status", "restore"))
    args, remainder = parser.parse_known_args(argv)
    handlers = {
        "doctor": doctor_main,
        "prepare": prepare_main,
        "run": run_main,
        "import": import_main,
        "status": status_main,
        "restore": restore_main,
    }
    handlers[args.command](remainder)


def _add_projection_options(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--projection", required=required, choices=sorted(SUPPORTED_PROJECTIONS))
    parser.add_argument("--input-fov", type=float, default=None)
    parser.add_argument("--lens-order", default=None)


def _indexes(value: str) -> list[int]:
    if not str(value).strip():
        return []
    try:
        return sorted({int(row.strip()) for row in str(value).split(",") if int(row.strip()) > 0})
    except ValueError as exc:
        raise ValueError("indexes must be a comma-separated list of positive integers") from exc


def _timeline_source_path(root: Path) -> Path:
    timeline = read_json(root / "timeline.json")
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must contain an array")
    values = {str(row.get("video_key") or "") for row in timeline if isinstance(row, dict) and str(row.get("video_key") or "")}
    if len(values) != 1:
        raise ValueError("panorama Bundle must reference exactly one source video")
    return Path(next(iter(values))).expanduser().resolve()


def _run_next_action(root: Path, run_id: str, analysis: dict[str, Any], *, execute_projection: bool) -> dict[str, Any]:
    if not execute_projection:
        return {
            "key": "execute_projection",
            "command": f'panorama-run "{root}" --run-id {run_id} --execute-projection',
        }
    first = next((row for row in analysis.get("tasks") or [] if row.get("task_type") == "panorama_six_view_semantic"), None)
    if isinstance(first, dict):
        return {
            "key": "prepare_panorama_session",
            "route": (first.get("route") or {}).get("route"),
            "timeline_index": first.get("timeline_index"),
            "sample_index": first.get("sample_index"),
        }
    return {"key": "review_gaps", "review_gaps": analysis.get("review_gaps") or []}


def _status_recovery(root: Path, run_id: str, *, plan: dict[str, Any], pack: dict[str, Any], analysis: dict[str, Any]) -> str:
    if not run_id:
        return f'panorama-prepare "{root}" --projection <type>'
    if plan and not pack:
        return f'panorama-run "{root}" --run-id {run_id} --execute-projection'
    if pack and not analysis:
        return f'panorama-run "{root}" --run-id {run_id}'
    return f'panorama-status "{root}" --run-id {run_id}'


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_panorama_restore_run_id(root: Path, requested: str) -> str:
    rows = [row for row in read_jsonl(root / "vision-analysis-runs.jsonl") if isinstance(row, dict)]
    if any(str(row.get("run_id") or "") == requested for row in rows):
        return requested
    projection_root = root / "runs" / f"panorama-{requested}"
    if not projection_root.is_dir():
        return requested
    pack_hashes: set[str] = set()
    sessions_root = root / "codex-vision-sessions"
    for task_pack in sessions_root.glob("*/task-pack.json") if sessions_root.is_dir() else []:
        pack = _read_mapping(task_pack)
        context = pack.get("viewport_context") if isinstance(pack.get("viewport_context"), dict) else {}
        projection_run = str(context.get("projection_run_id") or "")
        if not projection_run and str(context.get("viewport_pack_path") or ""):
            projection_run = requested if projection_root in Path(str(context["viewport_pack_path"])).parents else ""
        if projection_run == requested and str(pack.get("task_pack_sha256") or ""):
            pack_hashes.add(str(pack["task_pack_sha256"]))
    matches = [row for row in rows if str(row.get("task_pack_sha256") or "") in pack_hashes]
    return str(matches[-1].get("run_id") or requested) if matches else requested


def _print(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
