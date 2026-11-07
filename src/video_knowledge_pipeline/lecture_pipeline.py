from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .asr_environment import asr_environment_status
from .asr_runner import detect_asr_runners, plan_asr_run
from .asr_setup_plan import plan_asr_setup, render_asr_setup_plan_markdown
from .config import (
    DEFAULT_LOCAL_FRAME_BUDGET,
    DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
)
from .extractor_probe import detect_extractor_output
from .lecture_package import build_lecture_package, export_lecture_obsidian
from .models import EvidenceSegment, VideoMetadata, dataclass_to_dict, new_id, now_iso
from .orchestrator import (
    graph_candidates_for_video,
    init_project,
    render_video_evidence_card,
)
from .peepshow_adapter import import_peepshow_output
from .powershell import quote_powershell_argument as _quote_powershell_arg
from .powershell import quote_powershell_literal as _quote_powershell_string
from .storage import (
    append_jsonl,
    ensure_project_dirs,
    project_paths,
    read_json,
    write_json,
)
from .tool_research import recommended_trial_order
from .transcript import parse_transcript
from .vidclaude_adapter import import_vidclaude_cache
from .vidwise_adapter import import_vidwise_output
from .visual_tool_resolver import resolve_visual_extractor_command
from .webui_bridge import export_webui_bundle


def run_lecture_pipeline(
    root: str | Path,
    *,
    title: str,
    topic: str | None = None,
    vidclaude_cache: str | Path | None = None,
    peepshow_output: str | Path | None = None,
    vidwise_output: str | Path | None = None,
    media: str | Path | None = None,
    transcript: str | Path | None = None,
    webui_output_dir: str | Path | None = None,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    merge_window: float = 1.0,
    target: str = "bilinote",
    force_reimport: bool = False,
    allow_draft_obsidian_export: bool = False,
) -> dict[str, Any]:
    """Run the local lecture extraction glue pipeline over already-produced extractor outputs."""
    if transcript and not media:
        raise ValueError("media is required when transcript is provided")
    if not any([vidclaude_cache, peepshow_output, vidwise_output, transcript]):
        raise ValueError("at least one extractor output is required")

    root_path = Path(root)
    _ensure_project(root_path, title)
    import_topic = topic or title
    imports: dict[str, Any] = {}

    if vidclaude_cache:
        imports["vidclaude"] = _import_once(
            root_path,
            "vidclaude",
            vidclaude_cache,
            lambda: import_vidclaude_cache(root_path, vidclaude_cache, topic=import_topic),
            force=force_reimport,
        )
    if peepshow_output:
        imports["peepshow"] = _import_once(
            root_path,
            "peepshow",
            peepshow_output,
            lambda: import_peepshow_output(root_path, peepshow_output, topic=import_topic),
            force=force_reimport,
        )
    if vidwise_output:
        imports["vidwise"] = _import_once(
            root_path,
            "vidwise",
            vidwise_output,
            lambda: import_vidwise_output(root_path, vidwise_output, topic=import_topic),
            force=force_reimport,
        )
    if transcript:
        imports["transcript"] = _import_once(
            root_path,
            "transcript",
            transcript,
            lambda: import_transcript_source(root_path, media, transcript, topic=import_topic),
            force=force_reimport,
        )

    package = build_lecture_package(root_path, title=title, merge_window=merge_window)
    bundle = export_webui_bundle(root_path, output_dir=webui_output_dir, target=target)
    result: dict[str, Any] = {
        "project": str(root_path),
        "title": title,
        "topic": import_topic,
        "imports": imports,
        "package": package,
        "webui_bundle": bundle,
    }
    if vault:
        readiness = _bundle_readiness(bundle)
        result["review_readiness"] = readiness
        if not allow_draft_obsidian_export and readiness and not readiness.get("ready"):
            result["obsidian_export_blocked"] = {
                "reason": "draft_review_readiness_not_ready",
                "blockers": readiness.get("blockers", []),
                "next_action": readiness.get("next_action", {}),
                "hint": "Initial extraction exports are drafts. Pass allow_draft_obsidian_export=true only when you explicitly want draft Obsidian notes before review.",
            }
            return result
        result["obsidian_export"] = export_lecture_obsidian(root_path, vault, folder)
    return result


def run_detected_lecture_pipeline(
    root: str | Path,
    output_paths: list[str | Path],
    *,
    title: str,
    topic: str | None = None,
    webui_output_dir: str | Path | None = None,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    merge_window: float = 1.0,
    target: str = "bilinote",
    force_reimport: bool = False,
    allow_draft_obsidian_export: bool = False,
) -> dict[str, Any]:
    """Detect extractor output folders, then run the existing lecture pipeline."""
    probe = detect_extractor_output(output_paths, project=root, topic=topic or title)
    selected: dict[str, str] = {}
    skipped: list[dict[str, Any]] = []
    for candidate in probe.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("kind") or "")
        if not candidate.get("importable") or kind not in {"vidclaude", "peepshow", "vidwise"}:
            skipped.append(candidate)
            continue
        if kind in selected:
            skipped.append({**candidate, "skip_reason": f"{kind} already selected"})
            continue
        selected[kind] = str(candidate.get("path") or "")
    if not selected:
        raise ValueError("no importable extractor output detected")
    pipeline = run_lecture_pipeline(
        root,
        title=title,
        topic=topic,
        vidclaude_cache=selected.get("vidclaude"),
        peepshow_output=selected.get("peepshow"),
        vidwise_output=selected.get("vidwise"),
        webui_output_dir=webui_output_dir,
        vault=vault,
        folder=folder,
        merge_window=merge_window,
        target=target,
        force_reimport=force_reimport,
        allow_draft_obsidian_export=allow_draft_obsidian_export,
    )
    return {
        "project": str(Path(root)),
        "title": title,
        "probe": probe,
        "selected": selected,
        "skipped": skipped,
        "pipeline": pipeline,
    }


def _bundle_readiness(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(bundle.get("manifest_path") or ""))
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        return {}
    readiness = manifest.get("review_readiness")
    return readiness if isinstance(readiness, dict) else {}


def _import_once(root: Path, kind: str, source: str | Path, importer, *, force: bool = False) -> dict[str, Any]:
    paths = ensure_project_dirs(root)
    registry_path = paths["lecture_packages"] / "import-runs.json"
    registry = _read_import_registry(registry_path)
    key = _import_key(kind, source)
    if key in registry and not force:
        previous = registry[key]
        return {
            **previous,
            "skipped": True,
            "skip_reason": "already imported by lecture pipeline",
            "import_key": key,
            "registry_path": str(registry_path),
        }
    result = importer()
    record = {
        **result,
        "kind": kind,
        "source": _normalise_import_source(source),
        "imported_at": now_iso(),
        "skipped": False,
        "force_reimport": bool(force and key in registry),
    }
    registry[key] = record
    write_json(registry_path, registry)
    return {**record, "import_key": key, "registry_path": str(registry_path)}


def _read_import_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def lecture_import_status(root: str | Path) -> dict[str, Any]:
    """Return and write a human-readable status report for lecture import runs."""
    root_path = Path(root)
    paths = ensure_project_dirs(root_path)
    registry_path = paths["lecture_packages"] / "import-runs.json"
    markdown_path = paths["notes"] / "lecture-import-runs.md"
    registry = _read_import_registry(registry_path)
    imports = _registry_rows(registry)
    summary = {
        "import_count": len(imports),
        "forced_reimport_count": sum(1 for row in imports if row["force_reimport"]),
        "total_segments": sum(int(row.get("segment_count") or 0) for row in imports),
        "kinds": sorted({row["kind"] for row in imports if row["kind"]}),
    }
    markdown_path.write_text(_render_import_status_markdown(root_path, registry_path, summary, imports), encoding="utf-8")
    return {
        "project": str(root_path),
        "registry_path": str(registry_path),
        "markdown_path": str(markdown_path),
        "summary": summary,
        "imports": imports,
    }


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, record in sorted(registry.items()):
        if not isinstance(record, dict):
            continue
        rows.append(
            {
                "import_key": str(key),
                "kind": str(record.get("kind") or ""),
                "source": str(record.get("source") or ""),
                "video_id": str(record.get("video_id") or ""),
                "segment_count": int(record.get("segment_count") or 0),
                "imported_at": str(record.get("imported_at") or ""),
                "force_reimport": bool(record.get("force_reimport")),
                "segments_path": str(record.get("segments_path") or ""),
                "skipped_next_run": True,
                "skip_reason": "already imported by lecture pipeline",
            }
        )
    return rows


def _render_import_status_markdown(root: Path, registry_path: Path, summary: dict[str, Any], imports: list[dict[str, Any]]) -> str:
    lines = [
        "# Lecture Import Runs",
        "",
        f"- Project: `{root}`",
        f"- Registry: `{registry_path}`",
        f"- Import count: {summary['import_count']}",
        f"- Total segments: {summary['total_segments']}",
        f"- Forced reimports: {summary['forced_reimport_count']}",
        "",
        "## Next Run Behavior",
        "",
        "By default, every source listed below will be skipped on the next pipeline run. Use `--force-reimport` only after regenerating extractor output or correcting an import source.",
        "",
        "## Imports",
        "",
    ]
    if not imports:
        lines.append("No lecture extractor sources have been imported yet.")
        return "\n".join(lines)
    lines.extend(
        [
            "| Kind | Segments | Forced | Imported at | Source |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in imports:
        forced = "yes" if row["force_reimport"] else "no"
        lines.append(f"| {row['kind']} | {row['segment_count']} | {forced} | {row['imported_at']} | `{row['source']}` |")
    return "\n".join(lines)


def _import_key(kind: str, source: str | Path) -> str:
    return f"{kind}:{_normalise_import_source(source)}"


def _normalise_import_source(source: str | Path) -> str:
    return str(Path(source).expanduser().resolve())


def run_ready_lecture_pipeline(
    plan_path: str | Path,
    *,
    transcript: str | Path | None = None,
    webui_output_dir: str | Path | None = None,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    merge_window: float = 1.0,
    target: str = "bilinote",
    force_reimport: bool = False,
    allow_draft_obsidian_export: bool = False,
) -> dict[str, Any]:
    """Run the lecture pipeline using whichever planned outputs are ready."""
    path = Path(plan_path)
    plan = read_json(path)
    if not isinstance(plan, dict):
        raise ValueError("lecture pipeline plan must be a JSON object")
    planned = plan.get("planned_outputs") if isinstance(plan.get("planned_outputs"), dict) else {}
    status = status_lecture_pipeline_plan(path, transcript=transcript)
    ready_extractors = [str(name) for name in status.get("ready_extractors") or []]
    transcript_path = str(status.get("normalized_transcript") or "") if (status.get("ready") or {}).get("asr_transcript") else ""
    if not ready_extractors and not transcript_path:
        raise ValueError("no planned extractor output is ready; run an extractor or normalize ASR first")

    result = run_lecture_pipeline(
        str(plan.get("project", "")),
        title=str(plan.get("title", "")),
        topic=str(plan.get("topic") or plan.get("title", "")),
        vidclaude_cache=planned.get("vidclaude_cache") if "vidclaude" in ready_extractors else None,
        peepshow_output=planned.get("peepshow_output") if "peepshow" in ready_extractors else None,
        vidwise_output=planned.get("vidwise_output") if "vidwise" in ready_extractors else None,
        media=str(plan.get("media_path", "")) if transcript_path else None,
        transcript=transcript_path or None,
        webui_output_dir=webui_output_dir or planned.get("webui_output_dir"),
        vault=vault,
        folder=folder,
        merge_window=merge_window,
        target=target,
        force_reimport=force_reimport,
        allow_draft_obsidian_export=allow_draft_obsidian_export,
    )
    return {
        "plan_path": str(path),
        "used_extractors": ready_extractors,
        "used_transcript": transcript_path,
        "status": status,
        "pipeline": result,
    }


def prepare_lecture_workspace(
    root: str | Path,
    media: str | Path,
    *,
    title: str,
    topic: str | None = None,
    output_root: str | Path | None = None,
    asr_preset: str = "funasr",
    language: str = "zh",
    model: str | None = None,
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    fps: float = 1.0,
    target: str = "bilinote",
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare a self-contained handoff workspace for one lecture video."""
    plan = plan_lecture_pipeline(
        root,
        media,
        title=title,
        topic=topic,
        output_root=output_root,
        asr_preset=asr_preset,
        language=language,
        model=model,
        max_frames=max_frames,
        fps=fps,
        target=target,
        source_provenance=source_provenance,
    )
    status = status_lecture_pipeline_plan(plan["plan_path"])
    output_dir = Path(str(plan["output_root"]))
    mcp_status_args = {"plan_json": plan["plan_path"]}
    mcp_health_args = {
        "project": plan["project"],
        "plan_json": plan["plan_path"],
        "webui_output_dir": plan["planned_outputs"]["webui_output_dir"],
    }
    mcp_next_args: dict[str, Any] = {
        "project": plan["project"],
        "plan_json": plan["plan_path"],
        "webui_output_dir": plan["planned_outputs"]["webui_output_dir"],
        "folder": folder,
        "target": target,
        "allow_draft_obsidian_export": False,
    }
    mcp_run_ready_args: dict[str, Any] = {
        "plan_json": plan["plan_path"],
        "webui_output_dir": plan["planned_outputs"]["webui_output_dir"],
        "folder": folder,
        "target": target,
        "allow_draft_obsidian_export": False,
    }
    mcp_recommended_route_args: dict[str, Any] = {
        "plan_json": plan["plan_path"],
        "rank": 1,
        "execute": False,
        "normalize": True,
    }
    mcp_recommended_route_status_args: dict[str, Any] = {
        "plan_json": plan["plan_path"],
        "rank": 1,
    }
    mcp_recommended_route_queue_args: dict[str, Any] = {
        "plan_json": plan["plan_path"],
        "rank": 1,
        "execute": False,
        "normalize": True,
        "max_steps": 4,
    }
    mcp_workspace_advance_args: dict[str, Any] = {
        "plan_json": plan["plan_path"],
        "execute": False,
        "run_queue": True,
        "import_ready": True,
        "rank": 1,
        "normalize": True,
        "max_steps": 4,
        "webui_output_dir": plan["planned_outputs"]["webui_output_dir"],
        "folder": folder,
        "target": target,
        "allow_draft_obsidian_export": False,
    }
    mcp_workspace_advance_log_args: dict[str, Any] = {
        "project": plan["project"],
    }
    mcp_apply_bilinote_patch_args: dict[str, Any] = {
        "execute": False,
        "backup": True,
    }
    mcp_asr_args: dict[str, Any] = {
        "plan_json": plan["plan_path"],
        "execute": False,
        "normalize": True,
    }
    mcp_extractor_args = {
        name: {"plan_json": plan["plan_path"], "extractor": name, "execute": False}
        for name in ("vidclaude", "peepshow", "vidwise")
    }
    if vault:
        mcp_next_args["vault"] = str(vault)
        mcp_run_ready_args["vault"] = str(vault)
        mcp_workspace_advance_args["vault"] = str(vault)

    status_args_path = output_dir / "mcp-status-lecture-pipeline.args.json"
    health_args_path = output_dir / "mcp-lecture-project-health.args.json"
    next_args_path = output_dir / "mcp-lecture-next-step.args.json"
    run_ready_args_path = output_dir / "mcp-run-ready-lecture-pipeline.args.json"
    recommended_route_args_path = output_dir / "mcp-run-recommended-route.args.json"
    recommended_route_status_args_path = output_dir / "mcp-recommended-route-status.args.json"
    recommended_route_queue_args_path = output_dir / "mcp-recommended-route-queue.args.json"
    workspace_advance_args_path = output_dir / "mcp-recommended-workspace-advance.args.json"
    workspace_advance_log_args_path = output_dir / "mcp-recommended-workspace-advance-log.args.json"
    apply_bilinote_patch_args_path = output_dir / "mcp-apply-bilinote-patch.args.json"
    asr_args_path = output_dir / "mcp-run-asr-plan.args.json"
    asr_env_path = output_dir / "asr-environment.json"
    asr_env_script_path = output_dir / "asr-environment.ps1"
    asr_env_status_args_path = output_dir / "mcp-asr-environment-status.args.json"
    asr_env_status_dir = output_dir / "asr-environment-status"
    asr_setup_plan_path = output_dir / "asr-setup-plan.json"
    asr_setup_plan_markdown_path = output_dir / "asr-setup-plan.md"
    asr_setup_args_path = output_dir / "mcp-plan-asr-setup.args.json"
    extractor_args_paths = {
        name: output_dir / f"mcp-run-extractor-{name}.args.json"
        for name in ("vidclaude", "peepshow", "vidwise")
    }
    handoff_path = output_dir / "lecture-workspace.md"
    dashboard_path = output_dir / "lecture-workspace.html"
    write_json(status_args_path, mcp_status_args)
    write_json(health_args_path, mcp_health_args)
    write_json(next_args_path, mcp_next_args)
    write_json(run_ready_args_path, mcp_run_ready_args)
    write_json(recommended_route_args_path, mcp_recommended_route_args)
    write_json(recommended_route_status_args_path, mcp_recommended_route_status_args)
    write_json(recommended_route_queue_args_path, mcp_recommended_route_queue_args)
    write_json(workspace_advance_args_path, mcp_workspace_advance_args)
    write_json(workspace_advance_log_args_path, mcp_workspace_advance_log_args)
    write_json(apply_bilinote_patch_args_path, mcp_apply_bilinote_patch_args)
    write_json(asr_args_path, mcp_asr_args)
    asr_environment = build_asr_environment_export(plan, script_path=asr_env_script_path, status_output_dir=asr_env_status_dir)
    write_json(asr_env_path, asr_environment)
    asr_env_script_path.write_text(asr_environment["powershell"], encoding="utf-8")
    asr_setup_plan = plan_asr_setup(output_dir=output_dir, write=False, preferred=asr_preset)
    write_json(asr_setup_plan_path, asr_setup_plan)
    asr_setup_plan_markdown_path.write_text(render_asr_setup_plan_markdown(asr_setup_plan), encoding="utf-8")
    write_json(
        asr_setup_args_path,
        {
            "venv_dir": asr_setup_plan.get("target_venv_dir", ""),
            "output_dir": str(output_dir),
            "write": True,
            "preferred": asr_preset,
        },
    )
    write_json(
        asr_env_status_args_path,
        {
            "output_dir": str(asr_env_status_dir),
            "write": True,
        },
    )
    for name, args_path in extractor_args_paths.items():
        write_json(args_path, mcp_extractor_args[name])
    handoff_path.write_text(
        render_lecture_workspace_handoff(
            plan,
            status,
            status_args_path=status_args_path,
            health_args_path=health_args_path,
            next_args_path=next_args_path,
            run_ready_args_path=run_ready_args_path,
            recommended_route_args_path=recommended_route_args_path,
            recommended_route_status_args_path=recommended_route_status_args_path,
            recommended_route_queue_args_path=recommended_route_queue_args_path,
            workspace_advance_args_path=workspace_advance_args_path,
            workspace_advance_log_args_path=workspace_advance_log_args_path,
            apply_bilinote_patch_args_path=apply_bilinote_patch_args_path,
            asr_args_path=asr_args_path,
            asr_env_path=asr_env_path,
            asr_env_script_path=asr_env_script_path,
            asr_env_status_args_path=asr_env_status_args_path,
            asr_setup_plan_path=asr_setup_plan_path,
            asr_setup_plan_markdown_path=asr_setup_plan_markdown_path,
            asr_setup_args_path=asr_setup_args_path,
            extractor_args_paths=extractor_args_paths,
        ),
        encoding="utf-8",
    )
    dashboard_path.write_text(
        render_lecture_workspace_dashboard_html(
            plan,
            status,
            status_args_path=status_args_path,
            health_args_path=health_args_path,
            next_args_path=next_args_path,
            run_ready_args_path=run_ready_args_path,
            recommended_route_args_path=recommended_route_args_path,
            recommended_route_status_args_path=recommended_route_status_args_path,
            recommended_route_queue_args_path=recommended_route_queue_args_path,
            workspace_advance_args_path=workspace_advance_args_path,
            workspace_advance_log_args_path=workspace_advance_log_args_path,
            apply_bilinote_patch_args_path=apply_bilinote_patch_args_path,
            asr_args_path=asr_args_path,
            asr_env_path=asr_env_path,
            asr_env_script_path=asr_env_script_path,
            asr_env_status_args_path=asr_env_status_args_path,
            asr_setup_plan_path=asr_setup_plan_path,
            asr_setup_plan_markdown_path=asr_setup_plan_markdown_path,
            asr_setup_args_path=asr_setup_args_path,
            extractor_args_paths=extractor_args_paths,
        ),
        encoding="utf-8",
    )
    return {
        "project": plan["project"],
        "title": plan["title"],
        "media_path": plan["media_path"],
        "source_provenance": plan.get("source_provenance", {}),
        "output_root": plan["output_root"],
        "plan_path": plan["plan_path"],
        "plan_markdown_path": plan["markdown_path"],
        "status_path": status["status_path"],
        "status_markdown_path": status["markdown_path"],
        "handoff_path": str(handoff_path),
        "dashboard_path": str(dashboard_path),
        "mcp_status_args_path": str(status_args_path),
        "mcp_health_args_path": str(health_args_path),
        "mcp_next_args_path": str(next_args_path),
        "mcp_run_ready_args_path": str(run_ready_args_path),
        "mcp_recommended_route_args_path": str(recommended_route_args_path),
        "mcp_recommended_route_status_args_path": str(recommended_route_status_args_path),
        "mcp_recommended_route_queue_args_path": str(recommended_route_queue_args_path),
        "mcp_recommended_workspace_advance_args_path": str(workspace_advance_args_path),
        "mcp_recommended_workspace_advance_log_args_path": str(workspace_advance_log_args_path),
        "mcp_apply_bilinote_patch_args_path": str(apply_bilinote_patch_args_path),
        "mcp_asr_args_path": str(asr_args_path),
        "mcp_asr_environment_status_args_path": str(asr_env_status_args_path),
        "mcp_asr_setup_args_path": str(asr_setup_args_path),
        "asr_env_path": str(asr_env_path),
        "asr_env_script_path": str(asr_env_script_path),
        "asr_setup_plan_path": str(asr_setup_plan_path),
        "asr_setup_plan_markdown_path": str(asr_setup_plan_markdown_path),
        "mcp_extractor_args_paths": {name: str(path) for name, path in extractor_args_paths.items()},
        "next_step": status["next_step"],
    }


def build_asr_environment_export(
    plan: dict[str, Any],
    *,
    script_path: str | Path | None = None,
    status_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build reusable environment variables for the ASR command selected in a plan."""
    asr_plan = plan.get("asr_plan") if isinstance(plan.get("asr_plan"), dict) else {}
    command = asr_plan.get("command") if isinstance(asr_plan.get("command"), list) else []
    command_path = str(command[0] if command else "").strip()
    provider = str(asr_plan.get("provider") or asr_plan.get("preset") or "").strip()
    env_command = _asr_env_command_name(provider)
    variables: dict[str, str] = {}
    notes: list[str] = []

    if command_path:
        variables[env_command] = command_path
        path = Path(command_path)
        if path.parent and (path.is_absolute() or "\\" in command_path or "/" in command_path):
            variables["LECTURE_ASR_BIN_DIR"] = str(path.parent)
    else:
        notes.append("No ASR command was found in the lecture plan.")

    if command_path and not variables.get("LECTURE_ASR_BIN_DIR"):
        notes.append("The planned ASR command is not a path-like executable, so only the command override is exported.")

    if command_path and not Path(command_path).exists() and variables.get("LECTURE_ASR_BIN_DIR"):
        notes.append("The planned ASR executable path does not currently exist; run the ASR environment helper or update env vars.")

    powershell = _render_asr_environment_script(variables, notes=notes)
    status = asr_environment_status(output_dir=status_output_dir, write=False) if status_output_dir else {}
    return {
        "schema": "lecture_asr_environment.v1",
        "project": str(plan.get("project", "")),
        "plan_path": str(plan.get("plan_path", "")),
        "script_path": str(script_path or ""),
        "provider": provider,
        "preset": str(asr_plan.get("preset") or ""),
        "available": bool(asr_plan.get("available")),
        "command": command,
        "variables": variables,
        "load_command": f". {_quote_powershell_string(str(script_path))}" if script_path else "",
        "notes": notes,
        "powershell": powershell,
        "status": status,
        "status_next_action": status.get("next_action", {}) if isinstance(status, dict) else {},
        "status_mcp_args_path": status.get("mcp_args_path", "") if isinstance(status, dict) else "",
    }


def _asr_env_command_name(provider: str) -> str:
    if provider in {"funasr", "sensevoice"}:
        return "LECTURE_FUNASR_COMMAND"
    if provider == "whisperx":
        return "LECTURE_WHISPERX_COMMAND"
    if provider == "faster-whisper":
        return "LECTURE_FASTER_WHISPER_COMMAND"
    return "LECTURE_ASR_COMMAND"


def _render_asr_environment_script(variables: dict[str, str], *, notes: list[str]) -> str:
    lines = [
        "# Generated by prepare-lecture-workspace.",
        "# Dot-source this file before running guarded ASR commands:",
        "# . .\\asr-environment.ps1",
        "",
    ]
    if variables:
        for name, value in sorted(variables.items()):
            lines.append(f"$env:{name} = {_quote_powershell_string(value)}")
    else:
        lines.append("# No ASR environment variables were exported for this plan.")
    if notes:
        lines.extend(["", "# Notes"])
        lines.extend(f"# - {note}" for note in notes)
    lines.append("")
    return "\n".join(lines)


def render_lecture_workspace_handoff(
    plan: dict[str, Any],
    status: dict[str, Any],
    *,
    status_args_path: str | Path,
    health_args_path: str | Path,
    next_args_path: str | Path,
    run_ready_args_path: str | Path,
    recommended_route_args_path: str | Path,
    recommended_route_status_args_path: str | Path,
    recommended_route_queue_args_path: str | Path,
    workspace_advance_args_path: str | Path,
    workspace_advance_log_args_path: str | Path,
    apply_bilinote_patch_args_path: str | Path,
    asr_args_path: str | Path,
    asr_env_path: str | Path,
    asr_env_script_path: str | Path,
    asr_env_status_args_path: str | Path,
    asr_setup_plan_path: str | Path,
    asr_setup_plan_markdown_path: str | Path,
    asr_setup_args_path: str | Path,
    extractor_args_paths: dict[str, Path],
) -> str:
    """Render a compact operator handoff for a prepared lecture workspace."""
    commands = plan.get("commands", {}) if isinstance(plan.get("commands"), dict) else {}
    planned = plan.get("planned_outputs", {}) if isinstance(plan.get("planned_outputs"), dict) else {}
    routes = plan.get("recommended_routes") if isinstance(plan.get("recommended_routes"), list) else []
    lines = [
        f"# Lecture Workspace: {plan.get('title', 'Untitled')}",
        "",
        "## Inputs",
        "",
        f"- Project: `{plan.get('project', '')}`",
        f"- Media: `{plan.get('media_path', '')}`",
        f"- Output root: `{plan.get('output_root', '')}`",
        f"- Topic: {plan.get('topic', '')}",
        "",
        "## Main Files",
        "",
        f"- Plan JSON: `{plan.get('plan_path', '')}`",
        f"- Plan Markdown: `{plan.get('markdown_path', '')}`",
        f"- Status JSON: `{status.get('status_path', '')}`",
        f"- Status Markdown: `{status.get('markdown_path', '')}`",
        f"- MCP status args: `{status_args_path}`",
        f"- MCP health args: `{health_args_path}`",
        f"- MCP next-step args: `{next_args_path}`",
        f"- MCP run-ready args: `{run_ready_args_path}`",
        f"- MCP recommended-route args: `{recommended_route_args_path}`",
        f"- MCP recommended-route-status args: `{recommended_route_status_args_path}`",
        f"- MCP recommended-route-queue args: `{recommended_route_queue_args_path}`",
        f"- MCP recommended-workspace-advance args: `{workspace_advance_args_path}`",
        f"- MCP recommended-workspace-advance-log args: `{workspace_advance_log_args_path}`",
        f"- MCP apply-BiliNote-patch args: `{apply_bilinote_patch_args_path}`",
        f"- MCP ASR args: `{asr_args_path}`",
        f"- MCP ASR environment-status args: `{asr_env_status_args_path}`",
        f"- MCP ASR setup-plan args: `{asr_setup_args_path}`",
        f"- ASR environment JSON: `{asr_env_path}`",
        f"- ASR environment script: `{asr_env_script_path}`",
        f"- ASR setup-plan JSON: `{asr_setup_plan_path}`",
        f"- ASR setup-plan Markdown: `{asr_setup_plan_markdown_path}`",
        *[f"- MCP {name} args: `{path}`" for name, path in extractor_args_paths.items()],
        "",
        "## Planned Output Folders",
        "",
    ]
    for key, value in planned.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Recommended Routes", ""])
    if routes:
        for route in routes:
            if not isinstance(route, dict):
                continue
            marker = "recommended" if route.get("recommended") else f"rank {route.get('rank', '')}"
            route_status = "available" if route.get("available") else "missing"
            lines.append(f"- **{route.get('name', '')}** ({marker}, {route_status}): {route.get('reason', '')}")
    else:
        lines.append("- No route recommendation available.")

    lines.extend(
        [
            "",
            "## ASR Environment",
            "",
            "Load this before running guarded ASR if the planned ASR command points at a local tool environment:",
            "",
            "```powershell",
            f". `{asr_env_script_path}`",
            "```",
            "",
            f"- Environment JSON: `{asr_env_path}`",
            f"- MCP environment-status args: `{asr_env_status_args_path}`",
            f"- Setup plan JSON: `{asr_setup_plan_path}`",
            f"- Setup plan Markdown: `{asr_setup_plan_markdown_path}`",
            f"- MCP setup-plan args: `{asr_setup_args_path}`",
            "",
            "## Normal Flow",
            "",
            "1. Run one or more extractor commands below.",
            "2. Run status to confirm which outputs are ready.",
            "3. Run run-ready to import ready outputs and export the BiliNote/WebUI bundle.",
            "4. Import the WebUI bundle in BiliNote and review the timeline.",
            "",
        ]
    )
    _append_command(lines, "ASR", commands.get("asr"))
    _append_command(lines, "Guarded ASR run", _local_cli_command(["run-asr-plan", str(plan.get("plan_path", ""))]))
    _append_command(lines, "Normalize ASR", commands.get("normalize_asr"))
    _append_command(lines, "vidclaude", commands.get("vidclaude"))
    _append_command(lines, "Guarded vidclaude run", _local_cli_command(["run-extractor-plan", str(plan.get("plan_path", "")), "vidclaude"]))
    _append_command(lines, "peepshow", commands.get("peepshow"))
    _append_command(lines, "Guarded peepshow run", _local_cli_command(["run-extractor-plan", str(plan.get("plan_path", "")), "peepshow"]))
    _append_command(lines, "vidwise", commands.get("vidwise"))
    _append_command(lines, "Guarded vidwise run", _local_cli_command(["run-extractor-plan", str(plan.get("plan_path", "")), "vidwise"]))
    _append_command(lines, "Project health", _local_cli_command(["lecture-health", str(plan.get("project", "")), "--plan-json", str(plan.get("plan_path", ""))]))
    _append_command(lines, "Check status", _local_cli_command(["status-lecture-pipeline", str(plan.get("plan_path", ""))]))
    _append_command(lines, "Run ready outputs", _local_cli_command(["run-ready-lecture-pipeline", str(plan.get("plan_path", ""))]))

    lines.extend(
        [
            "",
            "## Agent Calls",
            "",
            "```powershell",
            f".\\scripts\\video-knowledge.ps1 mcp-call status_lecture_pipeline {status_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call lecture_project_health {health_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call lecture_next_step {next_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call run_ready_lecture_pipeline {run_ready_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call run_recommended_route {recommended_route_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call recommended_route_status {recommended_route_status_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call recommended_route_queue {recommended_route_queue_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call recommended_workspace_advance {workspace_advance_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call recommended_workspace_advance_log {workspace_advance_log_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call apply_bilinote_patch {apply_bilinote_patch_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call run_asr_plan {asr_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call asr_environment_status {asr_env_status_args_path}",
            f".\\scripts\\video-knowledge.ps1 mcp-call plan_asr_setup {asr_setup_args_path}",
            *[
                f".\\scripts\\video-knowledge.ps1 mcp-call run_extractor_plan {path}"
                for path in extractor_args_paths.values()
            ],
            "```",
            "",
            "## Current Status",
            "",
        ]
    )
    ready = status.get("ready") if isinstance(status.get("ready"), dict) else {}
    for name, is_ready in ready.items():
        lines.append(f"- `{name}`: {'ready' if is_ready else 'missing'}")
    lines.extend(["", f"Next step: {status.get('next_step', '')}", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_lecture_workspace_dashboard_html(
    plan: dict[str, Any],
    status: dict[str, Any],
    *,
    status_args_path: str | Path,
    health_args_path: str | Path,
    next_args_path: str | Path,
    run_ready_args_path: str | Path,
    recommended_route_args_path: str | Path,
    recommended_route_status_args_path: str | Path,
    recommended_route_queue_args_path: str | Path,
    workspace_advance_args_path: str | Path,
    workspace_advance_log_args_path: str | Path,
    apply_bilinote_patch_args_path: str | Path,
    asr_args_path: str | Path,
    asr_env_path: str | Path,
    asr_env_script_path: str | Path,
    asr_env_status_args_path: str | Path,
    asr_setup_plan_path: str | Path,
    asr_setup_plan_markdown_path: str | Path,
    asr_setup_args_path: str | Path,
    extractor_args_paths: dict[str, Path],
) -> str:
    """Render a static operator dashboard for the pre-extraction lecture workspace."""
    commands = plan.get("commands", {}) if isinstance(plan.get("commands"), dict) else {}
    planned = plan.get("planned_outputs", {}) if isinstance(plan.get("planned_outputs"), dict) else {}
    ready = status.get("ready") if isinstance(status.get("ready"), dict) else {}
    routes = plan.get("recommended_routes") if isinstance(plan.get("recommended_routes"), list) else []
    extractor_commands = {
        "ASR": commands.get("asr", ""),
        "Guarded ASR run": _local_cli_command(["run-asr-plan", str(plan.get("plan_path", ""))]),
        "Normalize ASR": commands.get("normalize_asr", ""),
        "vidclaude": commands.get("vidclaude", ""),
        "peepshow": commands.get("peepshow", ""),
        "vidwise": commands.get("vidwise", ""),
    }
    utility_commands = {
        "Load ASR environment": f". {asr_env_script_path}",
        "Project health": _local_cli_command(["lecture-health", str(plan.get("project", "")), "--plan-json", str(plan.get("plan_path", ""))]),
        "Check status": _local_cli_command(["status-lecture-pipeline", str(plan.get("plan_path", ""))]),
        "Run ready outputs": _local_cli_command(["run-ready-lecture-pipeline", str(plan.get("plan_path", ""))]),
    }
    mcp_commands = {
        "MCP status": f".\\scripts\\video-knowledge.ps1 mcp-call status_lecture_pipeline {status_args_path}",
        "MCP health": f".\\scripts\\video-knowledge.ps1 mcp-call lecture_project_health {health_args_path}",
        "MCP next step": f".\\scripts\\video-knowledge.ps1 mcp-call lecture_next_step {next_args_path}",
        "MCP run ready": f".\\scripts\\video-knowledge.ps1 mcp-call run_ready_lecture_pipeline {run_ready_args_path}",
        "MCP recommended route": f".\\scripts\\video-knowledge.ps1 mcp-call run_recommended_route {recommended_route_args_path}",
        "MCP recommended route status": f".\\scripts\\video-knowledge.ps1 mcp-call recommended_route_status {recommended_route_status_args_path}",
        "MCP recommended route queue": f".\\scripts\\video-knowledge.ps1 mcp-call recommended_route_queue {recommended_route_queue_args_path}",
        "MCP recommended workspace advance": f".\\scripts\\video-knowledge.ps1 mcp-call recommended_workspace_advance {workspace_advance_args_path}",
        "MCP recommended workspace advance log": f".\\scripts\\video-knowledge.ps1 mcp-call recommended_workspace_advance_log {workspace_advance_log_args_path}",
        "MCP apply BiliNote patch": f".\\scripts\\video-knowledge.ps1 mcp-call apply_bilinote_patch {apply_bilinote_patch_args_path}",
        "MCP guarded ASR": f".\\scripts\\video-knowledge.ps1 mcp-call run_asr_plan {asr_args_path}",
        "MCP ASR environment status": f".\\scripts\\video-knowledge.ps1 mcp-call asr_environment_status {asr_env_status_args_path}",
        "MCP ASR setup plan": f".\\scripts\\video-knowledge.ps1 mcp-call plan_asr_setup {asr_setup_args_path}",
        "ASR environment JSON": str(asr_env_path),
        "ASR setup-plan JSON": str(asr_setup_plan_path),
        "ASR setup-plan Markdown": str(asr_setup_plan_markdown_path),
        **{
            f"MCP guarded {name}": f".\\scripts\\video-knowledge.ps1 mcp-call run_extractor_plan {path}"
            for name, path in extractor_args_paths.items()
        },
    }
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  <title>{_html(plan.get('title', 'Lecture Workspace'))}</title>",
            "  <style>",
            _dashboard_css(),
            "  </style>",
            "</head>",
            "<body>",
            '  <main class="shell">',
            '    <section class="hero">',
            "      <div>",
            "        <p>Lecture Extract Workspace</p>",
            f"        <h1>{_html(plan.get('title', 'Untitled'))}</h1>",
            f"        <div class=\"meta\">{_html(plan.get('media_path', ''))}</div>",
            "      </div>",
            f"      <span class=\"badge\">{_html(status.get('next_step', ''))}</span>",
            "    </section>",
            _dashboard_section("Ready State", _status_cards(ready), "cards"),
            _dashboard_section("Recommended Routes", _route_cards(routes), "route-list"),
            _dashboard_section("Extractor Commands", _command_cards(extractor_commands), "grid"),
            _dashboard_section("Pipeline / Health", _command_cards(utility_commands), "grid"),
            _dashboard_section("Agent MCP Calls", _command_cards(mcp_commands), "grid"),
            _dashboard_section("Planned Outputs", _output_rows(planned), "table-wrap"),
            "  </main>",
            "  <script>",
            _dashboard_js(),
            "  </script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _dashboard_section(title: str, body: str, class_name: str) -> str:
    return "\n".join(
        [
            '    <section class="panel">',
            f"      <h2>{_html(title)}</h2>",
            f'      <div class="{class_name}">',
            body,
            "      </div>",
            "    </section>",
        ]
    )


def _status_cards(ready: dict[str, Any]) -> str:
    if not ready:
        return '        <div class="empty">No status data yet.</div>'
    rows = []
    for name, value in ready.items():
        state = "ready" if value else "missing"
        rows.append(f'        <div class="status {state}"><strong>{_html(name)}</strong><span>{state}</span></div>')
    return "\n".join(rows)


def _command_cards(commands: dict[str, Any]) -> str:
    rows = []
    for title, command in commands.items():
        command_text = str(command or "").strip()
        if not command_text:
            continue
        escaped_command = _html(command_text)
        rows.append(
            "\n".join(
                [
                    '        <article class="command-card">',
                    f"          <h3>{_html(title)}</h3>",
                    f"          <pre>{escaped_command}</pre>",
                    f'          <button type="button" data-copy="{escaped_command}">Copy command</button>',
                    "        </article>",
                ]
            )
        )
    return "\n".join(rows) if rows else '        <div class="empty">No commands available.</div>'


def _route_cards(routes: list[Any]) -> str:
    if not routes:
        return '        <div class="empty">No recommended routes available.</div>'
    rows = []
    for route in routes:
        if not isinstance(route, dict):
            continue
        available = bool(route.get("available"))
        recommended = bool(route.get("recommended"))
        command_text = str(route.get("command") or "").strip()
        escaped_command = _html(command_text)
        rows.append(
            "\n".join(
                [
                    f'        <article class="route-card {"available" if available else "missing"}">',
                    "          <div>",
                    f"            <h3>#{_html(route.get('rank', ''))} {_html(route.get('name', ''))}</h3>",
                    f"            <p>{_html(route.get('reason', ''))}</p>",
                    "          </div>",
                    f'          <span class="route-state">{"recommended" if recommended else ("available" if available else "missing")}</span>',
                    f"          <pre>{escaped_command}</pre>" if command_text else "",
                    f'          <button type="button" data-copy="{escaped_command}">Copy command</button>' if command_text else "",
                    "        </article>",
                ]
            )
        )
    return "\n".join(row for row in rows if row)


def _output_rows(planned: dict[str, Any]) -> str:
    if not planned:
        return '<div class="empty">No planned outputs.</div>'
    rows = ["<table><thead><tr><th>Name</th><th>Path</th></tr></thead><tbody>"]
    for key, value in planned.items():
        rows.append(f"<tr><td>{_html(key)}</td><td><code>{_html(value)}</code></td></tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _dashboard_css() -> str:
    return """
    :root { color-scheme: light; font-family: Inter, "Segoe UI", Arial, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f7f8; color: #172026; }
    .shell { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }
    .hero { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 20px 0 18px; border-bottom: 1px solid #d8e0e4; }
    .hero p { margin: 0 0 6px; color: #64727b; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    h1 { margin: 0; font-size: 28px; line-height: 1.2; }
    h2 { margin: 0 0 12px; font-size: 15px; }
    h3 { margin: 0 0 8px; font-size: 13px; }
    .meta { margin-top: 8px; color: #55636c; font-size: 13px; overflow-wrap: anywhere; }
    .badge { max-width: 360px; padding: 8px 10px; border: 1px solid #b9d6d6; background: #e9f7f6; color: #145555; border-radius: 6px; font-size: 13px; }
    .panel { margin-top: 18px; padding: 16px; border: 1px solid #d8e0e4; border-radius: 8px; background: #fff; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
    .route-list { display: grid; gap: 10px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
    .status { display: flex; justify-content: space-between; align-items: center; min-height: 44px; padding: 10px 12px; border-radius: 6px; border: 1px solid #d8e0e4; }
    .status.ready { border-color: #9bd5b5; background: #eefaf3; color: #14613a; }
    .status.missing { border-color: #e6c18b; background: #fff7e8; color: #7a4a0b; }
    .command-card { display: flex; min-height: 174px; flex-direction: column; border: 1px solid #d8e0e4; border-radius: 8px; padding: 12px; background: #fbfcfc; }
    .route-card { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px 12px; align-items: start; border: 1px solid #d8e0e4; border-radius: 8px; padding: 12px; background: #fbfcfc; }
    .route-card.available { border-color: #9bd5b5; background: #f4fbf7; }
    .route-card.missing { border-color: #e6c18b; background: #fffaf0; }
    .route-card pre, .route-card button { grid-column: 1 / -1; }
    .route-card p { margin: 0; color: #50616b; font-size: 12px; line-height: 1.5; }
    .route-state { border-radius: 999px; background: #e9f7f6; color: #145555; padding: 4px 8px; font-size: 12px; }
    pre { flex: 1; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.45; color: #23313a; }
    button { align-self: flex-start; margin-top: 12px; border: 1px solid #8bb9bd; background: #effafa; color: #13535a; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
    button:hover { background: #dff3f3; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #e5ebee; padding: 9px 8px; text-align: left; vertical-align: top; }
    code { overflow-wrap: anywhere; }
    .empty { color: #66747d; font-size: 13px; }
    @media (max-width: 720px) {
      .shell { width: min(100vw - 20px, 1180px); padding-top: 12px; }
      .hero { flex-direction: column; }
      h1 { font-size: 22px; }
      .badge { max-width: none; }
    }
    """.strip()


def _dashboard_js() -> str:
    return """
    document.querySelectorAll("[data-copy]").forEach((button) => {
      button.addEventListener("click", async () => {
        const text = button.getAttribute("data-copy") || "";
        try {
          await navigator.clipboard.writeText(text);
          const old = button.textContent;
          button.textContent = "Copied";
          setTimeout(() => { button.textContent = old; }, 1200);
        } catch {
          window.prompt("Copy command", text);
        }
      });
    });
    """.strip()


def _html(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)



def _ensure_project(root: Path, title: str) -> None:
    if not project_paths(root)["project"].exists():
        init_project(root, title)


def import_transcript_source(
    root: str | Path,
    media: str | Path,
    transcript: str | Path,
    *,
    topic: str,
) -> dict[str, Any]:
    """Import a normalized ASR transcript as full cue-level lecture evidence."""
    paths = ensure_project_dirs(root)
    media_path = Path(media)
    transcript_path = Path(transcript)
    cues = parse_transcript(transcript_path)
    if not cues:
        raise ValueError(f"transcript contains no cues: {transcript_path}")
    metadata = VideoMetadata(
        id=new_id("video_asr"),
        path=str(media_path),
        title=media_path.stem or transcript_path.stem,
        duration_seconds=max(cue.end for cue in cues),
        created_at=now_iso(),
    )
    segments = [
        EvidenceSegment(
            id=new_id("segment"),
            video_id=metadata.id,
            start=cue.start,
            end=cue.end,
            midpoint=(cue.start + cue.end) / 2,
            signals=["asr_transcript"],
            transcript_excerpt=cue.text,
            frame_paths=[],
            uncertainty="imported from normalized ASR transcript; verify speech recognition accuracy",
        )
        for cue in cues
    ]
    video_dir = paths["videos"] / metadata.id
    video_dir.mkdir(parents=True, exist_ok=True)
    write_json(video_dir / "metadata.json", dataclass_to_dict(metadata))
    write_json(video_dir / "segments.json", [dataclass_to_dict(segment) for segment in segments])
    copied_transcript = video_dir / transcript_path.name
    copied_transcript.write_text(transcript_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    card = render_video_evidence_card(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    card_path = paths["notes"] / f"{metadata.id}-transcript-evidence-card.md"
    card_path.write_text(card, encoding="utf-8")
    graph_rows = graph_candidates_for_video(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    append_jsonl(paths["graph"], graph_rows)
    return {
        "video_id": metadata.id,
        "imported_from": str(transcript_path),
        "media_path": str(media_path),
        "segment_count": len(segments),
        "card_path": str(card_path),
        "copied_transcript_path": str(copied_transcript),
        "segments_path": str(video_dir / "segments.json"),
        "graph_path": str(paths["graph"]),
    }


def plan_lecture_pipeline(
    root: str | Path,
    media: str | Path,
    *,
    title: str,
    topic: str | None = None,
    output_root: str | Path | None = None,
    asr_preset: str = "funasr",
    language: str = "zh",
    model: str | None = None,
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    fps: float = 1.0,
    target: str = "bilinote",
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a dry-run plan for running external extractors before the glue pipeline."""
    media_path = Path(media)
    if not media_path.exists():
        raise FileNotFoundError(f"media not found: {media_path}")

    root_path = Path(root)
    _ensure_project(root_path, title)
    paths = ensure_project_dirs(root_path)
    project_root = _project_root()
    output_dir = Path(output_root) if output_root else paths["lecture_packages"] / "extractor-runs" / _safe_stem(media_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "lecture-pipeline-plan.json"
    markdown_path = output_dir / "lecture-pipeline-plan.md"

    planned_outputs = {
        "vidclaude_cache": str(output_dir / "vidclaude-cache"),
        "peepshow_output": str(output_dir / "peepshow-out"),
        "vidwise_output": str(output_dir / "vidwise-out"),
        "webui_output_dir": str(output_dir / "webui-bundle"),
    }
    asr_plan = plan_asr_run(root_path, media_path, preset=asr_preset, language=language, model=model)
    run_pipeline_commands = {
        "vidclaude_only": _pipeline_command(
            root_path,
            title,
            topic or title,
            planned_outputs,
            target=target,
            extractors=["vidclaude"],
        ),
        "peepshow_only": _pipeline_command(
            root_path,
            title,
            topic or title,
            planned_outputs,
            target=target,
            extractors=["peepshow"],
        ),
        "vidwise_only": _pipeline_command(
            root_path,
            title,
            topic or title,
            planned_outputs,
            target=target,
            extractors=["vidwise"],
        ),
        "asr_transcript_only_template": _pipeline_command(
            root_path,
            title,
            topic or title,
            planned_outputs,
            target=target,
            extractors=["transcript"],
            media=str(media_path),
            transcript="<normalized-transcript.json>",
        ),
        "all_extractors": _pipeline_command(
            root_path,
            title,
            topic or title,
            planned_outputs,
            target=target,
            extractors=["vidclaude", "peepshow", "vidwise"],
        ),
    }
    extractor_commands = _extractor_command_metadata(project_root, media_path, planned_outputs, max_frames=max_frames, fps=fps)
    commands = {
        "asr": asr_plan["powershell"],
        "normalize_asr": _local_cli_command(["normalize-asr", str(root_path), asr_plan["expected_output_json"], "--provider", asr_plan["provider"], "--title", media_path.stem]),
        "vidclaude": str(extractor_commands["vidclaude"]["command"]),
        "peepshow": str(extractor_commands["peepshow"]["command"]),
        "vidwise": str(extractor_commands["vidwise"]["command"]),
        "run_pipeline": run_pipeline_commands["all_extractors"],
    }
    preflight = {
        "asr": detect_asr_runners(),
        "video_tools": _video_tool_status(),
    }
    recommended_routes = _recommended_execution_routes(preflight, commands, plan_path)
    plan = {
        "schema": "lecture_pipeline_plan.v1",
        "project": str(root_path),
        "title": title,
        "topic": topic or title,
        "media_path": str(media_path),
        "media_exists": media_path.exists(),
        "source_provenance": source_provenance or {},
        "output_root": str(output_dir),
        "planned_outputs": planned_outputs,
        "preflight": preflight,
        "recommended_routes": recommended_routes,
        "asr_plan": asr_plan,
        "extractor_commands": extractor_commands,
        "commands": commands,
        "run_pipeline_commands": run_pipeline_commands,
        "notes": [
            "This plan does not execute external extractors.",
            "Run one or more extractor commands, then use the matching command under run_pipeline_commands.",
            "After normalize_asr, replace <normalized-transcript.json> in asr_transcript_only_template with the returned json_path.",
            "Use run-asr-plan for guarded ASR execution plus automatic normalization when the selected local runner is available.",
            "Use run-extractor-plan for guarded vidclaude/peepshow/vidwise execution and readiness logging.",
            "Use run_pipeline_commands.all_extractors only after all planned extractor outputs exist.",
            "The final pipeline imports existing output folders and writes the BiliNote/WebUI bundle.",
        ],
    }
    write_json(plan_path, plan)
    markdown_path.write_text(render_lecture_pipeline_plan_markdown(plan), encoding="utf-8")
    return {**plan, "plan_path": str(plan_path), "markdown_path": str(markdown_path)}


def render_lecture_pipeline_plan_markdown(plan: dict[str, Any]) -> str:
    """Render a human-readable execution checklist for a lecture pipeline plan."""
    commands = plan.get("commands", {})
    pipeline_commands = plan.get("run_pipeline_commands", {})
    preflight = plan.get("preflight", {})
    video_tools = preflight.get("video_tools") or []
    asr_tools = (preflight.get("asr") or {}).get("tools") or []
    lines = [
        f"# Lecture Pipeline Plan: {plan.get('title', 'Untitled')}",
        "",
        "## Inputs",
        "",
        f"- Project: `{plan.get('project', '')}`",
        f"- Media: `{plan.get('media_path', '')}`",
        f"- Output root: `{plan.get('output_root', '')}`",
        f"- Topic: {plan.get('topic', '')}",
        "",
        "## Preflight",
        "",
        f"- ffmpeg: `{(preflight.get('asr') or {}).get('ffmpeg', '') or 'missing'}`",
        "",
        "### ASR",
        "",
    ]
    source = plan.get("source_provenance") if isinstance(plan.get("source_provenance"), dict) else {}
    if source:
        lines.extend(
            [
                "## Source Provenance",
                "",
                f"- URL: `{source.get('url', '')}`",
                f"- Download status: `{source.get('status', '')}`",
                f"- Local media: `{source.get('local_media_path', '')}`",
                f"- Manifest: `{source.get('manifest_path', '')}`",
                f"- Report: `{source.get('report_path', '')}`",
                f"- Provenance JSON: `{source.get('provenance_json_path', '')}`",
                "",
            ]
        )
    if asr_tools:
        for tool in asr_tools:
            status = "available" if tool.get("available") else "missing"
            lines.append(f"- `{tool.get('name', '')}`: {status}")
    else:
        lines.append("- No ASR tool status available.")
    lines.extend(["", "### Video Tools", ""])
    if video_tools:
        for tool in video_tools:
            status = "available" if tool.get("installed") else "missing"
            lines.append(f"- `{tool.get('name', '')}`: {status}")
    else:
        lines.append("- No video tool status available.")
    routes = plan.get("recommended_routes") if isinstance(plan.get("recommended_routes"), list) else []
    lines.extend(["", "## Recommended Routes", ""])
    if routes:
        for route in routes:
            marker = "recommended" if route.get("recommended") else f"rank {route.get('rank', '')}"
            status = "available" if route.get("available") else "missing"
            lines.append(f"- **{route.get('name', '')}** ({marker}, {status}): {route.get('reason', '')}")
            command = str(route.get("command") or "").strip()
            if command:
                lines.extend(["", "```powershell", command, "```", ""])
    else:
        lines.append("- No route recommendation available.")
    lines.extend(
        [
            "",
            "## Suggested Execution",
            "",
            "1. Run ASR if you need stronger transcript coverage.",
            "2. Normalize ASR output.",
            "3. Run one or more visual extractors.",
            "4. Use the matching final pipeline command.",
            "5. Import the WebUI bundle into BiliNote for human review.",
            "",
        ]
    )
    _append_command(lines, "ASR", commands.get("asr"))
    _append_command(lines, "Normalize ASR", commands.get("normalize_asr"))
    _append_command(lines, "vidclaude", commands.get("vidclaude"))
    _append_command(lines, "peepshow", commands.get("peepshow"))
    _append_command(lines, "vidwise", commands.get("vidwise"))
    lines.extend(["", "## Final Pipeline Commands", ""])
    _append_command(lines, "ASR transcript only", pipeline_commands.get("asr_transcript_only_template"))
    _append_command(lines, "vidclaude only", pipeline_commands.get("vidclaude_only"))
    _append_command(lines, "peepshow only", pipeline_commands.get("peepshow_only"))
    _append_command(lines, "vidwise only", pipeline_commands.get("vidwise_only"))
    _append_command(lines, "all extractors", pipeline_commands.get("all_extractors"))
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in plan.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def status_lecture_pipeline_plan(plan_path: str | Path, *, transcript: str | Path | None = None) -> dict[str, Any]:
    """Inspect planned extractor outputs and recommend the next runnable pipeline command."""
    path = Path(plan_path)
    plan = read_json(path)
    if not isinstance(plan, dict):
        raise ValueError("lecture pipeline plan must be a JSON object")
    planned = plan.get("planned_outputs") if isinstance(plan.get("planned_outputs"), dict) else {}
    ready = {
        "vidclaude": _vidclaude_ready(Path(str(planned.get("vidclaude_cache", "")))),
        "peepshow": _peepshow_ready(Path(str(planned.get("peepshow_output", "")))),
        "vidwise": _vidwise_ready(Path(str(planned.get("vidwise_output", "")))),
    }
    normalized_candidates = _normalized_transcript_candidates(Path(str(plan.get("project", ""))))
    normalized_transcript = str(Path(transcript).expanduser().resolve()) if transcript else _find_normalized_transcript(Path(str(plan.get("project", ""))))
    transcript_ready = bool(normalized_transcript and Path(normalized_transcript).exists())
    ready_extractors = [name for name, is_ready in ready.items() if is_ready]
    recommended_command = ""
    if ready_extractors or transcript_ready:
        recommended_command = _pipeline_command(
            Path(str(plan.get("project", ""))),
            str(plan.get("title", "")),
            str(plan.get("topic") or plan.get("title", "")),
            {key: str(value) for key, value in planned.items()},
            target="bilinote",
            extractors=[*ready_extractors, *(["transcript"] if transcript_ready else [])],
            media=str(plan.get("media_path", "")) if transcript_ready else None,
            transcript=normalized_transcript if transcript_ready else None,
        )
    status = {
        "plan_path": str(path),
        "project": str(plan.get("project", "")),
        "title": str(plan.get("title", "")),
        "ready": {
            **ready,
            "asr_transcript": transcript_ready,
        },
        "ready_extractors": ready_extractors,
        "normalized_transcript": normalized_transcript,
        "normalized_transcript_candidates": normalized_candidates,
        "recommended_pipeline_command": recommended_command,
        "next_step": _status_next_step(ready_extractors, transcript_ready),
    }
    status_path = path.with_name("lecture-pipeline-status.json")
    markdown_path = path.with_name("lecture-pipeline-status.md")
    write_json(status_path, status)
    markdown_path.write_text(render_lecture_pipeline_status_markdown(status), encoding="utf-8")
    return {**status, "status_path": str(status_path), "markdown_path": str(markdown_path)}


def render_lecture_pipeline_status_markdown(status: dict[str, Any]) -> str:
    """Render a human-readable readiness report for a lecture pipeline plan."""
    ready = status.get("ready") if isinstance(status.get("ready"), dict) else {}
    ready_extractors = status.get("ready_extractors") or []
    lines = [
        f"# Lecture Pipeline Status: {status.get('title', 'Untitled')}",
        "",
        "## Readiness",
        "",
    ]
    if ready:
        for name, is_ready in ready.items():
            lines.append(f"- `{name}`: {'ready' if is_ready else 'missing'}")
    else:
        lines.append("- No readiness data available.")

    lines.extend(["", "## Ready Extractors", ""])
    if ready_extractors:
        for name in ready_extractors:
            lines.append(f"- `{name}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Next Step", "", str(status.get("next_step", "")), ""])
    normalized_transcript = str(status.get("normalized_transcript") or "")
    if normalized_transcript:
        lines.extend(["## Normalized Transcript", "", f"`{normalized_transcript}`", ""])
    candidates = status.get("normalized_transcript_candidates") if isinstance(status.get("normalized_transcript_candidates"), list) else []
    if candidates:
        lines.extend(["## Normalized Transcript Candidates", "", "| Selected | Modified | Path |", "|---|---:|---|"])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            selected = "yes" if str(candidate.get("path") or "") == normalized_transcript else ""
            lines.append(f"| {selected} | {candidate.get('mtime', 0)} | `{candidate.get('path', '')}` |")
        lines.append("")

    recommended_command = str(status.get("recommended_pipeline_command") or "")
    if recommended_command:
        lines.extend(["## Recommended Pipeline Command", "", "```powershell", recommended_command, "```", ""])
    else:
        lines.extend(["## Recommended Pipeline Command", "", "No runnable pipeline command yet.", ""])

    return "\n".join(lines).rstrip() + "\n"


def _vidclaude_ready(path: Path) -> bool:
    return bool(str(path)) and (path / "meta.json").exists()


def _peepshow_ready(path: Path) -> bool:
    return bool(str(path)) and (path / "manifest.json").exists()


def _vidwise_ready(path: Path) -> bool:
    return bool(str(path)) and (path / "video.mp4").exists()


def _find_normalized_transcript(project: Path) -> str:
    candidates = _normalized_transcript_candidates(project)
    return str(candidates[0]["path"]) if candidates else ""


def _normalized_transcript_candidates(project: Path) -> list[dict[str, Any]]:
    if not project.exists():
        return []
    rows = []
    for path in (project / "transcripts").glob("**/normalized-transcript.json"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "path": str(path.resolve()),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )
    rows.sort(key=lambda row: (-float(row.get("mtime") or 0), str(row.get("path") or "")))
    return rows


def _status_next_step(ready_extractors: list[str], transcript_ready: bool) -> str:
    if ready_extractors or transcript_ready:
        return "run recommended_pipeline_command"
    return "run at least one extractor command or normalize ASR output"


def _append_command(lines: list[str], title: str, command: str | None) -> None:
    if not command:
        return
    lines.extend([f"### {title}", "", "```powershell", command, "```", ""])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem).strip("_") or "media"


def _video_tool_status() -> list[dict[str, Any]]:
    rows = []
    for row in recommended_trial_order():
        if row["name"] in {"vidclaude", "peepshow", "vidwise"}:
            rows.append(
                {
                    "name": row["name"],
                    "installed": row["installed"],
                    "installed_paths": row["installed_paths"],
                    "reuse_role": row["reuse_role"],
                    "notes": row["notes"],
                }
            )
    return rows


def _recommended_execution_routes(
    preflight: dict[str, Any],
    commands: dict[str, str],
    plan_path: Path,
) -> list[dict[str, Any]]:
    video_tools = preflight.get("video_tools") if isinstance(preflight.get("video_tools"), list) else []
    asr_preflight = preflight.get("asr") if isinstance(preflight.get("asr"), dict) else {}
    asr_tools = asr_preflight.get("tools") if isinstance(asr_preflight.get("tools"), list) else []
    installed_video = {str(tool.get("name") or "").lower(): tool for tool in video_tools if tool.get("installed")}
    available_asr = [tool for tool in asr_tools if tool.get("available")]
    extractor_commands = {name: resolve_visual_extractor_command(name) for name in ("vidclaude", "peepshow", "vidwise")}
    routes: list[dict[str, Any]] = []

    for name, role, reason in [
        ("vidclaude", "visual_timeline_extractor", "优先保留 OCR、transcript、scene/visual timeline 和 evidence，适合主抽取"),
        ("peepshow", "fast_frame_ocr_report", "快速生成帧、OCR、manifest 和 HTML report，适合先跑覆盖检查"),
        ("vidwise", "lightweight_fallback_extractor", "低成本 fallback，用于补轻量素材或快速失败兜底"),
    ]:
        tool = installed_video.get(name)
        command_meta = extractor_commands.get(name) or {}
        routes.append(
            {
                "name": name,
                "rank": 0,
                "recommended": False,
                "role": role,
                "phase": "visual_extraction",
                "available": bool(tool),
                "reason": reason if tool else f"{name} 未发现本地可用路径",
                "command_name": name,
                "command": str(commands.get(name) or ""),
                "mcp_tool": "run_extractor_plan",
                "mcp_args": {"plan_json": str(plan_path), "extractor": name, "execute": False},
                "paths": tool.get("installed_paths", []) if isinstance(tool, dict) else [],
                "command_source": command_meta.get("command_source", ""),
                "command_path": command_meta.get("command_path", ""),
                "command_prefix": command_meta.get("command_prefix", []),
            }
        )

    routes.append(
        {
            "name": "asr",
            "rank": 0,
            "recommended": False,
            "role": "strong_transcript_companion",
            "phase": "speech_extraction",
            "available": bool(available_asr),
            "reason": "强中文 ASR 用于完整语言信息；视觉抽取仍负责屏幕文字、公式、表格、代码和板书",
            "command_name": "guarded_asr",
            "command": _local_cli_command(["run-asr-plan", str(plan_path)]),
            "mcp_tool": "run_asr_plan",
            "mcp_args": {"plan_json": str(plan_path), "execute": False, "normalize": True},
            "paths": [str(tool.get("path") or tool.get("command") or tool.get("name") or "") for tool in available_asr],
        }
    )

    priority = {"vidclaude": 0, "peepshow": 1, "vidwise": 2, "asr": 3}
    routes.sort(key=lambda route: (0 if route.get("available") else 1, priority.get(str(route.get("name")), 99)))
    for index, route in enumerate(routes, start=1):
        route["rank"] = index
        route["recommended"] = index == 1
    return routes


def _vidclaude_command(project_root: Path, media: Path, output: str, *, max_frames: int, fps: float) -> str:
    return _powershell_join(
        [
            str(project_root / "scripts" / "run-vidclaude-study.ps1"),
            "-Video",
            str(media),
            "-Output",
            output,
            "-Mode",
            "standard",
            "-Fps",
            str(fps),
            "-MaxFrames",
            str(max_frames),
        ]
    )


def _extractor_command_metadata(
    project_root: Path,
    media: Path,
    planned_outputs: dict[str, str],
    *,
    max_frames: int,
    fps: float,
) -> dict[str, dict[str, Any]]:
    return {
        "vidclaude": {
            **resolve_visual_extractor_command("vidclaude"),
            "command": _vidclaude_command(project_root, media, planned_outputs["vidclaude_cache"], max_frames=max_frames, fps=fps),
            "output": planned_outputs["vidclaude_cache"],
        },
        "peepshow": {
            **resolve_visual_extractor_command("peepshow"),
            "command": _peepshow_command(media, planned_outputs["peepshow_output"], max_frames=max_frames, fps=fps),
            "output": planned_outputs["peepshow_output"],
        },
        "vidwise": {
            **resolve_visual_extractor_command("vidwise"),
            "command": _vidwise_command(media, planned_outputs["vidwise_output"]),
            "output": planned_outputs["vidwise_output"],
        },
    }


def _peepshow_command(media: Path, output: str, *, max_frames: int, fps: float) -> str:
    resolved = resolve_visual_extractor_command("peepshow")
    command = list(resolved.get("command_prefix") or ["npx", "peepshow"])
    min_frames = max(1, min(4, int(max_frames or 1)))
    command.extend(
        [
            str(media),
            "--output",
            output,
            "--emit",
            "json",
            "--fps",
            str(fps),
            "--max",
            str(max_frames),
            "--min",
            str(min_frames),
            "--ocr",
            "--no-transcribe",
            "--no-audio-events",
            "--gpu",
            "off",
        ]
    )
    return _powershell_join(command)


def _vidwise_command(media: Path, output: str) -> str:
    resolved = resolve_visual_extractor_command("vidwise")
    vidwise = (resolved.get("command_prefix") or ["vidwise"])[0]
    return _powershell_join([vidwise, str(media), "--model", "tiny", "--no-guide", "--frame-interval", "1", "--frame-threshold", "0.01", "--output-dir", output])


def _pipeline_command(
    root: Path,
    title: str,
    topic: str,
    planned_outputs: dict[str, str],
    *,
    target: str,
    extractors: list[str],
    media: str | None = None,
    transcript: str | None = None,
) -> str:
    args = [
        "run-lecture-pipeline",
        str(root),
        "--title",
        title,
        "--topic",
        topic,
    ]
    if "vidclaude" in extractors:
        args.extend(["--vidclaude-cache", planned_outputs["vidclaude_cache"]])
    if "peepshow" in extractors:
        args.extend(["--peepshow-output", planned_outputs["peepshow_output"]])
    if "vidwise" in extractors:
        args.extend(["--vidwise-output", planned_outputs["vidwise_output"]])
    if "transcript" in extractors:
        if not media or not transcript:
            raise ValueError("media and transcript are required for transcript pipeline commands")
        args.extend(["--media", media, "--transcript", transcript])
    args.extend(["--webui-output-dir", planned_outputs["webui_output_dir"], "--target", target])
    return _local_cli_command(args)


def _local_cli_command(args: list[str]) -> str:
    return _powershell_join([str(_project_root() / "scripts" / "video-knowledge.ps1"), *args])


def _powershell_join(args: list[str]) -> str:
    return " ".join(_quote_powershell_arg(str(arg)) for arg in args)
