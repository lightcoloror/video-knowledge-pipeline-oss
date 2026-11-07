from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .bundle_status import bundle_status_report
from .config import (
    DEFAULT_LOCAL_FRAME_BUDGET,
    DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
    public_vision_provider_profile,
    resolve_vision_execution_profile,
)
from .knowledge_coverage import audit_knowledge_coverage
from .knowledge_note_export import export_knowledge_note
from .local_video_run import prepare_local_video_run
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .multimodal_frame_analyzer import run_multimodal_frame_analysis
from .storage import write_json
from .temporal_frame_groups import run_temporal_frame_groups
from .temporal_visual_analyzer import run_temporal_visual_analysis
from .video_frame_router import run_video_frame_router
from .vision_acceptance import vision_acceptance_plan
from .vision_preflight import vision_execution_preflight
from .visual_structure import run_visual_structure_plan

ACCEPTANCE_SCHEMA = "video_knowledge_acceptance_run.v1"


def run_acceptance_run(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    title: str = "",
    copy_media: bool = False,
    execute_asr: bool = False,
    asr_preset: str = "sensevoice",
    asr_model: str = "iic/SenseVoiceSmall",
    transcript_path: str | Path | None = None,
    build_initial_bundle: bool = True,
    sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
    detect_scenes: bool = True,
    extract_frames: bool = True,
    execute_temporal_groups: bool = False,
    execute_vision: bool = False,
    execute_ebook_pipeline: bool = False,
    semantic_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    provider_config: dict[str, Any] | None = None,
    confirm_vision_calls: int | None = None,
    confirm_vision_indexes: str = "",
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run the local, evidence-preserving acceptance workflow for one video.

    The default is preview-safe: it prepares the local run and writes reports, but
    does not call cloud vision APIs, ebook HTTP parsing, or ASR execution unless
    the caller explicitly enables those branches.
    """
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    profile = resolve_vision_execution_profile(
        provider_config=provider_config,
        multimodal_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
    )
    provider = public_vision_provider_profile(profile["provider_config"])

    steps: list[dict[str, Any]] = []
    local_run = _step(
        steps,
        "prepare_local_video_run",
        "准备本地视频运行目录",
        lambda: prepare_local_video_run(
            media_path,
            root,
            title=title,
            copy_media=copy_media,
            plan_asr=True,
            execute_asr=execute_asr,
            asr_preset=asr_preset,
            asr_model=asr_model,
            transcript_path=transcript_path,
            build_initial_bundle=build_initial_bundle,
            sample_interval=sample_interval,
            max_frames=max_frames,
            sample_mode=sample_mode,
            detect_scenes=detect_scenes,
            extract_frames=extract_frames,
            timeout_seconds=timeout_seconds,
        ),
    )
    bundle_dir = _bundle_dir_from_local_run(local_run)

    if bundle_dir:
        _run_bundle_steps(
            steps,
            bundle_dir,
            title=title,
            execute_temporal_groups=execute_temporal_groups,
            execute_vision=execute_vision,
            execute_ebook_pipeline=execute_ebook_pipeline,
            semantic_limit=profile["multimodal_limit"],
            temporal_limit=profile["temporal_limit"],
            frame_count=profile["frame_count"],
            provider_config=profile["provider_config"],
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )
    else:
        steps.append(
            {
                "key": "bundle_steps",
                "label": "Bundle 后续步骤",
                "status": "skipped",
                "reason": "initial bundle was not created",
            }
        )

    result = {
        "schema": ACCEPTANCE_SCHEMA,
        "started_at": started_at,
        "finished_at": now_iso(),
        "title": title,
        "media_path": str(Path(media_path).expanduser().resolve()),
        "output_dir": str(root),
        "bundle_dir": str(bundle_dir) if bundle_dir else "",
        "mode": {
            "execute_asr": bool(execute_asr),
            "execute_temporal_groups": bool(execute_temporal_groups),
            "execute_vision": bool(execute_vision),
            "execute_ebook_pipeline": bool(execute_ebook_pipeline),
        },
        "provider": provider,
        "execution_profile": {
            "provider": provider,
            "semantic_limit": profile["multimodal_limit"],
            "temporal_limit": profile["temporal_limit"],
            "frame_count": profile["frame_count"],
        },
        "steps": steps,
    }
    result["summary"] = _summary(steps)
    json_path = root / "acceptance-run.json"
    report_path = root / "acceptance-report.md"
    result["json_path"] = str(json_path)
    result["report_path"] = str(report_path)
    write_json(json_path, result)
    report_path.write_text(render_acceptance_report(result), encoding="utf-8")
    return result


def run_acceptance_bundle(
    bundle_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    title: str = "",
    execute_temporal_groups: bool = False,
    execute_vision: bool = False,
    execute_ebook_pipeline: bool = False,
    semantic_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    provider_config: dict[str, Any] | None = None,
    confirm_vision_calls: int | None = None,
    confirm_vision_indexes: str = "",
) -> dict[str, Any]:
    """Run the acceptance workflow from an existing review bundle."""
    bundle_root = Path(bundle_dir).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve() if output_dir else bundle_root.parent
    root.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    profile = resolve_vision_execution_profile(
        provider_config=provider_config,
        multimodal_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
    )
    provider = public_vision_provider_profile(profile["provider_config"])
    steps: list[dict[str, Any]] = []
    _step(steps, "existing_bundle", "读取已有 Bundle", lambda: _validate_existing_bundle(bundle_root))
    _run_bundle_steps(
        steps,
        bundle_root,
        title=title,
        execute_temporal_groups=execute_temporal_groups,
        execute_vision=execute_vision,
        execute_ebook_pipeline=execute_ebook_pipeline,
        semantic_limit=profile["multimodal_limit"],
        temporal_limit=profile["temporal_limit"],
        frame_count=profile["frame_count"],
        provider_config=profile["provider_config"],
        confirm_vision_calls=confirm_vision_calls,
        confirm_vision_indexes=confirm_vision_indexes,
    )
    result = {
        "schema": ACCEPTANCE_SCHEMA,
        "started_at": started_at,
        "finished_at": now_iso(),
        "title": title,
        "media_path": "",
        "output_dir": str(root),
        "bundle_dir": str(bundle_root),
        "mode": {
            "execute_asr": False,
            "execute_temporal_groups": bool(execute_temporal_groups),
            "execute_vision": bool(execute_vision),
            "execute_ebook_pipeline": bool(execute_ebook_pipeline),
        },
        "provider": provider,
        "execution_profile": {
            "provider": provider,
            "semantic_limit": profile["multimodal_limit"],
            "temporal_limit": profile["temporal_limit"],
            "frame_count": profile["frame_count"],
        },
        "steps": steps,
    }
    result["summary"] = _summary(steps)
    json_path = root / "acceptance-run.json"
    report_path = root / "acceptance-report.md"
    result["json_path"] = str(json_path)
    result["report_path"] = str(report_path)
    write_json(json_path, result)
    report_path.write_text(render_acceptance_report(result), encoding="utf-8")
    return result


def render_acceptance_report(result: dict[str, Any]) -> str:
    bundle_dir = str(result.get("bundle_dir") or "")
    root = Path(str(result.get("output_dir") or "."))
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Video Knowledge Acceptance Run",
        "",
        f"- Title: {result.get('title') or '(untitled)'}",
        f"- Media: `{result.get('media_path', '')}`",
        f"- Output: `{result.get('output_dir', '')}`",
        f"- Bundle: `{bundle_dir or '(not created)'}`",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Workflow: `{summary.get('workflow_status', summary.get('status', 'unknown'))}`",
        f"- Content: `{summary.get('content_status', summary.get('status', 'unknown'))}`",
        "",
        "## Mode",
        "",
    ]
    mode = result.get("mode") if isinstance(result.get("mode"), dict) else {}
    for key in ("execute_asr", "execute_temporal_groups", "execute_vision", "execute_ebook_pipeline"):
        lines.append(f"- `{key}`: `{mode.get(key, False)}`")
    profile = result.get("execution_profile") if isinstance(result.get("execution_profile"), dict) else {}
    provider = profile.get("provider") if isinstance(profile.get("provider"), dict) else {}
    if profile:
        lines.extend(
            [
                "",
                "## Vision Execution Profile",
                "",
                f"- Provider: `{provider.get('provider', '')}` / `{provider.get('model', '')}`",
                f"- Semantic limit: `{profile.get('semantic_limit', 0)}`",
                f"- Temporal limit: `{profile.get('temporal_limit', 0)}`",
                f"- Frame count: `{profile.get('frame_count', 0)}`",
            ]
        )
    lines.extend(["", "## Steps", "", "| Step | Status | Artifact |", "|---|---|---|"])
    for step in result.get("steps") or []:
        if not isinstance(step, dict):
            continue
        artifact = _primary_artifact(step)
        lines.append(f"| {_md_cell(str(step.get('label') or step.get('key') or ''))} | `{step.get('status', '')}` | `{_md_cell(artifact)}` |")
    lines.extend(["", "## Human-readable Artifacts", ""])
    artifacts = _artifact_lines(root, bundle_dir)
    lines.extend(artifacts or ["- No downstream artifacts were created."])
    failures = [step for step in result.get("steps") or [] if isinstance(step, dict) and step.get("status") == "failed"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for step in failures:
            lines.append(f"- `{step.get('key')}`: {step.get('error', '')}")
    next_action = summary.get("next_action") if isinstance(summary.get("next_action"), dict) else {}
    if next_action:
        lines.extend(
            [
                "",
                "## Bundle Next Action",
                "",
                f"- Status: `{next_action.get('status', '')}`",
                f"- Key: `{next_action.get('key', '')}`",
                f"- Label: {next_action.get('label', '')}",
                f"- Tool: `{next_action.get('mcp_tool', '')}`",
                f"- Args: `{next_action.get('mcp_args_path', '')}`",
                f"- Human required: `{next_action.get('human_required', '')}`",
            ]
        )
        reason = str(next_action.get("reason") or next_action.get("hint") or "").strip()
        if reason:
            lines.append(f"- Reason: {reason}")
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "- 如果 `execute_vision=false`，当前多模态分支只是候选预览；配置 API 后再显式执行。",
            "- 如果 `execute_vision=true`，验收流程会先运行 preflight，并要求 `confirm_vision_calls` / `confirm_vision_indexes` 精确匹配后才调用模型。",
            "- 如果 `execute_temporal_groups=false`，当前连续片段理解没有真实 5-12 帧证据组；需要先执行帧组生成。",
            "- 图文截图解析仍走 `run_visual_structure`，底层复用 `ebook_markdown_pipeline`。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _run_bundle_steps(
    steps: list[dict[str, Any]],
    bundle_dir: Path,
    *,
    title: str,
    execute_temporal_groups: bool,
    execute_vision: bool,
    execute_ebook_pipeline: bool,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    provider_config: dict[str, Any] | None,
    confirm_vision_calls: int | None,
    confirm_vision_indexes: str,
) -> None:
    _step(steps, "video_frame_router", "画面类型路由", lambda: run_video_frame_router(bundle_dir))
    _step(
        steps,
        "visual_structure_preview",
        "图文截图解析分支",
        lambda: run_visual_structure_plan(
            bundle_dir,
            execute_ebook_pipeline=execute_ebook_pipeline,
            timeout_seconds=120,
        ),
    )
    _step(
        steps,
        "temporal_frame_groups",
        "连续片段帧组",
        lambda: run_temporal_frame_groups(
            bundle_dir,
            execute=execute_temporal_groups,
            frame_count=frame_count,
            limit=temporal_limit,
            timeout_seconds=120,
        ),
    )
    vision_gate = _acceptance_vision_execution_gate(
        bundle_dir,
        execute_vision=execute_vision,
        provider_config=provider_config,
        semantic_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
        confirm_vision_calls=confirm_vision_calls,
        confirm_vision_indexes=confirm_vision_indexes,
    )
    if execute_vision and vision_gate.get("preflight"):
        preflight_result = vision_gate["preflight"]
        _step(steps, "vision_execution_preflight", "真实视觉执行门禁", lambda: preflight_result)
    if vision_gate.get("gate"):
        gate_result = vision_gate["gate"]
        _step(steps, "multimodal_frame_analysis", "多模态单帧理解", lambda: _branch_gate_result(gate_result, branch="semantic_frame"))
    else:
        _step(
            steps,
            "multimodal_frame_analysis",
            "多模态单帧理解",
            lambda: run_multimodal_frame_analysis(
                bundle_dir,
                execute=execute_vision,
                provider_config=provider_config,
                limit=semantic_limit,
                confirm_vision_calls=confirm_vision_calls,
                confirm_vision_indexes=confirm_vision_indexes,
            ),
        )
    if vision_gate.get("gate"):
        gate_result = vision_gate["gate"]
        _step(steps, "temporal_visual_analysis", "连续片段理解", lambda: _branch_gate_result(gate_result, branch="temporal_sequence"))
    else:
        _step(
            steps,
            "temporal_visual_analysis",
            "连续片段理解",
            lambda: run_temporal_visual_analysis(
                bundle_dir,
                execute=execute_vision,
                provider_config=provider_config,
                frame_count=frame_count,
                limit=temporal_limit,
                confirm_vision_calls=confirm_vision_calls,
                confirm_vision_indexes=confirm_vision_indexes,
            ),
        )
    _step(
        steps,
        "vision_acceptance_plan",
        "多模态实测计划",
        lambda: vision_acceptance_plan(
            bundle_dir,
            provider_config=provider_config,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
        ),
    )
    _step(steps, "knowledge_coverage", "知识覆盖审计", lambda: audit_knowledge_coverage(bundle_dir))
    _step(
        steps,
        "export_knowledge_note",
        "导出人类可读知识笔记",
        lambda: export_knowledge_note(bundle_dir, title=title),
    )
    _step(steps, "bundle_status", "Bundle 状态报告", lambda: bundle_status_report(bundle_dir))


def _validate_existing_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest = bundle_dir / "manifest.json"
    timeline = bundle_dir / "timeline.json"
    if not manifest.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {bundle_dir}")
    if not timeline.exists():
        raise FileNotFoundError(f"bundle missing timeline.json: {bundle_dir}")
    review = bundle_dir / "review.html"
    return {
        "status": "ok",
        "bundle_dir": str(bundle_dir),
        "review_html": str(review) if review.exists() else "",
        "manifest_path": str(manifest),
        "timeline_path": str(timeline),
    }


def _acceptance_vision_execution_gate(
    bundle_dir: Path,
    *,
    execute_vision: bool,
    provider_config: dict[str, Any] | None,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    confirm_vision_calls: int | None,
    confirm_vision_indexes: str,
) -> dict[str, Any]:
    if not execute_vision:
        return {}
    preflight = vision_execution_preflight(
        bundle_dir,
        provider_config=provider_config,
        semantic_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
        include_semantic=True,
        include_temporal=True,
        write=True,
    )
    if not preflight.get("ready_to_execute"):
        return {"preflight": preflight, "gate": _preflight_blocked_gate(bundle_dir, preflight)}
    confirmation = _confirmation_gate(preflight, confirm_vision_calls=confirm_vision_calls, confirm_vision_indexes=confirm_vision_indexes)
    if confirmation:
        return {"preflight": preflight, "gate": confirmation}
    return {"preflight": preflight}


def _preflight_blocked_gate(bundle_dir: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    blockers = preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else []
    return {
        "schema": "lecture_acceptance_vision_gate.v1",
        "status": "vision_preflight_blocked",
        "bundle_dir": str(bundle_dir),
        "preflight_path": preflight.get("preflight_path", ""),
        "preflight_json_path": preflight.get("preflight_json_path", ""),
        "blockers": blockers,
        "summary": {
            "provider": preflight.get("provider") if isinstance(preflight.get("provider"), dict) else {},
            "ready_to_execute": False,
            "expected_api_calls": preflight.get("expected_api_calls", 0),
            "expected_indexes": _preflight_selected_index_string(preflight),
            "candidate_counts": preflight.get("candidate_counts") if isinstance(preflight.get("candidate_counts"), dict) else {},
            "blocker_keys": [str(item.get("key") or "") for item in blockers if isinstance(item, dict)],
            "hint": "Inspect vision-execution-preflight.md, fix blockers, then retry execute_vision.",
        },
    }


def _confirmation_gate(
    preflight: dict[str, Any],
    *,
    confirm_vision_calls: int | None,
    confirm_vision_indexes: str,
) -> dict[str, Any] | None:
    expected_calls = int(preflight.get("expected_api_calls") or 0)
    expected_indexes = _preflight_selected_index_string(preflight)
    calls_ok = confirm_vision_calls is not None and int(confirm_vision_calls) == expected_calls
    indexes_ok = str(confirm_vision_indexes or "").strip() == expected_indexes
    if calls_ok and indexes_ok:
        return None
    return {
        "schema": "lecture_acceptance_vision_gate.v1",
        "status": "vision_confirmation_required",
        "bundle_dir": str(preflight.get("bundle_dir") or ""),
        "preflight_path": preflight.get("preflight_path", ""),
        "preflight_json_path": preflight.get("preflight_json_path", ""),
        "summary": {
            "ready_to_execute": True,
            "expected_api_calls": expected_calls,
            "expected_indexes": expected_indexes,
            "received_confirm_vision_calls": confirm_vision_calls,
            "received_confirm_vision_indexes": str(confirm_vision_indexes or ""),
            "hint": "Retry with confirm_vision_calls and confirm_vision_indexes exactly matching the preflight output.",
        },
    }


def _branch_gate_result(gate: dict[str, Any], *, branch: str) -> dict[str, Any]:
    result = dict(gate)
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    result["summary"] = {**summary, "branch": branch, "model_called": False}
    return result


def _preflight_selected_index_string(preflight: dict[str, Any]) -> str:
    selected = preflight.get("selected_indexes") if isinstance(preflight.get("selected_indexes"), dict) else {}
    indexes: list[int] = []
    for key in ("semantic", "temporal"):
        for value in selected.get(key) or []:
            try:
                indexes.append(int(value))
            except (TypeError, ValueError):
                continue
    return ",".join(str(index) for index in indexes)


def _step(
    steps: list[dict[str, Any]],
    key: str,
    label: str,
    func: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = func()
    except Exception as exc:  # pragma: no cover - exercised through integration failures.
        item = {
            "key": key,
            "label": label,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        steps.append(item)
        return {}
    item = {
        "key": key,
        "label": label,
        "status": str(result.get("status") or "ok") if isinstance(result, dict) else "ok",
        "result": _compact_result(result),
    }
    steps.append(item)
    return result if isinstance(result, dict) else {}


def _bundle_dir_from_local_run(local_run: dict[str, Any]) -> Path | None:
    bundle = local_run.get("initial_bundle") if isinstance(local_run, dict) else None
    if not isinstance(bundle, dict) or str(bundle.get("status") or "") != "ok":
        return None
    value = str(bundle.get("bundle_dir") or "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.exists() else path


def _compact_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    keep = [
        "schema",
        "status",
        "bundle_dir",
        "json_path",
        "markdown_path",
        "report_path",
        "report_markdown_path",
        "coverage_markdown_path",
        "preflight_path",
        "preflight_json_path",
        "note_path",
        "full_transcript_path",
        "full_body_path",
        "review_html",
        "plan_path",
        "knowledge_coverage",
        "review_readiness",
        "summary",
    ]
    compact = {key: result[key] for key in keep if key in result}
    if isinstance(result.get("next_action"), dict):
        compact["next_action"] = _compact_next_action(result.get("next_action"))
    if isinstance(result.get("coverage"), dict):
        compact["coverage"] = _compact_coverage(result.get("coverage"))
    if "initial_bundle" in result:
        compact["initial_bundle"] = result["initial_bundle"]
    return compact


def _compact_next_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = [
        "status",
        "kind",
        "key",
        "label",
        "reason",
        "hint",
        "coverage_status",
        "mcp_tool",
        "mcp_args_path",
        "command",
        "human_required",
    ]
    compact = {key: value[key] for key in keep if key in value}
    blockers = value.get("blockers") if isinstance(value.get("blockers"), list) else []
    weak = value.get("weak_channels") if isinstance(value.get("weak_channels"), list) else []
    if blockers:
        compact["blocker_count"] = len(blockers)
        compact["blocker_keys"] = [str(item.get("key") or "") for item in blockers[:5] if isinstance(item, dict)]
    if weak:
        compact["weak_count"] = len(weak)
        compact["weak_keys"] = [str(item.get("key") or "") for item in weak[:5] if isinstance(item, dict)]
    return compact


def _compact_coverage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    blockers = value.get("blockers") if isinstance(value.get("blockers"), list) else []
    weak = value.get("weak_channels") if isinstance(value.get("weak_channels"), list) else []
    return {
        "status": value.get("status", ""),
        "timeline_items": value.get("timeline_items", 0),
        "items_with_visual_route": value.get("items_with_visual_route", 0),
        "items_with_visual_understanding": value.get("items_with_visual_understanding", 0),
        "items_with_temporal_understanding": value.get("items_with_temporal_understanding", 0),
        "semantic_frame_without_analysis": value.get("semantic_frame_without_analysis", 0),
        "temporal_sequence_without_analysis": value.get("temporal_sequence_without_analysis", 0),
        "missing_visual_understanding": value.get("missing_visual_understanding", 0),
        "blocker_count": len(blockers),
        "weak_count": len(weak),
        "blocker_keys": [str(item.get("key") or "") for item in blockers[:5] if isinstance(item, dict)],
        "weak_keys": [str(item.get("key") or "") for item in weak[:5] if isinstance(item, dict)],
        "next_action": _compact_next_action(value.get("next_action")),
    }


def _summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [step for step in steps if step.get("status") == "failed"]
    skipped = [step for step in steps if step.get("status") == "skipped"]
    workflow_status = "failed" if failed else ("partial" if skipped else "ok")
    content_status, next_action = _content_status(steps, workflow_status)
    return {
        "status": content_status if workflow_status == "ok" else workflow_status,
        "workflow_status": workflow_status,
        "content_status": content_status,
        "step_count": len(steps),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "failed_steps": [str(step.get("key") or "") for step in failed],
        "next_action": next_action,
    }


def _content_status(steps: list[dict[str, Any]], workflow_status: str) -> tuple[str, dict[str, Any]]:
    if workflow_status != "ok":
        return workflow_status, {}
    bundle_status = _step_result(steps, "bundle_status")
    status = str(bundle_status.get("status") or "").strip()
    next_action = _compact_next_action(bundle_status.get("next_action"))
    if status:
        return status, next_action
    coverage_result = _step_result(steps, "knowledge_coverage")
    coverage = coverage_result.get("coverage") if isinstance(coverage_result.get("coverage"), dict) else {}
    coverage_status = str(coverage.get("status") or "").strip()
    if coverage_status and coverage_status != "ok":
        next_action = _compact_next_action(coverage.get("next_action"))
        return f"coverage_{coverage_status}", next_action
    return "ready", {}


def _step_result(steps: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for step in steps:
        if step.get("key") != key:
            continue
        result = step.get("result")
        return result if isinstance(result, dict) else {}
    return {}


def _primary_artifact(step: dict[str, Any]) -> str:
    result = step.get("result") if isinstance(step.get("result"), dict) else {}
    for key in (
        "report_markdown_path",
        "preflight_path",
        "plan_path",
        "markdown_path",
        "coverage_markdown_path",
        "note_path",
        "review_html",
        "report_path",
        "json_path",
    ):
        value = str(result.get(key) or "").strip()
        if value:
            return value
    bundle = result.get("initial_bundle") if isinstance(result.get("initial_bundle"), dict) else {}
    return str(bundle.get("review_html") or bundle.get("bundle_dir") or "")


def _artifact_lines(root: Path, bundle_dir: str) -> list[str]:
    candidates = [
        root / "video-knowledge-run.md",
        Path(bundle_dir) / "review.html" if bundle_dir else None,
        Path(bundle_dir) / "bundle-status.md" if bundle_dir else None,
        Path(bundle_dir) / "knowledge-coverage.md" if bundle_dir else None,
        Path(bundle_dir) / "vision-acceptance-plan.md" if bundle_dir else None,
        Path(bundle_dir) / "exports" / "knowledge-note.md" if bundle_dir else None,
        Path(bundle_dir) / "exports" / "full-transcript.md" if bundle_dir else None,
    ]
    lines = []
    for path in candidates:
        if path and path.exists():
            lines.append(f"- `{path}`")
    return lines
