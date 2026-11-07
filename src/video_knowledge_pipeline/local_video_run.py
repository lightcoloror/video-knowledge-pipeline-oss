from __future__ import annotations

from .config import DEFAULT_LOCAL_FRAME_BUDGET, DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS, DEFAULT_LOCAL_FRAME_SAMPLING_MODE
from pathlib import Path
from typing import Any

from .asr_environment import asr_environment_status
from .asr_execution import run_asr_plan
from .asr_runner import plan_asr_run
from .entity_lexicon import build_entity_lexicon
from .lecture_package import build_lecture_package
from .local_tool_inventory import local_runtime_preflight
from .models import now_iso
from .orchestrator import add_video, init_project
from .powershell import quote_powershell_literal
from .storage import read_json, write_json
from .video_source import prepare_video_source
from .webui_bridge import export_webui_bundle


def prepare_local_video_run(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    title: str = "",
    copy_media: bool = False,
    plan_asr: bool = True,
    execute_asr: bool = False,
    asr_preset: str = "sensevoice",
    asr_model: str = "iic/SenseVoiceSmall",
    transcript_path: str | Path | None = None,
    build_initial_bundle: bool = False,
    sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
    detect_scenes: bool = True,
    extract_frames: bool = True,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Create a human-readable run folder for one local knowledge video."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    media = Path(media_path).expanduser().resolve()
    source_dir = root / "source"
    source = prepare_video_source(str(media), source_dir, execute=copy_media)
    selected_media = Path(str(source.get("local_media_path") or media))
    runtime_preflight = local_runtime_preflight()
    stage_results: list[dict[str, Any]] = [
        {"stage": "source", "status": "ok", "error_type": "", "error": ""}
    ]
    failed_stages: list[str] = []
    recovery_commands = [
        dict(row)
        for row in runtime_preflight.get("recovery_commands") or []
        if isinstance(row, dict)
    ]
    media_capabilities = (
        runtime_preflight.get("capabilities", {}).get("media", {})
        if isinstance(runtime_preflight.get("capabilities"), dict)
        else {}
    )
    missing_media = [
        name
        for name in ("ffmpeg", "ffprobe")
        if not isinstance(media_capabilities.get(name), dict)
        or not media_capabilities.get(name, {}).get("available")
    ]
    if missing_media:
        failed_stages.append("media_preflight")
        stage_results.append(
            {
                "stage": "media_preflight",
                "status": "failed",
                "error_type": "missing_media_tools",
                "error": "Missing local media tools: " + ", ".join(missing_media),
            }
        )
    else:
        stage_results.append(
            {"stage": "media_preflight", "status": "ok", "error_type": "", "error": ""}
        )
    try:
        asr_env = asr_environment_status()
        stage_results.append(
            {"stage": "asr_environment", "status": "ok", "error_type": "", "error": ""}
        )
    except Exception as exc:  # pragma: no cover - defensive environment boundary.
        asr_env = {"ok": False, "error": str(exc)}
        _record_stage_failure(stage_results, failed_stages, "asr_environment", exc)
    asr_plan: dict[str, Any] | None = None
    asr_run: dict[str, Any] | None = None
    pre_asr_context: dict[str, Any] | None = None
    if plan_asr:
        try:
            pre_asr_context = _prepare_pre_asr_context(root, title=title or media.stem)
            stage_results.append(
                {"stage": "pre_asr_context", "status": "ok", "error_type": "", "error": ""}
            )
        except Exception as exc:
            _record_stage_failure(stage_results, failed_stages, "pre_asr_context", exc)
        if pre_asr_context is not None:
            try:
                asr_plan = plan_asr_run(
                    root,
                    selected_media,
                    preset=asr_preset,
                    model=asr_model,
                    hotword=str(pre_asr_context.get("hotword_text") or ""),
                )
                stage_results.append(
                    {"stage": "asr_plan", "status": "ok", "error_type": "", "error": ""}
                )
            except Exception as exc:
                _record_stage_failure(stage_results, failed_stages, "asr_plan", exc)
                recovery_commands.append(
                    {
                        "stage": "asr_plan",
                        "key": "retry_prepare_local_video_run",
                        "command": _prepare_retry_command(media, root),
                        "reason": "Fix the reported ASR planning/runtime blocker, then rerun the same local preparation command.",
                    }
                )
        if execute_asr and asr_plan:
            try:
                asr_run = run_asr_plan(asr_plan["plan_path"], execute=True, timeout_seconds=timeout_seconds)
                stage_results.append(
                    {"stage": "asr_execute", "status": str(asr_run.get("status") or "ok"), "error_type": "", "error": str(asr_run.get("error") or "")}
                )
                if str(asr_run.get("status") or "ok") not in {"ok", "completed"}:
                    failed_stages.append("asr_execute")
            except Exception as exc:
                _record_stage_failure(stage_results, failed_stages, "asr_execute", exc)
                recovery_commands.append(
                    {
                        "stage": "asr_execute",
                        "key": "retry_asr_plan",
                        "command": f".\\scripts\\video-knowledge.ps1 run-asr-plan {quote_powershell_literal(str(asr_plan.get('plan_path') or ''))} --execute",
                        "reason": "Retry the existing ASR plan after resolving the reported runtime blocker.",
                    }
                )
    transcript = _selected_transcript(transcript_path=transcript_path, asr_run=asr_run)
    initial_bundle: dict[str, Any] | None = None
    if build_initial_bundle:
        initial_bundle = _build_initial_bundle(
            root,
            selected_media,
            title=title or media.stem,
            transcript_path=transcript,
            sample_interval=sample_interval,
            max_frames=max_frames,
            sample_mode=sample_mode,
            detect_scenes=detect_scenes,
            extract_frames=extract_frames,
        )
        bundle_status = str(initial_bundle.get("status") or "")
        if bundle_status == "ok":
            stage_results.append(
                {"stage": "initial_bundle", "status": "ok", "error_type": "", "error": ""}
            )
        else:
            failed_stages.append("initial_bundle")
            stage_results.append(
                {
                    "stage": "initial_bundle",
                    "status": "failed",
                    "error_type": "initial_bundle_failed",
                    "error": str(initial_bundle.get("error") or "initial bundle failed"),
                }
            )
            recovery_commands.append(
                {
                    "stage": "initial_bundle",
                    "key": "retry_initial_bundle",
                    "command": _prepare_retry_command(media, root, build_initial_bundle=True),
                    "reason": "Fix media/runtime blockers, then rebuild the initial bundle.",
                }
            )
    failed_stages = list(dict.fromkeys(failed_stages))
    recovery_commands = _dedupe_recovery_commands(recovery_commands)
    ok = not failed_stages
    report = {
        "schema": "video_knowledge_local_video_run.v1",
        "created_at": now_iso(),
        "ok": ok,
        "status": "ok" if ok else "partial_failure",
        "failed_stage": failed_stages[0] if failed_stages else "",
        "failed_stages": failed_stages,
        "stage_results": stage_results,
        "runtime_preflight": runtime_preflight,
        "recovery_commands": recovery_commands,
        "title": title or media.stem,
        "workspace_dir": str(root),
        "media_path": str(media),
        "selected_media_path": str(selected_media),
        "source": source,
        "asr_environment": _compact_asr_env(asr_env),
        "pre_asr_context": pre_asr_context,
        "asr_plan": asr_plan,
        "asr_run": asr_run,
        "transcript_path": str(transcript or ""),
        "initial_bundle": initial_bundle,
        "next_steps": _next_steps(
            root,
            asr_plan=asr_plan,
            asr_run=asr_run,
            execute_asr=execute_asr,
            transcript=transcript,
            initial_bundle=initial_bundle,
        ),
    }
    json_path = root / "video-knowledge-run.json"
    md_path = root / "video-knowledge-run.md"
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    write_json(json_path, report)
    md_path.write_text(render_local_video_run_markdown(report), encoding="utf-8")
    return report


def _prepare_pre_asr_context(root: Path, *, title: str) -> dict[str, Any]:
    """Create auditable, metadata-only ASR hints before the first decode."""

    manifest_path = root / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        payload = read_json(manifest_path)
        if isinstance(payload, dict):
            existing = dict(payload)
    existing.setdefault("title", title)
    write_json(manifest_path, existing)
    lexicon = build_entity_lexicon(root, phase="pre_asr", write=True)
    return {
        "phase": "pre_asr",
        "hotword_text": str(lexicon.get("hotword_text") or ""),
        "hotword_count": int(lexicon.get("hotword_variant_count") or 0),
        "lexicon_json": str(root / "entity-lexicon.pre-asr.json"),
        "hotword_audit_json": str(root / "entity-hotword-audit.pre-asr.json"),
        "evaluation_reference_used": False,
    }


def render_local_video_run_markdown(report: dict[str, Any]) -> str:
    asr_plan = report.get("asr_plan") if isinstance(report.get("asr_plan"), dict) else {}
    asr_run = report.get("asr_run") if isinstance(report.get("asr_run"), dict) else {}
    asr_env = report.get("asr_environment") if isinstance(report.get("asr_environment"), dict) else {}
    lines = [
        "# Video Knowledge Run",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- OK: `{report.get('ok', False)}`",
        f"- Failed stage: `{report.get('failed_stage', '')}`",
        f"- Title: {report.get('title', '')}",
        f"- Workspace: `{report.get('workspace_dir', '')}`",
        f"- Media: `{report.get('selected_media_path', '')}`",
        f"- Created: `{report.get('created_at', '')}`",
        "",
        "## ASR",
        "",
        f"- Environment OK: `{asr_env.get('ok', False)}`",
        f"- Available tools: `{', '.join(asr_env.get('available_tools') or [])}`",
        f"- Preset: `{asr_plan.get('preset', '')}`",
        f"- Plan: `{asr_plan.get('plan_path', '')}`",
        f"- Pre-ASR hotwords: `{(report.get('pre_asr_context') or {}).get('hotword_count', 0)}`",
        f"- Execute status: `{asr_run.get('status', 'not_run')}`",
        f"- Selected transcript: `{report.get('transcript_path', '')}`",
    ]
    failures = [row for row in report.get("stage_results") or [] if isinstance(row, dict) and row.get("status") == "failed"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(
                f"- `{failure.get('stage', '')}` / `{failure.get('error_type', '')}`: {failure.get('error', '')}"
            )
    if asr_plan.get("powershell"):
        lines.extend(["", "### ASR Command", "", "```powershell", str(asr_plan.get("powershell")), "```"])
    normalized = asr_run.get("normalized") if isinstance(asr_run.get("normalized"), dict) else {}
    if normalized:
        lines.extend(
            [
                "",
                "### Transcript Outputs",
                "",
                f"- JSON: `{normalized.get('json_path', '')}`",
                f"- SRT: `{normalized.get('srt_path', '')}`",
            ]
        )
    initial_bundle = report.get("initial_bundle") if isinstance(report.get("initial_bundle"), dict) else {}
    if initial_bundle:
        lines.extend([
            f"- Pre-ASR hotwords: `{(report.get('pre_asr_context') or {}).get('hotword_count', 0)}`",
            "",
                "## Initial Review Bundle",
                "",
                f"- Status: `{initial_bundle.get('status', '')}`",
                f"- WebUI: `{initial_bundle.get('review_html', '')}`",
                f"- Bundle: `{initial_bundle.get('bundle_dir', '')}`",
        ])
        if initial_bundle.get("error"):
            lines.append(f"- Error: `{initial_bundle.get('error', '')}`")
    lines.extend(["", "## Next Steps", ""])
    for step in report.get("next_steps") or []:
        if isinstance(step, dict):
            lines.append(f"- [{step.get('status', 'todo')}] `{step.get('key', '')}`: {step.get('label', '')}")
            if step.get("command"):
                lines.append(f"  - `{step.get('command')}`")
    recovery_commands = [row for row in report.get("recovery_commands") or [] if isinstance(row, dict)]
    if recovery_commands:
        lines.extend(["", "## Recovery Commands", ""])
        for row in recovery_commands:
            lines.append(f"- `{row.get('stage') or row.get('key', '')}`: `{row.get('command', '')}`")
            if row.get("reason"):
                lines.append(f"  - {row.get('reason')}")
    return "\n".join(lines).rstrip() + "\n"


def _record_stage_failure(
    stage_results: list[dict[str, Any]],
    failed_stages: list[str],
    stage: str,
    error: Exception,
) -> None:
    failed_stages.append(stage)
    stage_results.append(
        {
            "stage": stage,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )


def _prepare_retry_command(media: Path, root: Path, *, build_initial_bundle: bool = False) -> str:
    command = (
        ".\\scripts\\video-knowledge.ps1 prepare-local-video-run "
        f"{quote_powershell_literal(str(media))} {quote_powershell_literal(str(root))}"
    )
    return f"{command} --build-initial-bundle" if build_initial_bundle else command


def _dedupe_recovery_commands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        command = str(row.get("command") or "")
        key = (str(row.get("stage") or row.get("key") or ""), command)
        if not command or key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def _compact_asr_env(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(status.get("ok")),
        "venv_dir": str(status.get("venv_dir") or ""),
        "python": status.get("python") if isinstance(status.get("python"), dict) else {},
        "available_tools": status.get("available_tools") if isinstance(status.get("available_tools"), list) else [],
        "recommended_order": status.get("recommended_order") if isinstance(status.get("recommended_order"), list) else [],
    }


def _selected_transcript(*, transcript_path: str | Path | None, asr_run: dict[str, Any] | None) -> Path | None:
    if transcript_path:
        return Path(transcript_path).expanduser().resolve()
    if isinstance(asr_run, dict):
        normalized = asr_run.get("normalized")
        if isinstance(normalized, dict) and normalized.get("json_path"):
            return Path(str(normalized["json_path"])).expanduser().resolve()
    return None


def _build_initial_bundle(
    root: Path,
    media: Path,
    *,
    title: str,
    transcript_path: Path | None,
    sample_interval: float,
    max_frames: int,
    sample_mode: str,
    detect_scenes: bool,
    extract_frames: bool,
) -> dict[str, Any]:
    try:
        init_project(root, title)
        video = add_video(
            root,
            media,
            topic=title,
            transcript_path=transcript_path,
            sample_interval=sample_interval,
            max_frames=max_frames,
            sample_mode=sample_mode,
            detect_scenes=detect_scenes,
            extract_frames=extract_frames,
        )
        package = build_lecture_package(root, title=title)
        bundle = export_webui_bundle(root, output_dir=root / "webui-bundle")
        return {
            "status": "ok",
            "video": video,
            "package_path": package.get("package_path", ""),
            "bundle_dir": bundle.get("bundle_dir", ""),
            "review_html": bundle.get("review_html_path", ""),
            "manifest_path": bundle.get("manifest_path", ""),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "hint": "检查 ffmpeg/ffprobe、视频文件有效性、transcript 路径和帧抽取参数；修好后重新运行 --build-initial-bundle。",
        }


def _next_steps(
    root: Path,
    *,
    asr_plan: dict[str, Any] | None,
    asr_run: dict[str, Any] | None,
    execute_asr: bool,
    transcript: Path | None,
    initial_bundle: dict[str, Any] | None,
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    if asr_plan and not execute_asr and not transcript:
        steps.append(
            {
                "key": "run_asr_plan",
                "label": "执行本地 ASR，并自动归一化 transcript JSON/SRT。",
                "status": "todo",
                "command": f".\\scripts\\video-knowledge.ps1 run-asr-plan {asr_plan.get('plan_path', '')} --execute",
            }
        )
    if transcript:
        steps.append(
            {
                "key": "review_transcript",
                "label": "已选择 transcript；检查分段和术语，必要时重新跑 ASR 或导入修正版。",
                "status": "todo",
                "command": str(transcript),
            }
        )
    if asr_run and asr_run.get("status") == "ok":
        steps.append({"key": "review_transcript", "label": "人工检查 normalized transcript，必要时重分段或修正术语。", "status": "todo", "command": ""})
    if not initial_bundle:
        steps.append(
            {
                "key": "build_initial_bundle",
                "label": "抽帧并生成初始 webui-bundle。",
                "status": "todo",
                "command": f".\\scripts\\video-knowledge.ps1 prepare-local-video-run <media> {root} --build-initial-bundle",
            }
        )
    elif initial_bundle.get("status") == "ok":
        bundle_dir = str(initial_bundle.get("bundle_dir") or "<webui-bundle>")
        steps.extend(
            [
                {"key": "review_bundle", "label": "打开 review.html 人工检查基础时间轴和帧。", "status": "todo", "command": str(initial_bundle.get("review_html") or "")},
                {"key": "run_frame_router", "label": "对 bundle 执行画面路由。", "status": "todo", "command": f".\\scripts\\video-knowledge.ps1 run-video-frame-router {bundle_dir}"},
                {"key": "run_visual_branches", "label": "按路由执行图文解析、多模态单帧、多帧时序分析。", "status": "todo", "command": ""},
            ]
        )
    else:
        steps.append({"key": "fix_initial_bundle", "label": "初始 bundle 生成失败，先查看 video-knowledge-run.md 中的错误。", "status": "blocked", "command": ""})
    if not asr_plan:
        steps.insert(0, {"key": "plan_asr", "label": "生成 ASR 计划。", "status": "todo", "command": f".\\scripts\\video-knowledge.ps1 plan-asr {root} <media>"})
    return steps
