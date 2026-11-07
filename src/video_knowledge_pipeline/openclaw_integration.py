from __future__ import annotations

from .powershell import quote_powershell_literal as _ps_quote
from .config import DEFAULT_LOCAL_FRAME_BUDGET, DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS, DEFAULT_LOCAL_FRAME_SAMPLING_MODE
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .local_video_run import prepare_local_video_run
from .model_json import extract_last_json_document
from .models import now_iso
from .path_defaults import workspace_root

Runner = Callable[[list[str], dict[str, str], int], subprocess.CompletedProcess[str]]

_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".ts"}


def openclaw_video_plan(
    url_or_text: str,
    *,
    output_dir: str | Path = "",
    vdo_root: str | Path = "",
    vdo_output_dir: str | Path = "",
    backend: str = "",
    write_manifests: bool = True,
    include_manifests: bool = False,
    timeout_seconds: int = 120,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Plan a video-link download through video-download-orchestrator.

    This is intentionally planning-only. Real download execution stays owned by
    video-download-orchestrator and its OpenClaw confirmation boundary.
    """

    raw = str(url_or_text or "").strip()
    if not raw:
        return _openclaw_result(
            ok=False,
            status="empty_input",
            input_type="unknown",
            will_download=False,
            operator_boundary=_operator_boundary("planning_only"),
            next_actions=["provide_url_or_telegram_text"],
        )

    project_root = _project_root()
    plan_output_dir = Path(output_dir).expanduser().resolve() if output_dir else project_root / "openclaw-runs" / "download-plans"
    vdo_plan_dir = Path(vdo_output_dir).expanduser().resolve() if vdo_output_dir else plan_output_dir / "video-download-orchestrator"
    root = Path(vdo_root).expanduser().resolve() if vdo_root else _default_vdo_root()

    command = [
        sys.executable,
        "-m",
        "video_orchestrator.cli",
        "openclaw-plan",
        raw,
        "--output-dir",
        str(vdo_plan_dir),
        "--format",
        "json",
    ]
    if backend:
        command.extend(["--backend", backend])
    command.append("--write-manifests" if write_manifests else "--no-write-manifests")
    if include_manifests:
        command.append("--include-manifests")

    completed = _run_vdo_command(command, root, timeout_seconds, runner=runner)
    if completed.returncode != 0:
        return _openclaw_result(
            ok=False,
            status="download_plan_failed",
            input_type=_classify_input(raw),
            will_download=False,
            download_plan={},
            artifacts={"vdo_root": str(root), "vdo_output_dir": str(vdo_plan_dir)},
            operator_boundary=_operator_boundary("planning_only"),
            next_actions=["check_video_download_orchestrator", "retry_openclaw_video_plan"],
            errors=[_compact_process_error(completed)],
            commands={"host_plan": _powershell_command(command, root)},
        )

    payload = _parse_json_stdout(completed.stdout)
    if not isinstance(payload, dict):
        return _openclaw_result(
            ok=False,
            status="download_plan_parse_failed",
            input_type=_classify_input(raw),
            will_download=False,
            download_plan={},
            artifacts={"vdo_root": str(root), "vdo_output_dir": str(vdo_plan_dir)},
            operator_boundary=_operator_boundary("planning_only"),
            next_actions=["inspect_video_download_orchestrator_stdout"],
            errors=[{"message": "video-download-orchestrator returned non-JSON stdout", "stdout_excerpt": _excerpt(completed.stdout)}],
            commands={"host_plan": _powershell_command(command, root)},
        )

    will_download = bool(payload.get("will_download"))
    artifacts = _download_artifacts(payload, vdo_plan_dir=vdo_plan_dir, vdo_root=root)
    return _openclaw_result(
        ok=True,
        status=str(payload.get("status") or "planned"),
        input_type=_classify_input(raw),
        will_download=will_download,
        download_plan=payload,
        artifacts=artifacts,
        operator_boundary=_operator_boundary("planning_only"),
        next_actions=_plan_next_actions(payload),
        commands={
            "host_plan": _powershell_command(command, root),
            "host_execute_boundary": _vdo_execute_hint(raw, root, vdo_plan_dir),
            "openclaw_docker": "Use VDO_API_BASE=http://host.docker.internal:8921 for download planning, then pass the downloaded host path to openclaw-video-ingest.",
        },
        docker=_docker_notes(),
    )


def openclaw_video_ingest(
    media_path: str | Path,
    *,
    workspace: str | Path = "",
    title: str = "",
    copy_media: bool = False,
    plan_asr: bool = True,
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
    timeout_seconds: int = 1800,
    prepare_runner: Callable[..., dict[str, Any]] = prepare_local_video_run,
) -> dict[str, Any]:
    """Prepare a local or already-downloaded video for knowledge extraction."""

    media = Path(media_path).expanduser().resolve()
    if not media.exists():
        return _openclaw_result(
            ok=False,
            status="media_not_found",
            input_type="local_video_path",
            will_download=False,
            media_path=str(media),
            workspace=str(_default_workspace_for(media, workspace)),
            operator_boundary=_operator_boundary("local_ingest_only"),
            next_actions=["provide_existing_media_path", "run_openclaw_video_plan_for_links"],
        )

    run_workspace = _default_workspace_for(media, workspace)
    report = prepare_runner(
        media,
        run_workspace,
        title=title or media.stem,
        copy_media=copy_media,
        plan_asr=plan_asr,
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
    )
    artifacts = _ingest_artifacts(report)
    return _openclaw_result(
        ok=bool(report),
        status="ingested" if report else "ingest_failed",
        input_type="local_video_path",
        will_download=False,
        media_path=str(media),
        workspace=str(run_workspace),
        artifacts=artifacts,
        review_url_or_file=artifacts.get("review_html") or artifacts.get("run_markdown") or "",
        next_actions=_ingest_next_actions(report),
        operator_boundary=_operator_boundary("local_ingest_only"),
        local_run=report,
        docker=_docker_notes(media_path=str(media), workspace=str(run_workspace)),
    )


def openclaw_video_link(
    url_or_text: str,
    *,
    output_dir: str | Path = "",
    vdo_root: str | Path = "",
    vdo_output_dir: str | Path = "",
    backend: str = "",
    allow_download: bool = False,
    actor_id: str = "",
    confirm_download: bool = False,
    confirm_sensitive: bool = False,
    ingest_after_download: bool = False,
    downloaded_media_path: str | Path = "",
    workspace: str | Path = "",
    title: str = "",
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
    sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
    timeout_seconds: int = 1800,
    runner: Runner | None = None,
    prepare_runner: Callable[..., dict[str, Any]] = prepare_local_video_run,
) -> dict[str, Any]:
    """Plan a link and optionally hand an explicitly downloaded file into ingest."""

    plan = openclaw_video_plan(
        url_or_text,
        output_dir=output_dir,
        vdo_root=vdo_root,
        vdo_output_dir=vdo_output_dir,
        backend=backend,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    if not allow_download:
        plan["status"] = "planned_download_not_executed"
        plan["will_download"] = False
        plan["operator_boundary"] = _operator_boundary("planning_only")
        plan["next_actions"] = _append_unique(plan.get("next_actions"), "explicitly_execute_download_in_video_download_orchestrator")
        return plan

    raw = str(url_or_text or "").strip()
    root = Path(vdo_root).expanduser().resolve() if vdo_root else _default_vdo_root()
    vdo_plan_dir = Path(vdo_output_dir).expanduser().resolve() if vdo_output_dir else _project_root() / "openclaw-runs" / "download-plans" / "video-download-orchestrator"
    if not actor_id or not confirm_download:
        plan.update(
            {
                "ok": False,
                "status": "download_confirmation_required",
                "will_download": False,
                "operator_boundary": _operator_boundary("download_requires_explicit_confirmation"),
                "next_actions": [
                    "rerun_with_actor_id_and_confirm_download",
                    "or_run_video_download_orchestrator_openclaw_execute_manually",
                ],
                "commands": {
                    **(plan.get("commands") if isinstance(plan.get("commands"), dict) else {}),
                    "required_confirmed_execute": _vdo_execute_hint(raw, root, vdo_plan_dir),
                },
            }
        )
        return plan

    execute_command = [
        sys.executable,
        "-m",
        "video_orchestrator.cli",
        "openclaw-execute",
        raw,
        "--actor-id",
        actor_id,
        "--format",
        "json",
        "--confirm",
    ]
    if confirm_sensitive:
        execute_command.append("--confirm-sensitive")
    if backend:
        execute_command.extend(["--backend", backend])
    completed = _run_vdo_command(execute_command, root, timeout_seconds, runner=runner)
    execution_payload = _parse_json_stdout(completed.stdout) if completed.returncode == 0 else {}
    media = Path(downloaded_media_path).expanduser().resolve() if downloaded_media_path else _find_downloaded_media_path(execution_payload)
    result = _openclaw_result(
        ok=completed.returncode == 0,
        status="download_executed" if completed.returncode == 0 else "download_execute_failed",
        input_type=_classify_input(raw),
        will_download=True,
        download_plan=plan.get("download_plan") if isinstance(plan.get("download_plan"), dict) else {},
        media_path=str(media or ""),
        artifacts={
            **(plan.get("artifacts") if isinstance(plan.get("artifacts"), dict) else {}),
            "download_execution": execution_payload,
        },
        operator_boundary=_operator_boundary("download_explicitly_confirmed"),
        next_actions=["inspect_download_result", "run_openclaw_video_ingest"],
        errors=[] if completed.returncode == 0 else [_compact_process_error(completed)],
        commands={"host_execute": _powershell_command(execute_command, root)},
        docker=_docker_notes(media_path=str(media or "")),
    )
    if ingest_after_download and media and Path(media).exists():
        ingest = openclaw_video_ingest(
            media,
            workspace=workspace,
            title=title,
            max_frames=max_frames,
            sample_interval=sample_interval,
            sample_mode=sample_mode,
            timeout_seconds=timeout_seconds,
            prepare_runner=prepare_runner,
        )
        result["ingest"] = ingest
        result["workspace"] = ingest.get("workspace", "")
        result["review_url_or_file"] = ingest.get("review_url_or_file", "")
        result["next_actions"] = _append_unique(ingest.get("next_actions"), "review_download_and_ingest_logs")
    elif ingest_after_download:
        result["next_actions"] = _append_unique(result.get("next_actions"), "provide_downloaded_media_path_then_ingest")
    return result


def _run_vdo_command(command: list[str], vdo_root: Path, timeout_seconds: int, *, runner: Runner | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(vdo_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if runner:
        return runner(command, env, timeout_seconds)
    return subprocess.run(
        command,
        cwd=str(vdo_root),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _openclaw_result(
    *,
    ok: bool,
    status: str,
    input_type: str,
    will_download: bool,
    download_plan: dict[str, Any] | None = None,
    media_path: str = "",
    workspace: str = "",
    artifacts: dict[str, Any] | None = None,
    review_url_or_file: str = "",
    next_actions: list[str] | None = None,
    operator_boundary: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
    commands: dict[str, str] | None = None,
    docker: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "schema": "video_knowledge_pipeline.openclaw_integration.v1",
        "created_at": now_iso(),
        "ok": ok,
        "status": status,
        "input_type": input_type,
        "will_download": will_download,
        "download_plan": download_plan or {},
        "media_path": media_path,
        "workspace": workspace,
        "artifacts": artifacts or {},
        "review_url_or_file": review_url_or_file,
        "next_actions": next_actions or [],
        "operator_boundary": operator_boundary or {},
    }
    if errors:
        result["errors"] = errors
    if commands:
        result["commands"] = commands
    if docker:
        result["docker"] = docker
    result.update(extra)
    return result


def _classify_input(raw: str) -> str:
    if _URL_RE.fullmatch(raw.strip()):
        return "video_url"
    if _URL_RE.search(raw):
        return "telegram_text"
    suffix = Path(raw).suffix.lower()
    if suffix in _MEDIA_EXTENSIONS:
        return "local_video_path"
    return "telegram_text"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_vdo_root() -> Path:
    return workspace_root() / "video-download-orchestrator"


def _default_workspace_for(media: Path, workspace: str | Path) -> Path:
    if workspace:
        return Path(workspace).expanduser().resolve()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", media.stem).strip("-") or "video"
    return (_project_root() / "openclaw-runs" / safe_stem).resolve()


def _parse_json_stdout(stdout: str) -> dict[str, Any] | None:
    try:
        payload = extract_last_json_document(stdout, require_object=True)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _download_artifacts(payload: dict[str, Any], *, vdo_plan_dir: Path, vdo_root: Path) -> dict[str, Any]:
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    manifest_paths = []
    for item in items:
        if isinstance(item, dict) and item.get("manifest_path"):
            manifest_paths.append(str(item["manifest_path"]))
    return {
        "vdo_root": str(vdo_root),
        "vdo_output_dir": str(vdo_plan_dir),
        "vdo_batch_dir": str(payload.get("batch_dir") or ""),
        "download_manifest_paths": manifest_paths,
        "item_count": len(items),
    }


def _ingest_artifacts(report: dict[str, Any]) -> dict[str, Any]:
    initial_bundle = report.get("initial_bundle") if isinstance(report.get("initial_bundle"), dict) else {}
    return {
        "run_json": str(report.get("json_path") or ""),
        "run_markdown": str(report.get("markdown_path") or ""),
        "asr_plan": str((report.get("asr_plan") or {}).get("plan_path") if isinstance(report.get("asr_plan"), dict) else ""),
        "transcript_path": str(report.get("transcript_path") or ""),
        "bundle_dir": str(initial_bundle.get("bundle_dir") or ""),
        "review_html": str(initial_bundle.get("review_html") or ""),
        "manifest_path": str(initial_bundle.get("manifest_path") or ""),
    }


def _ingest_next_actions(report: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for step in report.get("next_steps") or []:
        if isinstance(step, dict) and step.get("key"):
            keys.append(str(step["key"]))
    return keys or ["inspect_video_knowledge_run"]


def _plan_next_actions(payload: dict[str, Any]) -> list[str]:
    actions = ["review_download_plan"]
    if payload.get("item_count") or payload.get("items"):
        actions.append("execute_download_in_video_download_orchestrator_after_confirmation")
        actions.append("then_run_openclaw_video_ingest_with_downloaded_media_path")
    else:
        actions.append("provide_supported_video_link_or_local_media_path")
    return actions


def _operator_boundary(kind: str) -> dict[str, Any]:
    boundaries = {
        "planning_only": {
            "kind": kind,
            "summary": "No real download is performed here. Link routing and download planning are delegated to video-download-orchestrator.",
            "requires_human_confirmation_for_download": True,
            "no_secrets_logged": True,
        },
        "local_ingest_only": {
            "kind": kind,
            "summary": "This only processes an existing local media file. ASR, vision API, ebook OCR, and exports keep their existing explicit execution gates.",
            "requires_human_confirmation_for_download": False,
            "no_secrets_logged": True,
        },
        "download_requires_explicit_confirmation": {
            "kind": kind,
            "summary": "Real download was requested but actor_id and confirm_download are required before calling video-download-orchestrator openclaw-execute.",
            "requires_human_confirmation_for_download": True,
            "no_secrets_logged": True,
        },
        "download_explicitly_confirmed": {
            "kind": kind,
            "summary": "Download execution was explicitly confirmed and delegated to video-download-orchestrator.",
            "requires_human_confirmation_for_download": True,
            "no_secrets_logged": True,
        },
    }
    return boundaries.get(kind, {"kind": kind, "no_secrets_logged": True})


def _docker_notes(*, media_path: str = "", workspace: str = "") -> dict[str, Any]:
    return {
        "openclaw_container_recommendation": "Mount VKP_WORKSPACE_ROOT into the OpenClaw container and translate mounted paths consistently before calling host tools.",
        "download_orchestrator_api": "For download planning from Docker, use VDO_API_BASE=http://host.docker.internal:8921 when the video-download-orchestrator HTTP bridge is running on the host.",
        "host_media_path": media_path,
        "host_workspace": workspace,
        "path_boundary": "The JSON returns Windows host paths. Docker callers must translate mounted paths consistently before passing them back to host-side tools.",
    }


def _vdo_execute_hint(raw: str, vdo_root: Path, output_dir: Path) -> str:
    command = [
        sys.executable,
        "-m",
        "video_orchestrator.cli",
        "openclaw-execute",
        raw,
        "--actor-id",
        "<telegram-user-id>",
        "--format",
        "json",
        "--confirm",
    ]
    return _powershell_command(command, vdo_root) + f"  # output-dir for planning artifacts: {output_dir}"


def _powershell_command(command: list[str], cwd: Path) -> str:
    quoted = " ".join(_ps_quote(part) for part in command)
    return f"Set-Location -LiteralPath {_ps_quote(str(cwd))}; $env:PYTHONPATH={_ps_quote(str(cwd / 'src'))}; {quoted}"



def _compact_process_error(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "returncode": completed.returncode,
        "stdout_excerpt": _excerpt(completed.stdout),
        "stderr_excerpt": _excerpt(completed.stderr),
    }


def _excerpt(value: str, limit: int = 1200) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _append_unique(values: Any, item: str) -> list[str]:
    result = [str(value) for value in values] if isinstance(values, list) else []
    if item not in result:
        result.append(item)
    return result


def _find_downloaded_media_path(payload: Any) -> Path | None:
    if not isinstance(payload, dict):
        return None
    candidates: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {"media_path", "output_file", "downloaded_file", "file_path", "path"} and isinstance(child, str):
                    candidates.append(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.suffix.lower() in _MEDIA_EXTENSIONS:
            return path.resolve()
    return None
