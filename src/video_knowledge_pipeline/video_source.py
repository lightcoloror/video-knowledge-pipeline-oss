from __future__ import annotations

from .config import DEFAULT_LOCAL_FRAME_BUDGET
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .model_json import extract_last_json_document
from .models import now_iso
from .path_defaults import workspace_root
from .storage import write_json


DEFAULT_ORCHESTRATOR_ROOT = workspace_root() / "video-download-orchestrator"
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts",
                    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac"}


def prepare_video_source(
    url: str,
    output_dir: str | Path,
    *,
    execute: bool = False,
    backend: str | None = None,
    orchestrator_root: str | Path | None = None,
    python: str | None = None,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """Prepare a local video file or plan/execute a URL download."""
    local = Path(url).expanduser()
    if local.exists() and local.is_file():
        return prepare_local_video_source(local, output_dir, copy_media=execute)

    return prepare_remote_video_source(
        url,
        output_dir,
        execute=execute,
        backend=backend,
        orchestrator_root=orchestrator_root,
        python=python,
        timeout_seconds=timeout_seconds,
    )


def prepare_local_video_source(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    copy_media: bool = False,
) -> dict[str, Any]:
    media = Path(media_path).expanduser().resolve()
    if not media.exists() or not media.is_file():
        raise FileNotFoundError(f"local media not found: {media}")
    if media.suffix.lower() not in MEDIA_EXTENSIONS:
        raise ValueError(f"unsupported local media extension: {media.suffix}")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    local_media = media
    copied_to = ""
    if copy_media:
        target = out / "source-media" / media.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != media:
            shutil.copy2(media, target)
        local_media = target.resolve()
        copied_to = str(local_media)
    payload = {
        "schema": "lecture_video_source.v1",
        "source_kind": "local_file",
        "url": "",
        "source_path": str(media),
        "execute": bool(copy_media),
        "status": "ready",
        "will_download": False,
        "copied_to": copied_to,
        "output_dir": str(out),
        "local_media_path": str(local_media),
        "local_media_candidates": [{"path": str(local_media), "size": local_media.stat().st_size, "mtime": local_media.stat().st_mtime}],
        "created_at": now_iso(),
        "next_action": {"key": "plan_asr", "hint": "本地视频已登记；下一步运行 plan-asr 或 prepare-local-video-run 生成完整交接报告。"},
    }
    return _write_source_outputs(out, payload)


def prepare_remote_video_source(
    url: str,
    output_dir: str | Path,
    *,
    execute: bool = False,
    backend: str | None = None,
    orchestrator_root: str | Path | None = None,
    python: str | None = None,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """Plan or execute a URL download through video-download-orchestrator."""
    root = Path(orchestrator_root or os.environ.get("LECTURE_VIDEO_ORCHESTRATOR_ROOT") or DEFAULT_ORCHESTRATOR_ROOT)
    src = root / "src"
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [
        python or sys.executable,
        "-m",
        "video_orchestrator.cli",
        "download",
        url,
        "--output-dir",
        str(out),
        "--format",
        "json",
    ]
    if not execute:
        command.insert(5, "--dry-run")
    if backend:
        command.extend(["--backend", backend])
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_pythonpath(str(src), env.get("PYTHONPATH", ""))

    payload = {
        "schema": "lecture_video_source.v1",
        "url": url,
        "execute": bool(execute),
        "status": "planned",
        "will_download": bool(execute),
        "orchestrator_root": str(root),
        "output_dir": str(out),
        "command": command,
        "created_at": now_iso(),
    }
    try:
        result = subprocess.run(
            command,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        payload.update({"status": "timeout", "returncode": None, "stdout": exc.stdout or "", "stderr": exc.stderr or ""})
        return _write_source_outputs(out, payload)

    payload.update({"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    parsed = _parse_json_stdout(result.stdout)
    payload["orchestrator_result"] = parsed
    if result.returncode != 0:
        payload["status"] = "failed"
        return _write_source_outputs(out, payload)

    payload["status"] = str(parsed.get("status") or ("finished" if execute else "planned"))
    payload["manifest_path"] = str(parsed.get("manifest_path") or ((parsed.get("artifacts") or {}).get("manifest_path") if isinstance(parsed.get("artifacts"), dict) else ""))
    payload["report_path"] = str((parsed.get("artifacts") or {}).get("report_path") if isinstance(parsed.get("artifacts"), dict) else "")
    payload["review_checklist_path"] = str((parsed.get("artifacts") or {}).get("review_checklist_path") if isinstance(parsed.get("artifacts"), dict) else "")
    media = _find_downloaded_media(out) if execute else {}
    payload["local_media_path"] = str(media.get("path") or "")
    payload["local_media_candidates"] = media.get("candidates") or []
    payload["next_action"] = _next_action(payload)
    return _write_source_outputs(out, payload)


def prepare_lecture_workspace_from_url(
    project: str | Path,
    url: str,
    *,
    title: str,
    topic: str | None = None,
    output_root: str | Path | None = None,
    download_output_dir: str | Path | None = None,
    execute: bool = False,
    backend: str | None = None,
    orchestrator_root: str | Path | None = None,
    python: str | None = None,
    asr_preset: str = "funasr",
    language: str = "zh",
    model: str | None = None,
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    fps: float = 1.0,
    target: str = "bilinote",
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    from .lecture_pipeline import prepare_lecture_workspace

    out_root = Path(output_root).expanduser().resolve() if output_root else Path(project).expanduser().resolve() / "lecture-packages" / "source-downloads"
    download_dir = Path(download_output_dir).expanduser().resolve() if download_output_dir else out_root / _safe_source_dir(url)
    source = prepare_video_source(
        url,
        download_dir,
        execute=execute,
        backend=backend,
        orchestrator_root=orchestrator_root,
        python=python,
        timeout_seconds=timeout_seconds,
    )
    media = str(source.get("local_media_path") or "")
    if not execute or not media:
        return {
            "schema": "lecture_workspace_from_url.v1",
            "status": "download_planned" if not execute else "download_finished_without_media",
            "project": str(project),
            "url": url,
            "title": title,
            "source": source,
            "workspace": None,
            "next_action": source.get("next_action"),
        }
    workspace = prepare_lecture_workspace(
        project,
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
        vault=vault,
        folder=folder,
        source_provenance=source,
    )
    return {
        "schema": "lecture_workspace_from_url.v1",
        "status": "workspace_prepared",
        "project": str(project),
        "url": url,
        "title": title,
        "source": source,
        "workspace": workspace,
    }


def _write_source_outputs(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    json_path = output_dir / "video-source-provenance.json"
    markdown_path = output_dir / "video-source-provenance.md"
    payload["provenance_json_path"] = str(json_path)
    payload["provenance_markdown_path"] = str(markdown_path)
    write_json(json_path, payload)
    markdown_path.write_text(_render_source_markdown(payload), encoding="utf-8")
    return payload


def _render_source_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Video Source Provenance",
        "",
        f"- URL: `{payload.get('url', '')}`",
        f"- Status: `{payload.get('status', '')}`",
        f"- Execute: `{payload.get('execute', False)}`",
        f"- Output: `{payload.get('output_dir', '')}`",
        f"- Local media: `{payload.get('local_media_path', '')}`",
        f"- Manifest: `{payload.get('manifest_path', '')}`",
        f"- Report: `{payload.get('report_path', '')}`",
        f"- Review checklist: `{payload.get('review_checklist_path', '')}`",
        "",
        "## Command",
        "",
        "```text",
        " ".join(str(part) for part in payload.get("command") or []),
        "```",
        "",
        "## Next Action",
        "",
        str((payload.get("next_action") or {}).get("hint") or ""),
    ]
    candidates = payload.get("local_media_candidates") if isinstance(payload.get("local_media_candidates"), list) else []
    if candidates:
        lines.extend(["", "## Media Candidates", "", "| Size | Path |", "|---:|---|"])
        for row in candidates:
            if isinstance(row, dict):
                lines.append(f"| {row.get('size', 0)} | `{row.get('path', '')}` |")
    return "\n".join(lines).rstrip() + "\n"


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    try:
        data = extract_last_json_document(stdout)
    except ValueError:
        return {} if not stdout.strip() else {"raw_stdout": stdout}
    return data if isinstance(data, dict) else {"result": data}


def _find_downloaded_media(output_dir: Path) -> dict[str, Any]:
    candidates = []
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if ".vdo" in path.parts or path.name.endswith(".part"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        candidates.append({"path": str(path.resolve()), "size": stat.st_size, "mtime": stat.st_mtime})
    candidates.sort(key=lambda row: (-int(row.get("size") or 0), -float(row.get("mtime") or 0), str(row.get("path") or "")))
    return {"path": candidates[0]["path"] if candidates else "", "candidates": candidates[:10]}


def _next_action(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") == "failed":
        return {"key": "inspect_download_failure", "hint": "查看 stderr、manifest、report 和 review-checklist 后再重试下载。"}
    if not payload.get("execute"):
        return {"key": "execute_download", "hint": "确认 dry-run manifest 后，用 execute=true 或 CLI --execute 真实下载。"}
    if not payload.get("local_media_path"):
        return {"key": "locate_downloaded_media", "hint": "下载命令结束但未发现视频文件；检查 report、输出目录和后端命令。"}
    return {"key": "prepare_lecture_workspace", "hint": "把 local_media_path 交给 prepare_lecture_workspace 进入课程抽取流程。"}


def _prepend_pythonpath(path: str, existing: str) -> str:
    return path if not existing else path + os.pathsep + existing


def _safe_source_dir(value: str) -> str:
    stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80].strip("_")
    return stem or "video-source"
