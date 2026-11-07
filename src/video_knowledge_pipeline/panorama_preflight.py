from __future__ import annotations

import json
import importlib.util
import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Callable

from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .panorama_contract import build_panorama_profile


PANORAMA_PREFLIGHT_SCHEMA = "video_knowledge_pipeline.panorama_preflight.v1"
CommandRunner = Callable[[list[str]], CompletedProcess[str]]
ToolResolver = Callable[[str], str]


def panorama_doctor(
    media_or_bundle: str | Path | None,
    *,
    projection: str | None = None,
    input_fov: float | int | str | None = None,
    lens_order: str | None = None,
    ffprobe_payload: dict[str, Any] | None = None,
    tool_resolver: ToolResolver = resolve_media_tool,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Run a bounded, read-only panorama capability and projection preflight."""

    runner = command_runner or _run_command
    ffmpeg = tool_resolver("ffmpeg")
    ffprobe = tool_resolver("ffprobe")
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not ffmpeg:
        blockers.append(_issue("ffmpeg_missing", "FFmpeg is not available through the existing media tool resolver"))
    if not ffprobe:
        blockers.append(_issue("ffprobe_missing", "FFprobe is not available through the existing media tool resolver"))

    v360 = False
    if ffmpeg:
        try:
            filter_result = runner([ffmpeg, "-hide_banner", "-filters"])
            v360 = filter_result.returncode == 0 and "v360" in str(filter_result.stdout or "")
            if not v360:
                blockers.append(_issue("v360_unavailable", "the resolved FFmpeg does not report the v360 filter"))
        except Exception as exc:  # pragma: no cover - platform-specific subprocess failures
            blockers.append(_issue("v360_probe_failed", _bounded_message(exc)))

    media_path = _resolve_media_path(media_or_bundle)
    profile: dict[str, Any] | None = None
    probe_payload = dict(ffprobe_payload) if isinstance(ffprobe_payload, dict) else None
    if media_or_bundle is not None and media_path is None:
        blockers.append(_issue("media_missing", "media or Bundle source path does not exist"))
    elif media_path is not None:
        if probe_payload is None and ffprobe:
            try:
                probe_result = runner(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-show_streams",
                        "-show_format",
                        "-of",
                        "json",
                        str(media_path),
                    ]
                )
                if probe_result.returncode != 0:
                    blockers.append(_issue("ffprobe_failed", str(probe_result.stderr or "ffprobe failed")[:500]))
                else:
                    try:
                        parsed = json.loads(str(probe_result.stdout or ""))
                        probe_payload = parsed if isinstance(parsed, dict) else None
                        if probe_payload is None:
                            blockers.append(_issue("ffprobe_invalid_json", "FFprobe output must be a JSON object"))
                    except json.JSONDecodeError:
                        blockers.append(_issue("ffprobe_invalid_json", "FFprobe returned invalid JSON"))
            except Exception as exc:  # pragma: no cover - platform-specific subprocess failures
                blockers.append(_issue("ffprobe_failed", _bounded_message(exc)))

        profile = build_panorama_profile(
            projection=projection,
            source_path=media_path,
            ffprobe_payload=probe_payload,
            input_fov=input_fov,
            lens_order=lens_order,
        )
        for issue in profile.get("issues") or []:
            if isinstance(issue, dict):
                blockers.append(_issue(str(issue.get("key") or "projection_invalid"), str(issue.get("message") or "")))
        if profile.get("aspect_ratio_hint") and not profile.get("ready"):
            warnings.append(
                _issue(
                    "aspect_ratio_not_authoritative",
                    "a 2:1 frame ratio is only a hint and cannot confirm a spherical projection",
                )
            )

    blockers = _dedupe_issues(blockers)
    tool_ready = bool(ffmpeg and ffprobe and v360) and not any(
        row["key"] in {"ffmpeg_missing", "ffprobe_missing", "v360_unavailable", "v360_probe_failed"}
        for row in blockers
    )
    ready_to_prepare = bool(tool_ready and media_path is not None and profile and profile.get("ready") and not blockers)
    recovery_projection = str(projection or "<type>")
    recovery_target = str(media_path or media_or_bundle or "<media-or-bundle>")
    recovery_command = f'panorama-prepare "{recovery_target}" --projection {recovery_projection}'

    return {
        "schema": PANORAMA_PREFLIGHT_SCHEMA,
        "ok": bool(tool_ready and not [row for row in blockers if row["key"] not in {"projection_required"}]),
        "ready_to_prepare": ready_to_prepare,
        "capabilities": {
            "ffmpeg": bool(ffmpeg),
            "ffprobe": bool(ffprobe),
            "v360": v360,
            "projection_backend": "ffmpeg_v360" if v360 else "unavailable",
            "optional_workers": {
                "py360convert": "available" if importlib.util.find_spec("py360convert") else "unavailable_not_installed",
                "equilib": "available" if importlib.util.find_spec("equilib") else "unavailable_not_installed",
            },
        },
        "profile": profile,
        "blockers": blockers,
        "warnings": warnings,
        "provider_health_check": "not_applicable",
        "unattended_execution": False,
        "recovery_command": recovery_command,
        "write_performed": False,
    }


def _resolve_media_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_dir():
        return None
    manifest_path = candidate / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    for key in ("video_key", "media_path", "video_path", "source_path"):
        raw = str(manifest.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = candidate / path
        if path.is_file():
            return path.resolve()
    return None


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=local_tool_subprocess_env(),
    )


def _issue(key: str, message: str) -> dict[str, str]:
    return {"key": key, "message": message}


def _dedupe_issues(values: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        key = str(value.get("key") or "")
        if key and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _bounded_message(value: object) -> str:
    return str(value or "unknown subprocess error")[:500]
