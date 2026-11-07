from __future__ import annotations

import os
import shutil
from pathlib import Path


KNOWN_FFMPEG_DIRS = (
    Path.home() / "AppData/Roaming/TRAE SOLO CN/ModularData/ai-agent/vm/tools/app/ffmpeg",
)
KNOWN_TESSERACT_DIRS = (
    Path.home() / "AppData/Roaming/TRAE SOLO CN/ModularData/ai-agent/vm/tools/app/tesseract",
)


def resolve_media_tool(name: str) -> str:
    """Resolve ffmpeg/ffprobe without requiring a global PATH install."""
    normalized = name.lower().removesuffix(".exe")
    if normalized not in {"ffmpeg", "ffprobe", "ffplay"}:
        raise ValueError(f"unsupported media tool: {name}")

    direct_env = {
        "ffmpeg": "FFMPEG_BINARY",
        "ffprobe": "FFPROBE_BINARY",
        "ffplay": "FFPLAY_BINARY",
    }[normalized]
    direct = _existing_file(os.environ.get(direct_env, ""))
    if direct:
        return direct

    for directory in _candidate_dirs():
        for filename in (normalized, f"{normalized}.exe"):
            candidate = directory / filename
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())

    return shutil.which(normalized) or ""


def media_tool_env() -> dict[str, str]:
    """Return environment values that make resolved media tools visible to children."""
    ffmpeg = resolve_media_tool("ffmpeg")
    ffprobe = resolve_media_tool("ffprobe")
    tesseract = resolve_tesseract()
    env: dict[str, str] = {}
    if ffmpeg:
        env["FFMPEG_BINARY"] = ffmpeg
    if ffprobe:
        env["FFPROBE_BINARY"] = ffprobe
    if tesseract:
        env["TESSERACT_BINARY"] = tesseract
        env["PEEPSHOW_TESSERACT"] = tesseract
    dirs = sorted({str(Path(path).parent) for path in (ffmpeg, ffprobe, tesseract) if path})
    if dirs:
        env["LECTURE_FFMPEG_DIR"] = dirs[0]
        existing_path = os.environ.get("PATH", "")
        env["PATH"] = os.pathsep.join([*dirs, existing_path]) if existing_path else os.pathsep.join(dirs)
    return env


def local_tool_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(media_tool_env())
    return env


def resolve_tesseract() -> str:
    direct = _existing_file(os.environ.get("TESSERACT_BINARY", ""))
    if direct:
        return direct
    direct = _existing_file(os.environ.get("PEEPSHOW_TESSERACT", ""))
    if direct:
        return direct
    for directory in _tesseract_candidate_dirs():
        for filename in ("tesseract", "tesseract.exe"):
            candidate = directory / filename
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())
    return shutil.which("tesseract") or ""


def _candidate_dirs() -> list[Path]:
    values = []
    env_dir = os.environ.get("LECTURE_FFMPEG_DIR", "").strip()
    if env_dir:
        values.append(Path(env_dir).expanduser())
    values.extend(KNOWN_FFMPEG_DIRS)
    return values


def _tesseract_candidate_dirs() -> list[Path]:
    values = []
    env_dir = os.environ.get("LECTURE_TESSERACT_DIR", "").strip()
    if env_dir:
        values.append(Path(env_dir).expanduser())
    values.extend(KNOWN_TESSERACT_DIRS)
    return values


def _existing_file(value: str | None) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.exists() and path.is_file():
        return str(path.resolve())
    found = shutil.which(value)
    return found or ""
