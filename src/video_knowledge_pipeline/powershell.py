from __future__ import annotations

import subprocess
from pathlib import Path

from .media_tools import local_tool_subprocess_env

_ARGUMENT_META_CHARACTERS = "'\"`$&|<>()[]{};"


def quote_powershell_literal(value: object) -> str:
    """Render one value as a PowerShell single-quoted string literal."""

    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def quote_powershell_argument(value: object) -> str:
    """Quote a command argument when the existing VKP contract requires it."""

    text = str(value)
    if not text:
        return "''"
    if any(char.isspace() for char in text) or any(
        char in text for char in _ARGUMENT_META_CHARACTERS
    ):
        return quote_powershell_literal(text)
    return text


def run_powershell_command(
    command: str,
    *,
    cwd: str | Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run one existing VKP PowerShell command with the established process contract."""

    timeout = int(timeout_seconds or 0) or None
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
        env=local_tool_subprocess_env(),
    )
