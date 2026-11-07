from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .file_hash import sha256_file


DEFAULT_ALLOWED_ROOTS_ENV = "VKP_MODEL_RUNTIME_ALLOWED_ROOTS"


def normalise_allowed_roots(
    values: list[str | Path] | tuple[str | Path, ...] | None,
    *,
    env_var: str = DEFAULT_ALLOWED_ROOTS_ENV,
    default_root: str | Path | None = None,
) -> tuple[Path, ...]:
    if values is None:
        configured = str(os.environ.get(env_var) or "").strip()
        if configured:
            values = [part for part in configured.split(os.pathsep) if part.strip()]
        else:
            values = [default_root or Path(__file__).resolve().parents[2]]
    roots = tuple(Path(value).expanduser().resolve() for value in values)
    if not roots:
        raise ValueError("artifact validation requires at least one allowed root")
    return roots


def validated_local_file(
    value: str | Path,
    *,
    label: str,
    allowed_roots: tuple[Path, ...],
) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} is required")
    lowered = raw.lower()
    if lowered.startswith(("data:", "http://", "https://")) or ";base64," in lowered:
        raise ValueError(f"{label} must be a legal local filesystem path")
    path = Path(raw).expanduser().resolve()
    if not any(path == root or path.is_relative_to(root) for root in allowed_roots):
        raise ValueError(f"{label} is outside allowed roots: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def artifact_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
