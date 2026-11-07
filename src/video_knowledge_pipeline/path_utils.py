from __future__ import annotations

from pathlib import Path


def file_uri_or_empty(path_text: str) -> str:
    """Return an absolute file URI, or an empty string when it cannot be built."""
    if not path_text:
        return ""
    try:
        return Path(path_text).expanduser().resolve().as_uri()
    except Exception:
        return ""
