from __future__ import annotations

import re
from typing import Any

_ASCII_CJK_RE = re.compile(r"[^0-9A-Za-z一-鿿]+")
_LOWER_ASCII_CJK_RE = re.compile(r"[^0-9a-z一-鿿]+")


def compact_ascii_cjk(value: Any) -> str:
    """Keep source ASCII alphanumerics and CJK, then case-fold the result."""

    return _ASCII_CJK_RE.sub("", str(value or "")).casefold()


def compact_ascii_cjk_after_lowering(value: Any) -> str:
    """Lower first so Unicode case variants may normalize into kept ASCII."""

    return _LOWER_ASCII_CJK_RE.sub("", str(value or "").lower())