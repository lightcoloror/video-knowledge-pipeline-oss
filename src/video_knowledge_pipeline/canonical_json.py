from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize with VKP's existing stable compact JSON contract."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash one value without changing VKP's established revision format."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
