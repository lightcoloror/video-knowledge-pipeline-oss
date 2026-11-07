from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline import artifact_freshness
from video_knowledge_pipeline.canonical_json import (
    canonical_json_bytes,
    canonical_json_sha256,
)


def test_canonical_json_preserves_existing_compact_unicode_contract() -> None:
    value = {"b": 2, "a": "课程"}
    assert canonical_json_bytes(value) == '{"a":"课程","b":2}'.encode()
    expected = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert canonical_json_sha256(value) == expected


def test_artifact_freshness_keeps_public_compatibility_export() -> None:
    assert artifact_freshness.canonical_json_sha256 is canonical_json_sha256


def test_no_module_embeds_compact_sorted_json_directly_in_sha256() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "video_knowledge_pipeline"
    direct_owners: set[str] = set()
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "sha256"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "hashlib"
            ):
                continue
            block = ast.get_source_segment(
                path.read_text(encoding="utf-8-sig"), node
            ) or ""
            if "json.dumps" in block and "sort_keys=True" in block and "separators=" in block:
                direct_owners.add(path.name)

    assert direct_owners == set()
