"""Local source seals built from VKP's existing tested-entry fingerprint policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .storage import read_json, write_json
from .tested_entry import dependency_snapshot, python_identity, read_git_head, source_snapshot

SCHEMA = "video_knowledge_pipeline.server_release.v1"


def create_server_release(source_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Seal the actual tree and environment; this does not publish or adopt it."""
    root, target = Path(source_root).resolve(), Path(output_path).resolve()
    if target.exists():
        raise ValueError("release exists; preserve it and choose another path")
    if target.is_relative_to(root) and not target.is_relative_to(root / ".local"):
        raise ValueError("release metadata must stay outside managed source")
    value = {
        "schema": SCHEMA, "git_head": read_git_head(root), "source": source_snapshot(root),
        "python": python_identity(), "dependencies": dependency_snapshot(),
        "publication": "local_only_not_published_or_adopted",
        "dependency_scope": "exact validation environment; not a tested Linux installation lock",
        "execution_safety_v1": False, "real_provider_verified": False,
    }
    value["release_sha256"] = canonical_json_sha256(value)
    write_json(target, value)
    return value


def verify_server_release(source_root: str | Path, release: dict[str, Any] | str | Path) -> dict[str, Any]:
    value = read_json(Path(release)) if not isinstance(release, dict) else release
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("invalid release schema")
    payload = {k: v for k, v in value.items() if k != "release_sha256"}
    if canonical_json_sha256(payload) != value.get("release_sha256"):
        raise ValueError("release document drift")
    root = Path(source_root).resolve()
    if read_git_head(root) != value["git_head"]:
        raise ValueError("release Git HEAD drift")
    if source_snapshot(root)["sha256"] != value["source"]["sha256"]:
        raise ValueError("release source drift")
    if python_identity() != value["python"]:
        raise ValueError("release Python drift")
    if dependency_snapshot()["sha256"] != value["dependencies"]["sha256"]:
        raise ValueError("release dependency drift")
    return {"release_sha256": value["release_sha256"], "source_sha256": value["source"]["sha256"],
            "git_head": value["git_head"], "verified": True}
