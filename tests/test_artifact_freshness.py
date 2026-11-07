from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.artifact_freshness import (
    build_dependency_snapshot,
    validate_dependency_snapshot,
)


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([{"index": 1, "text": "A"}]), encoding="utf-8")
    return bundle


def test_dependency_snapshot_detects_content_change_even_when_mtime_is_restored(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    timeline = bundle / "timeline.json"
    snapshot = build_dependency_snapshot(bundle, subject="bundle-export", inputs=[{"role": "timeline", "path": timeline}])
    original_mtime = timeline.stat().st_mtime

    timeline.write_text(json.dumps([{"index": 1, "text": "B"}]), encoding="utf-8")
    timeline.touch()
    import os

    os.utime(timeline, (original_mtime, original_mtime))
    result = validate_dependency_snapshot(bundle, snapshot)

    assert result["status"] == "stale"
    assert result["passed"] is False
    assert result["issues"][0]["key"] == "input_changed"
    assert "sha256" in result["issues"][0]["changed_fields"]


def test_dependency_snapshot_detects_missing_and_tampered_snapshot(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    snapshot = build_dependency_snapshot(bundle, subject="bundle-export", inputs=["timeline.json"])
    (bundle / "timeline.json").unlink()
    assert validate_dependency_snapshot(bundle, snapshot)["status"] == "missing"

    snapshot["subject"] = "tampered"
    assert validate_dependency_snapshot(bundle, snapshot)["status"] == "invalid"


def test_dependency_snapshot_rejects_artifact_outside_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside bundle"):
        build_dependency_snapshot(bundle, subject="bundle-export", inputs=[outside])
