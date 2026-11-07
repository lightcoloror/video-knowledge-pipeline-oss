from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.task_console import export_task_console


def _minimal_bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Storage preset fixture",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    return bundle


def test_task_console_exposes_two_non_destructive_storage_presets(
    tmp_path: Path,
) -> None:
    console = export_task_console(_minimal_bundle(tmp_path), write=False, refresh=False)
    commands = {
        row["key"]: row
        for row in console["commands"]
        if isinstance(row, dict) and row.get("key")
    }

    space_saving = commands["media_equivalence_space_saving"]
    archival = commands["media_equivalence_archive_lossless"]

    assert "--policy space_saving" in space_saving["command"]
    assert "默认节省空间" in space_saving["label"]
    assert space_saving["safety"] == "local_read_only"
    assert "--policy archive_lossless" in archival["command"]
    assert "档案级绝不降质" in archival["label"]
    assert archival["safety"] == "local_read_only"
    assert "delete" not in space_saving["command"].casefold()
    assert "delete" not in archival["command"].casefold()
