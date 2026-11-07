from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.video_workbench import export_video_workbench


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_workbench_embeds_fixed_wavesurfer_regions_and_local_review_controls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(root / "manifest.json", {"title": "shot review workbench"})
    _write_json(root / "timeline.json", [])
    _write_json(
        root / "exports" / "technical-shot-boundaries.json",
        {
            "schema": "video_knowledge_pipeline.technical_shot_boundaries.v1",
            "status": "completed",
            "ok": True,
            "boundary_kind": "technical_shot",
            "backend": "autoshot",
            "shots": [
                {"shot_id": "technical-shot-0001", "index": 1, "start": 0.0, "end": 1.0},
                {"shot_id": "technical-shot-0002", "index": 2, "start": 1.0, "end": 2.0},
            ],
        },
    )

    result = export_video_workbench(root, write=True)

    assert result["shot_review"]["status"] == "ready"
    assert result["shot_review"]["asset_source"]["version"] == "7.12.11"
    assert result["shot_review"]["asset_source"]["commit"] == "ae8d3cd32ebb27273051935c01fc6e4001cde3af"
    assert (root / "assets" / "wavesurfer-7.12.11" / "wavesurfer.min.js").is_file()
    assert (root / "assets" / "wavesurfer-7.12.11" / "regions.min.js").is_file()
    page = (root / "video-workbench.html").read_text(encoding="utf-8")
    assert "class SingleRegion" not in page  # implementation remains in pinned local asset
    assert "wavesurfer-7.12.11/wavesurfer.min.js" in page
    assert "wavesurfer-7.12.11/regions.min.js" in page
    assert "镜头边界与镜头语言复核" in page
    assert "保存到 VKP" in page
    glue = (root / "assets" / "shot-review-workbench.js").read_text(encoding="utf-8")
    assert "localStorage" in glue
    assert "VKP_SHOT_REVIEW_API" in glue
    assert "region-updated" in glue
    assert "shot-review-notes.json" in glue
