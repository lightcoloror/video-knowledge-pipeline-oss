from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.bundle_assets import repair_bundle_assets
from video_knowledge_pipeline.storage import read_json, write_json


def test_repair_external_absolute_asset_uses_portable_bundle_target(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "073_frame.jpg"
    source.write_bytes(b"frame")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    package_path = tmp_path / "lecture-package.json"
    write_json(
        package_path,
        {
            "timeline": [
                {
                    "index": 1,
                    "source_segment_ids": ["segment-1"],
                    "frame_paths": [str(source)],
                }
            ]
        },
    )
    write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "source_package": str(package_path),
        },
    )
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "source_segment_ids": ["segment-1"],
                "assets": [
                    {"source": str(source), "path": str(source), "copied": "false"}
                ],
            }
        ],
    )

    result = repair_bundle_assets(bundle)

    target = bundle / "assets" / "0001-073_frame.jpg"
    assert result["summary"]["copied"] == 1
    assert target.read_bytes() == b"frame"
    timeline = read_json(bundle / "timeline.json")
    assert timeline[0]["assets"][0]["path"] == "assets/0001-073_frame.jpg"
    assert timeline[0]["assets"][0]["copied"] == "true"
    manifest = read_json(bundle / "manifest.json")
    assert manifest["assets"][0]["path"] == "assets/0001-073_frame.jpg"
    package = read_json(package_path)
    assert package["timeline"][0]["frame_paths"] == [str(target)]
