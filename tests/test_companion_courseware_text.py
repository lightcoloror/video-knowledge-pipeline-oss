from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.companion_courseware_text import (
    import_companion_courseware_text,
    load_companion_courseware_text,
)
from video_knowledge_pipeline.knowledge_coverage import audit_knowledge_coverage
from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [{"index": 1, "start": 0, "end": 30, "transcript": "只有讲师画面。", "visual_route": "document_visual"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bundle


def test_companion_courseware_covers_screen_text_without_claiming_frame_ocr(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    source = tmp_path / "courseware.md"
    source.write_text("# 课程课件\n\n- 第一页：客户画像与信任。\n", encoding="utf-8")

    imported = import_companion_courseware_text(bundle, source, title="课程课件")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    loaded = load_companion_courseware_text(bundle, manifest)
    input_pack = build_smart_summary_input_pack(bundle, write=True)
    report = audit_knowledge_coverage(bundle, write=True)["coverage"]
    channels = {row["key"]: row for row in report["channels"]}

    assert imported["evidence_scope"] == "external_courseware_not_video_frame"
    assert imported["video_frame_ocr_required"] is False
    assert loaded is not None
    assert loaded["text"].startswith("# 课程课件")
    assert channels["screen_text"]["status"] == "covered_by_external_courseware"
    assert channels["structured_visual"]["status"] == "covered_by_external_courseware"
    assert channels["screen_text"]["blocker_count"] == 0
    assert report["companion_courseware"]["status"] == "covered_by_external_courseware"
    assert input_pack["companion_courseware"]["status"] == "covered_by_external_courseware"
    assert "客户画像与信任" in input_pack["companion_courseware"]["content_excerpt"]
    assert input_pack["companion_courseware"]["timestamp_mapping"] == "not_available"
