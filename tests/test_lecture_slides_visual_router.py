from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.video_frame_router import run_video_frame_router


def _write_bundle(root: Path, timeline: list[dict]) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False), encoding="utf-8"
    )


def test_plain_frame_is_not_automatically_promoted_to_semantic_model(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "plain-frame"
    _write_bundle(
        bundle,
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "frame_paths": ["assets/frame.jpg"],
                "transcript": "讲师继续说明课程内容。",
            }
        ],
    )

    result = run_video_frame_router(bundle, content_profile="general")

    assert result["summary"]["routes"]["unknown"] == 1
    assert result["summary"]["routes"]["semantic_frame"] == 0


def test_lecture_slides_profile_routes_static_screen_text_to_ocr_first(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "slides"
    _write_bundle(
        bundle,
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "frame_paths": ["assets/slide.jpg"],
                "material_types": ["screen"],
                "visual_text": "课程标题和一段已经提取出来的课件正文文字",
                "transcript": "这一页介绍方法。",
            }
        ],
    )

    result = run_video_frame_router(
        bundle, content_profile="lecture-slides-v1"
    )

    assert result["summary"]["content_profile"] == "lecture-slides-v1"
    assert result["summary"]["routes"]["document_visual"] == 1
    assert result["summary"]["routes"]["semantic_frame"] == 0
    assert "lecture_slides_ocr_first" in result["items"][0]["reasons"]


def test_lecture_slides_profile_keeps_gesture_as_semantic_candidate(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "slides-gesture"
    _write_bundle(
        bundle,
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "frame_paths": ["assets/slide.jpg"],
                "material_types": ["screen"],
                "transcript": "讲师用动作指向图中的空间关系。",
            }
        ],
    )

    result = run_video_frame_router(
        bundle, content_profile="lecture-slides-v1"
    )

    assert result["summary"]["routes"]["semantic_frame"] == 1
