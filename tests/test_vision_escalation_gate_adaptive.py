from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.vision_review_triage import vision_review_triage


def _temporal_bundle(
    tmp_path: Path,
    *,
    transcript: str,
    presenter_region: dict[str, float] | None = None,
    include_temporal_frames: bool = True,
) -> Path:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "before.png"
    second = assets / "after.png"
    Image.new("RGB", (160, 90), "white").save(first)
    changed = Image.new("RGB", (160, 90), "white")
    ImageDraw.Draw(changed).rectangle((0, 0, 31, 31), fill="black")
    changed.save(second)
    item: dict[str, object] = {
        "index": 1,
        "start": 0,
        "end": 5,
        "visual_route": "temporal_sequence",
        "transcript": transcript,
        "visual_text": "课程内容",
        "frame_paths": [str(first)],
    }
    if include_temporal_frames:
        item["temporal_frame_paths"] = [str(first), str(second)]
    if presenter_region:
        item["presenter_region"] = presenter_region
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", [item])
    return bundle


def test_missing_temporal_frames_require_local_recapture_before_multimodal(tmp_path: Path) -> None:
    bundle = _temporal_bundle(
        tmp_path,
        transcript="现在点击提交按钮。",
        include_temporal_frames=False,
    )

    result = vision_review_triage(bundle, write=False)

    assert result["temporal_indexes"] == []
    assert result["temporal_recapture_indexes"] == [1]
    row = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert row["frame_change_evidence"]["status"] == "not_available"
    assert row["local_prerequisite_action"] == "capture_temporal_frames"
    assert "temporal_evidence_missing_requires_recapture" in row["suppression_reasons"]


def test_localized_motion_at_arbitrary_position_is_not_assumed_to_be_presenter(tmp_path: Path) -> None:
    bundle = _temporal_bundle(tmp_path, transcript="普通讲解。")

    result = vision_review_triage(bundle, write=False)

    assert result["temporal_indexes"] == []
    row = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert row["frame_change_evidence"]["status"] == "localized_motion"
    assert row["frame_change_evidence"]["presenter_region_masked"] is False
    assert "localized_motion_without_operation_evidence" in row["suppression_reasons"]


def test_localized_operation_change_stays_temporal(tmp_path: Path) -> None:
    bundle = _temporal_bundle(tmp_path, transcript="现在点击左上角按钮。")

    result = vision_review_triage(bundle, write=False)

    assert result["temporal_indexes"] == [1]
    assert result["temporal_candidates"][0]["frame_change_evidence"]["status"] == "localized_motion"


def test_explicit_presenter_region_can_be_any_position_and_size(tmp_path: Path) -> None:
    bundle = _temporal_bundle(
        tmp_path,
        transcript="现在点击提交按钮。",
        presenter_region={"x": 0.0, "y": 0.0, "width": 0.25, "height": 0.45},
    )

    result = vision_review_triage(bundle, write=False)

    assert result["temporal_indexes"] == []
    row = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert row["frame_change_evidence"]["status"] == "explicit_overlay_only"
    assert row["frame_change_evidence"]["presenter_region_masked"] is True
    assert row["frame_change_evidence"]["overlay_region_source"] == "explicit_metadata"


def test_duplicate_suppression_keeps_stronger_later_evidence(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "same-1.png"
    second = assets / "same-2.png"
    Image.new("RGB", (160, 90), "white").save(first)
    Image.new("RGB", (160, 90), "white").save(second)
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "visual_route": "semantic_frame",
                "transcript": "这里是普通介绍。",
                "visual_text": "",
                "frame_paths": [str(first)],
            },
            {
                "index": 2,
                "start": 5,
                "end": 10,
                "visual_route": "semantic_frame",
                "transcript": "这里价格是 399 元。",
                "visual_text": "价格 299 元",
                "frame_paths": [str(second)],
            },
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["semantic_indexes"] == [2]
    first_row = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert first_row["duplicate_of_index"] == 2
    assert first_row["local_prerequisite_action"] == "none"
