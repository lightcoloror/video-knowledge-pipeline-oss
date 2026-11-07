from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.vision_review_triage import vision_review_triage


def test_fresh_triage_controls_visual_coverage_requirements(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    timeline = [
        {"index": 1, "visual_route": "semantic_frame"},
        {
            "index": 2,
            "visual_route": "semantic_frame",
            "visual_understanding": {
                "objects": ["slide"],
                "evidence_frame_paths": ["frame-2.jpg"],
            },
        },
        {
            "index": 3,
            "visual_route": "temporal_sequence",
            "temporal_visual_understanding": {
                "state_changes": ["slide changed"],
                "evidence_frame_paths": ["frame-3a.jpg", "frame-3b.jpg"],
            },
        },
    ]
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", timeline)
    write_json(
        bundle / "vision-review-triage.json",
        {
            "status": "ok",
            "mode": "triage",
            "total_items": 3,
            "semantic_indexes": [],
            "temporal_indexes": [],
            "visual_structure_first_indexes": [],
            "selected_counts": {"suppressed": 1},
        },
    )

    report = build_knowledge_coverage(
        {"schema": "lecture_webui_bundle.v1"},
        timeline,
        bundle_dir=bundle,
    )

    assert report["visual_requirement_policy"]["mode"] == "risk_based_triage"
    assert report["semantic_frame_without_analysis"] == 0
    assert report["temporal_sequence_without_analysis"] == 0
    channels = {row["key"]: row for row in report["channels"]}
    assert channels["semantic_frame_understanding"]["status"] == "ok"
    assert channels["semantic_frame_understanding"]["expected_count"] == 1
    assert channels["temporal_visual_understanding"]["status"] == "ok"
    assert channels["temporal_visual_understanding"]["expected_count"] == 1


def test_mismatched_triage_falls_back_to_full_route_requirements(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    timeline = [{"index": 1, "visual_route": "semantic_frame"}]
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", timeline)
    write_json(
        bundle / "vision-review-triage.json",
        {
            "status": "ok",
            "mode": "triage",
            "total_items": 99,
            "semantic_indexes": [],
            "temporal_indexes": [],
            "visual_structure_first_indexes": [],
        },
    )

    report = build_knowledge_coverage(
        {"schema": "lecture_webui_bundle.v1"},
        timeline,
        bundle_dir=bundle,
    )

    assert report["visual_requirement_policy"] == {
        "mode": "route_full_legacy",
        "fresh": False,
    }
    assert report["semantic_frame_without_analysis"] == 1


def test_static_temporal_sequence_is_suppressed_after_local_diff(
    tmp_path: Path,
) -> None:
    from PIL import Image

    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "static-1.png"
    second = assets / "static-2.png"
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
                "visual_route": "temporal_sequence",
                "transcript": "现在点击提交按钮。",
                "visual_text": "提交",
                "structured_visual": {"type": "slide"},
                "ocr_confidence": 0.97,
                "temporal_frame_paths": [str(first), str(second)],
            }
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["temporal_indexes"] == []
    row = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert row["frame_change_evidence"]["status"] == "static"
    assert (
        "static_sequence_suppresses_temporal_multimodal" in row["suppression_reasons"]
    )
    assert row["estimated_model_calls"] == 0


def test_changed_temporal_sequence_and_scene_boundary_stay_temporal(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    exports = bundle / "exports"
    assets.mkdir(parents=True)
    exports.mkdir()
    first = assets / "before.png"
    second = assets / "after.png"
    Image.new("RGB", (160, 90), "white").save(first)
    changed = Image.new("RGB", (160, 90), "white")
    ImageDraw.Draw(changed).rectangle((10, 10, 110, 70), fill="black")
    changed.save(second)
    write_json(
        exports / "scene-detection.json",
        {"boundaries": [{"seconds": 2.5, "reason": "pyscenedetect_scene_boundary"}]},
    )
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "visual_route": "temporal_sequence",
                "transcript": "打开页面后切换到结果。",
                "visual_text": "结果",
                "temporal_frame_paths": [str(first), str(second)],
            }
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["temporal_indexes"] == [1]
    row = result["temporal_candidates"][0]
    assert row["frame_change_evidence"]["status"] == "dynamic"
    assert row["scene_boundary_evidence"]["matched"] is True
    assert row["estimated_images"] == 2
    assert row["estimated_model_calls"] == 1


def test_complex_layout_routes_to_semantic_even_with_good_ocr(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    frame = bundle / "frame.png"
    frame.write_bytes(b"frame")
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "visual_route": "document_visual",
                "transcript": "比较左边和右边两个方案的箭头关系。",
                "visual_text": "方案 A 方案 B",
                "structured_visual": {
                    "type": "diagram",
                    "regions": [{"role": "left"}, {"role": "right"}],
                },
                "ocr_confidence": 0.98,
                "material_types": ["diagram", "multi_column"],
                "frame_paths": [str(frame)],
            }
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["semantic_indexes"] == [1]
    row = result["semantic_candidates"][0]
    assert "complex_relational_layout" in row["benefit_reasons"]
    assert row["ocr_evidence"]["confidence"] == 0.98
    assert row["recommended_execution_location"] == "local_preferred_remote_approved"


def test_adjacent_repeated_pages_are_deduplicated(tmp_path: Path) -> None:
    from PIL import Image

    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "page-1.png"
    second = assets / "page-2.png"
    Image.new("RGB", (160, 90), "white").save(first)
    Image.new("RGB", (160, 90), "white").save(second)
    rows = []
    for index, path in enumerate((first, second), start=1):
        rows.append(
            {
                "index": index,
                "start": (index - 1) * 5,
                "end": index * 5,
                "visual_route": "semantic_frame",
                "transcript": "这里的价格是 399 元。",
                "visual_text": "价格 299 元",
                "frame_paths": [str(path)],
            }
        )
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", rows)

    result = vision_review_triage(bundle, write=False)

    assert result["semantic_indexes"] == [1]
    duplicate = next(
        row for row in result["all_ranked_candidates"] if row["index"] == 2
    )
    assert duplicate["duplicate_of_index"] == 1
    assert duplicate["recommended_action"] == "none"
    assert result["selected_counts"]["suppressed"] == 1


def test_resolved_ocr_frame_suppresses_adjacent_unresolved_visual_duplicate(
    tmp_path: Path,
) -> None:
    from PIL import Image

    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "unresolved.png"
    second = assets / "resolved.png"
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
                "visual_route": "document_visual",
                "transcript": "继续说明方案。",
                "frame_paths": [str(first)],
            },
            {
                "index": 2,
                "start": 6,
                "end": 11,
                "visual_route": "semantic_frame",
                "transcript": "比较左右两种方案。",
                "visual_text": "方案 A 方案 B",
                "structured_visual": {
                    "type": "diagram",
                    "regions": [{"role": "left"}, {"role": "right"}],
                },
                "material_types": ["diagram", "multi_column"],
                "ocr_confidence": 0.98,
                "frame_paths": [str(second)],
            },
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["semantic_indexes"] == [2]
    assert result["visual_structure_first_indexes"] == []
    duplicate = next(
        row for row in result["all_ranked_candidates"] if row["index"] == 1
    )
    assert duplicate["duplicate_of_index"] == 2
    assert duplicate["recommended_action"] == "none"
    assert "resolved_ocr_near_duplicate_of_index_2" in duplicate["suppression_reasons"]


def test_completed_semantic_analysis_is_not_selected_again(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "visual_route": "semantic_frame",
                "transcript": "这里展示 399 元的左右关系。",
                "visual_text": "方案 A：299 元；方案 B：299 元",
                "structured_visual": {"type": "diagram"},
                "material_types": ["diagram", "multi_column"],
                "frame_paths": ["missing-but-declared.png"],
                "visual_understanding": {
                    "validation_status": "ok",
                    "spatial_relations": ["方案 A 位于左侧，方案 B 位于右侧"],
                    "evidence_frame_paths": ["missing-but-declared.png"],
                },
            }
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["semantic_indexes"] == []
    row = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert row["has_valid_visual_understanding"] is True
    assert row["duplicate_of_index"] is None
    assert "semantic_analysis_already_available" in row["suppression_reasons"]
    assert row["estimated_model_calls"] == 0


def test_completed_temporal_analysis_is_not_selected_or_recaptured(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "before.png"
    second = assets / "after.png"
    Image.new("RGB", (160, 90), "white").save(first)
    changed = Image.new("RGB", (160, 90), "white")
    ImageDraw.Draw(changed).rectangle((10, 10, 110, 70), fill="black")
    changed.save(second)
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "visual_route": "temporal_sequence",
                "transcript": "打开页面后切换到结果。",
                "visual_text": "结果",
                "temporal_frame_paths": [str(first), str(second)],
                "temporal_visual_understanding": {
                    "validation_status": "ok",
                    "state_changes": ["页面从输入态切换为结果态"],
                    "evidence_frame_paths": [str(first), str(second)],
                },
            }
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["temporal_indexes"] == []
    assert result["temporal_recapture_indexes"] == []
    row = next(row for row in result["all_ranked_candidates"] if row["index"] == 1)
    assert row["has_valid_temporal_visual_understanding"] is True
    assert "temporal_analysis_already_available" in row["suppression_reasons"]
    assert row["estimated_model_calls"] == 0


def test_completed_semantic_analysis_suppresses_identical_slide_beyond_time_window(
    tmp_path: Path,
) -> None:
    from PIL import Image

    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    first = assets / "first.png"
    second = assets / "second.png"
    Image.new("RGB", (160, 90), "white").save(first)
    Image.new("RGB", (160, 90), "white").save(second)
    shared_text = "方案 A 位于左侧，方案 B 位于右侧，并由箭头连接"
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "visual_route": "semantic_frame",
                "transcript": "先看这张关系图。",
                "visual_text": shared_text,
                "structured_visual": {"type": "diagram"},
                "material_types": ["diagram"],
                "frame_paths": [str(first)],
                "visual_understanding": {
                    "validation_status": "ok",
                    "spatial_relations": ["A 在左，B 在右"],
                    "evidence_frame_paths": [str(first)],
                },
            },
            {
                "index": 2,
                "start": 120,
                "end": 125,
                "visual_route": "semantic_frame",
                "transcript": "现在回到同一张关系图。",
                "visual_text": shared_text,
                "structured_visual": {"type": "diagram"},
                "material_types": ["diagram"],
                "frame_paths": [str(second)],
            },
        ],
    )

    result = vision_review_triage(bundle, write=False)

    assert result["semantic_indexes"] == []
    duplicate = next(
        row for row in result["all_ranked_candidates"] if row["index"] == 2
    )
    assert duplicate["duplicate_of_index"] == 1
    assert (
        "completed_semantic_analysis_duplicate_of_index_1"
        in duplicate["suppression_reasons"]
    )
    assert duplicate["estimated_model_calls"] == 0
