from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from video_knowledge_pipeline.temporal_visual_analyzer import (
    _normalise_temporal_understanding,
    _prompt,
)
from video_knowledge_pipeline.temporal_frame_preprocess import (
    build_temporal_frame_manifest,
    prepare_temporal_image_probe,
)
from video_knowledge_pipeline.temporal_frame_groups import plan_temporal_frame_candidates


def _frames(root: Path) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index in range(1, 9):
        image = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), "Static course slide", fill="black")
        if index >= 5:
            draw.rectangle((420, 180, 610, 330), fill="#dddddd")
            draw.text((440, 220), "changed", fill="black")
        path = root / f"frame_{index:02d}_{21_000 + (index - 1) * 1_143:010d}ms.jpg"
        image.save(path, quality=92)
        paths.append(str(path))
    return paths


def test_temporal_probe_builds_numbered_contact_sheet_and_bounded_representatives(tmp_path: Path) -> None:
    paths = _frames(tmp_path / "frames")

    report = prepare_temporal_image_probe(
        paths,
        output_dir=tmp_path / "prepared",
        use_contact_sheet=True,
        representative_limit=2,
        max_edge=640,
        jpeg_quality=75,
    )

    assert report["schema"] == "video_knowledge_pipeline.temporal_vlm_preprocess.v1"
    assert report["original_frame_count"] == 8
    assert report["frame_manifest"][0]["frame_id"] == "F01"
    assert report["frame_manifest"][-1]["frame_id"] == "F08"
    assert report["frame_manifest"][0]["timestamp_ms"] == 21_000
    assert report["representative_frame_count"] == 2
    assert report["sent_strategy"] == "contact_sheet_plus_representatives"
    assert len(report["image_paths"]) == 3
    assert Path(report["contact_sheet_path"]).is_file()
    assert len(report["frame_mapping"]) == 8
    assert {row["frame_id"] for row in report["frame_mapping"]} == {f"F{i:02d}" for i in range(1, 9)}
    assert report["implementation"]["library"] == "Pillow"


def test_temporal_prompt_uses_system_frame_manifest_and_readable_chinese(tmp_path: Path) -> None:
    paths = _frames(tmp_path / "frames")
    candidate = {
        "start": 21.0,
        "end": 29.0,
        "transcript": "讲师说明课程内容。",
        "visual_text": "高频问题的处理技巧",
        "frame_paths": paths,
        "frame_manifest": build_temporal_frame_manifest(paths),
    }

    prompt = _prompt(candidate)

    assert "expected_frame_count" in prompt
    assert '"frame_id": "F01"' in prompt
    assert '"frame_id": "F08"' in prompt
    assert "系统提供的帧数、帧 ID 和时间戳" in prompt
    assert "不要写成“四帧”" in prompt
    assert "�" not in prompt


def test_temporal_contract_rejects_model_frame_count_claim_that_conflicts_with_evidence(tmp_path: Path) -> None:
    paths = _frames(tmp_path / "frames")
    candidate = {
        "frame_paths": paths,
        "frame_manifest": build_temporal_frame_manifest(paths),
        "strict_frame_contract": True,
    }
    payload = {
        "event_sequence": ["四帧展示了同一课程标题页。"],
        "state_changes": ["讲师姿态略有变化。"],
        "observed_frame_count": 8,
        "observed_frame_ids": [f"F{i:02d}" for i in range(1, 9)],
        "per_frame_observations": [
            {"frame_id": f"F{i:02d}", "observation": "标题页可见"} for i in range(1, 9)
        ],
        "evidence_frame_paths": paths,
    }

    result = _normalise_temporal_understanding(payload, candidate)

    assert result["expected_frame_count"] == 8
    assert result["observed_frame_count"] == 8
    assert result["validation_status"] == "incomplete"
    assert "frame_count_claim_mismatch" in result["validation_issues"]


def test_temporal_contract_accepts_complete_per_frame_evidence(tmp_path: Path) -> None:
    paths = _frames(tmp_path / "frames")
    candidate = {
        "frame_paths": paths,
        "frame_manifest": build_temporal_frame_manifest(paths),
        "strict_frame_contract": True,
    }
    payload = {
        "event_sequence": ["F01 到 F08 均为同一标题页，讲师姿态轻微变化。"],
        "state_changes": ["F04 到 F05 的右下区域发生变化。"],
        "observed_frame_count": 8,
        "observed_frame_ids": [f"F{i:02d}" for i in range(1, 9)],
        "per_frame_observations": [
            {"frame_id": f"F{i:02d}", "observation": "标题页可见"} for i in range(1, 9)
        ],
        "evidence_frame_paths": paths,
    }

    result = _normalise_temporal_understanding(payload, candidate)

    assert result["validation_status"] == "ok"
    assert result["validation_issues"] == []
    assert result["expected_frame_ids"] == [f"F{i:02d}" for i in range(1, 9)]


def test_public_temporal_candidate_planner_preserves_existing_sampling_contract(tmp_path: Path) -> None:
    timeline = [
        {
            "index": 9,
            "visual_route": "mixed",
            "video_key": str(tmp_path / "video.mp4"),
            "start": 10.0,
            "end": 18.0,
            "video_duration_seconds": 30.0,
        }
    ]

    candidates = plan_temporal_frame_candidates(
        timeline,
        routes={"mixed"},
        frame_count=5,
        window_seconds=4.0,
        root=tmp_path,
    )

    assert len(candidates) == 1
    assert candidates[0]["index"] == 1
    assert candidates[0]["frame_count"] == 5
    assert candidates[0]["times"] == [10.0, 12.0, 14.0, 16.0, 18.0]
    assert candidates[0]["output_paths"][0].endswith("temporal-frames\\0001\\frame_01_0000010000ms.jpg")
