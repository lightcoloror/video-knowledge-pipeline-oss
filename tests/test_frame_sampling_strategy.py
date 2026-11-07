from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_pipeline import video
from video_knowledge_pipeline.orchestrator import _frame_sampling_plan
from video_knowledge_pipeline.supplemental_frame_sampling import plan_supplemental_frame_sampling
from video_knowledge_pipeline.video import fixed_timepoints


def test_balanced_long_video_sampling_covers_full_duration() -> None:
    duration = 5 * 60 * 60

    plan = _frame_sampling_plan(duration, sample_interval=5.0, max_frames=720, sample_mode="balanced-long-video")
    points = fixed_timepoints(duration, plan["effective_interval_seconds"], plan["fixed_budget"])

    assert plan["mode"] == "balanced-long-video"
    assert plan["effective_max_frames"] == 720
    assert 24.0 < plan["effective_interval_seconds"] < 26.0
    assert len(points) == 720
    assert points[0][0] == 0.0
    assert points[-1][0] > duration - 30


def test_dense_local_sampling_keeps_requested_interval_for_long_video() -> None:
    duration = 5 * 60 * 60

    plan = _frame_sampling_plan(duration, sample_interval=5.0, max_frames=720, sample_mode="dense-local")
    points = fixed_timepoints(duration, plan["effective_interval_seconds"], plan["fixed_budget"])

    assert plan["mode"] == "dense-local"
    assert plan["effective_interval_seconds"] == 5.0
    assert plan["effective_max_frames"] >= 3601
    assert len(points) >= 3601
    assert duration - 0.2 < points[-1][0] < duration


def test_fixed_timepoints_never_seeks_to_exact_eof() -> None:
    duration = 1067.237278

    points = fixed_timepoints(duration, interval=duration / 79, max_points=80)

    assert len(points) == 80
    assert points[-1][0] == duration - 0.1
    assert all(point < duration for point, _signal in points)


def test_video_probe_prefers_decodable_stream_duration_over_audio_tailed_container(
    tmp_path: Path, monkeypatch
) -> None:
    media = tmp_path / "audio-tailed.mp4"
    media.write_bytes(b"fixture")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "duration": "6181.493667",
                "width": 1920,
                "height": 1200,
                "avg_frame_rate": "15/1",
            },
            {"codec_type": "audio", "duration": "6181.538344"},
        ],
        "format": {"duration": "6181.538344"},
    }
    monkeypatch.setattr(video, "resolve_media_tool", lambda name: f"fixture-{name}")
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    metadata = video._probe_with_ffprobe(media)

    assert metadata["duration_seconds"] == 6181.493667
    points = fixed_timepoints(
        metadata["duration_seconds"], interval=metadata["duration_seconds"] / 79, max_points=80
    )
    assert points[-1][0] == metadata["duration_seconds"] - 0.1


def test_triage_first_reserves_budget_for_scene_or_semantic_points() -> None:
    duration = 5 * 60 * 60

    plan = _frame_sampling_plan(duration, sample_interval=5.0, max_frames=720, sample_mode="triage-first")

    assert plan["mode"] == "triage-first"
    assert plan["effective_max_frames"] == 720
    assert plan["fixed_budget"] == 504
    assert plan["reserved_for_scene_or_semantic"] == 216
    assert plan["effective_interval_seconds"] > 30



def _write_bundle(tmp_path: Path, *, media_path: str = "D:/videos/lesson.mp4", timeline: list[dict] | None = None) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {"schema": "lecture_webui_bundle.v1"}
    if media_path:
        manifest["media_path"] = media_path
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline or [], ensure_ascii=False), encoding="utf-8")
    return bundle


def test_supplemental_frame_sampling_plans_local_recapture_from_triage(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path,
        timeline=[
            {
                "index": 1,
                "start": 10,
                "end": 18,
                "visual_route": "document_visual",
                "transcript": "这里看屏幕上的工具价格 16k",
                "visual_text": "",
                "quality_issues": ["missing_visual_text"],
                "frame_paths": ["frames/001.jpg"],
            }
        ],
    )
    frame = bundle / "frames" / "001.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"fixture-frame")

    result = plan_supplemental_frame_sampling(bundle, max_items=1, max_frames_per_item=3, write=True)

    assert result["status"] == "ok"
    assert result["summary"]["planned_frames"] == 3
    assert result["summary"]["cloud_vision_allowed_by_default"] is False
    assert all(item["cloud_vision_allowed_by_default"] is False for item in result["items"])
    assert all(item["video_key"] == "D:/videos/lesson.mp4" for item in result["items"])
    assert (bundle / "supplemental-frame-sampling-plan.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_recapture"]["items"]
    assert manifest["frame_recapture"]["source"] == "supplemental_frame_sampling_plan"


def test_supplemental_frame_sampling_prefers_temporal_candidates(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path,
        timeline=[
            {
                "index": 1,
                "start": 20,
                "end": 30,
                "visual_route": "temporal_sequence",
                "transcript": "点击后台按钮然后切换页面演示流程",
                "visual_text": "页面",
                "quality_issues": ["temporal_sequence_without_analysis"],
                "frame_paths": ["frames/001.jpg"],
            }
        ],
    )

    result = plan_supplemental_frame_sampling(bundle, max_items=1, max_frames_per_item=5, write=False)

    assert result["items"]
    assert {item["recommended_action"] for item in result["items"]} == {"temporal_recapture"}
    assert len(result["items"]) == 5
    assert result["items"][0]["midpoint"] < result["items"][-1]["midpoint"]


def test_supplemental_frame_sampling_reports_missing_source_video(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path,
        media_path="",
        timeline=[
            {
                "index": 1,
                "start": 1,
                "end": 5,
                "visual_route": "semantic_frame",
                "transcript": "看这个工具名称",
                "visual_text": "",
                "quality_issues": ["semantic_frame_without_analysis"],
                "frame_paths": ["frames/001.jpg"],
            }
        ],
    )

    result = plan_supplemental_frame_sampling(bundle, max_items=1, write=False)

    assert result["status"] == "needs_source_video"
    assert result["summary"]["has_source_video"] is False
    assert result["next_actions"][0]["key"] == "set_source_video_path"
