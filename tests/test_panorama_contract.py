from __future__ import annotations

import pytest

from video_knowledge_pipeline.panorama_contract import (
    PANORAMA_PROFILE_SCHEMA,
    build_panorama_profile,
)


@pytest.mark.parametrize(
    ("projection", "stereo_mode"),
    [
        ("equirectangular_mono", "2d"),
        ("equirectangular_stereo_sbs", "sbs"),
        ("equirectangular_stereo_tb", "tb"),
        ("cubemap_3x2", "2d"),
        ("cubemap_6x1", "2d"),
        ("cubemap_1x6", "2d"),
    ],
)
def test_explicit_projection_builds_ready_profile(projection: str, stereo_mode: str) -> None:
    profile = build_panorama_profile(projection=projection, source_path="fixture.mp4")

    assert profile["schema"] == PANORAMA_PROFILE_SCHEMA
    assert profile["ready"] is True
    assert profile["projection"] == projection
    assert profile["stereo_mode"] == stereo_mode
    assert profile["projection_source"] == "explicit_override"
    assert len(profile["profile_sha256"]) == 64


def test_dual_fisheye_requires_explicit_fov_and_lens_order() -> None:
    blocked = build_panorama_profile(projection="dual_fisheye_explicit", source_path="fixture.mp4")
    ready = build_panorama_profile(
        projection="dual_fisheye_explicit",
        source_path="fixture.mp4",
        input_fov=190.0,
        lens_order="left_right",
    )

    assert blocked["ready"] is False
    assert {row["key"] for row in blocked["issues"]} == {"input_fov_required", "lens_order_required"}
    assert ready["ready"] is True
    assert ready["input_fov"] == 190.0
    assert ready["lens_order"] == "left_right"


@pytest.mark.parametrize(
    "kwargs, issue_key",
    [
        ({"projection": "flat"}, "projection_unsupported"),
        ({"projection": "dual_fisheye_explicit", "input_fov": 0, "lens_order": "left_right"}, "input_fov_invalid"),
        ({"projection": "dual_fisheye_explicit", "input_fov": 190, "lens_order": "clockwise"}, "lens_order_invalid"),
        ({"projection": "cubemap_3x2", "lens_order": "left_right"}, "lens_order_not_applicable"),
    ],
)
def test_invalid_projection_combinations_fail_closed(kwargs: dict[str, object], issue_key: str) -> None:
    profile = build_panorama_profile(source_path="fixture.mp4", **kwargs)

    assert profile["ready"] is False
    assert issue_key in {row["key"] for row in profile["issues"]}


def test_two_to_one_aspect_ratio_is_only_a_hint() -> None:
    profile = build_panorama_profile(
        source_path="ordinary-2-to-1.mp4",
        ffprobe_payload={"streams": [{"codec_type": "video", "width": 2000, "height": 1000}]},
    )

    assert profile["projection"] == "unknown"
    assert profile["ready"] is False
    assert profile["aspect_ratio_hint"] == "possible_equirectangular"
    assert "projection_required" in {row["key"] for row in profile["issues"]}


def test_spherical_metadata_detects_equirectangular_stereo() -> None:
    profile = build_panorama_profile(
        source_path="fixture.mp4",
        ffprobe_payload={
            "streams": [
                {
                    "codec_type": "video",
                    "width": 3840,
                    "height": 1920,
                    "tags": {"projection": "equirectangular", "stereo_mode": "left-right"},
                    "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}],
                }
            ]
        },
    )

    assert profile["ready"] is True
    assert profile["projection"] == "equirectangular_stereo_sbs"
    assert profile["projection_source"] == "ffprobe_metadata"
    assert profile["metadata_confidence"] == "confirmed"
    assert profile["metadata_evidence"]


def test_explicit_projection_conflicting_with_metadata_is_blocked() -> None:
    profile = build_panorama_profile(
        source_path="fixture.mp4",
        projection="cubemap_3x2",
        ffprobe_payload={
            "streams": [
                {
                    "codec_type": "video",
                    "tags": {"projection": "equirectangular"},
                    "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}],
                }
            ]
        },
    )

    assert profile["ready"] is False
    assert "projection_metadata_conflict" in {row["key"] for row in profile["issues"]}


def test_proprietary_source_requires_decode_evidence_and_explicit_projection() -> None:
    blocked = build_panorama_profile(source_path="camera.insv")
    allowed = build_panorama_profile(
        source_path="camera.insv",
        projection="equirectangular_mono",
        ffprobe_payload={"streams": [{"codec_type": "video", "codec_name": "hevc"}]},
    )

    assert blocked["ready"] is False
    assert "proprietary_source_requires_vendor_export" in {row["key"] for row in blocked["issues"]}
    assert allowed["ready"] is True


def test_profile_hash_is_stable_and_does_not_include_absolute_source_path() -> None:
    first = build_panorama_profile(projection="equirectangular_mono", source_path="C:/private/one.mp4")
    second = build_panorama_profile(projection="equirectangular_mono", source_path="D:/other/two.mp4")

    assert first["profile_sha256"] == second["profile_sha256"]
    assert "C:/private" not in str(first)
    assert "D:/other" not in str(second)
