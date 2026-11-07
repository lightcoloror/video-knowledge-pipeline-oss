from __future__ import annotations

import hashlib
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from video_knowledge_pipeline.panorama_contract import build_panorama_profile
from video_knowledge_pipeline.panorama_projection import (
    PANORAMA_PROJECTION_PLAN_SCHEMA,
    PANORAMA_VIEWPORT_PACK_SCHEMA,
    execute_panorama_projection,
    plan_panorama_projection,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, segment_count: int = 2, source_name: str = "source with spaces.mp4") -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = tmp_path / source_name
    source.write_bytes(b"synthetic panorama source")
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        "source_sha256": _sha256(source),
    }
    timeline = [
        {
            "index": index,
            "visual_route": "mixed",
            "video_key": str(source),
            "start": float((index - 1) * 10),
            "end": float(index * 10),
            "video_duration_seconds": float(segment_count * 10),
        }
        for index in range(1, segment_count + 1)
    ]
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    return bundle, source


def _profile(projection: str = "equirectangular_mono") -> dict[str, object]:
    kwargs: dict[str, object] = {"projection": projection, "source_path": "fixture.mp4"}
    if projection == "dual_fisheye_explicit":
        kwargs.update({"input_fov": 190.0, "lens_order": "left_right"})
    return build_panorama_profile(**kwargs)


def test_projection_plan_is_bounded_hash_bound_and_does_not_write(tmp_path: Path) -> None:
    bundle, source = _bundle(tmp_path, segment_count=3)
    before = sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*"))

    plan = plan_panorama_projection(
        bundle,
        profile=_profile(),
        indexes=[1, 3],
        frame_count=5,
        limit=1,
        face_size=128,
        run_id="bounded",
    )

    assert plan["schema"] == PANORAMA_PROJECTION_PLAN_SCHEMA
    assert plan["selected_indexes"] == [1]
    assert plan["segment_count"] == 1
    assert plan["sample_count"] == 5
    assert plan["viewport_count"] == 30
    assert len(plan["plan_sha256"]) == 64
    assert plan["source_sha256"] == _sha256(source)
    assert plan["estimated_output_bytes"] == 30 * 128 * 128 * 3
    assert {row["direction"] for row in plan["viewports"]} == {"front", "right", "back", "left", "up", "down"}
    assert all(not Path(row["path"]).is_absolute() for row in plan["viewports"])
    assert len({row["viewport_id"] for row in plan["viewports"]}) == 30
    assert str(source) in plan["viewports"][0]["command"]
    assert sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*")) == before


@pytest.mark.parametrize(
    ("projection", "expected"),
    [
        ("equirectangular_mono", "input=e:output=flat"),
        ("equirectangular_stereo_sbs", "input=e:output=flat:in_stereo=sbs"),
        ("equirectangular_stereo_tb", "input=e:output=flat:in_stereo=tb"),
        ("cubemap_3x2", "input=c3x2:output=flat"),
        ("cubemap_6x1", "input=c6x1:output=flat"),
        ("cubemap_1x6", "input=c1x6:output=flat"),
        ("dual_fisheye_explicit", "input=dfisheye:output=flat:ih_fov=190"),
    ],
)
def test_projection_plan_uses_ffmpeg_v360_contract(tmp_path: Path, projection: str, expected: str) -> None:
    bundle, _ = _bundle(tmp_path, segment_count=1)

    plan = plan_panorama_projection(
        bundle,
        profile=_profile(projection),
        frame_count=5,
        limit=1,
        run_id=projection,
    )

    assert expected in plan["viewports"][0]["filter"]
    assert plan["viewports"][0]["command"].count(str(Path(plan["viewports"][0]["absolute_output_path"]))) == 1


def test_execute_projection_calls_ffmpeg_only_for_selected_segments(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path, segment_count=2)
    plan = plan_panorama_projection(
        bundle,
        profile=_profile(),
        indexes=[2],
        frame_count=5,
        limit=16,
        face_size=64,
        run_id="execute",
    )
    commands: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        commands.append(command)
        output = Path(command[-1])
        output.write_bytes(("viewport:" + output.stem).encode("utf-8"))
        return CompletedProcess(command, 0, stdout="", stderr="")

    result = execute_panorama_projection(bundle, plan, command_runner=runner)

    assert result["status"] == "succeeded"
    assert len(commands) == 5 * 6
    assert result["pack"]["schema"] == PANORAMA_VIEWPORT_PACK_SCHEMA
    assert result["pack"]["selected_indexes"] == [2]
    assert len(result["pack"]["viewports"]) == 30
    assert all(len(row["sha256"]) == 64 for row in result["pack"]["viewports"])
    assert Path(result["pack_path"]).is_file()
    assert len(result["pack"]["geometric_coverage_sha256"]) == 64


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"frame_count": 4}, "frame_count"),
        ({"frame_count": 13}, "frame_count"),
        ({"limit": 17}, "limit"),
        ({"face_size": 32}, "face_size"),
    ],
)
def test_projection_plan_rejects_unbounded_parameters(tmp_path: Path, kwargs: dict[str, int], message: str) -> None:
    bundle, _ = _bundle(tmp_path, segment_count=1)

    with pytest.raises(ValueError, match=message):
        plan_panorama_projection(bundle, profile=_profile(), run_id="invalid", **kwargs)


def test_execution_rejects_timeline_hash_drift_before_creating_viewports(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path, segment_count=1)
    plan = plan_panorama_projection(bundle, profile=_profile(), frame_count=5, run_id="drift")
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="timeline hash drift"):
        execute_panorama_projection(bundle, plan, command_runner=lambda command: CompletedProcess(command, 0, "", ""))

    assert not (bundle / "runs" / "panorama-drift" / "viewports").exists()
