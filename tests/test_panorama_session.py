from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.codex_vision_session import import_codex_vision_results
from video_knowledge_pipeline.panorama_projection import PANORAMA_VIEWPORT_PACK_SCHEMA
from video_knowledge_pipeline.panorama_session import prepare_panorama_vision_session


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_with_pack(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    root.mkdir()
    timeline = [{"index": 1, "start": 0.0, "end": 10.0, "visual_route": "mixed", "transcript": "synthetic"}]
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    _write_json(root / "timeline.json", timeline)
    rows = []
    directions = [
        ("front", 0.0, 0.0),
        ("right", 90.0, 0.0),
        ("back", 180.0, 0.0),
        ("left", -90.0, 0.0),
        ("up", 0.0, 90.0),
        ("down", 0.0, -90.0),
    ]
    for direction, yaw, pitch in directions:
        path = root / "runs" / "panorama-fixture" / "viewports" / "0001" / "sample-01" / f"{direction}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{direction}".encode("utf-8"))
        rows.append(
            {
                "viewport_id": f"vp-{direction}",
                "timeline_index": 1,
                "sample_index": 1,
                "timestamp_seconds": 5.0,
                "direction": direction,
                "yaw": yaw,
                "pitch": pitch,
                "roll": 0.0,
                "h_fov": 90.0,
                "v_fov": 90.0,
                "path": path.relative_to(root).as_posix(),
                "source_sha256": "a" * 64,
                "filter": "v360=input=e:output=flat",
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    pack = {
        "schema": PANORAMA_VIEWPORT_PACK_SCHEMA,
        "run_id": "fixture",
        "profile": {"schema": "video_knowledge_pipeline.panorama_profile.v1", "projection": "equirectangular_mono", "ready": True},
        "plan_sha256": "b" * 64,
        "manifest_sha256": _sha256(root / "manifest.json"),
        "timeline_sha256": _sha256(root / "timeline.json"),
        "source_sha256": "a" * 64,
        "selected_indexes": [1],
        "segment_count": 1,
        "sample_count": 1,
        "frame_count": 1,
        "face_size": 64,
        "viewport_count": 6,
        "viewports": rows,
        "geometric_coverage_sha256": "c" * 64,
    }
    from video_knowledge_pipeline.panorama_projection import viewport_pack_sha256

    pack["task_pack_sha256"] = viewport_pack_sha256(pack)
    pack_path = root / "runs" / "panorama-fixture" / "viewport-pack.json"
    _write_json(pack_path, pack)
    return root, pack_path


def _filled_result(template_path: Path) -> Path:
    result = json.loads(template_path.read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-sol"
    row = result["items"][0]
    row["evidence_level"] = "confirmed"
    row["observed_frame_ids"] = list(row["expected_frame_ids"])
    for observation in row["panorama_viewport_observations"]:
        observation["observed"] = True
        observation["observation"] = f"visible content in {observation['direction']}"
    row["visual_understanding"] = {
        "objects": ["room"],
        "actions": [],
        "interface_state": "",
        "spatial_relations": ["content surrounds camera"],
        "instructor_focus": "",
        "non_text_information": ["six directional views inspected"],
        "confidence": 0.8,
        "keep_image_reason": "spherical context",
        "evidence_frame_paths": list(row["frame_paths"]),
    }
    path = template_path.with_name("result.json")
    _write_json(path, result)
    return path


def test_prepare_panorama_session_reuses_codex_pack_and_binds_exact_six_views(tmp_path: Path) -> None:
    bundle, viewport_pack = _bundle_with_pack(tmp_path)

    prepared = prepare_panorama_vision_session(bundle, viewport_pack_json=viewport_pack, timeline_index=1, sample_index=1)

    assert prepared["kind"] == "semantic"
    assert prepared["viewport_context"]["viewport_pack_sha256"]
    assert len(prepared["items"]) == 1
    assert len(prepared["items"][0]["frames"]) == 6
    assert [row["direction"] for row in prepared["items"][0]["viewport_context"]["viewports"]] == [
        "front", "right", "back", "left", "up", "down"
    ]
    assert all(Path(path).is_relative_to(bundle) for path in prepared["items"][0]["frame_paths"])


def test_panorama_import_is_atomic_and_writes_orthogonal_evidence(tmp_path: Path) -> None:
    bundle, viewport_pack = _bundle_with_pack(tmp_path)
    prepared = prepare_panorama_vision_session(bundle, viewport_pack_json=viewport_pack, timeline_index=1, sample_index=1)
    result_path = _filled_result(Path(prepared["result_template_path"]))

    imported = import_codex_vision_results(bundle, input_json=result_path)

    assert imported["status"] == "imported"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    panorama = timeline[0]["panorama_visual_understanding"]
    assert panorama["schema"] == "video_knowledge_pipeline.panorama_visual_understanding.v1"
    assert panorama["sessions"][0]["evidence_level"] == "confirmed"
    assert len(panorama["sessions"][0]["viewport_observations"]) == 6
    assert "visual_understanding" not in timeline[0]
    assert imported["vision_restore_hint"]["status"] == "ready"


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "yaw", "pack", "frame", "oversized"])
def test_panorama_import_rejects_invalid_or_stale_viewport_results_without_timeline_change(
    tmp_path: Path, mutation: str
) -> None:
    bundle, viewport_pack = _bundle_with_pack(tmp_path)
    prepared = prepare_panorama_vision_session(bundle, viewport_pack_json=viewport_pack, timeline_index=1, sample_index=1)
    result_path = _filled_result(Path(prepared["result_template_path"]))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    observations = result["items"][0]["panorama_viewport_observations"]
    if mutation == "missing":
        observations.pop()
    elif mutation == "extra":
        observations.append(dict(observations[0], viewport_id="vp-extra"))
    elif mutation == "duplicate":
        observations[-1] = dict(observations[0])
    elif mutation == "yaw":
        observations[0]["yaw"] = 45.0
    elif mutation == "pack":
        pack = json.loads(viewport_pack.read_text(encoding="utf-8"))
        pack["geometric_coverage_sha256"] = "d" * 64
        _write_json(viewport_pack, pack)
    elif mutation == "frame":
        Path(prepared["items"][0]["frame_paths"][0]).write_bytes(b"changed")
    elif mutation == "oversized":
        result["items"][0]["visual_understanding"]["objects"] = ["x" * (2 * 1024 * 1024)]
    _write_json(result_path, result)
    before = (bundle / "timeline.json").read_bytes()

    rejected = import_codex_vision_results(bundle, input_json=result_path)

    assert rejected["status"] == "rejected"
    assert rejected["updated_count"] == 0
    assert (bundle / "timeline.json").read_bytes() == before
    assert Path(rejected["raw_result_path"]).is_file()


def test_panorama_prepare_rejects_noncanonical_viewport_geometry(tmp_path: Path) -> None:
    bundle, viewport_pack = _bundle_with_pack(tmp_path)
    pack = json.loads(viewport_pack.read_text(encoding="utf-8"))
    pack["viewports"][0]["yaw"] = 45.0
    from video_knowledge_pipeline.panorama_projection import viewport_pack_sha256

    pack["task_pack_sha256"] = viewport_pack_sha256(pack)
    _write_json(viewport_pack, pack)

    with pytest.raises(ValueError, match="canonical geometry"):
        prepare_panorama_vision_session(bundle, viewport_pack_json=viewport_pack, timeline_index=1, sample_index=1)
