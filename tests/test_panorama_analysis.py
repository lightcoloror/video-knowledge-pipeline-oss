from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from video_knowledge_pipeline.panorama_analysis import (
    PANORAMA_ANALYSIS_PLAN_SCHEMA,
    plan_panorama_analysis,
    resolve_panorama_vision_route,
)
from video_knowledge_pipeline.panorama_projection import (
    PANORAMA_VIEWPORT_PACK_SCHEMA,
    viewport_pack_sha256,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, segments: int = 2, samples: int = 5) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    _write_json(
        root / "timeline.json",
        [
            {"index": index, "start": float(index - 1), "end": float(index), "visual_route": "mixed"}
            for index in range(1, segments + 1)
        ],
    )
    geometry = {
        "front": (0.0, 0.0),
        "right": (90.0, 0.0),
        "back": (180.0, 0.0),
        "left": (-90.0, 0.0),
        "up": (0.0, 90.0),
        "down": (0.0, -90.0),
    }
    viewports = []
    for index in range(1, segments + 1):
        for sample in range(1, samples + 1):
            for ordinal, (direction, (yaw, pitch)) in enumerate(geometry.items()):
                path = root / "runs" / "panorama-fixture" / "viewports" / f"{index:04d}" / f"sample-{sample:02d}" / f"{direction}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                value = min(255, ordinal * 30 + sample * (20 if direction == "right" else 2))
                Image.new("RGB", (32, 32), (value, value, value)).save(path)
                viewports.append(
                    {
                        "viewport_id": f"vp-{index}-{sample}-{direction}",
                        "timeline_index": index,
                        "sample_index": sample,
                        "timestamp_seconds": float(index * 10 + sample),
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
        "profile": {"projection": "equirectangular_mono", "ready": True},
        "plan_sha256": "b" * 64,
        "manifest_sha256": _sha256(root / "manifest.json"),
        "timeline_sha256": _sha256(root / "timeline.json"),
        "source_sha256": "a" * 64,
        "selected_indexes": list(range(1, segments + 1)),
        "segment_count": segments,
        "sample_count": segments * samples,
        "frame_count": samples,
        "face_size": 32,
        "viewport_count": len(viewports),
        "viewports": viewports,
        "geometric_coverage_sha256": "c" * 64,
    }
    pack["task_pack_sha256"] = viewport_pack_sha256(pack)
    pack_path = root / "runs" / "panorama-fixture" / "viewport-pack.json"
    _write_json(pack_path, pack)
    return root, pack_path


def test_analysis_plan_uses_six_faces_then_at_most_three_temporal_directions(tmp_path: Path) -> None:
    bundle, pack = _bundle(tmp_path)

    plan = plan_panorama_analysis(bundle, viewport_pack_json=pack, limit=1, write=False)

    assert plan["schema"] == PANORAMA_ANALYSIS_PLAN_SCHEMA
    assert plan["selected_indexes"] == [1]
    semantic = [row for row in plan["tasks"] if row["task_type"] == "panorama_six_view_semantic"]
    temporal = [row for row in plan["tasks"] if row["task_type"] == "panorama_temporal_direction"]
    assert len(semantic) == 1
    assert len(semantic[0]["frame_paths"]) == 6
    assert len(temporal) == 3
    assert all(5 <= len(row["frame_paths"]) <= 12 for row in temporal)
    assert all(row["probe"]["implementation"]["library"] == "Pillow" for row in temporal)
    assert temporal[0]["direction"] == "right"
    assert plan["remote_calls_made"] == 0
    assert plan["model_calls_made"] == 0
    assert not (bundle / "runs" / "panorama-analysis-fixture").exists()


def test_route_prefers_cache_then_loopback_local_vlm_then_authorized_provider() -> None:
    assert resolve_panorama_vision_route(cache_available=True)["route"] == "validated_cache"
    local = resolve_panorama_vision_route(
        local_vlm_preflight={
            "ready_to_execute": True,
            "provider": {"provider": "local_vlm", "base_url": "http://127.0.0.1:1234/v1"},
        }
    )
    assert local["route"] == "local_vlm"
    provider = resolve_panorama_vision_route(
        provider_preflight={"ready_to_execute": True, "consent": {"valid": True}}
    )
    assert provider["route"] == "authorized_provider"


def test_online_provider_without_existing_consent_is_never_selected() -> None:
    route = resolve_panorama_vision_route(
        provider_preflight={"ready_to_execute": True},
        interactive_session_available=False,
    )

    assert route["status"] == "unavailable"
    assert route["route"] == "review_gap"


def test_codex_session_is_interactive_fallback_and_unavailable_becomes_review_gap(tmp_path: Path) -> None:
    bundle, pack = _bundle(tmp_path, segments=1)

    interactive = plan_panorama_analysis(bundle, viewport_pack_json=pack, write=False)
    unavailable = plan_panorama_analysis(
        bundle,
        viewport_pack_json=pack,
        interactive_session_available=False,
        write=False,
    )

    assert {row["route"]["route"] for row in interactive["tasks"]} == {"codex_session"}
    assert unavailable["status"] == "review_gap"
    assert "panorama_visual_backend_unavailable" in {row["key"] for row in unavailable["review_gaps"]}


def test_write_mode_persists_plan_and_manifest_summary_without_model_calls(tmp_path: Path) -> None:
    bundle, pack = _bundle(tmp_path, segments=1)

    plan = plan_panorama_analysis(bundle, viewport_pack_json=pack, write=True)

    assert Path(plan["plan_path"]).is_file()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["panorama_analysis_plan"]["plan_sha256"] == plan["plan_sha256"]
    assert plan["remote_calls_made"] == 0
    assert plan["model_calls_made"] == 0
