from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.panorama_workbench import (
    PANORAMA_CAMERA_HINTS_SCHEMA,
    export_panorama_workbench,
    panorama_panel_html,
)
from video_knowledge_pipeline.video_workbench import _render_workbench_html


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "private panorama bundle"
    root.mkdir()
    directions = {
        "front": (0.0, 0.0),
        "right": (90.0, 0.0),
        "back": (180.0, 0.0),
        "left": (-90.0, 0.0),
        "up": (0.0, 90.0),
        "down": (0.0, -90.0),
    }
    viewport_rows = []
    observations = []
    for ordinal, (direction, (yaw, pitch)) in enumerate(directions.items(), start=1):
        path = root / "runs" / "panorama-card" / "viewports" / f"{direction}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{direction}".encode("utf-8"))
        viewport_rows.append(
            {
                "viewport_id": f"vp-{direction}",
                "timeline_index": 1,
                "sample_index": 1,
                "direction": direction,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
        )
        observations.append(
            {
                "viewport_id": f"vp-{direction}",
                "direction": direction,
                "yaw": yaw,
                "pitch": pitch,
                "roll": 0.0,
                "h_fov": 90.0,
                "v_fov": 90.0,
                "observed": True,
                "observation": direction,
                "salience_score": ordinal / 10,
            }
        )
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        "panorama_profile": {
            "schema": "video_knowledge_pipeline.panorama_profile.v1",
            "projection": "equirectangular_mono",
            "metadata_confidence": "confirmed",
            "profile_sha256": "a" * 64,
            "ready": True,
        },
        "panorama_projection_plan": {"run_id": "card", "plan_sha256": "b" * 64},
        "panorama_analysis_plan": {"selected_indexes": [1], "temporal_required_indexes": []},
    }
    timeline = [
        {
            "index": 1,
            "panorama_visual_understanding": {
                "schema": "video_knowledge_pipeline.panorama_visual_understanding.v1",
                "sessions": [
                    {
                        "session_id": "session-1",
                        "viewport_pack_sha256": "c" * 64,
                        "sample_index": 1,
                        "timestamp_seconds": 5.0,
                        "evidence_level": "confirmed",
                        "executor": "codex_session",
                        "model_actual": "gpt-5.6-sol",
                        "viewport_observations": observations,
                    }
                ],
            },
        }
    ]
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "timeline.json", timeline)
    _write_json(root / "runs" / "panorama-card" / "projection-plan.json", {"schema": "plan"})
    _write_json(root / "runs" / "panorama-card" / "viewport-pack.json", {"viewports": viewport_rows})
    _write_json(
        root / "runs" / "panorama-analysis-card" / "analysis-plan.json",
        {"semantic_task_count": 1, "temporal_task_count": 0, "review_gaps": []},
    )
    return root


def test_panorama_workbench_card_shows_six_faces_coverage_model_and_links(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    result = export_panorama_workbench(bundle, write=True)

    assert result["status"] == "complete"
    assert result["profile"]["projection"] == "equirectangular_mono"
    assert len(result["face_grid"]) == 6
    assert result["coverage"]["geometric_complete_segments"] == 1
    assert result["coverage"]["semantic_covered_segments"] == 1
    assert result["models"] == [
        {"executor": "codex_session", "model_actual": "gpt-5.6-sol", "evidence_level": "confirmed"}
    ]
    panel = panorama_panel_html(result)
    assert "全局不存在" in panel
    assert panel.count("<img") == 6
    assert "viewport_pack" in panel


def test_camera_hints_are_path_redacted_candidate_only_and_not_render_authority(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    result = export_panorama_workbench(bundle, write=True)
    hints = result["camera_hints"]
    serialized = json.dumps(hints, ensure_ascii=False)

    assert hints["schema"] == PANORAMA_CAMERA_HINTS_SCHEMA
    assert hints["candidate_only"] is True
    assert hints["review_required"] is True
    assert hints["render_authorized"] is False
    assert len(hints["hints"]) == 3
    assert all(row["status"] == "candidate_only" and row["review_required"] for row in hints["hints"])
    assert str(bundle) not in serialized
    assert '"source_path":' not in serialized
    assert "latitude" not in serialized
    assert "face_embedding" not in serialized
    assert (bundle / "exports" / "panorama-camera-hints.json").is_file()
    markdown = (bundle / "panorama-workbench.md").read_text(encoding="utf-8")
    assert str(bundle) not in markdown


def test_non_panorama_bundle_card_is_not_required() -> None:
    assert "未声明全景投影" in panorama_panel_html({"status": "not_required"})


def test_unified_workbench_html_contains_panorama_card_without_new_page(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    panorama = export_panorama_workbench(bundle, write=False)
    result = {
        "title": "Synthetic",
        "bundle_dir": str(bundle),
        "artifacts": [],
        "panorama": panorama,
        "shot_review": {},
        "ui_surface_registry": {},
        "ui_shell_assets": {},
    }

    rendered = _render_workbench_html(result, [])

    assert "全景球面分析" in rendered
    assert "equirectangular_mono" in rendered
    assert rendered.count("<img") >= 6
