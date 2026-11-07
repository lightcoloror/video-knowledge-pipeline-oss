from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from panorama_synthetic_fixture import generate_panorama_fixture
from video_knowledge_pipeline.codex_vision_session import import_codex_vision_results
from video_knowledge_pipeline.knowledge_coverage import audit_knowledge_coverage
from video_knowledge_pipeline.media_tools import resolve_media_tool
from video_knowledge_pipeline.panorama_cli import panorama_prepare, panorama_restore, panorama_run
from video_knowledge_pipeline.panorama_session import prepare_panorama_vision_session
from video_knowledge_pipeline.panorama_workbench import export_panorama_workbench


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(resolve_media_tool("ffmpeg") is None, reason="FFmpeg is required for the real v360 synthetic smoke")
def test_generated_erp_fixture_projects_imports_audits_and_restores_without_source_mutation(tmp_path: Path) -> None:
    fixture = generate_panorama_fixture(tmp_path / "fixture")
    source = Path(fixture["media_path"])
    source_before = _sha256(source)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "Synthetic panorama acceptance",
            "source_media_sha256": source_before,
        },
    )
    _write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": float(fixture["duration_seconds"]),
                "visual_route": "mixed",
                "video_key": str(source),
                "video_duration_seconds": float(fixture["duration_seconds"]),
                "transcript": "这是完全合成的全景验收转写，音轨不来自真实人物。",
                "visual_text": "PANORAMA OCR 360",
            }
        ],
    )

    prepared = panorama_prepare(
        bundle,
        projection="equirectangular_mono",
        indexes=[1],
        frame_count=5,
        limit=1,
        face_size=64,
        run_id="synthetic-e2e",
        write=True,
    )
    executed = panorama_run(bundle, run_id="synthetic-e2e", execute_projection=True)

    assert prepared["viewport_count"] == 30
    assert executed["status"] == "ready_for_analysis"
    assert executed["projection"]["status"] == "succeeded"
    assert executed["projection"]["pack"]["viewport_count"] == 30
    assert executed["analysis"]["semantic_task_count"] == 1
    assert executed["analysis"]["temporal_task_count"] <= 3
    assert _sha256(source) == source_before

    semantic = next(
        row
        for row in executed["analysis"]["tasks"]
        if row["task_type"] == "panorama_six_view_semantic"
    )
    session = prepare_panorama_vision_session(
        bundle,
        viewport_pack_json=executed["projection"]["pack_path"],
        timeline_index=1,
        sample_index=int(semantic["sample_index"]),
    )
    result_path = Path(session["result_template_path"]).with_name("synthetic-result.json")
    result = json.loads(Path(session["result_template_path"]).read_text(encoding="utf-8"))
    result["model_actual"] = "synthetic-human-fixture"
    row = result["items"][0]
    row["evidence_level"] = "confirmed"
    row["observed_frame_ids"] = list(row["expected_frame_ids"])
    row["visual_understanding"].update(
        {
            "objects": ["six labelled synthetic viewports", "seam marker"],
            "interface_state": "synthetic ERP acceptance fixture",
            "confidence": 1.0,
            "evidence_frame_paths": list(row["frame_paths"]),
        }
    )
    for position, observation in enumerate(row["panorama_viewport_observations"]):
        observation["observed"] = True
        observation["observation"] = f"observed synthetic direction {observation['direction']}"
        observation["salience_score"] = round(1.0 - position * 0.1, 2)
    _write_json(result_path, result)
    imported = import_codex_vision_results(bundle, input_json=result_path)

    assert imported["status"] == "imported"
    assert imported["updated_count"] == 1
    coverage = audit_knowledge_coverage(bundle, write=False)["coverage"]["panorama_coverage"]
    assert coverage["geometric_complete_segments"] == 1
    assert coverage["semantic_covered_segments"] == 1
    assert coverage["status"] == "review_gap"
    workbench = export_panorama_workbench(bundle)
    assert len(workbench["face_grid"]) == 6
    assert workbench["camera_hints"]["candidate_only"] is True
    assert workbench["camera_hints"]["render_authorized"] is False

    restore_plan = panorama_restore(bundle, run_id="synthetic-e2e", execute=False)
    vision_run_id = restore_plan["resolved_vision_run_id"]
    assert restore_plan["status"] == "restore_plan_ready"
    assert vision_run_id.startswith("codex_session_semantic-")
    restored = panorama_restore(
        bundle,
        run_id="synthetic-e2e",
        execute=True,
        confirm_run_id=vision_run_id,
    )
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert restored["status"] == "ok"
    assert not timeline[0].get("panorama_visual_understanding")
    assert timeline[0]["transcript"].startswith("这是完全合成")
    assert _sha256(source) == source_before
