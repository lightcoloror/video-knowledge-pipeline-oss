from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser, main as cli_main
from video_knowledge_pipeline.shot_breakdown import build_shot_breakdown
from video_knowledge_pipeline.video_workbench import export_video_workbench


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    frame_a = bundle / "frames" / "a.jpg"
    frame_b = bundle / "frames" / "b.jpg"
    frame_a.parent.mkdir(parents=True)
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")
    _write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": "参考片", "duration_seconds": 10})
    _write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 5,
                "corrected_transcript": "讲师介绍第一步。",
                "visual_text": "第一步",
                "temporal_visual_understanding": {"event_sequence": ["人物抬手指向屏幕"]},
                "frame_paths": [str(frame_a)],
                "tagger_tags": ["人物", "演示", "室内"],
                "environment_type": "indoor",
                "edit_role": "establishing",
            },
            {
                "index": 2,
                "start": 5,
                "end": 10,
                "transcript": "随后展示操作结果。",
                "visual_text": "操作结果",
                "temporal_visual_understanding": {"summary": "画面切换到操作结果"},
                "frame_paths": [str(frame_b)],
                "tagger_tags": ["屏幕", "演示"],
                "environment_type": "indoor",
                "edit_role": "information",
            },
        ],
    )
    _write_json(
        bundle / "exports" / "scene-detection.json",
        {
            "schema": "video_knowledge_pipeline.scene_detection.v1",
            "backend": "pyscenedetect",
            "boundary_kind": "technical_shot",
            "scenes": [
                {"index": 1, "start": 0, "end": 5},
                {"index": 2, "start": 5, "end": 10},
            ],
        },
    )
    return bundle


def test_shot_breakdown_fuses_existing_evidence_and_saved_reference_analysis(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    reference = tmp_path / "video-breakdown.json"
    _write_json(
        reference,
        {
            "shots": [
                {"index": 1, "shot_type": "medium shot", "camera_motion": {"type": "pan"}, "composition": {"thirds_score": 0.8}},
                {"index": 2, "shot_type": "close-up", "camera_motion": {"type": "static"}, "color_profile": {"temperature": "warm"}},
            ]
        },
    )

    result = build_shot_breakdown(bundle, reference_analysis_json=reference, write=True)

    assert result["status"] == "completed"
    assert result["input_provenance"]["reference_analysis"]["artifact_sha256"]
    assert result["input_provenance"]["reference_analysis"]["artifact_bytes"] == reference.stat().st_size
    assert result["shot_count"] == 2
    assert result["shots"][0]["facts"]["shot_type"] == "medium"
    assert result["shots"][0]["facts"]["camera_movement"] == "pan_or_tilt"
    assert result["shots"][0]["facts"]["composition"]["thirds_score"] == 0.8
    assert result["shots"][0]["facts"]["dialogue_or_narration"] == "讲师介绍第一步。"
    assert result["shots"][0]["facts"]["subject_action"] == ["人物抬手指向屏幕"]
    assert result["style_fingerprint"]["cuts_per_minute"] == 6.0
    assert result["readiness"]["ready_count"] == 2
    assert result["operator_boundary"]["cloud_calls_made"] == 0
    assert result["operator_boundary"]["timeline_mutated"] is False
    assert result["imitation_script"]["publication_allowed"] is False
    assert (bundle / "exports" / "shot-breakdown.json").exists()
    assert (bundle / "exports" / "style-fingerprint.json").exists()
    assert (bundle / "exports" / "imitation-script.md").exists()
    assert (bundle / "runs" / "shot-breakdown" / "run.json").exists()
    run = json.loads((bundle / "runs" / "shot-breakdown" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "needs_review"
    assert run["resource_requirements"] == {"cpu": 1, "gpu": 0, "network": 0}


def test_shot_breakdown_preserves_unknowns_and_blocks_generation_readiness(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    result = build_shot_breakdown(bundle, write=False)

    assert result["shots"][0]["facts"]["shot_type"] == "unknown"
    assert result["shots"][0]["facts"]["camera_movement"] == "unknown"
    assert "shot_type" in result["shots"][0]["unknown_fields"]
    assert result["readiness"]["ready"] is False
    assert "shot_type" in result["readiness"]["shots"][0]["blockers"]
    assert result["imitation_script"]["shots"][0]["human_confirmed"] is False


def test_shot_breakdown_marks_structured_shot_fact_provenance(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _write_json(
        bundle / "exports" / "shot-facts.json",
        {
            "schema": "video_knowledge_pipeline.shot_facts.v1",
            "shot_count": 2,
            "shots": [
                {
                    "index": 1,
                    "fields": {
                        "camera_movement": {
                            "value": "static",
                            "status": "inferred",
                            "confidence": 0.9,
                            "evidence_ids": ["frame:fixture"],
                            "source": "auto_scenes.optical_flow_analyzer",
                            "missing_evidence": [],
                        }
                    },
                }
            ],
        },
    )

    result = build_shot_breakdown(bundle, write=False)

    shot = result["shots"][0]
    assert shot["facts"]["camera_movement"] == "static"
    assert shot["field_provenance"]["camera_movement"] == (
        "shot_facts.v1:auto_scenes.optical_flow_analyzer"
    )


def test_shot_breakdown_cli_mcp_args_and_workbench_surface(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    parsed = build_parser().parse_args(["shot-breakdown", str(bundle), "--no-write"])
    assert parsed.command == "shot-breakdown"
    assert cli_main(["shot-breakdown", str(bundle), "--no-write"]) == 0
    build_shot_breakdown(bundle, write=True)
    audit = audit_bundle_mcp_args(bundle)
    row = next(item for item in audit["rows"] if item["key"] == "mcp_shot_breakdown_args")
    assert row["tool"] == "shot_breakdown"
    assert row["ok"] is True

    workbench = export_video_workbench(bundle, write=False)
    keys = {item["key"] for item in workbench["artifacts"]}
    assert "shot_breakdown_markdown" in keys
    capability = next(item for item in workbench["external_reuse_status"]["capabilities"] if item["key"] == "shot_breakdown_and_imitation")
    assert capability["status"] == "action_required"
