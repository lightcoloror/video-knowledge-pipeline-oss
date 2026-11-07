from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.general_tagger_adapter import (
    general_tagger_status,
    run_general_tagger,
)
from video_knowledge_pipeline.highlight_detection_adapter import run_highlight_detection
from video_knowledge_pipeline.media_capability_registry import media_capability_registry_status
from video_knowledge_pipeline.video_structure import build_video_structure


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    _write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "duration_seconds": 180,
        },
    )
    _write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 90,
                "frame_paths": [str(frame)],
                "transcript": "先介绍客户不回复的常见原因。",
                "visual_text": "客户不回复",
            },
            {
                "index": 2,
                "start": 90,
                "end": 180,
                "frame_paths": [str(frame)],
                "transcript": "接下来给出处理步骤和示例。",
                "visual_text": "处理步骤",
            },
        ],
    )
    input_pack = bundle / "exports" / "smart-summary-input-pack.json"
    _write_json(
        input_pack,
        {
            "title": "测试视频",
            "transcript_segments": [
                {
                    "index": 1,
                    "timeline_index": 1,
                    "start": 0,
                    "end": 90,
                    "raw_text": "先介绍客户不回复的常见原因。",
                    "evidence_inputs": {"tagger": ["讲师", "办公场景"]},
                },
                {
                    "index": 2,
                    "timeline_index": 2,
                    "start": 90,
                    "end": 180,
                    "raw_text": "接下来给出处理步骤和示例。",
                    "evidence_inputs": {"tagger": ["步骤", "演示"]},
                },
            ],
            "visual_digest": {"items": []},
        },
    )
    return bundle, input_pack


def test_local_video_structure_catalog_excludes_native_whole_video() -> None:
    status = media_capability_registry_status()

    local = {row["task"]: row for row in status["local_video_structure_capabilities"]}
    assert set(local) == {
        "shot_boundary_detection",
        "semantic_scene_segmentation",
        "storyline_structure",
        "highlight_detection",
        "general_image_tagging",
    }
    assert local["highlight_detection"]["implementation"] == "lighthouse_cg_detr"
    assert local["general_image_tagging"]["default_model"] == "ram_plus_swin_large_14m"
    assert status["excluded_capabilities"] == [
        {
            "task": "native_whole_video_understanding",
            "status": "disabled",
            "reason": "cost_latency_and_compute_policy",
        }
    ]


def test_video_structure_composes_shots_semantic_scenes_storyline_and_imported_highlights(
    tmp_path: Path,
) -> None:
    bundle, input_pack = _bundle(tmp_path)
    predictions = tmp_path / "highlights.json"
    _write_json(
        predictions,
        {
            "query": "找出可执行的客户沟通建议",
            "pred_relevant_windows": [[92.0, 145.0, 0.94]],
            "pred_saliency_scores": [0.1, 0.9],
        },
    )

    result = build_video_structure(
        bundle,
        title="测试视频",
        input_pack=input_pack,
        run_shot_detection=False,
        highlight_predictions_json=predictions,
    )

    assert result["status"] == "completed"
    assert result["shot_boundary_detection"]["status"] == "not_run"
    assert result["semantic_scene_segmentation"]["status"] == "completed"
    assert result["storyline_structure"]["items"]
    assert result["highlight_detection"]["status"] == "completed_import"
    assert result["highlight_detection"]["highlights"][0]["candidate_only"] is True
    assert result["operator_boundary"]["native_whole_video_understanding_enabled"] is False
    assert result["operator_boundary"]["cloud_calls_made"] == 0
    assert (bundle / "exports" / "video-structure.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_structure_json"] == "exports/video-structure.json"


def test_highlight_detection_preview_and_saved_prediction_import(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    predictions = tmp_path / "prediction.json"
    _write_json(
        predictions,
        {
            "pred_relevant_windows": [[3.0, 8.5, 0.75], [10.0, 15.0, 0.6]],
            "pred_saliency_scores": [0.4, 0.8, 0.3],
        },
    )

    preview = run_highlight_detection(
        bundle,
        query="人物完成关键动作",
        execute=False,
        write=False,
    )
    imported = run_highlight_detection(
        bundle,
        query="人物完成关键动作",
        predictions_json=predictions,
        write=True,
    )

    assert preview["status"] == "planned"
    assert preview["model"] == "cg_detr"
    assert preview["operator_boundary"]["no_remote_call"] is True
    assert preview["operator_boundary"]["gpu_required"] is True
    assert imported["status"] == "completed_import"
    assert len(imported["highlights"]) == 2
    assert imported["highlights"][0]["source"] == "lighthouse_cg_detr"
    assert (bundle / "exports" / "highlight-detection.json").exists()


def test_ram_plus_general_tagger_executes_injected_local_backend_and_imports_evidence(
    tmp_path: Path,
) -> None:
    bundle, _ = _bundle(tmp_path)
    checkpoint = tmp_path / "ram_plus_swin_large_14m.pth"
    checkpoint.write_bytes(b"fake-checkpoint")

    def backend(frame_path: Path) -> dict[str, object]:
        assert frame_path.exists()
        return {
            "tags_en": ["person", "presentation", "office"],
            "tags_zh": ["人物", "演示", "办公场景"],
        }

    result = run_general_tagger(
        bundle,
        checkpoint_path=checkpoint,
        execute=True,
        _inference_backend=backend,
    )

    assert result["status"] == "completed"
    assert result["model"] == "ram_plus_swin_large_14m"
    assert result["annotation_count"] == 2
    assert result["operator_boundary"]["remote_calls_made"] == 0
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert "办公场景" in timeline[0]["tagger_tags"]
    annotation = timeline[0]["tagger_annotations"][0]
    assert annotation["model"] == "ram_plus_swin_large_14m"
    assert annotation["artifact_sha256"]
    assert annotation["candidate_only"] is True
    assert (bundle / "exports" / "general-tagger.json").exists()


def test_general_tagger_reads_integrated_visual_evidence_frames(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    timeline_path = bundle / "timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    for row in timeline:
        row["integrated_visual"] = {
            "evidence_frame_paths": row.pop("frame_paths"),
        }
    _write_json(timeline_path, timeline)

    result = run_general_tagger(
        bundle,
        execute=True,
        import_annotations=False,
        write=False,
        _inference_backend=lambda _path: {
            "tags_en": ["person"],
            "tags_zh": ["人物"],
        },
    )

    assert result["status"] == "completed"
    assert result["annotation_count"] == 2
    assert result["annotations"][0]["artifact_path"].endswith("frame.jpg")

def test_general_tagger_status_selects_ram_plus_and_keeps_cl_compatibility(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ram-plus.pth"
    checkpoint.write_bytes(b"checkpoint")

    status = general_tagger_status(checkpoint_path=checkpoint)

    assert status["selected_model"] == "ram_plus_swin_large_14m"
    assert status["device_policy"]["gpu_required"] is True
    assert status["device_policy"]["cpu_fallback_allowed"] is False
    assert status["compatibility_baselines"] == ["cl_tagger", "wd_eva02_large_tagger_v3"]
    assert status["automatic_remote_fallback"] is False
    assert status["threshold_floor"] == 0.75


def test_cli_exposes_video_structure_highlight_and_general_tagger_front_doors() -> None:
    parser = build_parser()

    assert parser.parse_args(["video-structure", "bundle"]).command == "video-structure"
    highlight = parser.parse_args(["highlight-detection", "bundle", "--query", "动作"])
    assert highlight.command == "highlight-detection"
    assert highlight.device == "cuda"
    assert parser.parse_args(["general-tagger-status"]).command == "general-tagger-status"
    tagger = parser.parse_args(["run-general-tagger", "bundle"])
    assert tagger.command == "run-general-tagger"
    assert tagger.device == "cuda"
