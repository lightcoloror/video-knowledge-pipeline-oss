from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.scene_candidate_evidence import build_scene_candidate_evidence
from video_knowledge_pipeline.scene_taxonomy import explain_quality_for_shot_type


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_json(bundle / "manifest.json", {"duration_seconds": 60})
    _write_json(bundle / "timeline.json", [{"index": 1, "start": 0, "end": 60}])
    candidates = tmp_path / "candidates.json"
    _write_json(
        candidates,
        {
            "boundaries": [
                {
                    "seconds": 12.5,
                    "score": 0.91,
                    "taxonomy": {
                        "shot_type": "close-up",
                        "camera_movement": "static",
                    },
                    "quality": {"grade": "reject", "issues": ["blur"]},
                    "asr_evidence_ids": ["asr-1"],
                    "ocr_evidence_ids": ["ocr-1"],
                    "visual_evidence_ids": ["vision-1"],
                },
                {"seconds": 35.0, "score": 0.72, "reason": "feature_local_minimum"},
            ]
        },
    )
    taxonomy_prompt = tmp_path / "taxonomy.txt"
    taxonomy_prompt.write_text("shot type and camera movement taxonomy v1", encoding="utf-8")
    return bundle, candidates, taxonomy_prompt


def test_scene_candidates_are_content_addressed_and_never_mutate_timeline(tmp_path: Path) -> None:
    bundle, candidates, taxonomy_prompt = _fixture(tmp_path)
    timeline_path = bundle / "timeline.json"
    timeline_before = timeline_path.read_bytes()

    result = build_scene_candidate_evidence(
        bundle,
        candidates,
        model_id="fg-clip-large",
        model_commit="model-commit-1",
        language="zh-CN",
        taxonomy_prompt=taxonomy_prompt,
        write=True,
    )

    assert result["schema"] == "video_knowledge_pipeline.scene_candidate_evidence.v1"
    assert result["status"] == "needs_human_review"
    assert result["candidate_count"] == 2
    assert result["provenance"]["record_count"] == 2
    assert result["provenance"]["model_commit"] == "model-commit-1"
    assert result["provenance"]["language"] == "zh-CN"
    assert len(result["provenance"]["taxonomy_prompt_sha256"]) == 64
    assert result["candidates"][0]["candidate_only"] is True
    assert result["candidates"][0]["export_eligible"] is False
    assert result["candidates"][0]["taxonomy"]["normalized"]["shot_type"] == "close_up"
    explanation = result["candidates"][0]["shot_quality_explanation"]
    assert explanation["raw_quality"]["grade"] == "reject"
    assert explanation["raw_grade_preserved"] is True
    assert explanation["contextual_disposition"] == "review_context"
    assert timeline_path.read_bytes() == timeline_before
    assert result["timeline_invariant"]["timeline_sha256"] == hashlib.sha256(timeline_before).hexdigest()
    assert (bundle / "exports" / "scene-candidate-evidence.json").is_file()
    assert (bundle / "scene-candidate-review.todo.json").is_file()
    assert result["run_artifact"]["resource_requirements"] == {"cpu": 1, "gpu": 0, "network": 0}


def test_scene_candidate_cache_identity_is_stable_and_invalidates_on_model_change(tmp_path: Path) -> None:
    bundle, candidates, taxonomy_prompt = _fixture(tmp_path)
    first = build_scene_candidate_evidence(
        bundle,
        candidates,
        model_id="fg-clip-large",
        model_commit="model-commit-1",
        language="zh-CN",
        taxonomy_prompt=taxonomy_prompt,
        write=False,
    )
    same = build_scene_candidate_evidence(
        bundle,
        candidates,
        model_id="fg-clip-large",
        model_commit="model-commit-1",
        language="zh-CN",
        taxonomy_prompt=taxonomy_prompt,
        write=False,
    )
    changed = build_scene_candidate_evidence(
        bundle,
        candidates,
        model_id="fg-clip-large-v2",
        model_commit="model-commit-2",
        language="zh-CN",
        taxonomy_prompt=taxonomy_prompt,
        write=False,
    )

    assert first["provenance"]["cache_identity_sha256"] == same["provenance"]["cache_identity_sha256"]
    assert [row["candidate_id"] for row in first["candidates"]] == [
        row["candidate_id"] for row in same["candidates"]
    ]
    assert first["provenance"]["cache_identity_sha256"] != changed["provenance"]["cache_identity_sha256"]


def test_scene_candidate_requires_taxonomy_or_prompt_provenance(tmp_path: Path) -> None:
    bundle, candidates, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="taxonomy_prompt is required"):
        build_scene_candidate_evidence(
            bundle,
            candidates,
            model_id="fg-clip-large",
            language="zh-CN",
            taxonomy_prompt="",
            write=False,
        )


def test_scene_taxonomy_quality_explanation_preserves_raw_grade() -> None:
    result = explain_quality_for_shot_type(
        {"grade": "reject", "issues": ["soft focus"]},
        "close-up",
    )
    assert result["raw_quality"]["grade"] == "reject"
    assert result["contextual_disposition"] == "review_context"
    assert result["display_and_search_only"] is True


def test_scene_candidate_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "scene-candidate-evidence",
            "bundle",
            "candidates.json",
            "--model-id",
            "fg-clip-large",
            "--taxonomy-prompt",
            "taxonomy-v1",
            "--no-write",
        ]
    )
    assert args.command == "scene-candidate-evidence"
    assert args.model_id == "fg-clip-large"
    assert args.no_write is True

def test_transnetv2_saved_scenes_use_shot_candidate_contract(tmp_path: Path) -> None:
    bundle, _, taxonomy_prompt = _fixture(tmp_path)
    predictions = tmp_path / "transnetv2-scenes.json"
    _write_json(
        predictions,
        {
            "fps": 25,
            "threshold": 0.5,
            "scenes": [[0, 99], [100, 199], [200, 299]],
        },
    )

    result = build_scene_candidate_evidence(
        bundle,
        predictions,
        model_id="transnetv2",
        model_commit="85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed",
        taxonomy_prompt=taxonomy_prompt,
        source_format="transnetv2_scenes",
        write=False,
    )

    assert result["candidate_kind"] == "shot"
    assert [row["seconds"] for row in result["candidates"]] == [4.0, 8.0]
    assert all(
        row["schema"] == "video_knowledge_pipeline.candidate_shot_boundary.v1"
        for row in result["candidates"]
    )
    assert result["candidates"][0]["candidate_id"].startswith("shot-boundary-")
    assert result["candidates"][0]["source_coordinates"] == {
        "source_scene_index": 2,
        "start_frame": 100,
        "end_frame": 199,
    }
    provenance = result["provenance"]
    assert provenance["source_format"] == "transnetv2_scenes"
    assert provenance["frame_rate"] == 25.0
    assert provenance["prediction_threshold"] == 0.5
    assert provenance["model_execution_performed"] is False
    assert provenance["upstream_reference"]["api"] == "predictions_to_scenes"


def test_autoshot_saved_scenes_accept_dict_rows_and_explicit_fps(tmp_path: Path) -> None:
    bundle, _, taxonomy_prompt = _fixture(tmp_path)
    predictions = tmp_path / "autoshot-scenes.json"
    _write_json(
        predictions,
        {
            "scenes": [
                {"start_frame": 0, "end_frame": 49},
                {"start_frame": 50, "end_frame": 99},
            ]
        },
    )

    result = build_scene_candidate_evidence(
        bundle,
        predictions,
        model_id="autoshot",
        taxonomy_prompt=taxonomy_prompt,
        source_format="autoshot_scenes",
        frame_rate=25,
        write=False,
    )

    assert result["candidate_count"] == 1
    assert result["candidates"][0]["seconds"] == 2.0
    assert (
        result["provenance"]["upstream_reference"]["commit"]
        == "77c82ff826a9301bb173d9be786297a49d73d081"
    )


def test_saved_scenes_reject_overlap_and_missing_fps(tmp_path: Path) -> None:
    bundle, _, taxonomy_prompt = _fixture(tmp_path)
    predictions = tmp_path / "invalid-scenes.json"
    _write_json(predictions, {"scenes": [[0, 50], [50, 100]]})

    with pytest.raises(ValueError, match="frame_rate must be positive"):
        build_scene_candidate_evidence(
            bundle,
            predictions,
            model_id="transnetv2",
            taxonomy_prompt=taxonomy_prompt,
            source_format="transnetv2_scenes",
            write=False,
        )

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        build_scene_candidate_evidence(
            bundle,
            predictions,
            model_id="transnetv2",
            taxonomy_prompt=taxonomy_prompt,
            source_format="transnetv2_scenes",
            frame_rate=25,
            write=False,
        )


def test_scene_candidate_cli_accepts_saved_prediction_contract() -> None:
    args = build_parser().parse_args(
        [
            "scene-candidate-evidence",
            "bundle",
            "predictions.json",
            "--model-id",
            "transnetv2",
            "--taxonomy-prompt",
            "taxonomy-v1",
            "--source-format",
            "transnetv2_scenes",
            "--frame-rate",
            "25",
            "--no-write",
        ]
    )

    assert args.source_format == "transnetv2_scenes"
    assert args.frame_rate == 25.0
