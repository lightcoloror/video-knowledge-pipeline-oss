from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.filmed_structure import (
    BEAT_SCHEMA,
    SCENE_SCHEMA,
    _run_pelt,
    build_filmed_structure_plan,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    rows = []
    embedding_rows = []
    for index in range(1, 7):
        shot_id = f"technical-shot-{index:04d}"
        evidence_id = f"timeline:{index}"
        rows.append(
            {
                "shot_id": shot_id,
                "index": index,
                "start": float(index - 1),
                "end": float(index),
                "fields": {
                    "shot_type": {
                        "value": "medium" if index <= 3 else "close_up",
                        "status": "inferred",
                        "confidence": 0.8,
                        "evidence_ids": [evidence_id],
                        "source": "fixture.visual",
                        "missing_evidence": [],
                    },
                    "camera_movement": {
                        "value": "static",
                        "status": "inferred",
                        "confidence": 0.8,
                        "evidence_ids": [evidence_id],
                        "source": "fixture.visual",
                        "missing_evidence": [],
                    },
                    "dialogue_or_narration": {
                        "value": f"dialogue {index}",
                        "status": "confirmed",
                        "confidence": 1.0,
                        "evidence_ids": [evidence_id],
                        "source": "timeline.transcript",
                        "missing_evidence": [],
                    },
                    "screen_text": {
                        "value": f"screen {index}",
                        "status": "confirmed",
                        "confidence": 1.0,
                        "evidence_ids": [evidence_id],
                        "source": "timeline.visual_text",
                        "missing_evidence": [],
                    },
                },
            }
        )
        embedding_rows.append(
            {
                "shot_id": shot_id,
                "embedding": [0.0, 0.0] if index <= 2 else (
                    [4.0, 4.0] if index <= 4 else [8.0, 8.0]
                ),
            }
        )
    _write_json(
        root / "exports" / "shot-facts.json",
        {
            "schema": "video_knowledge_pipeline.shot_facts.v1",
            "status": "completed",
            "shot_count": 6,
            "shots": rows,
        },
    )
    embeddings = tmp_path / "embeddings.json"
    _write_json(
        embeddings,
        {"model": "BAAI/bge-m3", "shots": embedding_rows},
    )
    return root, embeddings


def test_pelt_scene_plan_never_assigns_story_roles_by_position(
    tmp_path: Path,
) -> None:
    root, embeddings = _bundle(tmp_path)

    result = build_filmed_structure_plan(
        root,
        shot_embeddings_json=embeddings,
        local_story_route_id="local-qwen35-9b",
        write=False,
        _change_point_runner=lambda features, penalty, min_size, jump: [2, 4, 6],
    )

    scenes = result["semantic_scene_plan"]
    beats = result["story_beat_plan"]
    assert scenes["schema"] == SCENE_SCHEMA
    assert scenes["status"] == "completed"
    assert scenes["scene_count"] == 3
    assert [row["shot_ids"] for row in scenes["scenes"]] == [
        ["technical-shot-0001", "technical-shot-0002"],
        ["technical-shot-0003", "technical-shot-0004"],
        ["technical-shot-0005", "technical-shot-0006"],
    ]
    assert beats["schema"] == BEAT_SCHEMA
    assert beats["position_only_role_inference"] is False
    assert {row["role"] for row in beats["beats"]} == {"unknown"}
    assert all(row["status"] == "unavailable" for row in beats["beats"])
    assert result["highlight_plan"]["maximum_scene_duration_seconds"] == 150.0
    assert result["operator_boundary"]["no_whole_video_vlm"] is True


def test_story_evidence_must_bind_scene_evidence_ids(tmp_path: Path) -> None:
    root, embeddings = _bundle(tmp_path)
    story = tmp_path / "story.json"
    _write_json(
        story,
        {
            "scenes": [
                {
                    "scene_id": "semantic-scene-0001",
                    "role": "setup",
                    "summary": "directly supported setup",
                    "confidence": 0.82,
                    "evidence_ids": ["timeline:1"],
                },
                {
                    "scene_id": "semantic-scene-0002",
                    "role": "payoff",
                    "summary": "not actually supported",
                    "confidence": 0.9,
                    "evidence_ids": ["timeline:999"],
                },
            ]
        },
    )

    result = build_filmed_structure_plan(
        root,
        shot_embeddings_json=embeddings,
        story_evidence_json=story,
        local_story_route_id="local-qwen35-9b",
        write=False,
        _change_point_runner=lambda features, penalty, min_size, jump: [2, 4, 6],
    )

    beats = result["story_beat_plan"]["beats"]
    assert beats[0]["role"] == "setup"
    assert beats[0]["status"] == "inferred"
    assert beats[0]["evidence_ids"] == ["timeline:1"]
    assert beats[1]["role"] == "unknown"
    assert beats[1]["status"] == "unavailable"
    assert beats[1]["summary"] == ""


def test_missing_bge_embeddings_is_explicitly_degraded(tmp_path: Path) -> None:
    root, _ = _bundle(tmp_path)

    result = build_filmed_structure_plan(
        root,
        write=False,
        _change_point_runner=lambda features, penalty, min_size, jump: [3, 6],
    )

    scene = result["semantic_scene_plan"]
    assert scene["status"] == "degraded_missing_bge_m3_embeddings"
    assert scene["embedding_provenance"]["status"] == "unavailable"
    assert result["status"] == "degraded"


def test_actual_pinned_ruptures_pelt_smoke() -> None:
    pytest.importorskip("ruptures", reason="optional local extra is not installed")
    features = [[0.0, 0.0] for _ in range(8)] + [
        [4.0, 4.0] for _ in range(8)
    ]

    breakpoints = _run_pelt(features, 2.0, 2, 1)

    assert breakpoints == [8, 16]


def test_filmed_structure_blocks_without_shot_facts(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1"})

    result = build_filmed_structure_plan(root, write=False)

    assert result["status"] == "blocked_missing_shot_facts"
    assert result["semantic_scene_plan"]["scene_count"] == 0
    assert result["operator_boundary"]["no_position_only_story_roles"] is True
