from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from video_knowledge_pipeline.personal_vlog import (
    EVIDENCE_SCHEMA,
    EVENT_PLAN_SCHEMA,
    PRESETS,
    PROJECT_SCHEMA,
    RECIPE_SCHEMA,
    PersonalVlogBoundaryError,
    PersonalVlogDriftError,
    build_asset_evidence,
    build_edit_recipes,
    build_event_plan,
    prepare_personal_vlog_project,
    public_recipe_projection,
    validate_project_sources,
)


def _write_fixture(root: Path, relative: str, content: bytes, offset: int) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    timestamp = 1_700_000_000 + offset
    os.utime(path, (timestamp, timestamp))
    return path


def _event_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "event"
    project = tmp_path / "project"
    source.mkdir()
    _write_fixture(source, "01-start.JPG", b"synthetic-image-one", 1)
    _write_fixture(source, "day/02-play.MP4", b"synthetic-video-one", 2)
    _write_fixture(source, "day/03-laugh.WAV", b"synthetic-audio-one", 3)
    _write_fixture(source, "notes.txt", b"unsupported but receipted", 4)
    return source, project


def _build_pipeline(tmp_path: Path):
    source, project_dir = _event_fixture(tmp_path)
    project = prepare_personal_vlog_project(source, project_dir, write=False)
    imported = {
        asset["asset_id"]: {
            "duration_seconds": (
                18.0
                if asset["media_type"] == "video"
                else 12.0 if asset["media_type"] == "audio" else None
            ),
            "width": 1920 if asset["media_type"] != "audio" else None,
            "height": 1080 if asset["media_type"] != "audio" else None,
            "technical_quality": 0.8,
            "activity_relevance": 0.9,
            "audio_value": 0.8 if asset["media_type"] != "image" else 0.0,
            "tags": ["play", "outdoors"],
            "appearance_cluster_ids": ["appearance_cluster_01"],
            "location_cluster_id": "location_cluster_01",
            "has_precise_location": True,
            # Inputs that must never be copied into public recipes.
            "latitude": 22.123456,
            "longitude": 113.123456,
            "face_embedding": [0.1, 0.2],
        }
        for asset in project["assets"]
    }
    evidence = build_asset_evidence(project, imported=imported)
    events = build_event_plan(project, evidence, manual_boundaries=[2, 3])
    recipes = build_edit_recipes(project, evidence, events, presets="all")
    return project, evidence, events, recipes


def test_prepare_inventory_is_bounded_content_addressed_and_deterministic(tmp_path: Path):
    source, project_dir = _event_fixture(tmp_path)

    first = prepare_personal_vlog_project(source, project_dir, write=False)
    second = prepare_personal_vlog_project(
        source, tmp_path / "another-project", write=False
    )

    assert first["schema"] == PROJECT_SCHEMA
    assert first["counts"] == {
        "enumerated": 4,
        "supported": 3,
        "unsupported": 1,
        "images": 1,
        "videos": 1,
        "audio": 1,
    }
    assert first["project_identity_sha256"] == second["project_identity_sha256"]
    assert first["project_id"] == second["project_id"]
    assert first["unsupported_files"][0]["status"] == "unsupported_format"
    assert all(asset["source_read_only"] for asset in first["assets"])
    assert all(Path(asset["source_path"]).is_absolute() for asset in first["assets"])
    assert not project_dir.exists()


def test_prepare_writes_only_project_metadata_and_never_copies_sources(tmp_path: Path):
    source, project_dir = _event_fixture(tmp_path)
    source_before = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    manifest = prepare_personal_vlog_project(source, project_dir, write=True)

    assert (project_dir / "personal-vlog-project.json").is_file()
    assert not (project_dir / "01-start.JPG").exists()
    assert manifest["source_policy"]["copied_into_project"] is False
    assert source_before == {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (lambda root: _write_fixture(root, ".hidden.JPG", b"x", 1), "hidden_entry"),
        (
            lambda root: _write_fixture(root, "a/b/c/deep.JPG", b"x", 1),
            "maximum_depth_exceeded",
        ),
    ],
)
def test_prepare_fails_closed_on_hidden_or_too_deep_entries(
    tmp_path: Path, setup, expected: str
):
    source = tmp_path / "event"
    source.mkdir()
    setup(source)

    with pytest.raises(PersonalVlogBoundaryError, match=expected):
        prepare_personal_vlog_project(source, tmp_path / "project", write=False)


def test_prepare_fails_closed_on_count_limit_and_overlapping_roots(tmp_path: Path):
    source = tmp_path / "event"
    source.mkdir()
    _write_fixture(source, "one.jpg", b"1", 1)
    _write_fixture(source, "two.jpg", b"2", 2)

    with pytest.raises(PersonalVlogBoundaryError, match="maximum_item_count"):
        prepare_personal_vlog_project(
            source, tmp_path / "project", max_items=1, write=False
        )
    with pytest.raises(PersonalVlogBoundaryError, match="must_not_overlap"):
        prepare_personal_vlog_project(source, source / "output", write=False)


def test_prepare_rejects_symlink_when_platform_allows_it(tmp_path: Path):
    source = tmp_path / "event"
    source.mkdir()
    target = _write_fixture(source, "real.jpg", b"real", 1)
    link = source / "linked.jpg"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows account")

    with pytest.raises(PersonalVlogBoundaryError, match="symlink|reparse"):
        prepare_personal_vlog_project(source, tmp_path / "project", write=False)


def test_source_hash_drift_is_rejected_before_evidence(tmp_path: Path):
    source, project_dir = _event_fixture(tmp_path)
    project = prepare_personal_vlog_project(source, project_dir, write=False)
    (source / "01-start.JPG").write_bytes(b"changed-after-inventory")

    with pytest.raises(PersonalVlogDriftError, match="source_hash_drift"):
        validate_project_sources(project)
    with pytest.raises(PersonalVlogDriftError, match="source_hash_drift"):
        build_asset_evidence(project)


def test_evidence_event_plan_and_recipes_preserve_contract_and_privacy(tmp_path: Path):
    project, evidence, events, recipes = _build_pipeline(tmp_path)

    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert events["schema"] == EVENT_PLAN_SCHEMA
    assert [event["member_asset_ids"] for event in events["events"]] == [
        [project["assets"][0]["asset_id"], project["assets"][1]["asset_id"]],
        [project["assets"][2]["asset_id"]],
    ]
    assert set(recipes) == set(PRESETS)
    for name, recipe in recipes.items():
        assert recipe["schema"] == RECIPE_SCHEMA
        assert recipe["preset"] == name
        assert recipe["target"] == PRESETS[name]
        assert recipe["output_mode"] == "candidate_only"
        assert recipe["review_required"] is True
        assert all(clip["candidate_only"] for clip in recipe["clips"])
        assert all(clip["deleted"] is False for clip in recipe["clips"])
        projected = public_recipe_projection(recipe)
        text = json.dumps(projected, ensure_ascii=False)
        assert str(project["event_root"]) not in text
        assert "22.123456" not in text
        assert "113.123456" not in text
        assert "face_embedding" not in text
        assert '"source_path":' not in text


def test_three_recipe_hashes_are_stable_and_vary_by_preset(tmp_path: Path):
    project, evidence, events, first = _build_pipeline(tmp_path)
    second = build_edit_recipes(project, evidence, events, presets="all")

    assert {name: row["recipe_sha256"] for name, row in first.items()} == {
        name: row["recipe_sha256"] for name, row in second.items()
    }
    assert len({row["recipe_sha256"] for row in first.values()}) == 3
    assert first["highlight"]["target"]["aspect_ratio"] == "9:16"
    assert first["standard"]["target"]["aspect_ratio"] == "16:9"
    assert first["long"]["target"]["aspect_ratio"] == "16:9"


def test_low_confidence_unselected_assets_are_never_marked_deleted(tmp_path: Path):
    source, project_dir = _event_fixture(tmp_path)
    for number in range(50):
        _write_fixture(source, f"extra/{number:02d}.jpg", f"image-{number}".encode(), 10 + number)
    project = prepare_personal_vlog_project(source, project_dir, write=False)
    imported = {
        asset["asset_id"]: {
            "technical_quality": 0.1,
            "activity_relevance": 0.1,
            "evidence_level": "candidate_weak",
        }
        for asset in project["assets"]
    }
    evidence = build_asset_evidence(project, imported=imported)
    events = build_event_plan(project, evidence, manual_boundaries=[len(project["assets"])])
    recipe = build_edit_recipes(project, evidence, events, presets=["highlight"])[
        "highlight"
    ]

    assert recipe["unselected_candidates"]
    assert all(row["deleted"] is False for row in recipe["unselected_candidates"])
    assert all(row["review_required"] for row in recipe["unselected_candidates"])


def test_manual_event_boundary_and_evidence_identity_drift_fail_closed(tmp_path: Path):
    project, evidence, _events, _recipes = _build_pipeline(tmp_path)
    with pytest.raises(PersonalVlogBoundaryError, match="out_of_range"):
        build_event_plan(project, evidence, manual_boundaries=[999])

    altered = dict(evidence)
    altered["project_identity_sha256"] = "0" * 64
    with pytest.raises(PersonalVlogDriftError, match="project_identity_mismatch"):
        build_event_plan(project, altered)
