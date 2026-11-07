from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.shot_review import (
    APPLIED_SCHEMA,
    NOTES_SCHEMA,
    apply_shot_review_notes,
    build_shot_review_template,
    shot_review_status,
)
from video_knowledge_pipeline.shot_language_analysis import run_shot_language_analysis
from video_knowledge_pipeline.technical_shot_detection import load_verified_technical_shots


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(root / "manifest.json", {"title": "review fixture"})
    _write_json(root / "timeline.json", [])
    _write_json(
        root / "exports" / "technical-shot-boundaries.json",
        {
            "schema": "video_knowledge_pipeline.technical_shot_boundaries.v1",
            "status": "completed",
            "ok": True,
            "boundary_kind": "technical_shot",
            "backend": "autoshot",
            "shots": [
                {"shot_id": "technical-shot-0001", "index": 1, "start": 0.0, "end": 1.0},
                {"shot_id": "technical-shot-0002", "index": 2, "start": 1.0, "end": 2.0},
            ],
        },
    )
    return root


def test_review_draft_is_hash_bound_and_formal_apply_is_derived(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    notes = build_shot_review_template(root, write=False)
    assert notes["schema"] == NOTES_SCHEMA
    notes["review_status"] = "human_confirmed"
    notes["review_id"] = "review-001"
    notes["shots"][0]["end"] = 1.1
    notes["shots"][1]["start"] = 1.1
    notes["field_corrections"] = [
        {"shot_id": "technical-shot-0001", "field": "shot_type", "value": "close_up"}
    ]

    result = apply_shot_review_notes(root, notes, write=True)

    assert result["schema"] == APPLIED_SCHEMA
    assert result["status"] == "completed"
    assert (root / "exports" / "technical-shot-boundaries.json").is_file()
    reviewed_path = root / "exports" / "technical-shot-boundaries.reviewed.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    assert reviewed["backend"] == "human_review"
    assert reviewed["shots"][0]["end"] == 1.1
    assert reviewed["timeline_mutated"] is False
    assert shot_review_status(root)["status"] == "active"
    shots, provenance = load_verified_technical_shots(root)
    assert shots[0]["end"] == 1.1
    assert provenance["compatibility"] == "human_reviewed_projection"

    facts = run_shot_language_analysis(root, execute=False, write=False)
    assert facts["shots"][0]["fields"]["shot_type"]["value"] == "close_up"
    assert facts["shots"][0]["fields"]["shot_type"]["status"] == "confirmed"
    assert facts["shot_review_provenance"]["review_id"] == "review-001"


def test_apply_rejects_stale_source_hash(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    notes = build_shot_review_template(root, write=False)
    notes["review_status"] = "human_confirmed"
    notes["review_id"] = "review-stale"
    source = root / "exports" / "technical-shot-boundaries.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source changed"):
        apply_shot_review_notes(root, notes, write=False)


def test_draft_cannot_be_formally_applied(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    notes = build_shot_review_template(root, write=False)
    with pytest.raises(ValueError, match="human_confirmed"):
        apply_shot_review_notes(root, notes, write=False)
