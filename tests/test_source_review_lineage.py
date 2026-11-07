from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from video_knowledge_pipeline.source_review_lineage import (
    discover_source_review_lineage,
    validate_bound_source_review_lineage,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(root: Path, media: Path, *, reviewed: bool) -> Path:
    root.mkdir(parents=True)
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    video_id = f"video_{digest[:12]}"
    _write_json(
        root / "manifest.json",
        {"schema": "lecture_webui_bundle.v1", "title": "同源采访", "source_artifacts_json": "source-artifacts.json"},
    )
    _write_json(
        root / "source-artifacts.json",
        {
            "artifacts": [
                {"kind": "media", "key": "video", "video_id": video_id, "path": str(media), "sha256": digest}
            ]
        },
    )
    if reviewed:
        _write_json(
            root / "speaker-review.json",
            {"schema": "speaker_review.v1", "status": "human_confirmed_count", "confirmed_participant_count": 3, "speaker_mappings": []},
        )
    return root


def test_discovers_same_source_review_but_does_not_silently_apply(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"synthetic media")
    current = _bundle(tmp_path / "runs" / "new" / "webui-bundle", media, reviewed=False)
    prior = _bundle(tmp_path / "outputs" / "old" / "webui-bundle", media, reviewed=True)

    result = discover_source_review_lineage(current, search_roots=[tmp_path / "outputs"])

    assert result["status"] == "prior_review_available"
    assert result["candidate_count"] == 1
    assert result["selected_candidate"]["bundle_dir"] == str(prior.resolve())
    assert result["applied"] is False
    assert result["selected_candidate"]["review_artifacts"][0]["binding_capability"] == "participant_count_only"
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "video_knowledge_pipeline"
        / "schemas"
        / "source-review-lineage.v1.schema.json"
    )
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(result)


def test_explicit_apply_binds_hashes_and_detects_review_drift(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"synthetic media")
    current = _bundle(tmp_path / "runs" / "new" / "webui-bundle", media, reviewed=False)
    prior = _bundle(tmp_path / "outputs" / "old" / "webui-bundle", media, reviewed=True)

    result = discover_source_review_lineage(current, search_roots=[tmp_path / "outputs"], apply=True)
    validation = validate_bound_source_review_lineage(current)

    assert result["status"] == "review_lineage_bound"
    assert validation["passed"] is True
    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inherited_review_artifacts"][0]["human_confirmed"] is True

    _write_json(prior / "speaker-review.json", {"status": "changed"})
    drift = validate_bound_source_review_lineage(current)
    assert drift["passed"] is False
    assert any(value.startswith("review_artifact_hash_drift:") for value in drift["failures"])


def test_same_title_with_different_media_is_not_a_match(tmp_path: Path) -> None:
    current_media = tmp_path / "current.mp4"
    prior_media = tmp_path / "prior.mp4"
    current_media.write_bytes(b"current")
    prior_media.write_bytes(b"prior")
    current = _bundle(tmp_path / "runs" / "same" / "webui-bundle", current_media, reviewed=False)
    _bundle(tmp_path / "outputs" / "same" / "webui-bundle", prior_media, reviewed=True)

    result = discover_source_review_lineage(current, search_roots=[tmp_path / "outputs"])

    assert result["status"] == "no_prior_review_found"
    assert result["candidate_count"] == 0


def test_nonempty_review_notes_are_human_lineage_but_empty_template_is_not(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"same")
    current = _bundle(tmp_path / "runs" / "new" / "webui-bundle", media, reviewed=False)
    prior = _bundle(tmp_path / "outputs" / "old" / "webui-bundle", media, reviewed=False)
    _write_json(prior / "review-notes.json", {"reviews": []})

    empty = discover_source_review_lineage(current, search_roots=[tmp_path / "outputs"], write=False)
    assert empty["candidate_count"] == 0

    _write_json(prior / "review-notes.json", {"reviews": [{"segment_id": "s1", "corrected_text": "人工纠正"}]})
    reviewed = discover_source_review_lineage(current, search_roots=[tmp_path / "outputs"], write=False)
    assert reviewed["candidate_count"] == 1
    assert reviewed["selected_candidate"]["review_artifacts"][0]["human_confirmed"] is True
