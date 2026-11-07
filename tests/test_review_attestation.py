from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser, main as cli_main
from video_knowledge_pipeline.review_attestation import (
    create_review_attestation,
    validate_review_attestation,
)


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([{"index": 1, "review_status": "confirmed"}]), encoding="utf-8")
    (bundle / "review-notes.json").write_text(json.dumps({"reviews": [{"timeline_index": 1}]}), encoding="utf-8")
    return bundle


def test_review_attestation_is_immutable_and_becomes_stale(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    created = create_review_attestation(
        bundle,
        target="timeline-review",
        artifact_paths=[
            {"role": "timeline", "path": "timeline.json"},
            {"role": "review_notes", "path": "review-notes.json"},
        ],
        approved_by="operator-a",
        comment="confirmed",
        write=True,
    )
    path = Path(created["path"])
    original = path.read_bytes()

    valid = validate_review_attestation(bundle, target="timeline-review")
    assert valid["status"] == "valid"
    assert valid["passed"] is True

    (bundle / "timeline.json").write_text(json.dumps([{"index": 1, "review_status": "changed"}]), encoding="utf-8")
    stale = validate_review_attestation(bundle, target="timeline-review")
    assert stale["status"] == "stale"
    assert stale["passed"] is False
    assert any(issue["key"] == "input_changed" for issue in stale["issues"])
    assert path.read_bytes() == original


def test_review_attestation_requires_visible_operator_and_detects_record_tamper(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    try:
        create_review_attestation(bundle, target="timeline-review", artifact_paths=["timeline.json"], approved_by="")
    except ValueError as exc:
        assert "approved_by" in str(exc)
    else:
        raise AssertionError("missing approved_by should fail")

    created = create_review_attestation(
        bundle,
        target="timeline-review",
        artifact_paths=["timeline.json"],
        approved_by="operator-a",
    )
    path = Path(created["path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["comment"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validate_review_attestation(bundle, attestation_path=path)["status"] == "invalid"


def test_review_attestation_cli_contract(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    argv = [
        "review-attestation-create",
        str(bundle),
        "--target",
        "timeline-review",
        "--artifact",
        "timeline=timeline.json",
        "--artifact",
        "review_notes=review-notes.json",
        "--approved-by",
        "operator-a",
    ]
    parsed = build_parser().parse_args(argv)
    assert parsed.command == "review-attestation-create"
    assert cli_main(argv) == 0
    assert cli_main(
        ["review-attestation-status", str(bundle), "--target", "timeline-review"]
    ) == 0
