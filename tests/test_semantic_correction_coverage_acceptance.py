from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.acceptance_check import (
    _acceptance_blockers,
    _acceptance_next_action,
)
from video_knowledge_pipeline.knowledge_coverage import (
    _transcript_semantic_correction_channel,
)


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "bundle"
    root.mkdir()
    transcript = root / "normalized-transcript.json"
    transcript.write_text(
        json.dumps(
            {"segments": [{"start": 0, "end": 10, "text": "合成语音。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root, {"normalized_transcript_json": transcript.name}


def test_unscanned_semantic_correction_is_a_machine_coverage_gap(
    tmp_path: Path, monkeypatch
) -> None:
    root, manifest = _bundle(tmp_path)
    monkeypatch.setattr(
        "video_knowledge_pipeline.transcript_semantic_correction.transcript_semantic_correction_status",
        lambda *_args, **_kwargs: {
            "status": "not_started",
            "ok": False,
            "candidate_discovery_status": "not_planned",
            "candidate_discovery_complete": False,
            "review_required_count": 0,
            "unadjudicated_candidate_count": 0,
            "artifact_identity": {"current": True},
            "next_action_key": "run_candidate_discovery",
        },
    )

    channel, summary = _transcript_semantic_correction_channel(
        root, manifest, transcript_item_count=1
    )

    assert channel["status"] == "blocked"
    assert channel["blocker_count"] == 1
    assert summary["candidate_discovery_status"] == "not_planned"
    assert summary["next_action_key"] == "run_candidate_discovery"


def test_human_review_is_weak_but_closed_semantics_are_covered(
    tmp_path: Path, monkeypatch
) -> None:
    root, manifest = _bundle(tmp_path)
    semantic_status = {
        "status": "impact_passed",
        "ok": False,
        "completion_gate": {
            "ok": False,
            "status": "blocked",
            "required_next_action_key": "review_candidates",
            "required_action_kind": "human",
        },
        "required_next_action_key": "review_candidates",
        "required_action_kind": "human",
        "candidate_discovery_status": "imported",
        "candidate_discovery_complete": True,
        "review_required_count": 2,
        "unadjudicated_candidate_count": 0,
        "artifact_identity": {"current": True},
        "corrected_transcript_exists": True,
        "next_action_key": "review_candidates",
    }
    monkeypatch.setattr(
        "video_knowledge_pipeline.transcript_semantic_correction.transcript_semantic_correction_status",
        lambda *_args, **_kwargs: semantic_status,
    )

    review_channel, review_summary = _transcript_semantic_correction_channel(
        root, manifest, transcript_item_count=1
    )
    assert review_channel["status"] == "weak"
    assert review_summary["review_required_count"] == 2

    semantic_status.update(
        {
            "ok": True,
            "completion_gate": {
                "ok": True,
                "status": "passed",
                "required_next_action_key": "none",
                "required_action_kind": "none",
            },
            "required_next_action_key": "none",
            "required_action_kind": "none",
            "review_required_count": 0,
        }
    )
    covered_channel, covered_summary = _transcript_semantic_correction_channel(
        root, manifest, transcript_item_count=1
    )
    assert covered_channel["status"] == "ok"
    assert covered_summary["artifact_identity_current"] is True


def test_acceptance_prioritizes_semantic_machine_work_and_high_risk_review() -> None:
    semantic = {
        "required": True,
        "status": "not_started",
        "ok": False,
        "completion_gate": {
            "ok": False,
            "status": "blocked",
            "required_next_action_key": "run_candidate_discovery",
            "required_action_kind": "machine",
        },
        "candidate_discovery_status": "not_planned",
        "candidate_discovery_complete": False,
        "artifact_identity_current": True,
        "review_required_count": 0,
        "unadjudicated_candidate_count": 0,
        "next_action_key": "run_candidate_discovery",
        "required_next_action_key": "run_candidate_discovery",
        "required_action_kind": "machine",
    }
    coverage = {
        "transcript_semantic_correction": semantic,
        "blockers": [],
    }
    blockers = _acceptance_blockers(coverage, {}, {}, {}, {})

    assert any(
        row["key"] == "transcript_semantic_correction_incomplete"
        and row["severity"] == "machine_action_available"
        for row in blockers
    )
    next_action = _acceptance_next_action(
        "machine_action_available", coverage, {}, {}, {}, {}, {}
    )
    assert next_action["key"] == "run_candidate_discovery"

    semantic.update(
        {
            "status": "impact_passed",
            "candidate_discovery_status": "imported",
            "candidate_discovery_complete": True,
            "review_required_count": 1,
            "completion_gate": {
                "ok": False,
                "status": "blocked",
                "required_next_action_key": "review_candidates",
                "required_action_kind": "human",
            },
            "required_next_action_key": "review_candidates",
            "required_action_kind": "human",
        }
    )
    blockers = _acceptance_blockers(coverage, {}, {}, {}, {})
    assert any(
        row["key"] == "transcript_semantic_high_risk_review"
        and row["severity"] == "human_review_required"
        for row in blockers
    )


def test_unadjudicated_candidates_block_coverage_and_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    root, manifest = _bundle(tmp_path)
    semantic_status = {
        "status": "impact_passed",
        "ok": False,
        "completion_gate": {
            "ok": False,
            "status": "blocked",
            "required_next_action_key": "validate_current_pack",
            "required_action_kind": "machine",
        },
        "required_next_action_key": "validate_current_pack",
        "required_action_kind": "machine",
        "candidate_discovery_status": "imported",
        "candidate_discovery_complete": True,
        "review_required_count": 0,
        "unadjudicated_candidate_count": 1,
        "artifact_identity": {"current": True},
        "corrected_transcript_exists": True,
        "next_action_key": "validate_current_pack",
    }
    monkeypatch.setattr(
        "video_knowledge_pipeline.transcript_semantic_correction.transcript_semantic_correction_status",
        lambda *_args, **_kwargs: semantic_status,
    )

    channel, summary = _transcript_semantic_correction_channel(
        root, manifest, transcript_item_count=1
    )

    assert channel["status"] == "blocked"
    assert channel["unadjudicated_candidate_count"] == 1
    assert summary["unadjudicated_candidate_count"] == 1
    coverage = {"transcript_semantic_correction": summary, "blockers": []}
    blockers = _acceptance_blockers(coverage, {}, {}, {}, {})
    assert any(
        row["key"] == "transcript_semantic_correction_incomplete"
        for row in blockers
    )
