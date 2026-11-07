from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser, main as cli_main
from video_knowledge_pipeline.review_attestation import create_review_attestation
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.video_edit_review_pack import (
    build_preference_evidence,
    build_token_silences,
    build_video_edit_review_pack,
    reclaim_silence_from_envelope,
    refine_edit_boundaries,
    validate_edit_artifacts,
)


def _tokens() -> list[dict[str, object]]:
    return [
        {"id": "tok-1", "start": 0.20, "end": 0.80, "original_text": "第", "isGap": False},
        {"id": "tok-2", "start": 1.20, "end": 1.80, "original_text": "一", "isGap": False},
        {"id": "tok-3", "start": 2.20, "end": 2.80, "original_text": "步", "isGap": False},
    ]


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    (bundle / "exports").mkdir(parents=True)
    write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "Edit fixture",
            "smart_summary_chapters": "exports/smart-summary-chapters.json",
        },
    )
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 3.0,
                "corrected_transcript": "第一步介绍方法。",
                "visual_text": "第一步",
                "temporal_visual_understanding": {"event_sequence": ["展示步骤"]},
                "temporal_frame_paths": ["temporal-frames/0001/frame_01.jpg"],
            },
            {"index": 2, "start": 3.0, "end": 6.0, "transcript": "第二步复核结果。"},
        ],
    )
    write_json(
        bundle / "exports" / "smart-summary-chapters.json",
        {
            "schema": "video_knowledge_pipeline.smart_summary_chapters.v1",
            "chapters": [
                {
                    "index": 1,
                    "title": "介绍与复核",
                    "start": 0.0,
                    "end": 6.0,
                    "summary_sentences": ["先介绍方法，再复核结果。"],
                    "citation_digest": [{"source_type": "temporal", "timeline_indexes": [1]}],
                }
            ],
        },
    )
    write_json(bundle / "transcript.tokens.json", _tokens())
    write_json(
        bundle / "edit.decisions.json",
        [{"id": "d-1", "start": 0.81, "end": 1.19, "kind": "silence", "action": "delete", "confirmed": True, "source": "user"}],
    )
    write_json(bundle / "delete_segments.json", [{"start": 0.83, "end": 1.15}])
    write_json(bundle / "ai_baseline.json", [])
    return bundle


def _attest_edit_handoff(bundle: Path) -> None:
    create_review_attestation(
        bundle,
        target="video-edit-handoff",
        artifact_paths=[
            {"role": "timeline", "path": "timeline.json"},
            {"role": "tokens", "path": "transcript.tokens.json"},
            {"role": "decisions", "path": "edit.decisions.json"},
            {"role": "delete_segments", "path": "delete_segments.json"},
            {"role": "ai_baseline", "path": "ai_baseline.json"},
        ],
        approved_by="test-operator",
    )


def test_boundary_refinement_uses_token_gaps_and_preserves_audit_fields() -> None:
    tokens = _tokens()
    silences = build_token_silences(tokens)
    result = refine_edit_boundaries(
        [{"id": "d-1", "start": 0.81, "end": 1.19, "action": "delete"}],
        tokens=tokens,
        silences=silences,
    )

    assert result["snapped_count"] == 1
    assert result["decisions"][0]["start"] == 0.83
    assert result["decisions"][0]["end"] == 1.15
    assert result["decisions"][0]["orig_start"] == 0.81
    assert result["decisions"][0]["orig_end"] == 1.19


def test_energy_envelope_reclaims_timestamp_overrun_silence() -> None:
    # Two voiced islands with a 120ms low-energy run between them.
    envelope = [-10.0] * 25 + [-42.0] * 12 + [-9.0] * 25
    words = [
        {"start": 0.0, "end": 0.32, "text": "前"},
        {"start": 0.30, "end": 0.62, "text": "后"},
    ]

    result = reclaim_silence_from_envelope(envelope, words, smooth_frames=0)

    assert result["reclaimed_count"] == 1
    assert result["reclaimed"][0] == {"start": 0.25, "end": 0.36}


def test_artifact_validation_accepts_boundary_refined_export_and_blocks_drift() -> None:
    tokens = _tokens()
    silences = build_token_silences(tokens)
    decisions = [{"start": 0.81, "end": 1.19, "action": "delete"}]

    accepted = validate_edit_artifacts(
        decisions,
        [{"start": 0.83, "end": 1.15}],
        tokens=tokens,
        silences=silences,
    )
    rejected = validate_edit_artifacts(
        decisions,
        [{"start": 0.70, "end": 1.15}],
        tokens=tokens,
        silences=silences,
    )

    assert accepted["ok"] is True
    assert accepted["comparison_mode"] == "boundary_refined"
    assert rejected["ok"] is False
    assert rejected["issues"][0]["reason"] == "delete_segments_mismatch"


def test_preference_evidence_requires_explicit_human_confirmation() -> None:
    baseline: list[dict[str, object]] = []
    final = [{"start": 0.85, "end": 1.15, "action": "delete", "kind": "silence", "reason": "long_gap"}]

    draft = build_preference_evidence(baseline, final, tokens=_tokens(), video_id="video-a", human_confirmed=False)
    confirmed = build_preference_evidence(baseline, final, tokens=_tokens(), video_id="video-a", human_confirmed=True)

    assert draft["eligible_difference_count"] == 0
    assert draft["differences"][0]["eligible_for_learning"] is False
    assert confirmed["eligible_difference_count"] == 1
    assert confirmed["differences"][0]["status"] == "observing"
    assert confirmed["differences"][0]["promotion_eligible"] is False


def test_review_pack_blocks_unconfirmed_edit_decisions(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    write_json(
        bundle / "edit.decisions.json",
        [{"id": "d-1", "start": 0.81, "end": 1.19, "kind": "silence", "action": "delete", "confirmed": False, "source": "ai"}],
    )

    result = build_video_edit_review_pack(bundle, write=False)

    assert result["edit_decisions_human_confirmed"] is False
    assert result["ready_for_single_ffmpeg"] is False
    assert any(row["reason"] == "unconfirmed_edit_decisions" for row in result["blockers"])


def test_review_pack_writes_storyboard_validation_manifest_run_and_cli(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _attest_edit_handoff(bundle)

    result = build_video_edit_review_pack(bundle, human_confirmed_diff=True, write=True)

    assert result["storyboard_scene_count"] == 1
    scene = result["storyboard_scenes"][0]
    assert scene["source"] == "vkp_smart_summary_candidate"
    assert scene["screen"] == "broll"
    assert scene["confirmed"] is False
    assert scene["evidence"]["citation_digest"]
    assert result["artifact_validation"]["ok"] is True
    assert result["ready_for_single_ffmpeg"] is True
    assert result["review_attestation"]["status"] == "valid"
    assert result["dependency_snapshot"]["snapshot_sha256"]
    assert result["preference_evidence"]["eligible_difference_count"] == 1
    assert (bundle / "exports" / "video-edit-review-pack.json").exists()
    assert (bundle / "exports" / "storyboard.candidates.json").exists()
    assert (bundle / "runs" / "video-edit-review-pack" / "run.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_edit_review_pack_markdown"] == "exports/video-edit-review-pack.md"
    audit = audit_bundle_mcp_args(bundle)
    assert manifest["video_edit_dependency_snapshot"] == "exports/video-edit-dependency-snapshot.json"
    edit_row = next(row for row in audit["rows"] if row["key"] == "mcp_video_edit_review_pack_args")
    assert edit_row["tool"] == "video_edit_review_pack"
    assert edit_row["ok"] is True
    run = json.loads((bundle / "runs" / "video-edit-review-pack" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "needs_review"

    parsed = build_parser().parse_args(["video-edit-review-pack", str(bundle), "--no-write"])
    assert parsed.command == "video-edit-review-pack"
    assert cli_main(["video-edit-review-pack", str(bundle), "--no-write"]) == 0
