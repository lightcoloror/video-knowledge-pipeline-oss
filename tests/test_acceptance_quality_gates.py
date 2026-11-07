from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.acceptance_check as acceptance


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "webui-bundle"
    (root / "exports").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1"}),
        encoding="utf-8",
    )
    (root / "timeline.json").write_text("[]", encoding="utf-8")
    return root


def _isolate_acceptance(monkeypatch) -> None:
    monkeypatch.setattr(
        acceptance,
        "audit_knowledge_coverage",
        lambda *_args, **_kwargs: {
            "coverage": {
                "status": "ok",
                "channels": [],
                "blockers": [],
                "weak_channels": [],
                "semantic_frame_without_analysis": 0,
                "temporal_sequence_without_analysis": 0,
            }
        },
    )
    monkeypatch.setattr(
        acceptance,
        "bundle_status_report",
        lambda *_args, **_kwargs: {
            "controlled_execution": {
                "status": "ready",
                "provider_health_status": "ready",
                "provider_health_safe_to_execute": True,
                "blockers": [],
            }
        },
    )
    monkeypatch.setattr(
        acceptance,
        "_note_quality",
        lambda _root: {"export_freshness": "fresh"},
    )
    monkeypatch.setattr(
        acceptance,
        "_review_lifecycle",
        lambda *_args: {"state": "not_prepared"},
    )
    monkeypatch.setattr(acceptance, "_review_closure", lambda _root: {})
    monkeypatch.setattr(acceptance, "_provider_matrix", lambda *_args: {})


def test_legacy_bundle_without_quality_artifacts_preserves_complete(
    tmp_path: Path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    _isolate_acceptance(monkeypatch)

    result = acceptance.acceptance_check(root, refresh=True, write=False)

    assert result["status"] == "complete"
    assert result["quality_gates"]["present"] is False


def test_transcript_quality_failed_makes_acceptance_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    _isolate_acceptance(monkeypatch)
    (root / "transcript-quality-gate.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_quality_gate.v1",
                "status": "failed",
                "ok": False,
                "fail_count": 2,
                "warning_count": 0,
            }
        ),
        encoding="utf-8",
    )

    result = acceptance.acceptance_check(root, refresh=True, write=False)

    assert result["status"] == "incomplete"
    assert result["next_action"]["key"] == "repair_transcript_quality"
    assert any(
        row["key"] == "transcript_quality_failed" for row in result["blockers"]
    )


def test_transcript_warning_is_accepted_only_with_known_gaps(
    tmp_path: Path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    _isolate_acceptance(monkeypatch)
    (root / "transcript-quality-gate.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_quality_gate.v1",
                "status": "warning",
                "ok": True,
                "fail_count": 0,
                "warning_count": 1,
            }
        ),
        encoding="utf-8",
    )

    result = acceptance.acceptance_check(root, refresh=True, write=False)

    assert result["status"] == "accepted_with_known_gaps"
    assert (
        result["summary"]["transcript_quality_classification"] == "warning"
    )


def test_smart_summary_failed_makes_acceptance_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    _isolate_acceptance(monkeypatch)
    (root / "exports" / "smart-summary-quality.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.smart_summary_quality.v1",
                "status": "failed",
                "passed": False,
                "automated_checks_passed": False,
                "production_ready": False,
            }
        ),
        encoding="utf-8",
    )

    result = acceptance.acceptance_check(root, refresh=True, write=False)

    assert result["status"] == "incomplete"
    assert result["next_action"]["key"] == "repair_smart_summary_quality"


def test_missing_human_key_points_requires_summary_review_not_visual_review(
    tmp_path: Path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    _isolate_acceptance(monkeypatch)
    (root / "exports" / "smart-summary-quality.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.smart_summary_quality.v1",
                "status": "blocked_missing_human_key_points",
                "passed": False,
                "automated_checks_passed": True,
                "production_ready": False,
            }
        ),
        encoding="utf-8",
    )

    result = acceptance.acceptance_check(root, refresh=True, write=False)

    assert result["status"] == "human_review_required"
    assert result["next_action"]["key"] == "provide_human_summary_key_points"
    assert result["next_action"]["mcp_tool"] == "smart_summary_quality_check"


def test_invalid_existing_quality_file_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    _isolate_acceptance(monkeypatch)
    (root / "transcript-quality-gate.json").write_text(
        "{not-json",
        encoding="utf-8",
    )

    result = acceptance.acceptance_check(root, refresh=True, write=False)

    assert result["status"] == "incomplete"
    assert (
        result["quality_gates"]["transcript"]["classification"] == "invalid"
    )
