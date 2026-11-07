from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import video_knowledge_pipeline.transcript_quality_gate as quality_gate_module
import video_knowledge_pipeline.transcript_semantic_batch as semantic_batch_module
from video_knowledge_pipeline.transcript_evidence_correction_pipeline import (
    _candidate_statistics,
    _pipeline_ok,
    _pipeline_status,
)
from video_knowledge_pipeline.transcript_quality_gate import (
    run_transcript_quality_gate,
)
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.transcript_semantic_correction import (
    transcript_semantic_correction_status,
)


def _status_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "candidate_count": 0,
        "semantic_candidate_count": 0,
        "targeted_window_count": 0,
        "execute_llm": False,
        "use_agent_substitute": False,
        "agent_name": "local_agent",
        "use_codex_substitute": False,
        "codex_result": None,
        "llm_result": None,
        "validation_result": None,
        "closure_result": None,
        "auto_apply_high_confidence": False,
        "candidate_discovery_status": "not_planned",
        "candidate_statistics": {},
        "corrected_transcript_exists": False,
        "quality_gate_result": None,
    }
    values.update(overrides)
    return values


def test_no_conflicts_does_not_hide_deferred_review_candidates() -> None:
    pack = {
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "needs_human_review": True,
                "llm_review_eligible": False,
                "llm_review_defer_reason": "needs_conflicting_external_evidence",
            },
            {
                "candidate_id": "candidate-b",
                "needs_human_review": True,
                "llm_review_eligible": False,
                "llm_review_defer_reason": "needs_conflicting_external_evidence",
            },
        ]
    }
    stats = _candidate_statistics(
        pack,
        {"llm_arbitration_count": 0},
        {"review_required_count": 0, "review_rows": []},
    )

    assert stats["discovered_candidate_count"] == 2
    assert stats["validated_candidate_count"] == 0
    assert stats["unvalidated_candidate_count"] == 2
    assert stats["deferred_candidate_count"] == 2
    assert stats["review_required_candidate_count"] == 2
    # Keep the validation-only meaning visible instead of presenting it as the
    # total outstanding review count.
    assert stats["validation_review_required_count"] == 0

    status = _pipeline_status(
        **_status_kwargs(
            semantic_candidate_count=2,
            candidate_statistics=stats,
        )
    )
    assert status == "needs_candidate_discovery_or_review"
    assert (
        _pipeline_ok(
            status,
            candidate_discovery_status="not_planned",
            candidate_statistics=stats,
            corrected_transcript_exists=False,
            quality_gate_result={"status": "failed", "ok": False},
        )
        is False
    )


def test_failed_quality_gate_or_missing_corrected_source_is_review_gap() -> None:
    stats = _candidate_statistics({"candidates": []}, {}, {})

    failed_quality_status = _pipeline_status(
        **_status_kwargs(
            candidate_discovery_status="no_suggestions",
            candidate_statistics=stats,
            corrected_transcript_exists=True,
            quality_gate_result={"status": "failed", "ok": False},
        )
    )
    assert failed_quality_status == "complete_with_review_gaps"
    assert (
        _pipeline_ok(
            failed_quality_status,
            candidate_discovery_status="no_suggestions",
            candidate_statistics=stats,
            corrected_transcript_exists=True,
            quality_gate_result={"status": "failed", "ok": False},
        )
        is False
    )

    missing_source_status = _pipeline_status(
        **_status_kwargs(
            candidate_discovery_status="no_suggestions",
            candidate_statistics=stats,
            corrected_transcript_exists=False,
            quality_gate_result={"status": "passed", "ok": True},
        )
    )
    assert missing_source_status == "complete_with_review_gaps"


def test_clean_completed_discovery_retains_legacy_success_status() -> None:
    stats = _candidate_statistics({"candidates": []}, {}, {})
    status = _pipeline_status(
        **_status_kwargs(
            candidate_discovery_status="no_suggestions",
            candidate_statistics=stats,
            corrected_transcript_exists=True,
            quality_gate_result={"status": "passed", "ok": True},
        )
    )
    assert status == "completed_no_semantic_candidates"
    assert (
        _pipeline_ok(
            status,
            candidate_discovery_status="no_suggestions",
            candidate_statistics=stats,
            corrected_transcript_exists=True,
            quality_gate_result={"status": "passed", "ok": True},
        )
        is True
    )


def test_direct_semantic_status_cannot_finish_without_scan_gate_and_canonical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "direct-status"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "direct status"})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {"schema": "pack", "candidate_count": 0, "candidates": []},
    )

    result = transcript_semantic_correction_status(root, write=False)

    assert result["status"] == "no_candidates"
    assert result["next_action_key"] == "none"
    assert result["required_next_action_key"] == "run_candidate_discovery"
    assert result["completion_gate"]["status"] == "blocked"
    assert result["completion_gate"]["blockers"][0]["key"] == (
        "candidate_discovery_incomplete"
    )
    assert result["ok"] is False
    assert result["candidate_discovery_complete"] is False
    assert result["corrected_transcript_exists"] is False
    assert result["quality_gate_ok"] is False


def test_batch_consumer_preserves_legacy_status_but_obeys_completion_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "batch-compatible-status"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "batch compatibility"})

    monkeypatch.setattr(
        semantic_batch_module,
        "transcript_semantic_correction_status",
        lambda *_args, **_kwargs: {
            "status": "no_candidates",
            "next_action_key": "none",
            "ok": False,
            "required_next_action_key": "run_candidate_discovery",
            "completion_gate": {
                "status": "blocked",
                "ok": False,
                "required_next_action_key": "run_candidate_discovery",
                "required_action_kind": "machine",
                "blockers": [
                    {
                        "key": "candidate_discovery_incomplete",
                        "next_action_key": "run_candidate_discovery",
                        "action_kind": "machine",
                    }
                ],
            },
            "candidate_discovery_status": "not_planned",
            "candidate_discovery_complete": False,
            "candidate_count": 0,
            "review_required_count": 0,
            "accepted_decision_count": 0,
            "final_residual_error_total": 0,
            "artifacts": {},
        },
    )
    monkeypatch.setattr(
        semantic_batch_module,
        "_canonical_integrity_state",
        lambda _root: {"passed": True, "status": "passed", "issues": []},
    )

    result = semantic_batch_module.transcript_semantic_acceptance(
        root,
        write=False,
    )

    assert result["ok"] is False
    assert result["status"] == "needs_semantic_correction_action"
    assert result["semantic_status"] == "no_candidates"
    assert result["item"]["next_action_key"] == "none"
    assert result["item"]["required_next_action_key"] == (
        "run_candidate_discovery"
    )
    assert result["item"]["acceptance_state"] == "needs_candidate_discovery"


def test_quality_gate_layers_keep_legacy_issue_list_and_counts(monkeypatch) -> None:
    cues = [
        SimpleNamespace(start=0.0, end=1.0, text=""),
        SimpleNamespace(start=1.0, end=2.0, text="这里仍然有买虫。"),
        SimpleNamespace(start=2.0, end=3.0, text="OK下一步。"),
    ]
    monkeypatch.setattr(quality_gate_module, "_read_manifest", lambda _root: {})
    monkeypatch.setattr(
        quality_gate_module,
        "_resolve_source",
        lambda _root, _manifest, _input: Path("corrected-transcript.json"),
    )
    monkeypatch.setattr(quality_gate_module, "parse_transcript", lambda _path: cues)
    monkeypatch.setattr(
        quality_gate_module,
        "assess_transcript_source_completeness",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "issues": [],
            "speech_completeness_verified": True,
        },
    )
    monkeypatch.setattr(
        quality_gate_module,
        "_semantic_evidence_status",
        lambda *_args, **_kwargs: {
            "status": "not_required",
            "required": False,
            "pending_candidate_count": 0,
            "window_count": 0,
            "plan_path": "",
        },
    )
    monkeypatch.setattr(
        quality_gate_module,
        "_unresolved_high_risk_term_issues",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        quality_gate_module, "_register_run", lambda *_args, **_kwargs: {}
    )

    result = run_transcript_quality_gate(Path("."), write=False)
    layers = result["quality_layers"]

    assert layers["source_integrity"]["issue_kind_counts"]["empty_segment"] == 1
    assert (
        layers["semantic_accuracy"]["issue_kind_counts"][
            "known_bad_term_residual"
        ]
        == 1
    )
    assert (
        layers["reader_projection"]["issue_kind_counts"][
            "missing_boundary_after_latin_token"
        ]
        == 1
    )
    assert sum(layer["issue_count"] for layer in layers.values()) == result[
        "issue_count"
    ]
    assert sum(layer["fail_count"] for layer in layers.values()) == result[
        "fail_count"
    ]
    assert sum(layer["warning_count"] for layer in layers.values()) == result[
        "warning_count"
    ]
    assert any(row["kind"] == "empty_segment" for row in result["issues"])
