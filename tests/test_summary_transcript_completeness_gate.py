from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_codex import (
    _summary_transcript_completeness_gate,
)


def _write_gate(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "transcript-quality-gate.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_missing_transcript_quality_artifact_preserves_legacy_summary_behavior(
    tmp_path: Path,
) -> None:
    result = _summary_transcript_completeness_gate(tmp_path)

    assert result["passed"] is True
    assert result["status"] == "not_available_legacy_compatible"


def test_failed_transcript_quality_blocks_summary_production_readiness(
    tmp_path: Path,
) -> None:
    _write_gate(
        tmp_path,
        {
            "status": "failed",
            "ok": False,
            "fail_count": 2,
            "source_completeness": {
                "applicable": True,
                "status": "failed",
                "speech_completeness_verified": False,
            },
        },
    )

    result = _summary_transcript_completeness_gate(tmp_path)

    assert result["passed"] is False
    assert result["status"] == "failed"


def test_unverified_single_pass_transcript_blocks_summary_production_readiness(
    tmp_path: Path,
) -> None:
    _write_gate(
        tmp_path,
        {
            "status": "warning",
            "ok": True,
            "fail_count": 0,
            "source_completeness": {
                "applicable": True,
                "status": "warning",
                "speech_completeness_verified": False,
            },
        },
    )

    result = _summary_transcript_completeness_gate(tmp_path)

    assert result["passed"] is False
    assert result["status"] == "unverified"


def test_source_bound_verified_transcript_allows_summary_quality_evaluation(
    tmp_path: Path,
) -> None:
    _write_gate(
        tmp_path,
        {
            "status": "warning",
            "ok": True,
            "fail_count": 0,
            "source_completeness": {
                "applicable": True,
                "status": "passed",
                "speech_completeness_verified": True,
            },
        },
    )

    result = _summary_transcript_completeness_gate(tmp_path)

    assert result["passed"] is True
    assert result["status"] == "verified"
