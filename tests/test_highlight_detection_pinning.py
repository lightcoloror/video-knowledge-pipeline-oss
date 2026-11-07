from __future__ import annotations

from pathlib import Path

import pytest

from video_knowledge_pipeline.highlight_detection_adapter import (
    LIGHTHOUSE_COMMIT,
    highlight_detection_status,
)


def test_lighthouse_status_verifies_fixed_source_and_never_downloads() -> None:
    status = highlight_detection_status()
    if "lighthouse_source_missing" in status["blockers"]:
        pytest.skip("optional reviewed Lighthouse source is not installed")

    assert status["source_verified"] is True
    assert status["source_commit"] == LIGHTHOUSE_COMMIT
    assert status["status"] == "needs_setup"
    assert "cg_detr_checkpoint_missing" in status["blockers"]
    assert status["operator_boundary"]["automatic_model_download"] is False
    assert status["operator_boundary"]["automatic_remote_fallback"] is False


def test_non_git_lighthouse_source_is_not_accepted(tmp_path: Path) -> None:
    source = tmp_path / "lighthouse"
    source.mkdir()

    status = highlight_detection_status(source_root=source)

    assert status["source_verified"] is False
    assert "lighthouse_source_commit_mismatch" in status["blockers"]
