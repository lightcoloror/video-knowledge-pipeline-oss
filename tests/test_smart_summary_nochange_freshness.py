from __future__ import annotations

import json
import os
from pathlib import Path

from video_knowledge_pipeline.smart_summary_codex import (
    _smart_summary_freshness_gate,
)


def test_freshness_accepts_verified_no_change_transcript(tmp_path: Path) -> None:
    summary = tmp_path / "smart-summary.codex.md"
    transcript = tmp_path / "source-arbitrated-transcript.json"
    summary.write_text("summary", encoding="utf-8")
    transcript.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
                "source": "transcript_semantic_correction_no_change",
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    os.utime(summary, (1, 1))
    os.utime(transcript, (2, 2))

    result = _smart_summary_freshness_gate(
        summary,
        {"passed": True, "transcript_source": str(transcript)},
    )

    assert result["passed"] is True
    assert result["status"] == "unchanged_transcript_after_review"
