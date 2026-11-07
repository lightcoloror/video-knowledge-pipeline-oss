from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.asr_response_quality import assess_asr_response
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_source_arbitration import (
    arbitrate_transcript_sources,
)


def test_arbitration_preserves_coarse_timing_for_vad_density(tmp_path: Path) -> None:
    """Keep estimated timing identifiable after the preferred transcript changes.

    Intent: prevent local arbitration/export from recreating false ASR retries.
    Decision: preserve the base cue's existing transformation provenance.
    Reason: arbitration changes text authority, not the origin of its timing.
    Evidence: a production Bundle returned to low_text_density after the sidecar
    dropped ``timing_estimation`` even though its normalized source was correct.
    Effective scope: source-arbitrated JSON metadata and downstream quality only.
    """

    root = tmp_path / "bundle"
    root.mkdir()
    write_json(
        root / "manifest.json",
        {"normalized_transcript_json": "normalized-transcript.json"},
    )
    transformation = {
        "type": "timing_estimation",
        "method": "character_proportional_within_source_window",
        "precision": "coarse",
        "source_window_start": 0.0,
        "source_window_end": 300.0,
        "source_record_index": 0,
    }
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {
                    "id": "segment-000001",
                    "start": 0.0,
                    "end": 300.0,
                    "text": "测试测试1234。",
                    "transformations": [transformation],
                }
            ]
        },
    )

    result = arbitrate_transcript_sources(root)
    corrected = read_json(root / "source-arbitrated-transcript.json")
    response_quality = assess_asr_response(
        corrected,
        vad_intervals=[{"start": 24.0, "end": 30.0}],
        media_duration_seconds=300.0,
    )

    assert result["status"] == "completed"
    assert corrected["segments"][0]["transformations"] == [transformation]
    assert response_quality["status"] == "passed"
    assert response_quality["coarse_timing_density"]["status"] == "passed"
    assert response_quality["coarse_timing_density"]["estimated_segment_count"] == 1
    assert response_quality["retry_plan"]["windows"] == []
