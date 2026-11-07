from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.transcript_postprocess import postprocess_asr_transcript


def test_readable_merge_does_not_assign_unlabeled_text_to_labeled_speaker(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "partial-label"
    bundle.mkdir()
    source = bundle / "normalized-transcript.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "a", "start": 0, "end": 1, "text": "尚未分人"},
                    {
                        "id": "b",
                        "start": 1,
                        "end": 2,
                        "text": "已有分人",
                        "speaker": "S01",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    postprocess_asr_transcript(
        bundle,
        input_path=source,
        target_seconds=30,
        max_chars=500,
        segment_policy="readable_merge",
        set_corrected=False,
        write=True,
    )
    payload = json.loads(
        (bundle / "postprocessed-transcript.json").read_text(encoding="utf-8")
    )

    assert len(payload["segments"]) == 2
    assert "speaker" not in payload["segments"][0]
    assert payload["segments"][1]["speaker"] == "S01"
