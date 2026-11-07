from __future__ import annotations

from pathlib import Path

from PIL import Image

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.vision_review_triage import vision_review_triage


def test_triage_uses_canonical_transcript_when_timeline_text_is_empty(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.png"
    Image.new("RGB", (160, 90), "white").save(frame)
    write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
        },
    )
    write_json(
        bundle / "source-arbitrated-transcript.json",
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "这里展示一个表格，请看一下。",
                }
            ]
        },
    )
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 5.0,
                "visual_route": "semantic_frame",
                "quality_issues": ["missing_transcript", "needs_human_review"],
                "frame_paths": [str(frame)],
            }
        ],
    )

    result = vision_review_triage(bundle, min_score=1, write=False)

    row = result["all_ranked_candidates"][0]
    assert "这里展示一个表格" in row["transcript_excerpt"]
    assert "missing_transcript" not in row["quality_issues"]
    assert "needs_human_review" in row["quality_issues"]
    assert "deictic_transcript_requires_screen_check" in row["semantic_reasons"]
    assert result["semantic_indexes"] == [1]
