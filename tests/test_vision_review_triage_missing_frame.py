from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.vision_review_triage import vision_review_triage


def test_triage_does_not_select_missing_frame_paths(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 5.0,
                "visual_route": "document_visual",
                "visual_text": "",
                "frame_paths": [str(bundle / "assets" / "missing.png")],
                "quality_issues": ["missing_visual_text"],
            }
        ],
    )

    result = vision_review_triage(bundle, min_score=1, write=False)

    row = result["all_ranked_candidates"][0]
    assert row["has_frame_evidence"] is False
    assert result["visual_structure_first_indexes"] == []
    assert result["semantic_indexes"] == []
    assert result["temporal_indexes"] == []
    assert result["frame_items"] == 0
