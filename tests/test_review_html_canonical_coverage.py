from __future__ import annotations

import json

from video_knowledge_pipeline.lecture_package import render_lecture_review_html


def test_review_html_coverage_counts_final_canonical_projection() -> None:
    """The visible transcript count must describe the rendered review rows.

    Intent: keep the review dashboard consistent with the transcript excerpts
    an operator actually sees.
    Decision: recompute only ``items_with_transcript`` from the final review
    projection while preserving every other manifest coverage metric.
    Reason: the manifest may describe a pre-canonical Timeline and report zero
    transcript rows even after canonical alignment succeeds.
    Evidence: an existing 80-item bundle rendered canonical excerpts while its
    stale coverage card still displayed zero.
    Effective scope: review HTML/package-data only; the persisted manifest
    coverage remains an audit record of the source Timeline.
    """

    html = render_lecture_review_html(
        {
            "title": "canonical coverage",
            "coverage": {
                "timeline_items": 2,
                "items_with_transcript": 0,
                "items_with_visual_text": 7,
            },
            "sources": [],
            "timeline": [
                {"index": 1, "start": 0.0, "end": 5.0, "transcript": ""},
                {"index": 2, "start": 10.0, "end": 15.0, "transcript": ""},
            ],
            "review_transcript_segments": [
                {"start": 0.0, "end": 5.0, "text": "已绑定的规范化逐字稿"},
            ],
        }
    )
    encoded = html.split(
        '<script id="package-data" type="application/json">',
        maxsplit=1,
    )[1].split("</script>", maxsplit=1)[0]
    package = json.loads(encoded)

    assert package["coverage"]["items_with_transcript"] == 1
    assert package["coverage"]["items_with_visual_text"] == 7
    assert package["timeline"][0]["review_transcript_source"] == (
        "source_arbitrated_transcript"
    )
    assert package["timeline"][1]["review_transcript_source"] == (
        "canonical_alignment_missing"
    )
    assert "<strong>1</strong><span>含转写</span>" in html
