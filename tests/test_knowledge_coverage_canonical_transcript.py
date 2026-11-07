from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.timeline_alignment_audit import timeline_alignment_audit


def test_coverage_reuses_fresh_canonical_transcript_alignment(tmp_path: Path) -> None:
    timeline = [
        {"index": 1, "start": 0.0, "end": 4.0, "transcript": ""},
        {"index": 2, "start": 4.0, "end": 8.0, "transcript": ""},
    ]
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"normalized_transcript_json": "normalized-transcript.json"}),
        encoding="utf-8",
    )
    (tmp_path / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.1, "end": 3.9, "text": "one"},
                    {"start": 4.1, "end": 7.9, "text": "two"},
                ]
            }
        ),
        encoding="utf-8",
    )
    timeline_alignment_audit(tmp_path, write=True)

    report = build_knowledge_coverage({}, timeline, bundle_dir=tmp_path)

    speech = next(row for row in report["channels"] if row["key"] == "speech")
    assert speech["covered_count"] == 2
    assert speech["coverage_percent"] == 100.0
    assert report["canonical_transcript_aligned_items"] == 2
    assert report["samples"]["missing_transcript"] == []


def test_coverage_rejects_stale_canonical_alignment(tmp_path: Path) -> None:
    timeline = [{"index": 1, "start": 0.0, "end": 4.0, "transcript": ""}]
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"normalized_transcript_json": "normalized-transcript.json"}),
        encoding="utf-8",
    )
    (tmp_path / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0.1, "end": 3.9, "text": "one"}]}),
        encoding="utf-8",
    )
    timeline_alignment_audit(tmp_path, write=True)
    timeline.append({"index": 2, "start": 4.0, "end": 8.0, "transcript": ""})
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")

    report = build_knowledge_coverage({}, timeline, bundle_dir=tmp_path)

    speech = next(row for row in report["channels"] if row["key"] == "speech")
    assert speech["covered_count"] == 0
    assert report["canonical_transcript_aligned_items"] == 0
