from __future__ import annotations

import shutil
from pathlib import Path

from video_knowledge_pipeline.review_session import prepare_review_session, review_closure_status
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_editor import apply_transcript_edits, prepare_transcript_edit_session


def _bundle() -> Path:
    root = Path("outputs/test-transcript-arbitration-review/bundle").resolve()
    shutil.rmtree(root.parent, ignore_errors=True)
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "Transcript arbitration review",
            "corrected_transcript_json": "source-arbitrated-transcript.json",
            "transcript_source_arbitration_json": "transcript-source-arbitration.json",
        },
    )
    write_json(
        root / "timeline.json",
        [
            {"index": 0, "start": 0.0, "end": 5.0, "transcript": "今天讲 Browser Use。"},
        ],
    )
    (root / "review.html").write_text("<html>review</html>", encoding="utf-8")
    write_json(
        root / "source-arbitrated-transcript.json",
        {
            "segments": [
                {"index": 0, "start": 0.0, "end": 5.0, "text": "今天讲 Browser Use。"},
            ]
        },
    )
    write_json(
        root / "transcript-source-arbitration.json",
        {
            "schema": "video_knowledge_pipeline.transcript_source_arbitration.v1",
            "review_rows": [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 5.0,
                    "corrected_text": "今天讲 Browser Use。",
                    "original_text": "今天讲 browser you。",
                    "chosen_source": "platform_subtitle",
                    "chosen_source_type": "platform_subtitle",
                    "confidence": 0.61,
                    "review_reason": "low_arbitration_confidence",
                    "alternatives": [
                        {"source_id": "asr", "source_type": "asr", "text": "今天讲 browser you。", "score": 2.1},
                        {"source_id": "platform_subtitle", "source_type": "subtitle", "text": "今天讲 Browser Use。", "score": 2.3},
                    ],
                }
            ],
        },
    )
    return root


def test_review_pack_includes_transcript_arbitration_targets() -> None:
    root = _bundle()

    session = prepare_review_session(root, refresh=False, limit=0, group_by="reason")

    targets = session["review_targets"]
    assert targets["by_reason"]["transcript_source_conflict"] == 1
    assert targets["by_reason"]["low_arbitration_confidence"] == 1
    item = next(row for row in targets["items"] if row.get("target_type") == "transcript_arbitration")
    assert item["suggested_status"] == "corrected_transcript"
    assert item["transcript_arbitration"]["chosen_source"] == "platform_subtitle"

    todo = read_json(root / "review-notes.todo.json")
    assert all("transcript_source_conflict" not in str(row.get("reason") or "") for row in todo["reviews"])

    closure = review_closure_status(root, write=False)
    assert closure["open_by_reason"]["transcript_source_conflict"] == 1

    edits = root / "transcript-edits.json"
    write_json(
        edits,
        {
            "schema": "video_knowledge_pipeline.transcript_edit_notes.v1",
            "segments": [
                {"index": 0, "corrected_text": "今天讲 Browser Use。"},
            ],
        },
    )
    apply_transcript_edits(root, edits_json=edits, write=True)

    closed_session = prepare_review_session(root, refresh=False, limit=0, group_by="reason")
    assert closed_session["review_targets"]["by_reason"].get("transcript_source_conflict", 0) == 0
    closed_closure = review_closure_status(root, write=False)
    assert closed_closure["open_by_reason"].get("transcript_source_conflict", 0) == 0
    assert closed_closure["open_by_reason"].get("low_arbitration_confidence", 0) == 0
    assert closed_closure["summary"]["closed"] >= 1
    closed_targets = closed_closure["closed_targets"]
    closed_item = next(row for row in closed_targets if row.get("target_type") == "transcript_arbitration")
    assert closed_item["closed"] is True
    assert closed_item["transcript_arbitration"]["human_corrected_text"] == "今天讲 Browser Use。"


def test_transcript_editor_shows_arbitration_conflicts() -> None:
    root = _bundle()

    session = prepare_transcript_edit_session(root)

    assert session["segments"][0]["arbitration_review"]["review_reason"] == "low_arbitration_confidence"
    html = (root / "transcript-editor.html").read_text(encoding="utf-8")
    assert "仲裁待复核" in html
    assert "字幕/ASR 仲裁冲突" in html
    assert "Browser Use" in html
