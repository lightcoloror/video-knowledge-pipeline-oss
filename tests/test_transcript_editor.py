from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.transcript_editor import apply_transcript_edits, prepare_transcript_edit_session


def _bundle(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"title": "Transcript UI"}, ensure_ascii=False), encoding="utf-8")
    (root / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 2, "transcript": "brother mc p 是浏览器工具"},
                {"index": 2, "start": 2, "end": 4, "transcript": "第二段不用改"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_prepare_transcript_edit_session_and_apply_human_edits(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)

    session = prepare_transcript_edit_session(bundle, write=True)

    assert session["schema"] == "video_knowledge_pipeline.transcript_edit_session.v1"
    assert session["summary"]["segments"] == 2
    assert (bundle / "transcript-editor.html").exists()
    html = (bundle / "transcript-editor.html").read_text(encoding="utf-8")
    assert "transcript-edits.json" in html
    assert "no_llm_call" not in html
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["transcript_editor_html"] == "transcript-editor.html"
    assert session["run_registry"]["run_type"] == "prepare_transcript_edit_session"
    assert session["run_registry"]["status"] == "needs_input"
    assert session["run_registry"]["parameters"]["segment_count"] == 2
    assert (bundle / "runs" / "prepare-transcript-edit-session" / "run.json").exists()

    edits = bundle / "transcript-edits.json"
    edits.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_edit_notes.v1",
                "segments": [
                    {"index": 0, "corrected_text": "Browser MCP 是浏览器工具"},
                    {"index": 1, "corrected_text": "第二段不用改"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    applied = apply_transcript_edits(bundle, edits_json=edits, write=True)

    assert applied["schema"] == "video_knowledge_pipeline.apply_transcript_edits.v1"
    assert applied["summary"]["corrected_segments"] == 1
    assert applied["run_registry"]["run_type"] == "apply_transcript_edits"
    assert applied["run_registry"]["status"] == "completed"
    assert applied["run_registry"]["parameters"]["corrected_segments"] == 1
    assert (bundle / "runs" / "apply-transcript-edits" / "run.json").exists()
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    run_types = {row["run_type"]: row["status"] for row in registry["runs"]}
    assert run_types["prepare_transcript_edit_session"] == "needs_input"
    assert run_types["apply_transcript_edits"] == "completed"
    corrected = json.loads((bundle / "human-corrected-transcript.json").read_text(encoding="utf-8"))
    assert corrected["schema"] == "video_knowledge_pipeline.human_corrected_transcript.v1"
    assert corrected["segments"][0]["corrected_text"] == "Browser MCP 是浏览器工具"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corrected_transcript_json"] == "human-corrected-transcript.json"
