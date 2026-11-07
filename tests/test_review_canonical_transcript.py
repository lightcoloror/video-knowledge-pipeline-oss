from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.lecture_package import (
    _review_transcript_segments,
    _timeline_for_review,
    render_lecture_review_html,
)
from video_knowledge_pipeline.webui_bridge import refresh_bundle_review_html


def _package(tmp_path: Path) -> dict:
    (tmp_path / "source-arbitrated-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 1.0, "end": 4.0, "text": "canonical 正文"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "corrected-transcript.json").write_text(
        json.dumps(
            {"segments": [{"start": 1.0, "end": 4.0, "text": "stale 正文"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "title": "Canonical Review",
        "review_artifacts": {
            "bundle_dir": str(tmp_path),
            "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
            "corrected_transcript_json": "corrected-transcript.json",
        },
        "timeline": [
            {"start": 1.0, "end": 4.0, "transcript": "旧 timeline 正文"},
            {"start": 20.0, "end": 22.0, "transcript": "未对齐旧正文"},
        ],
        "sources": [],
    }


def test_review_transcript_prefers_source_arbitrated_canonical(tmp_path: Path) -> None:
    segments = _review_transcript_segments(_package(tmp_path))
    assert [row["text"] for row in segments] == ["canonical 正文"]


def test_review_timeline_projects_canonical_and_quarantines_stale_text(tmp_path: Path) -> None:
    timeline = _timeline_for_review(_package(tmp_path))
    assert timeline[0]["transcript"] == "canonical 正文"
    assert timeline[0]["original_transcript"] == "旧 timeline 正文"
    assert timeline[0]["review_transcript_source"] == "source_arbitrated_transcript"
    assert "未对齐旧正文" not in timeline[1]["transcript"]
    assert timeline[1]["original_transcript"] == "未对齐旧正文"
    assert timeline[1]["review_transcript_source"] == "canonical_alignment_missing"


def test_review_html_embeds_projected_timeline_as_primary_data(tmp_path: Path) -> None:
    html = render_lecture_review_html(_package(tmp_path))
    assert "canonical 正文" in html
    assert '"transcript": "canonical 正文"' in html
    assert '"original_transcript": "旧 timeline 正文"' in html
    assert '"transcript": "未对齐旧正文"' not in html


def test_refresh_persists_canonical_binding_after_alignment_manifest_reload(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": package["title"],
                "corrected_transcript_json": "corrected-transcript.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "timeline.json").write_text(
        json.dumps(package["timeline"], ensure_ascii=False),
        encoding="utf-8",
    )

    refresh_bundle_review_html(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    canonical = tmp_path / "source-arbitrated-transcript.json"
    expected_hash = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert manifest["source_arbitrated_transcript_json"] == canonical.name
    assert manifest["corrected_transcript_json"] == canonical.name
    assert manifest["review_transcript_canonical"]["sha256"] == expected_hash
    assert manifest["review_transcript_canonical"]["raw_timeline_transcript_is_audit_only"] is True
    html = (tmp_path / "review.html").read_text(encoding="utf-8")
    assert '"transcript": "canonical 正文"' in html
