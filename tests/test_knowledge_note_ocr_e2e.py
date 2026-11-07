from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.ocr_backfill import run_ocr_backfill


def test_imported_ocr_evidence_reaches_final_knowledge_note(tmp_path: Path) -> None:
    bundle = tmp_path / "synthetic-bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "slide-0001.jpg"
    frame.write_bytes(b"synthetic-image-fixture-not-real-media")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Synthetic OCR Evidence"}),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 4,
                    "transcript": "老师说明这一页的流程。",
                    "visual_route": "document_visual",
                    "frame_paths": [str(frame)],
                    "tagger_tags": ["presentation screen", "indoor", "presentation screen"],
                    "quality_issues": ["missing_visual_text"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ocr_phrase = "合成 OCR 证据：确认需求 → 给出方案"
    imported = tmp_path / "reviewed-ocr.json"
    imported.write_text(
        json.dumps(
            {
                "schema": "lecture_ocr_backfill_input.v1",
                "items": [{"index": 1, "text": ocr_phrase, "source": "synthetic_reviewed_fixture"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    backfill = run_ocr_backfill(bundle, input_json=imported)
    exported = export_knowledge_note(
        bundle,
        title="Synthetic OCR Evidence",
        run_transcript_evidence_check=False,
    )

    assert backfill["backfill"]["updated_indexes"] == [1]
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["visual_text"] == ocr_phrase
    assert timeline[0]["ocr_backfilled_text"] == ocr_phrase
    input_pack = json.loads((bundle / "exports" / "smart-summary-input-pack.json").read_text(encoding="utf-8"))
    assert any(row.get("visual_text") == ocr_phrase for row in input_pack["visual_digest"]["items"])
    assert input_pack["evidence_trace"]["summary"]["ocr_or_ebook_items"] == 1
    full_transcript = Path(exported["full_transcript_path"]).read_text(encoding="utf-8")
    final_note = Path(exported["note_path"]).read_text(encoding="utf-8")
    assert f"画面文字/OCR：{ocr_phrase}" in full_transcript
    assert ocr_phrase in final_note
    assert "RAM++ 候选标签（未人工确认）" in final_note
    assert "presentation screen" in final_note
    assert "indoor" in final_note
