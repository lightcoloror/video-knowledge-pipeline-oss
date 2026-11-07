from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.targeted_visual_evidence import run_targeted_visual_evidence


def _bundle(root: Path) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "visual_route": "document_visual",
                    "frame_paths": ["frame-1.jpg"],
                    "visual_text": "",
                    "structured_visual": {},
                    "quality_issues": ["ocr_wrapper_only"],
                },
                {
                    "index": 2,
                    "visual_route": "semantic_frame",
                    "frame_paths": ["frame-2.jpg"],
                    "visual_text": "",
                },
                {
                    "index": 3,
                    "visual_route": "temporal_sequence",
                    "frame_paths": ["frame-3.jpg"],
                    "temporal_frame_paths": ["frame-3a.jpg", "frame-3b.jpg"],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _triage(*args, **kwargs):
    return {
        "mode": "triage",
        "selected_counts": {"visual_structure_first": 1, "semantic": 1, "temporal": 1},
        "visual_structure_first_indexes": [1],
        "semantic_indexes": [2],
        "temporal_indexes": [3],
    }


def test_targeted_visual_preview_never_calls_online_and_keeps_empty_results_blocked(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    monkeypatch.setattr("video_knowledge_pipeline.targeted_visual_evidence.vision_review_triage", _triage)
    monkeypatch.setattr(
        "video_knowledge_pipeline.targeted_visual_evidence.run_visual_structure_plan",
        lambda *args, **kwargs: {"summary": {"selected_indexes": [1], "execute_ebook_pipeline": False}},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.targeted_visual_evidence.run_screen_text_recovery",
        lambda *args, **kwargs: {"selected_indexes": [1], "crop_summary": {}, "ocr_summary": {"updated": 0}},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.targeted_visual_evidence.run_high_res_tile_plan",
        lambda *args, **kwargs: {"selected_indexes": [1], "summary": {"tiles_planned": 4}},
    )

    result = run_targeted_visual_evidence(bundle, allow_online_review=False, write=False)

    assert result["unresolved_document_indexes"] == [1]
    assert result["online_review"]["semantic_indexes"] == [1, 2]
    assert result["online_review"]["temporal_indexes"] == [3]
    assert result["online_review"]["executed"] is False
    assert result["human_review_indexes"] == [1, 2]
    assert result["operator_boundary"]["empty_wrapper_or_low_information_remains_blocked"] is True


def test_targeted_visual_local_ocr_resolution_prevents_unnecessary_tile_or_online_review(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    triage_calls: list[int] = []

    def retriage(*args, **kwargs):
        triage_calls.append(1)
        return _triage()

    monkeypatch.setattr("video_knowledge_pipeline.targeted_visual_evidence.vision_review_triage", retriage)
    monkeypatch.setattr(
        "video_knowledge_pipeline.targeted_visual_evidence.run_visual_structure_plan",
        lambda *args, **kwargs: {"summary": {"selected_indexes": [1], "execute_ebook_pipeline": True}},
    )

    def screen(*args, **kwargs):
        timeline_path = bundle / "timeline.json"
        rows = json.loads(timeline_path.read_text(encoding="utf-8"))
        rows[0]["visual_text"] = "Playwright MCP"
        rows[0]["structured_visual"] = {"type": "slide", "text": "Playwright MCP"}
        rows[0]["quality_issues"] = []
        timeline_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return {"selected_indexes": [1], "crop_summary": {"written": 1}, "ocr_summary": {"updated": 1}}

    monkeypatch.setattr("video_knowledge_pipeline.targeted_visual_evidence.run_screen_text_recovery", screen)
    monkeypatch.setattr(
        "video_knowledge_pipeline.targeted_visual_evidence.run_high_res_tile_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolved OCR must not tile")),
    )

    result = run_targeted_visual_evidence(
        bundle,
        execute_ebook=True,
        execute_crops=True,
        execute_ocr=True,
        allow_online_review=True,
        write=True,
    )

    assert result["unresolved_document_indexes"] == []
    assert result["online_review"]["semantic_indexes"] == [2]
    assert 1 not in result["online_review"]["semantic_indexes"]
    assert len(triage_calls) == 2
    assert result["triage"]["retriaged_after_local_evidence"] is True

def test_targeted_visual_rejects_execute_with_no_write(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    monkeypatch.setattr("video_knowledge_pipeline.targeted_visual_evidence.vision_review_triage", _triage)

    try:
        run_targeted_visual_evidence(bundle, execute_ebook=True, write=False)
    except ValueError as exc:
        assert "--no-write" in str(exc)
    else:
        raise AssertionError("execute with no-write must be rejected")
