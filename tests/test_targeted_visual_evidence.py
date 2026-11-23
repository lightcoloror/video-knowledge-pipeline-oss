from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.targeted_visual_evidence import (
    _unresolved_document_indexes,
    run_targeted_visual_evidence,
)
from video_knowledge_pipeline.visual_evidence import multimodal_document_ocr_state


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


def _multimodal_item(confidence) -> dict:
    return {
        "visual_text": "合成三步流程：整理、补充、排序。",
        "visual_understanding": {
            "objects": [{"type": "flowchart"}],
            "interface_state": "三个步骤由箭头从左到右连接",
            "spatial_relations": ["第二步位于第一步和第三步之间"],
            "confidence": confidence,
            "validation_status": "ok",
            "evidence_frame_paths": ["frame-1.jpg"],
        },
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


def test_targeted_visual_accepts_cross_validated_ocr_and_multimodal_document(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    timeline_path = bundle / "timeline.json"
    rows = json.loads(timeline_path.read_text(encoding="utf-8"))
    rows[0].update(
        {
            "visual_text": "三步流程：整理、补充、排序。",
            "structured_visual": [],
            "quality_issues": ["structured_visual_without_structure"],
            "visual_understanding": {
                "objects": [{"type": "flowchart"}],
                "interface_state": "三个步骤由箭头从左到右连接",
                "spatial_relations": ["第二步位于第一步和第三步之间"],
                "confidence": 0.95,
                "validation_status": "ok",
                "evidence_frame_paths": ["frame-1.jpg"],
            },
        }
    )
    timeline_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    assert _unresolved_document_indexes(bundle, [1]) == []


def test_multimodal_document_ocr_confidence_rejects_bool_nonfinite_and_out_of_range() -> None:
    for invalid in (True, False, float("inf"), float("-inf"), float("nan"), -0.1, 101, 150, "nan", None):
        state = multimodal_document_ocr_state(_multimodal_item(invalid))
        assert state["sufficient"] is False, repr(invalid)
        assert state["confidence"] == 0.0, repr(invalid)


def test_multimodal_document_ocr_confidence_accepts_fraction_percent_and_keeps_threshold() -> None:
    assert multimodal_document_ocr_state(_multimodal_item(0.95))["sufficient"] is True
    assert multimodal_document_ocr_state(_multimodal_item(0.85))["sufficient"] is True
    assert multimodal_document_ocr_state(_multimodal_item(90))["sufficient"] is True
    assert multimodal_document_ocr_state(_multimodal_item(100))["sufficient"] is True
    assert multimodal_document_ocr_state(_multimodal_item(0.84))["sufficient"] is False
    assert multimodal_document_ocr_state(_multimodal_item(1.5))["sufficient"] is False


def test_multimodal_ocr_state_whitespace_visual_text_falls_back_to_ocr() -> None:
    item = {
        "frame_path": "synthetic_frames/slide_001.png",
        "visual_text": "   \t\n  ",
        "ocr_text": "Sample OCR Content: Chapter 1 Overview",
        "visual_understanding": {
            "evidence_level": "confirmed",
            "confidence": 0.95,
            "interface_state": "slide_presentation",
            "objects": [{"name": "diagram", "type": "chart"}],
            "evidence_frame_paths": ["synthetic_frames/slide_001.png"],
        },
    }
    result = multimodal_document_ocr_state(item)
    assert result["ocr_available"] is True
    assert result["sufficient"] is True
    assert result["confidence"] == 0.95
    assert result["has_layout"] is True
    assert result["visual_model_complete"] is True
    assert result["visual_model_source"] == "model"


def test_multimodal_ocr_state_whitespace_human_text_falls_back_to_lower_precedence() -> None:
    item_visual = {
        "frame_path": "synthetic_frames/slide_002.png",
        "human_corrected_visual_text": "   \r\n   ",
        "visual_text": "Extracted Visual Text from Slide",
        "ocr_text": "Secondary OCR Content",
        "visual_understanding": {
            "evidence_level": "confirmed",
            "confidence": 0.95,
            "interface_state": "slide_presentation",
            "objects": [{"name": "table", "type": "grid"}],
            "evidence_frame_paths": ["synthetic_frames/slide_002.png"],
        },
    }
    result_visual = multimodal_document_ocr_state(item_visual)
    assert result_visual["ocr_available"] is True
    assert result_visual["sufficient"] is True
    assert result_visual["confidence"] == 0.95

    item_ocr = {
        "frame_path": "synthetic_frames/slide_003.png",
        "human_corrected_visual_text": " \t ",
        "visual_text": "  \n ",
        "ocr_text": "OCR-only synthetic content",
        "visual_understanding": {
            "evidence_level": "confirmed",
            "confidence": 0.9,
            "interface_state": "slide_presentation",
            "objects": [{"name": "flowchart", "type": "diagram"}],
            "evidence_frame_paths": ["synthetic_frames/slide_003.png"],
        },
    }
    result_ocr = multimodal_document_ocr_state(item_ocr)
    assert result_ocr["ocr_available"] is True
    assert result_ocr["sufficient"] is True
    assert result_ocr["confidence"] == 0.9


def test_multimodal_ocr_state_all_whitespace_text_is_insufficient() -> None:
    item = {
        "frame_path": "synthetic_frames/slide_004.png",
        "human_corrected_visual_text": "   ",
        "visual_text": " \t\n ",
        "ocr_text": "  \r\n ",
        "visual_understanding": {
            "evidence_level": "confirmed",
            "confidence": 0.95,
            "interface_state": "slide_presentation",
            "objects": [{"name": "diagram", "type": "chart"}],
            "evidence_frame_paths": ["synthetic_frames/slide_004.png"],
        },
    }
    result = multimodal_document_ocr_state(item)
    assert result["ocr_available"] is False
    assert result["sufficient"] is False
