from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.tile_result_import_builder import build_tile_result_import
from video_knowledge_pipeline.tile_result_merge import run_tile_result_merge


def _write_tile_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"title": "tile smoke"}, ensure_ascii=False), encoding="utf-8")
    (root / "timeline.json").write_text(json.dumps([{"index": 1, "quality_issues": ["ocr_text_empty"]}], ensure_ascii=False), encoding="utf-8")
    plan = {
        "items": [
            {
                "index": 1,
                "tiles": [
                    {"tile_id": "0001-01", "output_path": str(root / "high-res-tiles" / "tile-01.jpg")},
                    {"tile_id": "0001-02", "output_path": str(root / "high-res-tiles" / "tile-02.jpg")},
                    {"tile_id": "0001-03", "output_path": str(root / "high-res-tiles" / "tile-03.jpg")},
                    {"tile_id": "0001-04", "output_path": str(root / "high-res-tiles" / "tile-04.jpg")},
                    {"tile_id": "0001-05", "output_path": str(root / "high-res-tiles" / "tile-05.jpg")},
                ],
            }
        ]
    }
    (root / "high-res-tile-plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    results = root / "tile-results"
    results.mkdir()
    (results / "0001-01.json").write_text(
        json.dumps({"choices": [{"message": {"content": "OpenAI 视觉结果"}}], "model": "fixture-model"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (results / "0001-02.json").write_text(
        json.dumps({"result": [{"text": "客户特点", "confidence": 0.9}, {"text": "成交原则", "confidence": 0.8}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (results / "0001-03.json").write_text(
        json.dumps({"candidates": [{"content": {"parts": [{"text": "Gemini 视觉结果"}]}}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (results / "0001-04.json").write_text(
        json.dumps({"visual_understanding": {"summary": "表格结构", "screen_text": "问题链"}, "confidence": 0.82}, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def test_tile_result_import_build_consumes_common_ocr_and_vlm_shapes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    results_dir = _write_tile_bundle(bundle)

    result = build_tile_result_import(bundle, results_dir=results_dir, default_confidence=0.72, write=True)

    rows = result["tile_results_preview"]
    by_tile = {row["tile_id"]: row for row in rows}
    assert result["summary"]["matched_results"] == 4
    assert result["summary"]["pending_results"] == 1
    assert by_tile["0001-01"]["parse_source"] == "openai_choices"
    assert by_tile["0001-01"]["confidence"] == 0.72
    assert by_tile["0001-02"]["parse_source"] == "ocr_entries"
    assert by_tile["0001-02"]["confidence"] == 0.85
    assert "客户特点" in by_tile["0001-02"]["text"]
    assert by_tile["0001-03"]["parse_source"] == "gemini_candidates"
    assert by_tile["0001-03"]["confidence"] == 0.72
    assert by_tile["0001-04"]["structured_visual"]["type"] == "tile_visual_understanding"
    assert by_tile["0001-05"]["status"] == "pending_result"
    assert by_tile["0001-05"]["confidence"] == 0.0
    assert result["pending_items"][0]["reason"] == "tile_result_pending"
    assert result["pending_items"][0]["tile_id"] == "0001-05"

    import_run = json.loads((bundle / "runs" / "tile-result-import-build" / "run.json").read_text(encoding="utf-8"))
    assert import_run["status"] == "needs_input"
    assert import_run["failed_items"][0]["reason"] == "tile_result_pending"
    assert import_run["failed_items"][0]["tile_id"] == "0001-05"
    assert import_run["failed_items"][0]["suggested_next_tool"] == "tile_result_import_build"
    assert "tile-result-import-build" in import_run["failed_items"][0]["tile_result_import_command"]
    assert import_run["failed_items"][0]["evidence_paths"]
    assert "pending tiles" in " ".join(import_run["next_actions"])

    merge = run_tile_result_merge(bundle, input_json=bundle / "tile-result-import.json", execute=False, write=True)
    assert merge["summary"]["updates"] == 4
    assert merge["summary"]["review_targets"] == 1
