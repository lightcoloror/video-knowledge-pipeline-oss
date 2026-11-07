from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.smart_summary_chapters import build_smart_summary_chapter_pack


def _write_basic_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Run Registry Export"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 8,
                    "transcript": "这里用客户成交案例说明获取信任的关键动作和后续复盘流程。",
                    "visual_route": "document_visual",
                    "visual_text": "成交原则：信任、问题链、复盘",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_export_knowledge_note_registers_needs_review_without_chapter_links(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-review"
    _write_basic_bundle(bundle)

    result = export_knowledge_note(bundle, title="Run Registry Export")

    run = result["run_registry"]
    assert run["run_type"] == "knowledge_note_export"
    assert run["status"] == "completed"
    assert run["parameters"]["candidate_count"] >= 1
    assert run["parameters"]["chapter_links_available"] is True
    assert run["parameters"]["benchmark_script_status"] == "draft"
    assert run["parameters"]["benchmark_coverage_passed"] is True
    assert run["failed_items"] == []
    assert (bundle / "runs" / "knowledge-note-export" / "run.json").exists()
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    run_types = {row["run_type"]: row["status"] for row in registry["runs"]}
    assert run_types["knowledge_note_export"] == "completed"


def test_export_knowledge_note_registers_completed_with_chapter_links(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-completed"
    _write_basic_bundle(bundle)
    build_smart_summary_chapter_pack(bundle, title="Run Registry Export", target_chapters=1)

    result = export_knowledge_note(bundle, title="Run Registry Export")

    run = result["run_registry"]
    assert run["run_type"] == "knowledge_note_export"
    assert run["status"] == "completed"
    assert run["failed_items"] == []
    assert run["parameters"]["candidate_count"] >= 1
    assert run["parameters"]["linked_content_candidate_count"] >= 1
    artifact_keys = {row["key"] for row in run["artifacts"]}
    assert "knowledge_note" in artifact_keys
    assert "content_candidate_pack" in artifact_keys
    assert "content_material_card" in artifact_keys
    assert "benchmark_script_json" in artifact_keys
    assert "benchmark_script_markdown" in artifact_keys
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    run_types = {row["run_type"]: row["status"] for row in registry["runs"]}
    assert run_types["knowledge_note_export"] == "completed"
