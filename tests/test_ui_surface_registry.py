from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.ui_surface_registry import (
    GROUPS,
    SURFACES,
    export_ui_surface_registry,
    resolve_ui_surface_registry,
    surface_artifact_cards,
    ui_surface_definitions,
)
from video_knowledge_pipeline.storage import read_json


def test_surface_definitions_have_unique_authoritative_keys() -> None:
    definitions = ui_surface_definitions()

    assert len(definitions) == 14
    assert len({row["surface_id"] for row in definitions}) == len(definitions)
    assert len({row["artifact_key"] for row in definitions}) == len(definitions)
    assert [key for key, _label in GROUPS] == ["overview", "processing", "review", "editing", "evaluation", "settings"]
    assert next(row for row in definitions if row["surface_id"] == "video_workbench")["stability"] == "primary"
    assert next(row for row in definitions if row["surface_id"] == "transcript_editor")["stability"] == "compatibility"
    assert next(row for row in definitions if row["surface_id"] == "subtitle_editor")["description"].startswith("复用 MOYS")


def test_resolve_registry_reports_current_ready_and_missing_surfaces(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle with spaces"
    bundle.mkdir(exist_ok=True)
    (bundle / "task-console.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    manifest = {
        "task_console": "task-console.html",
        "review_html": "review.html",
        "subtitle_editor_html": "not-yet/subtitle-editor.html",
    }

    registry = resolve_ui_surface_registry(bundle, manifest, current_surface="video_workbench")
    by_id = {row["surface_id"]: row for row in registry["surfaces"]}

    assert registry["schema"] == "video_knowledge_pipeline.ui_surface_registry.v1"
    assert registry["surface_count"] == len(SURFACES)
    assert registry["ready_count"] == 3
    assert by_id["video_workbench"]["status"] == "current"
    assert by_id["video_workbench"]["status_label"] == "当前首页"
    assert by_id["task_console"]["status"] == "ready"
    assert by_id["task_console"]["href"] == "task-console.html"
    assert by_id["targeted_review"]["status"] == "not_generated"
    assert by_id["targeted_review"]["service_requirement"] == "targeted_review_loopback"
    assert by_id["subtitle_editor"]["href"] == "not-yet/subtitle-editor.html"
    assert "bundle with spaces" in by_id["task_console"]["resolved_generation_command"]
    assert sum(group["surface_count"] for group in registry["groups"]) == len(SURFACES)


def test_surface_artifact_cards_keep_specialized_pages_and_exclude_home(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    registry = resolve_ui_surface_registry(bundle, {}, current_surface="video_workbench")

    cards = surface_artifact_cards(registry)
    by_key = {row["key"]: row for row in cards}

    assert "video_workbench_html" not in by_key
    assert by_key["targeted_review_html"]["surface_group"] == "review"
    assert by_key["targeted_review_html"]["generation_command"] is None
    assert by_key["model_settings"]["write_mode"] == "loopback_writeback"
    assert by_key["quality_benchmark_html"]["status"] == "未生成"


def test_export_registry_writes_standalone_inventory_without_generating_surfaces(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "task-console.html").write_text("<html></html>", encoding="utf-8")

    registry = export_ui_surface_registry(
        bundle,
        {"task_console": "task-console.html"},
        current_surface="video_workbench",
        write=True,
    )

    assert registry["summary"]["ready_count"] == 2
    assert registry["summary"]["not_generated_count"] == len(SURFACES) - 2
    assert registry["summary"]["write_capable_count"] > 0
    assert registry["summary"]["service_required_count"] > 0
    assert registry["operator_boundary"]["inventory_only"] is True
    assert registry["operator_boundary"]["no_surface_generation"] is True
    assert read_json(bundle / "ui-surface-registry.json")["schema"] == registry["schema"]
    markdown = (bundle / "ui-surface-registry.md").read_text(encoding="utf-8")
    assert "VKP UI Surface Registry" in markdown
    assert "任务控制台" in markdown
    assert "模型与 API 设置" in markdown
    assert "does not generate pages" in markdown
    assert (bundle / "quality-console.html").exists() is False


def test_export_registry_plan_mode_does_not_write_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    registry = export_ui_surface_registry(bundle, {}, write=False)

    assert registry["paths"]["json"].endswith("ui-surface-registry.json")
    assert (bundle / "ui-surface-registry.json").exists() is False
    assert (bundle / "ui-surface-registry.md").exists() is False
