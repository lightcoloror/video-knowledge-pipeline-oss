from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.multimodal_frame_analyzer import _candidates
from video_knowledge_pipeline.task_console import export_task_console
from video_knowledge_pipeline.vision_review_queue import vision_review_queue
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight
from video_knowledge_pipeline.vision_review_triage import vision_review_triage


def _write_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "title": "queue test", "review_html": "review.html"}, ensure_ascii=False), encoding="utf-8")
    timeline = []
    for idx in range(1, 8):
        item = {
            "index": idx,
            "start": idx * 10,
            "end": idx * 10 + 8,
            "transcript": "这里看屏幕，展示工具名和操作步骤。",
            "visual_route": "semantic_frame",
            "frame_paths": [str(bundle / "assets" / f"{idx:04d}.jpg")],
            "quality_issues": ["semantic_frame_without_analysis", "missing_visual_understanding", "missing_visual_text"],
        }
        if idx == 2:
            item["visual_understanding"] = {"schema": "lecture_visual_understanding.v1", "objects": ["covered"], "validation_status": "ok"}
            item["quality_issues"] = ["missing_visual_text"]
        if idx == 3:
            item["visual_understanding"] = {"schema": "lecture_visual_understanding.v1", "parse_failed": True}
            item["quality_issues"] = ["model_output_parse_failed", "visual_understanding_incomplete"]
        timeline.append(item)
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    (bundle / "review.html").write_text("<html>review</html>", encoding="utf-8")


def test_vision_review_queue_batches_pending_and_failed_items(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    vision_review_triage(bundle, min_score=3, write=True)

    result = vision_review_queue(bundle, min_score=10, batch_size=2, max_items=4, write=True)

    assert result["schema"] == "video_knowledge_pipeline.vision_review_queue.v1"
    assert result["total_candidates"] == 1
    assert result["batch_counts"]["total"] == 1
    assert result["pending_items"] == 1
    assert result["batches"][0]["pending_indexes"] == [3]
    assert result["batches"][0]["failed_or_incomplete_indexes"] == [3]
    assert "-Indexes '3'" in result["batches"][0]["retry_command"]
    assert (bundle / "vision-review-queue.html").exists()
    assert (bundle / "vision-review-queue-run.ps1").exists()

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vision_review_queue_html"] == "vision-review-queue.html"
    assert manifest["mcp_vision_review_queue_args"] == "mcp-vision-review-queue.args.json"
    assert manifest["run_artifact_registry_json"] == "run-artifact-registry.json"
    assert (bundle / "runs" / "vision-review-queue" / "run.json").exists()
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert registry["runs"][0]["run_id"] == "vision-review-queue"
    run = json.loads((bundle / "runs" / "vision-review-queue" / "run.json").read_text(encoding="utf-8"))
    assert run["retry_command"] == result["batches"][0]["retry_command"]
    assert "-Indexes '3'" in run["retry_command"]
    assert len(run["failed_items"]) == 1
    reasons = {row["reason"] for row in run["failed_items"]}
    assert reasons == {"visual_understanding_failed_or_incomplete"}
    first_failed = next(row for row in run["failed_items"] if row["index"] == 3)
    assert first_failed["batch_id"] == result["batches"][0]["batch_id"]
    assert first_failed["batch_status"] == "failed"
    assert first_failed["pending_indexes"] == result["batches"][0]["pending_indexes"]
    assert first_failed["suggested_next_tool"] == "run_multimodal_frame_analysis"
    assert "-Execute" in first_failed["suggested_retry_command"]


def test_task_console_links_vision_review_queue(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    vision_review_triage(bundle, min_score=3, write=True)
    vision_review_queue(bundle, min_score=10, batch_size=2, max_items=4, write=True)

    console = export_task_console(bundle, write=True, refresh=False)

    command_keys = {row["key"] for row in console["commands"]}
    artifact_keys = {row["key"] for row in console["artifacts"]}
    html = (bundle / "task-console.html").read_text(encoding="utf-8")
    assert "vision_queue" in command_keys
    assert "vision_review_queue_html" in artifact_keys
    assert "run_artifact_registry_report" in artifact_keys
    assert "任务产物索引" in html
    assert "vision-review-queue" in html
    action_rows = {row["key"]: row for row in console["subqueue_action_plan"]["rows"]}
    assert action_rows["vision:review_triage"]["primary_command"]
    assert "-Indexes '3'" in action_rows["vision:review_triage"]["primary_command"]
    assert "Batch" in action_rows["vision:review_triage"]["blocked_reason"]


def test_vision_review_queue_cli_parser() -> None:
    args = build_parser().parse_args(["vision-review-queue", "bundle", "--batch-size", "8", "--max-items", "48", "--min-score", "11"])

    assert args.command == "vision-review-queue"
    assert args.batch_size == 8
    assert args.max_items == 48
    assert args.min_score == 11

def test_vision_preflight_retries_parse_failed_visual_understanding(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    result = vision_execution_preflight(
        bundle,
        provider_config={"provider": "custom_openai_compatible", "api_key": "test-key", "base_url": "http://127.0.0.1/v1", "model": "vision-test"},
        semantic_limit=10,
        include_temporal=False,
        semantic_indexes=[2, 3],
        write=False,
    )

    assert result["ready_to_execute"] is True
    assert result["selected_indexes"]["semantic"] == [3]
    assert result["confirmation"]["semantic_confirm_vision_indexes"] == "3"

def test_multimodal_candidates_retry_parse_failed_visual_understanding(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    indexes = [row["index"] for row in _candidates(bundle, timeline)]

    assert 2 not in indexes
    assert 3 in indexes
