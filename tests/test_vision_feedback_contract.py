from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.multimodal_frame_analyzer import (
    _new_vision_batch_progress,
    _write_vision_batch_progress,
    call_vision_model_with_retries,
)


def test_vision_progress_file_reconciles_counts(tmp_path: Path) -> None:
    candidates = [{"index": 2}, {"index": 5}]
    progress = _new_vision_batch_progress(
        tmp_path,
        kind="semantic_frame",
        candidates=candidates,
        execute=True,
        gate=None,
    )
    path = _write_vision_batch_progress(
        tmp_path,
        progress,
        results=[
            {"index": 2, "executed": True, "ok": True, "complete": True},
            {"index": 5, "executed": True, "ok": False, "truncated": True},
        ],
        current_position=2,
        status="partial_failure",
    )
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value["status"] == "partial_failure"
    assert value["selected_indexes"] == [2, 5]
    assert value["processed_count"] == 2
    assert value["complete_count"] == 1
    assert value["truncated_count"] == 1
    assert value["failed_count"] == 1
    assert "_started_perf_counter" not in value


def test_vision_retry_delay_is_exponential(monkeypatch) -> None:
    delays: list[float] = []
    responses = iter(
        [
            {"ok": False, "error": "timeout"},
            {"ok": False, "error": "connection reset"},
            {"ok": True, "content": "{}"},
        ]
    )
    monkeypatch.setattr("video_knowledge_pipeline.multimodal_frame_analyzer.time.sleep", delays.append)
    result = call_vision_model_with_retries(
        provider_config={},
        prompt="synthetic",
        image_paths=["synthetic.jpg"],
        attempts=3,
        delay_seconds=0.25,
        call_model=lambda **_: next(responses),
    )

    assert result["ok"] is True
    assert delays == [0.25, 0.5]
    assert [row.get("retry_delay_seconds") for row in result["attempts"]] == [
        0.25,
        0.5,
        None,
    ]


def test_cli_accepts_repeatable_indexes_and_explicit_temporal_tokens() -> None:
    parser = build_parser()
    semantic = parser.parse_args(
        ["run-multimodal-frame-analysis", "bundle", "--index", "2", "--index", "5"]
    )
    temporal = parser.parse_args(
        ["run-temporal-visual-analysis", "bundle", "--index", "8", "--max-tokens", "2048"]
    )
    tag_delta = parser.parse_args(
        ["run-temporal-tag-delta", "bundle", "--min-frames", "5", "--frame-count", "8", "--indexes", "2,5"]
    )

    assert semantic.index == [2, 5]
    assert temporal.index == [8]
    assert temporal.max_tokens == 2048
    assert tag_delta.min_frames == 5
    assert tag_delta.frame_count == 8
    assert tag_delta.indexes == "2,5"
