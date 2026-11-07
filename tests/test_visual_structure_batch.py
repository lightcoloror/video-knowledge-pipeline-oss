from __future__ import annotations

import subprocess
from pathlib import Path

from video_knowledge_pipeline.visual_structure_batch import (
    run_visual_structure_ebook_batches,
)


def test_visual_structure_ebook_batch_preview_skips_completed(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "video_knowledge_pipeline.visual_structure_batch._candidate_indexes",
        lambda _root, include_routes: [1, 2, 3, 4],
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.visual_structure_batch._successful_indexes",
        lambda _root: {1},
    )

    result = run_visual_structure_ebook_batches(
        root,
        execute=False,
        batch_size=2,
        write=False,
    )

    assert result["status"] == "planned"
    assert result["pending_indexes"] == [2, 3, 4]
    assert result["completed_indexes"] == [1]
    assert result["operator_boundary"]["child_process_per_batch"] is True
    assert result["progress_events"][-1]["status"] == "completed"


def test_visual_structure_ebook_batch_continues_after_failed_child(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    state = {1}
    calls: list[list[int]] = []
    monkeypatch.setattr(
        "video_knowledge_pipeline.visual_structure_batch._candidate_indexes",
        lambda _root, include_routes: [1, 2, 3, 4, 5],
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.visual_structure_batch._successful_indexes",
        lambda _root: set(state),
    )

    def fake_runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        values = [int(value) for value in command[command.index("--indexes") + 1].split(",")]
        calls.append(values)
        state.update(value for value in values if value != 3)
        return subprocess.CompletedProcess(
            command,
            1 if 3 in values else 0,
            stdout="{}",
            stderr="simulated child failure" if 3 in values else "",
        )

    result = run_visual_structure_ebook_batches(
        root,
        execute=True,
        batch_size=2,
        write=False,
        runner=fake_runner,
    )

    assert calls == [[2, 3], [4, 5]]
    assert result["status"] == "degraded"
    assert result["completed_indexes"] == [1, 2, 4, 5]
    assert result["failed_indexes"] == [3]
    assert result["failed_count"] == 1
    assert result["batch_results"][0]["error"] == "simulated child failure"
    assert result["progress_events"][-1]["status"] == "degraded"
    assert "--indexes 3" in result["retry_command"]
