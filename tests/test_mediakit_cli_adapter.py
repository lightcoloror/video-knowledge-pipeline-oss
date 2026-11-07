from __future__ import annotations

import subprocess
from pathlib import Path

from video_knowledge_pipeline.media_task_protocol import build_media_task_plan
from video_knowledge_pipeline.mediakit_cli_adapter import execute_mediakit_cli_task


def _plan(tmp_path: Path) -> dict:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"video-bytes")
    return build_media_task_plan(
        "scene_segmentation",
        execution_location="remote",
        route_id="mediakit-scene",
        route_revision="a" * 64,
        artifact_paths=[video],
        consent_id="consent-1",
        allowed_roots=[tmp_path],
    )


def test_official_cli_executes_consent_bound_local_upload(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "mediakit-cli.exe"
    executable.write_text("fixture", encoding="utf-8")
    invocations: list[list[str]] = []

    def fake_run(args, **kwargs):
        invocations.append(list(args))
        if "query-task" in args:
            return subprocess.CompletedProcess(args, 0, '{"task_id":"task-1","status":"succeeded","segments":[{"start":0,"end":3}]}', "")
        return subprocess.CompletedProcess(args, 0, '{"task_id":"task-1","status":"submitted"}', "")

    result = execute_mediakit_cli_task(
        _plan(tmp_path),
        api_key="test-key",
        command=str(executable),
        run=fake_run,
    )

    assert result["ok"] is True
    assert result["status"] == "succeeded"
    assert result["candidate_only"] is True
    assert result["network_audit"]["remote_requests_made"] is True
    assert result["network_audit"]["provider_managed_local_upload"] is True
    assert "--video-url" in invocations[0]
    assert "query-task" in invocations[1]


def test_official_cli_missing_is_an_honest_runtime_dependency_error(tmp_path: Path) -> None:
    result = execute_mediakit_cli_task(_plan(tmp_path), api_key="test-key", command=str(tmp_path / "missing-cli"))
    assert result["error"]["code"] == "mediakit_cli_unavailable"
    assert result["network_audit"]["remote_requests_made"] is False
