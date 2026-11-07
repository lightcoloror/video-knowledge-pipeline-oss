from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.local_video_run as local_video_run


def _stub_common(monkeypatch, media: Path) -> None:
    monkeypatch.setattr(
        local_video_run,
        "prepare_video_source",
        lambda *args, **kwargs: {
            "status": "ready",
            "source_kind": "local_file",
            "local_media_path": str(media),
        },
    )
    monkeypatch.setattr(
        local_video_run,
        "asr_environment_status",
        lambda: {"ok": True, "available_tools": ["fixture"], "recommended_order": []},
    )
    monkeypatch.setattr(
        local_video_run,
        "_prepare_pre_asr_context",
        lambda *args, **kwargs: {"hotword_text": "", "hotword_count": 0},
    )


def test_plan_asr_exception_still_writes_structured_failure_report(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "synthetic.mp4"
    media.write_bytes(b"synthetic-not-real-media")
    output = tmp_path / "run"
    _stub_common(monkeypatch, media)
    monkeypatch.setattr(
        local_video_run,
        "local_runtime_preflight",
        lambda: {
            "ok": True,
            "status": "ready",
            "capabilities": {"media": {"ffmpeg": {"available": True}, "ffprobe": {"available": True}}},
            "recovery_commands": [],
        },
        raising=False,
    )
    monkeypatch.setattr(
        local_video_run,
        "plan_asr_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic plan failure")),
    )

    result = local_video_run.prepare_local_video_run(media, output, title="Synthetic")

    assert result["ok"] is False
    assert result["status"] in {"partial_failure", "failed"}
    assert result["failed_stage"] == "asr_plan"
    assert "asr_plan" in result["failed_stages"]
    assert any(row["stage"] == "asr_plan" and row["status"] == "failed" for row in result["stage_results"])
    assert any(row["stage"] == "asr_plan" and row["command"] for row in result["recovery_commands"])
    json_path = output / "video-knowledge-run.json"
    markdown_path = output / "video-knowledge-run.md"
    assert json_path.exists()
    assert markdown_path.exists()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["failed_stage"] == "asr_plan"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "partial_failure" in markdown or "failed" in markdown
    assert "asr_plan" in markdown
    assert "synthetic plan failure" in markdown


def test_missing_media_tools_still_writes_partial_failure_report(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "synthetic.mp4"
    media.write_bytes(b"synthetic-not-real-media")
    output = tmp_path / "run"
    _stub_common(monkeypatch, media)
    monkeypatch.setattr(
        local_video_run,
        "local_runtime_preflight",
        lambda: {
            "ok": False,
            "status": "not_ready",
            "capabilities": {
                "media": {
                    "ffmpeg": {"available": False, "path": ""},
                    "ffprobe": {"available": False, "path": ""},
                }
            },
            "failed_checks": ["media:ffmpeg", "media:ffprobe"],
            "recovery_commands": [
                {"key": "configure_media_tools", "command": "set FFMPEG_BINARY=<ffmpeg-path>"}
            ],
        },
        raising=False,
    )

    result = local_video_run.prepare_local_video_run(
        media,
        output,
        title="Synthetic",
        plan_asr=False,
        build_initial_bundle=False,
    )

    assert result["ok"] is False
    assert result["status"] == "partial_failure"
    assert result["failed_stage"] == "media_preflight"
    assert "media_preflight" in result["failed_stages"]
    assert result["runtime_preflight"]["capabilities"]["media"]["ffmpeg"]["available"] is False
    assert result["recovery_commands"]
    assert (output / "video-knowledge-run.json").exists()
    assert "media_preflight" in (output / "video-knowledge-run.md").read_text(encoding="utf-8")
