from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline.asr_vad_chunking import prepare_asr_vad_chunks
from video_knowledge_pipeline.cli import main as cli_main


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"video-source")
    vad = tmp_path / "vad.json"
    vad.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.funasr_vad_segments.v1",
                "segments": [
                    {"id": "v1", "start": 0, "end": 20},
                    {"id": "v2", "start": 25, "end": 40},
                    {"id": "v3", "start": 200, "end": 220},
                ],
            }
        ),
        encoding="utf-8",
    )
    return media, vad


def test_prepare_asr_vad_chunks_preview_uses_vad_boundaries(
    tmp_path: Path,
) -> None:
    media, vad = _fixtures(tmp_path)

    result = prepare_asr_vad_chunks(
        media,
        vad,
        tmp_path / "chunks",
        max_request_seconds=60,
    )

    assert result["status"] == "planned"
    assert result["network_call"] is False
    assert result["bitrate_kbps"] == 64
    assert result["chunk_count"] == 2
    assert result["chunks"][0]["core_start"] == 0
    assert result["chunks"][0]["core_end"] == 40
    assert result["chunks"][0]["artifact_start"] == 0
    assert result["chunks"][0]["artifact_end"] == 41.5
    assert result["chunks"][0]["source_vad_segment_ids"] == ["v1", "v2"]
    assert result["chunks"][1]["artifact_start"] == 198.5
    assert result["chunks"][1]["artifact_end"] == 221.5
    assert "64k" in result["chunks"][0]["command"]
    assert Path(result["manifest_path"]).is_file()


def test_prepare_asr_vad_chunks_execute_hashes_all_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media, vad = _fixtures(tmp_path)
    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_vad_chunking.resolve_media_tool",
        lambda _: "ffmpeg-test",
    )

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        Path(command[-1]).write_bytes(("audio-" + Path(command[-1]).stem).encode())
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_vad_chunking.subprocess.run",
        fake_run,
    )

    result = prepare_asr_vad_chunks(
        media,
        vad,
        tmp_path / "chunks",
        max_request_seconds=60,
        execute=True,
    )

    assert result["status"] == "completed"
    assert result["ok"] is True
    assert result["completed_chunk_count"] == 2
    assert result["failed_chunk_count"] == 0
    assert all(row["output_sha256"] for row in result["chunks"])


def test_prepare_asr_vad_chunks_preserves_success_after_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media, vad = _fixtures(tmp_path)
    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_vad_chunking.resolve_media_tool",
        lambda _: "ffmpeg-test",
    )

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        target = Path(command[-1])
        if target.name == "asr-chunk-0002.mp3":
            return SimpleNamespace(returncode=1, stderr="fixture failure")
        target.write_bytes(b"successful-first-chunk")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_vad_chunking.subprocess.run",
        fake_run,
    )

    result = prepare_asr_vad_chunks(
        media,
        vad,
        tmp_path / "chunks",
        max_request_seconds=60,
        execute=True,
    )

    assert result["status"] == "degraded"
    assert result["ok"] is False
    assert result["completed_chunk_count"] == 1
    assert result["failed_chunk_count"] == 1
    assert result["chunks"][0]["status"] == "completed"
    assert result["chunks"][1]["status"] == "failed"
    assert result["chunks"][1]["stderr_tail"] == "fixture failure"


def test_prepare_asr_vad_chunks_cli_is_preview_only(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    media, vad = _fixtures(tmp_path)

    assert (
        cli_main(
            [
                "prepare-cloud-asr-chunks",
                str(media),
                str(vad),
                str(tmp_path / "chunks"),
                "--max-request-seconds",
                "60",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "planned"
    assert payload["execute"] is False
    assert payload["network_call"] is False
