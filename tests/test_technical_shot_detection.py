from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline.shot_boundary_runtime import run_shot_boundary_runtime
from video_knowledge_pipeline.shot_breakdown import build_shot_breakdown
from video_knowledge_pipeline.technical_shot_detection import (
    SCHEMA,
    load_verified_technical_shots,
    run_technical_shot_detection,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(
        root / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "fixture",
            "duration_seconds": 2,
        },
    )
    _write_json(
        root / "timeline.json",
        [
            {"index": 1, "start": 0.0, "end": 1.0, "transcript": "alpha"},
            {"index": 2, "start": 1.0, "end": 2.0, "transcript": "beta"},
        ],
    )
    return root


@pytest.mark.parametrize(
    ("backend", "source_format"),
    [
        ("autoshot", "autoshot_scenes"),
        ("omnishotcut", "omnishotcut_scenes"),
    ],
)
def test_saved_predictions_create_native_technical_shot_contract(
    tmp_path: Path,
    backend: str,
    source_format: str,
) -> None:
    root = _bundle(tmp_path)
    predictions = tmp_path / f"{backend}.json"
    _write_json(
        predictions,
        {
            "schema": "video_knowledge_pipeline.saved_shot_predictions.v1",
            "source_format": source_format,
            "fps": 25.0,
            "scenes": [[0, 24], [25, 49]],
        },
    )

    result = run_technical_shot_detection(
        root,
        backend=backend,
        predictions_json=predictions,
        write=True,
    )

    assert result["schema"] == SCHEMA
    assert result["status"] == "completed"
    assert result["boundary_kind"] == "technical_shot"
    assert result["shot_count"] == 2
    assert result["execution"]["upstream_commit"]
    assert result["operator_boundary"]["no_backend_fallback"] is True
    assert result["operator_boundary"]["no_chapter_or_timeline_range_fallback"] is True
    loaded, provenance = load_verified_technical_shots(root)
    assert len(loaded) == 2
    assert provenance["compatibility"] == "native_v2"

    breakdown = build_shot_breakdown(root, write=False)
    assert breakdown["schema"] == "video_knowledge_pipeline.shot_breakdown.v2"
    assert breakdown["status"] == "completed"
    assert breakdown["shot_count"] == 2


def test_shot_breakdown_blocks_instead_of_treating_timeline_as_shots(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    _write_json(
        root / "exports" / "scene-detection.json",
        {
            "schema": "video_knowledge_pipeline.scene_detection.v1",
            "scenes": [
                {"index": 1, "start": 0.0, "end": 1.0},
                {"index": 2, "start": 1.0, "end": 2.0},
            ],
        },
    )

    result = build_shot_breakdown(root, write=False)

    assert result["status"] == "blocked_missing_technical_shots"
    assert result["ok"] is False
    assert result["shot_count"] == 0
    assert result["shots"] == []
    assert result["operator_boundary"]["chapter_or_timeline_ranges_used_as_shots"] is False


def test_verified_legacy_pyscenedetect_is_read_only_compatible(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    _write_json(
        root / "exports" / "scene-detection.json",
        {
            "schema": "video_knowledge_pipeline.scene_detection.v1",
            "backend": "pyscenedetect",
            "boundary_kind": "technical_shot",
            "scenes": [
                {"index": 1, "start": 0.0, "end": 1.0},
                {"index": 2, "start": 1.0, "end": 2.0},
            ],
        },
    )

    shots, provenance = load_verified_technical_shots(root)

    assert len(shots) == 2
    assert provenance["compatibility"] == "verified_legacy_pyscenedetect"


def test_strict_pyscenedetect_does_not_accept_ffmpeg_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _bundle(tmp_path)
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"fixture")

    def fake_detection(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "backend": "ffmpeg_scene_fallback",
            "fallback_reason": "upstream unavailable",
            "scenes": [{"start": 0.0, "end": 2.0}],
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.technical_shot_detection.run_scene_detection",
        fake_detection,
    )

    result = run_technical_shot_detection(
        root,
        backend="pyscenedetect",
        media_path=media,
        strict=True,
        write=False,
    )

    assert result["status"] == "blocked_backend_fallback"
    assert result["ok"] is False
    assert result["shot_count"] == 0


def test_runtime_injection_preserves_explicit_backend_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _bundle(tmp_path)
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"fixture")
    monkeypatch.setattr(
        "video_knowledge_pipeline.technical_shot_detection._media_provenance",
        lambda path: {"status": "bound", "path": str(path), "sha256": "fixture"},
    )

    def fake_runtime(**kwargs: object) -> dict[str, object]:
        assert kwargs["backend"] == "autoshot"
        return {
            "source_format": "autoshot_scenes",
            "fps": 25.0,
            "scenes": [[0, 24], [25, 49]],
        }

    result = run_technical_shot_detection(
        root,
        backend="autoshot",
        media_path=media,
        _runtime_runner=fake_runtime,
        write=False,
    )

    assert result["status"] == "completed"
    assert result["execution"]["mode"] == "local_gpu_runtime"
    assert result["execution"]["fallback_used"] is False


def test_omnishotcut_missing_checkpoint_fails_closed(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"fixture")
    source = tmp_path / "omnishotcut"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="automatic download is disabled"):
        run_shot_boundary_runtime(
            backend="omnishotcut",
            media_path=media,
            source_root=source,
            checkpoint_path=tmp_path / "missing.ckpt",
        )


def test_runtime_worker_reuses_vkp_media_tool_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"media")
    source = tmp_path / "autoshot"
    source.mkdir()
    checkpoint = tmp_path / "autoshot.pth"
    checkpoint.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "video_knowledge_pipeline.shot_boundary_runtime._git_commit",
        lambda root: "77c82ff826a9301bb173d9be786297a49d73d081",
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.shot_boundary_runtime.probe_video",
        lambda path: SimpleNamespace(
            fps=25.0,
            duration_seconds=2.0,
            sha256="media-sha",
        ),
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.shot_boundary_runtime.local_tool_subprocess_env",
        lambda: {
            "PATH": r"C:\reviewed-ffmpeg",
            "FFMPEG_BINARY": r"C:\reviewed-ffmpeg\ffmpeg.exe",
        },
    )

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "source_format": "autoshot_scenes",
                    "fps": 25.0,
                    "scenes": [[0, 24], [25, 49]],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "video_knowledge_pipeline.shot_boundary_runtime.subprocess.run",
        fake_run,
    )

    result = run_shot_boundary_runtime(
        backend="autoshot",
        media_path=media,
        source_root=source,
        checkpoint_path=checkpoint,
    )

    assert captured["env"] == {
        "PATH": r"C:\reviewed-ffmpeg",
        "FFMPEG_BINARY": r"C:\reviewed-ffmpeg\ffmpeg.exe",
    }
    assert result["automatic_fallback"] is False
    assert result["network_calls"] == 0


def test_saved_vfr_predictions_use_exact_frame_timestamps(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    predictions = tmp_path / "vfr.json"
    _write_json(
        predictions,
        {
            "source_format": "autoshot_scenes",
            "fps": 25.0,
            "frame_timestamps_seconds": [0.0, 0.03, 0.09, 0.15, 0.22],
            "scenes": [[0, 1], [2, 4]],
        },
    )

    result = run_technical_shot_detection(
        root,
        backend="saved",
        predictions_json=predictions,
        source_format="autoshot_scenes",
        write=False,
    )

    assert result["shots"][0]["start"] == 0.0
    assert result["shots"][0]["end"] == 0.09
    assert result["shots"][1]["start"] == 0.09
    assert result["shots"][1]["end"] == pytest.approx(0.29)
    assert result["execution"]["time_basis"] == "vfr_frame_timestamps"


def test_saved_chunk_predictions_apply_absolute_time_offset(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    predictions = tmp_path / "chunk.json"
    _write_json(
        predictions,
        {
            "source_format": "omnishotcut_scenes",
            "fps": 25.0,
            "time_offset_seconds": 300.0,
            "scenes": [[0, 24], [25, 49]],
        },
    )

    result = run_technical_shot_detection(
        root,
        backend="saved",
        predictions_json=predictions,
        source_format="omnishotcut_scenes",
        write=False,
    )

    assert result["shots"][0]["start"] == 300.0
    assert result["shots"][1]["start"] == 301.0
    assert result["execution"]["time_offset_seconds"] == 300.0


def test_saved_vfr_predictions_reject_incomplete_timestamp_map(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    predictions = tmp_path / "broken-vfr.json"
    _write_json(
        predictions,
        {
            "source_format": "autoshot_scenes",
            "fps": 25.0,
            "frame_timestamps_seconds": [0.0, 0.04],
            "scenes": [[0, 2]],
        },
    )

    with pytest.raises(ValueError, match="does not cover every scene frame"):
        run_technical_shot_detection(
            root,
            backend="saved",
            predictions_json=predictions,
            source_format="autoshot_scenes",
            write=False,
        )
