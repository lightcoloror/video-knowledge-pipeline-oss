from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline.asr_vad_activity_audit import (
    audit_asr_vad_audio_activity,
)
from video_knowledge_pipeline.cli import main as cli_main


def _fixtures(tmp_path: Path, segments: list[dict[str, object]]) -> tuple[Path, Path]:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"local-media-fixture")
    vad = tmp_path / "vad.json"
    vad.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return media, vad


def test_vad_activity_audit_preview_does_not_run_probe(tmp_path: Path) -> None:
    media, vad = _fixtures(tmp_path, [{"start": 0, "end": 12}])

    result = audit_asr_vad_audio_activity(
        media,
        vad,
        output_path=tmp_path / "audit.json",
        duration_seconds=20,
        execute=False,
    )

    assert result["status"] == "planned"
    assert result["network_call"] is False
    assert result["duration_source"] == "explicit"
    assert result["source_media"]["sha256"] == hashlib.sha256(
        media.read_bytes()
    ).hexdigest()
    assert result["operator_boundary"]["candidate_evidence_only"] is True
    assert Path(result["output_path"]).is_file()


def test_vad_activity_audit_exposes_only_candidate_blind_spots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media, vad = _fixtures(
        tmp_path,
        [{"id": "v1", "start": 0, "end": 5}, {"id": "v2", "start": 10, "end": 12}],
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_vad_activity_audit.probe_video",
        lambda _: SimpleNamespace(duration_seconds=20.0),
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_vad_activity_audit.probe_audio_silence",
        lambda *_args, **_kwargs: {
            "ok": True,
            "status": "completed",
            "activity_intervals": [
                {"start": 0, "end": 5},
                {"start": 8, "end": 15},
            ],
        },
    )

    result = audit_asr_vad_audio_activity(media, vad, execute=True, write=False)

    assert result["status"] == "review_required"
    assert result["ok"] is True
    assert result["vad_coverage_verified"] is False
    assert result["candidate_gap_count"] == 2
    assert [(row["start"], row["end"]) for row in result["candidate_gaps"]] == [
        (8.0, 10.0),
        (12.0, 15.0),
    ]
    assert all(row["candidate_only"] for row in result["candidate_gaps"])
    assert result["operator_boundary"]["vad_segments_modified"] is False


def test_vad_activity_audit_passes_when_vad_covers_audio_activity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media, vad = _fixtures(tmp_path, [{"start": 0, "end": 20}])
    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_vad_activity_audit.probe_audio_silence",
        lambda *_args, **_kwargs: {
            "ok": True,
            "status": "completed",
            "activity_intervals": [{"start": 0, "end": 20}],
        },
    )

    result = audit_asr_vad_audio_activity(
        media, vad, duration_seconds=20, execute=True, write=False
    )

    assert result["status"] == "passed"
    assert result["vad_coverage_verified"] is True
    assert result["candidate_gap_count"] == 0


def test_vad_activity_audit_cli_defaults_to_preview(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    media, vad = _fixtures(tmp_path, [{"start": 0, "end": 12}])

    assert (
        cli_main(
            [
                "asr-vad-activity-audit",
                str(media),
                str(vad),
                "--duration-seconds",
                "20",
                "--no-write",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "planned"
    assert result["execute"] is False
    assert not Path(result["output_path"]).exists()
