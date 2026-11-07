from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_pipeline.models import TranscriptCue
from video_knowledge_pipeline.speaker_diarization_evidence import (
    assign_speaker_intervals,
    parse_sherpa_diarization_output,
    plan_sherpa_speaker_diarization,
    run_sherpa_speaker_diarization_plan,
)


def _write_wav(path: Path, *, sample_rate: int = 16000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"\x00\x00" * sample_rate * channels)


def _write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "seg-1", "start": 0.0, "end": 2.0, "text": "第一段"},
                    {"id": "seg-2", "start": 2.0, "end": 4.0, "text": "第二段"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_parse_official_sherpa_output_ignores_config_and_progress() -> None:
    rows = parse_sherpa_diarization_output(
        "\n".join(
            [
                'OfflineSpeakerDiarizationConfig(provider="cuda")',
                "Started",
                "0.125 -- 1.750 speaker_00",
                "progress 50.00%",
                "1.750 -- 3.000 speaker_01",
            ]
        )
    )

    assert rows == [
        {
            "interval_id": "speaker-interval-000001",
            "start": 0.125,
            "end": 1.75,
            "speaker": "speaker_00",
            "source_line": 3,
        },
        {
            "interval_id": "speaker-interval-000002",
            "start": 1.75,
            "end": 3.0,
            "speaker": "speaker_01",
            "source_line": 5,
        },
    ]


def test_overlap_assignment_preserves_source_and_never_nearest_fills() -> None:
    cues = [
        TranscriptCue(0.0, 4.0, "原文一", segment_id="seg-a"),
        TranscriptCue(4.0, 6.0, "原文二", segment_id="seg-b"),
        TranscriptCue(8.0, 9.0, "原文三", segment_id="seg-c"),
        TranscriptCue(
            9.0,
            10.0,
            "原文四",
            segment_id="seg-d",
            speaker="upstream-speaker",
        ),
    ]
    intervals = [
        {"interval_id": "i-1", "start": 0.0, "end": 3.0, "speaker": "speaker_00"},
        {"interval_id": "i-2", "start": 3.0, "end": 5.0, "speaker": "speaker_01"},
        {"interval_id": "i-3", "start": 5.0, "end": 6.0, "speaker": "speaker_00"},
        {"interval_id": "i-4", "start": 9.0, "end": 10.0, "speaker": "speaker_01"},
    ]

    result = assign_speaker_intervals(cues, intervals)
    segments = result["segments"]

    assert [row["id"] for row in segments] == ["seg-a", "seg-b", "seg-c", "seg-d"]
    assert [row["text"] for row in segments] == ["原文一", "原文二", "原文三", "原文四"]
    assert [(row["start"], row["end"]) for row in segments] == [
        (0.0, 4.0),
        (4.0, 6.0),
        (8.0, 9.0),
        (9.0, 10.0),
    ]
    assert segments[0]["speaker"] == "speaker_00"
    assert segments[1]["speaker"] == ""
    assert (
        segments[1]["metadata"]["speaker_diarization_candidate"]["state"]
        == "ambiguous_overlap"
    )
    assert segments[2]["speaker"] == ""
    assert (
        segments[2]["metadata"]["speaker_diarization_candidate"]["state"]
        == "uncovered"
    )
    assert segments[3]["speaker"] == "upstream-speaker"
    assert (
        segments[3]["metadata"]["speaker_diarization_candidate"]["state"]
        == "existing_conflicts_with_candidate"
    )
    assert result["policy"]["nearest_fill_enabled"] is False
    assert result["policy"]["segment_split_or_merge_allowed"] is False


def test_plan_requires_exact_runtime_models_and_gpu_without_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "bundle"
    media = tmp_path / "audio.wav"
    transcript = tmp_path / "transcript.json"
    segmentation = tmp_path / "segmentation.onnx"
    embedding = tmp_path / "embedding.onnx"
    _write_wav(media)
    _write_transcript(transcript)
    segmentation.write_bytes(b"segmentation")
    embedding.write_bytes(b"embedding")
    monkeypatch.setattr(
        "video_knowledge_pipeline.speaker_diarization_evidence._runtime_probe",
        lambda _path: {
            "status": "ready",
            "ready": True,
            "blocker": "",
            "official_contract_detected": True,
        },
    )

    plan = plan_sherpa_speaker_diarization(
        bundle,
        media,
        transcript,
        command=sys.executable,
        segmentation_model=segmentation,
        embedding_model=embedding,
        provider="cuda",
    )

    assert plan["status"] == "ready"
    assert plan["runtime"]["gpu_required"] is True
    assert plan["runtime"]["automatic_cpu_fallback"] is False
    assert "--segmentation.provider=cuda" in plan["command_argv"]
    assert "--embedding.provider=cuda" in plan["command_argv"]
    assert plan["operator_boundary"]["models_executed"] is False
    assert plan["operator_boundary"]["downloads_performed"] is False
    assert Path(plan["outputs"]["plan"]).is_file()


def test_plan_and_execute_fail_closed_on_artifact_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "bundle"
    media = tmp_path / "audio.wav"
    transcript = tmp_path / "transcript.json"
    segmentation = tmp_path / "segmentation.onnx"
    embedding = tmp_path / "embedding.onnx"
    _write_wav(media)
    _write_transcript(transcript)
    segmentation.write_bytes(b"segmentation")
    embedding.write_bytes(b"embedding")
    monkeypatch.setattr(
        "video_knowledge_pipeline.speaker_diarization_evidence._runtime_probe",
        lambda _path: {"status": "ready", "ready": True, "blocker": ""},
    )
    plan = plan_sherpa_speaker_diarization(
        bundle,
        media,
        transcript,
        command=sys.executable,
        segmentation_model=segmentation,
        embedding_model=embedding,
    )
    transcript.write_text('{"segments":[]}', encoding="utf-8")

    result = run_sherpa_speaker_diarization_plan(plan["outputs"]["plan"])

    assert result["status"] == "blocked"
    assert result["ok"] is False
    assert "transcript_sha256_changed" in result["blockers"]


def test_execute_writes_candidate_without_mutating_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "bundle"
    media = tmp_path / "audio.wav"
    transcript = tmp_path / "transcript.json"
    segmentation = tmp_path / "segmentation.onnx"
    embedding = tmp_path / "embedding.onnx"
    _write_wav(media)
    _write_transcript(transcript)
    source_before = transcript.read_bytes()
    segmentation.write_bytes(b"segmentation")
    embedding.write_bytes(b"embedding")
    monkeypatch.setattr(
        "video_knowledge_pipeline.speaker_diarization_evidence._runtime_probe",
        lambda _path: {"status": "ready", "ready": True, "blocker": ""},
    )
    plan = plan_sherpa_speaker_diarization(
        bundle,
        media,
        transcript,
        command=sys.executable,
        segmentation_model=segmentation,
        embedding_model=embedding,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.speaker_diarization_evidence.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                'segmentation provider="cuda"; embedding provider="cuda"\n'
                "0.000 -- 2.000 speaker_00\n"
                "2.000 -- 4.000 speaker_01\n"
            ),
            stderr="progress 100.00%\n",
        ),
    )

    result = run_sherpa_speaker_diarization_plan(
        plan["outputs"]["plan"],
        execute=True,
    )

    assert result["status"] == "needs_human_review"
    assert result["speaker_count"] == 2
    assert transcript.read_bytes() == source_before
    candidate_path = Path(result["artifacts"]["candidate_transcript"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert [row["speaker"] for row in candidate["segments"]] == [
        "speaker_00",
        "speaker_01",
    ]
    assert candidate["operator_boundary"]["primary_transcript_mutated"] is False
    assert (
        candidate["operator_boundary"]["human_confirmation_required_before_promotion"]
        is True
    )
