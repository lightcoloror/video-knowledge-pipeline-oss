from __future__ import annotations

import json
import subprocess
from pathlib import Path

import video_knowledge_pipeline.funasr_chunked_runner as runner
from video_knowledge_pipeline.asr_response_quality import assess_asr_response


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_out_of_bounds_word_timing_cannot_hide_external_vad_gap() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 2,
                    "text": "甲乙",
                    "words": [
                        {"word": "甲", "start": 0, "end": 1},
                        {"word": "乙", "start": 3, "end": 5},
                    ],
                }
            ]
        },
        vad_intervals=[{"start": 3, "end": 4}],
        media_duration_seconds=5,
    )

    assert result["status"] == "degraded"
    assert result["segments"][0]["coverage_evidence"] == "segment_bounds"
    assert (
        result["segments"][0]["coverage_evidence_reason"]
        == "word_timestamp_bounds_invalid"
    )
    assert result["speech_coverage"]["coverage_ratio"] == 0.0
    assert result["speech_coverage"]["gaps"][0]["start"] == 3.0
    assert result["speech_coverage"]["gaps"][0]["end"] == 4.0


def test_minor_provider_word_boundary_drift_is_clipped_and_accepted() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 2,
                    "text": "甲乙",
                    "words": [
                        {"word": "甲", "start": 0, "end": 1},
                        {"word": "乙", "start": 1, "end": 2.3},
                    ],
                }
            ]
        },
        vad_intervals=[{"start": 0, "end": 2}],
    )

    assert result["status"] == "passed"
    assert result["segments"][0]["coverage_evidence"] == "word_timestamps"
    assert result["segments"][0]["coverage_intervals"][-1]["end"] == 2.0


def test_cumulative_short_vad_gaps_do_not_pass_at_zero_coverage() -> None:
    result = assess_asr_response(
        {"segments": [{"start": 10, "end": 11, "text": "旁证文本"}]},
        vad_intervals=[
            {"start": 0, "end": 1.9},
            {"start": 3, "end": 4.9},
            {"start": 6, "end": 7.9},
        ],
        media_duration_seconds=12,
    )

    assert result["status"] == "degraded"
    assert result["speech_coverage"]["coverage_ratio"] == 0.0
    assert len(result["speech_coverage"]["gaps"]) == 3
    assert result["coverage_gap_count"] == 3


def test_one_small_gap_within_ratio_and_cumulative_budget_remains_noise() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 98.1,
                    "text": "内容" * 40,
                }
            ]
        },
        vad_intervals=[{"start": 0, "end": 100}],
    )

    assert result["status"] == "passed"
    assert result["speech_coverage"]["coverage_ratio"] == 0.981
    assert result["speech_coverage"]["gaps"] == []


def test_many_short_word_gaps_pass_when_long_media_coverage_stays_above_floor() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": float(position * 100),
                    "end": float(position * 100 + 99),
                    "text": f"第{position}段" + "内容" * 60,
                }
                for position in range(100)
            ]
        },
        vad_intervals=[{"start": 0.0, "end": 10000.0}],
        media_duration_seconds=10000.0,
    )

    assert result["status"] == "passed"
    assert result["speech_coverage"]["coverage_ratio"] == 0.99
    assert result["speech_coverage"]["gaps"] == []
    assert result["coverage_gap_count"] == 0


def test_one_material_gap_still_fails_even_above_relative_coverage_floor() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {"start": 0.0, "end": 4998.0, "text": "前段内容"},
                {"start": 5000.1, "end": 10000.0, "text": "后段内容"},
            ]
        },
        vad_intervals=[{"start": 0.0, "end": 10000.0}],
        media_duration_seconds=10000.0,
    )

    assert result["status"] == "degraded"
    assert result["speech_coverage"]["coverage_ratio"] > 0.95
    assert result["speech_coverage"]["gaps"] == [
        {
            "segment_id": "vad-gap-0001",
            "position": 0,
            "start": 4998.0,
            "end": 5000.1,
            "duration_seconds": 2.1,
            "reasons": ["missing_speech_coverage"],
            "original_text": "",
            "preserve_original_text": True,
        }
    ]
    assert result["coverage_gap_count"] == 1


def test_explicit_chunk_repair_replaces_prior_checkpoint_result(
    tmp_path: Path, monkeypatch
) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    chunks = [tmp_path / "chunk-0000.wav", tmp_path / "chunk-0001.wav"]
    for chunk in chunks:
        chunk.write_bytes(b"audio")
    output = tmp_path / "raw-asr-output.json"
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks)

    def first_run(command, **_kwargs):
        command = list(command)
        source = Path(_option(command, "--input"))
        Path(_option(command, "--output")).write_text(
            json.dumps({"result": [{"text": f"old-{source.stem}"}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", first_run)
    first = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
    )
    assert first["status"] == "completed"

    calls: list[str] = []

    def repair_run(command, **_kwargs):
        command = list(command)
        source = Path(_option(command, "--input"))
        calls.append(source.name)
        Path(_option(command, "--output")).write_text(
            json.dumps({"result": [{"text": "repaired"}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", repair_run)
    repaired = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
        chunk_indexes=[1],
    )

    assert calls == ["chunk-0001.wav"]
    assert repaired["status"] == "completed"
    assert repaired["canonical_complete"] is True
    assert repaired["successful_chunk_indexes"] == [0, 1]
    assert any(row.get("text") == "repaired" for row in repaired["chunk_results"])
    assert not any(row.get("text") == "old-chunk-0001" for row in repaired["chunk_results"])


def test_fresh_targeted_subset_is_not_misreported_as_canonical_complete(
    tmp_path: Path, monkeypatch
) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    chunks = [tmp_path / "chunk-0000.wav", tmp_path / "chunk-0001.wav"]
    for chunk in chunks:
        chunk.write_bytes(b"audio")
    output = tmp_path / "repair-only.json"
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks)

    def fake_run(command, **_kwargs):
        command = list(command)
        Path(_option(command, "--output")).write_text(
            json.dumps({"result": [{"text": "targeted repair"}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
        chunk_indexes=[1],
    )

    assert result["status"] == "partial_targeted_completed"
    assert result["ok"] is True
    assert result["canonical_complete"] is False
    assert result["targeted_repair_completed"] is True
    assert result["successful_chunk_indexes"] == [1]
