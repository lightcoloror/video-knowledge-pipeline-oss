from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.asr_ab_compare import compare_asr_ab_sample


def _write_reference(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 4.0, "text": "你好", "speaker": "S01"},
                    {"start": 4.0, "end": 8.0, "text": "请讲", "speaker": "S02"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_run(path: Path, *, campp_ready: bool) -> None:
    variants = [
        {
            "key": "sensevoice_full_punc",
            "status": "ok",
            "metrics": {
                "segment_count": 2,
                "char_count": 4,
                "punctuation_count": 1,
                "duration_seconds": 8.0,
                "speaker_count": 0,
                "speaker_labeled_segment_count": 0,
            },
        }
    ]
    if campp_ready:
        variants.append(
            {
                "key": "sensevoice_full_punc_campp",
                "status": "ok",
                "metrics": {
                    "segment_count": 2,
                    "char_count": 4,
                    "punctuation_count": 1,
                    "duration_seconds": 8.0,
                    "speaker_count": 2,
                    "speaker_labeled_segment_count": 2,
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "workspace_dir": str(path.parent),
                "sample_media_path": str(path.parent / "sample.wav"),
                "variants": variants,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_dialogue_reference_blocks_text_only_production_recommendation(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    run = tmp_path / "run.json"
    _write_reference(reference)
    _write_run(run, campp_ready=False)

    result = compare_asr_ab_sample(
        run,
        reference_transcript=reference,
        start_seconds=0.0,
        end_seconds=8.0,
        write=False,
    )

    assert result["primary_recommendation"] == "sensevoice_full_punc"
    assert result["production_recommendation"] == (
        "blocked_until_speaker_diarization_ready"
    )
    assert result["status"] == "primary_text_ready_speaker_diarization_pending"
    assert result["speaker_requirement"]["required"] is True
    assert result["speaker_requirement"]["min_speaker_count"] == 2
    assert result["gates"]["speaker_requirement_met"] is False
    assert result["gates"]["speaker_ready_variants"] == []


def test_fully_labeled_campp_candidate_meets_dialogue_speaker_gate(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    run = tmp_path / "run.json"
    _write_reference(reference)
    _write_run(run, campp_ready=True)

    result = compare_asr_ab_sample(
        run,
        reference_transcript=reference,
        start_seconds=0.0,
        end_seconds=8.0,
        write=False,
    )

    assert result["production_recommendation"] == (
        "blocked_until_speaker_quality_evaluation_passes"
    )
    assert result["speaker_evaluation_candidate"] == "sensevoice_full_punc_campp"
    assert result["status"] == "speaker_candidate_ready_for_diarization_evaluation"
    assert result["gates"]["speaker_requirement_met"] is True
    assert result["gates"]["speaker_ready_variants"] == [
        "sensevoice_full_punc_campp"
    ]
