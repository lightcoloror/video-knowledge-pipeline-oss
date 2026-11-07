from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_vad_profile_comparison import (
    LABEL_SCHEMA,
    compare_asr_vad_profiles,
)
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.video import sha256_file


def _profile_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"audio fixture")
    authoritative = tmp_path / "authoritative-vad.json"
    permissive = tmp_path / "permissive-vad.json"
    common = {
        "schema": "video_knowledge_pipeline.funasr_vad_segments.v1",
        "input": str(media.resolve()),
        "model": "fsmn-vad",
        "resolved_model": "D:/models/fsmn-vad",
        "model_revision": "v2.0.4",
        "device": "cuda",
    }
    write_json(
        authoritative,
        {
            **common,
            "evidence_profile": "authoritative",
            "candidate_only": False,
            "vad_settings": {
                "max_single_segment_time_ms": 30000,
                "speech_noise_threshold": 0.6,
                "max_end_silence_time_ms": 800,
            },
            "segments": [
                {"start": 0, "end": 5},
                {"start": 10, "end": 20},
            ],
        },
    )
    write_json(
        permissive,
        {
            **common,
            "evidence_profile": "candidate-permissive",
            "candidate_only": True,
            "vad_settings": {
                "max_single_segment_time_ms": 30000,
                "speech_noise_threshold": 0.35,
                "max_end_silence_time_ms": 1100,
            },
            "segments": [
                {"start": 0, "end": 5},
                {"start": 6, "end": 10},
                {"start": 10, "end": 21},
            ],
        },
    )
    audit = tmp_path / "activity-audit.json"
    write_json(
        audit,
        {
            "schema": "video_knowledge_pipeline.asr_vad_activity_audit.v1",
            "status": "review_required",
            "vad_sha256": sha256_file(authoritative),
            "audio_probe": {
                "activity_intervals": [
                    {"start": 0, "end": 24},
                ]
            },
            "candidate_gaps": [
                {
                    "candidate_id": "audio-activity-gap-0001",
                    "start": 5,
                    "end": 10,
                },
                {
                    "candidate_id": "audio-activity-gap-0002",
                    "start": 20,
                    "end": 24,
                },
            ],
        },
    )
    return authoritative, permissive, audit


def test_profile_comparison_keeps_same_model_support_candidate_only(
    tmp_path: Path,
) -> None:
    authoritative, permissive, audit = _profile_fixtures(tmp_path)

    result = compare_asr_vad_profiles(
        authoritative,
        permissive,
        audit,
        output_path=tmp_path / "comparison.json",
    )

    assert result["status"] == "awaiting_human_labels"
    assert result["same_model_support_counts"] == {
        "same_model_permissive_supported": 1,
        "same_model_permissive_partial": 1,
        "unresolved": 0,
    }
    assert [row["permissive_coverage_ratio"] for row in result["candidate_comparisons"]] == [
        0.8,
        0.25,
    ]
    assert result["decision_boundary"]["production_default_change_allowed"] is False
    assert result["decision_boundary"]["same_model_second_pass_is_independent_evidence"] is False
    assert (tmp_path / "comparison.json").is_file()
    template_path = Path(result["human_labels"]["template_path"])
    assert template_path.is_file()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["authoritative_vad_sha256"] == sha256_file(authoritative)
    assert template["permissive_vad_sha256"] == sha256_file(permissive)
    assert [row["label"] for row in template["labels"]] == ["", ""]


def test_profile_comparison_with_no_candidates_needs_no_human_labels(
    tmp_path: Path,
) -> None:
    authoritative, permissive, audit = _profile_fixtures(tmp_path)
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    audit_payload["status"] = "passed"
    audit_payload["candidate_gaps"] = []
    write_json(audit, audit_payload)

    result = compare_asr_vad_profiles(
        authoritative_vad_path=authoritative,
        permissive_vad_path=permissive,
        activity_audit_path=audit,
        output_path=tmp_path / "no-candidates.json",
        write=True,
    )

    assert result["status"] == "no_candidates"
    assert result["human_labels"]["candidate_count"] == 0
    assert result["human_labels"]["template_path"] == ""
    assert result["human_labels"]["template_written"] is False
    assert "no blind-spot calibration signal" in result["recommended_action"]
    assert not (tmp_path / "asr-vad-human-labels.template.json").exists()


def test_profile_comparison_calculates_human_labeled_screening_metrics(
    tmp_path: Path,
) -> None:
    authoritative, permissive, audit = _profile_fixtures(tmp_path)
    labels = tmp_path / "human-labels.json"
    write_json(
        labels,
        {
            "schema": LABEL_SCHEMA,
            "authoritative_vad_sha256": sha256_file(authoritative),
            "permissive_vad_sha256": sha256_file(permissive),
            "activity_audit_sha256": sha256_file(audit),
            "labels": [
                {
                    "candidate_id": "audio-activity-gap-0001",
                    "label": "speech",
                },
                {
                    "candidate_id": "audio-activity-gap-0002",
                    "label": "non_speech",
                },
            ],
        },
    )

    result = compare_asr_vad_profiles(
        authoritative,
        permissive,
        audit,
        labels_path=labels,
        write=False,
    )

    metrics = result["human_labels"]
    assert metrics["status"] == "calibration_labeled"
    assert metrics["confusion_counts"] == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 1,
    }
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_profile_comparison_rejects_stale_activity_audit(tmp_path: Path) -> None:
    authoritative, permissive, audit = _profile_fixtures(tmp_path)
    write_json(
        authoritative,
        {
            **__import__("json").loads(authoritative.read_text(encoding="utf-8")),
            "segments": [{"start": 0, "end": 4}],
        },
    )

    with pytest.raises(
        ValueError, match="does not bind the authoritative VAD hash"
    ):
        compare_asr_vad_profiles(
            authoritative,
            permissive,
            audit,
            write=False,
        )


def test_profile_comparison_rejects_non_permissive_candidate(tmp_path: Path) -> None:
    authoritative, permissive, audit = _profile_fixtures(tmp_path)
    payload = __import__("json").loads(permissive.read_text(encoding="utf-8"))
    payload["candidate_only"] = False
    write_json(permissive, payload)

    with pytest.raises(ValueError, match="must be explicitly candidate-only"):
        compare_asr_vad_profiles(
            authoritative,
            permissive,
            audit,
            write=False,
        )


def test_profile_comparison_cli_is_local_and_write_optional(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    authoritative, permissive, audit = _profile_fixtures(tmp_path)

    assert (
        cli_main(
            [
                "asr-vad-profile-compare",
                str(authoritative),
                str(permissive),
                str(audit),
                "--no-write",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["decision_boundary"]["network_call"] is False
    assert not Path(result["output_path"]).exists()
