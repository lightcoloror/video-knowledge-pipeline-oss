from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_targeted_retry_merge import (
    _crop_retry_text,
    merge_asr_targeted_retry_reports,
)
from video_knowledge_pipeline.storage import read_json, write_json


def _fixture(tmp_path: Path, *, ambiguous: bool = False) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"title": "targeted retry"})
    write_json(
        bundle / "source-arbitrated-transcript.json",
        {
            "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
            "segments": [
                {"index": 0, "source_segment_index": 0, "start": 0, "end": 10, "text": "保留前段"},
                {
                    "index": 1,
                    "source_segment_index": 1,
                    "start": 10,
                    "end": 13,
                    "text": "请逐字转写整个字。",
                    "raw_text": "请逐字转写整个字。",
                    "corrected_text": "请逐字转写整个字。",
                },
                {"index": 2, "source_segment_index": 2, "start": 13, "end": 20, "text": "保留后段"},
            ],
        },
    )
    artifact = tmp_path / "retry.wav"
    artifact.write_bytes(b"RIFF-targeted-retry")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    plan = tmp_path / "authorization-plan.json"
    write_json(
        plan,
        {
            "destination": "https://api.groq.com",
            "provider": "groq",
            "model": "whisper-large-v3-turbo",
            "route": {
                "route_id": "pool-groq-asr",
                "route_revision": "a" * 64,
                "virtual_model": "vkp-remote-asr-pool-groq-asr-aaaaaaaaaaaa",
            },
            "limits": {
                "max_logical_calls": 1,
                "max_calls_per_artifact": 1,
                "external_retries": 0,
                "max_estimated_cost_usd": 0.025,
            },
            "artifact_manifest": [
                {
                    "retry_id": "retry-0001",
                    "path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "sha256": digest,
                    "source_window_seconds": {"start": 9.0, "end": 14.0},
                    "source_segment_ids": ["1"],
                }
            ],
        },
    )
    consent = tmp_path / "consent.json"
    write_json(
        consent,
        {
            "schema": "video_knowledge_pipeline.model_connector_consent.v2",
            "consent_id": "consent-targeted-1",
            "user_confirmed_data_export": True,
            "operator_confirmation": {"confirmed": True},
            "instructions": "逐字转写这段中文音频。",
            "instruction_transport": "audit_only",
            "asr_prompt": "明亚保险 Excel",
            "asr_prompt_transport": "provider_audio_prompt",
            "artifacts": [
                {
                    "path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "sha256": digest,
                    "data_type": "audio",
                }
            ],
            "authorized_destinations": ["https://api.groq.com"],
            "route": {
                "route_id": "pool-groq-asr",
                "route_revision": "a" * 64,
                "virtual_model": "vkp-remote-asr-pool-groq-asr-aaaaaaaaaaaa",
            },
            "scope": {
                "max_calls": 1,
                "max_estimated_cost_usd": 0.025,
                "max_retries_per_call": 0,
            },
        },
    )
    execution = tmp_path / "connector-execution.json"
    segment = {
        "start": 0.0 if ambiguous else 1.0,
        "end": 6.0 if ambiguous else 4.0,
        "text": "这里是恢复后的正确正文。",
        "avg_logprob": -0.1,
        "compression_ratio": 1.1,
        "no_speech_prob": 0.01,
    }
    write_json(
        execution,
        {
            "task": "cloud_asr",
            "production_qualified": True,
            "artifact_paths": [str(artifact)],
            "consent_id": "consent-targeted-1",
            "consent_path": str(consent),
            "route": {
                "route_id": "pool-groq-asr",
                "route_revision": "a" * 64,
                "virtual_model": "vkp-remote-asr-pool-groq-asr-aaaaaaaaaaaa",
                "deployments": [
                    {
                        "provider": "groq_asr",
                        "litellm_provider": "groq",
                        "model": "whisper-large-v3-turbo",
                        "base_url": "https://api.groq.com/openai/v1",
                    }
                ]
            },
            "model_result": {
                "runtime_result": {
                    "request_options": {"max_retries": 0},
                    "raw_output": {"segments": [segment]},
                }
            },
        },
    )
    return bundle, plan, execution


def test_targeted_retry_replaces_only_the_authorized_bad_segment(tmp_path: Path) -> None:
    bundle, plan, execution = _fixture(tmp_path)

    result = merge_asr_targeted_retry_reports(bundle, plan, [execution], write=True)
    canonical = read_json(bundle / "source-arbitrated-transcript.json")

    assert result["status"] == "completed"
    assert result["production_qualified"] is True
    assert result["applied_retry_count"] == 1
    assert result["unchanged_segment_positions"] == [0, 2]
    assert canonical["segments"][0]["text"] == "保留前段"
    assert canonical["segments"][1]["text"] == "这里是恢复后的正确正文。"
    assert canonical["segments"][1]["raw_text"] == "请逐字转写整个字。"
    assert canonical["segments"][1]["start"] == 10
    assert canonical["segments"][1]["end"] == 13
    assert canonical["segments"][1]["asr_retry_transformations"][0]["consent_id"] == "consent-targeted-1"
    assert canonical["segments"][2]["text"] == "保留后段"
    assert (bundle / "asr-targeted-retry-merge-report.json").is_file()
    assert (bundle / "source-arbitrated-transcript.srt").is_file()


def test_targeted_retry_blocks_boundary_ambiguous_segment(tmp_path: Path) -> None:
    bundle, plan, execution = _fixture(tmp_path, ambiguous=True)

    result = merge_asr_targeted_retry_reports(bundle, plan, [execution], write=True)
    canonical = read_json(bundle / "source-arbitrated-transcript.json")

    assert result["status"] == "failed"
    assert result["applied_retry_count"] == 0
    assert "crosses target boundaries" in result["failures"][0]["error"]
    assert canonical["segments"][1]["text"] == "请逐字转写整个字。"
    assert not (bundle / "asr-targeted-retry-merge-report.json").exists()


def test_targeted_retry_rejects_authorization_hash_mismatch(tmp_path: Path) -> None:
    bundle, plan, execution = _fixture(tmp_path)
    payload = read_json(plan)
    payload["artifact_manifest"][0]["sha256"] = "0" * 64
    write_json(plan, payload)

    with pytest.raises(ValueError, match="hash mismatch"):
        merge_asr_targeted_retry_reports(bundle, plan, [execution], write=False)


def test_targeted_retry_can_match_zero_segment_identity(tmp_path: Path) -> None:
    bundle, plan, execution = _fixture(tmp_path)
    plan_payload = read_json(plan)
    plan_payload["artifact_manifest"][0]["source_window_seconds"] = {"start": 0.0, "end": 10.0}
    plan_payload["artifact_manifest"][0]["source_segment_ids"] = ["0"]
    write_json(plan, plan_payload)
    execution_payload = read_json(execution)
    execution_payload["model_result"]["runtime_result"]["raw_output"]["segments"] = [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "恢复后的第零段。",
            "avg_logprob": -0.1,
            "compression_ratio": 1.1,
            "no_speech_prob": 0.01,
        }
    ]
    write_json(execution, execution_payload)
    result = merge_asr_targeted_retry_reports(bundle, plan, [execution], write=False)
    assert result["status"] == "completed"
    assert result["applied"][0]["canonical_segment_position"] == 0


def test_targeted_retry_rejects_runtime_retry_drift(tmp_path: Path) -> None:
    bundle, plan, execution = _fixture(tmp_path)
    payload = read_json(execution)
    payload["model_result"]["runtime_result"]["request_options"]["max_retries"] = 1
    write_json(execution, payload)

    result = merge_asr_targeted_retry_reports(bundle, plan, [execution], write=False)

    assert result["status"] == "failed"
    assert result["applied_retry_count"] == 0
    assert "runtime retry limit differs" in result["failures"][0]["error"]


def test_targeted_retry_preserves_prior_recovery_history_and_cumulative_segments(
    tmp_path: Path,
) -> None:
    bundle, plan, execution = _fixture(tmp_path)
    canonical_path = bundle / "source-arbitrated-transcript.json"
    canonical = read_json(canonical_path)
    canonical["segments"][0]["asr_retry_transformations"] = [
        {"retry_id": "retry-prior", "source_segment_ids": ["0"]}
    ]
    canonical["targeted_asr_recovery"] = {
        "schema": "video_knowledge_pipeline.asr_targeted_retry_merge.v1",
        "status": "degraded",
        "authorization_plan": "prior-plan.json",
        "authorization_plan_sha256": "b" * 64,
        "expected_retry_count": 2,
        "applied_retry_count": 1,
        "failed_retry_count": 1,
        "applied": [{"source_segment_ids": ["0"]}],
        "failures": [{"source_segment_ids": ["1"]}],
    }
    write_json(canonical_path, canonical)

    result = merge_asr_targeted_retry_reports(bundle, plan, [execution], write=True)
    updated = read_json(canonical_path)
    recovery = updated["targeted_asr_recovery"]

    assert result["status"] == "completed"
    assert result["prior_recovery_count"] == 1
    assert result["cumulative_repaired_segment_count"] == 2
    assert result["cumulative_repaired_segment_positions"] == [0, 1]
    assert result["cumulative_transformation_count"] == 2
    assert recovery["history"][0]["authorization_plan"] == "prior-plan.json"
    assert recovery["history"][0]["status"] == "degraded"
    assert recovery["cumulative_repaired_segment_positions"] == [0, 1]
    assert recovery["cumulative_transformation_count"] == 2


def test_crop_retry_text_excludes_low_overlap_boundary_segment() -> None:
    text, evidence = _crop_retry_text(
        {
            "segments": [
                {"start": 2.0, "end": 6.0, "text": "target text"},
                {
                    "start": 29.96,
                    "end": 32.98,
                    "text": "adjacent text",
                },
            ]
        },
        snippet_start=122.32,
        target_start=123.82,
        target_end=153.8,
    )

    assert text == "target text"
    assert evidence["selected_segments"][0]["text"] == "target text"
    assert len(evidence["excluded_boundary_segments"]) == 1
    boundary = evidence["excluded_boundary_segments"][0]
    assert boundary["extends_right"] > 1.49
    assert boundary["overlap_ratio"] < 0.8
