from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_secondary_evidence import (
    close_secondary_asr_evidence,
)
from video_knowledge_pipeline.storage import read_json, write_json


def test_secondary_asr_evidence_builds_review_pack_without_changing_canonical(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, segments=_good_segments())
    canonical_before = _sha256(fixture["canonical"])

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    assert result["status"] == "needs_review"
    assert result["ok"] is True
    assert result["identity_validation"]["ok"] is True
    assert result["quality"]["accepted_segment_count"] == 2
    assert result["adjudication"]["patches_applied"] == 0
    assert result["canonical_integrity"]["unchanged"] is True
    assert _sha256(fixture["canonical"]) == canonical_before
    assert Path(result["artifacts"]["normalized_transcript"]).is_file()
    assert Path(result["artifacts"]["adjudication_pack"]).is_file()
    assert not (fixture["bundle"] / "asr-consensus-patched-transcript.json").exists()
    run = read_json(
        fixture["bundle"] / "runs" / "asr-secondary-evidence-closure" / "run.json"
    )
    assert run["status"] == "needs_review"
    assert run["operator_boundary"]["no_network_call"] is True


def test_secondary_asr_evidence_preserves_good_segments_when_one_is_blocked(
    tmp_path: Path,
) -> None:
    segments = _good_segments()
    segments[1]["compression_ratio"] = 8.0
    fixture = _fixture(tmp_path, segments=segments)
    execution = read_json(fixture["execution"])
    execution["status"] = "degraded"
    write_json(fixture["execution"], execution)
    canonical_before = _sha256(fixture["canonical"])

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    assert result["status"] == "degraded"
    assert result["ok"] is False
    assert result["quality"]["failed_segment_count"] == 1
    assert result["quality"]["accepted_segment_count"] == 1
    normalized = read_json(Path(result["artifacts"]["normalized_transcript"]))
    assert [row["segment_id"] for row in normalized["segments"]] == ["segment-000001"]
    assert _sha256(fixture["canonical"]) == canonical_before
    assert not (fixture["bundle"] / "asr-consensus-patched-transcript.json").exists()


def test_secondary_asr_evidence_blocks_route_identity_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, segments=_good_segments())
    execution = read_json(fixture["execution"])
    execution["route"]["route_revision"] = "tampered"
    write_json(fixture["execution"], execution)
    canonical_before = _sha256(fixture["canonical"])

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    assert result["status"] == "blocked"
    assert result["ok"] is False
    assert result["identity_validation"]["mismatch_count"] == 1
    assert result["identity_validation"]["mismatches"][0]["field"] == "route_revision"
    assert _sha256(fixture["canonical"]) == canonical_before
    assert "raw_asr_output" not in result["artifacts"]
    run = read_json(
        fixture["bundle"] / "runs" / "asr-secondary-evidence-closure" / "run.json"
    )
    assert [row["reason"] for row in run["failed_items"]] == ["identity_mismatch"]


@pytest.mark.parametrize(
    ("mismatch", "expected_field"),
    [
        ("provider", "provider"),
        ("model", "model"),
        ("artifact", "upload_manifest"),
    ],
)
def test_secondary_asr_evidence_blocks_provider_model_or_artifact_mismatch(
    tmp_path: Path,
    mismatch: str,
    expected_field: str,
) -> None:
    fixture = _fixture(tmp_path, segments=_good_segments())
    execution = read_json(fixture["execution"])
    if mismatch == "provider":
        execution["route"]["deployments"][0]["provider"] = "wrong_provider"
    elif mismatch == "model":
        execution["route"]["deployments"][0]["model"] = "wrong-model"
    else:
        execution["upload_manifest"]["files"][0]["sha256"] = "0" * 64
    write_json(fixture["execution"], execution)

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    fields = {row["field"] for row in result["identity_validation"]["mismatches"]}
    assert result["status"] == "blocked"
    assert expected_field in fields


def test_secondary_asr_evidence_fails_when_verbose_segments_are_missing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, segments=None)
    execution = read_json(fixture["execution"])
    execution["status"] = "failed"
    write_json(fixture["execution"], execution)
    canonical_before = _sha256(fixture["canonical"])

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    assert result["status"] == "failed"
    assert result["quality"]["status"] == "failed"
    assert result["quality"]["accepted_segment_count"] == 0
    assert _sha256(fixture["canonical"]) == canonical_before
    assert "normalized_transcript" not in result["artifacts"]


def test_secondary_asr_evidence_completes_when_clean_hypotheses_agree(
    tmp_path: Path,
) -> None:
    segments = _good_segments()
    segments[1]["text"] = "gamma zeta"
    fixture = _fixture(tmp_path, segments=segments)

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    assert result["status"] == "completed"
    assert result["ok"] is True
    assert result["consensus"]["conflict_count"] == 0
    assert result["adjudication"]["cluster_count"] == 0


def test_secondary_asr_evidence_no_write_is_preview_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, segments=_good_segments())
    canonical_before = _sha256(fixture["canonical"])

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
        write=False,
    )

    assert result["status"] == "preview"
    assert result["operator_boundary"]["no_network_call"] is True
    assert result["artifacts"] == {}
    assert _sha256(fixture["canonical"]) == canonical_before
    assert not (fixture["bundle"] / "asr-secondary-evidence").exists()
    assert not (fixture["bundle"] / "asr-consensus.json").exists()


def test_secondary_asr_evidence_preserves_untimed_text_as_degraded_candidates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, segments=None)
    execution = read_json(fixture["execution"])
    execution["status"] = "failed"
    execution["model_result"]["runtime_result"]["raw_output"]["text"] = (
        "alpha beta gamma delta"
    )
    write_json(fixture["execution"], execution)
    canonical_before = _sha256(fixture["canonical"])

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    assert result["status"] == "degraded"
    assert result["ok"] is False
    assert result["reason"] == (
        "untimed_secondary_text_preserved_as_inferred_candidates"
    )
    inference = result["quality"]["timing_inference"]
    assert inference["status"] == "candidate_alignment_available"
    assert inference["timing_inferred"] is True
    assert inference["provider_timestamps_present"] is False
    assert inference["segment_count"] == 2
    inferred = read_json(Path(result["artifacts"]["inferred_timing_transcript"]))
    assert [(row["start"], row["end"]) for row in inferred["segments"]] == [
        (0.0, 2.0),
        (2.0, 4.0),
    ]
    assert all(row["timing_inferred"] is True for row in inferred["segments"])
    assert inferred["operator_boundary"]["automatic_promotion_forbidden"] is True
    assert Path(result["artifacts"]["inferred_adjudication_pack"]).is_file()
    assert _sha256(fixture["canonical"]) == canonical_before
    assert not (fixture["bundle"] / "asr-consensus.json").exists()
    assert not (fixture["bundle"] / "asr-consensus-patched-transcript.json").exists()
    run = read_json(
        fixture["bundle"] / "runs" / "asr-secondary-evidence-closure" / "run.json"
    )
    assert run["status"] == "degraded"
    assert run["failed_items"][0]["reason"] == "quality_gate_failed"
    assert "candidate evidence only" in run["next_actions"][0]
    assert run["operator_boundary"]["secondary_never_auto_promoted"] is True


def test_secondary_asr_evidence_rejects_unrelated_untimed_text(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, segments=None)
    execution = read_json(fixture["execution"])
    execution["status"] = "failed"
    execution["model_result"]["runtime_result"]["raw_output"]["text"] = (
        "unrelated secondary candidate"
    )
    write_json(fixture["execution"], execution)

    result = close_secondary_asr_evidence(
        fixture["bundle"],
        connector_execution=fixture["execution"],
        prepared_suite=fixture["suite"],
    )

    assert result["status"] == "failed"
    assert result["quality"]["timing_inference"]["status"] == (
        "rejected_low_similarity"
    )
    assert "inferred_timing_transcript" not in result["artifacts"]


def _fixture(tmp_path: Path, *, segments: list[dict] | None) -> dict[str, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    canonical = bundle / "source-arbitrated-transcript.json"
    write_json(
        canonical,
        {
            "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
            "segments": [
                {
                    "id": "primary-000001",
                    "start": 0.0,
                    "end": 2.0,
                    "text": "alpha beta",
                },
                {
                    "id": "primary-000002",
                    "start": 2.0,
                    "end": 4.0,
                    "text": "gamma zeta",
                },
            ],
        },
    )
    write_json(
        bundle / "manifest.json", {"source_arbitrated_transcript_json": canonical.name}
    )

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"fixed-audio-fixture")
    artifact = {
        "path": str(audio.resolve()),
        "bytes": audio.stat().st_size,
        "sha256": _sha256(audio),
    }
    candidate_id = "secondary-asr--mistral-voxtral-mini"
    candidate = {
        "candidate_id": candidate_id,
        "connector_task": "cloud_asr",
        "model_type": "asr",
        "route_task": "asr",
        "virtual_model": "vkp-remote-asr-test",
        "provider": "mistral_asr",
        "model": "voxtral-mini-2602",
        "destination": "api.mistral.ai",
        "route_id": "route-mistral-asr",
        "route_revision": "revision-001",
        "artifacts": [artifact],
        "instructions": "Transcribe the supplied audio and return provider metadata.",
        "asr_prompt": "Context terms: alpha, beta, gamma, zeta.",
    }
    suite = tmp_path / "prepared-suite.json"
    write_json(
        suite,
        {
            "schema": "video_knowledge_pipeline.model_candidate_fixed_suite_prepared.v1",
            "candidates": [candidate],
        },
    )
    raw_output = {"text": ""}
    if segments is not None:
        raw_output["segments"] = segments
    execution = tmp_path / "connector-execution.json"
    write_json(
        execution,
        {
            "schema": "video_knowledge_pipeline.trusted_model_connector.v1",
            "ok": True,
            "status": "completed",
            "transport_ok": True,
            "contract_ok": True,
            "task": "cloud_asr",
            "model_type": "asr",
            "route": {
                "route_id": "route-mistral-asr",
                "route_revision": "revision-001",
                "virtual_model": "vkp-remote-asr-test",
                "deployments": [
                    {
                        "provider": "mistral_asr",
                        "model": "voxtral-mini-2602",
                        "base_url": "https://api.mistral.ai/v1",
                    }
                ],
            },
            "artifact_paths": [str(audio.resolve())],
            "upload_manifest": {"files": [artifact]},
            "model_result": {
                "runtime_result": {
                    "route_id": "route-mistral-asr",
                    "route_revision": "revision-001",
                    "task": "asr",
                    "virtual_model": "vkp-remote-asr-test",
                    "provider": "mistral_asr",
                    "deployment": {
                        "model": "voxtral-mini-2602",
                    },
                    "raw_output": raw_output,
                }
            },
        },
    )
    return {
        "bundle": bundle,
        "canonical": canonical,
        "suite": suite,
        "execution": execution,
    }


def _good_segments() -> list[dict]:
    return [
        {
            "id": "segment-000001",
            "start": 0.0,
            "end": 2.0,
            "text": "alpha beta",
            "avg_logprob": -0.2,
            "compression_ratio": 1.1,
            "no_speech_prob": 0.01,
        },
        {
            "id": "segment-000002",
            "start": 2.0,
            "end": 4.0,
            "text": "gamma delta",
            "avg_logprob": -0.25,
            "compression_ratio": 1.05,
            "no_speech_prob": 0.01,
        },
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
