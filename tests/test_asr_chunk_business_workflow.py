from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_chunk_batch_workflow import (
    build_asr_chunk_business_workflow,
)
from video_knowledge_pipeline.asr_vad_chunking import SCHEMA as CHUNK_MANIFEST_SCHEMA
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.model_business_authorization import (
    create_model_business_authorization,
    validate_model_business_authorization,
)
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.trusted_model_connector_policy import (
    ALLOWED_DESTINATIONS_ENV,
    ALLOWED_ROOTS_ENV,
    TrustedModelConnectorPolicy,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    max_calls: int = 2,
) -> tuple[Path, Path, Path, Path, TrustedModelConnectorPolicy]:
    bundle = tmp_path / "bundle"
    chunks_dir = bundle / "asr-chunks"
    chunks_dir.mkdir(parents=True)
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", [])
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-media")
    vad = tmp_path / "vad.json"
    write_json(vad, {"segments": [{"start": 0, "end": 200}]})

    chunks: list[dict[str, object]] = []
    for position in range(1, 3):
        path = chunks_dir / f"asr-chunk-{position:04d}.mp3"
        path.write_bytes(f"audio chunk {position}".encode())
        start = float((position - 1) * 100)
        chunks.append(
            {
                "chunk_id": f"asr-chunk-{position:04d}",
                "position": position,
                "status": "completed",
                "core_start": start,
                "core_end": start + 98,
                "artifact_start": max(0.0, start - 1.5),
                "artifact_end": start + 99.5,
                "output_path": str(path.resolve()),
                "output_bytes": path.stat().st_size,
                "output_sha256": _sha256(path),
            }
        )

    manifest_path = chunks_dir / "asr-vad-chunk-manifest.json"
    write_json(
        manifest_path,
        {
            "schema": CHUNK_MANIFEST_SCHEMA,
            "status": "completed",
            "ok": True,
            "source_path": str(source.resolve()),
            "source_bytes": source.stat().st_size,
            "source_sha256": _sha256(source),
            "vad_json": str(vad.resolve()),
            "vad_sha256": _sha256(vad),
            "chunk_count": 2,
            "completed_chunk_count": 2,
            "failed_chunk_count": 0,
            "chunks": chunks,
        },
    )

    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"asr.example"})
    )
    authorization = tmp_path / "business-authorization.json"
    create_model_business_authorization(
        tmp_path,
        bundle_dir=bundle,
        source_paths=[source],
        stages=[
            {
                "id": "cloud-asr-chunks",
                "task": "cloud_asr",
                "route_snapshot": {
                    "route_id": "cloud-asr-route",
                    "route_revision": "c" * 64,
                    "virtual_model": "vkp-remote-asr-fixture",
                    "execution_location": "remote",
                    "deployments": [
                        {
                            "provider": "custom_openai_compatible_asr",
                            "model": "whisper-test",
                            "base_url": "https://asr.example/v1",
                            "interface": "openai_compatible_asr",
                        }
                    ],
                },
                "allowed_producers": ["asr_vad_chunking"],
                "max_calls": max_calls,
                "max_estimated_cost_usd": 0.01 * max_calls,
                "max_cost_per_call_usd": 0.01,
                "max_retries_per_call": 0,
                "max_artifacts": 2,
                "max_total_bytes": 4096,
                "max_artifacts_per_child": 1,
                "max_bytes_per_child": 2048,
            }
        ],
        purpose="one confirmed chunked ASR workflow",
        max_calls=max_calls,
        max_estimated_cost_usd=0.01 * max_calls,
        confirm_data_export=True,
        output_path=authorization,
        policy=policy,
    )
    return manifest_path, authorization, bundle, source, policy


def test_business_authorization_mints_all_consents_and_compiles_workflow(
    tmp_path: Path,
) -> None:
    manifest, authorization, bundle, source, policy = _fixture(tmp_path)

    result = build_asr_chunk_business_workflow(
        manifest,
        authorization,
        stage_id="cloud-asr-chunks",
        producer="asr_vad_chunking",
        lineage_input_paths=[source],
        policy=policy,
    )

    assert result["status"] == "ready"
    assert result["new_user_confirmation_required"] is False
    assert len(result["child_consents"]) == 2
    assert all(Path(row["consent_path"]).is_file() for row in result["child_consents"])
    assert result["workflow"]["status"] == "ready"
    assert result["workflow"]["bundle_dir"] == str(bundle.resolve())
    assert validate_model_business_authorization(
        authorization, policy=policy
    )["remaining_calls"] == 0

    repeated = build_asr_chunk_business_workflow(
        manifest,
        authorization,
        stage_id="cloud-asr-chunks",
        producer="asr_vad_chunking",
        lineage_input_paths=[source],
        policy=policy,
    )
    assert all(
        row["status"] == "existing_child_consent"
        for row in repeated["child_consents"]
    )
    assert validate_model_business_authorization(
        authorization, policy=policy
    )["remaining_calls"] == 0


def test_business_workflow_preview_writes_no_child_or_workflow(
    tmp_path: Path,
) -> None:
    manifest, authorization, _bundle, source, policy = _fixture(tmp_path)

    result = build_asr_chunk_business_workflow(
        manifest,
        authorization,
        stage_id="cloud-asr-chunks",
        producer="asr_vad_chunking",
        lineage_input_paths=[source],
        policy=policy,
        write=False,
    )

    assert result["status"] == "preview"
    assert result["workflow"] == {}
    assert all(not Path(row["consent_path"]).exists() for row in result["child_consents"])
    assert not manifest.with_name("asr-chunk-batch-workflow.json").exists()
    assert validate_model_business_authorization(
        authorization, policy=policy
    )["remaining_calls"] == 2


def test_business_workflow_rejects_bundle_override(
    tmp_path: Path,
) -> None:
    manifest, authorization, _bundle, source, policy = _fixture(tmp_path)
    other_bundle = tmp_path / "other-bundle"
    other_bundle.mkdir()
    write_json(other_bundle / "manifest.json", {})
    write_json(other_bundle / "timeline.json", [])

    try:
        build_asr_chunk_business_workflow(
            manifest,
            authorization,
            stage_id="cloud-asr-chunks",
            producer="asr_vad_chunking",
            lineage_input_paths=[source],
            bundle_dir=other_bundle,
            policy=policy,
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("bundle override must be rejected")

def test_business_workflow_cli_preview_reuses_parent_without_new_confirmation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    manifest, authorization, _bundle, source, _policy = _fixture(tmp_path)
    monkeypatch.setenv(ALLOWED_ROOTS_ENV, str(tmp_path))
    monkeypatch.setenv(ALLOWED_DESTINATIONS_ENV, "asr.example")

    assert (
        cli_main(
            [
                "asr-chunk-business-workflow",
                str(manifest),
                str(authorization),
                "--stage-id",
                "cloud-asr-chunks",
                "--producer",
                "asr_vad_chunking",
                "--lineage-input",
                str(source),
                "--no-write",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "preview"
    assert payload["new_user_confirmation_required"] is False
    assert payload["provider_call_performed"] is False

def test_batch_preflight_blocks_before_any_child_write_when_budget_is_too_small(
    tmp_path: Path,
) -> None:
    manifest, authorization, _bundle, source, policy = _fixture(
        tmp_path, max_calls=1
    )

    with pytest.raises(ValueError, match="call limit"):
        build_asr_chunk_business_workflow(
            manifest,
            authorization,
            stage_id="cloud-asr-chunks",
            producer="asr_vad_chunking",
            lineage_input_paths=[source],
            policy=policy,
        )

    parent = read_json(authorization)
    assert parent["admissions"] == []
    assert parent["usage"]["calls_authorized"] == 0
    assert not (authorization.parent / "business-child-consents").exists()
