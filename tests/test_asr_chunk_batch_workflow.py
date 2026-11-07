from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_vad_activity_audit import SCHEMA as AUDIT_SCHEMA
from video_knowledge_pipeline.asr_chunk_batch_workflow import (
    build_asr_chunk_batch_workflow,
)
from video_knowledge_pipeline.asr_vad_chunking import SCHEMA as CHUNK_MANIFEST_SCHEMA
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
)
from video_knowledge_pipeline.storage import write_json


PROVIDER = {
    "provider": "custom_openai_compatible_asr",
    "base_url": "https://asr.example/v1",
    "model": "whisper-test",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk_manifest(tmp_path: Path) -> tuple[Path, list[Path]]:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-media")
    vad = tmp_path / "vad.json"
    write_json(vad, {"segments": [{"start": 0, "end": 200}]})
    artifacts: list[Path] = []
    chunks: list[dict[str, object]] = []
    for position in range(1, 3):
        path = chunk_dir / f"asr-chunk-{position:04d}.mp3"
        path.write_bytes(f"audio chunk {position}".encode())
        artifacts.append(path)
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
    manifest_path = chunk_dir / "asr-vad-chunk-manifest.json"
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
    return manifest_path, artifacts


def _activity_audit(
    tmp_path: Path, manifest_path: Path, *, status: str = "passed"
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = tmp_path / "asr-vad-activity-audit.json"
    write_json(
        path,
        {
            "schema": AUDIT_SCHEMA,
            "status": status,
            "vad_coverage_verified": status == "passed",
            "candidate_gap_count": 0 if status == "passed" else 1,
            "source_media": {"sha256": manifest["source_sha256"]},
            "vad_sha256": manifest["vad_sha256"],
        },
    )
    return path


def _consents(
    tmp_path: Path,
    artifacts: list[Path],
    *,
    retries: int = 0,
) -> list[Path]:
    paths: list[Path] = []
    for position, artifact in enumerate(artifacts, start=1):
        path = tmp_path / "consents" / f"chunk-{position:04d}.json"
        result = create_model_connector_consent(
            tmp_path,
            task="cloud_asr",
            artifact_paths=[artifact],
            provider_config=PROVIDER,
            output_path=path,
            max_calls=1,
            max_estimated_cost_usd=0.01,
            max_cost_per_call_usd=0.01,
            max_retries_per_call=retries,
            confirm_data_export=True,
        )
        paths.append(Path(result["consent_path"]))
    return paths


def test_build_asr_chunk_batch_workflow_reuses_broker_contract(
    tmp_path: Path,
) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(bundle / "timeline.json", [])

    result = build_asr_chunk_batch_workflow(
        manifest,
        consents,
        bundle_dir=bundle,
    )

    assert result["status"] == "ready"
    assert result["ok"] is True
    assert result["chunk_count"] == 2
    assert result["submission"]["performed"] is False
    assert result["submission"]["tool"] == "submit_consented_model_workflow_tool"
    arguments = result["submission"]["arguments"]
    assert arguments["max_parallel_global"] == 4
    assert arguments["max_parallel_per_destination"] == 2
    assert [row["id"] for row in arguments["nodes"]] == [
        "asr-chunk-0001",
        "asr-chunk-0002",
    ]
    assert all(row["depends_on"] == [] for row in arguments["nodes"])
    assert all(row["destination"] == "https://asr.example" for row in result["nodes"])
    assert Path(result["output_path"]).is_file()
    assert result["operator_boundary"]["batch_submitted"] is False


def test_asr_chunk_workflow_binds_passed_activity_audit(tmp_path: Path) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts)
    audit = _activity_audit(tmp_path, manifest)

    result = build_asr_chunk_batch_workflow(
        manifest,
        consents,
        activity_audit_path=audit,
    )

    assert result["activity_audit"]["status"] == "passed"
    assert result["activity_audit_sha256"] == _sha256(audit)
    assert result["operator_boundary"]["vad_activity_candidates_resolved"] is True


def test_asr_chunk_workflow_rejects_unresolved_activity_candidates(
    tmp_path: Path,
) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts)
    audit = _activity_audit(tmp_path, manifest, status="review_required")

    with pytest.raises(ValueError, match="unresolved candidate gaps"):
        build_asr_chunk_batch_workflow(
            manifest,
            consents,
            activity_audit_path=audit,
        )


def test_asr_chunk_batch_workflow_rejects_reordered_consents(
    tmp_path: Path,
) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts)

    with pytest.raises(ValueError, match="does not match ASR chunk order"):
        build_asr_chunk_batch_workflow(manifest, list(reversed(consents)))


def test_asr_chunk_batch_workflow_rejects_changed_chunk(
    tmp_path: Path,
) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts)
    artifacts[1].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="byte count changed"):
        build_asr_chunk_batch_workflow(manifest, consents)


def test_asr_chunk_batch_workflow_requires_zero_retry(
    tmp_path: Path,
) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts, retries=1)

    with pytest.raises(ValueError, match="max_retries_per_call=0"):
        build_asr_chunk_batch_workflow(manifest, consents)


def test_asr_chunk_batch_workflow_requires_one_consent_per_chunk(
    tmp_path: Path,
) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts)

    with pytest.raises(ValueError, match="exactly one consent"):
        build_asr_chunk_batch_workflow(manifest, consents[:1])


def test_asr_chunk_batch_workflow_cli_is_compile_only(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    manifest, artifacts = _chunk_manifest(tmp_path)
    consents = _consents(tmp_path, artifacts)

    assert (
        cli_main(
            [
                "asr-chunk-batch-workflow",
                str(manifest),
                "--consent-path",
                str(consents[0]),
                "--consent-path",
                str(consents[1]),
                "--no-write",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ready"
    assert payload["write"] is False
    assert payload["submission"]["performed"] is False
    assert not Path(payload["output_path"]).exists()
