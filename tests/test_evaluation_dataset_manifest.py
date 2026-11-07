from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.evaluation_dataset_manifest import (
    build_evaluation_dataset_manifest,
)


def _write(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_dataset_manifest_reports_real_readiness_without_download_or_extract(
    tmp_path: Path,
) -> None:
    for name in ("ClipShots-a", "ClipShots-b", "ClipShots-c"):
        _write(tmp_path / "archives" / "ClipShots" / "ClipShots" / name)
    _write(tmp_path / "raw" / "ClipShots" / "annotations" / "test.json")
    _write(tmp_path / "archives" / "AutoShot" / "AutoShot" / "video_download.zip")
    _write(tmp_path / "raw" / "AutoShot" / "README.md")
    _write(
        tmp_path / "subsets" / "AutoShot-GT-present-167" / "pilot-32-manifest.json",
        b"{}",
    )
    _write(
        tmp_path
        / "subsets"
        / "AutoShot-GT-present-167"
        / "autoshot-gpu-inference-pilot-32.json",
        b"{}",
    )
    _write(
        tmp_path
        / "subsets"
        / "AISHELL1-ModelScope-subset"
        / "speech_asr_aishell_subset_testsets.csv",
    )

    result = build_evaluation_dataset_manifest(tmp_path)

    assert result["schema"] == "video_knowledge_pipeline.evaluation_dataset_manifest.v1"
    assert len(result["manifest_sha256"]) == 64
    datasets = {row["dataset_id"]: row for row in result["datasets"]}
    assert datasets["clipshots"]["status"] == "archive_complete_not_extracted"
    assert datasets["clipshots"]["issues"] == ["video_media_not_extracted"]
    assert datasets["autoshot"]["status"] == "benchmark_completed"
    assert datasets["aishell1_modelscope_subset"]["status"] == "metadata_only"
    assert datasets["aishell1_modelscope_subset"]["issues"] == ["audio_payload_missing"]
    assert all(
        len(row["sha256"]) == 64
        for row in datasets["clipshots"]["archive_parts"]
    )


def test_dataset_manifest_identity_is_stable_when_only_generated_time_changes(
    tmp_path: Path,
) -> None:
    for name in ("ClipShots-a", "ClipShots-b", "ClipShots-c"):
        _write(tmp_path / "archives" / "ClipShots" / "ClipShots" / name)

    first = build_evaluation_dataset_manifest(tmp_path, include_sha256=False)
    second = build_evaluation_dataset_manifest(tmp_path, include_sha256=False)

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert json.dumps(first["datasets"], sort_keys=True) == json.dumps(
        second["datasets"], sort_keys=True
    )
