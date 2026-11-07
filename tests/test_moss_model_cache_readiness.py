from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import asr_runner


MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


def _snapshot(root: Path, *, complete: bool) -> Path:
    repo = root / "models--OpenMOSS-Team--MOSS-Transcribe-Diarize"
    revision = repo / "snapshots" / "abc123"
    revision.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123", encoding="utf-8")
    if complete:
        for name in asr_runner._MOSS_REQUIRED_SNAPSHOT_FILES:
            (revision / name).write_text("{}", encoding="utf-8")
        (revision / "model.safetensors").write_bytes(b"actual-weight-content")
    return revision


def test_moss_model_readiness_rejects_empty_huggingface_repo_shell(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip(
        "huggingface_hub",
        reason="optional local model-cache scanner is not installed",
    )
    cache = tmp_path / "hub"
    revision = _snapshot(cache, complete=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("VKP_LOCAL_MODEL_ROOT", str(tmp_path / "local-models"))

    result = asr_runner._model_ready(
        preset="moss-transcribe-diarize",
        model=MODEL_ID,
    )

    assert result["ready"] is False
    assert result["status"] == "incomplete_cache"
    assert result["cache_matches"] == []
    assert result["incomplete_cache_matches"][0]["path"] == str(revision.resolve())
    assert result["incomplete_cache_matches"][0]["status"] == (
        "missing_required_snapshot_files"
    )
    assert result["network_access"] == "disabled"


def test_moss_model_readiness_reuses_huggingface_cache_scanner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip(
        "huggingface_hub",
        reason="optional local model-cache scanner is not installed",
    )
    cache = tmp_path / "hub"
    revision = _snapshot(cache, complete=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("VKP_LOCAL_MODEL_ROOT", str(tmp_path / "local-models"))

    result = asr_runner._model_ready(
        preset="moss-transcribe-diarize",
        model=MODEL_ID,
    )

    assert result["ready"] is True
    assert result["status"] == "ready"
    assert result["cache_matches"] == [str(revision.resolve())]
    assert result["cache_scanner"]["available"] is True
    assert result["cache_scanner"]["version"]
    assert result["network_access"] == "disabled"


def test_moss_model_readiness_checks_all_indexed_weight_shards(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "moss-model"
    snapshot.mkdir()
    for name in asr_runner._MOSS_REQUIRED_SNAPSHOT_FILES:
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"part-one")

    incomplete = asr_runner._model_ready(
        preset="moss-transcribe-diarize",
        model=str(snapshot),
    )
    assert incomplete["ready"] is False
    assert incomplete["status"] == "incomplete_cache"
    assert incomplete["incomplete_cache_matches"][0]["status"] == (
        "incomplete_weight_shards"
    )
    assert incomplete["incomplete_cache_matches"][0]["missing_weight_shards"] == [
        "model-00002-of-00002.safetensors"
    ]

    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"part-two")
    complete = asr_runner._model_ready(
        preset="moss-transcribe-diarize",
        model=str(snapshot),
    )
    assert complete["ready"] is True
    assert complete["cache_matches"] == [str(snapshot.resolve())]


def test_moss_model_readiness_rejects_git_lfs_pointer_weights(tmp_path: Path) -> None:
    snapshot = tmp_path / "moss-lfs-pointer"
    snapshot.mkdir()
    for name in asr_runner._MOSS_REQUIRED_SNAPSHOT_FILES:
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 123456789\n",
        encoding="utf-8",
    )

    result = asr_runner._model_ready(
        preset="moss-transcribe-diarize",
        model=str(snapshot),
    )

    assert result["ready"] is False
    assert result["incomplete_cache_matches"][0]["status"] == (
        "missing_model_weights"
    )
