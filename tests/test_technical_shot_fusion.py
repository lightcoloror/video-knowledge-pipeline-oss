from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.technical_shot_fusion import (
    SCHEMA,
    fuse_technical_shot_boundaries,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _candidate(path: Path, backend: str, seconds: list[float], media_hash: str = "same") -> None:
    _write_json(
        path,
        {
            "schema": "video_knowledge_pipeline.technical_shot_boundaries.v1",
            "ok": True,
            "boundary_kind": "technical_shot",
            "backend": backend,
            "media": {"sha256": media_hash},
            "boundaries": [
                {"boundary_id": f"{backend}-{index}", "seconds": value}
                for index, value in enumerate(seconds, start=1)
            ],
        },
    )


def test_two_frame_tolerance_aggregates_but_does_not_auto_select(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_json(bundle / "manifest.json", {})
    first, second = tmp_path / "auto.json", tmp_path / "omni.json"
    _candidate(first, "autoshot", [1.0, 4.0])
    _candidate(second, "omnishotcut", [1.06, 6.0])

    result = fuse_technical_shot_boundaries(
        bundle,
        [first, second],
        frame_rate=25.0,
        tolerance_frames=2,
        write=False,
    )

    assert result["schema"] == SCHEMA
    assert result["candidate_count"] == 3
    assert result["agreement_count"] == 1
    assert result["review_count"] == 2
    assert result["candidates"][0]["backend_votes"] == ["autoshot", "omnishotcut"]
    assert all(row["automatically_selected"] is False for row in result["candidates"])
    assert result["operator_boundary"]["no_automatic_boundary_selection"] is True


def test_fusion_rejects_different_media_hashes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    first, second = tmp_path / "auto.json", tmp_path / "omni.json"
    _candidate(first, "autoshot", [1.0], media_hash="a")
    _candidate(second, "omnishotcut", [1.0], media_hash="b")

    with pytest.raises(ValueError, match="different media"):
        fuse_technical_shot_boundaries(
            bundle,
            [first, second],
            frame_rate=25.0,
            write=False,
        )


def test_overlapping_chunk_boundaries_cluster_with_two_frame_tolerance(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    first, second = tmp_path / "chunk-a.json", tmp_path / "chunk-b.json"
    _candidate(first, "autoshot-chunk-a", [304.96])
    _candidate(second, "autoshot-chunk-b", [305.02])

    result = fuse_technical_shot_boundaries(
        bundle,
        [first, second],
        frame_rate=25.0,
        tolerance_frames=2,
        write=False,
    )

    assert result["candidate_count"] == 1
    assert result["agreement_count"] == 1
    assert result["candidates"][0]["spread_seconds"] == pytest.approx(0.06)
    assert result["candidates"][0]["automatically_selected"] is False