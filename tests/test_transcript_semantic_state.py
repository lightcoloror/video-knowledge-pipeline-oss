from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.artifact_validation import sha256_file
from video_knowledge_pipeline.transcript_semantic_state import (
    DISCOVERY_SCAN_RECEIPT_NAME,
    DISCOVERY_SCAN_RECEIPT_SCHEMA,
    semantic_candidate_resolution,
    semantic_discovery_scan_state,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _state_bundle(tmp_path: Path, *, retained_count: int = 3) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(root / "manifest.json", {"title": "semantic state fixture"})
    _write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"index": index, "start": index * 2, "end": index * 2 + 1, "text": f"cue {index}"}
                for index in range(retained_count)
            ]
        },
    )
    _write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 0,
            "candidates": [],
        },
    )
    coverage = {
        "retained_cue_count": retained_count,
        "selected_count": retained_count,
        "selected_source_indexes": list(range(retained_count)),
        "covered_retained_cue_count": retained_count,
        "coverage_ratio": 1.0,
    }
    discovery_pack: dict[str, object] = {
        "schema": "video_knowledge_pipeline.transcript_semantic_candidate_discovery.v1",
        "status": "discovery_prompt_ready" if retained_count else "no_segments_selected",
        "pack_json": str(root / "transcript-semantic-correction-pack.json"),
        "cursor": 0,
        "next_cursor": None,
        "complete": True,
        "transcript_path": str(root / "normalized-transcript.json"),
        "transcript_sha256": sha256_file(root / "normalized-transcript.json"),
        "selected_source_indexes": list(range(retained_count)),
        "coverage": coverage,
    }
    _write_json(root / "transcript-semantic-candidate-discovery-pack.json", discovery_pack)
    return root, discovery_pack


def _write_scan_receipt(root: Path, discovery_pack: dict[str, object]) -> Path:
    pack_path = root / "transcript-semantic-candidate-discovery-pack.json"
    receipt_path = root / DISCOVERY_SCAN_RECEIPT_NAME
    coverage = dict(discovery_pack["coverage"])
    _write_json(
        receipt_path,
        {
            "schema": DISCOVERY_SCAN_RECEIPT_SCHEMA,
            "status": "completed",
            "complete": True,
            "cursor": discovery_pack["cursor"],
            "next_cursor": None,
            "transcript_sha256": discovery_pack["transcript_sha256"],
            "source_pack_sha256": sha256_file(
                root / "transcript-semantic-correction-pack.json"
            ),
            "discovery_pack_sha256": sha256_file(pack_path),
            "coverage": coverage,
            "batches": [
                {
                    "cursor": discovery_pack["cursor"],
                    "next_cursor": None,
                    "complete": True,
                    "discovery_pack_sha256": sha256_file(pack_path),
                }
            ],
        },
    )
    return receipt_path


def test_candidate_resolution_reads_all_formal_adjudication_arrays(tmp_path: Path) -> None:
    root = tmp_path / "resolution"
    root.mkdir()
    candidate_ids = [f"semcorr-{index:04d}" for index in range(1, 8)]
    _write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "candidates": [{"candidate_id": candidate_id} for candidate_id in candidate_ids]
        },
    )
    _write_json(
        root / "transcript-semantic-correction-validation.json",
        {
            "decisions": [{"candidate_id": candidate_ids[0]}],
            "accepted_decisions": [{"candidate_id": candidate_ids[1]}],
            "arbitrated_no_change_decisions": [{"candidate_id": candidate_ids[2]}],
            "rejected_decisions": [{"candidate_id": candidate_ids[3]}],
            "review_required_items": [{"candidate_id": candidate_ids[4]}],
            "review_required_decisions": [{"candidate_id": candidate_ids[5]}],
            "review_rows": [{"candidate_id": candidate_ids[6]}],
        },
    )

    result = semantic_candidate_resolution(root)

    assert result["candidate_count"] == 7
    assert result["adjudicated_candidate_count"] == 7
    assert result["unadjudicated_candidate_count"] == 0
    assert result["unadjudicated_candidate_ids"] == []


def test_bundle_local_hash_bound_scan_receipt_closes_discovery(tmp_path: Path) -> None:
    root, discovery_pack = _state_bundle(tmp_path)
    receipt_path = _write_scan_receipt(root, discovery_pack)

    result = semantic_discovery_scan_state(root)

    assert result["status"] == "complete"
    assert result["complete"] is True
    assert result["completion_source"] == "bundle_scan_receipt"
    assert result["receipt_status"] == "valid"
    assert result["receipt_path"] == str(receipt_path)
    assert result["identity_valid"] is True
    assert result["hash_drift"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (lambda receipt: receipt.pop("discovery_pack_sha256"), "receipt_discovery_pack_sha256_missing"),
        (lambda receipt: receipt.__setitem__("transcript_sha256", "0" * 64), "receipt_transcript_sha256_mismatch"),
        (lambda receipt: receipt.__setitem__("source_pack_sha256", "1" * 64), "receipt_source_pack_sha256_mismatch"),
        (lambda receipt: receipt.__setitem__("next_cursor", 8), "receipt_next_cursor_not_terminal"),
        (
            lambda receipt: receipt["coverage"].__setitem__("covered_retained_cue_count", 1),
            "receipt_coverage_mismatch",
        ),
    ],
)
def test_invalid_scan_receipt_fails_closed(
    tmp_path: Path,
    mutation,
    expected_issue: str,
) -> None:
    root, discovery_pack = _state_bundle(tmp_path)
    receipt_path = _write_scan_receipt(root, discovery_pack)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(receipt)
    _write_json(receipt_path, receipt)

    result = semantic_discovery_scan_state(root)

    assert result["complete"] is False
    assert result["receipt_status"] == "invalid"
    assert expected_issue in result["identity_issues"]


def test_existing_autopilot_job_remains_a_valid_completion_source(tmp_path: Path) -> None:
    root, discovery_pack = _state_bundle(tmp_path)
    pack_sha256 = sha256_file(root / "transcript-semantic-candidate-discovery-pack.json")
    _write_json(
        root / "runs" / "autopilot-compatible" / "autopilot-job.json",
        {
            "semantic_candidate_discovery": {
                "transcript_sha256": discovery_pack["transcript_sha256"],
                "last_discovery_pack_sha256": pack_sha256,
                "next_cursor": None,
                "complete": True,
                "coverage": discovery_pack["coverage"],
            }
        },
    )

    result = semantic_discovery_scan_state(root)

    assert result["complete"] is True
    assert result["completion_source"] == "autopilot_job"
    assert result["page_imported"] is True
    assert result["receipt_status"] == "missing"


def test_autopilot_completion_with_mismatched_coverage_fails_closed(tmp_path: Path) -> None:
    root, discovery_pack = _state_bundle(tmp_path)
    pack_sha256 = sha256_file(root / "transcript-semantic-candidate-discovery-pack.json")
    _write_json(
        root / "runs" / "autopilot-mismatched-coverage" / "autopilot-job.json",
        {
            "semantic_candidate_discovery": {
                "transcript_sha256": discovery_pack["transcript_sha256"],
                "last_discovery_pack_sha256": pack_sha256,
                "next_cursor": None,
                "complete": True,
                "coverage": {
                    "retained_cue_count": 1,
                    "covered_retained_cue_count": 1,
                },
            }
        },
    )

    result = semantic_discovery_scan_state(root)

    assert result["complete"] is False
    assert result["completion_source"] == "autopilot_job"
    assert result["coverage"]["retained_cue_count"] == 1


def test_manifest_residue_without_pack_cannot_close_empty_scan(tmp_path: Path) -> None:
    root = tmp_path / "manifest-residue"
    root.mkdir()
    _write_json(
        root / "manifest.json",
        {
            "transcript_semantic_candidate_discovery_summary": {
                "status": "no_segments_selected",
                "segment_count": 0,
            },
            "transcript_semantic_candidate_discovery_pack_json": (
                "transcript-semantic-candidate-discovery-pack.json"
            ),
        },
    )

    result = semantic_discovery_scan_state(root)

    assert result["status"] == "manifest_residue"
    assert result["complete"] is False
    assert result["hash_drift"] is True
    assert result["identity_valid"] is False


def test_empty_scan_without_hash_bound_receipt_is_not_complete(tmp_path: Path) -> None:
    root, _discovery_pack = _state_bundle(tmp_path, retained_count=0)

    result = semantic_discovery_scan_state(root)

    assert result["status"] == "empty_completion_receipt_missing"
    assert result["complete"] is False
    assert result["hash_drift"] is True
    assert "empty_completion_receipt_missing" in result["identity_issues"]


def test_empty_scan_with_valid_hash_bound_receipt_can_complete(tmp_path: Path) -> None:
    root, discovery_pack = _state_bundle(tmp_path, retained_count=0)
    _write_scan_receipt(root, discovery_pack)

    result = semantic_discovery_scan_state(root)

    assert result["status"] == "complete"
    assert result["complete"] is True
    assert result["coverage"]["retained_cue_count"] == 0
