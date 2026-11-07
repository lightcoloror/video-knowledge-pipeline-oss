from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.artifact_validation import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_semantic_correction import (
    build_transcript_semantic_candidate_discovery_pack,
    build_transcript_semantic_correction_pack,
    import_transcript_semantic_candidate_suggestions,
)
from video_knowledge_pipeline.transcript_semantic_state import (
    DISCOVERY_SCAN_RECEIPT_NAME,
    DISCOVERY_SCAN_RECEIPT_SCHEMA,
    semantic_discovery_scan_state,
)


def _bundle(tmp_path: Path, *, cue_count: int = 5) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(
        root / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "synthetic semantic discovery receipt",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(root / "timeline.json", [])
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {
                    "index": index,
                    "start": float(index * 3),
                    "end": float(index * 3 + 2),
                    "text": f"合成课程第{index}段内容",
                }
                for index in range(cue_count)
            ]
        },
    )
    build_transcript_semantic_correction_pack(root, limit=0, write=True)
    return root


def _prepare(root: Path, *, cursor: int = 0, limit: int = 2) -> dict[str, object]:
    pack = build_transcript_semantic_candidate_discovery_pack(
        root,
        cursor=cursor,
        limit=limit,
        write=True,
    )
    return pack


def _identity(root: Path, pack: dict[str, object]) -> dict[str, object]:
    return {
        "transcript_sha256": pack["transcript_sha256"],
        "discovery_pack_sha256": sha256_file(
            root / "transcript-semantic-candidate-discovery-pack.json"
        ),
        "cursor": pack["cursor"],
        "selected_source_indexes": pack["selected_source_indexes"],
    }


def _payload_path(
    root: Path,
    name: str,
    *,
    identity: dict[str, object] | None,
    suggestions: list[dict[str, object]] | None = None,
) -> Path:
    path = root / name
    payload: dict[str, object] = {
        "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1",
        "source": "synthetic_receipt_test",
        "suggestions": suggestions or [],
    }
    if identity is not None:
        payload["discovery_identity"] = identity
    write_json(path, payload)
    return path


def test_empty_completed_scan_preserves_admitted_pack_bytes_and_closure_identity(tmp_path):
    root=_bundle(tmp_path,cue_count=2)
    path=root/'transcript-semantic-correction-pack.json'
    pack=read_json(path)
    pack['updated_at']='2000-01-01T00:00:00'
    write_json(path,pack)
    before=path.read_bytes()
    discovery=_prepare(root,limit=10)
    payload=_payload_path(root,'empty-scan.json',identity=_identity(root,discovery))
    result=import_transcript_semantic_candidate_suggestions(root,input_json=payload,write=True)
    assert result['ok'] and result['progress_recorded'] and result['imported_candidate_count']==0
    assert path.read_bytes()==before
    assert semantic_discovery_scan_state(root)['complete']
    assert read_json(root/DISCOVERY_SCAN_RECEIPT_NAME)['source_pack_sha256']==sha256_file(path)


def _suggestion(index: int) -> dict[str, object]:
    return {
        "source_segment_index": index,
        "start": float(index * 3),
        "end": float(index * 3 + 2),
        "correction_type": "ordinary_word",
        "original_text": f"第{index}段",
        "candidate_text": f"第{index}节",
        "reason": "synthetic candidate only",
        "confidence": 0.6,
        "evidence_summary": "synthetic context",
        "needs_human_review": True,
    }


def test_legacy_suggestions_import_without_advancing_scan_progress(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path, cue_count=2)
    _prepare(root, limit=2)
    input_path = _payload_path(
        root,
        "legacy-suggestions.json",
        identity=None,
        suggestions=[_suggestion(0)],
    )

    result = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=input_path,
        write=True,
    )

    assert result["status"] == "imported"
    assert result["imported_candidate_count"] == 1
    assert result["progress_recorded"] is False
    assert result["progress_status"] == "identity_missing"
    assert not (root / DISCOVERY_SCAN_RECEIPT_NAME).exists()


def test_identity_bound_empty_terminal_page_writes_completion_receipt(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path, cue_count=2)
    page = _prepare(root, limit=2)
    generated_template = read_json(
        root / "transcript-semantic-candidate-discovery-template.json"
    )
    assert generated_template["discovery_identity"] == _identity(root, page)
    input_path = _payload_path(
        root,
        "empty-terminal.json",
        identity=_identity(root, page),
    )

    result = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=input_path,
        write=True,
    )
    receipt = read_json(root / DISCOVERY_SCAN_RECEIPT_NAME)
    scan = semantic_discovery_scan_state(root)

    assert result["status"] == "no_candidates_imported"
    assert result["progress_recorded"] is True
    assert result["progress_status"] == "completed"
    assert receipt["schema"] == DISCOVERY_SCAN_RECEIPT_SCHEMA
    assert receipt["complete"] is True
    assert receipt["next_cursor"] is None
    assert receipt["batches"][0]["suggestion_count"] == 0
    assert scan["complete"] is True
    assert scan["completion_source"] == "bundle_scan_receipt"


def test_identity_bound_multi_page_scan_is_sequential_and_complete(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path, cue_count=5)
    cursor = 0

    for expected_batch_count in (1, 2, 3):
        page = _prepare(root, cursor=cursor, limit=2)
        input_path = _payload_path(
            root,
            f"page-{expected_batch_count}.json",
            identity=_identity(root, page),
        )
        result = import_transcript_semantic_candidate_suggestions(
            root,
            input_json=input_path,
            write=True,
        )
        receipt = read_json(root / DISCOVERY_SCAN_RECEIPT_NAME)

        assert result["progress_recorded"] is True
        assert len(receipt["batches"]) == expected_batch_count
        if page["next_cursor"] is None:
            assert receipt["status"] == "completed"
            assert receipt["complete"] is True
        else:
            assert receipt["status"] == "in_progress"
            assert receipt["complete"] is False
            cursor = int(page["next_cursor"])

    assert [
        index
        for batch in receipt["batches"]
        for index in batch["selected_source_indexes"]
    ] == list(range(5))


def test_page_without_prior_receipt_is_rejected_without_pack_mutation(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path, cue_count=5)
    page = _prepare(root, cursor=2, limit=2)
    input_path = _payload_path(
        root,
        "out-of-order.json",
        identity=_identity(root, page),
        suggestions=[_suggestion(2)],
    )
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack_before = pack_path.read_bytes()

    result = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=input_path,
        write=True,
    )

    assert result["ok"] is False
    assert result["status"] == "progress_identity_rejected"
    assert "missing_prior_page_receipt" in result["progress_issues"]
    assert pack_path.read_bytes() == pack_before
    assert not (root / DISCOVERY_SCAN_RECEIPT_NAME).exists()


@pytest.mark.parametrize("failure_kind", ["pack_hash", "transcript_hash", "outside_index"])
def test_identity_drift_or_outside_index_fails_closed(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    root = _bundle(tmp_path, cue_count=4)
    page = _prepare(root, cursor=0, limit=2)
    identity = _identity(root, page)
    suggestions: list[dict[str, object]] = []
    if failure_kind == "pack_hash":
        identity["discovery_pack_sha256"] = "0" * 64
    elif failure_kind == "transcript_hash":
        identity["transcript_sha256"] = "1" * 64
    else:
        suggestions = [_suggestion(3)]
    input_path = _payload_path(
        root,
        f"invalid-{failure_kind}.json",
        identity=identity,
        suggestions=suggestions,
    )
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack_before = pack_path.read_bytes()

    result = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=input_path,
        write=True,
    )

    assert result["ok"] is False
    assert result["status"] == "progress_identity_rejected"
    assert result["progress_issues"]
    assert pack_path.read_bytes() == pack_before
    assert not (root / DISCOVERY_SCAN_RECEIPT_NAME).exists()


def test_same_page_is_idempotent_but_different_result_is_rejected(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path, cue_count=2)
    page = _prepare(root, limit=2)
    identity = _identity(root, page)
    first_path = _payload_path(root, "first.json", identity=identity)
    first = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=first_path,
        write=True,
    )
    receipt_path = root / DISCOVERY_SCAN_RECEIPT_NAME
    receipt_before = receipt_path.read_bytes()
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack_before = pack_path.read_bytes()

    repeated = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=first_path,
        write=True,
    )
    different_path = _payload_path(
        root,
        "different.json",
        identity=identity,
        suggestions=[_suggestion(0)],
    )
    rejected = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=different_path,
        write=True,
    )

    assert first["progress_recorded"] is True
    assert repeated["status"] == "already_imported"
    assert repeated["progress_status"] == "already_recorded"
    assert rejected["ok"] is False
    assert rejected["status"] == "progress_identity_rejected"
    assert "page_already_committed_with_different_result" in rejected["progress_issues"]
    assert receipt_path.read_bytes() == receipt_before
    assert pack_path.read_bytes() == pack_before


def test_transcript_changes_after_prepare_rejects_import(tmp_path: Path) -> None:
    root = _bundle(tmp_path, cue_count=2)
    page = _prepare(root, limit=2)
    input_path = _payload_path(
        root,
        "stale-transcript.json",
        identity=_identity(root, page),
    )
    transcript_path = root / "normalized-transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript["segments"][0]["text"] += "（已漂移）"
    write_json(transcript_path, transcript)
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack_before = pack_path.read_bytes()

    result = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=input_path,
        write=True,
    )

    assert result["ok"] is False
    assert "current_transcript_sha256_mismatch" in result["progress_issues"]
    assert pack_path.read_bytes() == pack_before
    assert not (root / DISCOVERY_SCAN_RECEIPT_NAME).exists()
