from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_semantic_correction import (
    build_transcript_semantic_correction_pack,
    transcript_semantic_correction_closure,
    transcript_semantic_correction_status,
)
from video_knowledge_pipeline.transcript_semantic_transaction import (
    apply_transcript_semantic_restore,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_semantic_bundle(tmp_path: Path, *, with_summary: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "semantic transaction fixture",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        root / "normalized-transcript.json",
        {"segments": [{"start": 0, "end": 4, "text": "这里演示 play right 自动化"}]},
    )
    write_json(
        root / "timeline.json",
        [{"index": 0, "start": 0, "end": 4, "visual_text": "Playwright 自动化"}],
    )
    if with_summary:
        (root / "exports" / "smart-summary.md").write_text(
            "生成方式：`codex_llm_rewrite_final`\n\n本节演示 play right 自动化。",
            encoding="utf-8",
        )
    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(row for row in pack["candidates"] if row["original_text"] == "play right")
    result_path = root / "transcript-semantic-correction-result.codex.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "source": "synthetic_test",
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": "proper_noun",
                    "original_text": "play right",
                    "corrected_text": "Playwright",
                    "confidence": 0.99,
                    "rationale": "Bound OCR evidence names the displayed tool.",
                    "evidence_ids": candidate["evidence_ids"],
                    "human_confirmed": False,
                    "needs_human_review": False,
                }
            ],
        },
    )
    return root, result_path


def test_closure_commits_one_hash_bound_transaction_and_restore(tmp_path: Path) -> None:
    root, result_path = _write_semantic_bundle(tmp_path)
    raw_before = (root / "normalized-transcript.json").read_bytes()
    timeline_before = (root / "timeline.json").read_bytes()

    closure = transcript_semantic_correction_closure(
        root,
        input_json=result_path,
        write=True,
    )

    receipt = closure["transaction_receipt"]
    assert receipt["status"] == "committed"
    assert receipt["artifact_count"] == 8
    assert Path(receipt["journal"]).is_file()
    assert Path(receipt["restore_plan"]).is_file()
    persisted = read_json(root / "transcript-semantic-correction-closure.json")
    assert persisted["pack_sha256"] == _sha(root / "transcript-semantic-correction-pack.json")
    assert persisted["validation_sha256"] == _sha(root / "transcript-semantic-correction-validation.json")
    assert persisted["input_identity"]["sha256"] == _sha(result_path)
    assert persisted["input_identity"]["path"] == str(result_path.resolve())

    status = transcript_semantic_correction_status(root, write=False)
    assert status["transaction_status"] == "committed"
    assert status["transaction_journal"] == receipt["journal"]
    assert status["transaction_restore_plan"] == receipt["restore_plan"]

    restored = apply_transcript_semantic_restore(
        root,
        receipt["restore_plan"],
        execute=True,
        confirm_transaction_id=receipt["transaction_id"],
    )
    assert restored["status"] == "restored"
    assert not (root / "source-arbitrated-transcript.json").exists()
    assert (root / "normalized-transcript.json").read_bytes() == raw_before
    assert (root / "timeline.json").read_bytes() == timeline_before


def test_closure_transaction_failure_rolls_back_all_targets_and_preserves_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, result_path = _write_semantic_bundle(tmp_path)
    old_canonical = b'{"segments":["operator baseline"]}'
    old_closure = b'{"status":"operator baseline"}'
    (root / "source-arbitrated-transcript.json").write_bytes(old_canonical)
    (root / "transcript-semantic-correction-closure.json").write_bytes(old_closure)
    raw_before = (root / "normalized-transcript.json").read_bytes()
    timeline_before = (root / "timeline.json").read_bytes()

    import video_knowledge_pipeline.transcript_semantic_transaction as transaction

    original_write = transaction.write_text_atomic

    def fail_on_direct_srt(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
        if path.parent == root and path.name == "source-arbitrated-transcript.srt":
            raise OSError("synthetic closure write failure")
        original_write(path, payload, encoding=encoding)

    monkeypatch.setattr(transaction, "write_text_atomic", fail_on_direct_srt)

    with pytest.raises(OSError, match="synthetic closure write failure"):
        transcript_semantic_correction_closure(root, input_json=result_path, write=True)

    assert (root / "source-arbitrated-transcript.json").read_bytes() == old_canonical
    assert (root / "transcript-semantic-correction-closure.json").read_bytes() == old_closure
    assert not (root / "source-arbitrated-transcript.srt").exists()
    assert (root / "normalized-transcript.json").read_bytes() == raw_before
    assert (root / "timeline.json").read_bytes() == timeline_before
    journals = list((root / "runs" / "transcript-semantic-correction").glob("*/transaction-journal.json"))
    assert len(journals) == 1
    assert read_json(journals[0])["status"] == "rolled_back"


def test_refresh_invalidates_old_summary_without_in_place_replacement(tmp_path: Path) -> None:
    root, result_path = _write_semantic_bundle(tmp_path, with_summary=True)
    summary_path = root / "exports" / "smart-summary.md"
    summary_before = summary_path.read_bytes()
    summary_before_sha = _sha(summary_path)
    raw_before = (root / "normalized-transcript.json").read_bytes()
    timeline_before = (root / "timeline.json").read_bytes()

    closure = transcript_semantic_correction_closure(
        root,
        input_json=result_path,
        refresh_exports=True,
        write=True,
    )

    assert closure["refresh_exports_status"] != "refreshed"
    assert closure["refresh_exports"]["requires_summary_regeneration"] is True
    assert closure["refresh_exports"]["semantic_acceptance_status"] != "accepted"
    assert closure["refresh_exports"]["readable_patch_status"] == "removed"
    summary_after = summary_path.read_text(encoding="utf-8")
    assert summary_path.read_bytes() != summary_before
    assert "needs_llm_summary" in summary_after
    assert "本节演示 Playwright 自动化" not in summary_after
    invalidation = read_json(root / "exports" / "smart-summary-invalidation.json")
    assert invalidation["status"] == "invalidated"
    archived = next(
        Path(row["archive_path"])
        for row in invalidation["invalidated_summaries"]
        if row["sha256"] == summary_before_sha
    )
    assert archived.read_bytes() == summary_before
    assert (root / "normalized-transcript.json").read_bytes() == raw_before
    assert (root / "timeline.json").read_bytes() == timeline_before
