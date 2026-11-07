from __future__ import annotations

import copy
from pathlib import Path

import pytest

from video_knowledge_pipeline import creator_report_review_notes as notes
from video_knowledge_pipeline.asr_diff_adjudication import (
    apply_asr_diff_adjudication, asr_diff_review_status, build_asr_diff_adjudication,
)
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from test_asr_fine_review_closure import prepare


def _review(root: Path, *, partial: bool = True):
    pack, decisions, payload = prepare(root)
    if partial:
        payload["rows"].pop()
        write_json(decisions, payload)
    result = apply_asr_diff_adjudication(root, decisions_json=decisions, review_only=True)
    return pack, result


def _snapshot(root: Path):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_pending_diff_is_located_without_marking_resolved_cues(tmp_path):
    pack, receipt = _review(tmp_path)
    cues = read_json(tmp_path / "canonical.json")["segments"]
    before = copy.deepcopy(cues)
    files = _snapshot(tmp_path)
    result = notes.collect_review_notes(tmp_path, cues)
    pending = {d["diff_id"] for d in pack["differences"]} - {
        row["diff_id"] for row in receipt["resolved"]}
    marked = [row for row in result["issues"] if row["code"] == "asr_difference_pending"]
    assert {row["diff_id"] for row in marked} == pending
    assert result["transcript"][0]["issues"] == []  # resolved numeric difference
    assert result["transcript"][2]["issues"][0]["diff_id"] in pending
    assert result["transcript"][2]["status"] == "review-required"
    assert result["transcript"][2]["text"] == "？"
    assert all(not Path(ref.split("#")[0]).is_absolute() for row in marked for ref in row["evidence_refs"])
    assert result == notes.collect_review_notes(tmp_path, cues)
    assert cues == before and _snapshot(tmp_path) == files


def test_completed_review_does_not_mark_resolved_differences(tmp_path):
    _review(tmp_path, partial=False)
    cues = read_json(tmp_path / "canonical.json")["segments"]
    result = notes.collect_review_notes(tmp_path, cues)
    assert result["issues"] == []
    assert all(row["issues"] == [] and row["status"] == "transcribed" for row in result["transcript"])


def test_pending_time_gap_stays_global_without_marking_neighbour_cues(tmp_path):
    pack, receipt = _review(tmp_path)
    # The projection cannot manufacture a cue when an alternate recognizer's
    # timed evidence lies in a gap in the supplied original cue sequence.
    cues = read_json(tmp_path / "canonical.json")["segments"][:-1]
    result = notes.collect_review_notes(tmp_path, cues)
    pending = [row for row in result["issues"] if row["code"] == "asr_difference_pending"]
    expected = {row["diff_id"] for row in pack["differences"]} - {
        row["diff_id"] for row in receipt["resolved"]}
    assert {row["diff_id"] for row in pending} == expected
    assert pending[0]["start"] == 13 and pending[0]["end"] == 13.1
    assert all(not cue["issues"] for cue in result["transcript"])


@pytest.mark.parametrize("artifact", ["asr-consensus-adjudication-pack.json", "reference.md", "asr-consensus.json"])
def test_changed_binding_is_global_and_cannot_locate_cues(tmp_path, artifact):
    _review(tmp_path)
    path = tmp_path / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    cues = read_json(tmp_path / "canonical.json")["segments"]
    result = notes.collect_review_notes(tmp_path, cues)
    assert result["issues"] == [{"code": "asr_review_stale", "message": "待核记录已过期",
                                 "evidence_refs": ["asr-consensus-adjudication-review.json"]}]
    assert all(row["issues"] == [] for row in result["transcript"])


def test_bound_external_original_media_preserves_native_pending_projection(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    _, decisions, payload = prepare(root)
    media = tmp_path / "original-source.wav"
    media.write_bytes(b"small synthetic media fixture; no decoding")
    manifest = read_json(root / "manifest.json")
    manifest["media_path"] = str(media)
    write_json(root / "manifest.json", manifest)
    comparison = read_json(root / "asr-consensus.json")
    comparison["media_path"] = str(media)
    write_json(root / "asr-consensus.json", comparison)
    pack = build_asr_diff_adjudication(root, cluster_token_gap=0)
    payload["rows"].pop()
    write_json(decisions, payload)
    receipt = apply_asr_diff_adjudication(root, decisions_json=decisions, review_only=True)
    current = asr_diff_review_status(root)
    assert current["status"] == "review-required" and current["resolved_count"] == 1
    cues = read_json(root / "canonical.json")["segments"]
    before = _snapshot(tmp_path)
    result = notes.collect_review_notes(root, cues)
    expected = {row["diff_id"] for row in pack["differences"]} - {
        row["diff_id"] for row in receipt["resolved"]}
    pending = [row for row in result["issues"] if row["code"] == "asr_difference_pending"]
    assert {row["diff_id"] for row in pending} == expected
    assert result["transcript"][2]["issues"][0]["diff_id"] in expected
    assert all(row["code"] != "asr_review_stale" for row in result["issues"])
    assert _snapshot(tmp_path) == before
    # External original media still follow the native byte/hash freshness gate.
    media.write_bytes(b"changed synthetic original")
    stale = notes.collect_review_notes(root, cues)
    assert [row["code"] for row in stale["issues"]] == ["asr_review_stale"]
    assert all(not cue["issues"] for cue in stale["transcript"])


@pytest.mark.parametrize("pointer", ["pack", "consensus", "relative_consensus"])
def test_outside_bundle_is_rejected_before_shared_reader(tmp_path, monkeypatch, pointer):
    root = tmp_path / "bundle"
    root.mkdir()
    _review(root)
    outside = tmp_path / "outside.json"
    receipt_path = root / "asr-consensus-adjudication-review.json"
    receipt = read_json(receipt_path)
    if pointer == "pack":
        outside.write_bytes((root / "asr-consensus-adjudication-pack.json").read_bytes())
        receipt["pack_binding"] = {"path": str(outside), "sha256": sha256_file(outside)}
    else:
        outside.write_bytes((root / "primary.json").read_bytes())
        comparison = read_json(root / "asr-consensus.json")
        comparison["primary_transcript"] = str(outside) if pointer == "consensus" else "primary.json"
        write_json(root / "asr-consensus.json", comparison)
        pack_path = root / "asr-consensus-adjudication-pack.json"
        pack = read_json(pack_path)
        pack["source_bindings"]["consensus"]["sha256"] = sha256_file(root / "asr-consensus.json")
        write_json(pack_path, pack)
        receipt["pack_binding"]["sha256"] = sha256_file(pack_path)
    write_json(receipt_path, receipt)
    monkeypatch.setattr(notes, "asr_diff_review_status", lambda root: pytest.fail("unsafe preflight reached reader"))
    result = notes.collect_review_notes(root, [{"start": 0, "end": 20, "text": "原文"}])
    assert result["issues"][0]["code"] == "asr_review_stale"
    assert result["transcript"][0]["issues"] == []
    assert str(tmp_path) not in result["issues"][0]["message"]


def test_empty_cue_is_retained_with_original_metadata_and_input_is_unchanged(tmp_path):
    cues = [{"start": 1, "end": 2, "text": "", "id": "empty", "metadata": {"keep": [1]}},
            {"start": 2, "end": 3, "text": "原文"}]
    before = copy.deepcopy(cues)
    result = notes.collect_review_notes(tmp_path, cues)
    assert len(result["transcript"]) == 2
    empty = result["transcript"][0]
    assert empty["text"] == "" and empty["id"] == "empty" and empty["metadata"] == {"keep": [1]}
    assert empty["issues"][0]["message"] == "此处原转写为空，待核"
    assert empty["status"] == "review-required"
    empty["metadata"]["keep"].append(2)
    assert cues == before
    assert not list(tmp_path.iterdir())
