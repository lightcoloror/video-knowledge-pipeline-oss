"""Synthetic, offline checks of the read-only reference-summary boundary."""
from pathlib import Path

import pytest

from video_knowledge_pipeline import smart_summary_handoff as handoff
from video_knowledge_pipeline.creator_report_reference_summary import load_reference_summary
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(root / "manifest.json", {"corrected_transcript_json": "corrected-transcript.json"})
    write_json(root / "corrected-transcript.json", [{"start": 0, "end": 2, "text": "合成输入待核。"}])
    write_json(root / "timeline.json", [{"start": 0, "end": 2, "transcript": "合成输入待核。"}])
    write_json(root / "transcript-quality-gate.json", {"status": "failed", "ok": False,
        "issues": [{"key": "synthetic_gap", "passed": False}]})
    return root


def _registered(root):
    source = root / "candidate.md"
    source.write_text("# 合成参考总结\n\n输入仍需核实。\n", encoding="utf-8")
    return handoff.register_review_candidate(root, source, gate={"blocking_reasons": ["synthetic_gap"]})


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_registered_current_candidate_is_reference_and_read_only(bundle):
    state = _registered(bundle)
    before = _files(bundle)
    result = load_reference_summary(bundle)
    assert result["status"] == "reference"
    assert result["text"].startswith(handoff.WATERMARK)
    assert "合成参考总结" in result["text"]
    assert result["issues"][0]["remaining_review_items"] == state["remaining_review_items"]
    assert result["sources"][0]["path"] == str((bundle / state["review_candidate_path"]).resolve())
    assert result["sources"][0]["sha256"] == state["review_candidate_sha256"]
    assert load_reference_summary(bundle) == result
    assert _files(bundle) == before
    assert read_json(bundle / "transcript-quality-gate.json")["status"] == "failed"
    assert read_json(bundle / handoff.STATE)["status"] == "review-required"
    assert not (bundle / "exports/smart-summary.codex.md").exists()


@pytest.mark.parametrize("changed", ["corrected-transcript.json", "timeline.json"])
def test_changed_bound_input_is_unavailable(bundle, changed):
    _registered(bundle)
    (bundle / changed).write_text("[]", encoding="utf-8")
    result = load_reference_summary(bundle)
    assert result["status"] == "unavailable" and not result["text"]
    assert result["issues"][0]["key"] == "reference_summary_binding_stale"


def test_new_preferred_input_does_not_reuse_old_binding(bundle):
    _registered(bundle)
    write_json(bundle / "human-corrected-transcript.json", [{"text": "new synthetic input"}])
    write_json(bundle / "manifest.json", {"human_corrected_transcript_json": "human-corrected-transcript.json",
                                          "corrected_transcript_json": "corrected-transcript.json"})
    result = load_reference_summary(bundle)
    assert result["status"] == "unavailable"
    assert result["issues"][0]["key"] == "reference_summary_binding_stale"


def test_unregistered_markdown_is_never_selected(bundle):
    path = bundle / "exports/smart-summary.llm.md"
    path.parent.mkdir()
    path.write_text("Arbitrary unregistered candidate", encoding="utf-8")
    result = load_reference_summary(bundle)
    assert result["status"] == "unavailable" and result["sources"] == []
    assert result["issues"][0]["key"] == "reference_summary_not_registered"


@pytest.mark.parametrize("field", ["raw_candidate_path", "review_candidate_path"])
def test_external_registered_path_is_rejected_without_reading(bundle, monkeypatch, field):
    state = _registered(bundle)
    outside = bundle.parent / "outside.md"
    outside.write_text("must not be read", encoding="utf-8")
    state[field] = str(outside)
    write_json(bundle / handoff.STATE, state)
    original = Path.read_bytes
    def safe_read(path):
        assert path.resolve() != outside.resolve(), "external candidate was read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", safe_read)
    result = load_reference_summary(bundle)
    assert result["status"] == "unavailable"
    assert "outside bundle" in result["issues"][0]["detail"]


def test_external_selected_transcript_is_rejected(bundle, monkeypatch):
    _registered(bundle)
    outside = bundle.parent / "outside.json"
    write_json(outside, [{"text": "must not be read"}])
    write_json(bundle / "manifest.json", {"source_arbitrated_transcript_json": str(outside)})
    original = Path.read_bytes
    def safe_read(path):
        assert path.resolve() != outside.resolve(), "external transcript was read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", safe_read)
    assert load_reference_summary(bundle)["status"] == "unavailable"


def test_tampered_candidate_is_unavailable(bundle):
    state = _registered(bundle)
    (bundle / state["raw_candidate_path"]).write_text("tampered", encoding="utf-8")
    result = load_reference_summary(bundle)
    assert result["status"] == "unavailable" and not result["text"]


def test_strict_summary_precedes_reference_without_mutating_strict_state(bundle, monkeypatch):
    _registered(bundle)
    strict_path = bundle / "exports/smart-summary.codex.md"
    strict_text = "# Strict synthetic summary\n"
    strict_path.write_text(strict_text, encoding="utf-8")
    before = _files(bundle)
    monkeypatch.setattr("video_knowledge_pipeline.creator_report_reference_summary.handoff_status",
                        lambda root: pytest.fail("reference lookup must not run"))
    result = load_reference_summary(bundle, strict_summary=strict_text, strict_summary_path=strict_path)
    assert result["status"] == "validated" and result["text"] == strict_text
    assert result["sources"][0]["role"] == "validated_smart_summary"
    assert result["issues"] == []
    assert _files(bundle) == before


def test_strict_source_mismatch_or_outside_is_not_replaced_with_reference(bundle):
    _registered(bundle)
    strict_path = bundle / "strict.md"
    strict_path.write_text("changed strict prose", encoding="utf-8")
    result = load_reference_summary(bundle, strict_summary="expected prose", strict_summary_path=strict_path)
    assert result["status"] == "unavailable"
    assert result["issues"][0]["key"] == "strict_summary_content_changed"
    outside = bundle.parent / "strict-outside.md"
    outside.write_text("expected prose", encoding="utf-8")
    assert load_reference_summary(bundle, strict_summary="expected prose", strict_summary_path=outside)["status"] == "unavailable"


def test_strict_path_alone_cannot_validate_arbitrary_markdown(bundle):
    path = bundle / "strict.md"
    path.write_text("unvalidated", encoding="utf-8")
    assert load_reference_summary(bundle, strict_summary_path=path)["status"] == "unavailable"


def test_input_pack_added_after_registration_invalidates_candidate(bundle):
    _registered(bundle)
    write_json(bundle / "exports/smart-summary-input-pack.json", {
        "transcript_source": str(bundle / "corrected-transcript.json"),
        "transcript_source_sha256": sha256_file(bundle / "corrected-transcript.json")})
    result = load_reference_summary(bundle)
    assert result["status"] == "unavailable"
    assert result["issues"][0]["key"] == "reference_summary_binding_stale"


def test_malformed_registration_is_unavailable(bundle):
    (bundle / "exports").mkdir()
    (bundle / handoff.STATE).write_text("not json", encoding="utf-8")
    result = load_reference_summary(bundle)
    assert result["status"] == "unavailable" and result["issues"]


@pytest.mark.parametrize("fresh_pack", [True, False])
def test_input_pack_binding_must_already_match_when_candidate_is_registered(bundle, fresh_pack):
    transcript = bundle / "corrected-transcript.json"
    write_json(bundle / "exports/smart-summary-input-pack.json", {
        "transcript_source": str(transcript),
        "transcript_source_sha256": sha256_file(transcript) if fresh_pack else "0" * 64})
    _registered(bundle)
    result = load_reference_summary(bundle)
    assert result["status"] == ("reference" if fresh_pack else "unavailable")
    if not fresh_pack:
        assert result["issues"][0]["detail"] == "summary_reference_input_pack_stale"
