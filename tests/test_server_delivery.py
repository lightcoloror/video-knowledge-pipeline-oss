"""Delivery-contract fixtures; these do not claim real PDF or video QA."""
from copy import deepcopy

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.server_delivery import map_server_delivery
from video_knowledge_pipeline.storage import write_json


def inputs(tmp_path, *, native_status="incomplete", blocker="screen_text", formal=False, constraints=None):
    native = {"schema": "lecture_acceptance_check.v1", "status": native_status,
              "summary": {"smart_summary_production_ready": formal},
              "blockers": [{"key": blocker, "severity": "blocking", "detail": "synthetic review gap"}] if blocker else []}
    native.update(constraints or {})
    rows = []
    for kind in ("html", "pdf", "md", "srt", "csv"):
        path = tmp_path / ("report." + kind)
        path.write_text("Synthetic contract payload. Not real PDF or media QA.", encoding="utf-8")
        qa = tmp_path / (kind + "-qa.json")
        write_json(qa, {"schema": "video_knowledge_pipeline.delivery_validation.v1", "status": "passed",
            "artifact_sha256": sha256_file(path), "native_acceptance_sha256": canonical_json_sha256(native),
            "evidence_scope": "synthetic_owner_QA_receipt_fixture",
            "checks": {k: True for k in ("content", "source_binding", "font", "images", "pagination", "timecodes",
                                           "shot_boundaries", "audiovisual_evidence", "links")}})
        rows.append({"path": path.name, "format": kind, "role": "shot_table" if kind == "csv" else
            "canonical_transcript" if kind == "srt" else "machine_draft", "sha256": sha256_file(path),
            "validation_receipt": qa.name, "validation_receipt_sha256": sha256_file(qa)})
    warnings = [{"key": blocker, "reason": "Synthetic localized uncertainty", "start": 2, "end": 3}] if blocker else []
    return native, rows, warnings


def mapped(tmp_path, native, rows, warnings, **kwargs):
    return map_server_delivery(native, execution_status=kwargs.pop("execution_status", "completed"),
        artifacts=rows, artifact_root=tmp_path, policy=kwargs.pop("policy", "reviewable_draft"), review_items=warnings, **kwargs)


def test_draft_is_readable_without_renaming_native_failure_or_charging(tmp_path):
    native, rows, warnings = inputs(tmp_path)
    before = deepcopy(native)
    result = mapped(tmp_path, native, rows, warnings)
    assert result["delivery_status"] == "reviewable_draft" and result["download_allowed"]
    assert result["workbench_status"] == "review_required" and result["candidate_only"]
    assert result["formal_completed"] is False and result["publication_allowed"] is False
    assert result["settlement_eligible"] is False and result["quality_gate_bypassed"] is False
    assert result["native_acceptance"] == before == native


def test_strict_policy_continues_blocking_the_same_known_quality_gap(tmp_path):
    native, rows, warnings = inputs(tmp_path)
    result = mapped(tmp_path, native, rows, warnings, policy="strict")
    assert not result["download_allowed"] and result["delivery_status"] == "blocked"
    assert "strict_quality_not_passed" in result["blockers"]


@pytest.mark.parametrize("status", ["failed", "running", "cancelled", "partial", "reconciliation_required"])
def test_execution_failure_is_not_a_reviewable_draft(tmp_path, status):
    native, rows, warnings = inputs(tmp_path)
    result = mapped(tmp_path, native, rows, warnings, execution_status=status)
    assert not result["download_allowed"] and "execution_not_completed" in result["blockers"]


@pytest.mark.parametrize("kind", ["html", "pdf", "md", "srt", "csv"])
def test_missing_or_empty_format_blocks_without_billing(tmp_path, kind):
    native, rows, warnings = inputs(tmp_path)
    (tmp_path / ("report." + kind)).write_bytes(b"")
    result = mapped(tmp_path, native, rows, warnings)
    assert not result["download_allowed"] and not result["settlement_eligible"]
    assert "five_format_delivery_incomplete" in result["blockers"]


@pytest.mark.parametrize("mutation", ["wrong_hash", "stale_QA", "missing_QA", "failed_QA", "fake_chapter_csv", "unknown_blocker", "no_review", "wrong_review"])
def test_evidence_and_quality_mapping_fail_closed(tmp_path, mutation):
    blocker = "unknown_new_blocker" if mutation == "unknown_blocker" else "screen_text"
    native, rows, warnings = inputs(tmp_path, blocker=blocker)
    if mutation == "wrong_hash":
        rows[0]["sha256"] = "0" * 64
    elif mutation == "stale_QA":
        native["summary"]["new_evidence"] = True
    elif mutation == "missing_QA":
        (tmp_path / rows[0]["validation_receipt"]).unlink()
    elif mutation == "failed_QA":
        qa = tmp_path / rows[0]["validation_receipt"]
        write_json(qa, {"status": "failed"})
        rows[0]["validation_receipt_sha256"] = sha256_file(qa)
    elif mutation == "fake_chapter_csv":
        rows[-1]["role"] = "chapter_table"
    elif mutation == "no_review":
        warnings = []
    elif mutation == "wrong_review":
        warnings[0]["key"] = "another_issue"
    result = mapped(tmp_path, native, rows, warnings)
    assert result["delivery_status"] == "blocked" and result["download_allowed"] is False


@pytest.mark.parametrize("check", ["font", "images", "pagination"])
def test_pdf_requires_content_QA_not_just_existence(tmp_path, check):
    native, rows, warnings = inputs(tmp_path)
    from video_knowledge_pipeline.storage import read_json
    row = rows[1]
    path = tmp_path / row["validation_receipt"]
    qa = read_json(path)
    qa["checks"][check] = False
    write_json(path, qa)
    row["validation_receipt_sha256"] = sha256_file(path)
    assert not mapped(tmp_path, native, rows, warnings)["download_allowed"]


def test_native_complete_needs_explicit_production_readiness_and_no_blockers(tmp_path):
    native, rows, warnings = inputs(tmp_path, native_status="complete", blocker=None, formal=True)
    result = mapped(tmp_path, native, rows, warnings, policy="strict")
    assert result["formal_completed"] and result["workbench_status"] == "delivered"
    assert result["settlement_eligible"] is False


def test_complete_process_and_files_do_not_substitute_for_missing_native_quality(tmp_path):
    native, rows, warnings = inputs(tmp_path, native_status="complete", blocker=None, formal=False)
    result = mapped(tmp_path, native, rows, warnings, policy="strict")
    assert not result["download_allowed"] and "formal_production_readiness_missing" in result["blockers"]


@pytest.mark.parametrize("field,value", [("candidate_only", True), ("publication_allowed", False), ("formal_completed", False)])
def test_explicit_native_restrictions_are_not_promoted_to_delivered(tmp_path, field, value):
    native, rows, warnings = inputs(tmp_path, native_status="complete", blocker=None, formal=True, constraints={field: value})
    strict = mapped(tmp_path, native, rows, warnings, policy="strict")
    assert not strict["download_allowed"] and not strict["formal_completed"]
    draft = mapped(tmp_path, native, rows, warnings)
    assert draft["workbench_status"] == "review_required" and draft["publication_allowed"] is False


@pytest.mark.parametrize("status", ["failed", "provider_blocked", "mysterious"])
def test_native_unavailable_and_unknown_statuses_stay_blocked(tmp_path, status):
    native, rows, warnings = inputs(tmp_path, native_status=status)
    assert not mapped(tmp_path, native, rows, warnings)["download_allowed"]


def test_no_speech_and_audio_only_cannot_fake_NA_in_v1_video_contract(tmp_path):
    native, rows, warnings = inputs(tmp_path)
    rows = [r for r in rows if r["format"] not in {"srt", "csv"}]
    result = mapped(tmp_path, native, rows, warnings)
    assert not result["download_allowed"] and "five_format_delivery_incomplete" in result["blockers"]
