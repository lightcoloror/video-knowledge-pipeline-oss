from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.model_output_contracts import (
    normalise_output_contract,
    validate_execution_report,
    validate_model_output,
)


def test_schema_alias_is_recorded_and_normalised_before_validation() -> None:
    result = validate_model_output(
        '{"people":[],"scene":"office","presentation_style":"slide",'
        '"non-text_visuals":[],"uncertainity":[]}',
        {
            "format": "json",
            "required_keys": {
                "people": "array",
                "scene": "string",
                "presentation_style": "string",
                "non_text_visuals": "array",
                "uncertainty": "array",
            },
            "aliases": {
                "non-text_visuals": "non_text_visuals",
                "uncertainity": "uncertainty",
            },
        },
    )

    assert result["transport_ok"] is True
    assert result["contract_ok"] is True
    assert result["quality_gate_passed"] is True
    assert result["applied_aliases"] == [
        {"alias": "non-text_visuals", "canonical": "non_text_visuals"},
        {"alias": "uncertainity", "canonical": "uncertainty"},
    ]


def test_summary_conflict_gate_requires_both_evidence_variants() -> None:
    contract = {
        "format": "json",
        "required_keys": {"title": "string", "summary": "string", "key_points": "array"},
        "required_all_terms": ["王飞", "王菲"],
    }

    failed = validate_model_output(
        {"title": "课程", "summary": "主讲王菲", "key_points": ["PPT 证据"]},
        contract,
    )
    passed = validate_model_output(
        {
            "title": "课程",
            "summary": "转写写王飞，PPT 写王菲，需复核",
            "key_points": ["证据冲突保留"],
        },
        contract,
    )

    assert failed["contract_ok"] is True
    assert failed["quality_gate_passed"] is False
    assert failed["quality_issues"] == [
        {"key": "required_term_missing", "detail": "王飞"}
    ]
    assert passed["quality_gate_passed"] is True


def test_correction_gate_rejects_overcorrection() -> None:
    contract = {
        "format": "json",
        "required_keys": {"decisions": "array"},
        "correction_policy": {
            "required_replacements": [{"source": "王飞", "replacement": "王菲"}],
            "allow_additional": False,
        },
    }
    result = validate_model_output(
        {
            "decisions": [
                {"source": "王飞", "replacement": "王菲"},
                {"source": "陌生客户", "replacement": "陌客"},
            ]
        },
        contract,
    )

    assert result["contract_ok"] is True
    assert result["quality_gate_passed"] is False
    assert result["quality_issues"] == [
        {"key": "unlisted_correction", "detail": "陌生客户 -> 陌客"}
    ]


def test_array_item_contract_rejects_missing_and_mistyped_decision_fields() -> None:
    contract = {
        "format": "json",
        "required_keys": {"decisions": "array"},
        "array_item_contracts": {
            "decisions": {
                "required_keys": {
                    "candidate_id": "string",
                    "confidence": "number",
                    "evidence_ids": "array",
                },
                "nonempty_keys": ["candidate_id", "evidence_ids"],
            }
        },
    }

    result = validate_model_output(
        {
            "decisions": [
                {
                    "candidate_id": "semcorr-1",
                    "confidence": "high",
                }
            ]
        },
        contract,
    )

    assert result["contract_ok"] is False
    assert result["quality_gate_passed"] is False
    assert {
        (row["key"], row["detail"]) for row in result["contract_issues"]
    } == {
        (
            "array_item_required_key_type_mismatch",
            "decisions[0].confidence: expected number",
        ),
        ("array_item_missing_required_key", "decisions[0].evidence_ids"),
    }


def test_temporal_each_group_requires_nonempty_state_change() -> None:
    contract = {
        "format": "json",
        "target": "temporal_each_group",
        "required_keys": {
            "event_sequence": "array",
            "state_changes": "array",
        },
        "nonempty_keys": ["state_changes"],
    }
    result = validate_model_output(
        {
            "groups": [
                {
                    "content": json.dumps(
                        {"event_sequence": ["frame 1", "frame 2"], "state_changes": []}
                    )
                }
            ]
        },
        contract,
    )

    assert result["contract_ok"] is True
    assert result["quality_gate_passed"] is False
    assert result["quality_issues"] == [
        {"key": "required_value_empty", "detail": "state_changes"}
    ]


def test_default_contract_and_missing_temporal_target_are_safe() -> None:
    assert validate_model_output("plain text")["quality_gate_passed"] is True
    normalised = normalise_output_contract(None)
    assert validate_model_output("plain text", normalised)["quality_gate_passed"] is True
    missing = validate_model_output(
        {"groups": []},
        {"target": "temporal_each_group", "format": "json"},
    )
    assert missing["contract_ok"] is False
    assert missing["contract_issues"] == [
        {"key": "contract_target_missing", "detail": "no temporal_each_group target found"}
    ]


def test_transport_contract_and_quality_statuses_are_distinct() -> None:
    contract = {"format": "json", "required_keys": {"title": "string"}}
    transport = validate_model_output({}, contract, transport_ok=False)
    contract_failure = validate_model_output("not json", contract)
    qualified = validate_model_output({"title": "ok"}, contract)

    assert transport["status"] == "transport_failed"
    assert contract_failure["status"] == "contract_failed"
    assert qualified["status"] == "qualified"


def test_saved_execution_validation_does_not_copy_content(tmp_path: Path) -> None:
    report = tmp_path / "connector-execution.json"
    report.write_text(
        json.dumps(
            {
                "ok": True,
                "model_result": {
                    "content": '{"title":"sensitive evidence"}',
                    "runtime_result": {"ok": True},
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_execution_report(
        report,
        {"format": "json", "required_keys": {"title": "string"}},
    )

    assert result["quality_gate_passed"] is True
    assert "sensitive evidence" not in json.dumps(result)
    assert result["content_persisted"] is False
