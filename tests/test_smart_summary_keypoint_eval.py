from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_codex import _human_key_point_recall
from video_knowledge_pipeline.smart_summary_keypoint_eval import (
    GOLDSET_SCHEMA,
    SCHEMA,
    evaluate_human_key_point_recall,
)


def test_structured_goldset_uses_explicit_aliases_and_preserves_lineage(
    tmp_path: Path,
) -> None:
    goldset = {
        "schema": GOLDSET_SCHEMA,
        "key_points": [
            {
                "id": "kp-child",
                "text": "孩子不带身故责任",
                "aliases": ["儿童方案不含身故责任"],
                "time_range": "00:48:00.000 - 00:49:00.000",
                "evidence_ids": ["seg-0048"],
                "source_kind": "human_confirmed",
            },
            {
                "id": "kp-action",
                "text": "顾问次日交付方案",
                "aliases": "经纪人明早提供对比方案",
                "evidence_ids": "seg-0059",
            },
        ],
    }
    source = tmp_path / "human-key-points.json"
    source.write_text(json.dumps(goldset, ensure_ascii=False), encoding="utf-8")

    result = evaluate_human_key_point_recall(
        goldset,
        "儿童方案不含身故责任。经纪人明早提供对比方案。",
        source_path=source,
    )

    assert result["schema"] == SCHEMA
    assert result["recall"] == 1.0
    assert result["source"]["path"] == str(source.resolve())
    assert len(result["source"]["sha256"]) == 64
    assert [row["method"] for row in result["decisions"]] == [
        "explicit_alias",
        "explicit_alias",
    ]
    assert result["decisions"][0]["time_range"] == "00:48:00.000 - 00:49:00.000"
    assert result["decisions"][0]["evidence_ids"] == ["seg-0048"]


def test_jieba_and_rapidfuzz_match_reordered_chinese_lexical_content() -> None:
    result = evaluate_human_key_point_recall(
        {"key_points": ["先确认需求再制定方案"]},
        "制定方案前，需要先确认家庭需求。",
    )

    decision = result["decisions"][0]
    assert decision["matched"] is True
    assert decision["method"] == "jieba_rapidfuzz_token_set"
    assert decision["jieba_token_set_ratio"] >= 90.0
    assert result["matcher"]["libraries"]["jieba"]["runtime_version"] == "0.42.1"
    assert result["matcher"]["libraries"]["rapidfuzz"]["runtime_version"] == "3.14.5"


def test_numeric_evidence_gate_rejects_lexically_similar_wrong_amount() -> None:
    result = evaluate_human_key_point_recall(
        {"key_points": ["家庭预算上限为2万元"]},
        "家庭预算上限为5万元。",
    )

    decision = result["decisions"][0]
    assert decision["matched"] is False
    assert decision["method"] == "numeric_evidence_missing"
    assert decision["missing_number_evidence"]
    assert result["recall"] == 0.0


def test_shared_generic_words_do_not_count_as_semantic_equivalence() -> None:
    result = evaluate_human_key_point_recall(
        {"key_points": ["投保人决定购买医疗险"]},
        "用户优先配置重疾险，医疗险以后补充。",
    )

    decision = result["decisions"][0]
    assert decision["matched"] is False
    assert decision["method"] == "not_matched"
    assert result["recall"] == 0.0


def test_malformed_goldset_fails_closed_instead_of_inflating_recall() -> None:
    result = evaluate_human_key_point_recall(
        {"key_points": [{"text": "确认家庭需求", "aliases": [42]}]},
        "先确认家庭需求。",
    )

    assert result["evaluated"] is False
    assert result["recall"] is None
    assert result["invalid_entries"][0]["index"] == 1


def test_quality_wrapper_reads_exports_goldset_and_returns_v2_decisions(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    path = exports / "human-key-points.json"
    path.write_text(
        json.dumps(
            {
                "schema": GOLDSET_SCHEMA,
                "key_points": [
                    {
                        "id": "kp-trust",
                        "text": "先建立信任再确认需求",
                        "evidence_ids": ["segment-0001"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = _human_key_point_recall(
        tmp_path,
        "沟通先建立信任再确认需求，最后制定方案。",
    )

    assert result["schema"] == SCHEMA
    assert result["evaluated"] is True
    assert result["recall"] == 1.0
    assert result["decisions"][0]["id"] == "kp-trust"
    assert result["source"]["sha256"]


def test_punctuation_only_key_point_fails_closed() -> None:
    result = evaluate_human_key_point_recall(
        {"key_points": ["……？！"]},
        "先确认家庭需求。",
    )

    assert result["evaluated"] is False
    assert result["invalid_entries"][0]["error"] == "key point text is empty"
