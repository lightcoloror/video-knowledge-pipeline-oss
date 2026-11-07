from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_semantic_correction import (
    _validate_decision,
    build_transcript_semantic_candidate_discovery_pack,
    build_transcript_semantic_correction_pack,
    import_transcript_semantic_candidate_suggestions,
)


NEW_HIGH_RISK_CASES = (
    ("negation", "不能投保", "可以投保"),
    ("condition", "如果满18岁", "只要满18岁"),
    ("exception", "这是除外责任", "这是包含责任"),
    ("person_name", "张三老师", "李四老师"),
    ("organization", "中医人寿", "中意人寿"),
    ("date_time", "日期是2025年5月", "日期是2026年5月"),
    ("money", "保费是100元", "保费是200元"),
    ("percentage", "比例是10%", "比例是20%"),
)


def _candidate(
    candidate_id: str,
    kind: str,
    original: str,
    corrected: str,
    *,
    evidence: list[dict],
) -> dict:
    return {
        "candidate_id": candidate_id,
        "segment_index": 0,
        "start": 0.0,
        "end": 2.0,
        "time_range": "00:00:00.000 - 00:00:02.000",
        "correction_type": kind,
        "risk_level": "high",
        "original_text": original,
        "candidate_text": corrected,
        "suggested_text": corrected,
        "context_text": original,
        "has_conflict": True,
        "needs_human_review": True,
        "evidence": evidence,
        "evidence_ids": [row["evidence_id"] for row in evidence],
        "evidence_source_types": sorted(
            {row["source_type"] for row in evidence}
        ),
    }


def _decision(candidate: dict, *, human_confirmed: bool = False) -> dict:
    return {
        "candidate_id": candidate["candidate_id"],
        "action": "replace",
        "correction_type": candidate["correction_type"],
        "original_text": candidate["original_text"],
        "corrected_text": candidate["candidate_text"],
        "confidence": 0.99,
        "rationale": "direct evidence supports the target span",
        "evidence_ids": candidate["evidence_ids"],
        "human_confirmed": human_confirmed,
        "needs_human_review": False,
    }


def test_new_semantic_anchor_types_fail_closed_with_only_one_direct_source() -> None:
    candidates = []
    decisions = []
    for index, (kind, original, corrected) in enumerate(
        NEW_HIGH_RISK_CASES,
        start=1,
    ):
        candidate = _candidate(
            f"risk-{index}",
            kind,
            original,
            corrected,
            evidence=[
                {
                    "evidence_id": f"ocr-{index}",
                    "source_type": "ocr",
                    "text": corrected,
                }
            ],
        )
        candidates.append(candidate)
        decisions.append(_decision(candidate))
    by_id = {row["candidate_id"]: row for row in candidates}
    rejected_by_type = {}
    for index, decision in enumerate(decisions):
        row = _validate_decision(
            decision,
            by_id,
            idx=index,
            min_confidence=0.88,
            strict_model_contract=True,
        )
        assert row["accepted"] is False
        rejected_by_type[row["correction_type"]] = row
    for kind, _original, _corrected in NEW_HIGH_RISK_CASES:
        row = rejected_by_type[kind]
        assert row["high_risk_semantic_anchor_change"] is True
        assert kind in row["semantic_anchor_change_types"]
        assert row["original_semantic_anchors"][kind]
        assert kind in row["corrected_semantic_anchors"]
        assert row["semantic_anchor_support_source_types"] == ["ocr"]
        assert (
            "semantic_anchor_change_requires_human_or_two_independent_direct_evidence"
            in row["reject_reasons"]
        )


def test_semantic_anchor_change_accepts_declared_distinct_strong_sources_or_human() -> None:
    candidate = _candidate(
        "organization-1",
        "organization",
        "中医人寿",
        "中意人寿",
        evidence=[
            {
                "evidence_id": "ocr-1",
                "source_type": "ocr",
                "text": "中意人寿",
                "evidence_level": "confirmed",
                "original_source_sha256": "a" * 64,
            },
            {
                "evidence_id": "subtitle-1",
                "source_type": "platform_subtitle",
                "text": "中意人寿",
                "original_source_sha256": "b" * 64,
            },
        ],
    )
    row = _validate_decision(
        _decision(candidate),
        {candidate["candidate_id"]: candidate},
        idx=0,
        min_confidence=0.88,
        strict_model_contract=True,
    )

    assert row["accepted"] is True
    assert row["semantic_anchor_support_source_types"] == [
        "ocr",
        "platform_subtitle",
    ]

    one_source = dict(candidate)
    one_source["evidence"] = candidate["evidence"][:1]
    one_source["evidence_ids"] = ["ocr-1"]
    one_source["evidence_source_types"] = ["ocr"]
    human = _validate_decision(
        _decision(one_source, human_confirmed=True),
        {one_source["candidate_id"]: one_source},
        idx=0,
        min_confidence=0.88,
        strict_model_contract=True,
    )
    assert human["accepted"] is True


def _write_bundle(tmp_path: Path, segments: list[dict]) -> Path:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "semantic safety fixture",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": segments})
    write_json(root / "timeline.json", [])
    return root


def test_candidate_has_read_only_bounded_context_speaker_and_chapter(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        [
            {"start": 0, "end": 2, "text": "前文第一条", "speaker": "spk-a"},
            {"start": 3, "end": 5, "text": "前文第二条", "speaker": "spk-a"},
            {"start": 10, "end": 12, "text": "这里有8个重点", "speaker": "spk-b"},
            {"start": 18, "end": 20, "text": "后文第一条", "speaker": "spk-b"},
            {"start": 40, "end": 42, "text": "超出窗口的后文", "speaker": "spk-c"},
        ],
    )
    write_json(
        root / "exports" / "smart-summary-chapters.json",
        {
            "chapters": [
                {
                    "index": 2,
                    "title": "保障方案",
                    "start": 6,
                    "end": 30,
                    "start_time": "00:00:06.000",
                    "end_time": "00:00:30.000",
                }
            ]
        },
    )

    pack = build_transcript_semantic_correction_pack(root, write=False)
    candidate = next(
        row
        for row in pack["candidates"]
        if row.get("segment_index") == 2
        and row.get("correction_type") == "number"
    )

    context = candidate["context_window"]
    assert [row["source_segment_index"] for row in context["previous"]] == [0, 1]
    assert [row["source_segment_index"] for row in context["next"]] == [3]
    assert context["duration_seconds"] <= 30
    assert candidate["speaker"] == "spk-b"
    assert candidate["chapter_index"] == 2
    assert candidate["chapter_title"] == "保障方案"
    assert candidate["edit_contract"] == {
        "target_segment_index": 2,
        "target_original_text": candidate["original_text"],
        "only_target_span_editable": True,
        "context_read_only": True,
    }


def test_generic_chinese_quantifiers_need_cross_source_difference(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        [
            {"start": 0, "end": 2, "text": "这是一个方案"},
            {"start": 3, "end": 5, "text": "我们只做一次演示"},
            {"start": 6, "end": 8, "text": "需要一天完成"},
            {
                "start": 9,
                "end": 12,
                "text": "保费100元，比例10%，日期2026年5月1日",
            },
        ],
    )

    pack = build_transcript_semantic_correction_pack(root, write=False)
    generic = {
        row["original_text"]
        for row in pack["candidates"]
        if row.get("reason") == "contains_number_or_amount"
    }
    assert not {"一个", "一次", "一天"} & generic
    assert {"100元", "10%"} <= generic
    assert any("2026年" in value for value in generic)

    write_json(
        root / "timeline.json",
        [{"index": 9, "start": 0, "end": 2, "visual_text": "这是两个方案"}],
    )
    conflict_pack = build_transcript_semantic_correction_pack(root, write=False)
    assert any(
        row.get("original_text") == "一个"
        and row.get("reason") == "contains_number_or_amount"
        for row in conflict_pack["candidates"]
    )


def test_candidate_discovery_sequentially_covers_every_retained_cue(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        [
            {"start": index * 3, "end": index * 3 + 2, "text": f"普通课程内容第{index}段"}
            for index in range(5)
        ],
    )
    transcript_path = root / "normalized-transcript.json"

    first = build_transcript_semantic_candidate_discovery_pack(
        root,
        limit=2,
        cursor=0,
        write=False,
    )
    second = build_transcript_semantic_candidate_discovery_pack(
        root,
        limit=2,
        cursor=first["next_cursor"],
        write=False,
    )
    third = build_transcript_semantic_candidate_discovery_pack(
        root,
        limit=2,
        cursor=second["next_cursor"],
        write=False,
    )

    assert first["transcript_path"] == str(transcript_path.resolve())
    assert first["transcript_sha256"] == sha256_file(transcript_path)
    assert first["selected_source_indexes"] == [0, 1]
    assert first["coverage"]["retained_cue_count"] == 5
    assert first["coverage"]["selected_count"] == 2
    assert first["complete"] is False
    assert first["next_cursor"] == 2
    assert second["selected_source_indexes"] == [2, 3]
    assert second["next_cursor"] == 4
    assert third["selected_source_indexes"] == [4]
    assert third["next_cursor"] is None
    assert third["complete"] is True
    assert {
        row["source_segment_index"]
        for batch in (first, second, third)
        for row in batch["segments"]
    } == set(range(5))
    assert all(
        "bounded_full_transcript_scan" in row["reasons"]
        for batch in (first, second, third)
        for row in batch["segments"]
    )
    assert not (root / "source-arbitrated-transcript.json").exists()


def test_bounded_discovery_contract_covers_homophones_and_collocations(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        [
            {"start": 0, "end": 4, "text": "客户依法享有保险权力。"},
            {"start": 5, "end": 9, "text": "这样可以提高保障的能力。"},
        ],
    )
    build_transcript_semantic_correction_pack(root, write=True)
    build_transcript_semantic_candidate_discovery_pack(root, write=True)
    prompt = (root / "transcript-semantic-candidate-discovery-prompt.md").read_text(
        encoding="utf-8"
    )
    assert "ordinary_homophone" in prompt
    assert "collocation_anomaly" in prompt

    suggestions = root / "synthetic-linguistic-suggestions.json"
    write_json(
        suggestions,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1",
            "source": "synthetic_linguistic_discovery",
            "suggestions": [
                {
                    "source_segment_index": 0,
                    "correction_type": "ordinary_homophone",
                    "original_text": "权力",
                    "candidate_text": "权利",
                    "reason": "同音普通词在当前语境中疑似误识别",
                    "confidence": 0.91,
                    "needs_human_review": True,
                },
                {
                    "source_segment_index": 1,
                    "correction_type": "collocation_anomaly",
                    "original_text": "提高保障的能力",
                    "candidate_text": "提高保障能力",
                    "reason": "局部搭配疑似异常",
                    "confidence": 0.9,
                    "needs_human_review": True,
                },
            ],
        },
    )
    imported = import_transcript_semantic_candidate_suggestions(
        root,
        input_json=suggestions,
        write=True,
    )
    assert imported["imported_candidate_count"] == 2
    pack = read_json(root / "transcript-semantic-correction-pack.json")
    linguistic = {
        row["correction_type"]: row
        for row in pack["candidates"]
        if row.get("correction_type")
        in {"ordinary_homophone", "collocation_anomaly"}
    }
    assert set(linguistic) == {"ordinary_homophone", "collocation_anomaly"}
    for candidate in linguistic.values():
        decision = _decision(candidate)
        decision["needs_human_review"] = False
        rejected = _validate_decision(
            decision,
            {candidate["candidate_id"]: candidate},
            idx=0,
            min_confidence=0.88,
        )
        assert rejected["accepted"] is False
        assert (
            "linguistic_candidate_requires_human_or_two_independent_direct_evidence"
            in rejected["reject_reasons"]
        )
        accepted = _validate_decision(
            _decision(candidate, human_confirmed=True),
            {candidate["candidate_id"]: candidate},
            idx=0,
            min_confidence=0.88,
        )
        assert accepted["accepted"] is True
