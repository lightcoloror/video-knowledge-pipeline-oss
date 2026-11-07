from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.evidence_conflict_index import build_evidence_conflict_index
from video_knowledge_pipeline.storage import write_json


def test_evidence_conflict_index_keeps_real_external_conflicts_only(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {})
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "schema": "pack",
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "start": 0,
                    "end": 4,
                    "time_range": "00:00:00.000 - 00:00:04.000",
                    "reason": "ordinary_word_conflict_between_asr_and_visual_text",
                    "risk_level": "medium",
                    "original_text": "Play right MCP",
                    "candidate_text": "Playwright MCP",
                    "context_text": "今天讲 Play right MCP",
                    "has_conflict": True,
                    "llm_review_eligible": True,
                    "evidence_source_types": ["ocr", "asr_or_subtitle"],
                    "source_support_summary": {
                        "has_source_conflict": True,
                        "supports_candidate": ["ocr"],
                        "supports_original": ["asr_or_subtitle"],
                    },
                    "evidence": [{"source_type": "ocr", "text": "Playwright MCP"}],
                },
                {
                    "candidate_id": "semcorr-0002",
                    "segment_index": 1,
                    "start": 4,
                    "end": 8,
                    "time_range": "00:00:04.000 - 00:00:08.000",
                    "reason": "numeric_or_step_risk_marker",
                    "risk_level": "low",
                    "original_text": "第二步",
                    "candidate_text": "",
                    "context_text": "第二步打开浏览器",
                    "has_conflict": False,
                    "llm_review_eligible": False,
                    "evidence_source_types": [],
                    "source_support_summary": {},
                    "evidence": [],
                },
            ],
        },
    )

    result = build_evidence_conflict_index(bundle, write=True)

    assert result["status"] == "conflicts_ready"
    assert result["candidate_count"] == 2
    assert result["conflict_count"] == 1
    assert result["conflicts"][0]["candidate_id"] == "semcorr-0001"
    assert result["conflicts"][0]["classification"] == "screen_text_conflict"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_conflict_index_json"] == "evidence-conflict-index.json"
    assert (bundle / "evidence-conflict-index.md").exists()


def test_evidence_conflict_index_accepts_tagger_web_and_multimodal_conflicts(tmp_path: Path) -> None:
    cases = [
        ("qinglong_tagger", "tagger_text_differs_from_transcript", "客户特点", "客户特征", "tagger_conflict"),
        ("web_context", "web_context_differs_from_transcript", "16k", "16k底薪", "web_context_conflict"),
        ("visual_understanding", "visual_understanding_differs_from_transcript", "火山 coding plan", "方舟 Coding Plan", "multimodal_conflict"),
    ]
    for source_type, reason, original, candidate, expected_class in cases:
        bundle = tmp_path / f"bundle-{source_type}"
        bundle.mkdir()
        write_json(bundle / "manifest.json", {})
        write_json(
            bundle / "transcript-semantic-correction-pack.json",
            {
                "schema": "pack",
                "candidates": [
                    {
                        "candidate_id": f"semcorr-{source_type}",
                        "segment_index": 0,
                        "start": 0,
                        "end": 5,
                        "time_range": "00:00:00.000 - 00:00:05.000",
                        "reason": reason,
                        "risk_level": "medium",
                        "original_text": original,
                        "candidate_text": candidate,
                        "context_text": f"这里讲 {original}",
                        "has_conflict": True,
                        "llm_review_eligible": True,
                        "evidence_source_types": [source_type, "asr_or_subtitle"],
                        "source_support_summary": {
                            "has_source_conflict": True,
                            "supports_candidate": [source_type],
                            "supports_original": ["asr_or_subtitle"],
                        },
                        "evidence": [{"source_type": source_type, "text": candidate}],
                    }
                ],
            },
        )

        result = build_evidence_conflict_index(bundle, write=True)

        assert result["conflict_count"] == 1
        assert result["conflicts"][0]["include_in_llm_arbitration"] is True
        assert result["conflicts"][0]["classification"] == expected_class
        assert result["summary"]["classification_counts"][expected_class] == 1


def test_evidence_conflict_index_prefers_explicit_preview_pack_over_stale_disk(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-preview"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {})
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {"schema": "pack", "candidates": []},
    )
    preview_pack = {
        "schema": "pack",
        "candidates": [
            {
                "candidate_id": "preview-conflict",
                "start": 0,
                "end": 5,
                "reason": "ordinary_word_conflict_between_asr_and_visual_text",
                "risk_level": "medium",
                "original_text": "中医人寿",
                "candidate_text": "中意人寿",
                "has_conflict": True,
                "llm_review_eligible": True,
                "evidence_source_types": ["ocr", "asr_or_subtitle"],
                "source_support_summary": {
                    "has_source_conflict": True,
                    "supports_candidate": ["ocr"],
                    "supports_original": ["asr_or_subtitle"],
                },
                "evidence": [{"source_type": "ocr", "text": "中意人寿"}],
            }
        ],
    }

    result = build_evidence_conflict_index(bundle, pack_data=preview_pack, write=False)

    assert result["candidate_count"] == 1
    assert result["conflict_count"] == 1
    assert result["conflicts"][0]["candidate_id"] == "preview-conflict"
    assert not (bundle / "evidence-conflict-index.json").exists()
