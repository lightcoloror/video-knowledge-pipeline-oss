from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import entity_lexicon as entity_lexicon_module
from video_knowledge_pipeline.entity_lexicon import build_entity_lexicon


def _bundle(
    root: Path, *, manifest: dict | None = None, timeline: list[dict] | None = None
) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest or {}, ensure_ascii=False), encoding="utf-8"
    )
    (root / "timeline.json").write_text(
        json.dumps(timeline or [], ensure_ascii=False), encoding="utf-8"
    )
    return root


def test_explicit_alias_creates_evidence_backed_correction_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        entity_lexicon_module,
        "_pinyin_backend",
        lambda: "unavailable_explicit_alias_only",
    )
    bundle = _bundle(
        tmp_path / "bundle",
        timeline=[{"index": 3, "transcript": "今天介绍米娅保险。"}],
    )
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical": "明亚保险",
                        "aliases": ["米娅保险", "名娅保险"],
                        "entity_type": "company",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_entity_lexicon(bundle, base_lexicon_json=lexicon, write=False)

    assert result["hotwords"] == ["明亚保险"]
    assert result["pinyin_backend"] == "unavailable_explicit_alias_only"
    assert result["correction_candidates"][0]["original_text"] == "米娅保险"
    assert result["correction_candidates"][0]["corrected_text"] == "明亚保险"
    assert result["correction_candidates"][0]["auto_apply_allowed"] is True


def test_pre_asr_hotwords_filter_transport_metadata_and_keep_explicit_domain_terms(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        manifest={"title": "方案制作原则"},
        timeline=[
            {
                "index": 1,
                "transcript": "使用MIAAPP中的PDF版本。",
                "visual_text": "COMPANY LOGO\n![img-0.jpeg](img-0.jpeg)\n明亚保险经纪 领航计划",
                "structured_visual": [
                    {
                        "schema": "lecture_visual_understanding.v1",
                        "provider": "mistral",
                        "model": "mistral-ocr-4-0",
                        "source": "online_ocr",
                        "image_path": "assets/img-0.jpeg",
                        "markdown": "明亚保险经纪 领航计划",
                    }
                ],
            }
        ],
    )
    post_asr_compat = {"phase": "post_asr", "marker": "keep"}
    (bundle / "entity-lexicon.json").write_text(
        json.dumps(post_asr_compat, ensure_ascii=False),
        encoding="utf-8",
    )
    lexicon = tmp_path / "industry-lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical": "明亚APP",
                        "aliases": ["MIAAPP"],
                        "entity_type": "product",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_entity_lexicon(
        bundle,
        base_lexicon_json=lexicon,
        phase="pre_asr",
        write=True,
    )

    folded = {str(value).casefold() for value in result["hotwords"]}
    assert "明亚app" in folded
    assert "mistral" not in folded
    assert "mistral-ocr-4-0" not in folded
    assert "company logo" not in folded
    assert not any("img-0.jpeg" in value for value in folded)
    assert result["phase"] == "pre_asr"
    assert result["correction_candidates"] == []
    assert result["operator_boundary"]["asr_input_allowed"] is True
    assert result["operator_boundary"]["asr_rerun_allowed"] is False
    assert result["hotword_audit"]["rejected_count"] >= 1
    assert (bundle / "entity-hotwords.pre-asr.txt").is_file()
    assert (bundle / "entity-hotword-audit.pre-asr.json").is_file()
    assert (
        json.loads((bundle / "entity-lexicon.json").read_text(encoding="utf-8"))
        == post_asr_compat
    )


def test_asr_only_entity_does_not_become_dynamic_hotword(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        timeline=[{"index": 1, "transcript": "欢迎使用虚构云智保产品。"}],
    )

    result = build_entity_lexicon(bundle, write=False)

    assert "虚构云智保" not in result["hotword_text"]
    assert result["operator_boundary"]["dynamic_terms_require_non_asr_support"] is True
    assert result["phase"] == "post_asr"
    assert (
        result["operator_boundary"]["post_asr_terms_are_correction_evidence_only"]
        is True
    )
    assert result["operator_boundary"]["asr_rerun_allowed"] is False


def test_ocr_or_metadata_can_supply_dynamic_hotword_but_pipeline_terms_are_filtered(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        manifest={"title": "明亚保险客户沟通课程 - SenseVoice ASR"},
        timeline=[
            {
                "index": 1,
                "transcript": "今天介绍米娅保险。",
                "visual_text": "明亚保险",
            }
        ],
    )

    result = build_entity_lexicon(bundle, write=False)
    hotwords = {str(value).casefold() for value in result["hotwords"]}

    assert any("明亚" in value for value in result["hotwords"])
    assert "sensevoice" not in hotwords
    assert "asr" not in hotwords
    assert (
        result["operator_boundary"]["pinyin_similarity_never_directly_replaces_text"]
        is True
    )


def test_generic_business_words_are_not_promoted_to_entities(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        manifest={"title": "没有客户时通过活动拓客和获客"},
    )

    result = build_entity_lexicon(bundle, write=False)

    assert "拓客" not in result["hotwords"]
    assert "获客" not in result["hotwords"]


def test_term_arbitration_glossary_shape_is_accepted(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        timeline=[
            {"index": 2, "transcript": "\u4eca\u5929\u4ecb\u7ecd\u7c73\u5a05\u3002"}
        ],
    )
    lexicon = tmp_path / "glossary.json"
    lexicon.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical_term": "\u660e\u4e9a",
                        "raw_mentions": ["\u7c73\u5a05", "\u540d\u5a05"],
                        "entity_type": "company",
                        "confidence": 0.99,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = build_entity_lexicon(bundle, base_lexicon_json=lexicon, write=False)
    assert result["terms"][0]["canonical"] == "\u660e\u4e9a"
    assert result["terms"][0]["aliases"] == [
        "\u660e\u4e9a",
        "\u7c73\u5a05",
        "\u540d\u5a05",
    ]
    assert result["correction_candidates"][0]["auto_apply_allowed"] is True


def test_unresolved_high_risk_alias_requires_review(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        timeline=[{"index": 1, "transcript": "We use my app for client work."}],
    )
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical": "MyApp",
                        "aliases": ["my app"],
                        "entity_type": "product",
                        "confidence": 0.99,
                        "review_required": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_entity_lexicon(bundle, base_lexicon_json=lexicon, write=False)

    assert result["status"] == "review_required"
    assert result["quality_gate_passed"] is False
    assert result["unresolved_high_risk_term_count"] == 1
    assert result["unresolved_high_risk_terms"][0]["original_text"] == "my app"
    assert (
        result["operator_boundary"][
            "unresolved_high_risk_terms_cannot_pass_quality_gate"
        ]
        is True
    )
