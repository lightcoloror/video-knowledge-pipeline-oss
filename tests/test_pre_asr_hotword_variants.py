from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.adaptive_asr_route import build_adaptive_asr_route
from video_knowledge_pipeline.entity_lexicon import build_entity_lexicon


def test_pre_asr_uses_safe_explicit_ascii_aliases_but_rejects_error_aliases(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"title":"方案制作原则"}', encoding="utf-8")
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    media = tmp_path / "lecture.wav"
    media.write_bytes(b"fixture")
    lexicon = tmp_path / "industry.json"
    lexicon.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical": "明亚APP",
                        "aliases": ["MIAAPP", "米娅APP"],
                        "entity_type": "product",
                    },
                    {
                        "canonical": "Excel",
                        "aliases": ["cell", "一个cell"],
                        "entity_type": "tool",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    built = build_entity_lexicon(bundle, base_lexicon_json=lexicon, phase="pre_asr", write=False)

    assert built["hotwords"] == ["Excel", "明亚APP"]
    assert built["hotword_variants"] == ["Excel", "cell", "明亚APP", "MIAAPP"]
    assert "米娅APP" not in built["hotword_text"]
    assert "一个cell" not in built["hotword_text"]
    assert built["hotword_audit"]["variant_policy"] == "canonical_plus_safe_explicit_ascii_aliases"

    calls: list[dict] = []

    def local_plan_builder(workspace, media_path, **kwargs):
        calls.append(kwargs)
        return {"plan_path": str(Path(workspace) / "plan.json"), "available": True}

    route = build_adaptive_asr_route(
        bundle,
        media,
        task_profile="terminology",
        base_lexicon_json=lexicon,
        local_plan_builder=local_plan_builder,
    )

    assert route["context"]["hotwords"] == ["Excel", "cell", "明亚APP", "MIAAPP"]
    assert calls[0]["hotword"] == "Excel cell 明亚APP MIAAPP"


def test_pre_asr_accepts_only_short_human_confirmed_review_terms(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"title":"方案制作原则"}', encoding="utf-8")
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    reviews = tmp_path / "review-notes.json"
    reviews.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_review_notes.v1",
                "reviews": [
                    {
                        "original_text": "MIAAPP",
                        "corrected_transcript": "明亚APP",
                        "status": "corrected_transcript",
                        "human_confirmed": True,
                    },
                    {
                        "original_text": "cell",
                        "corrected_transcript": "Excel",
                        "status": "corrected_transcript",
                        "human_confirmed": True,
                    },
                    {
                        "original_text": "unconfirmed",
                        "corrected_transcript": "不得进入",
                        "status": "corrected_transcript",
                        "human_confirmed": False,
                    },
                    {
                        "original_text": "整段",
                        "corrected_transcript": "这是一整句纠正文案；不得作为热词。",
                        "status": "corrected_transcript",
                        "human_confirmed": True,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_entity_lexicon(
        bundle,
        base_lexicon_json=reviews,
        phase="pre_asr",
        write=False,
    )

    assert result["hotwords"] == ["Excel", "明亚APP"]
    assert result["hotword_variants"] == ["Excel", "cell", "明亚APP", "MIAAPP"]
    assert "不得进入" not in result["hotword_text"]
    assert "一整句" not in result["hotword_text"]
