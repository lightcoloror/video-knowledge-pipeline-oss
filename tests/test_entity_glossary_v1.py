from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline import entity_lexicon as entity_lexicon_module
from video_knowledge_pipeline.entity_lexicon import build_entity_lexicon
from video_knowledge_pipeline.quick_health import build_quick_health


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bundle(root: Path, timeline: list[dict]) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"title": "synthetic entity glossary"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False), encoding="utf-8"
    )
    return root


def _confirmed_structured_visual(text: str) -> dict:
    return {
        "evidence_level": "confirmed",
        "markdown": text,
        "validation": {"status": "ok"},
    }


def test_entity_glossary_versions_confirmed_evidence_and_discovers_candidate_only_variants(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        entity_lexicon_module,
        "_pinyin_backend",
        lambda: "unavailable_explicit_alias_only",
    )
    monkeypatch.setattr(entity_lexicon_module, "_pinyin", lambda value: "")
    bundle = _bundle(
        tmp_path / "bundle",
        [
            {
                "index": 1,
                "start": 0,
                "end": 10,
                "transcript": "今天介绍中医人寿。",
                "structured_visual": _confirmed_structured_visual("中意人寿"),
            },
            {
                "index": 2,
                "start": 10,
                "end": 20,
                "transcript": "这里提到安恋人寿。",
                "structured_visual": _confirmed_structured_visual("安联人寿"),
            },
            {
                "index": 3,
                "start": 20,
                "end": 30,
                "transcript": "建军保费是一个观察指标。",
                "structured_visual": _confirmed_structured_visual("件均保费"),
            },
        ],
    )

    result = build_entity_lexicon(bundle, write=True)
    glossary = result["entity_glossary"]

    assert glossary["schema"] == "video_knowledge_pipeline.entity_glossary.v1"
    assert glossary["revision"] == 1
    assert set(glossary["source_hashes"]) == {"manifest", "timeline", "base_glossary"}
    assert glossary["capability"]["status"] == "degraded"
    assert glossary["capability"]["automatic_apply"] is False
    canonical_terms = {row["canonical"]: row for row in glossary["terms"]}
    assert {"中意人寿", "安联人寿", "件均保费"}.issubset(canonical_terms)
    for canonical in ("中意人寿", "安联人寿", "件均保费"):
        evidence = canonical_terms[canonical]["evidence"][0]
        assert evidence["evidence_level"] == "confirmed"
        assert evidence["path"].startswith("timeline.json#/")
        assert len(evidence["sha256"]) == 64

    candidates = {
        (row["original_text"], row["corrected_text"]): row
        for row in result["correction_candidates"]
    }
    for pair in (
        ("中医人寿", "中意人寿"),
        ("安恋人寿", "安联人寿"),
        ("建军保费", "件均保费"),
    ):
        assert pair in candidates
        assert candidates[pair]["auto_apply_allowed"] is False
        assert candidates[pair]["candidate_only"] is True
        assert candidates[pair]["matching_mode"] == "degraded_character_evidence"

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entity_glossary_json"] == "entity-glossary.json"
    assert (bundle / "entity-glossary.json").is_file()

    rerun = build_entity_lexicon(bundle, write=True)
    assert rerun["entity_glossary"]["revision"] == 1

    timeline_path = bundle / "timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline[0]["transcript"] += "补充。"
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    changed = build_entity_lexicon(bundle, write=True)
    assert changed["entity_glossary"]["revision"] == 2


def test_asr_only_text_never_promotes_a_canonical_glossary_term(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        [{"index": 1, "start": 0, "end": 5, "transcript": "欢迎中意人寿。"}],
    )

    result = build_entity_lexicon(bundle, write=False)

    assert result["entity_glossary"]["terms"] == []
    assert result["entity_glossary"]["operator_boundary"]["asr_single_source_cannot_promote"] is True


def test_quick_health_reports_entity_correction_backend_without_importing_models() -> None:
    result = build_quick_health(PROJECT_ROOT)
    check = next(
        row for row in result["checks"] if row["check_id"] == "entity_correction_capability"
    )

    assert check["status"] == "passed"
    assert "pypinyin_available=" in check["detail"]
    assert "entity_glossary=v1" in check["detail"]
    assert "automatic_apply=false" in check["detail"]
    assert "recovery=" in check["detail"]
    assert "pypinyin" not in result["runtime"]["heavy_modules_loaded"]
