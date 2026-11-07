from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.model_task_automation import (
    run_bilinote_mind_map_model,
    run_term_arbitration_model,
)
from video_knowledge_pipeline.storage import read_json, write_json


def _transcript_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "浏览器自动化训练",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"index": 0, "start": 0, "end": 4, "text": "今天比较 playright m c p。"},
                {"index": 1, "start": 4, "end": 8, "text": "屏幕展示正确工具名称。"},
            ]
        },
    )
    write_json(
        root / "timeline.json",
        [
            {"index": 1, "start": 0, "end": 4, "transcript": "今天比较 playright m c p。"},
            {"index": 2, "start": 4, "end": 8, "visual_text": "Playwright MCP"},
        ],
    )
    write_json(
        root / "term-resolution.json",
        {
            "terms": [
                {
                    "canonical_term": "playright m c p",
                    "raw_mentions": ["playright m c p", "Playwright MCP"],
                    "confidence": 0.4,
                    "needs_human_review": True,
                    "evidence": [
                        {"source": "asr", "timeline_index": 1, "mention": "playright m c p"},
                        {"source": "ocr", "timeline_index": 2, "mention": "Playwright MCP"},
                    ],
                }
            ]
        },
    )
    return root


def test_term_arbitration_model_preview_never_calls_provider(tmp_path: Path, monkeypatch) -> None:
    bundle = _transcript_bundle(tmp_path / "bundle")

    def fail_call(*args, **kwargs):
        raise AssertionError("preview must not call a provider")

    monkeypatch.setattr("video_knowledge_pipeline.model_task_automation.model_task_api_call", fail_call)
    result = run_term_arbitration_model(bundle)

    assert result["status"] == "planned"
    assert result["execute"] is False
    assert result["candidate_count"] >= 1


def test_term_arbitration_model_executes_validates_and_imports(tmp_path: Path, monkeypatch) -> None:
    bundle = _transcript_bundle(tmp_path / "bundle")

    def fake_call(*args, **kwargs):
        return {
            "ok": True,
            "status": "ok",
            "task": "term_arbitration",
            "model_type": "transcript_correction",
            "content": json.dumps(
                {
                    "decisions": [
                        {
                            "candidate_id": "term-1",
                            "canonical": "Playwright MCP",
                            "aliases": ["playright m c p", "Playwright MCP"],
                            "confidence": 0.97,
                            "action": "replace",
                            "rationale": "OCR and ASR conflict; OCR shows the product name.",
                            "evidence_indexes": [1, 2],
                            "needs_human_review": False,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "request_plan": {"adapter_backend": "builtin", "provider": {"model": "fixture"}},
        }

    monkeypatch.setattr("video_knowledge_pipeline.model_task_automation.model_task_api_call", fake_call)
    result = run_term_arbitration_model(bundle, provider_config={"provider": "fixture"}, execute=True)

    assert result["status"] == "completed"
    assert result["validation"]["status"] == "ready_for_import"
    glossary = read_json(bundle / "term-arbitration-glossary.json")
    assert any(row["canonical"] == "Playwright MCP" for row in glossary["terms"])


def test_bilinote_mind_map_model_executes_prompt_chunks(tmp_path: Path, monkeypatch) -> None:
    bundle = _transcript_bundle(tmp_path / "bundle")

    def fake_call(*args, **kwargs):
        return {
            "ok": True,
            "status": "ok",
            "task": "bilinote_mind_map",
            "model_type": "text_llm",
            "content": json.dumps(
                {
                    "title": "浏览器自动化",
                    "nodes": [{"title": "工具名称", "summary": "画面确认 Playwright MCP。", "children": []}],
                    "uncertain_terms": [],
                },
                ensure_ascii=False,
            ),
            "request_plan": {"adapter_backend": "builtin", "provider": {"model": "fixture"}},
        }

    monkeypatch.setattr("video_knowledge_pipeline.model_task_automation.model_task_api_call", fake_call)
    result = run_bilinote_mind_map_model(bundle, provider_config={"provider": "fixture"}, execute=True)

    assert result["status"] == "completed"
    assert result["rows"]
    assert (bundle / "exports" / "bilinote-mind-map-result.json").exists()
    assert read_json(bundle / "manifest.json")["bilinote_mind_map_result_json"] == "exports/bilinote-mind-map-result.json"


def test_term_arbitration_model_execute_no_write_returns_unpersisted_result(tmp_path: Path, monkeypatch) -> None:
    bundle = _transcript_bundle(tmp_path / "bundle")

    monkeypatch.setattr(
        "video_knowledge_pipeline.model_task_automation.model_task_api_call",
        lambda *args, **kwargs: {
            "ok": True,
            "status": "ok",
            "task": "term_arbitration",
            "model_type": "transcript_correction",
            "content": '{"decisions": []}',
            "request_plan": {"adapter_backend": "builtin", "provider": {}},
        },
    )
    result = run_term_arbitration_model(
        bundle, provider_config={"provider": "fixture"}, execute=True, write=False
    )

    assert result["status"] == "model_output_ready_not_persisted"
    assert result["model_result"]["decisions"] == []
    assert not (bundle / "term-arbitration-model-result.json").exists()