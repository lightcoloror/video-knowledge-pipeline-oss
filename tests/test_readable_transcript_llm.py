from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.transcript_readable_llm import _agent_substitute_polish_text, run_readable_transcript_llm_polish


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "readable-llm-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Readable LLM", "readable_transcript_json": "readable-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "readable-transcript.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.readable_transcript.v1",
                "segments": [
                    {"start": 0, "end": 4, "text": "那第二点就是时间比较短降低客户压力。"},
                    {"start": 4, "end": 8, "text": "同时展现同理心接下来开始约时间。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bundle


def test_readable_transcript_llm_polish_preview_writes_request_pack(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fail_call(*args, **kwargs):  # pragma: no cover
        raise AssertionError("provider should not be called in preview")

    monkeypatch.setattr("video_knowledge_pipeline.transcript_readable_llm.call_openai_compatible_text", fail_call)

    result = run_readable_transcript_llm_polish(bundle, execute=False)

    assert result["status"] == "planned"
    assert result["ok"] is True
    assert result["request_plan"]["call_count"] == 1
    assert (bundle / "exports" / "readable-transcript-llm-requests.json").exists()
    assert not (bundle / "llm-readable-transcript.json").exists()


def test_readable_transcript_llm_polish_execute_can_promote_corrected(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fake_call(*, provider_config, messages, temperature=0, response_format=None, max_tokens=None):
        assert response_format == {"type": "json_object"}
        return {
            "ok": True,
            "content": json.dumps(
                {
                    "segments": [
                        {"index": 0, "text": "那第二点，就是时间比较短，降低客户压力。"},
                        {"index": 1, "text": "同时展现同理心，接下来开始约时间。"},
                    ]
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr("video_knowledge_pipeline.transcript_readable_llm.call_openai_compatible_text", fake_call)

    result = run_readable_transcript_llm_polish(
        bundle,
        provider_config={"provider": "fixture", "base_url": "http://example.invalid/v1", "model": "fake", "api_key": "SECRET_SHOULD_NOT_LEAK"},
        execute=True,
        promote=True,
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    polished = json.loads((bundle / "llm-readable-transcript.json").read_text(encoding="utf-8"))
    corrected = json.loads((bundle / "corrected-transcript.json").read_text(encoding="utf-8"))
    report_text = (bundle / "readable-transcript-llm-polish.json").read_text(encoding="utf-8")
    assert result["status"] == "executed"
    assert result["ok"] is True
    assert result["quality"]["output_punctuation_density_per_1000_chars"] > result["quality"]["source_punctuation_density_per_1000_chars"]
    assert manifest["llm_readable_transcript_json"] == "llm-readable-transcript.json"
    assert manifest["corrected_transcript_json"] == "corrected-transcript.json"
    assert manifest["corrected_transcript_source"] == "llm_readable_transcript_polish"
    assert polished["segments"][0]["text"] == corrected["segments"][0]["text"]
    assert "SECRET_SHOULD_NOT_LEAK" not in report_text


def test_readable_transcript_llm_import_rejects_semantic_term_cleanup(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    import_path = bundle / "model-output.json"
    import_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"index": 0, "text": "可以帮你做保单整理，看看是否会有一些买虫的。"},
                    {"index": 1, "text": "他采取了一个二则一方式，同时展现同意心。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_readable_transcript_llm_polish(bundle, input_json=import_path, promote=True)

    assert result["status"] == "imported_with_rejections"
    assert result["ok"] is False
    assert {row["reason"] for row in result["failed_items"]} == {
        "readable_semantic_guard_failed"
    }
    assert not (bundle / "llm-readable-transcript.json").exists()
    assert not (bundle / "corrected-transcript.json").exists()

def test_agent_substitute_polish_uses_phrase_boundaries_without_semantic_cleanup() -> None:
    text = "那第二点呢就是时间比较短那同时展现了一个同意心那客户是这样回的说嗯那明晚八点o我找一下我的保单"

    polished = _agent_substitute_polish_text(text)

    assert "那第二点，呢" not in polished
    assert "那第二点呢，" in polished
    assert "明晚八点o" in polished
    assert "同意心" in polished
    assert "客户是这样回的说：" in polished
    assert "好，的" not in _agent_substitute_polish_text("好的那如果明天有变化")
    assert _agent_substitute_polish_text("那客户说").endswith("那客户说。")
    assert polished.endswith("。")


def test_readable_transcript_llm_polish_execute_requires_explicit_provider_config(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fail_call(*args, **kwargs):  # pragma: no cover
        raise AssertionError("provider should not be called without explicit provider_config")

    monkeypatch.setattr("video_knowledge_pipeline.transcript_readable_llm.call_openai_compatible_text", fail_call)

    result = run_readable_transcript_llm_polish(bundle, execute=True)

    assert result["status"] == "missing_provider_config"
    assert result["ok"] is False
    assert result["failed_items"][0]["reason"] == "missing_provider_config"
    assert result["request_plan"]["url"] == ""
    run = json.loads((bundle / "runs" / "readable-transcript-llm-polish" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "needs_input"

def test_readable_transcript_agent_substitute_runs_locally_and_can_promote(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fail_call(*args, **kwargs):  # pragma: no cover
        raise AssertionError("provider should not be called for agent_substitute")

    monkeypatch.setattr("video_knowledge_pipeline.transcript_readable_llm.call_openai_compatible_text", fail_call)

    result = run_readable_transcript_llm_polish(bundle, agent_substitute=True, agent_name="openclaw", promote=True)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    corrected = json.loads((bundle / "corrected-transcript.json").read_text(encoding="utf-8"))
    assert result["status"] == "agent_substitute_executed"
    assert result["agent_substitute"] is True
    assert result["agent_substitute_name"] == "openclaw"
    assert "openclaw" in result["compatible_agent_runtimes"]
    assert result["operator_boundary"]["no_cloud_call"] is True
    assert result["quality"]["output_punctuation_density_per_1000_chars"] >= result["quality"]["source_punctuation_density_per_1000_chars"]
    assert manifest["corrected_transcript_json"] == "corrected-transcript.json"
    assert corrected["segments"][0]["metadata"]["source"] == "llm_readable_transcript_polish"

def test_readable_transcript_codex_substitute_legacy_alias_still_works(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fail_call(*args, **kwargs):  # pragma: no cover
        raise AssertionError("provider should not be called for codex_substitute alias")

    monkeypatch.setattr("video_knowledge_pipeline.transcript_readable_llm.call_openai_compatible_text", fail_call)

    result = run_readable_transcript_llm_polish(bundle, codex_substitute=True, promote=False)

    assert result["status"] == "agent_substitute_executed"
    assert result["legacy_status"] == "codex_substitute_executed"
    assert result["agent_substitute_name"] == "codex"
    assert result["codex_substitute"] is True
