from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

import video_knowledge_pipeline.smart_summary_codex as smart_summary_codex
import video_knowledge_pipeline.smart_summary_section_llm as smart_summary_section_llm
from video_knowledge_pipeline.content_profile import apply_content_profile
from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.knowledge_note_export import _render_full_transcript
from video_knowledge_pipeline.production_artifact_gate import evaluate_production_artifact_gate
from video_knowledge_pipeline.smart_summary_global_reduce import run_smart_summary_global_reduce
from video_knowledge_pipeline.smart_summary_section_apply import apply_smart_summary_sections


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(root: Path, *, title: str = "测试视频") -> Path:
    root.mkdir(parents=True)
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": title})
    _write_json(root / "timeline.json", [{"index": 1, "start": 0, "end": 2, "transcript": "测试内容"}])
    _write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 2, "text": "测试内容"}]})
    return root


def _failed_transcript_gate(root: Path) -> None:
    _write_json(
        root / "transcript-quality-gate.json",
        {
            "schema": "video_knowledge_pipeline.transcript_quality_gate.v1",
            "status": "failed",
            "ok": False,
            "fail_count": 1,
            "warning_count": 0,
            "source_completeness": {
                "applicable": True,
                "status": "failed",
                "speech_completeness_verified": False,
            },
            "speaker_diarization": {"required": False, "passed": True, "status": "not_required"},
        },
    )


def test_failed_transcript_gate_blocks_formal_summary_and_exposes_six_states(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _failed_transcript_gate(bundle)

    result = evaluate_production_artifact_gate(bundle, discover_prior_reviews=False)

    assert result["formal_generation_allowed"] is False
    assert result["artifact_state"] == "review-required"
    assert result["execution_status"] == "blocked_by_quality_gate"
    assert result["transcript_quality"]["status"] == "failed"
    assert result["speaker_review_status"]["status"] == "not_required"
    assert result["semantic_fact_status"]["status"] == "not_required"
    assert result["privacy_status"]["status"] == "not_required"
    assert result["publication_readiness"]["status"] == "human_approval_required"
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "video_knowledge_pipeline"
        / "schemas"
        / "production-artifact-gate.v1.schema.json"
    )
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(result)


def test_all_summary_provider_frontdoors_stop_before_call_when_gate_failed(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _failed_transcript_gate(bundle)
    calls: list[str] = []

    def _unexpected_call(**_kwargs):
        calls.append("called")
        raise AssertionError("Provider must not be called")

    monkeypatch.setattr(smart_summary_codex, "call_openai_compatible_text", _unexpected_call)
    monkeypatch.setattr(smart_summary_section_llm, "call_openai_compatible_text", _unexpected_call)

    whole = smart_summary_codex.run_smart_summary_llm_rewrite(bundle, execute=True)
    section = smart_summary_section_llm.run_smart_summary_section_llm_rewrite(bundle, execute=True)
    reduce = run_smart_summary_global_reduce(bundle, execute=True)

    assert calls == []
    assert whole["status"] == "blocked_by_production_artifact_gate"
    assert section["status"] == "blocked_by_production_artifact_gate"
    assert reduce["status"] == "blocked_by_production_artifact_gate"
    assert whole["provider_call_performed"] is False
    assert section["provider_call_performed"] is False
    assert reduce["provider_call_performed"] is False


def test_failed_gate_blocks_manual_install_into_formal_codex_filename(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _failed_transcript_gate(bundle)
    candidate = tmp_path / "candidate.md"
    candidate.write_text("# 不能安装的机器总结\n", encoding="utf-8")

    result = smart_summary_codex.generate_smart_summary_with_codex(bundle, input_md=candidate)

    assert result["status"] == "blocked_by_production_artifact_gate"
    assert not (bundle / "exports" / "smart-summary.codex.md").exists()


def test_failed_gate_blocks_direct_section_apply_into_formal_codex_filename(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _failed_transcript_gate(bundle)
    revisions = tmp_path / "section-revisions.json"
    _write_json(
        revisions,
        {"rows": [{"section_id": "chapter-0001", "final_markdown": "# 不能安装的章节总结"}]},
    )

    result = apply_smart_summary_sections(bundle, input_json=revisions)

    assert result["status"] == "blocked_by_production_artifact_gate"
    assert result["quality_status"] == "blocked_review_required"
    assert result["production_artifact_gate"]["formal_generation_allowed"] is False
    assert not (bundle / "exports" / "smart-summary.codex.md").exists()


def test_failed_transcript_export_has_visible_review_required_watermark(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _failed_transcript_gate(bundle)

    text = _render_full_transcript(
        "测试视频",
        [{"index": 1, "start": 0, "end": 2, "transcript": "测试内容"}],
        bundle_dir=bundle,
        manifest={},
        sidecar={},
    )

    assert "review-required" in text
    assert "机器逐字稿" in text
    assert "transcript_quality:failed" in text


def test_medical_insurance_interview_requires_speaker_fact_and_privacy_review(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", title="客户医疗采访")
    profile_result = apply_content_profile(bundle, profile_id="medical-insurance-interview-v1")
    _write_json(
        bundle / "transcript-quality-gate.json",
        {
            "status": "passed",
            "ok": True,
            "fail_count": 0,
            "source_completeness": {"applicable": True, "status": "passed", "speech_completeness_verified": True},
            "speaker_diarization": {"required": True, "passed": False, "status": "speaker_diarization_required"},
        },
    )

    result = evaluate_production_artifact_gate(bundle, discover_prior_reviews=False)

    assert result["formal_generation_allowed"] is False
    assert result["content_profile"]["profile_id"] == "medical-insurance-interview-v1"
    assert "speaker_review:speaker_diarization_required" in result["blocking_reasons"]
    assert "semantic_fact:needs_human_review" in result["blocking_reasons"]
    assert "privacy:needs_human_review" in result["blocking_reasons"]
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "video_knowledge_pipeline"
        / "schemas"
        / "content-profile.v1.schema.json"
    )
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(profile_result)


def test_interview_profile_changes_section_prompt_contract(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle", title="客户采访")
    apply_content_profile(bundle, profile_id="interview-v1")
    _write_json(
        bundle / "transcript-quality-gate.json",
        {
            "status": "passed",
            "ok": True,
            "fail_count": 0,
            "source_completeness": {"applicable": True, "status": "passed", "speech_completeness_verified": True},
            "speaker_diarization": {"required": True, "passed": True, "status": "passed"},
        },
    )
    captured: list[list[dict]] = []

    monkeypatch.setattr(
        smart_summary_section_llm,
        "build_smart_summary_section_workflow",
        lambda *_args, **_kwargs: {
            "sections": [{"section_id": "chapter-0001", "title": "采访", "status": "ready", "evidence": {"summary_sentences": ["受访者讲述经历"]}}]
        },
    )
    monkeypatch.setattr(
        smart_summary_section_llm,
        "call_openai_compatible_text",
        lambda **kwargs: captured.append(kwargs["messages"]) or {"ok": False, "error": "fixture stop", "content": ""},
    )

    smart_summary_section_llm.run_smart_summary_section_llm_rewrite(
        bundle,
        execute=True,
        write=False,
        provider_config={
            "provider": "fixture-local",
            "base_url": "http://127.0.0.1:9/v1",
            "model": "fixture-model",
            "execution_location": "local",
            "adapter_backend": "openai_compatible",
        },
    )

    assert captured
    system = captured[0][0]["content"]
    assert "采访事实摘要" in system
    assert "不得生成方法论" in system


def test_medical_interview_gate_opens_only_after_required_human_reviews(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", title="客户医疗采访")
    apply_content_profile(bundle, profile_id="medical-insurance-interview-v1")
    _write_json(
        bundle / "transcript-quality-gate.json",
        {
            "status": "passed",
            "ok": True,
            "fail_count": 0,
            "source_completeness": {"applicable": True, "status": "passed", "speech_completeness_verified": True},
            "speaker_diarization": {"required": True, "passed": True, "status": "passed", "distinct_speaker_count": 3},
        },
    )
    _write_json(
        bundle / "speaker-review.json",
        {"status": "human_confirmed_roles", "speaker_mappings": [{"speaker_id": "speaker-1", "role": "受访者"}]},
    )
    _write_json(bundle / "privacy-review.json", {"status": "human_confirmed", "human_confirmed": True})
    _write_json(bundle / "medical-insurance-fact-review.json", {"status": "human_confirmed", "human_confirmed": True})

    result = evaluate_production_artifact_gate(bundle, discover_prior_reviews=False)

    assert result["formal_generation_allowed"] is True
    assert result["blocking_reasons"] == []
    assert result["publication_readiness"]["passed"] is False


def test_cli_exposes_profile_lineage_and_production_gate_commands() -> None:
    parser = build_parser()

    profile = parser.parse_args(["content-profile", "bundle", "--profile", "medical-insurance-interview-v1"])
    lineage = parser.parse_args(["source-review-lineage", "bundle", "--search-root", "outputs", "--apply"])
    gate = parser.parse_args(["production-artifact-gate", "bundle", "--artifact-kind", "smart_summary"])

    assert profile.profile == "medical-insurance-interview-v1"
    assert lineage.apply is True
    assert lineage.search_roots == ["outputs"]
    assert gate.artifact_kind == "smart_summary"
