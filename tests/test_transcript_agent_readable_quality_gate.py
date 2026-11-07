from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.local_targeted_asr_plan import build_local_targeted_asr_plan
from video_knowledge_pipeline.smart_summary_codex import _semantic_correction_quality_gate
from video_knowledge_pipeline.transcript_agent_readable import (
    run_agent_readable_transcript_rewrite,
)
from video_knowledge_pipeline.transcript_quality_gate import run_transcript_quality_gate


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "title": "agent readable fixture",
                "postprocessed_transcript_json": "postprocessed-transcript.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "postprocessed-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 4,
                        "text": "那第二点呢就是时间短那客户是这样回的说嗯那明晚八点o我找一下我的保单",
                    },
                    {
                        "start": 4,
                        "end": 8,
                        "text": "可以帮你做保单整理看看是否会有一些买虫的",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def test_agent_readable_transcript_rewrite_local_outputs_task_and_can_promote(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    result = run_agent_readable_transcript_rewrite(
        bundle, agent_name="openclaw", promote=True
    )

    assert result["status"] == "agent_substitute_executed"
    assert result["ok"] is True
    assert result["operator_boundary"]["no_cloud_call"] is True
    assert (bundle / "agent-readable-transcript-task.json").exists()
    assert (bundle / "agent-readable-transcript.json").exists()
    manifest = read_json(bundle / "manifest.json")
    assert (
        manifest["corrected_transcript_source"] == "agent_readable_transcript_rewrite"
    )
    corrected = read_json(bundle / "corrected-transcript.json")
    text = "\n".join(row["text"] for row in corrected["segments"])
    assert "那第二点，呢" not in text
    assert "明晚八点o" in text
    assert "买虫" in text


def test_agent_readable_transcript_rewrite_import_validates_rows(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    import_path = bundle / "agent-result.json"
    import_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "index": 0,
                        "text": "那第二点呢，就是时间短。那客户是这样回的说：嗯，那明晚八点o，我找一下我的保单。",
                    },
                    {"index": 99, "text": "bad"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_agent_readable_transcript_rewrite(
        bundle, input_json=import_path, agent_name="hermes_agent"
    )

    assert result["status"] == "imported_with_rejections"
    assert result["rejected_count"] == 1
    payload = read_json(bundle / "agent-readable-transcript.json")
    assert payload["segments"][0]["text"] == "那第二点呢，就是时间短。那客户是这样回的说：嗯，那明晚八点o，我找一下我的保单。"
    assert payload["segments"][1]["text"] == "可以帮你做保单整理看看是否会有一些买虫的"


def test_transcript_quality_gate_fails_on_residual_bad_terms_and_artifacts(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "corrected-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 4, "text": "那客户说："},
                    {
                        "start": 4,
                        "end": 8,
                        "text": "明晚八点o我找保单，看看有没有买虫的。",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = read_json(bundle / "manifest.json")
    manifest["corrected_transcript_json"] = "corrected-transcript.json"
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    result = run_transcript_quality_gate(bundle)

    assert result["status"] == "failed"
    assert result["ok"] is False
    kinds = {row["kind"] for row in result["issues"]}
    assert "colon_without_content" in kinds
    assert "known_bad_term_residual" in kinds
    assert "transcript-quality-gate.json" == result["artifacts"]["json"]


def test_transcript_quality_gate_keeps_semantic_correction_outside_readable_output(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    run_agent_readable_transcript_rewrite(bundle, agent_name="codex", promote=True)
    corrected = read_json(bundle / "corrected-transcript.json")
    corrected["segments"][1]["text"] = "可以帮你做保单整理，看看是否会有一些买重的。"
    (bundle / "corrected-transcript.json").write_text(
        json.dumps(corrected, ensure_ascii=False), encoding="utf-8"
    )

    result = run_transcript_quality_gate(bundle)

    assert result["status"] == "failed"
    assert result["fail_count"] == 2
    assert result["known_bad_counts"]["买虫"] == 1
    assert result["known_bad_counts"]["明晚八点o"] == 1


def test_transcript_quality_gate_fails_on_unresolved_high_risk_term(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "title": "high risk term fixture",
                "duration_seconds": 2,
                "corrected_transcript_json": "corrected-transcript.json",
                "entity_lexicon_json": "entity-lexicon.json",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "corrected-transcript.json").write_text(
        json.dumps(
            {"segments": [{"start": 0, "end": 2, "text": "We can use my app today."}]}
        ),
        encoding="utf-8",
    )
    (bundle / "entity-lexicon.json").write_text(
        json.dumps(
            {
                "unresolved_high_risk_terms": [
                    {
                        "candidate_id": "entity-0001",
                        "timeline_index": 1,
                        "original_text": "my app",
                        "corrected_text": "MyApp",
                        "entity_type": "product",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_transcript_quality_gate(bundle, write=False)

    assert result["status"] == "failed"
    assert result["ok"] is False
    assert "unresolved_high_risk_term" in {row["kind"] for row in result["issues"]}
    assert any("semantic review notes" in action for action in result["next_actions"])



def test_high_risk_semantic_candidates_require_local_second_asr_evidence(tmp_path: Path) -> None:
    bundle = tmp_path / "semantic-evidence"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {
            "title": "钟巍课程",
            "duration_seconds": 20,
            "corrected_transcript_json": "corrected-transcript.json",
            "transcript_semantic_correction_pack_json": "transcript-semantic-correction-pack.json",
        },
    )
    write_json(
        bundle / "corrected-transcript.json",
        {"segments": [{"start": 0, "end": 20, "text": "这是一段用于验证高风险事实证据门的完整转写内容。"}]},
    )
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "candidate_count": 3,
            "candidates": [
                {"candidate_id": "risk-person", "correction_type": "proper_noun", "risk_level": "high", "llm_review_defer_reason": "needs_conflicting_external_evidence", "start": 10, "end": 11, "original_text": "中威", "candidate_text": "钟巍", "evidence_source_types": ["asr"]},
                {"candidate_id": "risk-number", "correction_type": "number", "risk_level": "high", "llm_review_defer_reason": "needs_conflicting_external_evidence", "start": 12, "end": 13, "original_text": "1111年", "candidate_text": "2011年", "evidence_source_types": ["asr"]},
                {"candidate_id": "already-crosschecked", "correction_type": "proper_noun", "risk_level": "high", "llm_review_defer_reason": "needs_conflicting_external_evidence", "start": 15, "end": 16, "original_text": "错名", "candidate_text": "正名", "evidence_source_types": ["secondary_asr"]},
            ],
        },
    )

    plan = build_local_targeted_asr_plan(bundle, padding_seconds=3, write=True)

    assert plan["status"] == "planned"
    assert plan["selected_candidate_count"] == 2
    assert plan["retry_plan"]["window_count"] == 1
    window = plan["retry_plan"]["windows"][0]
    assert (window["start"], window["end"]) == (7.0, 16.0)
    assert window["candidate_ids"] == ["risk-person", "risk-number"]

    quality = run_transcript_quality_gate(bundle, write=False)
    assert quality["ok"] is False
    assert quality["semantic_evidence"]["required"] is True
    assert "semantic_evidence_pending" in {row["kind"] for row in quality["issues"]}

    summary_gate = _semantic_correction_quality_gate(bundle)
    assert summary_gate["passed"] is False
    assert summary_gate["targeted_evidence"]["required"] is True
