from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.transcript_evidence_correction_pipeline import run_transcript_evidence_correction_pipeline


def test_transcript_evidence_correction_pipeline_preview_writes_safe_chain(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {
            "title": "browser automation lesson",
            "normalized_transcript_json": "normalized-transcript.json",
            "platform_subtitle_path": "platform-subtitle.srt",
        },
    )
    write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0.0, "end": 4.0, "text": "今天讲 Play right MCP 的用法"},
                {"start": 4.0, "end": 8.0, "text": "第二步打开浏览器"},
            ]
        },
    )
    (bundle / "platform-subtitle.srt").write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:04,000",
                "今天讲 Playwright MCP 的用法。",
                "",
                "2",
                "00:00:04,000 --> 00:00:08,000",
                "第二步打开浏览器。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        bundle / "term-resolution.json",
        {
            "terms": [
                {
                    "canonical_term": "Playwright MCP",
                    "raw_mentions": ["Play right MCP", "Playwright MCP"],
                    "confidence": 0.95,
                }
            ]
        },
    )

    result = run_transcript_evidence_correction_pipeline(
        bundle,
        provider_config={"provider": "fixture", "api_key": "SENTINEL_SHOULD_NOT_BE_WRITTEN"},
        execute_llm=False,
        agent_name="openclaw",
        refresh_exports=False,
    )

    assert result["status"] in {
        "completed",
        "completed_no_text_changes",
        "no_safe_draft_decisions",
        "completed_no_semantic_candidates",
        "needs_candidate_discovery_or_review",
        "complete_with_review_gaps",
    }
    assert (bundle / "postprocessed-transcript.json").exists()
    assert (bundle / "evidence-conflict-index.json").exists()
    assert (bundle / "source-arbitrated-transcript.json").exists()
    assert (bundle / "corrected-transcript.json").exists()
    assert (bundle / "transcript-evidence-correction-pipeline.md").exists()
    assert (bundle / "readable-transcript-llm-polish.json").exists()
    assert (bundle / "exports" / "readable-transcript-llm-requests.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["postprocessed_transcript_json"] == "postprocessed-transcript.json"
    assert manifest["evidence_conflict_index_json"] == "evidence-conflict-index.json"
    assert manifest["evidence_conflict_llm_pack_json"] == "evidence-conflict-llm-pack.json"
    assert manifest["corrected_transcript_json"] == "corrected-transcript.json"

    corrected = json.loads((bundle / "source-arbitrated-transcript.json").read_text(encoding="utf-8"))
    assert corrected["schema"] == "video_knowledge_pipeline.source_arbitrated_transcript.v1"
    if "base_source" in corrected:
        assert corrected["base_source"]["source_type"] == "asr_explicit"
        assert corrected["base_source"]["path"].endswith("postprocessed-transcript.json")
    else:
        assert corrected["source"] == "transcript_semantic_correction"
    conflict_index = json.loads((bundle / "evidence-conflict-index.json").read_text(encoding="utf-8"))
    llm_pack = json.loads((bundle / "evidence-conflict-llm-pack.json").read_text(encoding="utf-8"))
    assert conflict_index["llm_arbitration_count"] == 1
    assert llm_pack["candidate_count"] == 1
    assert llm_pack["total_source_candidate_count"] > llm_pack["candidate_count"]
    assert llm_pack["operator_boundary"]["llm_pack_contains_only_evidence_conflicts"] is True
    assert llm_pack["operator_boundary"]["heuristic_risks_without_external_evidence_excluded"] is True
    assert [row["candidate_id"] for row in llm_pack["candidates"]] == [
        conflict_index["conflicts"][0]["candidate_id"]
    ]

    pipeline = json.loads((bundle / "transcript-evidence-correction-pipeline.json").read_text(encoding="utf-8"))
    assert pipeline["artifacts"]["evidence_conflict_llm_pack_json"].endswith("evidence-conflict-llm-pack.json")
    assert pipeline["artifacts"]["llm_readable_transcript_json"].endswith("llm-readable-transcript.json")
    assert pipeline["readable_llm_polish"]["status"] == "agent_substitute_executed"
    assert pipeline["readable_llm_polish"]["agent_substitute"] is True
    assert pipeline["readable_llm_polish"]["agent_substitute_name"] == "openclaw"
    assert pipeline["agent_review"]["result_json"].endswith("transcript-semantic-correction-result.codex.json")
    assert pipeline["codex_review"]["result_json"].endswith("transcript-semantic-correction-result.codex.json")
    assert manifest["mcp_transcript_evidence_correction_pipeline_args"] == "mcp-transcript-evidence-correction-pipeline.args.json"
    mcp_args = json.loads((bundle / "mcp-transcript-evidence-correction-pipeline.args.json").read_text(encoding="utf-8"))
    assert mcp_args["provider_config"] == {}
    assert mcp_args["use_agent_substitute"] is True
    assert mcp_args["agent_name"] == "local_agent"
    assert mcp_args["use_codex_substitute"] is False
    assert mcp_args["run_readable_llm"] is True
    assert mcp_args["execute_readable_llm"] is False
    assert mcp_args["promote_readable_llm"] is False
    assert mcp_args["readable_max_segments_per_batch"] == 40
    assert "SENTINEL" not in (bundle / "transcript-evidence-correction-pipeline.json").read_text(encoding="utf-8")



def test_transcript_evidence_pipeline_uses_secondary_asr_only_as_conflict_evidence(tmp_path: Path) -> None:
    bundle = tmp_path / "dual-asr-bundle"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {
            "title": "dual ASR lesson",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "今天讲 Play right MCP 的用法"},
                {"start": 5.0, "end": 10.0, "text": "第二步打开浏览器"},
            ]
        },
    )
    write_json(
        bundle / "qwen3-transcript.json",
        {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "今天讲 Playwright MCP 的用法"},
                {"start": 5.0, "end": 10.0, "text": "第二步打开浏览器"},
            ]
        },
    )

    result = run_transcript_evidence_correction_pipeline(
        bundle,
        asr_json=bundle / "normalized-transcript.json",
        secondary_asr_json=bundle / "qwen3-transcript.json",
        consensus_agreement_threshold=0.99,
        quality_profile="",
        execute_llm=False,
        use_agent_substitute=False,
        run_readable_llm=False,
        run_agent_readable_rewrite=False,
        run_postprocess=False,
        run_source_arbitration=False,
        materialise_corrected_alias=False,
        refresh_exports=False,
        write=True,
    )

    assert result["asr_consensus"]["status"] == "completed_with_conflicts"
    consensus = json.loads((bundle / "asr-consensus.json").read_text(encoding="utf-8"))
    assert consensus["operator_boundary"]["does_not_promote_secondary"] is True
    assert consensus["conflict_count"] >= 1
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["asr_secondary_transcript"].endswith("qwen3-transcript.json")
    conflicts = json.loads((bundle / "evidence-conflict-index.json").read_text(encoding="utf-8"))
    assert any(row["classification"] == "dual_asr_conflict" for row in conflicts["conflicts"])
    assert not (bundle / "asr-consensus-clips").exists()


def test_transcript_evidence_pipeline_preview_can_disable_readable_stages(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "preview-without-readable-stages"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {
            "title": "preview only",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "primary transcript"},
            ]
        },
    )
    write_json(
        bundle / "secondary.json",
        {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "secondary transcript"},
            ]
        },
    )

    result = run_transcript_evidence_correction_pipeline(
        bundle,
        asr_json=bundle / "normalized-transcript.json",
        secondary_asr_json=bundle / "secondary.json",
        execute_llm=False,
        use_agent_substitute=False,
        run_readable_llm=False,
        run_agent_readable_rewrite=False,
        run_postprocess=False,
        run_source_arbitration=False,
        materialise_corrected_alias=False,
        refresh_exports=False,
        write=False,
    )

    assert result["agent_readable_transcript_rewrite"] == {}
    assert result["transcript_quality_gate"] == {}
    assert not (bundle / "transcript-evidence-correction-pipeline.json").exists()

def test_transcript_evidence_pipeline_registers_multiple_secondary_asr_sources(tmp_path: Path) -> None:
    bundle = tmp_path / "multi-secondary"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"title": "multi secondary", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(bundle / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 5.0, "text": "Play right setup"}]})
    qwen = bundle / "qwen.json"
    mistral = bundle / "mistral.json"
    write_json(qwen, {"provider": "qwen3-asr", "segments": [{"start": 0.0, "end": 5.0, "text": "Playwright setup"}]})
    write_json(mistral, {"provider": "mistral_asr", "segments": [{"start": 0.0, "end": 5.0, "text": "Playwright setup"}]})

    result = run_transcript_evidence_correction_pipeline(
        bundle,
        asr_json=bundle / "normalized-transcript.json",
        secondary_asr_json=qwen,
        additional_secondary_asr_json=[mistral],
        execute_llm=False,
        use_agent_substitute=False,
        run_readable_llm=False,
        run_agent_readable_rewrite=False,
        run_postprocess=False,
        run_source_arbitration=False,
        materialise_corrected_alias=False,
        refresh_exports=False,
        write=True,
    )

    assert result["registered_secondary_asr_sources"] == [str(qwen.resolve()), str(mistral.resolve())]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["asr_secondary_transcripts"] == [str(qwen.resolve()), str(mistral.resolve())]
    pack = json.loads((bundle / "transcript-semantic-correction-pack.json").read_text(encoding="utf-8"))
    secondary_rows = [
        item
        for candidate in pack["candidates"]
        for item in candidate.get("evidence", [])
        if item.get("source_type") == "secondary_asr"
    ]
    assert {row.get("provider") for row in secondary_rows} >= {"qwen3-asr", "mistral_asr"}
    assert all(row.get("artifact_sha256") for row in secondary_rows)
