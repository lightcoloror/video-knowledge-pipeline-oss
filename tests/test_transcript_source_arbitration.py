from __future__ import annotations

import json
import shutil
from pathlib import Path

from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.term_arbitration_codex import build_term_arbitration_codex_pack, validate_term_arbitration_codex_result
from video_knowledge_pipeline.term_correction_impact import term_correction_impact_report
from video_knowledge_pipeline.transcript import parse_transcript
from video_knowledge_pipeline.transcript_source_arbitration import arbitrate_transcript_sources


def test_transcript_source_arbitration_promotes_high_confidence_terms() -> None:
    root = Path("outputs/test-transcript-source-arbitration/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "browser automation lesson",
            "normalized_transcript_json": "normalized-transcript.json",
            "platform_subtitle_path": "platform-subtitle.srt",
        },
    )
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0.0, "end": 4.0, "text": "今天讲 Play right MCP 的用法"},
                {"start": 4.0, "end": 8.0, "text": "第二步打开浏览器"},
            ]
        },
    )
    (root / "platform-subtitle.srt").write_text(
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
        root / "term-resolution.json",
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

    result = arbitrate_transcript_sources(root)

    assert result["status"] == "completed"
    assert result["summary"]["changed_segments"] == 1
    assert result["quality_summary"]["status"] == "changed_clean"
    assert result["quality_summary"]["high_confidence_term_replacements"] == 1
    assert result["quality_summary"]["can_use_as_summary_input"] is True
    assert result["quality_summary"]["summary_input_policy"]["mode"] == "corrected_clean"
    assert result["quality_summary"]["summary_input_policy"]["can_use_corrected_transcript"] is True
    assert result["quality_summary"]["review_required"] is False
    assert result["quality_summary"]["trusted_segment_indexes"]
    manifest = read_json(root / "manifest.json")
    assert manifest["corrected_transcript_json"] == "source-arbitrated-transcript.json"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["text"] == "今天讲 Playwright MCP 的用法"
    assert corrected["segments"][0]["chosen_source"] == "normalized_transcript_json"
    assert corrected["quality_summary"]["average_confidence"] > 0
    assert manifest["transcript_source_arbitration_quality"]["status"] == "changed_clean"
    report = (root / "transcript-source-arbitration.md").read_text(encoding="utf-8")
    assert "Arbitration Quality" in report
    assert "High-confidence term replacements" in report
    assert "Summary input mode" in report
    assert (root / "transcript-source-arbitration.md").exists()
    assert (root / "mcp-transcript-source-arbitration.args.json").exists()
    cues = parse_transcript(root / "source-arbitrated-transcript.json")
    assert cues[0].text == "今天讲 Playwright MCP 的用法"


def test_source_loading_ignores_derived_copies_and_deduplicates_normalized_json_srt(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(
        root / "manifest.json",
        {
            "normalized_transcript_srt": "normalized-transcript.srt",
            "transcript_json": "corrected-transcript.json",
            "transcript_srt": "corrected-transcript.srt",
            "source_transcript": "agent-readable-transcript.json",
            "transcript_path": "copied-transcript.json",
        },
    )
    normalized_segments = [
        {"start": 0.0, "end": 2.0, "text": "第一段内容。"},
        {"start": 2.0, "end": 4.0, "text": "第二段内容。"},
    ]
    write_json(root / "normalized-transcript.json", {"segments": normalized_segments})
    normalized_srt = (
        "1\n00:00:00,000 --> 00:00:02,000\n第一段内容。\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n第二段内容。\n"
    )
    (root / "normalized-transcript.srt").write_text(normalized_srt, encoding="utf-8")
    write_json(
        root / "corrected-transcript.json",
        {
            "schema": "video_knowledge_pipeline.corrected_transcript.v1",
            "segments": [{"start": 0.0, "end": 4.0, "text": "派生纠正版不应重新投票。"}],
        },
    )
    (root / "corrected-transcript.srt").write_text(
        "1\n00:00:00,000 --> 00:00:04,000\n派生纠正版不应重新投票。\n",
        encoding="utf-8",
    )
    write_json(
        root / "agent-readable-transcript.json",
        {
            "schema": "video_knowledge_pipeline.agent_readable_transcript.v1",
            "segments": [{"start": 0.0, "end": 4.0, "text": "代理可读稿也不应重新投票。"}],
        },
    )
    write_json(
        root / "copied-transcript.json",
        {
            "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
            "segments": [{"start": 0.0, "end": 4.0, "text": "改名后的派生稿也不应重新投票。"}],
        },
    )
    write_json(
        root / "timeline.json",
        [
            {
                "start": 0.0,
                "end": 2.0,
                "transcript": "错误的时间线 ASR 一。",
                "subtitle": "第一段内容。",
            },
            {
                "start": 2.0,
                "end": 4.0,
                "transcript": "错误的时间线 ASR 二。",
                "subtitle": "第二段内容。",
            },
        ],
    )

    result = arbitrate_transcript_sources(root, promote=False, write=False)

    assert result["source_count"] == 2
    assert [row["source_id"] for row in result["sources"]] == [
        "normalized-transcript.json",
        "timeline_subtitle",
    ]
    assert result["base_source"]["source_id"] == "normalized-transcript.json"
    assert result["base_source"]["path"].endswith("normalized-transcript.json")
    assert result["summary"]["segments"] == 2
    assert result["summary"]["review_segments"] == 0
    assert result["status"] == "completed"


def test_timeline_asr_is_used_only_when_no_file_transcript_exists(tmp_path: Path) -> None:
    root = tmp_path / "timeline-only"
    root.mkdir()
    write_json(root / "manifest.json", {})
    write_json(
        root / "timeline.json",
        [{"start": 1.0, "end": 3.0, "transcript": "只有时间线语音文本。"}],
    )

    result = arbitrate_transcript_sources(root, promote=False, write=False)

    assert result["source_count"] == 1
    assert result["base_source"]["source_id"] == "timeline_asr"
    assert result["summary"]["segments"] == 1


def test_explicit_asr_is_primary_and_explicit_corrected_fields_remain_allowed(tmp_path: Path) -> None:
    root = tmp_path / "explicit-sources"
    root.mkdir()
    write_json(
        root / "manifest.json",
        {
            "normalized_transcript_json": "normalized-transcript.json",
            "human_corrected_transcript_json": "corrected-transcript.json",
            "llm_corrected_transcript_json": "agent-readable-transcript.json",
        },
    )
    write_json(
        root / "normalized-transcript.json",
        {"segments": [{"start": 0.0, "end": 2.0, "text": "标准化 ASR。"}]},
    )
    write_json(
        root / "corrected-transcript.json",
        {
            "schema": "video_knowledge_pipeline.corrected_transcript.v1",
            "segments": [{"start": 0.0, "end": 2.0, "text": "人工纠正版。"}],
        },
    )
    write_json(
        root / "agent-readable-transcript.json",
        {
            "schema": "video_knowledge_pipeline.agent_readable_transcript.v1",
            "segments": [{"start": 0.0, "end": 2.0, "text": "LLM 纠正版。"}],
        },
    )
    explicit_asr = root / "source-arbitrated-explicit.json"
    write_json(
        explicit_asr,
        {
            "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
            "segments": [{"start": 0.0, "end": 2.0, "text": "显式指定 ASR。"}],
        },
    )

    result = arbitrate_transcript_sources(
        root,
        asr_json=explicit_asr,
        promote=False,
        write=False,
    )

    assert result["base_source"]["source_id"] == "explicit_asr"
    assert {row["source_id"] for row in result["sources"]} == {
        "explicit_asr",
        "human_corrected_transcript_json",
        "llm_corrected_transcript_json",
        "normalized_transcript_json",
    }


def test_transcript_source_arbitration_accepts_entity_lexicon_schema_and_chinese_aliases(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(root / "manifest.json", {"normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {"segments": [{"start": 0.0, "end": 3.0, "text": "欢迎米娅的伙伴参加名娅领航计划。"}]},
    )
    glossary = tmp_path / "entity-lexicon.json"
    write_json(
        glossary,
        {
            "terms": [
                {
                    "canonical_term": "明亚",
                    "raw_mentions": ["米娅", "名娅"],
                    "entity_type": "company",
                    "confidence": 0.99,
                    "review_required": False,
                }
            ]
        },
    )

    result = arbitrate_transcript_sources(root, glossary_json=glossary)

    assert result["summary"]["high_confidence_term_replacements"] == 2
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["text"] == "欢迎明亚的伙伴参加明亚领航计划。"


def test_transcript_source_arbitration_does_not_apply_review_required_chinese_alias(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(root / "manifest.json", {"normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 2.0, "text": "今天介绍米娅。"}]})
    glossary = tmp_path / "entity-lexicon.json"
    write_json(
        glossary,
        {
            "terms": [
                {
                    "canonical": "明亚",
                    "aliases": ["米娅"],
                    "confidence": 0.99,
                    "review_required": True,
                }
            ]
        },
    )

    result = arbitrate_transcript_sources(root, glossary_json=glossary)

    assert result["summary"]["high_confidence_term_replacements"] == 0
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["text"] == "今天介绍米娅。"

def test_transcript_source_arbitration_can_write_without_promoting() -> None:
    root = Path("outputs/test-transcript-source-arbitration-no-promote/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 2.0, "text": "原始文本"}]})

    result = arbitrate_transcript_sources(root, promote=False)

    assert result["status"] == "completed"
    manifest = read_json(root / "manifest.json")
    assert manifest["source_arbitrated_transcript_json"] == "source-arbitrated-transcript.json"
    assert "corrected_transcript_json" not in manifest


def test_term_arbitration_codex_pack_imports_tool_glossary() -> None:
    root = Path("outputs/test-term-arbitration-codex/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "browser automation lesson",
            "description": "浏览器自动化工具横评，重点比较 Playwright MCP、BrowserHarness、Browserbase。",
            "platform": "bilibili",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 5.0, "text": "今天横评 playright m c p 和 brow harness。"}]})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 5.0,
                "transcript": "今天横评 playright m c p 和 brow harness。",
                "tagger_labels": ["工具名", "横评"],
            },
            {
                "index": 2,
                "start": 5.0,
                "end": 9.0,
                "visual_text": "Playwright MCP / BrowserHarness",
                "structured_visual": {"markdown": "| Tool | Type |\n| Playwright MCP | Browser automation |\n| BrowserHarness | CDP control |"},
                "visual_understanding": {"objects": ["BrowserHarness workspace"], "non_text_info": "画面展示 BrowserHarness 控制台和 CDP 连接状态。"},
                "temporal_visual_understanding": {"event_sequence": ["打开 BrowserHarness", "连接已登录 Chrome"]},
                "tagger_visual_summary": "工具名对比页，出现 Playwright MCP 和 BrowserHarness。",
            },
        ],
    )
    write_json(
        root / "term-resolution.json",
        {
            "terms": [
                {
                    "canonical_term": "playright m c p",
                    "raw_mentions": ["playright m c p", "Playwright MCP"],
                    "confidence": 0.42,
                    "needs_human_review": True,
                    "source_counts": {"asr": 1, "ocr": 1},
                    "evidence": [
                        {"source": "asr", "timeline_index": 1, "mention": "playright m c p", "context": "今天横评 playright m c p 和 brow harness。"},
                        {"source": "ocr", "timeline_index": 2, "mention": "Playwright MCP", "context": "Playwright MCP / BrowserHarness"},
                    ],
                }
            ]
        },
    )

    planned = build_term_arbitration_codex_pack(root)

    assert planned["status"] == "draft_ready"
    assert planned["candidate_count"] >= 1
    assert planned["context_sources"]["metadata"]["description"].startswith("浏览器自动化工具横评")
    assert planned["context_sources"]["source_counts"]["visual_understanding"] == 1
    assert planned["context_sources"]["source_counts"]["structured_visual"] == 1
    playright_candidate = next(row for row in planned["candidates"] if "playright m c p" in row["raw_mentions"])
    assert any("visual_understanding" in row.get("source_channels", []) for row in playright_candidate["evidence"])
    prompt = (root / "term-arbitration-codex-prompt.md").read_text(encoding="utf-8")
    assert "Video-Level Context Sources" in prompt
    assert "Codex 临时代替在线文本大模型 API" in prompt
    assert "Semantic Arbitration Strategy" in prompt
    assert "BrowserHarness workspace" in prompt
    assert "工具名对比页" in prompt
    assert (root / "term-arbitration-codex-prompt.md").exists()
    assert (root / "term-arbitration-codex-draft.json").exists()
    assert (root / "term-arbitration-codex-result.template.json").exists()
    assert (root / "term-arbitration-codex-result.codex.md").exists()
    assert planned["artifacts"]["result_codex_markdown"] == "term-arbitration-codex-result.codex.md"
    assert planned["artifacts"]["mcp_validate_args"] == "mcp-term-arbitration-codex-validate.args.json"
    assert planned["artifacts"]["mcp_closure_codex_args"] == "mcp-term-correction-closure-codex.args.json"
    validate_args = read_json(root / "mcp-term-arbitration-codex-validate.args.json")
    assert validate_args["input_json"].endswith("term-arbitration-codex-result.codex.md")
    closure_codex_args = read_json(root / "mcp-term-correction-closure-codex.args.json")
    assert closure_codex_args["input_json"].endswith("term-arbitration-codex-result.codex.md")
    assert closure_codex_args["accept_draft"] is False
    assert planned["llm_semantic_arbitration"]["strategy"] == "codex_substitute_for_online_text_llm"
    assert planned["llm_semantic_arbitration"]["review_status"] == "codex_review_pending"
    assert planned["llm_semantic_arbitration"]["rule_draft_is_not_semantic_confirmation"] is True
    template = read_json(root / "term-arbitration-codex-result.template.json")
    assert template["reviewer"] == "codex_substitute_for_online_text_llm"
    assert template["decisions"]
    stub = (root / "term-arbitration-codex-result.codex.md").read_text(encoding="utf-8")
    assert "Codex Term Arbitration Response" in stub
    assert "```json" in stub
    assert "term-arbitration-codex-pack.json" in stub
    draft = read_json(root / "term-arbitration-codex-draft.json")
    assert any(row["canonical"] == "Playwright MCP" for row in draft["decisions"])
    assert any(row["canonical"] == "BrowserHarness" for row in draft["decisions"])
    assert all(row["action"] == "replace" for row in draft["decisions"] if row["canonical"] == "MCP")
    draft_glossary = read_json(root / "term-arbitration-glossary.json")
    assert draft_glossary["terms"] == []
    custom_stub = "# Custom Codex Response\n\n```json\n{\"schema\": \"custom\", \"decisions\": []}\n```\n"
    (root / "term-arbitration-codex-result.codex.md").write_text(custom_stub, encoding="utf-8")

    accepted_draft = build_term_arbitration_codex_pack(root, accept_draft=True)
    assert (root / "term-arbitration-codex-result.codex.md").read_text(encoding="utf-8") == custom_stub

    assert accepted_draft["status"] == "imported"
    assert accepted_draft["accept_draft"] is True
    assert accepted_draft["import_source"] == "codex_substitute_local_draft"
    assert accepted_draft["operator_boundary"]["auto_accepts_only_high_confidence_draft"] is True
    auto_result = read_json(root / "term-arbitration-codex-result.json")
    assert auto_result["source"] == "codex_substitute_local_draft"
    auto_glossary = read_json(root / "term-arbitration-glossary.json")
    assert any(row["canonical"] == "Playwright MCP" for row in auto_glossary["terms"])
    assert any(row["canonical"] == "BrowserHarness" for row in auto_glossary["terms"])
    auto_manifest = read_json(root / "manifest.json")
    assert auto_manifest["term_arbitration_codex_result_json"] == "term-arbitration-codex-result.json"
    assert auto_manifest["mcp_term_arbitration_codex_validate_args"] == "mcp-term-arbitration-codex-validate.args.json"
    assert auto_manifest["mcp_term_correction_closure_codex_args"] == "mcp-term-correction-closure-codex.args.json"
    assert auto_manifest["term_arbitration_codex_summary"]["import_source"] == "codex_substitute_local_draft"
    reviewed_path = root / "term-arbitration-codex-result.reviewed.json"
    write_json(
        reviewed_path,
        {
            "schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1",
            "decisions": [
                {
                    "candidate_id": "term-1",
                    "canonical": "Playwright MCP",
                    "aliases": ["playright m c p", "Playwright MCP"],
                    "confidence": 0.96,
                    "action": "replace",
                    "rationale": "OCR and topic context indicate the browser automation tool name.",
                    "evidence_indexes": [1, 2],
                    "needs_human_review": False,
                },
                {
                    "candidate_id": "term-2",
                    "canonical": "BrowserHarness",
                    "aliases": ["brow harness", "BrowserHarness"],
                    "confidence": 0.96,
                    "action": "replace",
                    "rationale": "OCR and topic context indicate the browser automation tool name.",
                    "evidence_indexes": [1, 2],
                    "needs_human_review": False,
                }
            ],
        },
    )

    imported = build_term_arbitration_codex_pack(root, input_json=reviewed_path)

    assert imported["status"] == "imported"
    assert imported["llm_semantic_arbitration"]["review_status"] == "codex_or_llm_reviewed_import"
    codex_response_path = root / "term-arbitration-codex-result.codex.md"
    codex_response_path.write_text(
        "Here is the reviewed JSON:\n\n```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1",
                "source": "codex_reviewed_import",
                "decisions": [
                    {
                        "candidate_id": "term-1",
                        "canonical": "Playwright MCP",
                        "aliases": ["playright m c p", "Playwright MCP"],
                        "confidence": 0.97,
                        "action": "replace",
                        "rationale": "Codex semantic review confirms the tool name from OCR and context.",
                        "evidence_indexes": [1, 2],
                        "needs_human_review": False,
                    },
                    {
                        "candidate_id": "term-2",
                        "canonical": "BrowserHarness",
                        "aliases": ["brow harness", "BrowserHarness"],
                        "confidence": 0.96,
                        "action": "replace",
                        "rationale": "Codex semantic review confirms the tool name from OCR and context.",
                        "evidence_indexes": [1, 2],
                        "needs_human_review": False,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    imported_from_markdown = build_term_arbitration_codex_pack(root, input_json=codex_response_path)
    assert imported_from_markdown["status"] == "imported"
    assert imported_from_markdown["import_source"] == "codex_reviewed_import"
    assert imported_from_markdown["accepted_decision_count"] == 2
    glossary = read_json(root / "term-arbitration-glossary.json")
    assert glossary["terms"][0]["canonical"] == "Playwright MCP"
    assert "playright m c p" in glossary["terms"][0]["aliases"]
    manifest = read_json(root / "manifest.json")
    assert manifest["term_arbitration_glossary_json"] == "term-arbitration-glossary.json"
    arbitration = arbitrate_transcript_sources(root)
    assert arbitration["summary"]["changed_segments"] == 1
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["text"] == "今天横评 Playwright MCP 和 BrowserHarness。"

def test_validate_term_arbitration_codex_result_preflights_markdown() -> None:
    root = Path("outputs/test-term-arbitration-codex-validation/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "validate codex response"})
    write_json(
        root / "term-arbitration-codex-pack.json",
        {
            "candidates": [
                {"id": "term-1", "evidence": [{"timeline_index": 1}]},
                {"id": "term-2", "evidence": [{"timeline_index": 2}]},
            ]
        },
    )
    response = root / "term-arbitration-codex-result.codex.md"
    response.write_text(
        "Codex response:\n\n```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1",
                "decisions": [
                    {"candidate_id": "term-1", "canonical": "Playwright MCP", "aliases": ["playright m c p"], "confidence": 0.96, "action": "replace", "needs_human_review": False, "rationale": "OCR supports it.", "evidence_indexes": [1]},
                    {"candidate_id": "term-2", "canonical": "MaybeTerm", "aliases": ["maybe term"], "confidence": 0.52, "action": "review", "needs_human_review": True, "rationale": "Not enough evidence."},
                    {"candidate_id": "term-3", "canonical": "UnsafeTerm", "aliases": ["unsafe term"], "confidence": 0.97, "action": "replace", "needs_human_review": False},
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    result = validate_term_arbitration_codex_result(root, input_json=response)

    assert result["status"] == "ready_for_import"
    assert result["ok"] is True
    assert result["accepted_decision_count"] == 1
    assert result["rejected_decision_count"] == 2
    assert result["rejected_decisions"][0]["rejection_reasons"] == ["action_not_replace", "confidence_below_minimum", "needs_human_review"]
    unsafe = next(row for row in result["rejected_decisions"] if row["canonical"] == "UnsafeTerm")
    assert "missing_semantic_rationale" in unsafe["rejection_reasons"]
    assert "missing_evidence_indexes" in unsafe["rejection_reasons"]
    assert "unknown_candidate_id" in unsafe["rejection_reasons"]
    assert "term-correction-closure" in result["next_actions"][0]
    assert (root / "term-arbitration-codex-validation.md").exists()
    validation_md = (root / "term-arbitration-codex-validation.md").read_text(encoding="utf-8")
    assert "Rejection Guidance" in validation_md
    assert "missing_semantic_rationale" in validation_md
    assert "do not invent IDs" in validation_md
    assert "review their rejection_reasons" in validation_md
    manifest = read_json(root / "manifest.json")
    assert manifest["term_arbitration_codex_validation_markdown"] == "term-arbitration-codex-validation.md"
    args = read_json(root / "mcp-term-arbitration-codex-validate.args.json")
    assert args["input_json"].endswith("term-arbitration-codex-result.codex.md")

def test_term_correction_closure_accepts_draft_and_refreshes_exports() -> None:
    from video_knowledge_pipeline.term_correction_closure import run_term_correction_closure

    root = Path("outputs/test-term-correction-closure/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "term closure lesson",
            "description": "浏览器自动化课程，正确工具名是 Playwright MCP 和 BrowserHarness。",
            "platform": "bilibili",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 5.0, "text": "今天横评 playright m c p 和 brow harness。"}]})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 5.0,
                "transcript": "今天横评 playright m c p 和 brow harness。",
                "visual_text": "Playwright MCP / BrowserHarness",
                "structured_visual": {"markdown": "| Tool | Type |\n| Playwright MCP | Browser automation |\n| BrowserHarness | CDP control |"},
            }
        ],
    )
    write_json(
        root / "term-resolution.json",
        {
            "terms": [
                {
                    "canonical_term": "playright m c p",
                    "raw_mentions": ["playright m c p", "Playwright MCP"],
                    "confidence": 0.42,
                    "needs_human_review": True,
                    "source_counts": {"asr": 1, "ocr": 1},
                }
            ]
        },
    )

    result = run_term_correction_closure(root, accept_draft=True)

    assert result["status"] in {"completed", "needs_smart_summary_fix"}
    assert result["steps"]["term_arbitration_codex_validation"]["status"] == "skipped_no_input_json"
    assert result["steps"]["term_arbitration_codex"]["status"] == "imported"
    assert result["steps"]["transcript_source_arbitration"]["ok"] is True
    assert result["steps"]["term_correction_impact"]["final_export_alias_total"] == 0
    assert (root / "term-correction-closure.md").exists()
    assert (root / "term-arbitration-glossary.json").exists()
    assert (root / "source-arbitrated-transcript.json").exists()
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["text"] == "今天横评 Playwright MCP 和 BrowserHarness。"
    assert (root / "exports" / "smart-summary.md").exists()
    manifest = read_json(root / "manifest.json")
    assert manifest["term_correction_closure_markdown"] == "term-correction-closure.md"
    assert manifest["mcp_term_correction_closure_args"] == "mcp-term-correction-closure.args.json"

def test_term_correction_closure_imports_codex_markdown_response() -> None:
    from video_knowledge_pipeline.term_correction_closure import run_term_correction_closure

    root = Path("outputs/test-term-correction-closure-codex-import/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "term closure codex import lesson",
            "description": "浏览器自动化课程，正确工具名是 Playwright MCP 和 BrowserHarness。",
            "platform": "bilibili",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 5.0, "text": "今天横评 playright m c p 和 brow harness。"}]})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 5.0,
                "transcript": "今天横评 playright m c p 和 brow harness。",
                "visual_text": "Playwright MCP / BrowserHarness",
                "structured_visual": {"markdown": "| Tool | Type |\n| Playwright MCP | Browser automation |\n| BrowserHarness | CDP control |"},
            }
        ],
    )
    write_json(
        root / "term-resolution.json",
        {
            "terms": [
                {
                    "canonical_term": "playright m c p",
                    "raw_mentions": ["playright m c p", "Playwright MCP"],
                    "confidence": 0.42,
                    "needs_human_review": True,
                    "source_counts": {"asr": 1, "ocr": 1},
                }
            ]
        },
    )
    codex_response = root / "term-arbitration-codex-result.codex.md"
    codex_response.write_text(
        "Codex semantic decision:\n\n```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1",
                "source": "codex_reviewed_import",
                "decisions": [
                    {
                        "candidate_id": "term-1",
                        "canonical": "Playwright MCP",
                        "aliases": ["playright m c p", "Playwright MCP"],
                        "confidence": 0.97,
                        "action": "replace",
                        "rationale": "OCR and course context confirm Playwright MCP.",
                        "evidence_indexes": [1],
                        "needs_human_review": False,
                    },
                    {
                        "candidate_id": "term-2",
                        "canonical": "BrowserHarness",
                        "aliases": ["brow harness", "BrowserHarness"],
                        "confidence": 0.96,
                        "action": "replace",
                        "rationale": "OCR and course context confirm BrowserHarness.",
                        "evidence_indexes": [1],
                        "needs_human_review": False,
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    result = run_term_correction_closure(root, input_json=codex_response)

    assert result["status"] in {"completed", "needs_smart_summary_fix"}
    assert result["accept_draft"] is False
    assert result["input_json"].endswith("term-arbitration-codex-result.codex.md")
    assert result["semantic_review_status"] == "codex_or_llm_reviewed_import"
    assert result["steps"]["term_arbitration_codex_validation"]["status"] == "ready_for_import"
    assert result["steps"]["term_arbitration_codex"]["status"] == "imported"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["text"] == "今天横评 Playwright MCP 和 BrowserHarness。"
    manifest = read_json(root / "manifest.json")
    assert manifest["term_correction_closure_summary"]["semantic_review_status"] == "codex_or_llm_reviewed_import"
    args = read_json(root / "mcp-term-correction-closure.args.json")
    assert args["input_json"].endswith("term-arbitration-codex-result.codex.md")
    run = read_json(root / "runs" / "term-correction-closure" / "run.json")
    assert "--input-json" in run["retry_command"]

def test_term_correction_closure_stops_on_invalid_codex_markdown_response() -> None:
    from video_knowledge_pipeline.term_correction_closure import run_term_correction_closure

    root = Path("outputs/test-term-correction-closure-invalid-codex-import/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "invalid codex import lesson", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 5.0, "text": "今天横评 playright m c p。"}]})
    codex_response = root / "term-arbitration-codex-result.codex.md"
    codex_response.write_text(
        "Codex semantic decision with no accepted replacements:\n\n```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.term_arbitration_codex_result.v1",
                "source": "codex_reviewed_import",
                "decisions": [
                    {
                        "candidate_id": "term-1",
                        "canonical": "Playwright MCP",
                        "aliases": ["playright m c p", "Playwright MCP"],
                        "confidence": 0.51,
                        "action": "review",
                        "rationale": "Evidence is too weak; keep for human review.",
                        "evidence_indexes": [1],
                        "needs_human_review": True,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    result = run_term_correction_closure(root, input_json=codex_response)

    assert result["status"] == "needs_term_review"
    assert result["semantic_review_status"] == "codex_validation_failed"
    assert result["steps"]["term_arbitration_codex_validation"]["status"] == "no_accepted_decisions"
    assert result["steps"]["term_arbitration_codex"]["status"] == "skipped_validation_failed"
    assert result["accepted_validation_decisions"] == 0
    assert result["rejected_validation_decisions"] == 1
    assert not (root / "term-arbitration-glossary.json").exists()
    assert not (root / "source-arbitrated-transcript.json").exists()
    assert (root / "term-correction-closure.md").exists()
    manifest = read_json(root / "manifest.json")
    assert manifest["term_correction_closure_summary"]["term_validation_status"] == "no_accepted_decisions"
    args = read_json(root / "mcp-term-correction-closure.args.json")
    assert args["input_json"].endswith("term-arbitration-codex-result.codex.md")
    run = read_json(root / "runs" / "term-correction-closure" / "run.json")
    assert any(item["reason"] == "no_accepted_decisions" for item in run["failed_items"])


def test_term_correction_impact_report_measures_final_export_residuals() -> None:
    root = Path("outputs/test-term-correction-impact/bundle").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "exports").mkdir()
    write_json(
        root / "manifest.json",
        {
            "title": "browser automation lesson",
            "normalized_transcript_json": "normalized-transcript.json",
            "source_arbitrated_transcript_json": "source-arbitrated-transcript.json",
            "source_arbitrated_transcript_markdown": "source-arbitrated-transcript.md",
            "term_arbitration_glossary_json": "term-arbitration-glossary.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0.0, "end": 5.0, "text": "今天横评 playright m c p 和 brow harness。"}]})
    write_json(
        root / "term-arbitration-glossary.json",
        {
            "schema": "video_knowledge_pipeline.term_arbitration_glossary.v1",
            "terms": [
                {"canonical": "Playwright MCP", "aliases": ["playright m c p", "Playwright MCP"], "confidence": 0.96, "review_required": False},
                {"canonical": "BrowserHarness", "aliases": ["brow harness", "BrowserHarness"], "confidence": 0.96, "review_required": False},
            ],
        },
    )
    write_json(root / "source-arbitrated-transcript.json", {"segments": [{"start": 0.0, "end": 5.0, "text": "今天横评 Playwright MCP 和 BrowserHarness。"}]})
    (root / "source-arbitrated-transcript.md").write_text("今天横评 Playwright MCP 和 BrowserHarness。", encoding="utf-8")
    (root / "exports" / "full-transcript.md").write_text("今天横评 Playwright MCP 和 BrowserHarness。", encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text("# Smart Summary\n\n本视频横评 Playwright MCP 和 BrowserHarness。", encoding="utf-8")

    result = term_correction_impact_report(root)

    assert result["status"] == "passed"
    assert result["source_alias_total"] == 2
    assert result["final_export_alias_total"] == 0
    assert result["final_clean_rate"] == 1.0
    assert (root / "term-correction-impact-report.json").exists()
    assert (root / "term-correction-impact-report.md").exists()
    manifest = read_json(root / "manifest.json")
    assert manifest["term_correction_impact_summary"]["status"] == "passed"
