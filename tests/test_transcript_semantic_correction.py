from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.content_asset_status import content_asset_status
from video_knowledge_pipeline.review_session import prepare_review_session
from video_knowledge_pipeline.smart_summary_codex import smart_summary_quality_check
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_semantic_summary_impact import transcript_semantic_summary_impact_report
from video_knowledge_pipeline.transcript_semantic_correction import (
    _sidecar_evidence_for_cue,
    build_transcript_semantic_correction_codex_draft,
    import_transcript_semantic_candidate_suggestions,
    build_transcript_semantic_candidate_discovery_pack,
    build_transcript_semantic_candidate_discovery_llm_draft,
    build_transcript_semantic_candidate_discovery_codex_draft,
    build_transcript_semantic_correction_llm_draft,
    build_transcript_semantic_correction_pack,
    import_transcript_semantic_review_notes,
    transcript_semantic_correction_closure,
    transcript_semantic_correction_impact_report,
    transcript_semantic_correction_model_instructions,
    transcript_semantic_correction_output_contract,
    transcript_semantic_correction_readable_impact_report,
    transcript_semantic_correction_status,
    validate_transcript_semantic_correction,
    validate_transcript_semantic_model_output,
)


def _smart_summary_input_text(root: Path) -> str:
    payload = read_json(root / "exports" / "smart-summary-input-pack.json")
    return "\n".join(str(row.get("raw_text") or "") for row in payload.get("transcript_segments") or [])

def _llm_summary_text(text: str) -> str:
    return f"生成方式：`codex_llm_rewrite_final`\n\n{text}"

def _write_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "semantic correction fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0, "end": 4, "text": "今天讲 play right m c p 和 16k 底薪"},
                {"start": 5, "end": 9, "text": "然后点击登录并保存配置"},
            ]
        },
    )
    write_json(
        root / "timeline.json",
        [
            {"index": 0, "start": 0, "end": 4, "visual_text": "Playwright MCP 16k 底薪", "tags": ["工具名", "数字"]},
            {"index": 1, "start": 5, "end": 9, "visual_understanding": {"summary": "讲师演示点击登录按钮并保存配置"}},
        ],
    )
    (root / "exports" / "full-transcript.md").write_text("今天讲 play right m c p 和 16k 底薪", encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text("课程提到 play right m c p", encoding="utf-8")
    return root


def test_semantic_model_contract_rejects_gemini_shape_and_exact_text_drift(
    tmp_path: Path,
) -> None:
    root = _write_bundle(tmp_path)
    pack = build_transcript_semantic_correction_pack(root, write=False)
    candidate = next(row for row in pack["candidates"] if row.get("evidence"))
    malformed = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
        "source": "online_llm_review",
        "decisions": [
            {
                "candidate_id": candidate["candidate_id"],
                "original_text": f"{candidate['original_text']} changed",
                "accept": False,
                "needs_human_review": True,
                "reason": "uncertain",
            }
        ],
    }

    validation = validate_transcript_semantic_model_output(malformed, pack)

    assert validation["contract_ok"] is False
    assert validation["quality_gate_passed"] is False
    assert validation["rejected_decision_count"] == 1
    assert any(
        row["detail"] == "decisions[0].confidence"
        for row in validation["contract_issues"]
    )
    assert any(
        row["key"] == "original_text_mismatch"
        for row in validation["quality_issues"]
    )

    contract = transcript_semantic_correction_output_contract()
    assert contract["array_item_contracts"]["decisions"]["required_keys"][
        "evidence_ids"
    ] == "array"
    instructions = transcript_semantic_correction_model_instructions("Review candidates.")
    assert instructions.count("VKP_STRICT_TRANSCRIPT_SEMANTIC_CORRECTION_V1") == 1
    assert "Review candidates.\n\nVKP_STRICT_TRANSCRIPT_SEMANTIC_CORRECTION_V1" in instructions
    assert transcript_semantic_correction_model_instructions(instructions) == instructions

    supported = next(
        row for row in pack["candidates"] if row.get("original_text") == "play right"
    )
    valid = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
        "source": "online_llm_review",
        "decisions": [
            {
                "candidate_id": supported["candidate_id"],
                "action": "replace",
                "correction_type": supported["correction_type"],
                "original_text": supported["original_text"],
                "corrected_text": "Playwright",
                "confidence": 0.99,
                "rationale": "OCR and transcript context support the canonical tool name.",
                "evidence_ids": [
                    row["evidence_id"] for row in supported["evidence"]
                ],
                "human_confirmed": False,
                "needs_human_review": False,
            }
        ],
    }
    qualified = validate_transcript_semantic_model_output(valid, pack)
    assert qualified["contract_ok"] is True
    assert qualified["quality_gate_passed"] is True
    assert qualified["accepted_decision_count"] == 1



def test_domain_semantic_suspect_words_enter_correction_pack(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "insurance semantic suspects", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {
                    "start": 0,
                    "end": 12,
                    "text": "可以帮你做保单整理查缺补漏看看是否会有一些买虫的",
                },
                {
                    "start": 12,
                    "end": 24,
                    "text": "他采取了一个二则一的方式并不断展示自己的同意心",
                },
            ]
        },
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)

    rows = pack["candidates"]
    pairs = {(row.get("original_text"), row.get("candidate_text")) for row in rows}
    assert ("买虫的", "买重的") in pairs
    assert ("二则一", "二择一") in pairs or ("二则一的方式", "二择一的方式") in pairs
    assert ("同意心", "同理心") in pairs
    suspect_rows = [row for row in rows if row.get("reason") == "known_domain_semantic_suspect"]
    assert suspect_rows
    assert all(row["llm_review_eligible"] is True for row in suspect_rows)
    assert not (root / "source-arbitrated-transcript.json").exists()

def test_insurance_domain_variants_are_candidate_only(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "insurance terminology candidates",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {
                    "start": 0,
                    "end": 30,
                    "text": (
                        "\u5408\u4fdd\u4eba\u5458\u8bf4\u53ef\u80fd\u805a\u5b9d\uff0c\u5305\u4f53\u4ea7\u54c1\u53c8\u8981\u505c\u7626\uff1b"
                        "\u8ddf\u79d1\u6280\u8bb2\u4f4f\u4f60\u68c0\u548c\u84dd\u5c3e\u708e\u6848\u4f8b\u65f6\u4e0d\u8981\u8bef\u89e3\u4e3a\u805a\u8d54\uff0c"
                        "\u6700\u540e\u627e\u5230\u5ba2\u6237\u5174\u8d77\u70b9\u548c\u7537\u6027\u7684\u5b9d\u5b9d\u753b\u50cf\u3002"
                    ),
                }
            ]
        },
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)

    pairs = {
        (row.get("original_text"), row.get("candidate_text"))
        for row in pack["candidates"]
    }
    assert {
        ("\u5408\u4fdd\u4eba\u5458", "\u6838\u4fdd\u4eba\u5458"),
        ("\u805a\u5b9d", "\u62d2\u4fdd"),
        ("\u5305\u4f53", "\u6807\u4f53"),
        ("\u505c\u7626", "\u505c\u552e"),
        ("\u8ddf\u79d1\u6280\u8bb2", "\u8ddf\u5ba2\u6237\u8bb2"),
        ("\u4f4f\u4f60\u68c0", "\u91cd\u75be\u9669"),
        ("\u84dd\u5c3e\u708e", "\u9611\u5c3e\u708e"),
        ("\u805a\u8d54", "\u62d2\u8d54"),
        ("\u5174\u8d77\u70b9", "\u5174\u8da3\u70b9"),
        ("\u7537\u6027\u7684\u5b9d\u5b9d", "\u7537\u6027\u7684\u5b9d\u7238"),
    }.issubset(pairs)
    suspect_rows = [
        row
        for row in pack["candidates"]
        if row.get("reason") == "known_domain_semantic_suspect"
    ]
    assert suspect_rows
    assert all(row["candidate_only"] is True for row in suspect_rows)
    assert all(
        row["automatic_application_allowed"] is False
        for row in suspect_rows
    )
    assert all(row["needs_human_review"] is True for row in suspect_rows)
    assert not (root / "source-arbitrated-transcript.json").exists()



def test_semantic_pack_skips_corrected_transcript_json_when_loading_raw_cues(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "raw source drift", "transcript_json": "corrected-transcript.json"})
    write_json(root / "corrected-transcript.json", {"segments": [{"start": 0, "end": 12, "text": "看看是否会有一些买重的"}]})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 12, "text": "看看是否会有一些买虫的"}]})

    pack = build_transcript_semantic_correction_pack(root, write=True)

    pairs = {(row.get("original_text"), row.get("candidate_text")) for row in pack["candidates"]}
    assert ("买虫的", "买重的") in pairs


def test_domain_semantic_suspect_words_codex_draft_can_close(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "insurance semantic suspects", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0, "end": 12, "text": "看看是否会有一些买虫的"},
                {"start": 12, "end": 24, "text": "他采取了一个二则一的方式并不断展示自己的同意心"},
            ]
        },
    )

    build_transcript_semantic_correction_pack(root, write=True)
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert draft["status"] == "draft_ready"
    assert any(row["original_text"] == "买虫的" and row["corrected_text"] == "买重的" for row in draft["decisions"])
    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)
    assert validation["accepted_decision_count"] >= 2
    closure = transcript_semantic_correction_closure(root, input_json=root / "transcript-semantic-correction-result.codex.md", auto_apply=True, write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    text = "\n".join(row["text"] for row in corrected["segments"])
    assert "买重的" in text
    assert "二择一" in text
    assert "同理心" in text


def test_domain_semantic_suspect_normalizes_ok_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "appointment confirmation", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {"segments": [{"start": 0, "end": 8, "text": "客户是这样回的说嗯那明晚八点o我找一下我的保单"}]},
    )

    build_transcript_semantic_correction_pack(root, write=True)
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)

    assert any(row["original_text"] == "明晚八点o" and row["corrected_text"] == "明晚八点 OK" for row in draft["decisions"])
    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)
    assert validation["accepted_decision_count"] >= 1
    closure = transcript_semantic_correction_closure(root, input_json=root / "transcript-semantic-correction-result.codex.md", auto_apply=True, write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    text = "\n".join(row["text"] for row in corrected["segments"])
    assert "明晚八点 OK" in text
    assert "明晚八点o" not in text


def test_domain_semantic_suspect_normalizes_spaced_ok_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "bundle-spaced-ok"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "appointment confirmation", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {"segments": [{"start": 0, "end": 8, "text": "客户是这样回的说嗯那明晚八点 o 我找一下我的保单"}]},
    )

    build_transcript_semantic_correction_pack(root, write=True)
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)

    assert any(row["original_text"] == "明晚八点 o" and row["corrected_text"] == "明晚八点 OK" for row in draft["decisions"])


def test_semantic_candidate_discovery_pack_imports_suggestions_as_candidates_only(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    pack = build_transcript_semantic_correction_pack(root, write=True)
    before_count = pack["candidate_count"]

    discovery = build_transcript_semantic_candidate_discovery_pack(root, limit=5, write=True)

    assert discovery["status"] == "discovery_prompt_ready"
    assert discovery["segment_count"] > 0
    assert (root / "transcript-semantic-candidate-discovery-prompt.md").exists()
    assert (root / "transcript-semantic-candidate-discovery-template.json").exists()

    suggestion_path = root / "transcript-semantic-candidate-suggestions.codex.md"
    suggestion_path.write_text(
        "```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1",
                "source": "codex_candidate_discovery_test",
                "suggestions": [
                    {
                        "source_segment_index": 0,
                        "start": 0,
                        "end": 4,
                        "correction_type": "ordinary_word",
                        "original_text": "今天讲",
                        "candidate_text": "今天主要讲",
                        "reason": "上下文可能漏了主要二字，仅作为候选复核。",
                        "confidence": 0.61,
                        "evidence_summary": "ASR segment contains the source span; no direct application allowed.",
                        "needs_human_review": True,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    imported = import_transcript_semantic_candidate_suggestions(root, input_json=suggestion_path, write=True)

    assert imported["status"] == "imported"
    assert imported["imported_candidate_count"] == 1
    merged_pack = read_json(root / "transcript-semantic-correction-pack.json")
    assert merged_pack["candidate_count"] == before_count + 1
    imported_candidate = next(row for row in merged_pack["candidates"] if row.get("candidate_id") in imported["imported_candidate_ids"])
    assert imported_candidate["discovered_by"] == "codex_candidate_discovery_test"
    assert imported_candidate["needs_human_review"] is True
    assert imported_candidate["correction_type"] == "ordinary_word"
    assert not (root / "source-arbitrated-transcript.json").exists()


def test_semantic_candidate_discovery_codex_draft_writes_suggestions_only(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    build_transcript_semantic_correction_pack(root, write=True)

    result = build_transcript_semantic_candidate_discovery_codex_draft(root, limit=5, max_suggestions=5, write=True)

    assert result["status"] in {"codex_suggestions_ready", "no_safe_codex_suggestions"}
    assert result["segment_count"] > 0
    assert (root / "transcript-semantic-candidate-suggestions.codex.md").exists()
    assert (root / "transcript-semantic-candidate-suggestions.codex.json").exists()
    assert (root / "mcp-transcript-semantic-candidate-discovery-codex-draft.args.json").exists()
    assert not (root / "source-arbitrated-transcript.json").exists()
    if result["suggestion_count"]:
        payload = read_json(root / "transcript-semantic-candidate-suggestions.codex.json")
        assert payload["operator_boundary"]["suggestions_only"] is True
        imported = import_transcript_semantic_candidate_suggestions(root, input_json=root / "transcript-semantic-candidate-suggestions.codex.md", write=True)
        assert imported["suggestion_count"] == result["suggestion_count"]
        assert not (root / "source-arbitrated-transcript.json").exists()

def test_semantic_candidate_discovery_imports_segment_boundary_review_candidate(tmp_path: Path) -> None:
    root = tmp_path / "bundle-boundary"
    (root / "exports").mkdir(parents=True)
    long_text = "首先我们来看客户特点然后我们讲成交原则然后再看获取信任的动作接下来继续拆解陌生客户的沟通节奏最后总结整个流程怎么落地执行"
    write_json(root / "manifest.json", {"title": "boundary fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 36, "text": long_text}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 36, "transcript": long_text}])

    build_transcript_semantic_correction_pack(root, write=True)
    discovery = build_transcript_semantic_candidate_discovery_codex_draft(root, limit=5, max_suggestions=5, write=True)

    assert discovery["status"] == "codex_suggestions_ready"
    suggestions = read_json(root / "transcript-semantic-candidate-suggestions.codex.json")["suggestions"]
    boundary = next(row for row in suggestions if row["correction_type"] == "segment_boundary")
    expected_boundary_label = "\u3010\u5f85\u65ad\u53e5\u590d\u6838\u3011"
    assert boundary["candidate_text"] == expected_boundary_label
    assert boundary["needs_human_review"] is True

    imported = import_transcript_semantic_candidate_suggestions(root, input_json=root / "transcript-semantic-candidate-suggestions.codex.md", write=True)
    assert imported["imported_candidate_count"] >= 1
    pack = read_json(root / "transcript-semantic-correction-pack.json")
    imported_candidate = next(row for row in pack["candidates"] if row.get("correction_type") == "segment_boundary" and row.get("candidate_text") == expected_boundary_label)
    assert imported_candidate["risk_level"] == "high"
    assert imported_candidate["needs_human_review"] is True
    assert imported_candidate["candidate_text"] == expected_boundary_label

def test_semantic_candidate_discovery_high_confidence_visual_support_reaches_readable_outputs(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "semantic generic correction fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {"segments": [{"start": 0, "end": 6, "text": "今天讲客户信任流程这一页"}]},
    )
    write_json(
        root / "timeline.json",
        [
            {
                "index": 0,
                "start": 0,
                "end": 6,
                "visual_text": "客户信任建立流程",
                "ocr_evidence_level": "confirmed",
                "structured_visual": {
                    "title": "客户信任建立流程",
                    "evidence_level": "confirmed",
                },
                "tags": ["课程概念", "屏幕文字"],
            }
        ],
    )
    (root / "exports" / "full-transcript.md").write_text("今天讲客户信任流程这一页", encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text("本节讲客户信任流程这一页", encoding="utf-8")

    build_transcript_semantic_correction_pack(root, write=True)
    discovery = build_transcript_semantic_candidate_discovery_codex_draft(root, limit=5, max_suggestions=5, write=True)
    assert discovery["status"] == "codex_suggestions_ready"
    suggestions = read_json(root / "transcript-semantic-candidate-suggestions.codex.json")["suggestions"]
    assert any(row["candidate_text"] == "客户信任建立流程" and row["needs_human_review"] is False for row in suggestions)

    imported = import_transcript_semantic_candidate_suggestions(root, input_json=root / "transcript-semantic-candidate-suggestions.codex.md", write=True)
    assert imported["imported_candidate_count"] >= 1
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert any(row["corrected_text"] == "客户信任建立流程" for row in draft["decisions"])
    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)
    assert validation["accepted_decision_count"] >= 1
    closure = transcript_semantic_correction_closure(root, input_json=root / "transcript-semantic-correction-result.codex.md", refresh_exports=True, write=True)
    assert closure["status"] == "completed"
    assert closure["refresh_exports_status"] == "needs_summary_regeneration"
    assert closure["refresh_exports"]["requires_summary_regeneration"] is True
    assert closure["refresh_exports"]["semantic_acceptance_status"] != "accepted"

    corrected = read_json(root / "source-arbitrated-transcript.json")
    corrected_text = "\n".join(row["text"] for row in corrected["segments"])
    assert "客户信任建立流程" in corrected_text
    assert "今天讲客户信任流程这一页" not in corrected_text
    assert "客户信任建立流程" in (root / "exports" / "full-transcript.md").read_text(encoding="utf-8")
    assert "客户信任建立流程" in _smart_summary_input_text(root)
    readable = transcript_semantic_correction_readable_impact_report(root, write=False)
    assert readable["status"] == "passed"

def test_semantic_candidate_discovery_decisive_number_support_reaches_readable_outputs(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    raw = "\u8fd9\u4e2a\u5c97\u4f4d\u662f1k\u5e95\u85aa"
    corrected = "\u8fd9\u4e2a\u5c97\u4f4d\u662f16k\u5e95\u85aa"
    write_json(root / "manifest.json", {"title": "semantic number correction fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 6, "text": raw}]})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 0,
                "start": 0,
                "end": 6,
                "visual_text": "16k\u5e95\u85aa",
                "ocr_evidence_level": "confirmed",
                "structured_visual": {
                    "salary": "16k\u5e95\u85aa",
                    "evidence_level": "confirmed",
                },
                "tags": ["screen text", "salary"],
            }
        ],
    )
    (root / "exports" / "full-transcript.md").write_text(raw, encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text(raw, encoding="utf-8")

    build_transcript_semantic_correction_pack(root, write=True)
    discovery = build_transcript_semantic_candidate_discovery_codex_draft(root, limit=5, max_suggestions=5, write=True)
    assert discovery["status"] == "codex_suggestions_ready"
    suggestions = read_json(root / "transcript-semantic-candidate-suggestions.codex.json")["suggestions"]
    assert any(row["correction_type"] == "number" and row["original_text"] == "1k" and row["candidate_text"] == "16k" and row["needs_human_review"] is False for row in suggestions)

    imported = import_transcript_semantic_candidate_suggestions(root, input_json=root / "transcript-semantic-candidate-suggestions.codex.md", write=True)
    assert imported["suggestion_count"] >= 1
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert any(row["correction_type"] == "number" and row["original_text"] == "1k" and row["corrected_text"] == "16k" for row in draft["decisions"])
    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)
    assert validation["accepted_decision_count"] >= 1
    closure = transcript_semantic_correction_closure(root, input_json=root / "transcript-semantic-correction-result.codex.md", refresh_exports=True, write=True)
    assert closure["status"] == "completed"
    assert closure["refresh_exports_status"] == "needs_summary_regeneration"
    assert closure["refresh_exports"]["requires_summary_regeneration"] is True
    assert closure["refresh_exports"]["semantic_acceptance_status"] != "accepted"

    corrected_doc = read_json(root / "source-arbitrated-transcript.json")
    corrected_text = "\n".join(row["text"] for row in corrected_doc["segments"])
    assert corrected in corrected_text
    assert raw not in corrected_text
    assert corrected in (root / "exports" / "full-transcript.md").read_text(encoding="utf-8")
    assert corrected in _smart_summary_input_text(root)
    readable = transcript_semantic_correction_readable_impact_report(root, write=False)
    assert readable["status"] == "passed"

def test_semantic_candidate_discovery_decisive_action_support_reaches_readable_outputs(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    raw = "\u7136\u540e\u6253\u5f00\u767b\u5f55\u9875\u9762"
    corrected = "\u7136\u540e\u70b9\u51fb\u767b\u5f55\u9875\u9762"
    write_json(root / "manifest.json", {"title": "semantic action correction fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 6, "text": raw}]})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 0,
                "start": 0,
                "end": 6,
                "visual_understanding": {"summary": "\u8bb2\u5e08\u70b9\u51fb\u767b\u5f55\u6309\u94ae\u5e76\u4fdd\u5b58\u914d\u7f6e"},
                "temporal_visual_understanding": {"steps": ["\u70b9\u51fb\u767b\u5f55\u6309\u94ae", "\u4fdd\u5b58\u914d\u7f6e"]},
            }
        ],
    )
    (root / "exports" / "full-transcript.md").write_text(raw, encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text(raw, encoding="utf-8")

    build_transcript_semantic_correction_pack(root, write=True)
    discovery = build_transcript_semantic_candidate_discovery_codex_draft(root, limit=5, max_suggestions=5, write=True)
    assert discovery["status"] == "codex_suggestions_ready"
    suggestions = read_json(root / "transcript-semantic-candidate-suggestions.codex.json")["suggestions"]
    assert any(row["correction_type"] == "action" and row["original_text"] == "\u6253\u5f00" and row["candidate_text"] == "\u70b9\u51fb" and row["needs_human_review"] is False for row in suggestions)

    imported = import_transcript_semantic_candidate_suggestions(root, input_json=root / "transcript-semantic-candidate-suggestions.codex.md", write=True)
    assert imported["suggestion_count"] >= 1
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert any(row["correction_type"] == "action" and row["original_text"] == "\u6253\u5f00" and row["corrected_text"] == "\u70b9\u51fb" for row in draft["decisions"])
    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)
    assert validation["accepted_decision_count"] >= 1
    closure = transcript_semantic_correction_closure(root, input_json=root / "transcript-semantic-correction-result.codex.md", refresh_exports=True, write=True)
    assert closure["status"] == "completed"
    assert closure["refresh_exports_status"] == "needs_summary_regeneration"
    assert closure["refresh_exports"]["requires_summary_regeneration"] is True
    assert closure["refresh_exports"]["semantic_acceptance_status"] != "accepted"

    corrected_doc = read_json(root / "source-arbitrated-transcript.json")
    corrected_text = "\n".join(row["text"] for row in corrected_doc["segments"])
    assert corrected in corrected_text
    assert raw not in corrected_text
    assert corrected in (root / "exports" / "full-transcript.md").read_text(encoding="utf-8")
    assert corrected in _smart_summary_input_text(root)
    readable = transcript_semantic_correction_readable_impact_report(root, write=False)
    assert readable["status"] == "passed"

def test_correlated_subtitles_require_review_before_updating_readable_outputs(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    raw = "\u4eca\u5929\u8bb2playright"
    corrected = "\u4eca\u5929\u8bb2Playwright"
    write_json(
        root / "manifest.json",
        {
            "title": "subtitle arbitration fixture",
            "normalized_transcript_json": "normalized-transcript.json",
            "platform_subtitle_path": "platform-subtitle.json",
            "embedded_subtitle_path": "embedded-subtitle.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 6, "text": raw}]})
    write_json(root / "platform-subtitle.json", {"segments": [{"start": 0, "end": 6, "text": corrected}]})
    write_json(root / "embedded-subtitle.json", {"segments": [{"start": 0, "end": 6, "text": corrected}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 6}])
    (root / "exports" / "full-transcript.md").write_text(raw, encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text(raw, encoding="utf-8")

    pack = build_transcript_semantic_correction_pack(root, write=True)
    subtitle_candidate = next(row for row in pack["candidates"] if row.get("reason") == "subtitle_text_differs_from_transcript" and row.get("candidate_text") == "Playwright")
    assert set(subtitle_candidate["evidence_source_types"]) >= {"platform_subtitle", "embedded_subtitle"}
    support_summary = subtitle_candidate["source_support_summary"]
    assert set(support_summary["supports_candidate"]) >= {"platform_subtitle", "embedded_subtitle"}
    assert "asr_or_subtitle" in support_summary["supports_original"]
    assert support_summary["dominant_side"] == "conflict_needs_review"
    assert support_summary["lineage"]["independent_source_count"] == 0
    assert support_summary["needs_review_by_source_vote"] is True
    assert support_summary["candidate_weight"] > support_summary["original_weight"]
    assert support_summary["source_reliability"]["platform_subtitle"] == 50
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert any(row["original_text"] == "playright" and row["corrected_text"] == "Playwright" for row in draft["decisions"])
    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)
    assert validation["accepted_decision_count"] == 0
    assert not (root / "source-arbitrated-transcript.json").exists()
    # Explicit synthetic human review remains the established escape hatch;
    # two potentially copied subtitles are not automatic factual approval.
    reviewed = dict(draft)
    reviewed["decisions"] = [dict(row, human_confirmed=True) for row in draft["decisions"]]
    reviewed_path = root / "human-reviewed-result.json"
    write_json(reviewed_path, reviewed)
    validation = validate_transcript_semantic_correction(root, input_json=reviewed_path, write=True)
    assert validation["accepted_decision_count"] >= 1
    closure = transcript_semantic_correction_closure(root, input_json=reviewed_path, refresh_exports=True, write=True)
    assert closure["status"] == "completed"
    assert closure["refresh_exports_status"] == "needs_summary_regeneration"
    assert closure["refresh_exports"]["requires_summary_regeneration"] is True
    assert closure["refresh_exports"]["semantic_acceptance_status"] != "accepted"

    corrected_doc = read_json(root / "source-arbitrated-transcript.json")
    corrected_text = "\n".join(row["text"] for row in corrected_doc["segments"])
    assert corrected in corrected_text
    assert raw not in corrected_text
    assert corrected in (root / "exports" / "full-transcript.md").read_text(encoding="utf-8")
    assert corrected in _smart_summary_input_text(root)
    readable = transcript_semantic_correction_readable_impact_report(root, write=False)
    assert readable["status"] == "passed"

def test_platform_subtitle_does_not_override_strong_visual_original_support(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "exports").mkdir(parents=True)
    raw = "今天讲Playwright"
    wrong_subtitle = "今天讲Playright"
    write_json(
        root / "manifest.json",
        {
            "title": "subtitle opposition fixture",
            "normalized_transcript_json": "normalized-transcript.json",
            "platform_subtitle_path": "platform-subtitle.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 6, "text": raw}]})
    write_json(root / "platform-subtitle.json", {"segments": [{"start": 0, "end": 6, "text": wrong_subtitle}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 6, "visual_text": "Playwright"}])
    (root / "exports" / "full-transcript.md").write_text(raw, encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text(raw, encoding="utf-8")

    pack = build_transcript_semantic_correction_pack(root, write=True)
    subtitle_candidate = next(row for row in pack["candidates"] if row.get("reason") == "subtitle_text_differs_from_transcript")
    support_summary = subtitle_candidate["source_support_summary"]
    assert "platform_subtitle" in support_summary["supports_candidate"]
    assert "ocr" in support_summary["supports_original"]
    assert support_summary["has_source_conflict"] is True
    assert support_summary["dominant_side"] == "original"
    assert support_summary["original_weight"] > support_summary["candidate_weight"]
    assert support_summary["needs_review_by_source_vote"] is True

    status = transcript_semantic_correction_status(root, write=True)
    source_vote = status["source_vote_summary"]
    assert source_vote["needs_review_by_source_vote_count"] >= 1
    assert source_vote["by_dominant_side"]["original"] >= 1

    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert not any(row.get("corrected_text") == "Playright" for row in draft["decisions"])

def test_semantic_candidate_discovery_llm_draft_preview_writes_prompt_only(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    build_transcript_semantic_correction_pack(root, write=True)

    result = build_transcript_semantic_candidate_discovery_llm_draft(root, execute=False, limit=5, write=True)

    assert result["status"] == "planned"
    assert result["execute"] is False
    assert result["segment_count"] > 0
    assert result["suggestion_count"] == 0
    assert (root / "transcript-semantic-candidate-discovery-llm-prompt.md").exists()
    assert (root / "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json").exists()
    assert not (root / "transcript-semantic-candidate-suggestions.llm.json").exists()
    assert not (root / "source-arbitrated-transcript.json").exists()
def test_semantic_correction_codex_draft_applies_known_terms_across_timeline_fallback(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "browser automation terms"})
    write_json(
        root / "timeline.json",
        [
            {"index": 0, "start": 0, "end": 4, "transcript": "今天讲 playright m c p"},
            {"index": 1, "start": 4, "end": 8, "transcript": "继续比较 playright m c p 和 chrom"},
        ],
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)
    assert pack["status"] == "pack_ready"
    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert draft["status"] == "draft_ready"
    assert any(row["corrected_text"] == "Playwright" for row in draft["decisions"])
    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)
    assert validation["accepted_decision_count"] >= 2

    closure = transcript_semantic_correction_closure(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=True)

    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    texts = "\n".join(row["text"] for row in corrected["segments"])
    assert "Playwright MCP" in texts
    assert "Chrome" in texts
    assert "playright" not in texts
    assert "m c p" not in texts
    assert "chrom" not in texts



def test_semantic_closure_refresh_exports_updates_readable_outputs(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    pack = build_transcript_semantic_correction_pack(root, write=True)
    playright = next(row for row in pack["candidates"] if row["original_text"] == "play right")
    mcp = next(row for row in pack["candidates"] if row["original_text"] == "m c p")
    result_path = root / "transcript-semantic-correction-result.codex.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": playright["candidate_id"],
                    "segment_index": playright["segment_index"],
                    "action": "replace",
                    "original_text": "play right",
                    "corrected_text": "Playwright",
                    "correction_type": "proper_noun",
                    "confidence": 0.96,
                    "rationale": "OCR evidence and full context support the browser automation tool name.",
                    "evidence_ids": playright["evidence_ids"],
                },
                {
                    "candidate_id": mcp["candidate_id"],
                    "segment_index": mcp["segment_index"],
                    "action": "replace",
                    "original_text": "m c p",
                    "corrected_text": "MCP",
                    "correction_type": "proper_noun",
                    "confidence": 0.96,
                    "rationale": "OCR evidence and full context support the browser automation tool acronym.",
                    "evidence_ids": mcp["evidence_ids"],
                },
            ],
        },
    )

    closure = transcript_semantic_correction_closure(root, input_json=result_path, refresh_exports=True, write=True)

    full_transcript = (root / "exports" / "full-transcript.md").read_text(encoding="utf-8")
    full_body = (root / "exports" / "full-body.md").read_text(encoding="utf-8")
    smart_summary_input = _smart_summary_input_text(root)
    closure_report = (root / "transcript-semantic-correction-closure.md").read_text(encoding="utf-8")
    readable_impact = read_json(root / "transcript-semantic-readable-impact-report.json")

    assert closure["status"] == "completed"
    assert closure["refresh_exports_status"] == "needs_summary_regeneration"
    assert closure["refresh_exports"]["readable_impact_status"] == "passed"
    assert closure["refresh_exports"]["semantic_acceptance_status"] != "accepted"
    assert closure["refresh_exports"]["requires_summary_regeneration"] is True
    assert (
        closure["refresh_exports"]["canonical_transcript_integrity"]["passed"] is True
    )
    assert readable_impact["status"] == "passed"
    assert "Playwright MCP" in full_transcript
    assert "play right m c p" not in full_transcript
    assert "Playwright MCP" in full_body
    assert "play right m c p" not in full_body
    assert "00:00:" not in full_body
    assert "Playwright MCP" in smart_summary_input
    assert "play right m c p" not in smart_summary_input
    assert "Refresh exports status: `pending_downstream_refresh`" in closure_report

def test_semantic_correction_pack_groups_repeated_variants_by_canonical_hint(tmp_path: Path) -> None:
    root = tmp_path / "bundle-groups"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "browser tool variants", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0, "end": 4, "text": "这里对比 browser base 的能力"},
                {"start": 5, "end": 9, "text": "后面继续说 browse base 的登录态"},
            ]
        },
    )
    write_json(
        root / "timeline.json",
        [
            {"index": 0, "start": 0, "end": 4, "visual_text": "Browserbase dashboard"},
            {"index": 1, "start": 5, "end": 9, "visual_text": "Browserbase session"},
        ],
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)

    assert pack["candidate_group_count"] >= 1
    assert all(row.get("candidate_group_id") for row in pack["candidates"])
    assert all("canonical_hint" in row for row in pack["candidates"])
    browser_groups = [row for row in pack["candidate_groups"] if row.get("canonical_hint") == "Browserbase"]
    assert browser_groups
    group = browser_groups[0]
    assert group["candidate_count"] >= 2
    assert {"browser base", "browse base"} <= set(group["variant_texts"])
    assert "proper_noun" in group["correction_types"]
    assert "action" in group["correction_types"]
    assert "ocr" in group["evidence_source_types"]

    status = transcript_semantic_correction_status(root)
    assert status["candidate_group_count"] == pack["candidate_group_count"]
    assert any(row.get("canonical_hint") == "Browserbase" for row in status["candidate_group_preview"])


def test_semantic_correction_pack_validation_closure_and_impact(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)

    pack = build_transcript_semantic_correction_pack(root, write=True)
    assert pack["candidate_count"] >= 2
    assert (root / "transcript-semantic-correction-pack.json").exists()
    term_candidate = next(row for row in pack["candidates"] if row["correction_type"] in {"proper_noun", "term"})

    result_path = root / "transcript-semantic-correction-result.codex.md"
    result_path.write_text(
        "```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
                "decisions": [
                    {
                        "candidate_id": term_candidate["candidate_id"],
                        "accept": True,
                        "correction_type": term_candidate["correction_type"],
                        "original_text": term_candidate["original_text"],
                        "corrected_text": "MCP" if term_candidate["original_text"] == "m c p" else "Playwright MCP",
                        "confidence": 0.93,
                        "rationale": "OCR evidence shows the canonical tool name.",
                        "evidence_ids": term_candidate["evidence_ids"],
                        "human_confirmed": False,
                        "needs_human_review": False,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=True)
    assert validation["accepted_decision_count"] == 1
    closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["source"] == "transcript_semantic_correction"
    assert corrected["segments"][0]["changed"] is True
    manifest = read_json(root / "manifest.json")
    assert manifest["corrected_transcript_source"] == "transcript_semantic_correction"

    impact = transcript_semantic_correction_impact_report(root, write=True)
    assert impact["status"] == "needs_fix"
    assert impact["final_residual_error_total"] > 0
    failed_quality = smart_summary_quality_check(root, require_codex=False, write=False)
    failed_checks = {row["key"]: row for row in failed_quality["checks"]}
    assert failed_checks["transcript_semantic_correction_impact"]["passed"] is False
    assert failed_quality["transcript_semantic_correction_gate"]["status"] == "impact_needs_fix"

    (root / "exports" / "full-transcript.md").write_text(corrected["segments"][0]["text"], encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text(_llm_summary_text(corrected["segments"][0]["text"]), encoding="utf-8")
    impact = transcript_semantic_correction_impact_report(root, write=True)
    assert impact["status"] == "passed"
    status_before_readable = transcript_semantic_correction_status(root)
    assert status_before_readable["status"] == "needs_readable_impact_report"
    failed_readable_quality = smart_summary_quality_check(root, require_codex=False, write=False)
    assert failed_readable_quality["transcript_semantic_correction_gate"]["status"] == "needs_readable_impact_report"
    readable_impact = transcript_semantic_correction_readable_impact_report(root, write=True)
    assert readable_impact["status"] == "passed"
    status_before_summary = transcript_semantic_correction_status(root)
    assert status_before_summary["status"] == "needs_summary_impact_report"
    assert status_before_summary["next_action_key"] == "run_summary_impact"
    failed_summary_quality = smart_summary_quality_check(root, require_codex=False, write=False)
    assert failed_summary_quality["transcript_semantic_correction_gate"]["status"] == "needs_summary_impact_report"
    material_card = root / "exports" / "content-material-card.json"
    material_card_md = root / "exports" / "content-material-card.md"
    candidate_pack = root / "exports" / "content-candidate-pack.json"
    candidate_pack_md = root / "exports" / "content-candidate-pack.md"
    write_json(
        material_card,
        {
            "material_id": "semantic-gate-fixture",
            "source_path": str(root),
            "source_type": "video",
            "source_fact_status": "candidate",
            "evidence_tier": "transcript_evidence",
            "privacy_level": "private",
            "desensitized": True,
            "compliance_risk": "low",
            "fact_check_status": "needs_review",
            "target_layer": "content_asset",
            "publish_surface": "none",
            "content_stage": "draft",
            "cta_type": "none",
            "crm_followup_needed": False,
            "owner_thread": "video-knowledge-pipeline",
            "next_action": "review",
            "blocked_reason": "",
            "review_required": True,
            "publication_allowed": False,
            "allowed_as_inspiration": True,
            "allowed_as_fact": False,
            "circle_of_friends_status": "needs_review_inspiration",
        },
    )
    material_card_md.write_text("# material card\n", encoding="utf-8")
    write_json(candidate_pack, {"review_required": True, "publication_allowed": False, "allowed_as_inspiration": True, "allowed_as_fact": False, "candidate_count": 0, "candidates": []})
    candidate_pack_md.write_text("# candidate pack\n", encoding="utf-8")
    manifest = read_json(root / "manifest.json")
    manifest["content_assets"] = {
        "content_material_card_path": str(material_card),
        "content_material_card_markdown_path": str(material_card_md),
        "content_candidate_pack_path": str(candidate_pack),
        "content_candidate_pack_markdown_path": str(candidate_pack_md),
    }
    write_json(root / "manifest.json", manifest)
    asset_status_before_summary = content_asset_status(root, write=False)
    assert asset_status_before_summary["ok"] is False
    assert asset_status_before_summary["status"] == "semantic_correction_needs_action"
    assert asset_status_before_summary["semantic_correction_asset_gate"]["status"] == "needs_summary_impact_report"
    assert asset_status_before_summary["semantic_correction_summary_impact_status"] == "missing"
    assert "finish_transcript_semantic_correction_before_handoff" in asset_status_before_summary["next_actions"]
    summary_impact = transcript_semantic_summary_impact_report(root, write=True)
    assert summary_impact["status"] == "passed"
    semantic_status = transcript_semantic_correction_status(root)
    assert semantic_status["status"] == "impact_passed"
    assert semantic_status["closure_status"] == "completed"
    assert semantic_status["closure_applied_correction_count"] == 1
    assert semantic_status["corrected_transcript_exists"] is True
    assert semantic_status["summary_impact_status"] == "passed"
    assert semantic_status["summary_absorption_rate"] > 0
    ui_summary = semantic_status["ui_summary"]
    assert ui_summary["ui_state"] == "closed_and_export_checked"
    assert ui_summary["accepted_decision_count"] == 1
    assert ui_summary["applied_correction_count"] == 1
    assert ui_summary["applied_correction_type_counts"][term_candidate["correction_type"]] == 1
    assert ui_summary["evidence_source_counts"]
    assert ui_summary["export_chain"]["readable_impact_status"] == "passed"
    assert ui_summary["export_chain"]["summary_impact_status"] == "passed"
    asset_status_after_summary = content_asset_status(root, write=False)
    assert asset_status_after_summary["ok"] is True
    assert asset_status_after_summary["status"] == "ready_for_inspiration_review"
    assert asset_status_after_summary["semantic_correction_asset_gate"]["status"] == "passed"
    assert asset_status_after_summary["semantic_correction_summary_impact_status"] == "passed"
    assert asset_status_after_summary["semantic_correction_summary_residual_original_total"] == 0
    passed_quality = smart_summary_quality_check(root, require_codex=False, write=False)
    passed_checks = {row["key"]: row for row in passed_quality["checks"]}
    assert passed_checks["transcript_semantic_correction_impact"]["passed"] is True
    assert passed_quality["transcript_semantic_correction_gate"]["status"] == "impact_passed"



def test_semantic_summary_impact_report_proves_smart_summary_absorbed_correction(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    pack = build_transcript_semantic_correction_pack(root, write=True)
    term_candidate = next(row for row in pack["candidates"] if row["correction_type"] in {"proper_noun", "term"})
    result_path = root / "transcript-semantic-correction-result.codex.md"
    corrected_text = "MCP" if term_candidate["original_text"] == "m c p" else "Playwright MCP"
    result_path.write_text(
        "```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
                "decisions": [
                    {
                        "candidate_id": term_candidate["candidate_id"],
                        "accept": True,
                        "correction_type": term_candidate["correction_type"],
                        "original_text": term_candidate["original_text"],
                        "corrected_text": corrected_text,
                        "confidence": 0.93,
                        "rationale": "OCR evidence shows the canonical tool name.",
                        "evidence_ids": term_candidate["evidence_ids"],
                        "human_confirmed": False,
                        "needs_human_review": False,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    validate_transcript_semantic_correction(root, input_json=result_path, write=True)
    baseline = root / "exports" / "smart-summary.before.md"
    baseline.write_text(f"旧总结仍写成 {term_candidate['original_text']}。", encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text(_llm_summary_text(f"新总结已经写成 {corrected_text}。"), encoding="utf-8")

    report = transcript_semantic_summary_impact_report(root, baseline_summary_path=baseline, write=True)

    assert report["status"] == "passed"
    assert report["summary_residual_original_total"] == 0
    assert report["summary_corrected_hit_total"] >= 1
    assert report["baseline_residual_delta"] >= 1
    assert report["corrections"][0]["summary_absorption_proven"] is True
    assert (root / "transcript-semantic-summary-impact-report.json").exists()
    assert (root / "transcript-semantic-summary-impact-report.md").exists()



def test_semantic_summary_impact_does_not_treat_case_fixed_term_as_residual(tmp_path: Path) -> None:
    root = tmp_path / "bundle-case-summary"
    exports = root / "exports"
    exports.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "case summary absorption"})
    write_json(
        root / "transcript-semantic-correction-validation.json",
        {
            "accepted_decision_count": 1,
            "accepted_decisions": [
                {
                    "candidate_id": "semcorr-term-0001",
                    "action": "replace",
                    "correction_type": "term",
                    "original_text": "javascript",
                    "corrected_text": "JavaScript",
                    "confidence": 0.96,
                }
            ],
        },
    )
    baseline = exports / "smart-summary.before.md"
    baseline.write_text("旧总结把工具名写成 javascript。", encoding="utf-8")
    (exports / "smart-summary.md").write_text(_llm_summary_text("新总结明确写成 JavaScript，并用于浏览器自动化。"), encoding="utf-8")

    report = transcript_semantic_summary_impact_report(root, baseline_summary_path=baseline, write=True)

    assert report["status"] == "passed"
    assert report["summary_residual_original_total"] == 0
    assert report["summary_corrected_hit_total"] == 1
    assert report["baseline_residual_delta"] == 1
    row = report["corrections"][0]
    assert row["current_original_count"] == 0
    assert row["current_corrected_count"] == 1
    assert row["summary_absorption_proven"] is True
    assert row["sample_residual_lines"] == []

def test_semantic_summary_impact_accepts_corrected_context_absorption(tmp_path: Path) -> None:
    root = tmp_path / "bundle-context-summary"
    exports = root / "exports"
    exports.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "context summary absorption", "source_arbitrated_transcript_json": "source-arbitrated-transcript.json"})
    write_json(root / "timeline.json", [])
    write_json(
        root / "transcript-semantic-correction-validation.json",
        {
            "accepted_decision_count": 1,
            "accepted_decisions": [
                {
                    "candidate_id": "semcorr-boundary-0001",
                    "action": "replace",
                    "correction_type": "segment_boundary",
                    "original_text": "第一步先分析客户特点然后建立信任第二步确认需求",
                    "corrected_text": "第一步，分析客户特点。第二步，建立信任并确认需求。",
                    "confidence": 0.96,
                    "start": 0,
                    "end": 18,
                }
            ],
        },
    )
    write_json(
        root / "source-arbitrated-transcript.json",
        {
            "segments": [
                {
                    "start": 0,
                    "end": 18,
                    "text": "第一步，分析客户特点。第二步，建立信任并确认需求。",
                    "semantic_corrections": [
                        {
                            "candidate_id": "semcorr-boundary-0001",
                            "correction_type": "segment_boundary",
                            "application": "segment_split",
                        }
                    ],
                }
            ]
        },
    )
    (exports / "smart-summary.md").write_text(_llm_summary_text("这段课程把成交前置动作拆成两步：分析客户特点，并围绕建立信任和确认需求推进。"), encoding="utf-8")

    report = transcript_semantic_summary_impact_report(root, write=True)

    assert report["status"] == "passed"
    assert report["summary_corrected_hit_total"] == 0
    assert report["summary_context_keyword_hit_total"] >= 2
    assert report["summary_absorption_proven_count"] == 1
    row = report["corrections"][0]
    assert row["absorption_method"] == "corrected_context_keywords"
    assert "分析客户特点" in row["summary_context_keyword_hits"]
    assert "建立信任" in row["summary_context_keyword_hits"]
def test_semantic_correction_validation_rejects_risky_number_without_strong_evidence(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    pack = build_transcript_semantic_correction_pack(root, write=True)
    number_candidate = next(row for row in pack["candidates"] if row["correction_type"] == "number")
    result_path = root / "number-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": number_candidate["candidate_id"],
                    "accept": True,
                    "correction_type": "number",
                    "original_text": number_candidate["original_text"],
                    "corrected_text": "26000",
                    "confidence": 0.9,
                    "rationale": "Looks likely from context.",
                    "evidence_ids": number_candidate["evidence_ids"],
                    "human_confirmed": False,
                }
            ],
        },
    )
    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=False)
    assert validation["accepted_decision_count"] == 0
    assert "number_requires_stronger_evidence_or_human_confirmation" in validation["rejected_decisions"][0]["reject_reasons"]

def test_semantic_correction_validation_rejects_mislabeled_fact_value_change_without_strong_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle-fact-risk"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "fact risk fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "他说底薪是 16k"}]})

    pack = build_transcript_semantic_correction_pack(root, write=True)
    number_candidate = next(row for row in pack["candidates"] if row["correction_type"] == "number")
    result_path = root / "mislabeled-number-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": number_candidate["candidate_id"],
                    "accept": True,
                    "correction_type": "ordinary_word",
                    "original_text": "16k",
                    "corrected_text": "26000",
                    "confidence": 0.99,
                    "rationale": "The meaning appears to be the full salary number.",
                    "evidence_ids": number_candidate["evidence_ids"],
                    "human_confirmed": False,
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=False)

    assert validation["accepted_decision_count"] == 0
    rejected = validation["rejected_decisions"][0]
    reasons = rejected["reject_reasons"]
    assert rejected["high_risk_fact_change"] is True
    assert rejected["original_fact_values"] == ["16k"]
    assert rejected["corrected_fact_values"] == ["26000"]
    assert validation["review_rows"][0]["high_risk_fact_change"] is True
    assert validation["review_rows"][0]["corrected_fact_values"] == ["26000"]
    assert "fact_value_requires_stronger_evidence_or_human_confirmation" in reasons
    assert "unsafe_fact_value_without_strong_evidence" in reasons

def test_semantic_correction_pack_flags_ordinary_word_conflict_from_visual_text_without_subtitle(tmp_path: Path) -> None:
    root = tmp_path / "bundle-ordinary-visual-conflict"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "ordinary visual conflict fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 5, "text": "今天我们讲客户新任建立方法"}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 5, "visual_text": "客户信任建立方法"}])

    pack = build_transcript_semantic_correction_pack(root, write=True)

    candidate = next(row for row in pack["candidates"] if row.get("reason") == "ordinary_word_conflict_between_asr_and_visual_text")
    assert candidate["correction_type"] == "ordinary_word"
    assert candidate["original_text"] == "新"
    assert candidate["candidate_text"] == "信"
    assert candidate["has_conflict"] is True
    assert "ocr" in candidate["evidence_source_types"]


def test_semantic_correction_pack_flags_ordinary_word_conflict_between_asr_and_subtitle(tmp_path: Path) -> None:
    root = tmp_path / "bundle-ordinary-conflict"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "ordinary conflict fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 5, "text": "今天我们讲客户新任建立方法"}]})
    (root / "platform-subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:05,000\n今天我们讲客户信任建立方法\n", encoding="utf-8")
    write_json(root / "timeline.json", [])

    pack = build_transcript_semantic_correction_pack(root, write=True)

    candidate = next(row for row in pack["candidates"] if row.get("reason") == "ordinary_word_conflict_between_asr_and_subtitle")
    assert candidate["correction_type"] == "ordinary_word"
    assert candidate["original_text"] == "新"
    assert candidate["candidate_text"] == "信"
    assert candidate["has_conflict"] is True
    assert "platform_subtitle" in candidate["evidence_source_types"]


def test_semantic_correction_pack_keeps_numeric_subtitle_conflict_as_number(tmp_path: Path) -> None:
    root = tmp_path / "bundle-number-subtitle-conflict"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "number subtitle conflict fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "他说底薪是一万六千块"}]})
    (root / "platform-subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:04,000\n他说底薪是 16k\n", encoding="utf-8")
    write_json(root / "timeline.json", [])

    pack = build_transcript_semantic_correction_pack(root, write=True)

    assert any(row["correction_type"] == "number" and row.get("candidate_text") == "16k" for row in pack["candidates"])
    assert not any(row.get("reason") == "ordinary_word_conflict_between_asr_and_subtitle" for row in pack["candidates"])


def test_semantic_correction_pack_flags_chinese_fact_values_and_visual_numeric_conflict(tmp_path: Path) -> None:
    root = tmp_path / "bundle-chinese-number"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "chinese number fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "他说底薪是一万六千块"}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 4, "visual_text": "16k 底薪"}])

    pack = build_transcript_semantic_correction_pack(root, write=True)

    number_candidates = [row for row in pack["candidates"] if row["correction_type"] == "number"]
    assert any(row["original_text"] == "一万六千块" and row["reason"] == "contains_number_or_amount" for row in number_candidates)
    conflict = next(row for row in number_candidates if row.get("candidate_text") == "16k")
    assert conflict["original_text"] == "一万六千块"
    assert conflict["reason"] == "visual_text_differs_from_transcript"
    assert conflict["needs_human_review"] is True


def test_semantic_correction_pack_flags_platform_subtitle_as_conflict_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle-subtitle-conflict"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "subtitle conflict fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "今天用 open client 做自动化"}]})
    (root / "platform-subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:04,000\n今天用 OpenClaw 做自动化\n", encoding="utf-8")
    write_json(root / "timeline.json", [])

    pack = build_transcript_semantic_correction_pack(root, write=True)

    conflict = next(row for row in pack["candidates"] if row.get("reason") == "subtitle_text_differs_from_transcript")
    assert conflict["correction_type"] == "proper_noun"
    assert conflict["original_text"] == "open client"
    assert conflict["candidate_text"] == "OpenClaw"
    assert "platform_subtitle" in conflict["evidence_source_types"]


def test_semantic_correction_validation_rejects_action_change_without_visual_or_human_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle-action-risk"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "action risk fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "然后点击登录并保存配置"}]})

    pack = build_transcript_semantic_correction_pack(root, write=True)
    action_candidate = next(row for row in pack["candidates"] if row["correction_type"] == "action")
    result_path = root / "action-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": action_candidate["candidate_id"],
                    "accept": True,
                    "correction_type": "action",
                    "original_text": "点击登录",
                    "corrected_text": "点击注册",
                    "confidence": 0.97,
                    "rationale": "The tutorial context may be about account creation.",
                    "evidence_ids": action_candidate["evidence_ids"],
                    "human_confirmed": False,
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=False)

    assert validation["accepted_decision_count"] == 0
    rejected = validation["rejected_decisions"][0]
    assert rejected["high_risk_action_change"] is True
    assert rejected["original_action_values"] == ["点击", "登录"]
    assert rejected["corrected_action_values"] == ["点击", "注册"]
    assert "action_change_requires_visual_temporal_or_human_confirmation" in rejected["reject_reasons"]
    assert "unsafe_action_without_visual_or_human_evidence" in rejected["reject_reasons"]
    assert validation["review_rows"][0]["high_risk_action_change"] is True


def test_semantic_correction_validation_accepts_action_change_with_visual_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle-action-visual"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "action visual fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "然后点击登录并保存配置"}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 4, "visual_understanding": {"summary": "讲师演示点击注册按钮，然后保存配置"}}])

    pack = build_transcript_semantic_correction_pack(root, write=True)
    action_candidate = next(row for row in pack["candidates"] if row["correction_type"] == "action")
    visual_evidence_ids = [row["evidence_id"] for row in action_candidate["evidence"] if row.get("source_type") == "visual_understanding"]
    result_path = root / "action-visual-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": action_candidate["candidate_id"],
                    "accept": True,
                    "correction_type": "action",
                    "original_text": "点击登录",
                    "corrected_text": "点击注册",
                    "confidence": 0.97,
                    "rationale": "The visual evidence explicitly says the instructor clicks the register button.",
                    "evidence_ids": visual_evidence_ids,
                    "human_confirmed": False,
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=False)

    assert validation["accepted_decision_count"] == 1
    accepted = validation["accepted_decisions"][0]
    assert accepted["high_risk_action_change"] is True
    assert accepted["corrected_action_values"] == ["点击", "注册"]

def test_semantic_correction_validation_tracks_llm_confirmed_no_change_without_rejection(tmp_path: Path) -> None:
    root = tmp_path / "bundle-no-change"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "no change fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "他说要沟通二十分钟"}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 4, "visual_text": "沟通二十分钟"}])

    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(row for row in pack["candidates"] if row["correction_type"] == "number")
    result_path = root / "no-change-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "accept": False,
                    "correction_type": candidate["correction_type"],
                    "original_text": candidate["original_text"],
                    "corrected_text": candidate["original_text"],
                    "confidence": 0.95,
                    "rationale": "OCR 和 ASR 都显示二十分钟，确认无需修改。",
                    "evidence_ids": candidate["evidence_ids"],
                    "needs_human_review": False,
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=True)

    assert validation["status"] == "arbitrated_no_change"
    assert validation["ok"] is True
    assert validation["accepted_decision_count"] == 0
    assert validation["arbitrated_no_change_count"] == 1
    assert validation["rejected_decision_count"] == 0
    assert validation["review_required_count"] == 0
    assert validation["arbitrated_no_change_decisions"][0]["semantic_decision_status"] == "arbitrated_no_change"

    closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)
    assert closure["status"] == "completed_no_text_changes"
    assert closure["ok"] is True
    assert closure["arbitrated_no_change_count"] == 1


def test_semantic_correction_llm_selection_defers_low_evidence_number_candidates(tmp_path: Path) -> None:
    root = tmp_path / "bundle-llm-eligibility"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "eligibility fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0, "end": 4, "text": "他说沟通二十分钟"},
                {"start": 5, "end": 9, "text": "今天我们讲客户新任建立方法"},
            ]
        },
    )
    write_json(root / "timeline.json", [{"index": 1, "start": 5, "end": 9, "visual_text": "客户信任建立方法"}])

    pack = build_transcript_semantic_correction_pack(root, write=True)
    number_candidate = next(row for row in pack["candidates"] if row.get("reason") == "contains_number_or_amount")
    conflict_candidate = next(row for row in pack["candidates"] if row.get("reason") == "ordinary_word_conflict_between_asr_and_visual_text")

    assert number_candidate["llm_review_eligible"] is False
    assert number_candidate["llm_review_defer_reason"] == "needs_conflicting_external_evidence"
    assert conflict_candidate["llm_review_eligible"] is True

    result = build_transcript_semantic_correction_llm_draft(root, execute=False, limit=10, write=True)
    prompt = (root / "transcript-semantic-correction-llm-prompt.md").read_text(encoding="utf-8")

    assert result["candidate_selection"]["strategy"] == "source_conflict_first"
    assert result["candidate_selection"]["eligible_candidate_count"] >= 1
    assert result["candidate_selection"]["deferred_low_evidence_candidate_count"] >= 1
    assert conflict_candidate["candidate_id"] in result["candidate_selection"]["selected_candidate_ids"]
    assert number_candidate["candidate_id"] not in result["candidate_selection"]["selected_candidate_ids"]
    assert "客户信任建立方法" in prompt
    assert "二十分钟" not in prompt


def test_semantic_correction_pack_flags_deictic_transcript_with_visual_concept(tmp_path: Path) -> None:
    root = tmp_path / "bundle-concept-gap"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "concept gap fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 5, "text": "这里这个很重要大家看一下"}]})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 0,
                "start": 0,
                "end": 5,
                "visual_text": "客户信任建立流程：确认需求，给出解决方案",
                "tagger_tags": ["步骤", "概念"],
                "tagger_annotations": [{"text": "重点概念：客户信任建立流程"}],
            }
        ],
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)

    concept = next(row for row in pack["candidates"] if row["reason"] == "deictic_or_low_information_transcript_with_support_concept")
    assert concept["correction_type"] == "concept"
    assert concept["risk_level"] == "medium"
    assert "客户信任建立流程" in concept["candidate_text"]
    assert {"ocr", "tagger"} <= set(concept["evidence_source_types"])
    assert concept["has_conflict"] is True
    status = transcript_semantic_correction_status(root, write=True)
    attention = status["semantic_attention_preview"]
    assert attention
    assert attention[0]["candidate_id"] == concept["candidate_id"]
    assert attention[0]["correction_type"] == "concept"
    assert "客户信任建立流程" in attention[0]["suggested_text"]
    assert attention[0]["priority_score"] > 0


def test_semantic_correction_pack_uses_subtitle_metadata_tagger_and_visual_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "Browserbase 与 Playwright MCP 横评",
            "description": "本视频对比 Browserbase、Playwright MCP 和 Chrome DevTools。",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 6, "text": "今天讲 browser base 和 play right m c p"}]})
    (root / "platform-subtitle.srt").write_text(
        "1\n00:00:00,000 --> 00:00:06,000\n今天讲 Browserbase 和 Playwright MCP\n",
        encoding="utf-8",
    )
    write_json(
        root / "timeline.json",
        [
            {
                "index": 0,
                "start": 0,
                "end": 6,
                "visual_text": "Browserbase Playwright MCP",
                "tagger_tags": ["工具名", "横评"],
                "tagger_annotations": [{"time": 1.2, "tags": ["工具名"], "text": "画面出现 Browserbase 和 Playwright MCP"}],
                "integrated_visual": {"tagger_visual_summary": "工具名对比页"},
            }
        ],
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)

    assert pack["candidate_count"] >= 1
    source_types = {source for row in pack["candidates"] for source in row.get("evidence_source_types", [])}
    assert {"platform_subtitle", "page_metadata", "tagger", "ocr"} <= source_types
    reasons = {row.get("reason") for row in pack["candidates"]}
    assert {"subtitle_text_differs_from_transcript", "visual_text_differs_from_transcript"} & reasons
    assert any(row.get("candidate_text") in {"Browserbase", "Playwright"} or "Browser" in row.get("candidate_text", "") for row in pack["candidates"])



def test_semantic_correction_pack_falls_back_to_timeline_and_flags_mojibake(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "timeline-only mojibake"})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 4,
                "transcript": "��� playright m c p ���",
                "visual_text": "Playwright MCP",
            }
        ],
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)

    assert pack["status"] == "pack_ready"
    assert pack["candidate_count"] >= 1
    mojibake = next(row for row in pack["candidates"] if row["reason"] == "transcript_text_mojibake_or_decoding_error")
    assert mojibake["risk_level"] == "high"
    assert mojibake["needs_human_review"] is True
    assert "ocr" in mojibake["evidence_source_types"]

def test_semantic_correction_validation_requires_known_evidence_and_marks_conflict(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "semantic correction conflict", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "今天讲 browser base"}]})
    (root / "platform-subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:04,000\n今天讲 Browserbase\n", encoding="utf-8")
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 4, "visual_text": "Browserbase"}])
    pack = build_transcript_semantic_correction_pack(root, write=True)
    conflict_candidate = next(row for row in pack["candidates"] if row.get("has_conflict"))

    bad_result = root / "bad-result.json"
    write_json(
        bad_result,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": conflict_candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": conflict_candidate["correction_type"],
                    "original_text": conflict_candidate["original_text"],
                    "corrected_text": "Browserbase",
                    "confidence": 0.9,
                    "semantic_rationale": "字幕和画面均显示 Browserbase。",
                    "evidence_ids": ["not-in-pack"],
                    "safe_to_apply": True,
                    "needs_human_review": False,
                }
            ],
        },
    )
    bad_validation = validate_transcript_semantic_correction(root, input_json=bad_result, write=False)
    bad_reasons = bad_validation["rejected_decisions"][0]["reject_reasons"]
    assert "unknown_evidence_ids" in bad_reasons
    assert "conflict_not_marked_for_review" in bad_reasons

    good_result = root / "good-result.json"
    write_json(
        good_result,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": conflict_candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": conflict_candidate["correction_type"],
                    "original_text": conflict_candidate["original_text"],
                    "corrected_text": "Browserbase",
                    "confidence": 0.96,
                    "semantic_rationale": "平台字幕和画面文字共同支持 Browserbase。",
                    "evidence_ids": conflict_candidate["evidence_ids"],
                    "safe_to_apply": True,
                    "needs_human_review": False,
                }
            ],
        },
    )
    good_validation = validate_transcript_semantic_correction(root, input_json=good_result, write=False)
    assert good_validation["accepted_decision_count"] == 1
    assert good_validation["accepted_decisions"][0]["semantic_rationale"]


def test_semantic_correction_rejected_decisions_flow_to_review_session_and_status(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "semantic correction review", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "今天讲 browser base"}]})
    (root / "platform-subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:04,000\n今天讲 Browserbase\n", encoding="utf-8")
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 4, "transcript": "今天讲 browser base", "visual_text": "Browserbase"}])
    (root / "exports").mkdir(exist_ok=True)
    write_json(root / "exports" / "smart-summary-chapters.json", {"chapters": [{"index": 1, "title": "浏览器自动化工具", "start": 0, "end": 10, "start_time": "00:00:00.000", "end_time": "00:00:10.000"}]})
    (root / "review.html").write_text("<html></html>", encoding="utf-8")
    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(row for row in pack["candidates"] if row.get("has_conflict"))
    result_path = root / "semantic-review-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": candidate["correction_type"],
                    "original_text": candidate["original_text"],
                    "corrected_text": "Browserbase",
                    "confidence": 0.5,
                    "semantic_rationale": "证据冲突且置信不足，需要人工确认。",
                    "evidence_ids": candidate["evidence_ids"],
                    "safe_to_apply": True,
                    "needs_human_review": False,
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=True)

    assert validation["accepted_decision_count"] == 0
    assert validation["review_required_count"] == 1
    review_payload = read_json(root / "transcript-semantic-correction-review.json")
    assert review_payload["review_required_count"] == 1
    assert review_payload["items"][0]["target_type"] == "transcript_semantic_correction"
    status = transcript_semantic_correction_status(root, write=True)
    assert status["review_required_count"] == 1
    assert status["candidate_type_counts"].get("term", 0) >= 1 or status["candidate_type_counts"].get("proper_noun", 0) >= 1
    assert status["evidence_source_counts"]["ocr"] >= 1
    assert status["validation_rejection_reason_counts"]["confidence_below_threshold"] == 1
    assert status["review_required_items"][0]["candidate_id"] == candidate["candidate_id"]
    assert status["review_required_items"][0]["chapter_index"] == 1
    assert status["chapter_risk_summary"][0]["chapter_title"] == "浏览器自动化工具"
    assert status["chapter_risk_summary"][0]["review_required_count"] == 1
    assert status["review_required_preview"][0]["candidate_id"] == candidate["candidate_id"]
    asset_status = content_asset_status(root, write=False)
    assert asset_status["semantic_correction_review_count"] == 1
    assert asset_status["semantic_correction_chapter_risk_summary"][0]["chapter_index"] == 1

    session = prepare_review_session(root, limit=0, include_closed=False)
    targets = session["review_targets"]["items"]
    semantic_targets = [row for row in targets if row.get("target_type") == "transcript_semantic_correction"]
    assert len(semantic_targets) == 1
    assert semantic_targets[0]["transcript_semantic_correction"]["candidate_id"] == candidate["candidate_id"]


def test_import_transcript_semantic_review_notes_promotes_human_confirmed_decision(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "semantic review import", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "今天讲 browser base"}]})
    (root / "platform-subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:04,000\n今天讲 Browserbase\n", encoding="utf-8")
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 4, "transcript": "今天讲 browser base", "visual_text": "Browserbase"}])
    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(row for row in pack["candidates"] if row.get("has_conflict"))
    review_notes = root / "semantic-review-notes.json"
    write_json(
        review_notes,
        {
            "schema": "lecture_review_notes.v1",
            "reviews": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": "accept_correction",
                    "corrected_text": "Browserbase",
                    "comment": "人工看视频和画面后确认工具名为 Browserbase。",
                    "evidence_ids": candidate["evidence_ids"],
                },
                {
                    "candidate_id": "missing-candidate",
                    "status": "accept_correction",
                    "corrected_text": "MissingTool",
                    "comment": "这条应该被跳过，因为候选不存在。",
                }
            ],
        },
    )

    imported = import_transcript_semantic_review_notes(root, review_json=review_notes, write=True)

    assert imported["decision_count"] == 1
    assert imported["skipped_count"] == 1
    imported_result = read_json(Path(imported["result_json"]))
    assert imported_result["import_summary"]["decision_count"] == 1
    assert imported_result["import_summary"]["skipped_count"] == 1
    assert imported_result["import_summary"]["skipped"][0]["candidate_id"] == "missing-candidate"
    assert Path(imported["result_json"]).exists()
    assert imported["validation"]["accepted_decision_count"] == 1
    closure = transcript_semantic_correction_closure(root, input_json=imported["result_json"], write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["text"] == "今天讲 Browserbase"
    manifest = read_json(root / "manifest.json")
    assert manifest["transcript_semantic_correction_result_review_json"] == "transcript-semantic-correction-result.review.json"
    assert manifest["mcp_import_transcript_semantic_review_notes_args"] == "mcp-import-transcript-semantic-review-notes.args.json"


def test_import_transcript_semantic_review_notes_preserves_split_segments(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    raw_text = "第一步先分析客户特点然后建立信任第二步再确认需求第三步处理异议最后总结成交原则这些内容需要拆开否则后续动作容易混淆"
    write_json(root / "manifest.json", {"title": "semantic split review import", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 10, "end": 42, "text": raw_text}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 10, "end": 42, "transcript": raw_text, "tagger_tags": ["步骤"]}])
    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(row for row in pack["candidates"] if row["correction_type"] in {"punctuation", "segment_boundary"})
    review_notes = root / "semantic-split-review-notes.json"
    write_json(
        review_notes,
        {
            "schema": "lecture_review_notes.v1",
            "reviews": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": "accept_correction",
                    "correction_type": "segment_boundary",
                    "comment": "人工确认这一条 ASR 实际包含两个步骤，应拆成两个可读时间段。",
                    "evidence_ids": candidate["evidence_ids"],
                    "segments": [
                        {"start": 10, "end": 16, "text": "第一步，先分析客户特点，然后建立信任。"},
                        {"start": 16, "end": 42, "text": "第二步，再确认需求。第三步，处理异议。最后，总结成交原则。"},
                    ],
                }
            ],
        },
    )

    imported = import_transcript_semantic_review_notes(root, review_json=review_notes, write=True)

    assert imported["decision_count"] == 1
    imported_result = read_json(Path(imported["result_json"]))
    assert imported_result["decisions"][0]["segments"][0]["text"].startswith("第一步")
    assert imported["validation"]["accepted_decision_count"] == 1
    closure = transcript_semantic_correction_closure(root, input_json=imported["result_json"], write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["summary"]["segments"] == 2
    assert corrected["segments"][0]["semantic_corrections"][0]["application"] == "segment_split"


def test_import_transcript_semantic_review_notes_preserves_merge_segment_indexes(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    first = "第一步先分析客户特点"
    second = "然后建立信任"
    write_json(root / "manifest.json", {"title": "semantic merge review import", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 10, "end": 13, "text": first}, {"start": 13, "end": 16, "text": second}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 10, "end": 13, "transcript": first, "tagger_tags": ["步骤"]}, {"index": 1, "start": 13, "end": 16, "transcript": second, "tagger_tags": ["步骤"]}])
    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = pack["candidates"][0]
    review_notes = root / "semantic-merge-review-notes.json"
    write_json(
        review_notes,
        {
            "schema": "lecture_review_notes.v1",
            "reviews": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": "accept_correction",
                    "correction_type": "segment_boundary",
                    "corrected_text": "第一步，先分析客户特点，然后建立信任。",
                    "comment": "人工确认两个 ASR 短段实际是同一个步骤，应合并为一个语义段。",
                    "evidence_ids": candidate["evidence_ids"],
                    "merge_segment_indexes": [0, 1],
                }
            ],
        },
    )

    imported = import_transcript_semantic_review_notes(root, review_json=review_notes, write=True)

    assert imported["decision_count"] == 1
    imported_result = read_json(Path(imported["result_json"]))
    assert imported_result["decisions"][0]["merge_segment_indexes"] == [0, 1]
    assert imported["validation"]["accepted_decision_count"] == 1
    closure = transcript_semantic_correction_closure(root, input_json=imported["result_json"], write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["summary"]["segments"] == 1
    assert corrected["segments"][0]["semantic_corrections"][0]["application"] == "segment_merge"
def test_semantic_correction_punctuation_candidate_and_whole_segment_closure(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    raw_text = "第一步先分析客户特点然后建立信任第二步再确认需求第三步处理异议最后总结成交原则这些内容需要连起来理解否则后面的执行动作会被误解"
    corrected_text = "第一步，先分析客户特点，然后建立信任。第二步，再确认需求。第三步，处理异议。最后，总结成交原则。这些内容需要连起来理解，否则后面的执行动作会被误解。"
    write_json(root / "manifest.json", {"title": "punctuation fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 10, "end": 42, "text": raw_text}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 10, "end": 42, "transcript": raw_text, "tagger_tags": ["步骤", "结论"]}])

    pack = build_transcript_semantic_correction_pack(root, write=True)

    candidate = next(row for row in pack["candidates"] if row["correction_type"] in {"punctuation", "segment_boundary"})
    assert candidate["needs_human_review"] is True
    assert candidate["original_text"] == raw_text
    result_path = root / "punctuation-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": candidate["correction_type"],
                    "original_text": candidate["original_text"],
                    "corrected_text": corrected_text,
                    "confidence": 0.98,
                    "semantic_rationale": "人工根据完整语义确认原 ASR 段落缺少标点和边界，需要整段改写为可读句子。",
                    "evidence_ids": candidate["evidence_ids"],
                    "safe_to_apply": True,
                    "needs_human_review": False,
                    "human_confirmed": True,
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=True)
    assert validation["accepted_decision_count"] == 1
    closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    segment = corrected["segments"][0]
    assert segment["text"] == corrected_text
    assert segment["raw_text"] == raw_text
    assert segment["structure_changed"] is True
    assert segment["semantic_corrections"][0]["application"] == "whole_segment_text"


def test_semantic_correction_segment_boundary_can_split_one_asr_segment(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    raw_text = "第一步先分析客户特点然后建立信任第二步再确认需求第三步处理异议最后总结成交原则这些内容需要拆开否则后续动作容易混淆"
    write_json(root / "manifest.json", {"title": "segment split fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 10, "end": 42, "text": raw_text}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 10, "end": 42, "transcript": raw_text, "tagger_tags": ["步骤"]}])

    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(row for row in pack["candidates"] if row["correction_type"] in {"punctuation", "segment_boundary"})
    result_path = root / "segment-split-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": "segment_boundary",
                    "original_text": candidate["original_text"],
                    "confidence": 0.98,
                    "semantic_rationale": "人工确认这一条 ASR 实际包含两个步骤，应拆成两个可读时间段。",
                    "evidence_ids": candidate["evidence_ids"],
                    "safe_to_apply": True,
                    "human_confirmed": True,
                    "segments": [
                        {"start": 10, "end": 16, "text": "第一步，先分析客户特点，然后建立信任。"},
                        {"start": 16, "end": 42, "text": "第二步，再确认需求。第三步，处理异议。最后，总结成交原则。"},
                    ],
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=True)
    assert validation["accepted_decision_count"] == 1
    assert validation["accepted_decisions"][0]["segments"][0]["text"].startswith("第一步")
    closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["summary"]["segments"] == 2
    assert corrected["segments"][0]["text"] == "第一步，先分析客户特点，然后建立信任。"
    assert corrected["segments"][1]["text"].startswith("第二步")
    assert "最后" in corrected["segments"][1]["text"]
    assert corrected["segments"][0]["source_segment_index"] == 0
    assert corrected["segments"][0]["semantic_corrections"][0]["application"] == "segment_split"
    assert corrected["segments"][0]["semantic_corrections"][0]["split_segment_count"] == 2


def test_semantic_correction_segment_boundary_can_merge_multiple_asr_segments(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    first = "第一步先分析客户特点"
    second = "然后建立信任"
    write_json(root / "manifest.json", {"title": "segment merge fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 10, "end": 13, "text": first}, {"start": 13, "end": 16, "text": second}]})
    write_json(root / "timeline.json", [{"index": 0, "start": 10, "end": 13, "transcript": first, "tagger_tags": ["步骤"]}, {"index": 1, "start": 13, "end": 16, "transcript": second, "tagger_tags": ["步骤"]}])

    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = pack["candidates"][0]
    result_path = root / "segment-merge-result.json"
    write_json(
        result_path,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": "segment_boundary",
                    "original_text": f"{first} {second}",
                    "corrected_text": "第一步，先分析客户特点，然后建立信任。",
                    "confidence": 0.98,
                    "semantic_rationale": "人工确认两个 ASR 短段实际是同一个步骤，应合并为一个语义段。",
                    "evidence_ids": candidate["evidence_ids"],
                    "safe_to_apply": True,
                    "human_confirmed": True,
                    "merge_segment_indexes": [0, 1],
                }
            ],
        },
    )

    validation = validate_transcript_semantic_correction(root, input_json=result_path, write=True)
    assert validation["accepted_decision_count"] == 1
    assert validation["accepted_decisions"][0]["merge_segment_indexes"] == [0, 1]
    closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)
    assert closure["status"] == "completed"
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["summary"]["segments"] == 1
    assert corrected["segments"][0]["text"] == "第一步，先分析客户特点，然后建立信任。"
    assert corrected["segments"][0]["source_segment_indexes"] == [0, 1]
    assert corrected["segments"][0]["semantic_corrections"][0]["application"] == "segment_merge"
    assert corrected["segments"][0]["semantic_corrections"][0]["merged_segment_count"] == 2


def test_import_transcript_semantic_review_notes_can_close_keep_original(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "semantic keep original", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "今天讲 browser base"}]})
    (root / "platform-subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:04,000\n今天讲 Browserbase\n", encoding="utf-8")
    write_json(root / "timeline.json", [{"index": 0, "start": 0, "end": 4, "transcript": "今天讲 browser base", "visual_text": "Browserbase"}])
    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(row for row in pack["candidates"] if row.get("has_conflict"))
    bad_result = root / "bad-result.json"
    write_json(
        bad_result,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "action": "replace",
                    "correction_type": candidate["correction_type"],
                    "original_text": candidate["original_text"],
                    "corrected_text": "Browserbase",
                    "confidence": 0.5,
                    "semantic_rationale": "证据冲突且置信不足，需要人工确认。",
                    "evidence_ids": candidate["evidence_ids"],
                    "safe_to_apply": True,
                }
            ],
        },
    )
    initial_validation = validate_transcript_semantic_correction(root, input_json=bad_result, write=True)
    assert initial_validation["review_required_count"] == 1

    review_notes = root / "semantic-review-notes.json"
    write_json(
        review_notes,
        {
            "schema": "lecture_review_notes.v1",
            "reviews": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": "keep_original",
                    "comment": "人工看完整上下文后确认这里按原转写保留。",
                    "evidence_ids": candidate["evidence_ids"],
                }
            ],
        },
    )

    imported = import_transcript_semantic_review_notes(root, review_json=review_notes, write=True)

    assert imported["decision_count"] == 1
    assert imported["validation"]["accepted_decision_count"] == 1
    assert imported["validation"]["review_required_count"] == 0
    status = transcript_semantic_correction_status(root, write=True)
    assert status["review_required_count"] == 0
    assert status["review_closure_summary"]["closed_review_decision_count"] == 1
    assert status["review_closure_summary"]["open_review_required_count"] == 0
    assert status["review_closure_summary"]["actions"]["keep_original"] == 1
    asset_status = content_asset_status(root, write=False)
    assert asset_status["semantic_correction_review_closure_summary"]["closed_review_decision_count"] == 1
    closure = transcript_semantic_correction_closure(root, input_json=imported["result_json"], write=True)
    assert closure["status"] == "completed_no_text_changes"


def test_semantic_correction_llm_draft_preview_writes_prompt_without_provider(tmp_path: Path) -> None:
    root = tmp_path / "llm-preview"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "llm preview"})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "title": "llm preview",
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "start": 0,
                    "end": 4,
                    "time_range": "00:00:00.000 - 00:00:04.000",
                    "correction_type": "proper_noun",
                    "risk_level": "medium",
                    "original_text": "titok",
                    "context_text": "这里讲 titok 平台",
                    "reason": "ascii_tool_or_proper_noun_in_chinese_transcript",
                    "evidence_ids": ["asr_segment_0"],
                    "evidence": [{"evidence_id": "asr_segment_0", "source_type": "asr_or_subtitle", "text": "这里讲 titok 平台"}],
                }
            ],
        },
    )
    initial_status = transcript_semantic_correction_status(root, write=True)
    assert initial_status["llm_draft_status"] == "not_planned"
    assert initial_status["llm_draft_next_action"] == "run_llm_draft_preview"

    result = build_transcript_semantic_correction_llm_draft(root, execute=False, limit=1, write=True)
    assert result["status"] == "planned"
    assert result["ok"] is True
    assert result["execute"] is False
    assert (root / "transcript-semantic-correction-llm-prompt.md").exists()
    assert not (root / "transcript-semantic-correction-result.llm.json").exists()

    prompt_status = transcript_semantic_correction_status(root, write=True)
    assert prompt_status["llm_draft_status"] == "prompt_ready"
    assert prompt_status["llm_draft_next_action"] == "execute_llm_or_use_codex"
    assert "execute_llm_or_use_codex" in prompt_status["commands"]
    assert "validate_llm_result" in prompt_status["commands"]
    assert "LLM draft status" in (root / "transcript-semantic-correction-status.md").read_text(encoding="utf-8")

    write_json(
        root / "transcript-semantic-correction-result.llm.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
            "source": "text_llm_semantic_review",
            "decisions": [
                {
                    "candidate_id": "semcorr-0001",
                    "accept": False,
                    "original_text": "titok",
                    "corrected_text": "TikTok",
                    "confidence": 0.2,
                    "rationale": "证据不足",
                    "evidence_ids": ["asr_segment_0"],
                    "needs_human_review": True,
                }
            ],
        },
    )
    executed_status = transcript_semantic_correction_status(root, write=True)
    assert executed_status["llm_draft_status"] == "executed"
    assert executed_status["llm_draft_next_action"] == "validate_llm_result"


def test_semantic_correction_llm_draft_prioritizes_attention_candidates(tmp_path: Path) -> None:
    root = tmp_path / "llm-attention"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "attention priority", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0, "end": 4, "text": "今天讲 titok 平台"},
                {"start": 10, "end": 14, "text": "这里这个很重要大家看一下"},
            ]
        },
    )
    write_json(
        root / "timeline.json",
        [
            {"index": 0, "start": 0, "end": 4, "visual_text": "TikTok platform"},
            {
                "index": 1,
                "start": 10,
                "end": 14,
                "visual_text": "客户信任建立流程：确认需求，给出解决方案",
                "tagger_annotations": [{"text": "重点概念：客户信任建立流程"}],
            },
        ],
    )

    pack = build_transcript_semantic_correction_pack(root, write=True)
    concept = next(row for row in pack["candidates"] if row["correction_type"] == "concept")

    result = build_transcript_semantic_correction_llm_draft(root, execute=False, limit=1, write=True)
    prompt = (root / "transcript-semantic-correction-llm-prompt.md").read_text(encoding="utf-8")

    assert result["candidate_count"] == 1
    assert result["candidate_selection"]["strategy"] == "source_conflict_first"
    assert result["candidate_selection"]["attention_candidate_count"] >= 1
    assert result["candidate_selection"]["selected_candidate_ids"] == [concept["candidate_id"]]
    assert "客户信任建立流程" in prompt
    assert "source_conflict_first" in prompt
    assert "titok" not in prompt


def _readable_impact_acceptance_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "readable-impact-presence"
    (root / "exports").mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "readable impact presence"})
    write_json(root / "transcript-semantic-correction-validation.json", {
        "accepted_decisions": [{"candidate_id": "semcorr-0001", "action": "replace",
            "original_text": "我买重疾。", "corrected_text": "我买重疾险。", "correction_type": "ordinary_word",
            "confidence": 0.96}]
    })
    return root


def test_readable_impact_cannot_pass_when_full_transcript_is_missing(tmp_path: Path) -> None:
    root = _readable_impact_acceptance_fixture(tmp_path)
    (root / "exports" / "smart-summary.md").write_text(_llm_summary_text("讲述者研究重疾险。"), encoding="utf-8")
    result = transcript_semantic_correction_readable_impact_report(root, write=True)
    assert result["ok"] is False
    assert result["status"] == "missing_required_documents"
    assert result["missing_required_documents"] == ["full_transcript"]
    assert "full_transcript" in (root / "transcript-semantic-readable-impact-report.md").read_text(encoding="utf-8")


def test_readable_impact_cannot_pass_for_empty_full_transcript(tmp_path: Path) -> None:
    root = _readable_impact_acceptance_fixture(tmp_path)
    (root / "exports" / "full-transcript.md").write_text("\n \t\n", encoding="utf-8")
    result = transcript_semantic_correction_readable_impact_report(root, write=False)
    assert result["ok"] is False
    assert result["missing_required_documents"] == ["full_transcript"]


def test_readable_impact_does_not_require_an_uninstalled_final_summary(tmp_path: Path) -> None:
    root = _readable_impact_acceptance_fixture(tmp_path)
    (root / "exports" / "full-transcript.md").write_text("我买重疾险。", encoding="utf-8")
    result = transcript_semantic_correction_readable_impact_report(root, write=False)
    assert result["ok"] is True
    assert result["missing_required_documents"] == []
    assert result["required_documents"] == ["full_transcript"]
    assert result["smart_summary_evaluation"] == "pending_llm_not_required"


def test_semantic_readable_impact_ignores_artifact_paths(tmp_path: Path) -> None:
    root = tmp_path / "readable-impact"
    (root / "exports").mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "readable impact"})
    write_json(
        root / "transcript-semantic-correction-validation.json",
        {
            "accepted_decisions": [
                {"candidate_id": "semcorr-0001", "action": "replace", "original_text": "tiktok", "corrected_text": "TikTok", "correction_type": "proper_noun", "confidence": 0.96}
            ]
        },
    )
    (root / "exports" / "full-transcript.md").write_text("正文已经是 TikTok\n路径 D:\\tmp\\tiktok-demo\\file.json", encoding="utf-8")
    (root / "exports" / "smart-summary.md").write_text("总结已经是 TikTok\n- 来源: D:\\tmp\\tiktok-demo\\file.json", encoding="utf-8")
    (root / "exports" / "knowledge-note.md").write_text("审计可保留 raw tiktok", encoding="utf-8")
    result = transcript_semantic_correction_readable_impact_report(root, write=True)
    assert result["status"] == "passed"
    assert result["required_readable_residual_total"] == 0
    assert (root / "transcript-semantic-readable-impact-report.md").exists()





def test_semantic_status_reruns_stale_closure_after_new_validation(tmp_path: Path) -> None:
    root = tmp_path / "bundle-stale-closure"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "stale closure fixture"})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "correction_type": "proper_noun",
                    "risk_level": "safe_auto_apply",
                    "original_text": "browser base",
                    "suggested_text": "Browserbase",
                    "start": 0,
                    "end": 3,
                    "evidence_source_types": ["ocr_ebook"],
                }
            ],
            "updated_at": "2026-07-07T09:17:00",
        },
    )
    write_json(
        root / "transcript-semantic-correction-validation.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_validation.v1",
            "status": "accepted_with_rejections",
            "accepted_decision_count": 1,
            "accepted_decisions": [
                {
                    "candidate_id": "semcorr-0001",
                    "action": "replace",
                    "correction_type": "proper_noun",
                    "original_text": "browser base",
                    "corrected_text": "Browserbase",
                    "confidence": 0.94,
                    "evidence_ids": ["ev-001"],
                }
            ],
            "review_rows": [],
            "updated_at": "2026-07-07T09:18:14",
        },
    )
    write_json(
        root / "transcript-semantic-correction-closure.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_closure.v1",
            "status": "no_safe_decisions",
            "ok": False,
            "applied_correction_count": 0,
            "changed_segment_count": 0,
            "updated_at": "2026-07-07T09:17:20",
        },
    )
    write_json(
        root / "transcript-semantic-correction-impact-report.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_impact.v1",
            "status": "needs_fix",
            "final_residual_error_total": 1,
            "updated_at": "2026-07-07T09:17:30",
        },
    )

    status = transcript_semantic_correction_status(root, write=False)

    assert status["status"] == "needs_closure"
    assert status["next_action_key"] == "run_closure"
    assert status["accepted_decision_count"] == 1
    assert status["closure_status"] == "no_safe_decisions"


def test_transcript_semantic_status_detects_codex_candidate_suggestions(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    build_transcript_semantic_correction_pack(root, write=True)
    build_transcript_semantic_candidate_discovery_pack(root, limit=5, write=True)
    suggestion_path = root / "transcript-semantic-candidate-suggestions.codex.md"
    suggestion_path.write_text(
        "```json\n"
        + json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1",
                "source": "codex_candidate_discovery_test",
                "suggestion_count": 1,
                "suggestions": [
                    {
                        "source_segment_index": 0,
                        "original_text": "今天讲",
                        "candidate_text": "今天主要讲",
                        "correction_type": "ordinary_word",
                        "confidence": 0.61,
                        "reason": "低信息开头，需进入复核。",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    status = transcript_semantic_correction_status(root, write=False)

    assert status["candidate_discovery_status"] == "suggestions_ready"
    assert status["candidate_discovery_next_action"] == "import_candidate_suggestions"
    assert status["candidate_discovery_suggestion_count"] == 1
    assert status["candidate_discovery_artifacts"]["codex_suggestions_markdown"].endswith("transcript-semantic-candidate-suggestions.codex.md")


def test_semantic_correction_codex_draft_accepts_safe_spaced_acronym_normalization(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "spaced acronym fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "今天讲 s e o"}]})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "start": 0,
                    "end": 4,
                    "time_range": "00:00:00.000 - 00:00:04.000",
                    "correction_type": "proper_noun",
                    "risk_level": "medium",
                    "original_text": "s e o",
                    "candidate_text": "SEO",
                    "context_text": "今天讲 s e o",
                    "reason": "odd_spaced_letters_or_acronym",
                    "needs_human_review": False,
                    "evidence_ids": ["asr_segment_0", "metadata_manifest"],
                    "evidence_source_types": ["asr_or_subtitle", "page_metadata"],
                    "evidence": [
                        {"evidence_id": "asr_segment_0", "source_type": "asr_or_subtitle", "text": "今天讲 s e o"},
                        {"evidence_id": "metadata_manifest", "source_type": "page_metadata", "text": "title: spaced acronym fixture"},
                    ],
                }
            ],
        },
    )

    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert draft["status"] == "draft_ready"
    assert draft["decision_count"] == 1
    assert draft["decisions"][0]["original_text"] == "s e o"
    assert draft["decisions"][0]["corrected_text"] == "SEO"
    assert draft["decisions"][0]["safe_to_apply"] is True

    validation = validate_transcript_semantic_correction(root, input_json=root / "transcript-semantic-correction-result.codex.md", write=False)
    assert validation["accepted_decision_count"] == 1
    assert validation["accepted_decisions"][0]["corrected_text"] == "SEO"

def test_semantic_correction_codex_draft_does_not_attach_known_term_to_number_candidate_context(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "number context fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "一个 a i 风口"}]})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "start": 0,
                    "end": 4,
                    "time_range": "00:00:00.000 - 00:00:04.000",
                    "correction_type": "number",
                    "risk_level": "high",
                    "original_text": "一个",
                    "candidate_text": "",
                    "context_text": "一个 a i 风口",
                    "reason": "contains_number_or_amount",
                    "needs_human_review": True,
                    "evidence_ids": ["asr_segment_0", "metadata_manifest"],
                    "evidence_source_types": ["asr_or_subtitle", "page_metadata"],
                    "evidence": [
                        {"evidence_id": "asr_segment_0", "source_type": "asr_or_subtitle", "text": "一个 a i 风口"},
                        {"evidence_id": "metadata_manifest", "source_type": "page_metadata", "text": "title: number context fixture"},
                    ],
                }
            ],
        },
    )

    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert draft["status"] == "no_safe_draft_decisions"
    assert draft["decision_count"] == 0
    assert not any(row.get("corrected_text") == "AI" for row in draft["decisions"])

def test_semantic_correction_codex_draft_does_not_attach_context_term_to_different_proper_candidate(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "proper context fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "a i 工具和 s e o"}]})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "start": 0,
                    "end": 4,
                    "time_range": "00:00:00.000 - 00:00:04.000",
                    "correction_type": "proper_noun",
                    "risk_level": "medium",
                    "original_text": "s e o",
                    "candidate_text": "SEO",
                    "context_text": "a i 工具和 s e o",
                    "reason": "odd_spaced_letters_or_acronym",
                    "needs_human_review": False,
                    "evidence_ids": ["asr_segment_0", "metadata_manifest"],
                    "evidence_source_types": ["asr_or_subtitle", "page_metadata"],
                    "evidence": [
                        {"evidence_id": "asr_segment_0", "source_type": "asr_or_subtitle", "text": "a i 工具和 s e o"},
                        {"evidence_id": "metadata_manifest", "source_type": "page_metadata", "text": "title: proper context fixture"},
                    ],
                }
            ],
        },
    )

    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert draft["status"] == "draft_ready"
    assert draft["decision_count"] == 1
    assert draft["decisions"][0]["original_text"] == "s e o"
    assert draft["decisions"][0]["corrected_text"] == "SEO"
    assert not any(row.get("corrected_text") == "AI" for row in draft["decisions"])

def test_semantic_correction_codex_draft_keeps_unknown_spaced_acronym_for_review(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "unknown acronym fixture", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "这里可能是 g p d"}]})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "start": 0,
                    "end": 4,
                    "time_range": "00:00:00.000 - 00:00:04.000",
                    "correction_type": "proper_noun",
                    "risk_level": "medium",
                    "original_text": "g p d",
                    "candidate_text": "GPD",
                    "context_text": "这里可能是 g p d",
                    "reason": "odd_spaced_letters_or_acronym",
                    "needs_human_review": False,
                    "evidence_ids": ["asr_segment_0", "metadata_manifest"],
                    "evidence_source_types": ["asr_or_subtitle", "page_metadata"],
                    "evidence": [
                        {"evidence_id": "asr_segment_0", "source_type": "asr_or_subtitle", "text": "这里可能是 g p d"},
                        {"evidence_id": "metadata_manifest", "source_type": "page_metadata", "text": "title: unknown acronym fixture"},
                    ],
                }
            ],
        },
    )

    draft = build_transcript_semantic_correction_codex_draft(root, write=True)
    assert draft["status"] == "no_safe_draft_decisions"
    assert draft["decision_count"] == 0
    assert not any(row.get("corrected_text") == "GPD" for row in draft["decisions"])

def test_review_notes_rebind_shifted_candidate_id_by_unique_original(tmp_path: Path) -> None:
    root = tmp_path / "rebind"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "rebind", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 3, "text": "use cell table"}]})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "candidate_count": 2,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "original_text": "other",
                    "candidate_text": "Other",
                    "correction_type": "proper_noun",
                    "evidence_ids": ["ocr-other"],
                    "evidence": [{"evidence_id": "ocr-other", "source_type": "ocr", "text": "Other"}],
                },
                {
                    "candidate_id": "semcorr-0002",
                    "segment_index": 0,
                    "original_text": "cell",
                    "candidate_text": "Excel",
                    "correction_type": "proper_noun",
                    "evidence_ids": ["ocr-excel"],
                    "evidence": [{"evidence_id": "ocr-excel", "source_type": "ocr", "text": "Excel"}],
                },
            ],
        },
    )
    notes = root / "notes.json"
    write_json(
        notes,
        {
            "reviews": [
                {
                    "candidate_id": "semcorr-0001",
                    "original_text": "cell",
                    "correction_type": "proper_noun",
                    "status": "accept_correction",
                    "corrected_text": "Excel",
                    "comment": "Human confirmed from the slide.",
                }
            ]
        },
    )

    imported = import_transcript_semantic_review_notes(root, review_json=notes, write=True)

    assert imported["decision_count"] == 1
    decision = read_json(Path(imported["result_json"]))["decisions"][0]
    assert decision["candidate_id"] == "semcorr-0002"
    assert decision["rebound_from_candidate_id"] == "semcorr-0001"
    assert decision["evidence_ids"] == ["ocr-excel"]
    assert imported["validation"]["accepted_decision_count"] == 1


def test_review_notes_ambiguous_original_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "ambiguous"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "ambiguous"})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "candidate_count": 3,
            "candidates": [
                {"candidate_id": "semcorr-0001", "segment_index": 0, "original_text": "other", "correction_type": "ordinary_word"},
                {"candidate_id": "semcorr-0002", "segment_index": 1, "original_text": "cell", "correction_type": "proper_noun"},
                {"candidate_id": "semcorr-0003", "segment_index": 2, "original_text": "cell", "correction_type": "proper_noun"},
            ],
        },
    )
    notes = root / "notes.json"
    write_json(notes, {"reviews": [{"candidate_id": "semcorr-0001", "original_text": "cell", "status": "accept_correction", "corrected_text": "Excel"}]})

    imported = import_transcript_semantic_review_notes(root, review_json=notes, write=False)

    assert imported["decision_count"] == 0
    assert imported["skipped_count"] == 1


def test_semantic_closure_is_idempotent_when_human_correction_already_present(tmp_path: Path, monkeypatch) -> None:
    from video_knowledge_pipeline import transcript_semantic_correction as semantic
    monkeypatch.setattr(semantic, 'now_iso', lambda: '2026-01-01T00:00:00+00:00')
    root = tmp_path / "idempotent"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "idempotent", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 3, "text": "use Excel table"}]})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "original_text": "cell",
                    "candidate_text": "Excel",
                    "correction_type": "proper_noun",
                    "evidence_ids": ["ocr-excel"],
                    "evidence": [{"evidence_id": "ocr-excel", "source_type": "ocr", "text": "Excel"}],
                }
            ],
        },
    )
    result_path = root / "result.json"
    write_json(
        result_path,
        {
            "decisions": [
                {
                    "candidate_id": "semcorr-0001",
                    "action": "replace",
                    "correction_type": "proper_noun",
                    "original_text": "cell",
                    "corrected_text": "Excel",
                    "confidence": 1.0,
                    "semantic_rationale": "Human confirmed from the slide.",
                    "evidence_ids": ["ocr-excel"],
                    "safe_to_apply": True,
                    "human_confirmed": True,
                }
            ]
        },
    )

    closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)

    assert closure["status"] == "completed"
    assert closure["applied_correction_count"] == 1
    assert closure["changed_segment_count"] == 0
    assert closure["applied_corrections"][0]["application"] == "already_present"
    assert read_json(root / "source-arbitrated-transcript.json")["segments"][0]["text"] == "use Excel table"

    canonical_bytes = (root / "source-arbitrated-transcript.json").read_bytes()
    monkeypatch.setattr(semantic, 'now_iso', lambda: '2026-01-01T00:00:02+00:00')
    second_closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)

    assert second_closure["canonical_write_status"] == "unchanged"
    assert (root / "source-arbitrated-transcript.json").read_bytes() == canonical_bytes


def test_semantic_validation_identity_preserves_bytes_but_not_changed_inputs(tmp_path: Path, monkeypatch) -> None:
    from video_knowledge_pipeline import transcript_semantic_correction as semantic
    from video_knowledge_pipeline.file_hash import sha256_file
    root = tmp_path / 'validation-identity'
    write_json(root / 'manifest.json', {})
    write_json(root / 'transcript-semantic-correction-pack.json', {'candidates': []})
    answer = root / 'answer.json'
    write_json(answer, {'decisions': [], 'candidate_selection': {'selected_candidate_count': 0}})
    monkeypatch.setattr(semantic, 'now_iso', lambda: '2026-01-01T00:00:00+00:00')
    first = semantic.validate_transcript_semantic_correction(root, input_json=answer)
    report = root / 'transcript-semantic-correction-validation.json'
    before = report.read_bytes()
    monkeypatch.setattr(semantic, 'now_iso', lambda: '2026-01-01T00:00:02+00:00')
    repeated = semantic.validate_transcript_semantic_correction(root, input_json=answer)
    assert report.read_bytes() == before
    assert repeated['updated_at'] == first['updated_at']
    assert repeated['input_sha256'] == sha256_file(answer)
    # Equivalent JSON with different bytes still has a distinct raw identity.
    answer.write_bytes(answer.read_bytes() + b'\n')
    changed = semantic.validate_transcript_semantic_correction(root, input_json=answer)
    assert changed['input_sha256'] != repeated['input_sha256']
    assert report.read_bytes() != before
    before = report.read_bytes()
    semantic.validate_transcript_semantic_correction(root, input_json=answer, min_confidence=0.99)
    assert report.read_bytes() != before


def test_sequential_semantic_closures_preserve_prior_validated_corrections(tmp_path: Path) -> None:
    root = tmp_path / "sequential"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "sequential", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 5, "text": "MIAAPP fee"}]})

    def write_round(candidate_id: str, original: str, corrected: str, result_name: str) -> Path:
        evidence_id = f"human-{candidate_id}"
        write_json(
            root / "transcript-semantic-correction-pack.json",
            {
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "segment_index": 0,
                        "original_text": original,
                        "candidate_text": corrected,
                        "correction_type": "proper_noun",
                        "evidence_ids": [evidence_id],
                        "evidence": [{"evidence_id": evidence_id, "source_type": "human_note", "text": corrected}],
                    }
                ],
            },
        )
        result_path = root / result_name
        write_json(
            result_path,
            {
                "decisions": [
                    {
                        "candidate_id": candidate_id,
                        "action": "replace",
                        "correction_type": "proper_noun",
                        "original_text": original,
                        "corrected_text": corrected,
                        "confidence": 1.0,
                        "semantic_rationale": "Human confirmed.",
                        "evidence_ids": [evidence_id],
                        "safe_to_apply": True,
                        "human_confirmed": True,
                    }
                ]
            },
        )
        return result_path

    first_result = write_round("semcorr-0001", "MIAAPP", "Mingya APP", "first.json")
    first = transcript_semantic_correction_closure(root, input_json=first_result, write=True)
    assert first["cumulative_decision_count"] == 1

    second_result = write_round("semcorr-0002", "fee", "premium", "second.json")
    second = transcript_semantic_correction_closure(root, input_json=second_result, write=True)

    corrected = read_json(root / "source-arbitrated-transcript.json")
    ledger = read_json(root / "transcript-semantic-correction-decision-ledger.json")
    assert second["current_accepted_decision_count"] == 1
    assert second["cumulative_decision_count"] == 2
    assert second["applied_correction_count"] == 2
    assert corrected["segments"][0]["text"] == "Mingya APP premium"
    assert corrected["segments"][0]["raw_text"] == "MIAAPP fee"
    assert ledger["decision_count"] == 2
    status = transcript_semantic_correction_status(root)
    assert status["accepted_decision_count"] == 2
    assert status["ui_summary"]["accepted_decision_count"] == 2


def test_semantic_closure_uses_richest_upstream_base(tmp_path: Path) -> None:
    root = tmp_path / "rich-upstream-base"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "rich base", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 5, "text": "MIAAPP"}]})
    write_json(root / "corrected-transcript.json", {"segments": [{"start": 0, "end": 5, "text": "full context MIAAPP tail"}]})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "original_text": "MIAAPP",
                    "candidate_text": "Mingya APP",
                    "correction_type": "proper_noun",
                    "evidence_ids": ["human-mingya"],
                    "evidence": [{"evidence_id": "human-mingya", "source_type": "human_note", "text": "Mingya APP"}],
                }
            ],
        },
    )
    result_path = root / "result.json"
    write_json(
        result_path,
        {
            "decisions": [
                {
                    "candidate_id": "semcorr-0001",
                    "action": "replace",
                    "correction_type": "proper_noun",
                    "original_text": "MIAAPP",
                    "corrected_text": "Mingya APP",
                    "confidence": 1.0,
                    "semantic_rationale": "Human confirmed.",
                    "evidence_ids": ["human-mingya"],
                    "safe_to_apply": True,
                    "human_confirmed": True,
                }
            ]
        },
    )

    closure = transcript_semantic_correction_closure(root, input_json=result_path, write=True)

    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert closure["status"] == "completed"
    assert corrected["segments"][0]["text"] == "full context Mingya APP tail"

def test_semantic_validation_requires_two_independent_secondary_asr_sources(tmp_path: Path) -> None:
    root = tmp_path / "dual-asr"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "dual ASR"})
    evidence = [
        {"evidence_id": "primary", "source_type": "asr_or_subtitle", "text": "fee"},
        {"evidence_id": "secondary-qwen", "source_type": "secondary_asr", "provider": "qwen3-asr", "artifact_sha256": "a" * 64, "text": "premium"},
        {"evidence_id": "secondary-mistral", "source_type": "secondary_asr", "provider": "mistral_asr", "artifact_sha256": "b" * 64, "text": "premium"},
        {"evidence_id": "suggestion", "source_type": "candidate_discovery_suggestion", "text": "premium"},
    ]
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "original_text": "fee",
                    "candidate_text": "premium",
                    "correction_type": "ordinary_word",
                    "has_conflict": True,
                    "evidence_ids": [row["evidence_id"] for row in evidence],
                    "evidence": evidence,
                }
            ],
        },
    )

    def decision(evidence_ids: list[str]) -> dict[str, object]:
        return {
            "candidate_id": "semcorr-0001",
            "action": "replace",
            "correction_type": "ordinary_word",
            "original_text": "fee",
            "corrected_text": "premium",
            "confidence": 0.96,
            "semantic_rationale": "Independent ASR evidence agrees.",
            "evidence_ids": evidence_ids,
            "safe_to_apply": True,
            "human_confirmed": False,
        }

    one_path = root / "one.json"
    write_json(one_path, {"decisions": [decision(["secondary-qwen", "suggestion"])]})
    one = validate_transcript_semantic_correction(root, input_json=one_path, write=False)
    assert one["accepted_decision_count"] == 0
    assert "insufficient_independent_evidence_for_asr_conflict" in one["rejected_decisions"][0]["reject_reasons"]

    two_path = root / "two.json"
    write_json(two_path, {"decisions": [decision(["secondary-qwen", "secondary-mistral", "suggestion"])]})
    two = validate_transcript_semantic_correction(root, input_json=two_path, write=False)
    assert two["accepted_decision_count"] == 1


def test_sidecar_evidence_merges_all_overlapping_cues_from_one_source() -> None:
    cue = type("Cue", (), {"start": 10.0, "end": 20.0, "text": "primary text"})()
    first = type("Cue", (), {"start": 9.0, "end": 15.0, "text": "first half"})()
    second = type("Cue", (), {"start": 15.0, "end": 21.0, "text": "second half contains target"})()

    rows = _sidecar_evidence_for_cue(
        cue,
        [
            {
                "source_id": "secondary_qwen",
                "source_type": "secondary_asr",
                "provider": "qwen3-asr",
                "artifact_sha256": "a" * 64,
                "cues": [first, second],
            }
        ],
    )

    assert len(rows) == 1
    assert rows[0]["start"] == 9.0
    assert rows[0]["end"] == 21.0
    assert rows[0]["text"] == "first half second half contains target"


def test_pack_rehydrates_persisted_candidate_with_current_sidecar_evidence(tmp_path: Path) -> None:
    root = tmp_path / "rehydrate"
    root.mkdir()
    write_json(
        root / "manifest.json",
        {
            "title": "rehydrate",
            "normalized_transcript_json": "normalized-transcript.json",
            "asr_secondary_transcripts": ["qwen.json", "mistral.json"],
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": [{"start": 0, "end": 10, "text": "fee"}]})
    write_json(root / "timeline.json", [])
    write_json(
        root / "qwen.json",
        {"provider": "qwen3-asr", "segments": [{"start": 0, "end": 5, "text": "first half"}, {"start": 5, "end": 10, "text": "premium"}]},
    )
    write_json(root / "mistral.json", {"provider": "mistral_asr", "segments": [{"start": 0, "end": 10, "text": "premium"}]})
    write_json(
        root / "transcript-semantic-candidate-suggestions-imported.json",
        {
            "imported_candidates": [
                {
                    "candidate": {
                        "candidate_id": "semcorr-9000",
                        "segment_index": 0,
                        "start": 0,
                        "end": 10,
                        "original_text": "fee",
                        "candidate_text": "premium",
                        "correction_type": "ordinary_word",
                        "reason": "independent ASR agreement",
                        "needs_human_review": False,
                        "evidence": [{"evidence_id": "stale", "source_type": "secondary_asr", "text": "stale"}],
                    }
                }
            ]
        },
    )

    pack = build_transcript_semantic_correction_pack(root, write=False)
    candidate = next(row for row in pack["candidates"] if row.get("candidate_text") == "premium")
    secondary = [row for row in candidate["evidence"] if row.get("source_type") == "secondary_asr"]

    assert candidate["candidate_id"] == "semcorr-9000"
    assert len(secondary) == 2
    assert "first half premium" in secondary[0]["text"]
    assert secondary[1]["text"] == "premium"
    assert all(row.get("artifact_sha256") for row in secondary)


def test_semantic_status_rejects_validation_bound_to_shifted_candidate(tmp_path: Path) -> None:
    root = tmp_path / "stale"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "stale"})
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "candidate_count": 1,
            "candidates": [{"candidate_id": "semcorr-0001", "segment_index": 0, "original_text": "other", "correction_type": "proper_noun"}],
        },
    )
    stale_decision = {"candidate_id": "semcorr-0001", "original_text": "cell", "corrected_text": "Excel"}
    write_json(
        root / "transcript-semantic-correction-validation.json",
        {"accepted_decision_count": 1, "accepted_decisions": [stale_decision], "review_required_count": 0, "updated_at": "2026-07-20T10:00:00"},
    )
    write_json(root / "transcript-semantic-correction-closure.json", {"status": "completed", "ok": True, "updated_at": "2026-07-20T10:01:00"})
    write_json(root / "transcript-semantic-correction-impact-report.json", {"status": "passed", "updated_at": "2026-07-20T10:02:00"})
    write_json(root / "transcript-semantic-readable-impact-report.json", {"status": "passed", "updated_at": "2026-07-20T10:03:00"})
    write_json(root / "transcript-semantic-summary-impact-report.json", {"status": "passed", "updated_at": "2026-07-20T10:04:00"})

    status = transcript_semantic_correction_status(root, write=False)

    assert status["status"] == "stale_validation_pack"
    assert status["ok"] is False
    assert status["artifact_identity"]["current"] is False
    assert status["artifact_identity"]["issues"][0]["key"] == "validated_candidate_original_mismatch"


def test_canonical_source_mode_and_visual_transport_filter_regressions(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    normalized = read_json(root / "normalized-transcript.json")
    normalized["segments"][0]["text"] = "今天讲保证方案"
    write_json(root / "normalized-transcript.json", normalized)
    write_json(
        root / "source-arbitrated-transcript.json",
        {
            "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
            "segments": [
                {"start": 0, "end": 4, "text": "今天讲保障方案"},
                {"start": 5, "end": 9, "text": "然后点击登录并保存配置"},
            ],
        },
    )
    manifest = read_json(root / "manifest.json")
    manifest["source_arbitrated_transcript_json"] = "source-arbitrated-transcript.json"
    write_json(root / "manifest.json", manifest)
    timeline = read_json(root / "timeline.json")
    timeline[0]["visual_text"] = (
        "![img-0.jpeg](img-0.jpeg) "
        "D:\\bundle\\assets\\frame_01_0001ms.jpg "
        "video_knowledge_pipeline.visual_text.v1 "
        "明亚保险经纪"
    )
    write_json(root / "timeline.json", timeline)

    raw_pack = build_transcript_semantic_correction_pack(
        root, source_mode="raw", write=False
    )
    canonical_pack = build_transcript_semantic_correction_pack(
        root, source_mode="canonical", write=False
    )

    assert raw_pack["source_mode"] == "raw"
    assert canonical_pack["source_mode"] == "canonical"
    assert any(
        row.get("original_text") == "保证方案"
        and row.get("suggested_text") == "保障方案"
        for row in raw_pack["candidates"]
    )
    assert not any(
        row.get("original_text") == "保证方案"
        for row in canonical_pack["candidates"]
    )
    semantic_fields = json.dumps(
        [
            {
                "original_text": row.get("original_text"),
                "suggested_text": row.get("suggested_text"),
                "reason": row.get("reason"),
            }
            for row in canonical_pack["candidates"]
        ],
        ensure_ascii=False,
    )
    assert "img-0" not in semantic_fields
    assert "frame_01_0001ms" not in semantic_fields
    assert not any(
        row.get("reason") == "visual_text_differs_from_transcript"
        and str(row.get("suggested_text") or "") == "0"
        for row in canonical_pack["candidates"]
    )

def test_domain_lexicon_requires_one_independent_secondary_asr(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    normalized = read_json(root / "normalized-transcript.json")
    normalized["segments"][0]["text"] = "属于市场第一批兑的产品"
    write_json(root / "normalized-transcript.json", normalized)
    secondary = root / "secondary-asr.json"
    write_json(
        secondary,
        {
            "provider": "fixture-secondary-asr",
            "segments": [
                {"start": 0, "end": 4, "text": "属于市场第一梯队的产品"}
            ],
        },
    )
    manifest = read_json(root / "manifest.json")
    manifest["asr_secondary_transcripts"] = [secondary.name]
    write_json(root / "manifest.json", manifest)
    pack = build_transcript_semantic_correction_pack(root, write=True)
    candidate = next(
        row for row in pack["candidates"]
        if row.get("original_text") == "第一批兑"
    )
    lexicon_ids = [
        row["evidence_id"] for row in candidate["evidence"]
        if row.get("source_type") == "explicit_domain_lexicon"
    ]
    secondary_ids = [
        row["evidence_id"] for row in candidate["evidence"]
        if row.get("source_type") == "secondary_asr"
    ]
    assert len(lexicon_ids) == 1
    assert len(secondary_ids) == 1
    decision = {
        "candidate_id": candidate["candidate_id"],
        "action": "replace",
        "correction_type": "ordinary_word",
        "original_text": "第一批兑",
        "corrected_text": "第一梯队",
        "confidence": 0.93,
        "semantic_rationale": "显式领域词库与一份独立 ASR 在同一时间窗内一致。",
        "evidence_ids": candidate["evidence_ids"],
        "safe_to_apply": True,
        "needs_human_review": False,
        "human_confirmed": False,
    }
    accepted_input = root / "domain-accepted.json"
    write_json(accepted_input, {"decisions": [decision]})
    accepted = validate_transcript_semantic_correction(
        root, input_json=accepted_input, write=False
    )
    assert accepted["accepted_decision_count"] == 1

    lexicon_only_input = root / "domain-lexicon-only.json"
    lexicon_only = dict(decision)
    lexicon_only["evidence_ids"] = [
        value for value in candidate["evidence_ids"] if value not in secondary_ids
    ]
    write_json(lexicon_only_input, {"decisions": [lexicon_only]})
    rejected = validate_transcript_semantic_correction(
        root, input_json=lexicon_only_input, write=False
    )
    assert rejected["accepted_decision_count"] == 0
    assert "insufficient_independent_evidence_for_asr_conflict" in rejected["rejected_decisions"][0]["reject_reasons"]
