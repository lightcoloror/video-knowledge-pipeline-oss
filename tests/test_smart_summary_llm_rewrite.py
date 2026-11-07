from __future__ import annotations

import json
import os
from pathlib import Path

from video_knowledge_pipeline.smart_summary_codex import (
    EVIDENCE_BOUNDARY,
    _normalise_llm_markdown,
    _render_llm_rewrite_pack,
    _timestamps,
    _compression_target,
    prepare_smart_summary_llm_rewrite,
    run_smart_summary_llm_rewrite,
    smart_summary_quality_check,
)
from video_knowledge_pipeline.smart_summary_section_llm import run_smart_summary_section_llm_rewrite
from video_knowledge_pipeline.transcript_postprocess import postprocess_asr_transcript
from video_knowledge_pipeline.task_console import export_task_console


def _bundle(root: Path) -> Path:
    bundle = root / "bundle-llm-rewrite"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "LLM Rewrite Video", "media_path": "D:/media/llm.mp4"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 8, "transcript": "开头讲客户画像和信任建立。", "visual_route": "document_visual"},
                {"index": 2, "start": 600, "end": 612, "transcript": "中段讲问题链和需求确认。", "visual_route": "semantic_frame"},
                {"index": 3, "start": 1200, "end": 1215, "transcript": "结尾讲复盘清单和下一步动作。", "visual_route": "temporal_sequence"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 8, "text": "开头讲客户画像、客户顾虑和建立信任的基本原则。"},
                    {"start": 600, "end": 612, "text": "中段展开陌客沟通的问题链和需求确认，强调连续提问。"},
                    {"start": 1200, "end": 1215, "text": "结尾要求把跟进动作和复盘清单沉淀下来。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-pack.json").write_text(
        json.dumps({"status": "no_candidates", "candidate_count": 0, "candidates": [], "candidate_groups": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle


def test_prepare_smart_summary_llm_rewrite_writes_handoff_pack(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    result = prepare_smart_summary_llm_rewrite(bundle)

    assert result["status"] == "ready_for_codex_rewrite"
    assert result["provider"] == "codex_manual"
    assert result["operator_boundary"]["cloud_llm_calls"] == "not_executed_by_this_command"
    assert result["operator_boundary"]["writes_final_summary"] is False
    assert Path(result["prompt_path"]).exists()
    assert Path(result["template_path"]).exists()
    assert not Path(result["expected_output_path"]).exists()
    assert "generate-smart-summary-with-codex" in result["install_command"]
    assert result["run_artifact"]["run_type"] == "smart_summary_llm_rewrite"
    assert result["run_artifact"]["status"] == "needs_input"

    pack = Path(result["prompt_path"]).read_text(encoding="utf-8")
    assert "这是成品智能总结" in pack
    assert "每一项都要带 `HH:MM:SS` 来源时间" in pack
    assert "不要依据第一人称" in pack
    assert "不要调用云端" in pack
    assert "Write the final Markdown" in pack
    assert "## Chapter Evidence" in pack

    status = json.loads((bundle / "exports" / "smart-summary-llm-rewrite-status.json").read_text(encoding="utf-8"))
    assert status["expected_output_path"].endswith("smart-summary.llm.md")


def test_llm_rewrite_pack_keeps_chapter_evidence_through_long_video_end(tmp_path: Path) -> None:
    chapters = [
        {
            "start": index * 600,
            "end": (index + 1) * 600,
            "title": f"第 {index + 1} 节",
            "summary_sentences": [f"第 {index + 1} 节的证据。"],
        }
        for index in range(13)
    ]

    pack = _render_llm_rewrite_pack(
        tmp_path,
        title="长视频",
        provider="fixture",
        pack={},
        memory_pack={},
        chapter_pack={"chapters": chapters, "course_map": {}},
        baseline_path=tmp_path / "missing.md",
        output_path=tmp_path / "smart-summary.llm.md",
    )

    assert "### 02:00:00.000 - 02:10:00.000 第 13 节" in pack

def test_normalise_llm_markdown_adds_reusable_evidence_boundary() -> None:
    content = _normalise_llm_markdown("# 示例课程 - 智能总结\n\n## 一句话概览\n\n概览。", title="示例课程")

    assert EVIDENCE_BOUNDARY in content
    assert "生成方式：`codex_llm_rewrite_final`" in content


def test_timestamps_accepts_minute_second_sections_without_parsing_iso_created_time() -> None:
    values = _timestamps("### 00:00 - 05:21\n\n- Created: `2026-07-23T16:36:46`")

    assert values == [0.0, 321.0]


def test_task_console_exposes_smart_summary_llm_rewrite_command(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    prepare_smart_summary_llm_rewrite(bundle)

    result = export_task_console(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    commands = {row["key"]: row for row in result["commands"]}

    assert "smart_summary_llm_rewrite" in commands
    assert "smart_summary_llm_rewrite_run" in commands
    assert commands["smart_summary_llm_rewrite"]["safety"] == "local_no_cloud"
    assert manifest["smart_summary_llm_rewrite_pack_markdown"] == "exports/smart-summary-llm-rewrite-pack.md"
    assert manifest["mcp_prepare_smart_summary_llm_rewrite_args"] == "mcp-prepare-smart-summary-llm-rewrite.args.json"
    assert manifest["mcp_run_smart_summary_llm_rewrite_args"] == "mcp-run-smart-summary-llm-rewrite.args.json"
    assert (bundle / "mcp-prepare-smart-summary-llm-rewrite.args.json").exists()
    assert (bundle / "mcp-run-smart-summary-llm-rewrite.args.json").exists()



def test_smart_summary_production_wrapper_uses_local_proxy_runtime(monkeypatch) -> None:
    import video_knowledge_pipeline.online_model_gateway as online_gateway
    import video_knowledge_pipeline.smart_summary_codex as summary_module
    from video_knowledge_pipeline.text_llm_gateway import resolve_text_provider_config

    calls: list[dict[str, object]] = []

    def keep_explicit(model_type: str, explicit: dict[str, object] | None = None, **kwargs: object) -> dict[str, object]:
        return dict(explicit or {})

    def fake_runtime(task: str, **kwargs: object) -> dict[str, object]:
        calls.append({"task": task, **kwargs})
        return {"ok": True, "status": "completed", "content": "# Proxy summary"}

    monkeypatch.setattr(online_gateway, "resolve_model_api_provider_config", keep_explicit)
    monkeypatch.setattr(online_gateway, "model_runtime_request", fake_runtime)
    provider = resolve_text_provider_config(
        {
            "provider": "openai_compatible",
            "adapter_backend": "proxy",
            "base_url": "http://127.0.0.1:8776/v1",
            "model": "vkp-local-text-test",
            "location": "local",
            "execution_location": "local",
            "route_id": "pool-local-text",
            "route_revision": "c" * 64,
        }
    )

    result = summary_module.call_openai_compatible_text(
        provider_config=provider,
        messages=[{"role": "user", "content": "rewrite"}],
    )

    assert result["ok"] is True
    assert result["content"] == "# Proxy summary"
    assert calls[0]["task"] == "summary_rewrite"
    assert calls[0]["execution_location"] == "local"
    assert calls[0]["messages"] == [{"role": "user", "content": "rewrite"}]

def test_run_smart_summary_llm_rewrite_preview_does_not_call_provider(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fail_call(*args, **kwargs):  # pragma: no cover - should never be reached
        raise AssertionError("provider should not be called in preview mode")

    monkeypatch.setattr("video_knowledge_pipeline.smart_summary_codex.call_openai_compatible_text", fail_call)

    result = run_smart_summary_llm_rewrite(
        bundle,
        provider_config={"provider": "fixture", "base_url": "http://example.invalid/v1", "model": "fake-model", "api_key": "SENTINEL_API_KEY_SHOULD_NOT_APPEAR"},
        execute=False,
    )

    assert result["status"] == "planned"
    assert result["execute"] is False
    assert result["provider"]["api_key_configured"] is True
    assert "SENTINEL_API_KEY_SHOULD_NOT_APPEAR" not in json.dumps(result, ensure_ascii=False)
    assert (bundle / "exports" / "smart-summary-llm-run-status.json").exists()
    assert not (bundle / "exports" / "smart-summary.llm.md").exists()


def test_run_smart_summary_llm_rewrite_executes_and_installs_final_summary(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fake_call(*, provider_config, messages, temperature=0):
        assert provider_config["api_key"] == "SENTINEL_API_KEY_SHOULD_NOT_APPEAR"
        assert messages and "这是成品智能总结" in messages[-1]["content"]
        return {
            "ok": True,
            "content": """# LLM Rewrite Video - 智能总结

生成方式：`codex_llm_rewrite_final`。

## 基本信息
- 视频名：LLM Rewrite Video
- 覆盖范围：00:00:00 - 00:20:15

## 一句话概览
这节课围绕客户画像、信任建立、问题链和复盘清单，讲如何把陌客沟通变成可执行的成交流程。

## 核心主题
课程主线是先识别客户顾虑，再通过连续提问确认需求，最后沉淀跟进动作。

## 分段总结
- 00:00:00 - 00:00:08：开头解释客户画像和建立信任的基本原则。
- 00:10:00 - 00:10:12：中段展开陌客沟通的问题链和需求确认。
- 00:20:00 - 00:20:15：结尾要求把跟进动作和复盘清单沉淀下来。

## 关键观点
- 信任建立不是话术堆砌，而是围绕客户顾虑给出连续证据。
- 问题链能把陌客沟通从闲聊推向需求确认。

## 可执行动作清单
- 先写出客户画像和常见顾虑。
- 为每类顾虑准备连续提问。
- 每次沟通后记录跟进动作和复盘清单。

## 高频话术
- “你现在最担心的是哪一块？”
- “如果这个问题解决了，下一步你会怎么判断？”

## 待复核点
- 视觉证据未执行或不足，屏幕细节仍需复核。
""",
        }

    monkeypatch.setattr("video_knowledge_pipeline.smart_summary_codex.call_openai_compatible_text", fake_call)

    result = run_smart_summary_llm_rewrite(
        bundle,
        provider_config={"provider": "fixture", "base_url": "http://example.invalid/v1", "model": "fake-model", "api_key": "SENTINEL_API_KEY_SHOULD_NOT_APPEAR"},
        execute=True,
    )

    # Successful generation is not successful admission: this deliberately small
    # fixture does not satisfy the final evidence/quality gates.
    assert result["status"] == "blocked_quality"
    assert result["ok"] is False
    assert (bundle / "exports" / "smart-summary.llm.md").exists()
    assert not (bundle / "exports" / "smart-summary.codex.md").exists()
    assert result["install_result"]["status"] == "blocked_quality"
    assert Path(result["install_result"]["validation_receipt"]).is_file()
    assert result["quality"]["is_codex_summary"] is True



def test_postprocess_asr_transcript_writes_corrected_sidecar(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    result = postprocess_asr_transcript(bundle, target_seconds=30, max_chars=120)

    assert result["status"] == "completed"
    assert result["source_segment_count"] == 3
    assert result["postprocessed_segment_count"] <= 3
    assert (bundle / "postprocessed-transcript.json").exists()
    assert (bundle / "corrected-transcript.json").exists()
    assert (bundle / "corrected-transcript.srt").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corrected_transcript_json"] == "corrected-transcript.json"
    assert manifest["transcript_json"] == "corrected-transcript.json"
    corrected = json.loads((bundle / "corrected-transcript.json").read_text(encoding="utf-8"))
    assert any(str(row.get("text", "")).endswith("。") for row in corrected["segments"])


def test_run_smart_summary_section_llm_rewrite_preview_does_not_call_provider(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)

    def fail_call(*args, **kwargs):  # pragma: no cover - should never be reached
        raise AssertionError("section provider should not be called in preview mode")

    monkeypatch.setattr("video_knowledge_pipeline.smart_summary_section_llm.call_openai_compatible_text", fail_call)

    result = run_smart_summary_section_llm_rewrite(
        bundle,
        provider_config={"provider": "fixture", "base_url": "http://example.invalid/v1", "model": "fake-model", "api_key": "SENTINEL_API_KEY_SHOULD_NOT_APPEAR"},
        execute=False,
        limit=2,
    )

    assert result["status"] == "planned"
    assert result["execute"] is False
    assert result["selected_section_count"] == 2
    assert "SENTINEL_API_KEY_SHOULD_NOT_APPEAR" not in json.dumps(result, ensure_ascii=False)
    assert (bundle / "exports" / "smart-summary-section-llm-rewrite.md").exists()
    assert (bundle / "mcp-smart-summary-section-llm-rewrite.args.json").exists()


def test_run_smart_summary_section_llm_rewrite_rejects_empty_provider_output(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    calls: list[int] = []

    def fake_call(*, provider_config, messages, temperature=0, max_tokens=1200):
        calls.append(1)
        if len(calls) == 1:
            return {"ok": True, "content": ""}
        return {
            "ok": True,
            "content": """### `00:10:00 - 00:10:12` 中段问题链

本节关键在于把陌生客户沟通从泛聊推进到需求确认。讲师建议先围绕客户当前问题发问，再把问题链拆成可执行动作，避免直接抛方案。

#### 关键观点
- 客户的问题需要通过连续提问被澄清，而不是靠顾问猜测。

#### 可执行动作
- 先确认客户最担心的问题，再记录下一步跟进动作。

- 视觉证据边界：本节涉及屏幕内容时仍需 OCR、多模态或人工复核。
""",
        }

    monkeypatch.setattr("video_knowledge_pipeline.smart_summary_section_llm.call_openai_compatible_text", fake_call)

    result = run_smart_summary_section_llm_rewrite(
        bundle,
        provider_config={"provider": "fixture", "base_url": "http://example.invalid/v1", "model": "fake-model", "api_key": "SENTINEL_API_KEY_SHOULD_NOT_APPEAR"},
        execute=True,
        limit=2,
        min_section_chars=80,
    )

    assert result["status"] == "partial_failed"
    assert result["ok"] is False
    assert result["failed_section_count"] == 1
    assert result["failed_items"][0]["reason"] == "empty_model_output"
    assert result["calls"][0]["accepted"] is False
    assert result["calls"][1]["accepted"] is True
    assert not (bundle / "exports" / "smart-summary.codex.md").exists()
    revisions = json.loads((bundle / "exports" / "smart-summary-section-llm-revisions.json").read_text(encoding="utf-8"))
    assert len(revisions["rows"]) == 1



def test_smart_summary_quality_requires_corrected_transcript_input(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    exports = bundle / "exports"
    exports.mkdir(exist_ok=True)
    summary = exports / "smart-summary.codex.md"
    summary.write_text(
        """# LLM Rewrite Video - 智能总结

生成方式：`codex_llm_rewrite_final`。

## 基本信息
- 视频名：LLM Rewrite Video
- 覆盖范围：00:00:00 - 00:20:15

## 一句话概览
这节课完整讲解客户画像、信任建立、问题链推进和复盘清单，帮助把陌生客户沟通整理成可执行的成交动作。

## 核心主题
课程主线是先识别客户顾虑，再通过连续提问确认需求，最后沉淀跟进动作。

## 分段总结
- 00:00:00 - 00:00:08：开头解释客户画像和建立信任的基本原则。
- 00:10:00 - 00:10:12：中段展开陌客沟通的问题链和需求确认。
- 00:20:00 - 00:20:15：结尾要求把跟进动作和复盘清单沉淀下来。

## 关键观点
- 00:00:00 信任建立不是话术堆砌，而是围绕客户顾虑给出连续证据。
- 00:10:00 问题链能把陌客沟通从闲聊推向需求确认。

## 可执行动作清单
- 00:00:00 先写出客户画像和常见顾虑。
- 00:10:00 为每类顾虑准备连续提问。
- 00:20:00 每次沟通后记录跟进动作和复盘清单。

## 高频话术
- 00:00:00 “你现在最担心的是哪一块？”
- 00:10:00 “如果这个问题解决了，下一步你会怎么判断？”

## 待复核点
- 视觉证据未执行或不足，屏幕细节仍需复核。
""",
        encoding="utf-8",
    )

    quality = smart_summary_quality_check(bundle, summary_path=summary, require_codex=True, write=False)
    checks = {row["key"]: row for row in quality["checks"]}

    assert checks["corrected_transcript_input"]["passed"] is False
    assert "corrected/source-arbitrated transcript" in checks["corrected_transcript_input"]["detail"]

    postprocess_asr_transcript(bundle, target_seconds=30, max_chars=120)
    corrected_path = bundle / "corrected-transcript.json"
    os.utime(corrected_path, None)
    quality_after = smart_summary_quality_check(bundle, summary_path=summary, require_codex=True, write=False)
    checks_after = {row["key"]: row for row in quality_after["checks"]}
    assert checks_after["corrected_transcript_input"]["passed"] is True
    assert checks_after["summary_after_corrected_transcript"]["passed"] is False
    assert quality_after["summary_freshness_gate"]["status"] == "summary_stale_after_transcript_update"

    os.utime(summary, None)
    refreshed_quality = smart_summary_quality_check(bundle, summary_path=summary, require_codex=True, write=False)
    refreshed_checks = {row["key"]: row for row in refreshed_quality["checks"]}
    assert refreshed_checks["summary_after_corrected_transcript"]["passed"] is True


def test_smart_summary_quality_prefers_canonical_source_over_stale_readable_sidecar(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_arbitrated_transcript_json"] = "source-arbitrated-transcript.json"
    manifest["readable_transcript_json"] = "readable-transcript.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    canonical = bundle / "source-arbitrated-transcript.json"
    canonical.write_text(
        json.dumps({"segments": [{"start": 0, "end": 8, "text": "人工确认后的 canonical 正文。"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "readable-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 8, "text": "尚未回流的旧正文。"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    exports = bundle / "exports"
    exports.mkdir()
    (exports / "smart-summary-input-pack.json").write_text(
        json.dumps(
            {
                "transcript_source": str(canonical),
                "transcript_source_label": "source_arbitrated_transcript",
                "transcript_source_decision": {
                    "uses_corrected_transcript": True,
                    "selected_label": "source_arbitrated_transcript",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = exports / "smart-summary.md"
    summary.write_text("# 摘要\n\n## 一句话概览\n\n待完善。\n", encoding="utf-8")

    quality = smart_summary_quality_check(bundle, summary_path=summary, require_codex=True, write=False)

    assert quality["transcript_source_gate"]["passed"] is True
    assert quality["transcript_source_gate"]["status"] == "corrected"
    assert quality["transcript_source_gate"]["transcript_source"] == str(canonical)
    assert quality["transcript_source_gate"]["transcript_source_label"] == "source_arbitrated_transcript"

def test_smart_summary_quality_blocks_canonical_transcript_with_review_gap(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_arbitrated_transcript_json"] = (
        "source-arbitrated-transcript.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    canonical = bundle / "source-arbitrated-transcript.json"
    canonical.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
                "segments": [
                    {
                        "index": 1,
                        "start": 0,
                        "end": 30,
                        "text": "\u6210\u529f\u5730\u52fe\u8d77\u4e86\u5ba2\u6237\u7684\u5174\u8da3\u70b9\u3002",
                        "needs_human_review": True,
                    }
                ],
                "quality_summary": {
                    "status": "needs_review",
                    "review_required": True,
                    "can_use_as_summary_input": False,
                    "review_segment_refs": [
                        {
                            "index": 1,
                            "start": 0,
                            "end": 30,
                            "reason": "upstream_asr_review:low_text_density",
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    exports = bundle / "exports"
    exports.mkdir()
    (exports / "smart-summary-input-pack.json").write_text(
        json.dumps(
            {
                "transcript_source": str(canonical),
                "transcript_source_label": "source_arbitrated_transcript",
                "transcript_source_decision": {
                    "uses_corrected_transcript": True,
                    "selected_label": "source_arbitrated_transcript",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = exports / "smart-summary.md"
    summary.write_text(
        "# \u6458\u8981\n\n## \u4e00\u53e5\u8bdd\u6982\u89c8\n\n\u5f85\u5b8c\u5584\u3002\n", encoding="utf-8"
    )

    quality = smart_summary_quality_check(
        bundle, summary_path=summary, require_codex=True, write=False
    )
    checks = {row["key"]: row for row in quality["checks"]}

    assert quality["transcript_source_gate"]["passed"] is False
    assert (
        quality["transcript_source_gate"]["status"]
        == "transcript_content_gaps"
    )
    assert quality["transcript_source_gate"]["review_segment_count"] == 1
    assert checks["corrected_transcript_input"]["passed"] is False
    assert quality["passed"] is False


def test_human_key_point_recall_is_not_passed_when_not_evaluated(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    exports = bundle / "exports"
    exports.mkdir(exist_ok=True)
    summary = exports / "smart-summary.md"
    summary.write_text(
        "# \u6458\u8981\n\n## \u4e00\u53e5\u8bdd\u6982\u89c8\n\n\u5f85\u5b8c\u5584\u3002\n", encoding="utf-8"
    )

    quality = smart_summary_quality_check(
        bundle, summary_path=summary, require_codex=False, write=False
    )
    check = next(
        row
        for row in quality["checks"]
        if row["key"] == "human_key_point_recall"
    )

    assert check["evaluated"] is False
    assert check["passed"] is None
    assert check["status"] == "not_evaluated"
    assert "automated_checks_passed" in quality
    assert quality["quality_evidence_complete"] is False
    assert quality["production_ready"] is False
    assert quality["passed"] is False
    assert quality["status"] != "passed"



def test_postprocess_asr_transcript_readable_mode_writes_readable_sidecar(tmp_path: Path) -> None:
    bundle = tmp_path / "readable-transcript-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 4, "text": "那第二点就是时间比较短"},
                    {"start": 4, "end": 8, "text": "同时展现了一个同理心"},
                    {"start": 8, "end": 12, "text": "接下来客户开始约时间"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = postprocess_asr_transcript(
        bundle,
        target_seconds=30,
        max_chars=120,
        segment_policy="readable_merge",
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    readable = json.loads((bundle / "readable-transcript.json").read_text(encoding="utf-8"))
    corrected = json.loads((bundle / "corrected-transcript.json").read_text(encoding="utf-8"))
    text = readable["segments"][0]["text"]
    assert result["punctuation_mode"] == "readable"
    assert manifest["readable_transcript_json"] == "readable-transcript.json"
    assert manifest["corrected_transcript_json"] == "corrected-transcript.json"
    assert "，" in text
    assert text.endswith("。")
    assert corrected["segments"][0]["text"] == text


def test_postprocess_asr_transcript_conservative_mode_keeps_terminal_only(tmp_path: Path) -> None:
    bundle = tmp_path / "conservative-transcript-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 4, "text": "第一段没有内部标点"}, {"start": 4, "end": 8, "text": "第二段继续说明"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    postprocess_asr_transcript(bundle, target_seconds=30, max_chars=120, punctuation_mode="conservative")

    corrected = json.loads((bundle / "corrected-transcript.json").read_text(encoding="utf-8"))
    text = corrected["segments"][0]["text"]
    assert "，" not in text
    assert text.endswith("。")


def test_long_video_compression_target_requires_complete_navigation() -> None:
    assert _compression_target(
        transcript_chars=46681,
        transcript_max=10214.035,
        coverage_ratio=1.0,
        section_coverage_passed=True,
    ) == (0.05, 0.30)
    assert _compression_target(
        transcript_chars=46681,
        transcript_max=10214.035,
        coverage_ratio=1.0,
        section_coverage_passed=False,
    ) == (0.12, 0.30)
