from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_codex import (
    smart_summary_quality_check,
    write_smart_summary_dependency_snapshot,
)


def test_quality_number_gate_uses_only_trusted_ebook_ocr_evidence(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    exports = bundle / "exports"
    exports.mkdir(parents=True)
    title = "2026\u5e747\u670824\u65e5\u5168\u56fd\u5927\u65e9\u4f1a"
    (bundle / "manifest.json").write_text(json.dumps({"title": title, "source_arbitrated_transcript_json": "source-arbitrated-transcript.json"}), encoding="utf-8")
    transcript = bundle / "source-arbitrated-transcript.json"
    transcript.write_text(json.dumps({"schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1", "segments": [{"start": 0, "end": 1200, "text": "evidence " * 400}]}, ensure_ascii=False), encoding="utf-8")
    (bundle / "transcript-semantic-correction-pack.json").write_text(json.dumps({"status": "no_candidates", "candidate_count": 0}), encoding="utf-8")
    (exports / "smart-summary-input-pack.json").write_text(json.dumps({"transcript_source": str(transcript), "transcript_source_label": "source_arbitrated_transcript", "transcript_source_decision": {"uses_corrected_transcript": True, "selected_label": "source_arbitrated_transcript"}}), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([
        {"index": 1, "start": 60, "end": 70, "ebook_pipeline_status": {"ok": True, "artifact_path": "ebook/book.md"}, "structured_visual": [{"source": "ebook_markdown_pipeline", "artifact_path": "ebook/book.md", "markdown": "\u5ba2\u6237100%\u6765\u81ea\u4e92\u8054\u7f51\uff0c\u4fdd\u8d393000\u4e07"}]},
        {"index": 2, "start": 80, "end": 90, "ebook_pipeline_status": {"ok": False, "artifact_path": "ebook/failed.md"}, "structured_visual": [{"source": "ebook_markdown_pipeline", "artifact_path": "ebook/failed.md", "markdown": "\u4e1a\u7ee93300"}]},
    ], ensure_ascii=False), encoding="utf-8")
    summary = exports / "smart-summary.codex.md"
    summary.write_text("\\n\\n".join([
        f"# {title} - \u667a\u80fd\u603b\u7ed3",
        "\u751f\u6210\u65b9\u5f0f\uff1a`codex_llm_rewrite_final`",
        f"## \u4e00\u53e5\u8bdd\u6982\u89c8\\n{title}\u663e\u793a\u5ba2\u6237100%\u6765\u81ea\u4e92\u8054\u7f51\uff0c\u4fdd\u8d393000\u4e07\uff1b3300\u4ecd\u5f85\u8bc1\u636e\u6838\u9a8c\u3002",
        "## \u6838\u5fc3\u4e3b\u9898\\n\u56de\u5230\u8bc1\u636e\u505a\u51b3\u7b56\u3002",
        "## \u5206\u6bb5\u603b\u7ed3\\n- 00:00:00 - 00:20:00\uff1a\u6309\u65f6\u95f4\u590d\u76d8\u8bfe\u7a0b\u3002",
        "## \u5173\u952e\u89c2\u70b9\\n- 00:01:00 \u6570\u5b57\u9700\u56de\u94fe\u539f\u59cb\u8bc1\u636e\u3002",
        "## \u53ef\u6267\u884c\u52a8\u4f5c\u6e05\u5355\\n- 00:02:00 \u5148\u6838\u9a8c\u518d\u4f7f\u7528\u3002",
        "## \u9ad8\u9891\u8bdd\u672f\\n- 00:03:00 \u8bf7\u5148\u770b\u8bc1\u636e\u3002",
        "## \u5f85\u590d\u6838\u70b9\\n- \u89c6\u89c9\u8bc1\u636e\u672a\u6267\u884c\u6216\u5f85\u590d\u6838\u3002",
    ]), encoding="utf-8")

    result = smart_summary_quality_check(bundle, summary_path=summary, write=False)
    checks = {row["key"]: row for row in result["checks"]}

    assert checks["number_consistency"]["passed"] is False
    assert result["quality_metrics"]["unsupported_numbers"] == ["3300"]
    support = result["quality_metrics"]["number_evidence"]["supporting_claims"]
    assert support["100%"][0]["source_kind"] == "ebook_ocr"
    assert support["100%"][0]["timeline_index"] == 1
    assert support["3000"][0]["artifact_path"] == "ebook/book.md"


def test_quality_marks_summary_stale_after_timeline_evidence_changes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    exports = bundle / "exports"
    exports.mkdir(parents=True)
    transcript = bundle / "source-arbitrated-transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
                "segments": [{"start": 0, "end": 600, "text": "evidence " * 400}],
            }
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps({"source_arbitrated_transcript_json": "source-arbitrated-transcript.json"}),
        encoding="utf-8",
    )
    (exports / "smart-summary-input-pack.json").write_text(
        json.dumps(
            {
                "transcript_source": str(transcript),
                "transcript_source_decision": {"uses_corrected_transcript": True},
            }
        ),
        encoding="utf-8",
    )
    timeline = bundle / "timeline.json"
    timeline.write_text(json.dumps([{"start": 0, "end": 600, "text": "initial OCR evidence"}]), encoding="utf-8")
    summary = exports / "smart-summary.codex.md"
    summary.write_text(
        "\n".join(
            [
                "生成方式：`codex_llm_rewrite_final`",
                "## 一句话概览\n这是一条足够长的概览，用于验证证据快照失效。",
                "## 核心主题\n- 证据优先。",
                "## 分段总结\n- 00:00:00 - 00:10:00：按证据回顾。",
                "## 关键观点\n- 00:01:00 先核验。",
                "## 可执行动作清单\n- 00:02:00 记录来源。",
                "## 高频话术\n- 00:03:00 请看证据。",
                "## 待复核点\n- 视觉证据待复核。",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = write_smart_summary_dependency_snapshot(bundle)
    assert snapshot["status"] == "recorded"
    fresh = smart_summary_quality_check(bundle, summary_path=summary, write=False)
    fresh_check = {row["key"]: row for row in fresh["checks"]}["summary_after_evidence_update"]
    assert fresh_check["passed"] is True

    # The input pack includes generated metadata and may be rebuilt by export. It
    # must not become a false evidence change; its stable source artifacts are
    # already tracked by the shared dependency snapshot.
    (exports / "smart-summary-input-pack.json").write_text(
        json.dumps(
            {
                "transcript_source": str(transcript),
                "transcript_source_decision": {"uses_corrected_transcript": True},
                "generated_at": "later",
            }
        ),
        encoding="utf-8",
    )
    refreshed_pack = smart_summary_quality_check(bundle, summary_path=summary, write=False)
    refreshed_pack_check = {row["key"]: row for row in refreshed_pack["checks"]}["summary_after_evidence_update"]
    assert refreshed_pack_check["passed"] is True

    timeline.write_text(json.dumps([{"start": 0, "end": 600, "text": "updated OCR evidence"}]), encoding="utf-8")
    stale = smart_summary_quality_check(bundle, summary_path=summary, write=False)
    stale_check = {row["key"]: row for row in stale["checks"]}["summary_after_evidence_update"]
    assert stale_check["passed"] is False
    assert stale["summary_evidence_freshness_gate"]["status"] == "summary_stale_after_evidence_stale"
