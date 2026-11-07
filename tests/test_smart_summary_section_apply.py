from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from video_knowledge_pipeline.smart_summary_section_apply import apply_smart_summary_sections


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundle() -> Path:
    root = Path("outputs") / ("pytest-smart-summary-section-apply-" + uuid4().hex) / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    _write_json(root / "manifest.json", {"title": "Section Apply Smoke", "media_path": "D:/media/smoke.mp4", "source_arbitrated_transcript_json": "source-arbitrated-transcript.json"})
    transcript_payload = {
        "segments": [
            {"start": 0, "end": 10, "text": "开头讲客户画像和信任建立。"},
            {"start": 1200, "end": 1210, "text": "中段讲问题链和需求确认。"},
            {"start": 2400, "end": 2410, "text": "后段讲跟进动作和复盘记录。"},
            {"start": 3590, "end": 3600, "text": "结尾总结客户沟通要沉淀成清单。"},
        ]
    }
    _write_json(root / "normalized-transcript.json", transcript_payload)
    _write_json(root / "source-arbitrated-transcript.json", transcript_payload)
    _write_json(
        root / "timeline.json",
        [
            {"index": 0, "start": 0, "end": 10, "transcript": "开头讲客户画像和信任建立。"},
            {"index": 1, "start": 1200, "end": 1210, "transcript": "中段讲问题链和需求确认。"},
            {"index": 2, "start": 2400, "end": 2410, "transcript": "后段讲跟进动作和复盘记录。"},
            {"index": 3, "start": 3590, "end": 3600, "transcript": "结尾总结客户沟通要沉淀成清单。"},
        ],
    )
    _write_json(root / "transcript-semantic-correction-pack.json", {"candidate_count": 0, "candidates": []})
    _write_json(
        exports / "human-key-points.json",
        {"key_points": ["\u5ba2\u6237\u6c9f\u901a\u3001\u786e\u8ba4\u95ee\u9898\u3001\u5b89\u6392\u52a8\u4f5c\u548c\u590d\u76d8"]},
    )
    _write_json(
        exports / "smart-summary-section-workflow.json",
        {
            "schema": "video_knowledge_pipeline.smart_summary_section_workflow.v1",
            "bundle_dir": str(root),
            "title": "Section Apply Smoke",
            "sections": [
                {"section_id": "chapter-0001", "chapter_index": 1, "title": "开头", "start": 0, "end": 10, "start_time": "00:00:00.000", "end_time": "00:00:10.000", "status": "needs_rewrite", "reasons": ["global_summary_quality_failed"]},
                {"section_id": "chapter-0002", "chapter_index": 2, "title": "中段", "start": 1200, "end": 1210, "start_time": "00:20:00.000", "end_time": "00:20:10.000", "status": "needs_rewrite", "reasons": ["global_summary_quality_failed"]},
                {"section_id": "chapter-0003", "chapter_index": 3, "title": "后段", "start": 2400, "end": 2410, "start_time": "00:40:00.000", "end_time": "00:40:10.000", "status": "needs_rewrite", "reasons": ["global_summary_quality_failed"]},
                {"section_id": "chapter-0004", "chapter_index": 4, "title": "结尾", "start": 3590, "end": 3600, "start_time": "00:59:50.000", "end_time": "01:00:00.000", "status": "needs_rewrite", "reasons": ["global_summary_quality_failed"]},
            ],
        },
    )
    _write_json(exports / "smart-summary-section-todo.json", {"rows": [{"section_id": "chapter-0001", "draft_markdown": ""}]})
    return root


def _revision_text(label: str) -> str:
    return f"{label}讲关键观点和方法论。核心判断是客户沟通要先确认问题，再安排动作。可执行动作：先记录事实，再确认下一步，最后复盘结果。可复用表达：你可以说我们先把已确认的点复述一遍。视觉证据未执行/待复核。"


def test_empty_section_revisions_need_input() -> None:
    root = _bundle()
    result = apply_smart_summary_sections(root, input_json=root / "exports" / "smart-summary-section-todo.json", write=True)
    assert result["status"] == "needs_section_revisions"
    assert result["quality_passed"] is False
    assert result["run_artifact"]["status"] == "needs_input"


def test_complete_section_revisions_install_codex_summary() -> None:
    root = _bundle()
    revisions = root / "section-revisions.json"
    _write_json(
        revisions,
        {
            "rows": [
                {"section_id": "chapter-0001", "final_markdown": _revision_text("开头")},
                {"section_id": "chapter-0002", "final_markdown": _revision_text("中段")},
                {"section_id": "chapter-0003", "final_markdown": _revision_text("后段")},
                {"section_id": "chapter-0004", "final_markdown": _revision_text("结尾")},
            ]
        },
    )
    result = apply_smart_summary_sections(root, input_json=revisions, require_all_sections=True, write=True)
    assert result["status"] == "blocked_quality"
    assert result["installed_section_count"] == 0
    assert result["candidate_section_count"] == 4
    # Section assembly remains a valid intermediate candidate, but the repeated
    # location labels and template actions must not masquerade as a mature final
    # reader summary without the evidence-bound global Reduce stage.
    assert result["quality_passed"] is False
    checks = {
        row["key"]: row
        for row in result["codex_status"]["quality"]["checks"]
    }
    assert checks["reader_semantic_maturity"]["passed"] is False
    assert result["run_artifact"]["status"] == "needs_retry"
    assert not (root / "exports" / "smart-summary.codex.md").exists()
    assert (root / "exports" / "smart-summary-section-candidate.md").exists()
