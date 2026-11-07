from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.knowledge_note_export import (
    _build_benchmark_script,
    _content_candidate_items,
    _render_benchmark_script_markdown,
    export_knowledge_note,
)


def _quality_ready() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "status": "formal_generation_allowed",
            "formal_generation_allowed": True,
            "blocking_reasons": [],
        },
        {"status": "passed", "passed": True},
    )


def _long_timeline() -> list[dict[str, object]]:
    return [
        {"index": 1, "start": 0, "end": 20, "transcript": "开场提出一个明确的问题，并说明接下来会给出判断框架。"},
        {"index": 2, "start": 50, "end": 70, "transcript": "前段补充背景和定义，让观众知道这个问题为什么值得处理。"},
        {"index": 3, "start": 120, "end": 140, "transcript": "中段展开核心论证、方法步骤和一个可以回看的案例。"},
        {"index": 4, "start": 210, "end": 230, "transcript": "后段讨论限制条件并把前面的论证收束成行动框架。"},
        {"index": 5, "start": 270, "end": 300, "transcript": "结尾复盘关键判断，并邀请观众根据自己的情况继续核对。"},
    ]


def test_long_benchmark_uses_all_timeline_strata_and_passes_coverage(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    production_gate, summary_quality = _quality_ready()

    script = _build_benchmark_script(
        bundle,
        "Long benchmark",
        _long_timeline(),
        production_artifact_gate=production_gate,
        smart_summary_quality=summary_quality,
    )
    markdown = _render_benchmark_script_markdown(script)

    assert script["status"] == "needs_review"
    assert script["final_allowed"] is False
    assert script["coverage"]["passed"] is True
    assert script["coverage"]["selected_strata_count"] == 5
    assert script["coverage"]["latest_evidence_ratio"] == 1.0
    assert script["selection"]["selected_strata"] == [
        "opening",
        "early",
        "middle",
        "late",
        "conclusion",
    ]
    assert [row["timeline_index"] for row in script["structure_analysis"]] == [1, 2, 3, 4, 5]
    assert len(script["timestamp_evidence_links"]) >= 5
    assert "## 结构拆解" in markdown
    assert "## 开头钩子" in markdown
    assert "## 论证节奏 / Beats" in markdown
    assert "## 视觉表达" in markdown
    assert "## CTA" in markdown
    assert "## 原创性说明" in markdown
    assert "## 风险与待复核项" in markdown


def test_long_benchmark_stays_draft_when_late_evidence_is_missing(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    timeline = [
        {"index": 1, "start": 0, "end": 20, "transcript": "开头有很多文字，分数很高。" * 20},
        {"index": 2, "start": 40, "end": 60, "transcript": "前段证据。"},
        {"index": 3, "start": 70, "end": 75, "transcript": "仍然只有前段证据。"},
        {"index": 4, "start": 280, "end": 300},
    ]
    production_gate, summary_quality = _quality_ready()

    script = _build_benchmark_script(
        bundle,
        "Incomplete benchmark",
        timeline,
        production_artifact_gate=production_gate,
        smart_summary_quality=summary_quality,
    )

    assert script["status"] == "draft"
    assert script["final_allowed"] is False
    assert script["coverage"]["passed"] is False
    assert "long_video_requires_four_timeline_strata" in script["coverage"]["blocking_reasons"]
    assert "long_video_latest_evidence_before_eighty_percent" in script["coverage"]["blocking_reasons"]
    assert script["coverage"]["latest_evidence_ratio"] == 0.25
    assert any("覆盖门未通过" in note for note in script["risk_notes"])


def test_knowledge_export_writes_new_benchmark_artifacts_and_keeps_legacy_draft(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    timeline = _long_timeline()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Benchmark export",
                "normalized_transcript_json": "normalized-transcript.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False), encoding="utf-8"
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": row["start"], "end": row["end"], "text": row["transcript"]}
                    for row in timeline
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = export_knowledge_note(
        bundle,
        title="Benchmark export",
        run_transcript_evidence_check=False,
    )

    benchmark_json = Path(result["benchmark_script_json_path"])
    benchmark_markdown = Path(result["benchmark_script_markdown_path"])
    legacy_markdown = Path(result["content_assets"]["short_video_script_drafts_path"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))

    assert benchmark_json.exists()
    assert benchmark_markdown.exists()
    assert payload["schema"] == "video_knowledge_pipeline.benchmark_script.v1"
    assert payload["status"] == "draft"
    assert payload["final_allowed"] is False
    assert result["content_assets"]["benchmark_script_json_path"] == str(benchmark_json)
    assert result["content_assets"]["benchmark_script_markdown_path"] == str(benchmark_markdown)
    assert result["run_registry"]["status"] == "completed"
    assert manifest["benchmark_script_json"] == "exports/benchmark-script.json"
    assert manifest["benchmark_script_markdown"] == "exports/benchmark-script.md"
    legacy_text = legacy_markdown.read_text(encoding="utf-8")
    assert "Selection: `timeline_stratified`" in legacy_text
    assert "Timeline stratum: `opening`" in legacy_text
    assert "Timeline stratum: `conclusion`" in legacy_text
    assert [row["index"] for row in _content_candidate_items(timeline)[:5]] == [1, 2, 3, 4, 5]
