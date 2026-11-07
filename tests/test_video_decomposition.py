from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser, main as cli_main
from video_knowledge_pipeline.video_decomposition import (
    FINDING_DIMENSIONS,
    REPORT_SCHEMA,
    VideoDecompositionContractError,
    build_video_decomposition_report,
    compare_video_decomposition_reports,
    validate_video_decomposition_report,
    video_decomposition_report_status,
)
from video_knowledge_pipeline.video_workbench import export_video_workbench


CONSUMER_SRC = Path(
    os.environ.get("VKP_VIDEO_CREATION_PIPELINE_SRC") or Path(__file__).resolve().parents[2] / "ai-video-tools-20260708" / "mvp-video-pipeline" / "src"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, title: str = "拆解测试", ordinal: int = 1) -> Path:
    root = tmp_path / f"bundle-{ordinal}"
    exports = root / "exports"
    source = root / "source"
    frame = root / "frames" / "001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(f"frame-{ordinal}".encode())
    transcript = root / "source-arbitrated-transcript.json"
    _write_json(
        transcript,
        {
            "schema": "video_knowledge_pipeline.source_arbitrated_transcript.v1",
            "segments": [
                {"start": 0, "end": 4, "text": f"第{ordinal}条视频先展示结果。"},
                {"start": 4, "end": 9, "text": "随后解释关键步骤。"},
                {"start": 9, "end": 14, "text": "最后给出行动建议。"},
            ],
        },
    )
    _write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0,
                "end": 4,
                "corrected_transcript": f"第{ordinal}条视频先展示结果。",
                "visual_text": "结果展示",
                "visual_understanding": {"objects": ["演示画面"], "actions": ["展示结果"]},
                "frame_paths": ["frames/001.jpg"],
            },
            {"index": 2, "start": 4, "end": 9, "corrected_transcript": "随后解释关键步骤。"},
            {
                "index": 3,
                "start": 9,
                "end": 14,
                "corrected_transcript": "最后给出行动建议。",
                "temporal_visual_understanding": {"event_sequence": ["讲解收束"]},
            },
        ],
    )
    exports.mkdir(parents=True)
    (exports / "smart-summary.codex.md").write_text(
        "# 智能总结\n\n## 核心主题\n\n展示结果、解释步骤并给出行动建议。\n",
        encoding="utf-8",
    )
    (exports / "companion-courseware-text.md").write_text(
        "# 配套课件\n\n结果、步骤、行动。\n",
        encoding="utf-8",
    )
    _write_json(
        source / "page-metadata.json",
        {"schema": "video_knowledge_pipeline.page_metadata.v1", "title": title},
    )
    _write_json(root / "run-artifact-registry.json", {"schema": "registry", "runs": []})
    _write_json(
        root / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": title,
            "source_arbitrated_transcript_json": transcript.name,
            "page_metadata_json": "source/page-metadata.json",
            "companion_courseware_text_markdown": "exports/companion-courseware-text.md",
        },
    )
    return root


def _rehash(report: dict) -> None:
    value = dict(report)
    value.pop("report_sha256", None)
    report["report_sha256"] = canonical_json_sha256(value)


def test_report_is_read_only_exact_and_consumer_compatible(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    protected = [
        bundle / "timeline.json",
        bundle / "source-arbitrated-transcript.json",
        bundle / "exports" / "smart-summary.codex.md",
        bundle / "run-artifact-registry.json",
    ]
    before = {path: _sha(path) for path in protected}

    report = build_video_decomposition_report(bundle, write=True)

    assert report["schema"] == REPORT_SCHEMA
    assert set(report) >= {
        "report_id",
        "title",
        "source_artifacts",
        "modality_coverage",
        "findings",
        "structure_segments",
        "creative_strategy",
        "report_sha256",
    }
    assert len(report["single_video_layers"]) == 5
    assert report["operator_boundary"]["run_registry_mutated"] is False
    assert {path: _sha(path) for path in protected} == before
    assert video_decomposition_report_status(bundle)["status"] == "fresh"
    assert (bundle / "exports" / "video-decomposition-report.json").is_file()
    assert (bundle / "exports" / "video-decomposition-report.md").is_file()

    if not CONSUMER_SRC.is_dir():
        pytest.skip("video creation consumer checkout is not available")
    sys.path.insert(0, str(CONSUMER_SRC))
    try:
        from video_creation_pipeline.decomposition_adoption import (
            validate_vkp_decomposition_report,
        )

        validate_vkp_decomposition_report(report)
    finally:
        sys.path.remove(str(CONSUMER_SRC))


def test_reader_sections_do_not_promote_preroll_to_hook_or_positioning(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    timeline_path = bundle / "timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline[0]["corrected_transcript"] = "能听得见声音吗？再等两分钟。"
    timeline[-1]["corrected_transcript"] = "好的，谢谢大家。"
    _write_json(timeline_path, timeline)
    summary_path = bundle / "exports" / "smart-summary.codex.md"
    summary_path.write_text(
        "# 智能总结\n\n"
        "生成方式：online_llm_section_rewrite\n\n"
        "## 基本信息\n\n- 视频名：测试视频\n\n"
        "## 一句话概览\n\n这份课程解释客户经营的核心方法与执行步骤。\n\n"
        "## 核心主题 / 课程主线\n\n"
        "- 课程主线：从客户问题出发，依次讲解方法、案例与行动。\n"
        "- 章节来源：内部生成元数据，不应进入拆解定位。\n",
        encoding="utf-8",
    )

    report = build_video_decomposition_report(bundle, write=True)
    by_dimension = {
        dimension: [row for row in report["findings"] if row["dimension"] == dimension]
        for dimension in ("positioning", "content_value", "hook", "payoff")
    }

    positioning = by_dimension["positioning"][0]
    content_value = by_dimension["content_value"][0]
    assert positioning["status"] == "confirmed"
    assert "课程主线" in positioning["claim"]
    assert "能听得见" not in positioning["claim"]
    assert "生成方式" not in positioning["claim"]
    assert any(evidence_id.startswith("smart-summary-") for evidence_id in positioning["evidence_ids"])
    assert "这份课程解释客户经营" in content_value["claim"]
    assert "基本信息" not in content_value["claim"]
    assert by_dimension["hook"][0]["status"] == "unavailable"
    assert by_dimension["hook"][0]["evidence_ids"] == []
    assert by_dimension["payoff"][0]["status"] == "unavailable"
    assert by_dimension["payoff"][0]["evidence_ids"] == []
    assert not any(
        row["dimension"] == "reusable_framework" and row["status"] == "confirmed"
        for row in report["findings"]
    )
    assert all(row["label"].startswith("时间证据锚点") for row in report["structure_segments"])

    summary_path.write_text(summary_path.read_text(encoding="utf-8") + "\n输入已变化。\n", encoding="utf-8")
    assert video_decomposition_report_status(bundle)["status"] == "stale"


def test_report_invalidates_when_timeline_changes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    report = build_video_decomposition_report(bundle, write=True)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["corrected_transcript"] = "输入已经变化。"
    _write_json(bundle / "timeline.json", timeline)

    status = video_decomposition_report_status(bundle)

    assert status["status"] == "stale"
    assert any(row["key"] == "input_changed" for row in status["issues"])
    with pytest.raises(VideoDecompositionContractError, match="changed after"):
        validate_video_decomposition_report(report)


def test_inferred_finding_cannot_override_confirmed_finding(tmp_path: Path) -> None:
    report = build_video_decomposition_report(_bundle(tmp_path), write=False)
    source_ids = [row["artifact_id"] for row in report["source_artifacts"]]
    assert len(source_ids) >= 2
    report["findings"].append(
        {
            "finding_id": "positioning-inferred-duplicate",
            "dimension": "positioning",
            "status": "inferred",
            "claim": "候选推断不应覆盖已确认定位。",
            "time_range": {"scope": "global", "start_s": None, "end_s": None},
            "source_modalities": ["transcript"],
            "evidence_ids": source_ids[:2],
            "evidence_paths": [],
            "missing_evidence": [],
            "direct_evidence_kind": "",
        }
    )
    _rehash(report)

    with pytest.raises(VideoDecompositionContractError, match="cannot override"):
        validate_video_decomposition_report(report, check_source_artifacts=False)


def test_sensitive_gaps_remain_unavailable_without_guesses(tmp_path: Path) -> None:
    report = build_video_decomposition_report(_bundle(tmp_path), write=False)
    by_dimension = {row["dimension"]: row for row in report["findings"]}
    for dimension in ("bgm_identity", "author_identity", "behind_the_scenes", "performance_metrics"):
        finding = by_dimension[dimension]
        assert finding["status"] == "unavailable"
        assert finding["evidence_ids"] == []
        assert finding["evidence_paths"] == []
        assert finding["missing_evidence"]


def test_five_report_comparison_uses_cards_and_uniform_matrix(tmp_path: Path) -> None:
    report_paths: list[Path] = []
    for ordinal in range(1, 6):
        bundle = _bundle(tmp_path, title=f"视频 {ordinal}", ordinal=ordinal)
        build_video_decomposition_report(bundle, write=True)
        report_paths.append(bundle / "exports" / "video-decomposition-report.json")
    output = tmp_path / "comparison"

    small_comparison = compare_video_decomposition_reports(report_paths[:2], write=False)
    comparison = compare_video_decomposition_reports(report_paths, output_dir=output, write=True)

    assert small_comparison["layout"] == "wide_table"
    assert comparison["layout"] == "cards_and_matrix"
    assert len(comparison["cards"]) == 5
    assert {row["dimension"] for row in comparison["uniform_matrix"]} == FINDING_DIMENSIONS
    markdown = (output / "video-decomposition-comparison.md").read_text(encoding="utf-8")
    assert "## 视频卡片" in markdown
    assert "## 统一尺度矩阵" in markdown


def test_cli_mcp_and_workbench_surface_freshness(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    parsed = build_parser().parse_args(["video-decomposition-report", str(bundle), "--no-write"])
    assert parsed.command == "video-decomposition-report"
    assert cli_main(["video-decomposition-report", str(bundle), "--no-write"]) == 0
    build_video_decomposition_report(bundle, write=True)
    audit = audit_bundle_mcp_args(bundle)
    row = next(item for item in audit["rows"] if item["key"] == "mcp_video_decomposition_report_args")
    assert row["tool"] == "video_decomposition_report"
    assert row["ok"] is True

    workbench = export_video_workbench(bundle, write=False)
    card = next(item for item in workbench["artifacts"] if item["key"] == "video_decomposition_report_json")
    assert card["status"] == "fresh"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["visual_text"] = "更新后的画面文字"
    _write_json(bundle / "timeline.json", timeline)
    stale = export_video_workbench(bundle, write=False)
    stale_card = next(item for item in stale["artifacts"] if item["key"] == "video_decomposition_report_json")
    assert stale_card["status"] == "stale"
