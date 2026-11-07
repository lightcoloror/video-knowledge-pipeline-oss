from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.bundle_next import bundle_next_action
from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.smart_summary_chapters import _chapter_citation_digest, _chapter_evidence_trace
from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
from video_knowledge_pipeline.task_console import _temporal_tag_delta_html
from video_knowledge_pipeline.video_workbench import _evidence_status_panel_html
from video_knowledge_pipeline.visual_evidence import continuous_tag_language_gate, evidence_level_counts


def _weak_tag_delta() -> dict:
    return {
        "schema": "video_knowledge_pipeline.temporal_tag_delta.v2",
        "decision": "coarse_weak_evidence",
        "evidence_level": "candidate_weak",
        "stable_tags": ["讲师", "课件"],
        "coarse_summary": "画面标签显示可能持续包含：讲师、课件；不据此推断动作或因果。",
        "raw_observations": [
            {"frame_id": "frame-1", "frame_path": "f1.jpg", "tags": ["讲师", "课件"]},
            {"frame_id": "frame-2", "frame_path": "f2.jpg", "tags": ["讲师", "课件"]},
        ],
        "run_id": "synthetic-tag-run",
    }


def _timeline_item(*, decision: str = "coarse_weak_evidence") -> dict:
    tag = _weak_tag_delta()
    tag["decision"] = decision
    if decision != "coarse_weak_evidence":
        tag["evidence_level"] = "candidate_weak" if decision == "temporal_multimodal_required" else "unavailable"
    return {
        "index": 1,
        "start": 0,
        "end": 10,
        "transcript": "讲师介绍课件内容。",
        "visual_route": "temporal_sequence",
        "temporal_frame_paths": ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"],
        "temporal_tag_delta": tag,
    }


def test_continuous_tag_is_separate_weak_evidence_and_closes_static_machine_gap() -> None:
    timeline = [_timeline_item()]
    levels = evidence_level_counts(timeline)
    coverage = build_knowledge_coverage({}, timeline)
    temporal_channel = next(row for row in coverage["channels"] if row["key"] == "temporal_visual_understanding")

    assert levels["continuous_tag"]["candidate_weak"] == 1
    assert levels["temporal"]["confirmed"] == 0
    assert coverage["items_with_continuous_tag_weak_evidence"] == 1
    assert temporal_channel["status"] == "candidate_weak"
    assert temporal_channel["covered_count"] == 1
    assert temporal_channel not in coverage["blockers"]


def test_candidate_weak_temporal_channel_does_not_request_unnecessary_multimodal() -> None:
    item = _timeline_item()
    item["visual_text"] = "静态课件标题"
    item["assets"] = [{"path": "f1.jpg", "exists": True}]
    coverage = build_knowledge_coverage({}, [item])

    assert coverage["status"] == "ok"
    assert coverage["next_action"]["key"] == "ready_for_review"


def test_dynamic_tag_delta_does_not_close_temporal_understanding_gap() -> None:
    coverage = build_knowledge_coverage({}, [_timeline_item(decision="temporal_multimodal_required")])
    temporal_channel = next(row for row in coverage["channels"] if row["key"] == "temporal_visual_understanding")
    assert temporal_channel["status"] == "blocked"
    assert temporal_channel["covered_count"] == 0


def test_smart_summary_and_citation_digest_preserve_candidate_weak_level(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"title": "Synthetic"}), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([_timeline_item()], ensure_ascii=False), encoding="utf-8")

    pack = build_smart_summary_input_pack(bundle, write=False)
    trace = pack["evidence_trace"]
    chapter_trace = _chapter_evidence_trace(trace, pack["transcript_segments"], 0, 10)
    citations = _chapter_citation_digest(pack["transcript_segments"], chapter_trace)

    assert pack["visual_digest"]["items"][0]["continuous_tag_weak_evidence"].startswith("画面标签显示可能")
    assert trace["continuous_tag_items"][0]["evidence_level"] == "candidate_weak"
    weak_citation = next(row for row in citations if row["source_type"] == "continuous_tag_weak_evidence")
    assert weak_citation["evidence_level"] == "candidate_weak"
    assert weak_citation["evidence_paths"]


def test_summary_quality_rejects_unqualified_or_action_claim_from_tags(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "timeline.json").write_text(json.dumps([_timeline_item()], ensure_ascii=False), encoding="utf-8")

    timeline = [_timeline_item()]
    unqualified = continuous_tag_language_gate(timeline, "## 分段总结\n画面包含讲师和课件。\n## 待复核点\n")
    qualified = continuous_tag_language_gate(timeline, "## 分段总结\n画面标签显示可能包含讲师和课件。\n## 待复核点\n")
    action = continuous_tag_language_gate(timeline, "## 分段总结\n画面标签显示讲师可能点击课件。\n## 待复核点\n")

    assert unqualified["passed"] is False
    assert qualified["passed"] is True
    assert action["passed"] is False
    assert any(row.get("reason") == "tag_only_action_or_causality_claim" for row in action["violations"])


def test_bundle_next_inserts_temporal_tagging_before_multimodal(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"):
        (bundle / name).write_bytes(name.encode())
    item = _timeline_item()
    item.pop("temporal_tag_delta")
    manifest = {
        "title": "Synthetic",
        "knowledge_coverage": {
            "status": "blocked",
            "blockers": [],
            "weak_channels": [],
            "next_action": {
                "key": "temporal_visual_understanding",
                "mcp_tool": "run_temporal_visual_analysis",
                "label": "连续片段理解",
            },
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")

    action = bundle_next_action(bundle, refresh=False)["next_action"]
    assert action["key"] == "temporal_tag_delta"
    assert action["mcp_tool"] == "run_temporal_tag_delta"
    assert Path(action["mcp_args_path"]).is_file()


def test_bundle_next_repairs_short_temporal_group_before_tagging(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    item = _timeline_item()
    item.pop("temporal_tag_delta")
    item["temporal_frame_paths"] = item["temporal_frame_paths"][:4]
    manifest = {
        "knowledge_coverage": {
            "status": "blocked",
            "blockers": [],
            "weak_channels": [],
            "next_action": {
                "key": "temporal_visual_understanding",
                "mcp_tool": "run_temporal_visual_analysis",
                "label": "连续片段理解",
            },
        }
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([item]), encoding="utf-8")

    action = bundle_next_action(bundle, refresh=False)["next_action"]

    assert action["key"] == "temporal_frame_groups"
    assert action["mcp_tool"] == "run_temporal_frame_groups"


def test_workbench_and_task_console_show_continuous_tag_status(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"temporal_tag_delta": {"status": "completed", "run_id": "run-1"}}),
        encoding="utf-8",
    )
    status = {
        "visual_evidence_levels": evidence_level_counts([_timeline_item()]),
        "temporal_tag_delta": {"status": "completed", "run_id": "run-1", "commands": {"run": "run-tag", "resume": "resume-tag"}},
    }
    workbench = _evidence_status_panel_html(status)
    console = _temporal_tag_delta_html(status, root)
    assert "连续打标" in workbench and "candidate_weak" not in workbench
    assert "运行连续打标" in workbench and "恢复连续打标" in workbench
    assert "candidate_weak" in console and "run-1" in console
