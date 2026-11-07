from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_pipeline.asr_runner import _command_for_preset
from video_knowledge_pipeline.cli import _mcp_callables
from video_knowledge_pipeline.qwen3_forced_aligner_runner import _cue_chunks, _timestamps_monotonic
from video_knowledge_pipeline.scene_detection_adapter import _normalise_scene_list, _uniform_cap
from video_knowledge_pipeline.smart_summary_global_reduce import _normalise_markdown, _reduce_prompt_plan, _shape_quality, _write, run_smart_summary_global_reduce
from video_knowledge_pipeline.smart_summary_codex import _quality_numbers
from video_knowledge_pipeline.summary_consistency import run_summary_consistency_check
from video_knowledge_pipeline.video_evidence_query import apply_video_evidence_confirmation


class _Time:
    def __init__(self, seconds: float):
        self.seconds = seconds

    def get_seconds(self) -> float:
        return self.seconds


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_phase17_reuse_closure_tools_are_registered_for_agents() -> None:
    tools = _mcp_callables()
    assert "asr_diff_adjudication" in tools
    assert "apply_asr_diff_adjudication" in tools
    assert "scene_detection" in tools
    assert "smart_summary_global_reduce" in tools
    assert "summary_consistency_check" in tools
    assert "video_evidence_query_plan" in tools
    assert "apply_video_evidence_confirmation" in tools


def test_qwen_forced_aligner_plan_uses_transcript_and_official_runner() -> None:
    command = _command_for_preset(
        preset="qwen3-forced-aligner",
        command="qwen-asr",
        media=Path("media.wav"),
        output_dir=Path("."),
        output_json=Path("output.json"),
        language="zh",
        model=None,
        punc_model=None,
        spk_model=None,
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
        vad_max_single_segment_time_ms=30000,
        chunk_boundary_mode="fixed_duration",
        chunk_overlap_seconds=5.0,
        python_executable="python",
        local_device="cuda",
        use_python_runner=True,
        qwen_timestamps=True,
        alignment_transcript=Path("transcript.json"),
    )

    assert "video_knowledge_pipeline.qwen3_forced_aligner_runner" in command
    assert command[command.index("--transcript") + 1] == "transcript.json"


def test_qwen_forced_aligner_chunks_and_monotonic_timestamp_gate() -> None:
    cues = [
        SimpleNamespace(start=0.0, end=10.0, text="第一句。"),
        SimpleNamespace(start=10.0, end=20.0, text="第二句。"),
        SimpleNamespace(start=320.0, end=330.0, text="第三句。"),
    ]

    chunks = _cue_chunks(cues, max_seconds=300)

    assert len(chunks) == 2
    assert chunks[0]["cue_indexes"] == [0, 1]
    assert _timestamps_monotonic([{"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0}]) is True
    assert _timestamps_monotonic([{"start": 1.0, "end": 1.5}, {"start": 0.5, "end": 1.0}]) is False


def test_pyscenedetect_boundaries_are_normalised_and_uniformly_capped() -> None:
    scenes = [(_Time(0), _Time(10)), (_Time(10), _Time(20)), (_Time(20), _Time(30)), (_Time(30), _Time(40))]

    rows, points = _normalise_scene_list(scenes, max_points=2)

    assert len(rows) == 4
    assert [row["seconds"] for row in points] == [10.0, 30.0]
    assert _uniform_cap(list(range(10)), max_points=3) == [0, 4, 9]

def test_smart_summary_number_gate_ignores_markdown_list_ordinals() -> None:
    assert _quality_numbers("1. 第一步\n2) 第二步\n3、第三步\n比例 15%") == ["15%"]


def test_global_reduce_blocks_incomplete_map_and_never_reads_raw_asr(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_json(root / "manifest.json", {"title": "课程"})
    _write_json(
        root / "exports" / "smart-summary-section-workflow.json",
        {"sections": [{"section_id": "s1"}, {"section_id": "s2"}]},
    )
    _write_json(
        root / "exports" / "smart-summary-section-llm-revisions.json",
        {"rows": [{"section_id": "s1", "final_markdown": "第一章内容"}]},
    )
    _write_json(root / "exports" / "course-map.json", {"mainline": "课程主线"})
    _write_json(root / "normalized-transcript.json", {"segments": [{"text": "不应直接进入 Reduce"}]})

    result = run_smart_summary_global_reduce(root, execute=False, write=False)

    assert result["status"] == "blocked_incomplete_map"
    assert result["operator_boundary"]["does_not_read_raw_asr"] is True


def test_global_reduce_balances_over_budget_map_without_dropping_late_sections(tmp_path: Path) -> None:
    rows = [
        {
            "section_id": f"s{index}",
            "title": f"第{index}章",
            "time_range": f"00:{index:02d}:00 - 00:{index + 1:02d}:00",
            "final_markdown": f"开头{index}-" + ("甲" * 1200) + f"-结尾{index}",
        }
        for index in range(1, 7)
    ]

    # The current full reader contract plus all six minimum excerpts needs
    # more than the historical 3000 characters. Keep both the fitting case
    # and the explicit fail-closed case; never remove late sections to fit.
    plan = _reduce_prompt_plan(tmp_path / "bundle", rows, {}, max_input_chars=3600)

    assert plan["full_input_chars"] > len(plan["prompt"])
    assert len(plan["prompt"]) <= 3600
    assert plan["all_sections_included"] is True
    assert plan["clipped_section_ids"] == ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert "开头1" in plan["prompt"]
    assert "结尾6" in plan["prompt"]
    assert "本章中段已按上下文预算压缩" in plan["prompt"]
    assert "900–1300" in plan["prompt"]
    assert "基本信息" in plan["prompt"]
    assert "每一项都必须带来源时间戳" in plan["prompt"]
    assert "00:01:00" in plan["prompt"]
    assert "00:07:00" in plan["prompt"]

    root = tmp_path / "bundle"
    _write_json(root / "manifest.json", {"title": "长课程"})
    _write_json(root / "exports" / "smart-summary-section-workflow.json", {"sections": [{"section_id": row["section_id"]} for row in rows]})
    _write_json(root / "exports" / "smart-summary-section-llm-revisions.json", {"rows": rows})
    _write_json(root / "exports" / "course-map.json", {"mainline": "全片主线"})

    blocked = run_smart_summary_global_reduce(root, execute=False, max_input_chars=3000, write=False)
    assert blocked["status"] == "blocked_reduce_input_budget"
    assert blocked["ok"] is False
    assert blocked["reduce_stage"]["minimum_required_input_chars"] > 3000
    assert blocked["reduce_stage"]["all_sections_included"] is True

    result = run_smart_summary_global_reduce(root, execute=False, max_input_chars=3600, write=False)

    assert result["status"] == "planned"
    assert result["reduce_stage"]["prompt_within_budget"] is True
    assert result["reduce_stage"]["all_sections_included"] is True
    assert result["reduce_stage"]["clipped_section_ids"] == ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert result["operator_boundary"]["late_chapters_not_dropped"] is True


def test_global_reduce_normalises_final_marker_for_installer() -> None:
    content = _normalise_markdown(
        "# 基本信息\n\n- 标题：课程\n\n# 一句话概览\n\n内容",
        title="课程",
    )

    assert content.startswith("# 课程 - 智能总结")
    assert "## 基本信息" in content
    assert "## 一句话概览" in content
    assert "生成方式：`codex_llm_rewrite_final`。" in content
    assert content.count("codex_llm_rewrite_final") == 1

def test_global_reduce_shape_requires_real_markdown_headings() -> None:
    rows = [{"time_range": "00:00:00.000 - 00:01:00.000"}]
    headings = (
        "基本信息",
        "一句话概览",
        "核心主题 / 课程主线",
        "分段总结",
        "关键观点 / 方法论",
        "可执行动作清单",
        "高频话术 / 可复用表达",
        "待复核点 / 低置信内容",
    )
    content = "生成方式：`codex_llm_rewrite_final`。\n" + "\n".join(
        f"## {heading}\n00:00:00.000 00:01:00.000 {'内容' * 50}" for heading in headings
    )

    assert _shape_quality(content, expected_ids={"s1"}, rows=rows)["passed"] is True
    assert _shape_quality(content.replace("## ", "# "), expected_ids={"s1"}, rows=rows)["passed"] is False

def test_global_reduce_reuses_persisted_candidate_without_model_call(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    _write_json(root / "manifest.json", {"title": "课程"})
    _write_json(exports / "smart-summary-section-workflow.json", {"sections": [{"section_id": "s1"}]})
    _write_json(
        exports / "smart-summary-section-llm-revisions.json",
        {
            "rows": [
                {
                    "section_id": "s1",
                    "time_range": "00:00:00.000 - 00:01:00.000",
                    "final_markdown": "章节内容",
                }
            ]
        },
    )
    _write_json(exports / "course-map.json", {"mainline": "课程主线"})
    headings = (
        "基本信息",
        "一句话概览",
        "核心主题 / 课程主线",
        "分段总结",
        "关键观点 / 方法论",
        "可执行动作清单",
        "高频话术 / 可复用表达",
        "待复核点 / 低置信内容",
    )
    candidate = "生成方式：`codex_llm_rewrite_final`。\n" + "\n".join(
        f"## {heading}\n00:00:00.000 00:01:00.000 {'内容' * 50}" for heading in headings
    )
    exports.mkdir(parents=True, exist_ok=True)
    (exports / "smart-summary-global-reduce-candidate.md").write_text(candidate, encoding="utf-8")
    monkeypatch.setattr(
        "video_knowledge_pipeline.smart_summary_global_reduce.model_task_api_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model call must be skipped")),
    )

    result = run_smart_summary_global_reduce(
        root,
        reuse_candidate=True,
        execute=False,
        install=False,
        write=False,
    )

    assert result["status"] == "completed"
    assert result["ok"] is True
    assert result["model_call"]["reused_candidate"] is True


def test_global_reduce_failed_candidate_is_preserved_even_with_existing_codex_summary(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    (exports / "smart-summary.codex.md").write_text("existing", encoding="utf-8")

    _write(root, {"status": "reduce_quality_failed"}, prompt="prompt", candidate="candidate", write=True)

    assert (exports / "smart-summary.codex.md").read_text(encoding="utf-8") == "existing"
    assert (exports / "smart-summary-global-reduce-candidate.md").read_text(encoding="utf-8") == "candidate"


def test_summary_consistency_allows_unknown_but_blocks_new_number(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_json(root / "manifest.json", {"title": "课程"})
    _write_json(
        root / "exports" / "smart-summary-section-llm-revisions.json",
        {
            "rows": [
                {
                    "section_id": "s1",
                    "title": "Playwright 获客流程",
                    "final_markdown": "使用 Playwright 做三步获客流程。",
                }
            ]
        },
    )
    summary = root / "exports" / "smart-summary.codex.md"
    summary.write_text("# 总结\nPlaywright 流程包含四步。另有 UnknownTool 待核实。", encoding="utf-8")

    result = run_summary_consistency_check(root, write=False)

    assert result["status"] == "conflict"
    assert any(row["value"] == "四" for row in result["conflicts"])
    assert any(row.get("entity") == "UnknownTool" for row in result["unknown_or_insufficient"])


def test_video_evidence_confirmation_requires_evidence_for_confirmed(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_json(root / "manifest.json", {"title": "课程"})
    _write_json(
        root / "exports" / "video-evidence-query-plan.json",
        {
            "query": "工具名",
            "fine_stage": {
                "candidates": [
                    {
                        "candidate_id": "evidence-query-001",
                        "time_range": "00:00:10 - 00:00:20",
                    }
                ]
            },
        },
    )
    decisions = root / "decisions.json"
    _write_json(decisions, {"rows": [{"candidate_id": "evidence-query-001", "status": "confirmed", "evidence_paths": []}]})

    result = apply_video_evidence_confirmation(root, decisions_json=decisions, write=False)

    assert result["ok"] is False
    assert result["invalid_rows"][0]["reason"] == "confirmed_requires_evidence"
