from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.smart_summary_global_reduce as global_reduce
import video_knowledge_pipeline.smart_summary_codex as smart_summary_codex
from video_knowledge_pipeline.smart_summary_reader_plan import (
    SCHEMA,
    evaluate_reader_markdown_semantics,
    normalize_reader_plan_candidate,
    parse_reader_plan,
    render_reader_summary,
    validate_reader_plan,
)


def _fact_pack() -> dict:
    sections = []
    snippets = (
        "先确认客户真正想解决的问题，再选择合适的方法。",
        "持续提供有用信息，信任才会逐步形成。",
        "沟通时先问现状和目标，再讨论具体方案。",
        "把后续动作和完成时间记录下来，避免口头约定丢失。",
    )
    for index, snippet in enumerate(snippets, start=1):
        evidence_id = f"ev-{index:04d}"
        fact_type = "actions" if index == 4 else "key_points"
        sections.append(
            {
                "section_id": f"chapter-{index:04d}",
                "facts": [
                    {
                        "fact_type": fact_type,
                        "text": snippet,
                        "fact_status": "candidate_evidence",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "evidence_refs": [
                    {
                        "evidence_id": evidence_id,
                        "source_kind": "asr",
                        "fact_status": "candidate_evidence",
                        "snippet": snippet,
                    }
                ],
            }
        )
    sections.append(
        {
            "section_id": "review-only",
            "facts": [],
            "evidence_refs": [
                {
                    "evidence_id": "gap-0001",
                    "source_kind": "review_gap",
                    "fact_status": "review_gap_not_fact",
                    "snippet": "某个数字听不清",
                }
            ],
        }
    )
    return {"sections": sections}


def _claim(index: int, *, title: str, text: str) -> dict:
    return {
        "title": title,
        "text": text,
        "time_ranges": [f"00:0{index - 1}:00.000 - 00:0{index}:00.000"],
        "evidence_ids": [f"ev-{index:04d}"],
        "source_modalities": ["asr"],
    }


def _valid_plan() -> dict:
    return {
        "schema": SCHEMA,
        "source_section_ids": [f"chapter-{index:04d}" for index in range(1, 5)],
        "overview": {
            "text": "这段内容说明如何从确认需求、持续提供价值到形成信任，并把沟通结果落实为后续行动。",
            "evidence_ids": ["ev-0001", "ev-0002", "ev-0003", "ev-0004"],
        },
        "core_insights": [
            _claim(1, title="先确认真实需求", text="不要急着介绍方案，先明确对方当前处境和真正目标。"),
            _claim(2, title="用持续价值建立信任", text="稳定提供有用信息，比一次性说服更容易形成长期信任。"),
            _claim(3, title="让沟通形成清晰路径", text="按现状、目标和方案的顺序推进，减少信息错位。"),
            _claim(4, title="把共识转成下一步", text="明确负责人和完成时间，避免行动只停留在口头。"),
        ],
        "themes": [
            {
                "title": "识别问题与目标",
                "time_range": "00:00:00.000 - 00:01:20.000",
                "summary": "先了解当前情况和期待结果，再决定后续沟通重点。",
                "problem": "一开始就讲方案容易答非所问。",
                "reason": "需求和目标尚未被确认。",
                "method": "用开放问题确认现状、限制和目标。",
                "case": "讲者示范了先询问现状再讨论方案的顺序。",
                "action": "整理三项确认问题用于下一次沟通。",
                "evidence_ids": ["ev-0001"],
                "source_modalities": ["asr"],
            },
            {
                "title": "持续价值与信任",
                "time_range": "00:01:20.000 - 00:02:40.000",
                "summary": "信任来自持续且相关的帮助，而不是频繁推销。",
                "problem": "只在成交前联系会让关系缺少基础。",
                "reason": "对方没有持续感受到沟通价值。",
                "method": "围绕真实问题分阶段提供信息。",
                "case": "讲者将持续提供价值与信任形成联系起来。",
                "action": "制定后续信息触达节奏并记录反馈。",
                "evidence_ids": ["ev-0002"],
                "source_modalities": ["asr"],
            },
            {
                "title": "方案沟通与行动闭环",
                "time_range": "00:02:40.000 - 00:04:00.000",
                "summary": "讨论方案后要留下清楚的下一步，才能形成可执行闭环。",
                "problem": "口头共识容易在沟通结束后丢失。",
                "reason": "负责人和时间点没有被记录。",
                "method": "按现状、目标、方案和行动顺序收束沟通。",
                "case": "讲者要求把后续动作和完成时间记录下来。",
                "action": "确认负责人、动作和完成时间。",
                "evidence_ids": ["ev-0003", "ev-0004"],
                "source_modalities": ["asr"],
            },
        ],
        "principles": [
            _claim(1, title="需求优先于方案", text="方案选择应建立在明确需求之上，而不是反过来。"),
            _claim(2, title="信任来自持续贡献", text="持续解决小问题能够为后续重要决策积累信任。"),
            _claim(4, title="行动必须可追踪", text="明确负责人和时间点，才能判断沟通是否真正落地。"),
        ],
        "actions": [
            {
                "text": "确认客户的现状、限制和目标。",
                "time_range": "00:00:00.000 - 00:01:00.000",
                "evidence_ids": ["ev-0001"],
            },
            {
                "text": "记录后续动作、负责人和完成时间。",
                "time_range": "00:03:00.000 - 00:04:00.000",
                "evidence_ids": ["ev-0004"],
            },
        ],
        "reusable_expressions": [
            {
                "text": "先确认客户真正想解决的问题，再选择合适的方法。",
                "kind": "verbatim_quote",
                "time_range": "00:00:00.000 - 00:01:00.000",
                "evidence_ids": ["ev-0001"],
            }
        ],
        "review_items": [
            {
                "text": "一处数字无法从现有转写确认。",
                "time_range": "00:02:00.000 - 00:02:10.000",
                "evidence_ids": ["gap-0001"],
                "missing_evidence": "清晰音频或人工听审",
            }
        ],
    }


def test_reader_plan_validates_and_renders_without_internal_fields() -> None:
    plan = _valid_plan()
    result = validate_reader_plan(
        plan,
        fact_pack=_fact_pack(),
        expected_section_ids=set(plan["source_section_ids"]),
    )

    assert result["passed"] is True
    markdown = render_reader_summary(
        plan,
        title="客户沟通课程",
        first_time="00:00:00.000",
        last_time="00:04:00.000",
    )
    assert "## 一句话概览" in markdown
    assert "### 识别问题与目标" in markdown
    assert "source_kind" not in markdown
    assert "evidence_id" not in markdown
    assert evaluate_reader_markdown_semantics(markdown)["passed"] is True


def test_reader_plan_blocks_review_only_evidence_and_unknown_ids() -> None:
    plan = _valid_plan()
    plan["core_insights"][0]["evidence_ids"] = ["gap-0001"]
    plan["actions"][0]["evidence_ids"] = ["missing-9999"]

    result = validate_reader_plan(
        plan,
        fact_pack=_fact_pack(),
        expected_section_ids=set(plan["source_section_ids"]),
    )

    assert result["passed"] is False
    assert any("review_only_evidence_promoted" in error for error in result["errors"])
    assert any("unknown_evidence" in error for error in result["errors"])


def test_reader_plan_blocks_overlap_fake_quote_and_non_action() -> None:
    plan = _valid_plan()
    plan["themes"][1]["time_range"] = "00:01:00.000 - 00:02:40.000"
    plan["actions"][0]["text"] = "课程背景介绍"
    plan["reusable_expressions"][0]["text"] = "这是模型自己改写的漂亮金句"

    result = validate_reader_plan(
        plan,
        fact_pack=_fact_pack(),
        expected_section_ids=set(plan["source_section_ids"]),
    )

    assert result["passed"] is False
    assert any("theme_overlap_or_out_of_order" in error for error in result["errors"])
    assert any("non_action_item" in error for error in result["errors"])
    assert any("verbatim_quote_not_found" in error for error in result["errors"])


def test_reader_plan_normalizes_provider_labels_without_adding_facts() -> None:
    plan = _valid_plan()
    plan["actions"][0]["text"] = "强制客户反馈使用体验。"
    plan["actions"].append(
        {
            "text": "课程背景介绍",
            "time_range": "00:00:00.000 - 00:01:00.000",
            "evidence_ids": ["ev-0001"],
        }
    )
    plan["reusable_expressions"][0]["text"] = "模型概括出的近似表达"
    plan["review_items"].extend(
        [
            {
                "text": "schema: 内部标题需要修正。",
                "time_range": "",
                "evidence_ids": [],
                "missing_evidence": "内部字段",
            },
            {
                "text": "说话人提到一项数据，需要外部核真。",
                "time_range": "00:00:00.000 - 00:01:00.000",
                "evidence_ids": ["ev-0001"],
                "missing_evidence": "缺少外部数据验证",
            },
        ]
    )

    normalized = normalize_reader_plan_candidate(plan, fact_pack=_fact_pack())
    repaired = normalized["plan"]
    validation = validate_reader_plan(
        repaired,
        fact_pack=_fact_pack(),
        expected_section_ids=set(repaired["source_section_ids"]),
    )

    assert validation["passed"] is True
    assert repaired["actions"][0]["text"].startswith("强制")
    assert len(repaired["actions"]) == 2
    assert all(row["text"] != "课程背景介绍" for row in repaired["actions"])
    assert repaired["reusable_expressions"][0]["kind"] == "reusable_expression"
    assert all("schema:" not in row["text"] for row in repaired["review_items"])
    assert all("外部核真" not in row["text"] for row in repaired["review_items"])
    assert {row["kind"] for row in normalized["repairs"]} == {
        "drop_internal_meta_review",
        "drop_external_fact_check_for_speaker_claim",
        "drop_non_action_item",
        "downgrade_unverified_verbatim_quote",
    }


def test_reader_plan_accepts_three_distinct_core_insights_for_short_content() -> None:
    plan = _valid_plan()
    plan["core_insights"] = plan["core_insights"][:3]

    result = validate_reader_plan(
        plan,
        fact_pack=_fact_pack(),
        expected_section_ids=set(plan["source_section_ids"]),
    )

    assert result["passed"] is True
    assert result["core_insight_count"] == 3


def test_reader_plan_compresses_overlong_overview_at_sentence_boundary() -> None:
    plan = _valid_plan()
    plan["overview"]["text"] = (
        "这是第一句，概括采访背景与核心主题。"
        "这是第二句，说明关键决策与主要经历。"
        "这是第三句，补充行动、隐私和待复核边界。" * 12
    )

    normalized = normalize_reader_plan_candidate(plan, fact_pack=_fact_pack())
    repaired = normalized["plan"]
    validation = validate_reader_plan(
        repaired,
        fact_pack=_fact_pack(),
        expected_section_ids=set(repaired["source_section_ids"]),
    )

    assert validation["passed"] is True
    assert 24 <= len(repaired["overview"]["text"]) <= 240
    assert repaired["overview"]["text"].endswith("。")
    assert any(row["kind"] == "compress_overlong_overview" for row in normalized["repairs"])


def test_interview_plan_anonymizes_unbound_identity_and_drops_prescriptive_action() -> None:
    plan = _valid_plan()
    plan["overview"]["text"] = "陈女士讲述了自己的诊治与保险经历，采访者记录其原意。"
    plan["actions"] = [
        {
            "text": "应优先选择非手术治疗，以避免延误。",
            "time_range": "00:01:00.000 - 00:02:00.000",
            "evidence_ids": ["ev-0001"],
        }
    ]
    plan["themes"][0]["method"] = "选择放疗方案，避免再次接受手术。"
    plan["themes"][0]["action"] = "应优先选择非手术治疗，以避免延误。"
    fact_pack = _fact_pack()
    fact_pack["content_profile"] = "interview"

    normalized = normalize_reader_plan_candidate(plan, fact_pack=fact_pack)
    repaired = normalized["plan"]
    validation = validate_reader_plan(
        repaired,
        fact_pack=fact_pack,
        expected_section_ids=set(repaired["source_section_ids"]),
    )

    assert validation["passed"] is True
    assert "陈女士" not in json.dumps(repaired, ensure_ascii=False)
    assert "受访者" in repaired["overview"]["text"]
    assert repaired["actions"] == []
    assert repaired["themes"][0]["action"] == ""
    assert repaired["themes"][0]["method"].startswith("受访者当时的个人选择是：")
    assert {
        "anonymize_unbound_interview_identity",
        "drop_interview_prescriptive_action",
        "drop_interview_prescriptive_theme_action",
        "attribute_interview_high_stakes_method",
    }.issubset({row["kind"] for row in normalized["repairs"]})

    rendered = render_reader_summary(
        repaired,
        title="客户采访",
        first_time="00:00:00.000",
        last_time="00:03:00.000",
    )
    assert "## 核心主题 / 内容主线" in rendered
    assert "- 个人选择/经历：" in rendered
    assert "优先选择非手术" not in rendered


def test_explicit_medical_interview_renders_fact_first_sections() -> None:
    rendered = render_reader_summary(
        _valid_plan(),
        title="客户医疗采访",
        first_time="00:00:00.000",
        last_time="00:03:00.000",
        content_profile="medical-insurance-interview-v1",
    )

    assert "## 核心主题 / 事实主线" in rendered
    assert "## 事实时间线" in rendered
    assert "## 受访者原话与感受" in rendered
    assert "## 明确后续事项" in rendered
    assert "## 原话摘录" in rendered
    assert "## 待核实事项 / 隐私与发布边界" in rendered
    assert "## 关键观点 / 方法论" not in rendered
    assert "## 高频话术 / 可复用表达" not in rendered


def test_reader_plan_normalizes_overlapping_theme_windows_and_action_verbs() -> None:
    plan = _valid_plan()
    plan["themes"][0]["time_range"] = "00:00:00.000 - 00:01:20.000"
    plan["themes"][1]["time_range"] = "00:01:00.000 - 00:02:00.000"
    plan["themes"][2]["time_range"] = "00:00:00.000 - 00:03:00.000"
    plan["actions"][0]["text"] = "将已打磨内容标准化并保持持续输出。"

    normalized = normalize_reader_plan_candidate(plan, fact_pack=_fact_pack())
    repaired = normalized["plan"]
    validation = validate_reader_plan(
        repaired,
        fact_pack=_fact_pack(),
        expected_section_ids=set(repaired["source_section_ids"]),
    )

    assert validation["passed"] is True
    assert repaired["themes"][0]["time_range"] == "00:00:00.000 - 00:01:00.000"
    assert repaired["themes"][2]["time_range"] == "00:02:00.000 - 00:03:00.000"
    assert {row["kind"] for row in normalized["repairs"]} == {
        "partition_overlapping_theme_ranges",
        "clamp_containing_theme_start",
    }

def test_parse_reader_plan_rejects_prose_wrapper() -> None:
    wrapped = "下面是结果：\n" + json.dumps(_valid_plan(), ensure_ascii=False)
    result = parse_reader_plan(wrapped)

    assert result["ok"] is False
    assert result["errors"][0].startswith("invalid_json")


def test_semantic_gate_detects_meta_overview_and_fake_action() -> None:
    markdown = """# 示例 - 智能总结

## 一句话概览
这份总结围绕章节修订展开，下面给出课程内容。

## 分段总结
### 章节一：我们来看看。（00:00:00.000 - 00:10:00.000）
内容。

## 可执行动作清单
- 00:00:00：视觉证据待复核。
"""
    result = evaluate_reader_markdown_semantics(markdown)

    assert result["passed"] is False
    assert any("overview_describes_generation" in problem for problem in result["problems"])
    assert any("weak_theme_title" in problem for problem in result["problems"])
    assert any("non_action_item" in problem for problem in result["problems"])


def test_global_reduce_executes_structured_reader_plan_contract(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"title": "客户沟通课程"}, ensure_ascii=False),
        encoding="utf-8",
    )
    section_ids = [f"chapter-{index:04d}" for index in range(1, 5)]
    rows = [
        {
            "section_id": section_id,
            "title": f"主题{index}",
            "time_range": f"00:0{index - 1}:00.000 - 00:0{index}:00.000",
            "final_markdown": f"第{index}章先整理证据，再形成行动。",
        }
        for index, section_id in enumerate(section_ids, start=1)
    ]
    (exports / "smart-summary-section-workflow.json").write_text(
        json.dumps({"sections": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    (exports / "smart-summary-section-llm-revisions.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    (exports / "course-map.json").write_text(
        json.dumps({"mainline": "需求、信任、沟通和行动"}, ensure_ascii=False),
        encoding="utf-8",
    )

    fact_pack = _fact_pack()
    fact_pack.update(
        {
            "schema": global_reduce.CHAPTER_FACT_PACK_SCHEMA,
            "revision": "a" * 64,
            "summary": {
                "evidence_bound_sections": 4,
                "review_only_sections": 0,
                "unbound_section_ids": [],
                "evidence_reference_count": 5,
                "review_only_evidence_count": 1,
                "source_kinds": ["asr"],
            },
        }
    )
    monkeypatch.setattr(global_reduce, "_chapter_fact_pack", lambda *args, **kwargs: fact_pack)
    monkeypatch.setattr(
        global_reduce,
        "resolve_text_provider_config",
        lambda value: {"provider": "fake", "model": "fake-reader-model"},
    )
    captured: dict = {}

    def fake_call(task: str, **kwargs):
        captured.update({"task": task, **kwargs})
        return {"ok": True, "error": "", "content": json.dumps(_valid_plan(), ensure_ascii=False)}

    monkeypatch.setattr(global_reduce, "model_task_api_call", fake_call)

    result = global_reduce.run_smart_summary_global_reduce(
        root,
        execute=True,
        install=False,
        write=True,
    )

    assert result["ok"] is False
    assert result["status"] == "waiting_claim_review"
    assert captured["task"] == "smart_summary_global_reduce"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["execute"] is True
    assert (exports / "smart-summary-reader-plan.json").exists()
    assert (exports / "smart-summary-global-reduce-raw-response.txt").exists()
    assert not (exports / "smart-summary.codex.md").exists()
    summary = render_reader_summary(_valid_plan(), title="synthetic", first_time="00:00:00", last_time="00:04:00")
    assert "## 一句话概览" in summary
    assert "source_kind" not in summary
    assert "evidence_id" not in summary

def test_global_reduce_reuse_normalizes_saved_reader_plan_without_model_call(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"title": "短内容"}, ensure_ascii=False),
        encoding="utf-8",
    )
    section_ids = [f"chapter-{index:04d}" for index in range(1, 5)]
    rows = [
        {
            "section_id": section_id,
            "title": f"主题{index}",
            "time_range": f"00:0{index - 1}:00.000 - 00:0{index}:00.000",
            "final_markdown": f"第{index}章先整理证据，再形成行动。",
        }
        for index, section_id in enumerate(section_ids, start=1)
    ]
    (exports / "smart-summary-section-workflow.json").write_text(
        json.dumps({"sections": rows}, ensure_ascii=False), encoding="utf-8"
    )
    (exports / "smart-summary-section-llm-revisions.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8"
    )
    (exports / "course-map.json").write_text(
        json.dumps({"mainline": "需求、信任、沟通和行动"}, ensure_ascii=False),
        encoding="utf-8",
    )
    fact_pack = _fact_pack()
    fact_pack.update(
        {
            "schema": global_reduce.CHAPTER_FACT_PACK_SCHEMA,
            "revision": "b" * 64,
            "summary": {
                "evidence_bound_sections": 4,
                "review_only_sections": 0,
                "unbound_section_ids": [],
                "evidence_reference_count": 5,
                "review_only_evidence_count": 1,
                "source_kinds": ["asr"],
            },
        }
    )
    plan = _valid_plan()
    plan["core_insights"] = plan["core_insights"][:3]
    plan["actions"].append(
        {
            "text": "课程背景介绍",
            "time_range": "00:00:00.000 - 00:01:00.000",
            "evidence_ids": ["ev-0001"],
        }
    )
    (exports / "smart-summary-reader-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(global_reduce, "_chapter_fact_pack", lambda *args, **kwargs: fact_pack)
    monkeypatch.setattr(
        global_reduce,
        "model_task_api_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reuse must not call a model")
        ),
    )

    result = global_reduce.run_smart_summary_global_reduce(
        root,
        reuse_candidate=True,
        execute=False,
        install=False,
        write=False,
    )

    assert result["ok"] is False
    assert result["status"] == "waiting_claim_review"
    assert result["model_call"]["reused_candidate"] is True
    assert result["reader_plan"]["validation"]["passed"] is True
    assert {
        row["kind"] for row in result["reader_plan"]["validation"]["normalizations"]
    } == {"drop_non_action_item"}

def test_hidden_final_marker_is_accepted_by_canonical_selector(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    content = render_reader_summary(
        _valid_plan(),
        title="reader summary",
        first_time="00:00:00.000",
        last_time="00:04:00.000",
    )
    path = exports / "smart-summary.codex.md"
    path.write_text(content, encoding="utf-8")

    assert "<!-- codex_llm_rewrite_final -->" in content
    assert not any(line.startswith("\u751f\u6210\u65b9\u5f0f") for line in content.splitlines())
    assert smart_summary_codex._existing_codex_summary(root) == path.resolve()

def test_quality_semantic_gate_receives_full_reader_markdown(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    content = render_reader_summary(
        _valid_plan(),
        title="reader summary",
        first_time="00:00:00.000",
        last_time="00:04:00.000",
    )
    path = exports / "smart-summary.codex.md"
    path.write_text(content, encoding="utf-8")
    captured: dict[str, str] = {}

    def fake_semantics(value: str) -> dict:
        captured["content"] = value
        return {"passed": True, "problems": []}

    monkeypatch.setattr(
        smart_summary_codex, "evaluate_reader_markdown_semantics", fake_semantics
    )
    smart_summary_codex.smart_summary_quality_check(
        root, summary_path=path, require_codex=True, write=False
    )

    assert "## \u4e00\u53e5\u8bdd\u6982\u89c8" in captured["content"]
