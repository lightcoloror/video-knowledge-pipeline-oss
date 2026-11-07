from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.quality_benchmark import _summary_blind_acceptance
from video_knowledge_pipeline.summary_blind_review import CRITERIA, apply_summary_blind_review, build_summary_blind_review


def _write_summary(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n生成方式：`hidden-source`。\n\n## 总结\n\n{body}\n\n- 辅助文件：C:\\private\\artifact.md\n", encoding="utf-8")


def test_summary_blind_review_hides_sources_and_applies_role_scores(tmp_path: Path) -> None:
    baseline = tmp_path / "old-vkp" / "smart-summary.md"
    candidate = tmp_path / "new-vkp" / "smart-summary.md"
    reference = tmp_path / "getbrain-smart-summary.md"
    _write_summary(baseline, "旧版 VKP", "结构较散，行动建议较少。")
    _write_summary(candidate, "本地 SenseVoice ASR 版", "结构清楚，覆盖完整，行动建议明确。")
    _write_summary(reference, "得到大脑智能总结", "结构清楚，阅读自然。")
    quality_manifest = tmp_path / "quality-benchmark-manifest.json"
    quality_manifest.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.quality_benchmark_manifest.v1",
                "samples": [],
                "summary_blind_review": {
                    "required": True,
                    "target_mean_improvement": 0.5,
                    "target_reference_readability_gap": 0.3,
                    "items": [
                        {
                            "item_id": "summary-01",
                            "video_title": "活动获客课程",
                            "bundle_dir": str(tmp_path / "bundle"),
                            "baseline_summary_path": str(baseline),
                            "candidate_summary_path": str(candidate),
                            "reference_summary_path": str(reference),
                            "baseline_score": None,
                            "candidate_score": None,
                            "reference_score": None,
                            "review_status": "todo",
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    built = build_summary_blind_review(quality_manifest, write=True)

    assert built["status"] == "ready"
    public_text = (tmp_path / "summary-blind-review.json").read_text(encoding="utf-8")
    page = (tmp_path / "summary-blind-review.html").read_text(encoding="utf-8")
    assert "SenseVoice" not in public_text
    assert "得到大脑" not in public_text
    assert str(candidate) not in public_text
    assert "C:\\private" not in public_text
    assert "活动获客课程" in public_text
    assert "视频 1 | 活动获客课程" in page
    assert "本页评测视频" in page
    assert "视频名称公开；各版本的生成来源保持匿名" in page
    assert "版本 A" in page and "版本 B" in page and "版本 C" in page
    private = json.loads((tmp_path / "summary-blind-review.private.json").read_text(encoding="utf-8"))
    role_by_label = {label: mapping["role"] for label, mapping in private["items"][0]["labels"].items()}
    role_score = {"baseline": 3.0, "candidate": 4.0, "reference": 4.2}
    label_scores = {
        label: {key: role_score[role] for key, _display in CRITERIA}
        for label, role in role_by_label.items()
    }
    scores = {
        "schema": "video_knowledge_pipeline.summary_blind_review_scores.v1",
        "items": [{"item_id": "summary-01", "label_scores": label_scores, "winner_label": next(label for label, role in role_by_label.items() if role == "reference"), "notes": "fixture"}],
    }
    scores_path = tmp_path / "summary-blind-review-scores.json"
    scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    result = apply_summary_blind_review(tmp_path / "summary-blind-review.private.json", scores_path, write=True)

    assert result["status"] == "completed"
    updated = json.loads(quality_manifest.read_text(encoding="utf-8"))
    item = updated["summary_blind_review"]["items"][0]
    assert item["baseline_score"] == 3.0
    assert item["candidate_score"] == 4.0
    assert item["reference_score"] == 4.2
    run = json.loads((tmp_path / "quality-benchmark.json").read_text(encoding="utf-8"))
    summary = run["acceptance"]["summary_blind_review"]
    assert summary["ready"] is True
    assert summary["mean_improvement"] == 1.0
    assert summary["mean_reference_readability_gap"] == 0.2
    assert summary["passed"] is True


def test_summary_blind_review_blocks_when_candidate_is_missing(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.md"
    baseline.write_text("# baseline", encoding="utf-8")
    manifest = tmp_path / "quality-benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "summary_blind_review": {
                    "items": [
                        {
                            "item_id": "summary-01",
                            "baseline_summary_path": str(baseline),
                            "candidate_summary_path": "",
                        }
                    ]
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = build_summary_blind_review(manifest, write=False)

    assert result["status"] == "blocked"
    assert result["missing"][0]["missing_roles"] == ["candidate"]

def test_summary_blind_review_does_not_penalize_candidate_readability_that_exceeds_reference() -> None:
    result = _summary_blind_acceptance(
        {
            "target_mean_improvement": 0.5,
            "target_reference_readability_gap": 0.3,
            "items": [
                {
                    "baseline_score": 3.0,
                    "candidate_score": 4.5,
                    "reference_summary_path": "evaluation-only.md",
                    "candidate_dimension_scores": {"readability": 4.8},
                    "reference_dimension_scores": {"readability": 4.1},
                    "review_status": "completed",
                }
            ],
        }
    )

    assert result["passed"] is True
    assert result["mean_reference_readability_gap"] == 0.0
    assert result["mean_candidate_readability_margin"] == 0.7
    assert result["reference_readability_gap_semantics"] == "max(0, reference_readability - candidate_readability)"

def test_summary_blind_review_excludes_rule_draft_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.md"
    candidate = tmp_path / "candidate.md"
    reference = tmp_path / "reference.md"
    baseline.write_text(
        "# 旧规则版\n\n生成方式：codex_assisted_draft。\n\n## 可执行动作清单\n\n- 然后呢，就是说，这个那个。\n",
        encoding="utf-8",
    )
    _write_summary(candidate, "LLM 候选", "先确认客户问题，再根据证据给出三步行动方案。")
    _write_summary(reference, "外部参考", "先梳理问题，再设计行动。")
    manifest = tmp_path / "quality-benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "summary_blind_review": {
                    "items": [
                        {
                            "item_id": "summary-llm-only",
                            "video_title": "LLM 总结验收",
                            "baseline_summary_path": str(baseline),
                            "candidate_summary_path": str(candidate),
                            "reference_summary_path": str(reference),
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_summary_blind_review(manifest, write=True)

    assert result["status"] == "ready"
    public = json.loads((tmp_path / "summary-blind-review.json").read_text(encoding="utf-8"))
    private = json.loads((tmp_path / "summary-blind-review.private.json").read_text(encoding="utf-8"))
    assert len(public["items"][0]["versions"]) == 2
    assert public["items"][0]["excluded_non_llm_versions"] == 1
    assert {row["role"] for row in private["items"][0]["excluded_roles"]} == {"baseline"}
    assert {mapping["role"] for mapping in private["items"][0]["labels"].values()} == {"candidate", "reference"}

def test_summary_blind_review_preserves_logseq_hierarchy_and_tables(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.md"
    reference = tmp_path / "reference.md"
    _write_summary(candidate, "LLM 候选", "## 主线\n\n- 第一步\n  - 证据")
    reference.write_text(
        "📑 智能总结\n\t\t- 录音信息\n\t\t\t- **时长**：17 分钟\n\t\t- 录音总结\n\t\t\t- **关键观点**\n\t\t\t\t- 先建立信任\n\n| 项目 | 内容 |\n| --- | --- |\n| 类型 | 培训 |\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "quality-benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "summary_blind_review": {
                    "items": [
                        {
                            "item_id": "hierarchy",
                            "video_title": "层级测试",
                            "candidate_summary_path": str(candidate),
                            "reference_summary_path": str(reference),
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    build_summary_blind_review(manifest, write=True)

    page = (tmp_path / "summary-blind-review.html").read_text(encoding="utf-8")
    assert "<h1>📑 智能总结</h1>" in page
    assert "<li>录音信息\n<ul>" in page
    assert "<strong>时长</strong>：17 分钟" in page
    assert "<li><strong>关键观点</strong>\n<ul>" in page
    assert "<table>" in page
