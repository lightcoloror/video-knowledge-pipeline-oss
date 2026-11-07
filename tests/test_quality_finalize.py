from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.quality_finalize import finalize_quality_outputs


def _bundle(root: Path, *, corrected: bool = True) -> Path:
    root.mkdir()
    manifest = {}
    if corrected:
        (root / "corrected-transcript.json").write_text(
            json.dumps({"segments": [{"start": 0, "end": 60, "text": "有标点的纠正版逐字稿。"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest["corrected_transcript_json"] = "corrected-transcript.json"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_quality_finalize_requires_corrected_transcript(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", corrected=False)

    result = finalize_quality_outputs(bundle, write=False)

    assert result["status"] == "blocked_missing_corrected_transcript"
    assert result["operator_boundary"]["raw_asr_is_not_summary_input"] is True


def test_quality_finalize_stops_before_summary_when_transcript_gate_fails(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.run_transcript_quality_gate",
        lambda *args, **kwargs: {"status": "failed", "ok": False, "fail_count": 2, "next_actions": ["fix transcript"]},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.build_semantic_chapter_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chapters must not run")),
    )

    result = finalize_quality_outputs(bundle, write=False)

    assert result["status"] == "blocked_transcript_quality"
    assert result["next_actions"] == ["fix transcript"]


def test_quality_finalize_preview_builds_semantic_chapters_then_plans_llm(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    order: list[str] = []
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.run_transcript_quality_gate",
        lambda *args, **kwargs: {"status": "passed", "ok": True, "fail_count": 0},
    )

    def semantic(*args, **kwargs):
        order.append("semantic")
        assert kwargs["chapter_mode"] == "semantic"
        return {"status": "completed", "ok": True, "chapter_count": 1, "chapters": [{"chapter_id": "chapter-1"}]}

    def chapters(*args, **kwargs):
        order.append("chapters")
        assert kwargs["chapter_mode"] == "semantic"
        return {"status": "completed", "ok": True, "chapter_count": 1, "chapters": [{"chapter_id": "chapter-1"}]}

    def section_llm(*args, **kwargs):
        order.append("llm")
        assert kwargs["execute"] is False
        assert kwargs["install"] is True
        assert kwargs["require_all_sections"] is True
        return {"status": "planned", "ok": True, "next_actions": ["execute sections"]}

    monkeypatch.setattr("video_knowledge_pipeline.quality_finalize.build_semantic_chapter_plan", semantic)
    monkeypatch.setattr("video_knowledge_pipeline.quality_finalize.build_smart_summary_chapter_pack", chapters)
    monkeypatch.setattr("video_knowledge_pipeline.quality_finalize.run_smart_summary_section_llm_rewrite", section_llm)

    result = finalize_quality_outputs(bundle, execute_llm=False, write=False)

    assert result["status"] == "planned_section_llm"
    assert order == ["semantic", "chapters", "llm"]


def test_quality_finalize_only_completes_after_final_summary_quality_passes(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.run_transcript_quality_gate",
        lambda *args, **kwargs: {"status": "passed", "ok": True, "fail_count": 0},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.build_semantic_chapter_plan",
        lambda *args, **kwargs: {"status": "completed", "ok": True, "chapters": [{"chapter_id": "chapter-1"}]},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.build_smart_summary_chapter_pack",
        lambda *args, **kwargs: {"status": "completed", "ok": True, "chapters": [{"chapter_id": "chapter-1"}]},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.run_smart_summary_section_llm_rewrite",
        lambda *args, **kwargs: {"status": "completed", "ok": True},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.run_smart_summary_global_reduce",
        lambda *args, **kwargs: {"status": "completed", "ok": True},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.run_summary_consistency_check",
        lambda *args, **kwargs: {"status": "passed_with_unknowns", "ok": True, "quality": {"high_risk_conflicts": 0, "unknown_items": 1}},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_finalize.smart_summary_quality_check",
        lambda *args, **kwargs: {"status": "passed", "passed": True},
    )
    monkeypatch.setattr("video_knowledge_pipeline.knowledge_note_export.export_knowledge_note",
                        lambda *a, **k: {"status": "completed", "ok": True})
    monkeypatch.setattr("video_knowledge_pipeline.workflow_completion.workflow_completion_status",
                        lambda *a, **k: {"status": "completed", "completed": True, "next_actions": []})

    result = finalize_quality_outputs(bundle, execute_llm=True, write=True)

    assert result["status"] == "completed"
    assert result["ok"] is True


def test_quality_finalize_resumes_claim_review_without_repeating_any_model_stage(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "exports").mkdir()
    (bundle / "exports/smart-summary-global-reduce.json").write_text(
        json.dumps({"status": "waiting_claim_review", "ok": False}), encoding="utf-8")
    module = "video_knowledge_pipeline.quality_finalize."
    monkeypatch.setattr(module + "run_transcript_quality_gate", lambda *a, **k: {"status": "passed", "fail_count": 0})
    def forbidden(*a, **k):
        raise AssertionError("pending review must not rebuild chapters or regenerate model results")
    for name in ("build_semantic_chapter_plan", "build_smart_summary_chapter_pack", "run_smart_summary_section_llm_rewrite"):
        monkeypatch.setattr(module + name, forbidden)
    def reduce(*a, **k):
        assert k['reuse_candidate'] is True and k['execute'] is False
        return {"status": "waiting_claim_review", "ok": False, "next_actions": ["prepare review"],
                "claim_review": {"passed": False}, "claim_review_task": {"phase": "review"}}
    monkeypatch.setattr(module + "run_smart_summary_global_reduce", reduce)
    result = finalize_quality_outputs(bundle, execute_llm=True, write=True)
    assert result['status'] == 'waiting_claim_review' and not result['ok']
    assert result['claim_review_task']['phase'] == 'review'
    assert result['operator_boundary']['claim_review_resume_never_repeats_generation']
    assert result['next_actions'] == ['prepare review']
    def reviewed(*a, **k):
        assert k['reuse_candidate'] is True and k['execute'] is False
        return {'status': 'completed', 'ok': True}
    monkeypatch.setattr(module + 'run_smart_summary_global_reduce', reviewed)
    monkeypatch.setattr(module + 'run_summary_consistency_check', lambda *a, **k: {'quality': {'high_risk_conflicts': 0}})
    monkeypatch.setattr(module + 'smart_summary_quality_check', lambda *a, **k: {'passed': True})
    monkeypatch.setattr("video_knowledge_pipeline.knowledge_note_export.export_knowledge_note",
                        lambda *a, **k: {"status": "completed", "ok": True})
    monkeypatch.setattr("video_knowledge_pipeline.workflow_completion.workflow_completion_status",
                        lambda *a, **k: {"status": "completed", "completed": True, "next_actions": []})
    final = finalize_quality_outputs(bundle, execute_llm=True, write=True)
    assert final['status'] == 'completed' and final['ok']
